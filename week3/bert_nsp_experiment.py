#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForNextSentencePrediction

DATASET_ID = "G-reen/cc-re-2020"
TOPICS = ["Politics","Finance & Business","Sports & Fitness","Science & Tech.","Entertainment"]
FORMATS = {"News Article","News (Org.)"}

def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def split_sents(text):
    text = norm(text)
    xs = re.split(r'(?<=[.!?])["”’\']?\s+(?=["“‘\']?[A-Z0-9])', text)
    return [norm(x) for x in xs if len(norm(x).split()) >= 6]

def make_pair(text, rng):
    s = split_sents(text)
    if len(s) < 4: return None
    for _ in range(20):
        na, nb = rng.randint(1,3), rng.randint(1,3)
        if len(s) < na + nb: continue
        st = rng.randint(0, len(s)-na-nb)
        A = " ".join(s[st:st+na])
        B = " ".join(s[st+na:st+na+nb])
        if 10 <= len(A.split()) <= 220 and 10 <= len(B.split()) <= 220:
            return A,B,na,nb
    return None

def collect(per_topic, seed, min_topic_prob=0.70, min_format_prob=0.70):
    rng = random.Random(seed)
    counts = {t:0 for t in TOPICS}
    out = []
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    seen = set()
    for row in ds:
        t = row.get("topic")
        if t not in counts or counts[t] >= per_topic: continue
        if row.get("format") not in FORMATS: continue
        if float(row.get("topic_prob") or 0) < min_topic_prob: continue
        if float(row.get("format_prob") or 0) < min_format_prob: continue
        text = norm(row.get("text"))
        if len(text.split()) < 80: continue
        key = row.get("id") or row.get("url") or text[:200]
        if key in seen: continue
        pair = make_pair(text, rng)
        if not pair: continue
        A,B,na,nb = pair
        out.append(dict(doc_id=str(key), topic=t, url=str(row.get("url") or ""),
                        fetch_date=str(row.get("date") or ""), A=A, original_B=B,
                        a_sentences=na, b_sentences=nb, b_words=len(B.split())))
        counts[t] += 1
        seen.add(key)
        if all(v >= per_topic for v in counts.values()): break
    if not all(v >= per_topic for v in counts.values()):
        raise RuntimeError(f"insufficient samples: {counts}")
    return pd.DataFrame(out)

def pick(anchor, pool, rng):
    pool = pool[pool.doc_id != anchor.doc_id].copy()
    pool["dist"] = (pool.b_words - anchor.b_words).abs()
    pool = pool.sort_values("dist").head(min(12,len(pool)))
    return pool.iloc[rng.randrange(len(pool))]

def build_pairs(docs, seed):
    rng = random.Random(seed)
    rows=[]
    for aid,a in docs.reset_index(drop=True).iterrows():
        same = pick(a, docs[docs.topic==a.topic], rng)
        other = pick(a, docs[docs.topic!=a.topic], rng)
        base=dict(anchor_id=aid, anchor_doc_id=a.doc_id, anchor_topic=a.topic,
                  A=a.A, true_B=a.original_B, true_B_words=a.b_words)
        for cond,b,gold in [("original",a,1),("same_topic",same,0),("other_topic",other,0)]:
            B = a.original_B if cond=="original" else b.original_B
            rows.append({**base, "condition":cond, "B":B, "B_doc_id":b.doc_id,
                         "B_topic":b.topic, "B_words":len(B.split()), "gold_is_next":gold})
    return pd.DataFrame(rows)

@torch.inference_mode()
def score(df, model_name, batch, maxlen, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name).to(device).eval()
    scores=[]
    for i in range(0,len(df),batch):
        x=df.iloc[i:i+batch]
        enc=tok(x.A.tolist(), x.B.tolist(), padding=True, truncation="longest_first",
                max_length=maxlen, return_tensors="pt")
        enc={k:v.to(device) for k,v in enc.items()}
        logits=model(**enc).logits
        p=torch.softmax(logits,dim=-1)[:,0]  # class 0 = IsNext
        scores.extend(p.cpu().numpy().tolist())
    z=df.copy()
    z["isnext_score"]=scores
    z["pred_is_next"]=(z.isnext_score>=0.5).astype(int)
    return z

def rank(pos, neg):
    return float(np.mean((pos>neg) + 0.5*(pos==neg)))

def metrics_from_arrays(o,s,x):
    return {
      "TPR_original":float(np.mean(o>=.5)),
      "FAR_same":float(np.mean(s>=.5)),
      "FAR_other":float(np.mean(x>=.5)),
      "Delta_FAR":float(np.mean(s>=.5)-np.mean(x>=.5)),
      "Rank_same":rank(o,s),
      "Rank_other":rank(o,x),
      "MeanScore_original":float(np.mean(o)),
      "MeanScore_same":float(np.mean(s)),
      "MeanScore_other":float(np.mean(x)),
      "ScoreGap_same_minus_other":float(np.mean(s-x))
    }

def analyze(scored, nboot, seed):
    p=scored.pivot(index="anchor_id",columns="condition",values="isnext_score")
    o=p.original.to_numpy(); s=p.same_topic.to_numpy(); x=p.other_topic.to_numpy()
    m=metrics_from_arrays(o,s,x)
    rng=np.random.default_rng(seed); rec=[]
    for b in range(nboot):
        ix=rng.integers(0,len(p),len(p))
        mm=metrics_from_arrays(o[ix],s[ix],x[ix]); mm["bootstrap_id"]=b; rec.append(mm)
    boot=pd.DataFrame(rec)
    summary=[]
    for k,v in m.items():
        summary.append({"metric":k,"estimate":v,
                        "ci_low":float(boot[k].quantile(.025)),
                        "ci_high":float(boot[k].quantile(.975))})
    return m,boot,pd.DataFrame(summary)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="week3/bert_nsp_results")
    ap.add_argument("--per-topic",type=int,default=40)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--model",default="google-bert/bert-base-uncased")
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--maxlen",type=int,default=512)
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    docs=collect(args.per_topic,args.seed); docs.to_csv(out/"documents.csv",index=False)
    pairs=build_pairs(docs,args.seed+1); pairs.to_csv(out/"pairs.csv",index=False)
    device="cuda" if torch.cuda.is_available() else "cpu"
    scored=score(pairs,args.model,args.batch,args.maxlen,device)
    scored.to_csv(out/"pair_scores.csv",index=False)
    m,boot,summary=analyze(scored,args.bootstrap,args.seed+2)
    boot.to_csv(out/"bootstrap.csv",index=False); summary.to_csv(out/"summary.csv",index=False)
    with open(out/"summary.json","w") as f:
        json.dump({"dataset":DATASET_ID,"topics":TOPICS,"n_docs":len(docs),"n_pairs":len(pairs),
                   "model":args.model,"device":device,"metrics":m},f,indent=2)
    same=scored[scored.condition=="same_topic"].sort_values("isnext_score",ascending=False).head(10)
    other=scored[scored.condition=="other_topic"].sort_values("isnext_score",ascending=False).head(10)
    report=["# BERT NSP same-topic confusion experiment","",
            f"- Documents: {len(docs)} ({args.per_topic} × {len(TOPICS)} topics)",
            f"- Pairs: {len(pairs)}","- Model: "+args.model,f"- Device: {device}","",
            "## Main metrics","",summary.to_markdown(index=False),"",
            "## Interpretation","",
            "- Main hypothesis: FAR_same > FAR_other (Delta_FAR > 0).",
            "- Secondary signal: Rank_same < Rank_other.",
            "- If Delta_FAR 95% bootstrap CI excludes 0, same-topic distractors systematically fool NSP more often.",
            "","## Highest-scoring same-topic false pairs",""]
    for _,r in same.iterrows():
        report += [f"### anchor {int(r.anchor_id)} · score={r.isnext_score:.4f}",
                   f"- topic: {r.anchor_topic}",f"- A: {r.A}",f"- false B: {r.B}",f"- true B: {r.true_B}",""]
    (out/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(m,indent=2))

if __name__=="__main__":
    main()
