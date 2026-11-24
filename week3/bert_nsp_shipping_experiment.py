#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForNextSentencePrediction

DATASET_ID = "Ktzoras/shipping_news_articles_summary"

CATEGORIES = {
    "freight_market": [
        "freight rate","freight rates","freight market","shipping rate","shipping rates",
        "charter rate","charter rates","spot rate","spot rates","hire rate","hire rates",
        "freight index","bdi","baltic dry","container rate","container rates","tanker rate","tanker rates",
        "lng freight","dry bulk market","tanker market","container market"
    ],
    "ports_congestion": [
        "port congestion","congestion","port delay","port delays","berth","terminal congestion",
        "container terminal","port operations","port strike","dockworker","dockworkers","stevedore",
        "port throughput","port capacity","port closure","port disruption","port authority",
        "port","ports","terminal","terminals","harbor","harbour","dock","docks"
    ],
    "routes_geopolitics": [
        "suez canal","panama canal","red sea","cape of good hope","strait of hormuz",
        "strait of malacca","shipping route","shipping routes","route planning","reroute","rerouting",
        "diversion","diverted","transit route","canal transit","geopolitical","sanctions",
        "houthi","black sea","ukraine","middle east"
    ],
    "regulation_decarbonization": [
        "imo","decarbonisation","decarbonization","emissions","emission","carbon tax","carbon price",
        "net zero","net-zero","fuel eu","fueleu","eu ets","ets","greenhouse gas","ghg",
        "alternative fuel","methanol","ammonia","hydrogen","lng fuel","energy efficiency",
        "environmental regulation","regulation","regulatory"
    ],
    "vessel_carrier_operations": [
        "fleet","vessel","vessels","shipowner","shipowners","carrier","carriers","shipping line",
        "shipping lines","newbuild","newbuilding","orderbook","scrubber","charter","chartering",
        "acquisition","investment","merger","alliance","capacity","deployment","service network",
        "vessel operation","ship management"
    ],
}

def norm(x: str) -> str:
    x = str(x or "").replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def repair_sentence_boundaries(text: str) -> str:
    # Some rows concatenate sentences without whitespace: "...matter said.“Some..."
    # Add whitespace only when punctuation is immediately followed by an uppercase/opening quote.
    text = re.sub(r'([.!?]["”’]?)((?=[“"\']?[A-Z]))', r'\1 ', text)
    return text

def split_sents(text: str) -> list[str]:
    text = repair_sentence_boundaries(norm(text))
    xs = re.split(r'(?<=[.!?])["”’\']?\s+(?=["“‘\']?[A-Z0-9])', text)
    return [norm(x) for x in xs if len(norm(x).split()) >= 6]

def phrase_count(text: str, phrase: str) -> int:
    return text.count(phrase)

def classify(title: str, content: str):
    title_l = norm(title).lower()
    body_l = norm(content).lower()[:5000]
    scores = {}
    for cat, kws in CATEGORIES.items():
        score = 0
        hits = []
        for kw in kws:
            tc = phrase_count(title_l, kw)
            bc = phrase_count(body_l, kw)
            if tc or bc:
                score += 4 * tc + min(bc, 3)
                hits.append((kw, tc, bc))
        scores[cat] = score

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    best, best_score = ranked[0]
    second_score = ranked[1][1]

    # Require a clear signal. This avoids forcing ambiguous shipping articles into a subtopic.
    if best_score < 4:
        return None, scores, []
    if best_score - second_score < 2:
        return None, scores, []

    hits = []
    for kw in CATEGORIES[best]:
        tc = phrase_count(title_l, kw)
        bc = phrase_count(body_l, kw)
        if tc or bc:
            hits.append(f"{kw}:title={tc},body={bc}")
    return best, scores, hits

def make_pair(text: str, rng: random.Random):
    s = split_sents(text)
    if len(s) < 4:
        return None
    for _ in range(30):
        na, nb = rng.randint(1,3), rng.randint(1,3)
        if len(s) < na + nb:
            continue
        start = rng.randint(0, len(s)-na-nb)
        A = " ".join(s[start:start+na])
        B = " ".join(s[start+na:start+na+nb])
        wa, wb = len(A.split()), len(B.split())
        if 10 <= wa <= 220 and 10 <= wb <= 220:
            return A, B, na, nb
    return None

def collect(per_category: int, seed: int, year: int):
    rng = random.Random(seed)
    counts = {c:0 for c in CATEGORIES}
    rows = []
    seen_links = set()

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    for row in ds:
        link = norm(row.get("link"))
        title = norm(row.get("title"))
        content = norm(row.get("content"))
        date = str(row.get("date") or "")

        if not link or link in seen_links:
            continue
        if year and not date.startswith(str(year)):
            continue
        if len(content.split()) < 100:
            continue

        category, scores, hits = classify(title, content)
        if category is None or counts[category] >= per_category:
            continue

        pair = make_pair(content, rng)
        if pair is None:
            continue
        A, B, na, nb = pair

        rows.append({
            "doc_id": link,
            "link": link,
            "date": date,
            "title": title,
            "category": category,
            "category_score": scores[category],
            "category_margin": scores[category] - sorted(scores.values(), reverse=True)[1],
            "keyword_hits": " | ".join(hits),
            "A": A,
            "original_B": B,
            "a_sentences": na,
            "b_sentences": nb,
            "a_words": len(A.split()),
            "b_words": len(B.split()),
        })
        seen_links.add(link)
        counts[category] += 1

        if all(v >= per_category for v in counts.values()):
            break

    if not all(v >= per_category for v in counts.values()):
        raise RuntimeError(f"Not enough clear documents for year={year}: {counts}")

    return pd.DataFrame(rows)

def pick(anchor, pool, rng):
    pool = pool[pool.doc_id != anchor.doc_id].copy()
    if len(pool) == 0:
        raise RuntimeError("No donor candidates")
    pool["len_dist"] = (pool.b_words - anchor.b_words).abs()
    pool = pool.sort_values(["len_dist","category_score"], ascending=[True,False]).head(min(12,len(pool)))
    return pool.iloc[rng.randrange(len(pool))]

def build_pairs(docs: pd.DataFrame, seed: int):
    rng = random.Random(seed)
    out = []
    docs = docs.reset_index(drop=True)

    for aid, a in docs.iterrows():
        same = pick(a, docs[docs.category == a.category], rng)
        other = pick(a, docs[docs.category != a.category], rng)

        base = {
            "anchor_id": aid,
            "anchor_doc_id": a.doc_id,
            "anchor_category": a.category,
            "anchor_title": a.title,
            "A": a.A,
            "true_B": a.original_B,
            "true_B_words": a.b_words,
        }
        candidates = [
            ("original", a, a.original_B, 1),
            ("same_subtopic", same, same.original_B, 0),
            ("other_subtopic", other, other.original_B, 0),
        ]
        for condition, donor, B, gold in candidates:
            out.append({
                **base,
                "condition": condition,
                "B": B,
                "B_doc_id": donor.doc_id,
                "B_category": donor.category,
                "B_title": donor.title,
                "B_words": len(B.split()),
                "gold_is_next": gold,
            })

    return pd.DataFrame(out)

@torch.inference_mode()
def score_pairs(df, model_name, batch_size, max_length, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = BertForNextSentencePrediction.from_pretrained(model_name).to(device).eval()

    scores = []
    for i in range(0, len(df), batch_size):
        x = df.iloc[i:i+batch_size]
        enc = tok(
            x.A.tolist(),
            x.B.tolist(),
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k:v.to(device) for k,v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:,0]  # class 0 = IsNext
        scores.extend(probs.cpu().numpy().tolist())

    z = df.copy()
    z["isnext_score"] = scores
    z["pred_is_next"] = (z.isnext_score >= 0.5).astype(int)
    return z

def rank_metric(pos, neg):
    return float(np.mean((pos > neg).astype(float) + 0.5*(pos == neg).astype(float)))

def metrics_from_arrays(o, s, x):
    return {
        "TPR_original": float(np.mean(o >= 0.5)),
        "FAR_same_subtopic": float(np.mean(s >= 0.5)),
        "FAR_other_subtopic": float(np.mean(x >= 0.5)),
        "Delta_FAR": float(np.mean(s >= 0.5) - np.mean(x >= 0.5)),
        "Rank_same": rank_metric(o, s),
        "Rank_other": rank_metric(o, x),
        "SameScore_gt_OtherScore": float(np.mean(s > x) + 0.5*np.mean(s == x)),
        "MeanScore_original": float(np.mean(o)),
        "MeanScore_same": float(np.mean(s)),
        "MeanScore_other": float(np.mean(x)),
        "MeanScoreGap_same_minus_other": float(np.mean(s-x)),
    }

def analyze(scored, nboot, seed):
    p = scored.pivot(index="anchor_id", columns="condition", values="isnext_score")
    o = p.original.to_numpy()
    s = p.same_subtopic.to_numpy()
    x = p.other_subtopic.to_numpy()
    m = metrics_from_arrays(o,s,x)

    rng = np.random.default_rng(seed)
    rec = []
    for b in range(nboot):
        idx = rng.integers(0, len(p), len(p))
        mm = metrics_from_arrays(o[idx], s[idx], x[idx])
        mm["bootstrap_id"] = b
        rec.append(mm)
    boot = pd.DataFrame(rec)

    summary = []
    for k,v in m.items():
        summary.append({
            "metric":k,
            "estimate":v,
            "ci_low":float(boot[k].quantile(0.025)),
            "ci_high":float(boot[k].quantile(0.975)),
        })
    return m, boot, pd.DataFrame(summary)

def category_metrics(scored):
    rows = []
    for cat, g in scored.groupby("anchor_category"):
        p = g.pivot(index="anchor_id", columns="condition", values="isnext_score")
        o = p.original.to_numpy(); s = p.same_subtopic.to_numpy(); x = p.other_subtopic.to_numpy()
        m = metrics_from_arrays(o,s,x)
        m["category"] = cat
        m["n"] = len(p)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("category")

def donor_stats(pairs):
    rows = []
    for cond in ["same_subtopic","other_subtopic"]:
        g = pairs[pairs.condition == cond]
        vc = g.B_doc_id.value_counts()
        rows.append({
            "condition":cond,
            "n_pairs":len(g),
            "unique_donors":int(vc.size),
            "max_reuse":int(vc.max()),
            "donors_reused_2plus":int((vc >= 2).sum()),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="week3/bert_nsp_shipping_results")
    ap.add_argument("--per-category", type=int, default=40)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="google-bert/bert-base-uncased")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    docs = collect(args.per_category, args.seed, args.year)
    docs.to_csv(out/"documents.csv", index=False)

    pairs = build_pairs(docs, args.seed+1)
    pairs.to_csv(out/"pairs.csv", index=False)
    donor_stats(pairs).to_csv(out/"donor_stats.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scored = score_pairs(pairs, args.model, args.batch, args.maxlen, device)
    scored.to_csv(out/"pair_scores.csv", index=False)

    m, boot, summary = analyze(scored, args.bootstrap, args.seed+2)
    boot.to_csv(out/"bootstrap.csv", index=False)
    summary.to_csv(out/"summary.csv", index=False)
    category_metrics(scored).to_csv(out/"category_metrics.csv", index=False)

    paired = scored.pivot(index="anchor_id", columns="condition", values=["isnext_score","pred_is_next"])
    paired.to_csv(out/"paired_scores.csv")

    top_same = scored[scored.condition=="same_subtopic"].sort_values("isnext_score", ascending=False).head(15)
    top_other = scored[scored.condition=="other_subtopic"].sort_values("isnext_score", ascending=False).head(15)
    top_same.to_csv(out/"top_same_false_candidates.csv", index=False)
    top_other.to_csv(out/"top_other_false_candidates.csv", index=False)

    meta = {
        "dataset": DATASET_ID,
        "split":"train",
        "year_filter": args.year,
        "categories": list(CATEGORIES),
        "n_docs":len(docs),
        "n_pairs":len(pairs),
        "model":args.model,
        "device":device,
        "seed_docs":args.seed,
        "seed_negatives":args.seed+1,
        "seed_bootstrap":args.seed+2,
        "metrics":m,
    }
    (out/"summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = [
        "# BERT NSP shipping-domain hard-negative experiment","",
        f"- Dataset: {DATASET_ID}",
        f"- Year filter: {args.year}",
        f"- Documents: {len(docs)} ({args.per_category} × {len(CATEGORIES)} subtopics)",
        f"- Pairs: {len(pairs)}",
        f"- Model: {args.model}",
        f"- Device: {device}","",
        "## Subtopic counts","",
        docs.category.value_counts().sort_index().to_string(),"",
        "## Main metrics","",
        summary.to_markdown(index=False),"",
        "## Category metrics","",
        category_metrics(scored).to_markdown(index=False),"",
        "## Donor reuse","",
        donor_stats(pairs).to_markdown(index=False),"",
        "## Highest-scoring same-subtopic false candidates",""
    ]
    for _,r in top_same.head(10).iterrows():
        report += [
            f"### anchor {int(r.anchor_id)} · score={r.isnext_score:.6f}",
            f"- category: {r.anchor_category}",
            f"- anchor title: {r.anchor_title}",
            f"- donor title: {r.B_title}",
            f"- A: {r.A}",
            f"- false B: {r.B}",
            f"- true B: {r.true_B}",
            ""
        ]
    (out/"REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(summary.to_string(index=False))
    print(category_metrics(scored).to_string(index=False))
    print(donor_stats(pairs).to_string(index=False))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
