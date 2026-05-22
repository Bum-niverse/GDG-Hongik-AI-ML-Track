# Week 5 Assignment - 하이퍼파라미터 튜닝과 K-Fold

## 📌1. 목표

이번 주차에서는 와인 데이터를 사용해서 결정트리 모델을 학습시키고, 하이퍼파라미터 튜닝과 K-Fold 교차검증을 실습하였다.

지난 주차에서는 로지스틱 회귀와 SGDClassifier를 통해 분류 모델을 학습하고, 에포크 수에 따라 정확도가 달라지는 것을 확인했다. 이번 주차에서는 모델 자체의 설정값인 하이퍼파라미터를 바꿔가며 성능이 어떻게 달라지는지 확인하였다.

이번 과제의 핵심은 다음과 같다.

- 와인 데이터셋을 불러온다.
- 입력 데이터와 정답 데이터를 분리한다.
- Train set, Validation set, Test set을 나눈다.
- DecisionTreeClassifier를 사용해 모델을 학습한다.
- `max_depth` 값을 바꿔가며 성능을 비교한다.
- K-Fold 교차검증을 사용해 여러 번 검증 점수를 확인한다.
- K값에 따라 평균 검증 점수가 어떻게 달라지는지 확인한다.

이번 주차에서는 단순히 모델을 한 번 학습시키고 테스트 점수만 확인하는 것이 아니라, 검증 세트를 따로 만들고 여러 조건을 비교하면서 더 적절한 모델 설정을 찾는 과정을 학습하였다.

* * *

## 📌2. 데이터 준비

이번 실습에서는 와인 데이터를 사용하였다.

```python
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')
```

여기서 입력 데이터는 모델이 와인을 분류하기 위해 참고하는 값이고, 타깃 데이터는 모델이 맞춰야 하는 정답이다.

즉, 알코올 도수, 당도, pH 값을 보고 와인의 클래스를 예측하는 문제이다.

먼저 전체 데이터를 훈련 세트와 테스트 세트로 나누었다.

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.2,
    random_state=42
)
```

훈련 세트는 모델을 학습시키는 데 사용하고, 테스트 세트는 최종 성능을 확인하는 데 사용한다.

이전 주차에서도 훈련 데이터와 테스트 데이터를 나누는 것이 중요하다고 배웠다. 모델이 이미 본 데이터로 평가하면 실제보다 성능이 좋게 나올 수 있기 때문에, 테스트 세트는 마지막 평가용으로 따로 두는 것이 필요하다.

* * *

##  📌4. 검증 세트

이번 주차에서 새롭게 다룬 개념 중 하나는 검증 세트이다.

```python
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input,
    train_target,
    test_size=0.2,
    random_state=42
)
```

기존에는 훈련 세트와 테스트 세트만 나누었지만, 이번에는 훈련 세트를 다시 나누어 실제 학습에 사용할 데이터와 검증에 사용할 데이터를 만들었다.

정리하면 다음과 같다.

Train set: 모델을 학습시키는 데이터
Validation set: 하이퍼파라미터를 비교하기 위한 중간 평가 데이터
Test set: 최종 성능을 확인하는 데이터

검증 세트가 필요한 이유는 테스트 세트를 계속 사용하면 테스트 세트에 맞춰서 모델을 고르게 될 수 있기 때문이다. 그래서 여러 설정을 비교할 때는 검증 세트를 사용하고, 테스트 세트는 마지막에 한 번 확인하는 것이 더 적절하다고 이해했다.

* * *

##  📌5. 결정트리 모델

이번 실습에서는 결정트리 모델을 사용하였다.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
```

결정트리는 데이터를 조건에 따라 계속 나누면서 분류하는 모델이다.

예를 들어 와인 데이터에서는 당도, 알코올 도수, pH 같은 특성을 기준으로 데이터를 나누고, 최종적으로 어떤 클래스에 속하는지 판단한다.

결정트리는 구조를 시각화할 수 있다는 장점이 있다.

```python
from sklearn.tree import plot_tree

plot_tree(dt)
```

트리를 직접 확인하면 모델이 어떤 특성을 기준으로 데이터를 나누는지 볼 수 있다. 그래서 다른 모델보다 예측 과정을 이해하기 쉽다고 느꼈다.

* * *

##  📌6. max_depth

과제 1에서는 결정트리의 max_depth 값을 바꿔가며 성능을 비교하였다.

```python
max_depths = [1, 3, 5, 10]
```

max_depth는 결정트리의 최대 깊이를 제한하는 하이퍼파라미터이다.

```python
for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(sub_input, sub_target)

    train_score = dt.score(sub_input, sub_target)
    val_score = dt.score(val_input, val_target)
    test_score = dt.score(test_input, test_target)

    print("max_depth:", depth)
    print("Train score:", train_score)
    print("Validation score:", val_score)
    print("Test score:", test_score)
    print()
```
max_depth 값이 너무 작으면 트리가 너무 단순해져서 데이터를 충분히 학습하지 못할 수 있다. 이 경우 과소적합이 발생할 수 있다.

반대로 max_depth 값이 너무 크면 트리가 너무 복잡해져서 훈련 데이터에만 지나치게 맞춰질 수 있다. 이 경우 과대적합이 발생할 수 있다.

그래서 max_depth 값을 여러 개 비교하면서 훈련 점수와 검증 점수를 함께 확인해야 한다.

* * *

##  📌7. 하이퍼파라미터 튜닝

이번 주차에서 중요한 개념은 하이퍼파라미터 튜닝이었다.

하이퍼파라미터는 모델이 직접 학습해서 정하는 값이 아니라, 사람이 미리 정해줘야 하는 설정값이다.

이번 과제에서는 max_depth가 하이퍼파라미터에 해당한다.

```python
DecisionTreeClassifier(max_depth=depth, random_state=42)
```

max_depth를 1, 3, 5, 10으로 바꿔가며 점수를 비교해보면, 같은 데이터와 같은 모델을 사용하더라도 설정값에 따라 성능이 달라지는 것을 확인할 수 있다.

이번 실습을 통해 모델을 선택하는 것만큼 모델의 설정값을 잘 정하는 것도 중요하다는 것을 알게 되었다.

* * *

##  📌8. K-Fold 교차검증

과제 2에서는 K-Fold 교차검증을 사용하였다.

```python
from sklearn.model_selection import cross_validate, KFold
```

K-Fold 교차검증은 훈련 데이터를 K개의 부분으로 나눈 뒤, 그중 하나를 검증 세트로 사용하고 나머지를 훈련 세트로 사용하는 과정을 반복하는 방법이다.

이번 과제에서는 K값을 3, 5, 10으로 설정하였다.

```python
k_values = [3, 5, 10]
```

각 K값에 대해 교차검증을 수행하였다.

```python
for k in k_values:
    splitter = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_validate(
        dt,
        train_input,
        train_target,
        cv=splitter
    )

    print("K:", k)
    print("Fold scores:", scores['test_score'])
    print("Mean score:", np.mean(scores['test_score']))
    print()
```

여기서 Fold scores는 각 Fold에서 나온 검증 점수이고, Mean score는 그 점수들의 평균이다.

K-Fold를 사용하면 한 번 나눈 검증 세트에만 의존하지 않고, 여러 번 검증한 결과를 평균으로 볼 수 있다. 그래서 모델 성능을 조금 더 안정적으로 판단할 수 있다.

* * *

## 📌9. K값 비교

이번 과제에서는 K값을 3, 5, 10으로 바꿔가며 평균 검증 점수를 확인하였다.

K값이 작으면 한 번에 사용하는 검증 세트의 크기가 커지고, 반복 횟수는 적어진다.
반대로 K값이 커지면 더 여러 번 검증할 수 있지만, 계산량이 늘어난다.

이번 실습에서는 K값을 바꿨을 때 Fold별 점수와 평균 점수가 어떻게 달라지는지 확인하였다.

```python
print("Fold scores:", scores['test_score'])
print("Mean score:", np.mean(scores['test_score']))
```

K-Fold를 사용하면서 한 번의 검증 점수만 보고 모델을 판단하는 것보다 여러 번 검증한 평균 점수를 보는 것이 더 안정적이라고 느꼈다.

* * *

##  📌10. 정리

이번 주차에서는 와인 데이터를 사용해서 결정트리 모델을 학습하고, 하이퍼파라미터 튜닝과 K-Fold 교차검증을 실습하였다.

결정트리는 데이터를 조건에 따라 나누면서 분류하는 모델이고, max_depth를 통해 트리의 깊이를 제한할 수 있었다.
max_depth 값이 너무 작으면 모델이 단순해져서 과소적합이 발생할 수 있고, 너무 크면 훈련 데이터에 너무 맞춰져 과대적합이 발생할 수 있다.

이번 주차에서 정리할 내용은 다음과 같다.

와인 데이터에서는 alcohol, sugar, pH를 입력 데이터로 사용한다.
class는 모델이 맞춰야 하는 정답 데이터이다.
Train set은 모델 학습에 사용한다.
Validation set은 하이퍼파라미터를 비교할 때 사용한다.
Test set은 최종 성능 확인에 사용한다.
결정트리는 조건을 기준으로 데이터를 나누며 분류하는 모델이다.
max_depth는 결정트리의 최대 깊이를 제한하는 하이퍼파라미터이다.
max_depth가 너무 작으면 과소적합이 발생할 수 있다.
max_depth가 너무 크면 과대적합이 발생할 수 있다.
K-Fold 교차검증은 데이터를 K개로 나누어 여러 번 검증하는 방법이다.
Fold별 점수를 평균내면 모델 성능을 더 안정적으로 확인할 수 있다.

결국 이번 주차의 핵심은 모델의 성능을 한 번의 점수만 보고 판단하지 않고, 검증 세트와 교차검증을 사용해서 더 적절한 모델 설정을 찾는 것이었다.

* * *

## 📌11. 느낀점

이번 주차에서는 결정트리 모델을 사용해서 와인 데이터를 분류하고, max_depth 값과 K-Fold 교차검증에 대해 실습하였다.

가장 먼저 새롭게 느껴졌던 부분은 검증 세트였다. 이전에는 주로 훈련 세트와 테스트 세트만 나누어서 모델을 학습하고 평가했는데, 이번에는 훈련 세트를 다시 나누어 검증 세트를 만들었다. 처음에는 테스트 세트로 평가하면 되지 않나 싶었지만, 하이퍼파라미터를 비교할 때 테스트 세트를 계속 사용하면 결국 테스트 세트에 맞는 모델을 고르게 될 수 있다는 점을 이해하게 되었다.

max_depth를 바꿔가며 결정트리의 성능을 비교한 것도 기억에 남았다. max_depth가 작으면 모델이 너무 단순해지고, 너무 크면 훈련 데이터에만 과하게 맞춰질 수 있었다. 그래서 모델을 복잡하게 만드는 것이 항상 좋은 것은 아니라는 것을 다시 확인할 수 있었다.

K-Fold 교차검증도 처음에는 조금 복잡하게 느껴졌지만, 여러 번 나누어 검증하고 그 평균을 본다는 방식으로 이해하니 납득이 되었다. 한 번 나눈 검증 세트의 점수만 보면 우연히 높거나 낮게 나올 수 있는데, K-Fold를 사용하면 여러 번의 검증 결과를 평균으로 볼 수 있어서 더 안정적인 평가가 가능하다고 느꼈다.

이번 과제를 하면서 모델 성능을 확인할 때 단순히 테스트 정확도 하나만 보면 부족하다는 것을 알게 되었다. 앞으로는 모델을 학습할 때 훈련 점수, 검증 점수, 테스트 점수를 구분해서 보고, 하이퍼파라미터를 바꿨을 때 성능이 어떻게 달라지는지도 같이 확인해야겠다고 생각했다.
