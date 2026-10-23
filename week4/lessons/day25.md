# Day 25. Self-Attention 직접 구현 + Causal Mask

커리큘럼 6-4(Self-Attention 직접 구현), 6-5(Causal Mask 적용)를 하루에 합니다.
mini GPT의 심장이라서 오늘은 조금 길어요. 천천히 shape를 하나씩 찍어가며 진행하세요.

## 오늘의 목표

- `Head` 클래스 하나로 **Causal Self-Attention**을 처음부터 끝까지 직접 구현한다
- 각 단계의 shape를 말로 설명할 수 있다
- 마스크가 미래 Token을 정말 가리는지 테스트로 확인한다

## 오늘 만들 파일

| 파일 | 내용 |
|---|---|
| `minigpt/attention.py` | `Head` 클래스 (TODO 7개) |
| `tests/test_day25.py` | 확인용 테스트 (이미 있음) |

## 개념 복습

오늘 구현할 공식은 이것 하나예요.

```text
Attention(Q, K, V) = softmax(QKᵀ / √dk + mask) V
```

| 단계 | 배운 곳 |
|---|---|
| Q, K, V를 Linear로 만들기 | `week2/day12.ipynb` |
| `QKᵀ`와 `transpose(-2, -1)` | `week1/day02.ipynb` 5절, `week2/day12.ipynb` |
| `√dk`로 나누는 이유 | `week2/day13.ipynb` |
| `tril` 마스크, `-inf`, `register_buffer` | `week2/day14.ipynb` |
| Softmax와 `dim=-1` | `week1/day05.ipynb` |

## 단계별 구현 가이드

예시 크기: `B=2, T=8, C=n_embd=32, head_size=8, block_size=8`

### `__init__`

**TODO 1. Key / Query / Value Linear**

```text
nn.Linear(n_embd, head_size, bias=False)  × 3개
이름: self.key, self.query, self.value
```

Attention에서는 보통 bias를 쓰지 않아요. 테스트가 bias가 없는지도 확인합니다.

**TODO 2. `tril` buffer**

`(block_size, block_size)` 크기의 아래 삼각형(대각선 포함) 1 행렬을 `"tril"`이라는 이름으로 `register_buffer` 하세요.

> 💡 buffer는 `model.to("mps")` 할 때 같이 옮겨지지만 Optimizer가 학습하지는 않아요. 마스크처럼 "고정된 값"에 딱 맞아요.

마지막 줄의 `raise NotImplementedError(...)`는 다 채운 뒤 **지워야** 해요. (forward도 마찬가지)

### `forward`

| 단계 | 하는 일 | 결과 shape |
|---|---|---|
| TODO 3 | `k = self.key(x)`, q, v도 같은 방식 | `(B, T, hs)` 각각 |
| TODO 4 | `q @ k.transpose(-2, -1)`에 `head_size ** -0.5` 곱하기 | `(B, T, T)` |
| TODO 5 | `self.tril[:T, :T] == 0`인 칸을 `float("-inf")`로 (`masked_fill`) | `(B, T, T)` |
| TODO 6 | `F.softmax(..., dim=-1)` → `self.dropout(...)` | `(B, T, T)` |
| TODO 7 | weight `@ v` | `(B, T, hs)` |

### 자주 하는 실수

| 실수 | 증상 |
|---|---|
| `k.T` 사용 | batch 축까지 뒤집혀서 shape 에러 또는 이상한 결과 |
| `√C`(n_embd)로 나눔 | 기준 결과와 숫자가 달라짐. `√head_size`가 맞아요 |
| 마스크를 `[:T, :T]`로 안 자름 | T < block_size일 때 shape 에러 |
| softmax **후에** 0으로 지움 | 행 합이 1이 안 됨 (`week2/day14.ipynb` 참고) |
| `softmax(dim=1)` | 에러 없이 틀린 결과. 꼭 `dim=-1` |
| `masked_fill`의 결과를 다시 저장 안 함 | `masked_fill`은 새 텐서를 돌려줘요. `wei = wei.masked_fill(...)` |

## 실행해서 확인하기

`week4` 폴더에서 실행하세요.

```bash
../.venv/bin/python -m pytest tests/test_day25.py -v
../.venv/bin/python -m minigpt.attention
```

테스트 7개가 확인하는 것:

1. Q/K/V가 bias 없는 `nn.Linear(n_embd, head_size)`인가
2. `tril`이 parameter가 아니라 buffer인가, 모양이 맞는가
3. 출력 shape `(B, T, head_size)`
4. T가 block_size보다 짧아도 동작하는가
5. `F.scaled_dot_product_attention(..., is_causal=True)`와 결과가 같은가
6. 마지막 Token만 바꿨을 때 앞쪽 출력이 그대로인가 (커닝 테스트)
7. 첫 Token의 출력이 자기 Value와 같은가

데모를 실행하면 이렇게 나오면 성공이에요.

```text
x: (2, 8, 32) → out: (2, 8, 8)
```

## 막혔을 때

먼저 각 줄 다음에 `print(q.shape)`처럼 shape를 찍어보세요. 그래도 안 되면 `solutions/minigpt/attention.py`와 비교해보세요. 단, 최소 한 번은 직접 끝까지 시도한 뒤에 보는 걸 추천해요.

## 체크 질문

1. `wei`의 shape가 `(B, T, T)`인 이유는? `wei[b, i, j]`는 무엇을 뜻하나요?
2. 왜 `√head_size`로 나누나요? 안 나누면 어떤 일이 생기나요?
3. `tril`을 parameter가 아니라 buffer로 등록하는 이유는?
4. 마스크를 softmax **전에** 적용해야 하는 이유는?
5. 첫 번째 Token의 출력이 자기 Value와 똑같은 이유는?

**내 답:**

1.
2.
3.
4.
5.
