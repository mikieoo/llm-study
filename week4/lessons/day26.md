# Day 26. Multi-Head Attention 구현

## 오늘의 목표
- Day 25에서 만든 `Head`를 여러 개 병렬로 돌리는 `MultiHeadAttention`을 완성해요
- `nn.ModuleList`가 왜 필요한지 알아요
- Head 결과를 이어 붙이는(concat) 축과 shape 변화를 설명할 수 있어요

## 오늘 만들 파일
| 파일 | 내용 |
|---|---|
| `minigpt/multihead.py` | `MultiHeadAttention` (`__init__`, `forward`) |
| `tests/test_day26.py` | 확인용 테스트 (이미 있어요) |

> ⚠️ 이 파일은 `from .attention import Head`로 **Day 25의 `Head`를 그대로 가져다 써요.** Day 25 테스트가 통과한 상태여야 오늘 테스트도 통과할 수 있어요.

## 개념 복습
자세한 내용은 `week2/day14.ipynb` 3절에 있어요.

```text
x (B, T, C)
 ├─ Head 1 → (B, T, hs)
 ├─ Head 2 → (B, T, hs)
 ├─ Head 3 → (B, T, hs)
 └─ Head 4 → (B, T, hs)
        ↓ concat (dim=-1)
     (B, T, 4·hs) = (B, T, C)
        ↓ proj (Linear C → C)
     (B, T, C)
```

- Head마다 서로 다른 Wq, Wk, Wv를 가지니까 **서로 다른 관계**(바로 앞 글자, 같은 단어의 앞부분 등)를 볼 수 있어요.
- `head_size = n_embd // n_head`로 두면 이어 붙였을 때 다시 `C`가 돼서, Block에 넣고 Residual(`x + attn(x)`)을 더할 수 있어요.
- 마지막 `proj`는 여러 Head가 가져온 정보를 한 번 더 섞어 주는 Linear예요.

## 단계별 구현 가이드

### `__init__`
| TODO | 할 일 | 힌트 |
|---|---|---|
| 1 | `head_size` 계산 | `C=32, n_head=4`이면 `8` |
| 2 | `self.heads` | `nn.ModuleList([Head(...) for _ in range(n_head)])` 형태 |
| 3 | `self.proj` | 입력도 출력도 `n_embd` |
| 4 | `self.dropout` | `nn.Dropout(dropout)` |

**흔한 실수: 파이썬 `list`에 Head를 담기**
```python
self.heads = [Head(...) for _ in range(n_head)]   # ❌
```
이렇게 하면 에러는 안 나지만, PyTorch가 안에 든 Head를 **하위 모듈로 인식하지 못해요.** 그러면
- `model.parameters()`에 Head의 Weight가 빠져서 Optimizer가 학습을 안 시키고
- `model.to("mps")`를 해도 Head는 CPU에 남아요.

`nn.ModuleList`는 "모듈을 담는 리스트"라서 이 문제가 없어요.

### `forward`
```text
① 각 Head에 x 넣기          h(x)            → (B, T, hs) × n_head
② 마지막 축으로 이어 붙이기  torch.cat(…, dim=-1) → (B, T, C)
③ proj                        → (B, T, C)
④ dropout                     → (B, T, C)
```

**흔한 실수: 이어 붙이는 축**
- `dim=0`으로 붙이면 batch 축이 늘어나서 `(n_head·B, T, hs)`가 돼요.
- `dim=1`로 붙이면 Token 축이 늘어나서 `(B, n_head·T, hs)`가 돼요.
- 우리가 원하는 건 **Token마다 벡터를 길게** 만드는 것이니까 `dim=-1`이에요.

## 실행해서 확인하기
`week4` 폴더에서 실행하세요.

```bash
../.venv/bin/python -m pytest tests/test_day26.py -v
../.venv/bin/python -m minigpt.multihead
```

두 번째 명령은 이렇게 나오면 돼요.
```text
x: (2, 8, 32) → out: (2, 8, 32)
```

테스트가 확인하는 것:
- 출력 shape `(B, T, C)`, block_size보다 짧은 입력도 동작
- `heads`가 `nn.ModuleList`이고 Head가 `n_head`개, 각 Head 출력 크기가 `n_embd // n_head`
- `proj`가 `(C → C)` Linear
- 출력이 `proj(concat(모든 Head))`와 같은지
- 미래 Token을 바꿔도 앞쪽 출력이 그대로인지 (Causal)
- `n_embd`가 `n_head`로 안 나누어떨어지면 `AssertionError`

## 도전 과제 (선택)
`week2/day14.ipynb`의 **구현 B**처럼 Head를 따로 만들지 않고 `nn.Linear(C, 3C)` 하나로 Q/K/V를 한 번에 만든 뒤 `view` / `transpose`로 Head를 나누는 버전을 만들어 보세요. 결과는 같고 GPU에서 더 빨라요. nanoGPT가 이 방식을 써요. 오늘 테스트와 별개로, 다른 파일에서 연습하면 돼요.

## 막혔을 때
먼저 15분 정도 직접 해본 다음 `solutions/minigpt/multihead.py`와 비교해 보세요. 완성본으로 테스트를 돌려 보려면:

```bash
MINIGPT_IMPL=solutions ../.venv/bin/python -m pytest tests/test_day26.py -v
```

## 체크 질문
1. `head_size`를 `n_embd // n_head`로 정하는 이유는?
2. Head들을 파이썬 `list`가 아니라 `nn.ModuleList`에 담아야 하는 이유는?
3. `torch.cat`을 `dim=-1`로 하는 이유는? `dim=1`로 하면 shape가 어떻게 되나요?
4. Head를 4개 쓰는 것과 크기가 4배인 Head 1개를 쓰는 것은 파라미터 수가 같은데, 무엇이 다른가요?
5. 마지막 `proj`는 어떤 역할을 하나요?

**내 답:**

1.
2.
3.
4.
5.
