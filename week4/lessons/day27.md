# Day 27. Transformer Block + GPT Model 구현

## 오늘의 목표

- FFN(`FeedForward`)을 만든다 (커리큘럼 6-7)
- Attention과 FFN을 Pre-LN 방식으로 묶은 `Block`을 만든다 (커리큘럼 6-8)
- Embedding, Block × N, LayerNorm, `lm_head`를 이어 **GPT 모델 전체**를 완성한다 (커리큘럼 6-9)

오늘이 끝나면 `model(idx, targets)` 한 줄로 logits와 loss가 나와요. 내일은 이걸 학습시키기만 하면 돼요.

## 오늘 만들 파일

| 파일 | 채울 곳 | 쓰는 부품 |
|---|---|---|
| `minigpt/block.py` | `FeedForward.__init__/forward`, `Block.__init__/forward` | Day 26 `MultiHeadAttention` |
| `minigpt/model.py` | `GPT.__init__`, `GPT.forward` | Day 24 `GPTEmbedding`, 오늘 만든 `Block` |

`GPTConfig`, `_init_weights`, `num_params`는 이미 완성돼 있어요. 그대로 두세요.

> 💡 `Block`은 Day 25~26의 Attention을, `GPT`는 Day 24의 Embedding을 가져다 써요. 앞 Day 테스트가 통과한 상태여야 오늘 테스트도 통과해요.

## 개념 복습

| 개념 | 한 줄 요약 | 다시 볼 곳 |
|---|---|---|
| FFN | Token마다 따로 C → 4C → C, 사이에 GELU | `week3/day17.ipynb`, `week1/day04.ipynb` |
| Residual | `x + f(x)`: 원래 값은 두고 바뀐 만큼만 더하기 | `week3/day16.ipynb` |
| LayerNorm | Token 벡터 하나(C축)를 평균 0, 분산 1로 | `week3/day16.ipynb` |
| Pre-LN Block | LayerNorm을 attn/ffwd **안쪽**에 | `week3/day15.ipynb` |
| GPT 구조 | Embedding → Block × N → ln_f → lm_head | `week3/day19.ipynb` |
| Cross Entropy | `(B, T, V)`를 `(B*T, V)`로 펴서 계산 | `week1/day06.ipynb` |

**커리큘럼 6-8 그림에 대해:** 커리큘럼 그림은 `Attention → Residual → FFN → Residual`로 LayerNorm이 빠져 있어요. 실제 GPT-2 이후 모델은 LayerNorm을 넣고, 그것도 **Pre-LN** 방식으로 넣어요. 우리도 Pre-LN을 써요.

```text
커리큘럼 6-8 (LayerNorm 없음)      우리가 만들 것 (Pre-LN)
x → Attention → +x                 x = x + attn(ln1(x))
  → FFN       → +x                 x = x + ffwd(ln2(x))
```

## 단계별 구현 가이드

### 1. `FeedForward`

`self.net` 하나에 4개를 순서대로 넣어요.

```text
(B, T, C) → Linear(C, 4C) → (B, T, 4C) → GELU → Linear(4C, C) → (B, T, C) → Dropout
```

`forward`는 `self.net(x)`를 돌려주면 끝이에요.

자주 하는 실수:
- GELU를 빼먹기. Linear 두 개가 하나로 합쳐져서 늘렸다 줄이는 의미가 없어져요 (Day 4).
- 두 번째 Linear를 `Linear(4C, 4C)`로 만들기. 출력이 C가 아니면 Residual 덧셈에서 shape 에러가 나요.

### 2. `Block.__init__`

부품 4개를 **정해진 이름으로** 만드세요. 테스트와 `GPT`가 이 이름을 그대로 써요.

| 이름 | 무엇 |
|---|---|
| `self.ln1` | `nn.LayerNorm(n_embd)` |
| `self.attn` | `MultiHeadAttention(n_embd, n_head, block_size, dropout)` |
| `self.ln2` | `nn.LayerNorm(n_embd)` |
| `self.ffwd` | `FeedForward(n_embd, dropout)` |

### 3. `Block.forward`

딱 두 줄이에요.

```text
x: (B, T, C)
① ln1(x)            → (B, T, C)   정규화
② attn(①)           → (B, T, C)   Token끼리 소통
③ x = x + ②         → (B, T, C)   Residual
④ ln2(x) → ffwd     → (B, T, C)   Token마다 생각
⑤ x = x + ④         → (B, T, C)   Residual
```

자주 하는 실수:
- `x = self.ln1(x + self.attn(x))`처럼 쓰기. 이건 Post-LN이에요. 테스트 `test_block_matches_pre_ln_formula`가 잡아내요.
- `x = self.attn(self.ln1(x))`처럼 `x +`를 빼먹기. 정보가 사라지고 깊게 쌓으면 학습이 안 돼요. `test_block_residual`이 잡아내요.
- 두 번째 줄에서 `ln1`을 다시 쓰기. LayerNorm마다 학습되는 γ/β가 따로 있어서 두 개가 필요해요.

### 4. `GPT.__init__`

| 이름 | 무엇 | 출력 shape |
|---|---|---|
| `self.embedding` | `GPTEmbedding(vocab_size, n_embd, block_size, dropout)` | `(B, T, C)` |
| `self.blocks` | `nn.Sequential(*[Block(...) for _ in range(n_layer)])` | `(B, T, C)` |
| `self.ln_f` | `nn.LayerNorm(n_embd)` | `(B, T, C)` |
| `self.lm_head` | `nn.Linear(n_embd, vocab_size)` | `(B, T, V)` |

값은 전부 `config.xxx`에서 꺼내 쓰세요. 다 만든 뒤 `self.apply(self._init_weights)`가 실행되도록 `raise` 줄만 지우면 돼요.

`nn.Sequential`에 리스트를 넣을 때는 `*`로 풀어줘야 해요. `nn.Sequential([...])`는 에러가 나요.

### 5. `GPT.forward`

```text
idx      (B, T)       Token ID
embedding(idx)        → (B, T, C)
blocks(...)           → (B, T, C)
ln_f(...)             → (B, T, C)
lm_head(...)          → (B, T, V)   = logits
```

loss는 `targets`가 있을 때만 계산해요.

```text
logits  (B, T, V)  →  view(B*T, V)
targets (B, T)     →  view(B*T)
F.cross_entropy(펴진 logits, 펴진 targets)
```

`targets`가 `None`이면 `loss = None`으로 두고 `return logits, loss`로 둘 다 돌려주세요. 생성할 때(Day 28)는 targets 없이 부르거든요.

자주 하는 실수:
- `ln_f`를 빼먹기. Pre-LN은 마지막 Block의 출력이 정규화되지 않은 상태라 한 번 더 정규화해야 해요 (`week3/day19.ipynb`).
- `F.cross_entropy(logits, targets)`처럼 3차원 그대로 넣기. 에러가 나거나 축이 뒤바뀐 엉뚱한 값이 나와요.
- softmax를 먼저 씌우고 cross_entropy에 넣기. `F.cross_entropy`는 **logits**를 받아요 (Day 6).

## 실행해서 확인하기

week4 폴더에서 실행해요.

```bash
cd week4
../.venv/bin/python -m pytest tests/test_day27.py -v
```

테스트가 확인하는 것:

| 테스트 | 확인 내용 |
|---|---|
| `test_feedforward_*` | 출력 shape, 가운데가 4C인지, GELU가 있는지 |
| `test_block_shape` / `has_parts` | 입력 shape = 출력 shape, 부품 이름 |
| `test_block_residual` | attn, ffwd가 0을 내면 Block이 입력을 그대로 통과시키는지 |
| `test_block_matches_pre_ln_formula` | Pre-LN 순서 |
| `test_block_is_causal` / `test_gpt_is_causal` | 미래 Token을 바꿔도 앞쪽 출력이 그대로인지 |
| `test_gpt_initial_loss_near_log_vocab` | 학습 전 loss ≈ log(65) ≈ 4.17 |
| `test_gpt_loss_matches_cross_entropy` | loss 계산 방식 |

모델을 직접 돌려볼 수도 있어요.

```bash
../.venv/bin/python -m minigpt.block
../.venv/bin/python -m minigpt.model
```

`minigpt.model`을 실행하면 이런 식으로 나오면 성공이에요.

```text
params: 29761
logits: (2, 8, 65) | loss: 4.142 | log(V): 4.174
```

## 막혔을 때

먼저 테스트 메시지를 읽고, 위 가이드의 shape 표와 비교해 보세요. 그래도 안 되면 완성본과 비교하세요.

- `solutions/minigpt/block.py`
- `solutions/minigpt/model.py`

완성본으로 테스트가 통과하는지 확인하려면:

```bash
MINIGPT_IMPL=solutions ../.venv/bin/python -m pytest tests/test_day27.py -v
```

## 체크 질문

1. `Block`의 입력 shape와 출력 shape가 같아야 하는 이유는?
2. Pre-LN과 Post-LN은 코드로 어떻게 다른가? 우리는 왜 Pre-LN을 쓰는가?
3. `test_block_residual`에서 attn과 ffwd의 Weight를 전부 0으로 만들었는데 출력이 입력과 같았다. 왜 그런가?
4. `GPT.forward`에서 `ln_f`는 왜 필요한가?
5. 학습 전 loss가 log(vocab_size) 근처로 나오는 이유는?

**내 답:**

1.
2.
3.
4.
5.
