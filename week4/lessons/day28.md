# Day 28. GPT 학습 및 Text Generation

커리큘럼 6-10, 6-11 · 4주차 마지막 날

## 오늘의 목표

- 학습 루프의 핵심 5줄을 직접 채워서 mini GPT를 **진짜로 학습**시킨다
- `generate`를 직접 구현해서 학습한 모델로 **글을 생성**한다
- loss 숫자를 읽을 줄 안다 (시작값, 줄어드는 모습, train과 val의 차이)
- 4주 동안 배운 전체 흐름을 한 번에 설명할 수 있다

## 오늘 만들 파일

| 파일 | 할 일 |
|---|---|
| `minigpt/sampling.py` | `generate()` 반복문 안쪽 (TODO 1~6) |
| `minigpt/trainer.py` | `train()` 안의 학습 한 step (batch 뽑기 + 4줄) |
| `tests/test_day28.py` | 이미 있어요. 내 코드를 검사해요 |

`trainer.py`의 `estimate_loss()`와 평가·기록 부분은 이미 채워져 있어요. 비어 있는 건 핵심 줄뿐이에요.

---

## 개념 복습

### 학습 루프 (week1/day07.ipynb 8절)

```python
x, y = get_batch(...)        # 0. 데이터 한 묶음
_, loss = model(x, y)        # 1. 예측 + loss
optimizer.zero_grad()        # 2. 지난 gradient 지우기
loss.backward()              # 3. gradient 계산
optimizer.step()             # 4. Weight 수정
```

1주차 Day 7에서 소용돌이 데이터를 분류할 때 쓴 루프와 **완전히 같아요.** 모델이 GPT로 바뀌었을 뿐이에요.

### 생성 루프 (week3/day20.ipynb)

```text
지금까지의 Token
 ↓  최근 block_size개만 자르기
 ↓  model → logits (B, T, V)
 ↓  마지막 위치만 (B, V)
 ↓  / temperature → softmax → 확률
 ↓  multinomial로 1개 뽑기
 ↓  뒤에 붙이기
 ↻  반복
```

Day 20에서 Bigram 모델로 했던 것과 같고, 모델만 GPT로 바뀌어요.

---

## 단계별 구현 가이드

### 1. `generate()` (sampling.py)

| TODO | 할 일 | shape |
|---|---|---|
| 1 | `idx_cond = idx[:, -block_size:]` 처럼 뒤에서 block_size개만 | `(B, ≤block_size)` |
| 2 | `logits, _ = model(idx_cond)` | `(B, T, V)` |
| 3 | 마지막 위치만 꺼내고 temperature로 나누기 | `(B, V)` |
| 4 | `top_k`가 있으면 k등보다 작은 점수를 `-inf`로 | `(B, V)` |
| 5 | `F.softmax(…, dim=-1)` → `torch.multinomial(probs, num_samples=1)` | `(B, V)` → `(B, 1)` |
| 6 | `torch.cat([idx, idx_next], dim=1)` | `(B, T+1)` |

**왜 잘라야 하나요?** Position Embedding 표의 크기가 `block_size`라서, 그보다 긴 입력은 모델에 넣을 수 없어요 (week2/day10.ipynb). 생성을 계속하면 문장이 계속 길어지니 매번 최근 것만 잘라 넣어요.

**top_k 힌트**
```python
v, _ = torch.topk(logits, k)          # v: 각 행의 상위 k개 값 (B, k), 내림차순
# v[:, [-1]] 은 k등 값 (B, 1) → 이보다 작은 칸을 -inf로
```
`-inf`는 softmax를 거치면 정확히 0이 돼요 (week1/day05.ipynb 8-1절).

**자주 하는 실수**
- `logits[:, -1]` 대신 `logits[-1]`을 쓰면 **마지막 batch**를 꺼내게 돼요. 축 순서는 `(B, T, V)`예요.
- `argmax`로 고르면 매번 같은 글자만 반복해요. 확률대로 뽑으려면 `multinomial`이에요 (week3/day20.ipynb).
- `torch.cat`의 `dim=1`을 빼먹으면 batch 방향으로 붙어요.

### 2. `train()` (trainer.py)

평가 코드 아래 TODO 자리에 5줄을 넣으면 돼요. `get_batch`는 Day 23에서 만든 함수이고, 마지막 인자로 `device`를 꼭 넘겨야 해요. 모델이 GPU(`mps`)에 있으면 데이터도 같은 곳에 있어야 하거든요.

**자주 하는 실수**
- `zero_grad()`를 빼면 gradient가 계속 쌓여서 학습이 이상해져요 (week1/day07.ipynb 6절).
- `optimizer.step()`을 `loss.backward()`보다 먼저 쓰면 아무것도 안 바뀌어요.

---

## 실행해서 확인하기

모든 명령은 **`week4` 폴더에서** 실행해요.

### 1) 테스트

```bash
../.venv/bin/python -m pytest tests/test_day28.py -v
```

`generate` 테스트는 일부러 아주 작은 가짜 모델을 써서 Day 27이 안 끝났어도 따로 확인할 수 있어요. `train` 테스트는 Day 23의 `get_batch`를 쓰니까 Day 23이 먼저 통과해야 해요.

Day 22~28을 한 번에 확인하려면:

```bash
../.venv/bin/python -m pytest tests/ -v
```

### 2) 진짜 학습: `train.py`

`train.py`가 지금까지 만든 모든 조각을 연결해 줘요.

```text
data/input.txt → CharTokenizer (Day 22) → train/val 나누기 (Day 23)
 → GPT (Day 24~27) → train() (오늘) → out/ckpt.pt 저장 → generate() (오늘)로 맛보기 출력
```

```bash
../.venv/bin/python train.py                     # 내가 만든 minigpt/ 로 학습
../.venv/bin/python train.py --impl solutions    # 완성본으로 학습 (비교용)
```

기본 설정은 `n_embd 128, n_head 4, n_layer 4, block_size 64, batch 32, 3000 step, lr 1e-3`이에요. Apple Silicon에서는 GPU(`mps`)를 자동으로 써요.

**어떤 숫자가 나와야 하나요?**

| 시점 | val loss | 의미 |
|---|---|---|
| step 0 | 약 4.17 | `log(65)`. 65글자 중 아무거나 찍는 수준 (week1/day06.ipynb) |
| 학습 중 | 쭉 내려감 | 단어 모양, 줄바꿈, 이름: 형식을 배우는 중 |
| 끝 | 아래 실측값 참고 | 셰익스피어 "같은" 글을 쓰는 수준 |

**실측값** (완성본 `--impl solutions`, 기본 설정, Apple Silicon `mps`, 2026-10-23)

```text
[data] 1,115,394 글자 | vocab 65 | train 1,003,854 / val 111,540
[model] 816,705 params | device mps | impl solutions
step     0 | train loss 4.2001 | val loss 4.2000
step   500 | train loss 2.0385 | val loss 2.0971
step  1000 | train loss 1.7881 | val loss 1.9164
step  1500 | train loss 1.6650 | val loss 1.8185
step  2000 | train loss 1.5917 | val loss 1.7646
step  2500 | train loss 1.5459 | val loss 1.7217
step  3000 | train loss 1.4870 | val loss 1.6830
[train] 92초
```

학습이 끝난 뒤 `generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 20 --seed 0`으로 생성한 결과예요.

```text
ROMEO:
No, a king to the hage of help,
I help to man a no false there live of still a two
come and the murders in up in our breath,
By like me in and not for the horse, the shall king
and lady parle that part, both the hath prince
As are hath for the capse
```

대본 형식(이름 + 줄바꿈), 진짜 영어 단어, 그럴듯한 리듬은 배웠지만 뜻은 아직 안 통해요. 81만 파라미터, 1분 반 학습으로는 여기까지예요. 내가 만든 코드로 돌렸을 때 val loss가 **1.6~1.8** 근처면 잘 만든 거예요. (랜덤 시드와 기기에 따라 조금씩 달라요)

학습이 끝나면 `out/` 폴더에 세 파일이 생겨요.

| 파일 | 내용 |
|---|---|
| `out/ckpt.pt` | 학습한 Weight + 설정 + 글자 목록 |
| `out/loss.png` | train / val loss 곡선 |
| `out/history.json` | 평가할 때마다 기록한 loss 숫자 |

`out/`은 `.gitignore`에 들어 있어서 GitHub에는 올라가지 않아요.

### 3) 글 생성: `generate.py`

```bash
../.venv/bin/python generate.py --prompt "ROMEO:"
../.venv/bin/python generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 20 --max_new_tokens 500
../.venv/bin/python generate.py --prompt "JULIET:" --seed 42      # 같은 결과를 다시 보고 싶을 때
```

| 옵션 | 효과 |
|---|---|
| `--prompt` | 이 글자들로 시작해요. 학습 데이터에 없는 글자(예: 한글)는 쓸 수 없어요 |
| `--temperature` | 낮으면 안전하고 반복적, 높으면 엉뚱하고 다양해요 (week1/day05.ipynb 7절) |
| `--top_k` | 상위 k개 글자 안에서만 뽑아요. 이상한 글자가 튀어나오는 걸 막아요 |
| `--max_new_tokens` | 몇 글자를 새로 쓸지 |
| `--seed` | 랜덤 고정 |

같은 prompt로 temperature를 0.5 / 1.0 / 1.5로 바꿔가며 비교해 보세요.

---

## 더 해보기

### 내 텍스트로 학습하기

```bash
../.venv/bin/python train.py --data data/my_text.txt --out out_my
../.venv/bin/python generate.py --ckpt out_my/ckpt.pt --prompt "오늘"
```

한국어 글로도 학습할 수 있어요. 다만 글자 단위 Tokenizer라 한글은 글자 종류(vocab)가 많아지고, 데이터가 최소 수십만 글자는 있어야 그럴듯한 결과가 나와요. 데이터가 작으면 아래 과적합이 금방 생겨요.

### 모델 키우기

```bash
../.venv/bin/python train.py --n_embd 256 --n_head 8 --n_layer 6 --block_size 128 --max_steps 5000 --lr 5e-4 --out out_big
```

- 키우면 loss는 더 내려가지만 학습 시간이 꽤 길어져요.
- `n_embd`는 `n_head`로 나누어떨어져야 해요 (Day 26).
- 모델이 커지면 learning rate는 보통 조금 낮춰요.

### 과적합(overfitting) 읽는 법

`out/loss.png`에서 두 선을 비교해요.

```text
train loss ↓  val loss ↓     → 잘 배우는 중
train loss ↓  val loss → ↑   → 과적합: 학습 데이터를 "외우는" 중
```

val loss가 다시 올라가기 시작하면 더 학습해도 소용없어요. 데이터를 늘리거나, `--dropout`을 높이거나, 모델을 줄이거나, 그 시점에서 멈춰요.

---

## 막혔을 때

먼저 테스트 오류 메시지를 읽고, 위 "자주 하는 실수"를 확인해 보세요. 그래도 안 되면 완성본과 비교해요.

- `solutions/minigpt/sampling.py`
- `solutions/minigpt/trainer.py`

`--impl solutions`로 학습이 되는데 내 코드로는 안 된다면, 어느 Day 파일이 문제인지 `pytest tests/ -v`로 찾을 수 있어요.

---

## 4주 마무리

축하해요. 이제 이 흐름을 **직접 짠 코드로** 처음부터 끝까지 돌릴 수 있어요.

```text
문장
 ↓  Tokenizer                     minigpt/tokenizer.py   (Day 22, week2/day08)
Token ID
 ↓  Embedding                     minigpt/embedding.py   (Day 24, week2/day09)
 ↓  + Position                                           (week2/day10)
 ↓  Self-Attention (Q/K/V, √dk, mask, softmax, ×V)
 │                                minigpt/attention.py   (Day 25, week2/day11~13)
 ↓  Multi-Head                    minigpt/multihead.py   (Day 26, week2/day14)
 ↓  Transformer Block × N         minigpt/block.py       (Day 27, week3/day15~17)
 ↓  LayerNorm → Linear            minigpt/model.py       (Day 27, week3/day19)
 ↓  Softmax
Next Token                        minigpt/sampling.py    (오늘, week3/day20)
 ↓  반복 생성
```

학습은 `minigpt/trainer.py`(오늘)가 `Cross Entropy`(week1/day06)와 `Backprop + AdamW`(week1/day07)로 해요.

### 다음 단계

커리큘럼의 "추가 학습 단계" 순서예요.

```text
Pre-training → Instruction Tuning → Fine-tuning → LoRA
 → RLHF / Preference Learning → Inference → KV Cache → Quantization → RAG
```

| 주제 | 한 줄 설명 | 오늘 코드와의 연결 |
|---|---|---|
| Pre-training | 아주 많은 텍스트로 다음 Token 예측 학습 | 오늘 한 것이 바로 이것 (규모만 다름) |
| Instruction Tuning | 질문-답변 형식 데이터로 추가 학습 | 같은 `train()`, 데이터만 바뀜 |
| Fine-tuning | 특정 목적 데이터로 이어서 학습 | `ckpt.pt`를 불러와 이어서 학습 |
| LoRA | 큰 Weight는 얼리고 작은 행렬 두 개만 학습 | `nn.Linear` 옆에 작은 Linear 추가 |
| RLHF / Preference | 사람이 더 좋아하는 답이 나오도록 학습 | loss 함수가 바뀜 |
| KV Cache | 생성할 때 지난 K, V를 저장해 재계산을 없앰 | `generate()`가 매번 전체를 다시 계산하는 걸 개선 |
| Quantization | Weight를 8bit/4bit로 줄여 가볍게 | 같은 모델, 숫자 표현만 바뀜 |
| RAG | 관련 문서를 찾아 prompt에 붙여 넣기 | `--prompt`를 똑똑하게 만드는 것 |

---

## 체크 질문

1. 학습 시작 직후 val loss가 약 4.17인 이유는?
   - 내 답:
2. `generate()`에서 입력을 `block_size`로 자르지 않으면 어떤 일이 생기나요?
   - 내 답:
3. 학습(Training)과 생성(Inference)은 무엇이 다른가요? (정답 유무, 한 번에 처리하는 위치 수)
   - 내 답:
4. temperature와 top_k는 각각 생성 결과를 어떻게 바꾸나요?
   - 내 답:
5. train loss는 계속 내려가는데 val loss가 올라가기 시작했다면, 무슨 일이 생긴 걸까요? 어떻게 대처하나요?
   - 내 답:
