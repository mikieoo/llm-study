# Day 23. Training Dataset 만들기 (커리큘럼 6-2)

## 오늘의 목표

- 긴 텍스트 전체를 **train / val**로 나눌 수 있다
- 텍스트에서 길이 `block_size`짜리 조각을 무작위로 뽑아 **(x, y) batch**를 만들 수 있다
- `y`가 왜 `x`를 한 칸 민 것인지 설명할 수 있다

## 오늘 만들 파일

| 파일 | 할 일 |
|---|---|
| `minigpt/data.py` | `train_val_split`, `get_batch`의 `TODO` 채우기 (`load_text`는 이미 완성) |
| `tests/test_day23.py` | 이미 있어요. 실행만 하면 돼요 |

---

## 1. 개념 복습

### x와 y는 한 칸 밀린 관계예요

`week3/day20.ipynb`에서 본 그대로예요. 커리큘럼 6-2의 예시를 Token 단위로 보면 이렇습니다.

```text
data = 안 녕 하 세 요

x    = 안 녕 하 세
y    = 녕 하 세 요
```

| 위치 t | 모델이 보는 것 (x[:t+1]) | 맞혀야 하는 것 (y[t]) |
|---|---|---|
| 0 | 안 | 녕 |
| 1 | 안 녕 | 하 |
| 2 | 안 녕 하 | 세 |
| 3 | 안 녕 하 세 | 요 |

조각 하나(길이 T)에서 **문제가 T개** 나와요. Causal Mask(`week2/day14.ipynb`) 덕분에 T개를 한 번에 학습할 수 있어요.

> 💡 커리큘럼 5-3의 단어 예시는 `x: 나는 오늘 학교` / `y: 오늘 학교 에`가 맞아요. 원문의 `y: 는 오늘 학교 에`는 오타예요.

### 왜 train / val로 나누나요?

- **train**: 모델이 실제로 보고 배우는 데이터
- **val**: 학습에 쓰지 않고 남겨두는 시험 문제

train loss만 보면, 모델이 문장을 **외워서** 낮아진 건지 **일반화해서** 낮아진 건지 알 수 없어요. 한 번도 안 본 val의 loss도 같이 내려가야 진짜로 배운 거예요. Day 28에서 두 곡선을 같이 그려 볼 거예요.

---

## 2. 단계별 구현 가이드

### 2-1. `train_val_split(data, train_ratio=0.9)`

**입력**: Token ID가 한 줄로 늘어선 1차원 텐서 `data` (shape `(N,)`)
**출력**: `(train, val)` 튜플

1. 자를 위치 `n`을 계산해요: `N × train_ratio`
2. 앞부분 `data[:n]`은 train, 뒷부분 `data[n:]`은 val이에요

**자주 하는 실수**
- `len(data) * train_ratio`는 `float`이에요. 슬라이싱하려면 `int(...)`로 바꿔야 해요.
- **섞으면 안 돼요.** 텍스트는 앞뒤 문맥이 이어져 있어서, 섞으면 val이 train과 같은 장면에서 나와요. 그러면 시험 문제가 교과서에 그대로 실린 셈이에요.
- `train_ratio` 인자를 무시하고 `0.9`를 코드에 박아 넣지 마세요. 테스트가 `0.8`로도 확인해요.

### 2-2. `get_batch(data, block_size, batch_size, device="cpu")`

**입력**: 1차원 텐서 `data`, 조각 길이 `block_size`(T), 조각 개수 `batch_size`(B)
**출력**: `x`, `y`, 둘 다 shape `(B, T)`

**TODO 1: 시작 위치 뽑기**

```python
ix = torch.randint(최댓값, (batch_size,))   # shape (B,)
```

`최댓값`이 이 함수의 핵심이에요. `torch.randint(high, ...)`는 `0 ~ high-1` 사이에서 뽑아요.

- `x`는 `i`부터 `i + block_size - 1`까지 써요
- `y`는 한 칸 더 뒤인 `i + block_size`까지 써요
- 그러니까 `i + block_size`가 `len(data) - 1` 이하여야 해요

종이에 `len(data) = 20`, `block_size = 8`로 놓고 마지막 시작 위치를 직접 계산해 보세요. 테스트는 범위를 넘지 않는지와, **가능한 마지막 위치까지 뽑히는지**를 둘 다 확인해요.

**TODO 2, 3: 조각 잘라서 쌓기**

```text
ix = [3, 17]  (B = 2)

data[3 : 3+T]    →  x의 0번째 행
data[17 : 17+T]  →  x의 1번째 행
torch.stack(...)  →  (2, T)
```

`y`도 똑같이 하되, 시작이 한 칸 뒤예요. 리스트 컴프리헨션과 `torch.stack`을 쓰면 한 줄씩이면 돼요.

**TODO 4: device로 옮기기**

`x.to(device)`, `y.to(device)`. Day 28에서 Apple Silicon GPU(`"mps"`)로 학습할 때 필요해요.

### shape 요약

| 변수 | shape | 의미 |
|---|---|---|
| `data` | `(N,)` | 전체 텍스트의 Token ID |
| `ix` | `(B,)` | 조각 시작 위치 B개 |
| `x` | `(B, T)` | 입력 |
| `y` | `(B, T)` | 정답 (x를 한 칸 민 것) |

---

## 3. 실행해서 확인하기

`week4` 폴더에서 실행하세요.

```bash
# 테스트
../.venv/bin/python -m pytest tests/test_day23.py -v

# 직접 출력해 보기 (파일 맨 아래 __main__ 블록)
../.venv/bin/python -m minigpt.data
```

`__main__` 블록은 `torch.arange(100)`을 data로 써요. 값이 곧 위치라서 결과를 눈으로 확인하기 쉬워요.

```text
train: 90 | val: 10
x: [[a, a+1, ..., a+7], [b, b+1, ..., b+7]]
y: [[a+1, a+2, ..., a+8], [b+1, ..., b+8]]
```

각 행에서 `y`가 `x`보다 정확히 1씩 크면 성공이에요. `a`, `b` 값은 seed에 따라 달라요.

### 실제 데이터로 해보기

Day 22의 Tokenizer와 연결해서 진짜 Shakespeare 조각을 뽑아 보세요. `week4` 폴더에서 `../.venv/bin/python`을 실행한 뒤 입력하면 돼요.

```python
import torch
from minigpt.tokenizer import CharTokenizer
from minigpt.data import load_text, train_val_split, get_batch

text = load_text("data/input.txt")
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
train, val = train_val_split(data)
print(len(train), len(val))

x, y = get_batch(train, block_size=16, batch_size=2)
for i in range(2):
    print(repr(tok.decode(x[i])), "→", repr(tok.decode(y[i])))
```

`y`의 글자들이 `x`보다 한 글자씩 앞으로 당겨져 있는 게 보일 거예요.

---

## 4. 막혔을 때

먼저 테스트 실패 메시지를 읽어 보세요. 무엇이 틀렸는지 알려줘요. 그래도 막히면 `solutions/minigpt/data.py`와 비교해 보세요. 다만 **최소 15분은 직접 시도한 뒤에** 보는 걸 추천해요. 특히 `torch.randint`의 최댓값은 직접 계산해 봐야 오래 기억에 남아요.

---

## 5. 체크 질문

1. `y`는 왜 `x`를 한 칸 민 것인가요? 길이 T인 조각 하나에서 학습 문제는 몇 개 나오나요?
2. train / val을 나눌 때 왜 섞지 않고 앞뒤로 자르나요?
3. `get_batch`에서 시작 위치의 최댓값을 `len(data) - block_size`로 잡는 이유는 무엇인가요? `len(data) - block_size + 1`로 잡으면 어떤 일이 생기나요?
4. 시작 위치를 고정하지 않고 매번 무작위로 뽑는 이유는 무엇인가요?
5. `batch_size = 32`, `block_size = 64`이면 `get_batch` 한 번으로 학습 문제가 몇 개 만들어지나요?

**내 답:**

1.
2.
3.
4.
5.
