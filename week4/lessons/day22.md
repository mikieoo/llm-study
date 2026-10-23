# Day 22. Character Tokenizer

## 오늘의 목표

- 4주차 진행 방식을 익힌다 (skeleton → 테스트 → 완성본과 비교)
- 학습 데이터(Tiny Shakespeare)를 직접 열어보고 어떤 글자들이 있는지 본다
- 글자 단위 Tokenizer `CharTokenizer`를 직접 구현한다 (커리큘럼 6-1)

---

## 4주차는 이렇게 진행해요

1~3주차는 노트북으로 개념을 익혔어요. 4주차는 **`.py` 파일로 mini GPT를 처음부터 직접 만드는 주**예요. 하루에 부품을 하나씩 만들고, Day 28에 다 조립해서 학습시켜요.

### 폴더 구조

```text
week4/
├── lessons/          ← 날마다 읽을 설명서 (지금 보고 있는 파일)
├── minigpt/          ← 내가 채울 코드 (TODO 포함)
├── tests/            ← 내 코드가 맞는지 확인하는 테스트
├── solutions/minigpt/← 완성본 (막혔을 때만!)
├── data/input.txt    ← 학습 데이터
├── train.py          ← Day 28: 학습 실행
└── generate.py       ← Day 28: 텍스트 생성
```

### 하루 흐름

1. `lessons/dayNN.md`를 읽는다
2. `minigpt/` 안의 오늘 파일에서 `# TODO`와 `raise NotImplementedError(...)` 부분을 채운다
3. 테스트를 돌린다 → 전부 `passed`가 나올 때까지 고친다
4. 그래도 막히면 `solutions/minigpt/`의 같은 파일과 비교한다

> 💡 모든 명령은 **`week4` 폴더에서** 실행해요. VS Code 터미널에서 `cd week4` 먼저 하세요.

| Day | 오늘 만들 파일 | 커리큘럼 |
|---|---|---|
| 22 | `minigpt/tokenizer.py` | 6-1 |
| 23 | `minigpt/data.py` | 6-2 |
| 24 | `minigpt/embedding.py` | 6-3 |
| 25 | `minigpt/attention.py` | 6-4, 6-5 |
| 26 | `minigpt/multihead.py` | 6-6 |
| 27 | `minigpt/block.py`, `minigpt/model.py` | 6-7, 6-8, 6-9 |
| 28 | `minigpt/sampling.py`, `minigpt/trainer.py` + 학습 | 6-10, 6-11 |

---

## 학습 데이터 살펴보기

`data/input.txt`는 **Tiny Shakespeare**예요. 셰익스피어 희곡을 이어 붙인 약 111만 글자짜리 영어 텍스트로, 글자 단위 GPT 실습에서 가장 많이 쓰는 데이터예요.

```bash
head -20 data/input.txt
```

```text
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.
```

Python으로 글자 종류도 세어보세요.

```bash
../.venv/bin/python -c "t=open('data/input.txt').read(); c=sorted(set(t)); print(len(t), len(c)); print(repr(''.join(c)))"
```

글자 종류는 **65개**예요 (대소문자 알파벳, 숫자 일부, 문장부호, 공백, 줄바꿈). 이게 곧 우리 모델의 `vocab_size`가 돼요.

> 나중에 내 텍스트(한국어 소설 등)로도 학습할 수 있어요. Day 28에서 `--data` 옵션으로 바꾸면 돼요.

---

## 개념 복습

`week2/day08.ipynb` 2절에서 이미 만들어봤어요.

```text
"hello"
  ↓ encode   (글자 → ID)
[3, 2, 4, 4, 5]
  ↓ decode   (ID → 글자)
"hello"
```

| 이름 | 뜻 | 예 (`"hello world"`) |
|---|---|---|
| `chars` | 등장한 글자를 중복 없이 **정렬**한 리스트 = Vocabulary | `[' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w']` |
| `stoi` | string to int: 글자 → ID | `{' ': 0, 'd': 1, ...}` |
| `itos` | int to string: ID → 글자 | `{0: ' ', 1: 'd', ...}` |
| `vocab_size` | 글자 종류 수 | `8` |

---

## 단계별 구현 가이드

파일: `minigpt/tokenizer.py`

### 1. `__init__`

- `set(text)`로 중복 제거 → `sorted(...)`로 정렬 → `self.chars`
  - 정렬하는 이유: 같은 텍스트면 **항상 같은 ID**가 붙도록 (set은 순서가 정해져 있지 않아요)
- `enumerate(self.chars)`로 `(ID, 글자)` 쌍을 돌면서 `self.stoi`, `self.itos` 딕셔너리 만들기
- 마지막 줄의 `raise NotImplementedError(...)`는 **지워야** 해요

### 2. `vocab_size`

- `self.chars`의 길이

### 3. `encode(s)`

- 문자열의 글자 하나하나를 `self.stoi`로 바꿔서 **리스트**로 반환

### 4. `decode(ids)`

- ID 하나하나를 `self.itos`로 바꿔서 **이어 붙인 문자열** 반환 (`"".join(...)`)
- Day 28에서 `decode`에 `torch.Tensor`가 들어와요. 텐서 원소는 `int`가 아니라 0차원 텐서라서 딕셔너리 키로 바로 못 써요 → `int(i)`로 바꿔서 찾기

`from_chars`는 이미 완성돼 있어요. 학습한 모델을 저장할 때 `chars`만 저장해두고, 불러올 때 이걸로 Tokenizer를 똑같이 복원해요.

### 자주 하는 실수

| 증상 | 원인 |
|---|---|
| `test_encode_order_follows_sorted_vocab` 실패 | `sorted`를 빼먹음 |
| `decode`가 `TypeError: unhashable type` 또는 `KeyError` | 텐서 원소를 `int(i)`로 안 바꿈 |
| `encode`가 문자열을 돌려줌 | 리스트 컴프리헨션 `[... for ch in s]` 대신 다른 걸 씀 |

---

## 실행해서 확인하기

```bash
# 테스트 (8개 전부 passed가 목표)
../.venv/bin/python -m pytest tests/test_day22.py -v

# 직접 돌려보기 (파일 맨 아래 __main__ 부분이 실행됨)
../.venv/bin/python -m minigpt.tokenizer
```

정답이면 이렇게 나와요.

```text
vocab: [' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'] | size: 8
encode('hello') = [3, 2, 4, 4, 5]
decode(...)     = hello
```

---

## 막혔을 때

먼저 테스트 실패 메시지를 읽어보세요. 무엇이 틀렸는지 알려줘요.
그래도 모르겠으면 `solutions/minigpt/tokenizer.py`와 비교하세요. **먼저 10분은 혼자 시도해보기!**

완성본으로 테스트가 통과하는지 확인하고 싶으면:

```bash
MINIGPT_IMPL=solutions ../.venv/bin/python -m pytest tests/test_day22.py -v
```

---

## 체크 질문

1. `chars`를 만들 때 왜 `sorted`를 하나요?
2. Tiny Shakespeare의 `vocab_size`는 얼마이고, 이 숫자는 나중에 GPT의 어느 부분 크기를 정하나요? (힌트: `week3/day19.ipynb`)
3. 글자 단위 Tokenizer의 장점과 단점은? (힌트: `week2/day08.ipynb`의 비교표)
4. 학습 데이터에 없던 글자(예: `"한"`)를 `encode`하면 어떻게 되나요? 왜 그런가요?

**내 답:**

1.
2.
3.
4.
