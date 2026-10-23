# Day 24. Embedding 구현 (커리큘럼 6-3)

## 오늘의 목표
- Token ID `(B, T)`를 벡터 `(B, T, C)`로 바꾸는 **GPT의 첫 번째 층**을 직접 만든다
- Token Embedding과 Position Embedding을 **더해서** 쓰는 이유를 코드로 확인한다
- `block_size`보다 긴 입력을 막는 이유를 안다

## 오늘 만들 파일
| 파일 | 내용 |
|---|---|
| `minigpt/embedding.py` | `GPTEmbedding` 클래스 (TODO 5개) |
| `tests/test_day24.py` | 확인용 테스트 (수정하지 마세요) |

---

## 개념 복습

### Token Embedding (week2/day09.ipynb)
`nn.Embedding(vocab_size, n_embd)`는 **표(행렬)** 예요. Token ID가 행 번호이고, 그 행을 꺼내면 벡터가 나와요.

```text
token_embedding.weight : (V, C)
idx                    : (B, T)        ← 행 번호들
token_embedding(idx)   : (B, T, C)     ← 행을 꺼내서 C 축이 붙음
```

### Position Embedding (week2/day10.ipynb)
Attention은 순서를 모르기 때문에 "몇 번째 자리인지"도 벡터로 알려줘야 해요.
GPT-2와 우리 mini GPT는 **위치도 학습하는 표**를 써요. 표 크기는 `(block_size, C)`예요.

```text
position_embedding.weight         : (block_size, C)
torch.arange(T)                   : (T,)      ← [0, 1, ..., T-1]
position_embedding(torch.arange(T)): (T, C)
```

### 둘을 더한다
```text
tok : (B, T, C)
pos :    (T, C)       ← 앞에 B 축이 없지만 broadcasting으로 B개 문장 모두에 같은 위치 벡터가 더해짐
out : (B, T, C)
```

💡 이어 붙이지(concat) 않고 **더하는** 이유: 길이 C를 그대로 유지해야 뒤의 Block들이 `(B, T, C)`를 받을 수 있어요. 학습하면서 모델이 알아서 "의미 정보"와 "위치 정보"를 같은 벡터 안에 섞어 담아요.

---

## 단계별 구현 가이드

### `__init__`
| TODO | 할 일 | 결과 |
|---|---|---|
| 1 | `self.token_embedding` = Token 종류 수 × 벡터 길이 표 | weight `(vocab_size, n_embd)` |
| 2 | `self.position_embedding` = 위치 개수 × 벡터 길이 표 | weight `(block_size, n_embd)` |

TODO를 채운 뒤 `raise NotImplementedError(...)` 줄은 **지우세요.** 그 아래 `dropout`, `block_size` 줄은 그대로 둬요.

### `forward`
| TODO | 할 일 | shape |
|---|---|---|
| 3 | Token 벡터 꺼내기 | `(B, T, C)` |
| 4 | 위치 `0..T-1`의 벡터 꺼내기 | `(T, C)` |
| 5 | 더하고 `self.dropout` 통과시켜 반환 | `(B, T, C)` |

### 자주 하는 실수
- **`torch.arange(self.block_size)`로 위치를 만들기**: 입력이 `block_size`보다 짧으면(예: 생성 초반 T=1) shape가 안 맞아요. 반드시 **T개**만 꺼내세요.
- **device를 안 맞추기**: `torch.arange(T)`는 기본이 CPU예요. 모델을 MPS(맥 GPU)로 옮기면 에러가 나요. `device=idx.device`를 붙이세요.
- **dropout 빼먹기**: 지금은 `dropout=0.0`이라 티가 안 나지만, 학습(Day 28)에서는 0.1을 써요.
- **`raise` 줄을 안 지우기**: TODO를 다 채웠는데 `NotImplementedError`가 나면 이걸 확인하세요.

---

## 실행해서 확인하기
`week4` 폴더에서 실행하세요.

```bash
# 테스트 (7개 모두 통과하면 완료)
../.venv/bin/python -m pytest tests/test_day24.py -v

# 직접 돌려보기
../.venv/bin/python -m minigpt.embedding
# idx: (2, 8) → out: (2, 8, 32)
```

테스트가 확인하는 것:
- 두 표의 크기
- 출력 shape `(B, T, C)`
- 출력 == `token_embedding.weight[idx] + position_embedding.weight[arange(T)]`
- 같은 Token이라도 위치가 다르면 다른 벡터
- `T < block_size`에서도 동작, `T > block_size`면 AssertionError
- train 모드에서만 dropout 적용

## 막혔을 때
먼저 week2/day09.ipynb, day10.ipynb의 코드를 다시 보고 15분은 직접 시도해 보세요.
그래도 안 되면 `solutions/minigpt/embedding.py`와 비교해 보세요. 완성본으로 테스트를 돌려보려면:

```bash
MINIGPT_IMPL=solutions ../.venv/bin/python -m pytest tests/test_day24.py -v
```

---

## 체크 질문
1. `token_embedding`과 `position_embedding`의 weight shape는 각각 무엇이고, 왜 행 개수가 다른가?
2. `tok + pos`에서 shape `(B, T, C)`와 `(T, C)`가 어떻게 더해지는가?
3. 위치 번호를 `torch.arange(self.block_size)`가 아니라 `torch.arange(T)`로 만들어야 하는 이유는?
4. 입력 길이가 `block_size`보다 길면 왜 안 되는가? 생성할 때는 이 문제를 어떻게 피하는가? (힌트: week3/day20.ipynb)

**내 답:**

1.
2.
3.
4.
