# LLM / Transformer 학습 커리큘럼

## 목표

이 커리큘럼의 목표는 **LLM을 단순히 사용하는 수준을 넘어, 내부 구조와 동작 원리를 이해하고 직접 작은 GPT 모델을 구현할 수 있는 수준**까지 가는 것입니다.

학습 순서는 다음과 같습니다.

```text
딥러닝 최소 기초
    ↓
NLP 기초
    ↓
Attention
    ↓
Transformer
    ↓
GPT 구조
    ↓
mini GPT 직접 구현
```

수학을 먼저 깊게 파기보다는 **개념 이해 → 코드 실습 → 필요한 수학 보충** 순서로 진행합니다.

---

# 1단계. 딥러닝 최소 기초

## 목표

Transformer와 GPT 코드를 볼 때 기본적인 딥러닝 연산을 이해할 수 있도록 합니다.

## 1-1. 벡터와 행렬

### 학습 내용

- Scalar
- Vector
- Matrix
- Tensor
- Shape
- 행렬 곱셈
- Dot Product

### 반드시 이해할 것

```text
벡터
= 숫자 여러 개를 묶어놓은 것

행렬
= 벡터를 여러 개 쌓아놓은 것
```

예:

```text
[175, 70, 25]
```

LLM 내부에서는 단어나 문장도 결국 숫자 벡터로 처리됩니다.

```text
"고양이"

↓

[0.21, -0.53, 0.87, ...]
```

### Transformer와의 연결

나중에 Attention에서 등장하는 아래 연산을 이해하기 위한 기반입니다.

```text
Q = XWq
K = XWk
V = XWv
```

### 실습

- Python list로 벡터 표현
- NumPy 또는 PyTorch Tensor 만들기
- 행렬 곱 직접 계산하기
- `torch.matmul()` 사용하기

### 체크 질문

- 벡터와 행렬의 차이는 무엇인가?
- 행렬 곱을 왜 사용하는가?
- `(3, 4)` shape와 `(4, 5)` shape를 곱하면 결과 shape는 무엇인가?

---

## 1-2. Linear Layer

### 학습 내용

Linear Layer의 기본 공식:

```text
y = xW + b
```

구성 요소:

```text
x = 입력
W = Weight
b = Bias
y = 출력
```

PyTorch:

```python
import torch.nn as nn

linear = nn.Linear(3, 5)
```

### 반드시 이해할 것

Linear Layer는 입력 벡터를 다른 형태의 벡터로 변환합니다.

```text
입력 벡터
   ↓
Linear Layer
   ↓
새로운 벡터
```

### 체크 질문

- Weight는 누가 정하는가?
- Bias는 왜 필요한가?
- `nn.Linear(3, 5)`에서 3과 5는 무엇을 의미하는가?

---

## 1-3. Activation Function

### 학습 내용

- ReLU
- GELU
- Sigmoid
- Tanh

특히 LLM에서는 GELU 계열을 자주 사용합니다.

### 핵심 질문

왜 Linear Layer만 계속 쌓으면 안 되는가?

```text
Linear
 ↓
Linear
 ↓
Linear
```

이 구조는 결국 하나의 Linear 연산으로 합쳐질 수 있습니다.

그래서 비선형성을 넣기 위해 Activation Function을 사용합니다.

```text
Linear
 ↓
Activation
 ↓
Linear
```

### 실습

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print(F.relu(x))
print(F.gelu(x))
```

---

## 1-4. Softmax

### 학습 내용

Softmax는 여러 숫자를 확률처럼 변환합니다.

예:

```text
[2.0, 1.0, 0.1]

↓

[0.66, 0.24, 0.10]
```

전체 합은 1이 됩니다.

### LLM과의 연결

LLM의 마지막 출력은 각 Token의 점수입니다.

```text
"오늘 날씨가"

↓

좋다     8.2
춥다     6.1
자동차  -2.4
```

Softmax를 적용하면:

```text
좋다      75%
춥다      20%
자동차     1%
...
```

이 확률을 기반으로 다음 Token을 선택합니다.

---

## 1-5. Loss Function

### 학습 내용

- Prediction
- Target
- Loss
- Cross Entropy Loss

### 핵심 개념

```text
모델 예측
   ↓
정답과 비교
   ↓
얼마나 틀렸는지 계산
   ↓
Loss
```

LLM에서는 주로 Cross Entropy Loss를 사용합니다.

### 실습

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])

loss = F.cross_entropy(logits, target)

print(loss)
```

---

## 1-6. Backpropagation

### 학습 내용

- Forward
- Loss
- Gradient
- Backward
- Weight Update

전체 흐름:

```text
입력
 ↓
모델
 ↓
예측
 ↓
Loss 계산
 ↓
Backpropagation
 ↓
Gradient 계산
 ↓
Weight 수정
```

PyTorch:

```python
loss.backward()
```

### 목표

미분 공식을 외우는 것이 아니라 다음을 이해하면 됩니다.

> Loss를 줄이는 방향으로 Weight를 수정하기 위해 Gradient를 계산한다.

---

## 1-7. Optimizer

### 학습 내용

- Gradient Descent
- SGD
- Adam
- AdamW

PyTorch:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 학습 루프

```python
for step in range(1000):

    prediction = model(x)

    loss = loss_function(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

### 1단계 완료 기준

다음 흐름을 설명할 수 있으면 완료입니다.

```text
입력
 ↓
Linear
 ↓
Activation
 ↓
Linear
 ↓
Softmax
 ↓
Prediction
 ↓
Loss
 ↓
Backward
 ↓
Weight Update
```

---

# 2단계. NLP 기초

## 목표

텍스트가 어떻게 숫자로 바뀌어 Transformer에 들어가는지 이해합니다.

---

## 2-1. Token

학습 내용:

- Character Token
- Word Token
- Subword Token
- Token ID

예:

```text
"안녕하세요"

↓

["안", "녕", "하", "세", "요"]
```

또는 실제 LLM에서는:

```text
["안녕", "하세요"]
```

처럼 나뉠 수도 있습니다.

---

## 2-2. Tokenizer

### 학습 내용

- Vocabulary
- Token ID
- Encode
- Decode
- BPE 개념

예:

```text
"고양이"

↓

Token ID

↓

15342
```

---

## 2-3. Embedding

Token ID를 벡터로 변환합니다.

```text
15342

↓

Embedding

↓

[0.13, -0.72, 0.44, ...]
```

### 학습 내용

- Embedding Matrix
- Embedding Dimension
- Semantic Similarity

PyTorch:

```python
embedding = nn.Embedding(vocab_size, embedding_dim)
```

---

## 2-4. Positional Information

Transformer는 기본적으로 순서를 알지 못합니다.

따라서:

```text
나는
오늘
학교에
간다
```

에서 각 Token의 위치 정보를 추가해야 합니다.

### 학습 내용

- Positional Encoding
- Positional Embedding
- RoPE 개념

---

# 3단계. Attention

## 목표

Transformer의 핵심인 Self-Attention을 이해합니다.

---

## 3-1. Self-Attention

문장:

```text
철수는 학교에 갔다.
그는 공부를 했다.
```

`그는`이라는 Token이 `철수`와 관련 있다는 것을 계산합니다.

```text
그는 → 철수      높음
그는 → 학교      낮음
그는 → 공부      중간
```

---

## 3-2. Query / Key / Value

Embedding에서 세 개의 벡터를 만듭니다.

```text
X
│
├─ Wq → Q
├─ Wk → K
└─ Wv → V
```

### 개념

```text
Query
= 내가 찾고 싶은 정보

Key
= 각 Token이 가진 특징

Value
= 실제 가져올 정보
```

---

## 3-3. Attention Score

공식:

```text
QKᵀ
```

Query와 Key의 유사도를 계산합니다.

---

## 3-4. Scaled Dot-Product Attention

핵심 공식:

```text
Attention(Q, K, V)

=

softmax(QKᵀ / √dk)V
```

단계별로 이해합니다.

```text
Q × Kᵀ
 ↓
Scale
 ↓
Softmax
 ↓
Attention Weight
 ↓
× V
 ↓
Output
```

### 실습 목표

PyTorch를 이용해 Attention을 직접 구현합니다.

---

## 3-5. Causal Mask

GPT에서는 미래 Token을 보면 안 됩니다.

```text
나는 오늘 학교에 간다

현재 Token: 오늘

볼 수 있음:
나는
오늘

볼 수 없음:
학교에
간다
```

Mask:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

---

## 3-6. Multi-Head Attention

Attention을 여러 개 병렬로 사용합니다.

```text
Head 1
Head 2
Head 3
Head 4
   ↓
Concat
   ↓
Linear
```

각 Head는 서로 다른 관계를 학습할 수 있습니다.

---

# 4단계. Transformer

## 목표

Attention을 포함한 Transformer Block 전체 구조를 이해합니다.

---

## 4-1. Transformer Block

기본 구조:

```text
입력
 ↓
Multi-Head Self-Attention
 ↓
Residual Connection
 ↓
LayerNorm
 ↓
Feed Forward Network
 ↓
Residual Connection
 ↓
LayerNorm
```

---

## 4-2. Residual Connection

기존 입력을 유지하면서 새로운 정보를 추가합니다.

```text
x
│
├──────────────┐
↓              │
Attention      │
↓              │
+ ←────────────┘
```

---

## 4-3. Layer Normalization

학습을 안정화합니다.

학습 내용:

- 평균
- 분산
- Normalization
- LayerNorm

---

## 4-4. Feed Forward Network

각 Token을 개별적으로 처리하는 작은 신경망입니다.

```text
Linear
 ↓
GELU
 ↓
Linear
```

---

## 4-5. Transformer Block 반복

```text
Embedding
 ↓
Block 1
 ↓
Block 2
 ↓
Block 3
 ↓
...
 ↓
Block N
```

---

# 5단계. GPT 구조

## 목표

Transformer와 GPT의 차이를 이해합니다.

---

## 5-1. Original Transformer

원본 Transformer:

```text
Encoder
+
Decoder
```

---

## 5-2. GPT

GPT는 Decoder-style Transformer 구조를 사용합니다.

```text
Token
 ↓
Embedding
 ↓
Transformer Block
 ↓
Transformer Block
 ↓
...
 ↓
Linear
 ↓
Softmax
 ↓
Next Token
```

---

## 5-3. Next Token Prediction

예:

```text
입력:
나는 오늘 학교

정답:
에
```

학습 데이터:

```text
x: 나는 오늘 학교
y: 는 오늘 학교 에
```

즉 한 칸씩 밀린 구조입니다.

---

## 5-4. Autoregressive Generation

```text
나는
 ↓
나는 오늘
 ↓
나는 오늘 학교에
 ↓
나는 오늘 학교에 갔다
```

한 Token씩 생성합니다.

---

## 5-5. Training과 Inference 차이

### Training

정답을 알고 있습니다.

```text
Input
 ↓
Prediction
 ↓
Target과 비교
 ↓
Loss
 ↓
Backpropagation
```

### Inference

정답이 없습니다.

```text
Input
 ↓
Next Token Prediction
 ↓
Token 선택
 ↓
Input에 추가
 ↓
반복
```

---

# 6단계. mini GPT 직접 구현

## 목표

PyTorch로 작은 GPT를 직접 구현합니다.

---

## 6-1. Character Tokenizer

직접 Vocabulary를 만듭니다.

```python
chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

---

## 6-2. Training Dataset 만들기

```text
안녕하세요

↓

x = 안녕하세
y = 녕하세요
```

---

## 6-3. Embedding 구현

```python
token_embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)
```

---

## 6-4. Self-Attention 직접 구현

직접 구현할 요소:

```text
Q
K
V
Attention Score
Softmax
Weighted Sum
```

---

## 6-5. Causal Mask 적용

미래 Token을 보지 못하도록 Mask를 구현합니다.

---

## 6-6. Multi-Head Attention 구현

여러 Self-Attention Head를 병렬로 실행합니다.

---

## 6-7. Feed Forward Network 구현

```python
nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd)
)
```

---

## 6-8. Transformer Block 구현

구조:

```text
Attention
 ↓
Residual
 ↓
FFN
 ↓
Residual
```

---

## 6-9. GPT Model 구현

최종 구조:

```text
Token Embedding
+
Position Embedding
 ↓
Transformer Block × N
 ↓
LayerNorm
 ↓
Linear
 ↓
Vocabulary Logits
```

---

## 6-10. 학습

```python
for step in range(max_steps):

    logits, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

---

## 6-11. Text Generation

예:

```text
Input:

"오늘"

↓

Model

↓

"오늘 날씨가 좋습니다..."
```

---

# 추가 학습 단계

mini GPT 구현 후에는 다음 순서로 확장합니다.

```text
Transformer
 ↓
GPT
 ↓
Pre-training
 ↓
Instruction Tuning
 ↓
Fine-tuning
 ↓
LoRA
 ↓
RLHF / Preference Learning
 ↓
Inference
 ↓
KV Cache
 ↓
Quantization
 ↓
RAG
```

---

# 추천 학습 일정

## 1주차 — 딥러닝 최소 기초

### Day 1
- Vector
- Matrix
- Tensor
- Shape

### Day 2
- Matrix Multiplication
- Dot Product

### Day 3
- Linear Layer
- Weight
- Bias

### Day 4
- Activation Function
- ReLU
- GELU

### Day 5
- Softmax
- Probability

### Day 6
- Loss
- Cross Entropy

### Day 7
- Backpropagation
- Optimizer
- 간단한 Neural Network 학습

---

## 2주차 — NLP + Attention

### Day 8
- Token
- Vocabulary
- Tokenizer

### Day 9
- Embedding

### Day 10
- Positional Encoding

### Day 11
- Self-Attention 개념

### Day 12
- Query / Key / Value

### Day 13
- Scaled Dot-Product Attention

### Day 14
- Causal Mask
- Multi-Head Attention

---

# 3주차 — Transformer + GPT

### Day 15
- Transformer Block

### Day 16
- Residual Connection
- LayerNorm

### Day 17
- Feed Forward Network

### Day 18
- Encoder / Decoder

### Day 19
- GPT 구조

### Day 20
- Next Token Prediction

### Day 21
- 전체 구조 복습

---

# 4주차 — mini GPT 구현

### Day 22
- Character Tokenizer

### Day 23
- Dataset 구성

### Day 24
- Embedding

### Day 25
- Self-Attention 구현

### Day 26
- Multi-Head Attention 구현

### Day 27
- Transformer Block 구현

### Day 28
- GPT 학습 및 Text Generation

---

# 최종적으로 설명할 수 있어야 하는 질문

## 딥러닝

- Weight는 무엇인가?
- Bias는 왜 필요한가?
- Linear Layer는 어떤 역할을 하는가?
- Activation Function은 왜 필요한가?
- Softmax는 왜 사용하는가?
- Loss는 무엇인가?
- Backpropagation은 왜 필요한가?
- Optimizer는 무엇을 하는가?

## NLP

- Token은 무엇인가?
- Tokenizer는 왜 필요한가?
- Token ID와 Embedding의 차이는 무엇인가?
- Embedding은 무엇인가?
- Position 정보는 왜 필요한가?

## Attention

- Self-Attention은 무엇인가?
- Query / Key / Value는 무엇인가?
- QKᵀ는 왜 계산하는가?
- Softmax는 Attention에서 왜 사용하는가?
- V는 어떤 역할을 하는가?
- Multi-Head Attention은 왜 필요한가?
- Causal Mask는 왜 필요한가?

## Transformer

- Transformer Block에는 무엇이 들어가는가?
- Residual Connection은 왜 사용하는가?
- LayerNorm은 왜 필요한가?
- FFN은 Attention과 어떤 차이가 있는가?

## GPT

- GPT는 Transformer와 어떤 관계인가?
- GPT는 왜 다음 Token을 예측하는가?
- Training과 Inference는 무엇이 다른가?
- GPT는 문장을 어떻게 생성하는가?

---

# 학습 원칙

## 1. 수식을 먼저 외우지 않는다

예:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dk)V
```

공식을 암기하기 전에 각 단계의 의미를 이해합니다.

---

## 2. 코드와 개념을 같이 본다

예:

```python
attention = q @ k.transpose(-2, -1)
```

를 봤을 때:

```text
Query와 Key의 관계를 계산하는 코드
```

라고 연결할 수 있어야 합니다.

---

## 3. 직접 값을 출력한다

Tensor의 shape와 값을 계속 확인합니다.

```python
print(x.shape)
print(q.shape)
print(k.shape)
print(attention.shape)
```

Transformer는 Shape를 이해하면 훨씬 쉽게 이해할 수 있습니다.

---

## 4. 작은 모델부터 만든다

처음부터 실제 GPT 규모를 따라 하지 않습니다.

예:

```text
Embedding Dimension = 32
Head = 4
Layer = 2
```

정도의 작은 모델로 시작합니다.

---

## 5. 마지막에는 직접 설명한다

각 단계가 끝날 때마다 코드 없이 다음 질문에 답해봅니다.

> 지금 입력 데이터가 모델 내부에서 어떤 과정을 거쳐 출력으로 변환되는가?

최종적으로 아래 흐름을 설명할 수 있으면 목표 달성입니다.

```text
문장
 ↓
Tokenizer
 ↓
Token ID
 ↓
Embedding
 ↓
Position
 ↓
Self-Attention
 ↓
Transformer Block × N
 ↓
Linear
 ↓
Softmax
 ↓
Next Token
 ↓
반복 생성
```
