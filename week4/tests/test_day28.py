"""Day 28 테스트: generate (sampling.py), train (trainer.py)

generate 테스트는 일부러 아주 작은 가짜 모델을 써요.
그래서 Day 27의 GPT가 완성되지 않았어도 generate만 따로 확인할 수 있어요.
train 테스트는 Day 23의 get_batch(minigpt/data.py)를 사용해요.
"""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    """마지막 Token만 보고 다음 Token 점수를 내는 Bigram 모델 (week3/day20.ipynb)"""

    def __init__(self, vocab_size=10, block_size=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(block_size=block_size, vocab_size=vocab_size)
        self.table = nn.Embedding(vocab_size, vocab_size)
        self.seen_lengths = []

    def forward(self, idx, targets=None):
        B, T = idx.shape
        self.seen_lengths.append(T)
        assert T <= self.config.block_size, "block_size보다 긴 입력이 모델에 들어왔어요"
        logits = self.table(idx)                       # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


def manual_greedy(model, idx, n):
    for _ in range(n):
        logits, _ = model(idx[:, -model.config.block_size:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def test_generate_shape_and_prefix():
    from minigpt.sampling import generate

    model = TinyLM()
    idx = torch.tensor([[1, 2], [3, 4]])
    out = generate(model, idx, max_new_tokens=5)
    assert tuple(out.shape) == (2, 7), f"결과 shape가 (B, T + max_new_tokens) = (2, 7)이어야 해요. 지금: {tuple(out.shape)}"
    assert torch.equal(out[:, :2], idx), "처음 넣은 prompt가 결과 앞부분에 그대로 남아 있어야 해요"
    assert out.dtype == torch.long, "생성된 Token ID는 정수(long)여야 해요"
    assert int(out.min()) >= 0 and int(out.max()) < 10, "Token ID가 vocab 범위를 벗어났어요"


def test_generate_crops_to_block_size():
    from minigpt.sampling import generate

    model = TinyLM(block_size=4)
    idx = torch.randint(10, (1, 9))                    # block_size(4)보다 긴 prompt
    out = generate(model, idx, max_new_tokens=6)
    assert tuple(out.shape) == (1, 15), "prompt가 길어도 결과 길이는 T + max_new_tokens여야 해요"
    assert max(model.seen_lengths) <= 4, "모델에 넣기 전에 최근 block_size개만 잘라야 해요"


def test_generate_top_k_1_is_greedy():
    from minigpt.sampling import generate

    model = TinyLM(seed=3)
    idx = torch.tensor([[0], [5]])
    torch.manual_seed(123)
    out = generate(model, idx, max_new_tokens=8, top_k=1)
    expected = manual_greedy(model, idx, 8)
    assert torch.equal(out, expected), "top_k=1이면 항상 확률 1등 Token만 골라야 해요 (greedy와 같아야 함)"


def test_generate_low_temperature_is_greedy():
    from minigpt.sampling import generate

    model = TinyLM(seed=7)
    idx = torch.tensor([[2]])
    torch.manual_seed(0)
    out = generate(model, idx, max_new_tokens=8, temperature=1e-3)
    expected = manual_greedy(model, idx, 8)
    assert torch.equal(out, expected), "temperature가 아주 작으면 거의 1등만 뽑혀야 해요. logits를 temperature로 나눴나요?"


def test_generate_samples_not_always_greedy():
    from minigpt.sampling import generate

    model = TinyLM(seed=1)
    with torch.no_grad():
        model.table.weight.zero_()                     # 모든 Token 확률이 똑같은 모델
    idx = torch.zeros(1, 1, dtype=torch.long)
    torch.manual_seed(0)
    out = generate(model, idx, max_new_tokens=40)
    assert len(set(out[0, 1:].tolist())) > 3, "확률이 고르면 여러 Token이 골고루 나와야 해요. argmax 대신 multinomial로 뽑았나요?"


def test_train_lowers_loss():
    from minigpt.trainer import train

    torch.manual_seed(0)
    data = torch.tensor([0, 1, 2, 3, 4] * 200)         # 0→1→2→3→4→0 반복: 다음 Token이 완전히 정해진 데이터
    train_data, val_data = data[:900], data[900:]
    model = TinyLM(vocab_size=5, block_size=8)

    history = train(
        model, train_data, val_data,
        max_steps=150, batch_size=8, lr=0.1, eval_interval=50, eval_iters=5,
        device="cpu", log=lambda *a: None,
    )
    assert isinstance(history, list) and len(history) >= 2, "train은 평가 기록(history) 리스트를 돌려줘야 해요"
    assert set(history[0]) >= {"step", "train", "val"}, "history 항목에는 step / train / val이 있어야 해요"
    first, last = history[0]["val"], history[-1]["val"]
    assert last < first * 0.5, "학습 후 val loss가 충분히 줄지 않았어요. zero_grad → backward → step 순서를 확인해보세요"
