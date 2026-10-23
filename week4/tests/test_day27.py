"""Day 27 테스트: FeedForward / Block / GPT

    ../.venv/bin/python -m pytest tests/test_day27.py -v

Block 은 Day 26 의 MultiHeadAttention(→ Day 25 Head)을 쓰고,
GPT 는 Day 24 의 GPTEmbedding 을 써요. 앞 Day 가 끝나 있어야 통과합니다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C, N_HEAD, BLOCK = 32, 4, 8


def _small_config(vocab_size=65):
    from minigpt.model import GPTConfig

    return GPTConfig(vocab_size=vocab_size, block_size=BLOCK, n_embd=C, n_head=N_HEAD, n_layer=2, dropout=0.0)


# ---------------------------------------------------------------- FeedForward
def test_feedforward_shape():
    from minigpt.block import FeedForward

    torch.manual_seed(0)
    ffwd = FeedForward(C)
    out = ffwd(torch.randn(2, 5, C))
    assert out.shape == (2, 5, C), f"FeedForward 출력 shape 가 {tuple(out.shape)} 예요. 입력과 같은 (B, T, C) 여야 해요"


def test_feedforward_hidden_is_4C():
    from minigpt.block import FeedForward

    ffwd = FeedForward(C)
    linears = [m for m in ffwd.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2, f"FeedForward 안에 Linear 가 {len(linears)}개 있어요. 늘리는 것 하나, 줄이는 것 하나여야 해요"
    up, down = linears
    assert (up.in_features, up.out_features) == (C, 4 * C), "첫 번째 Linear 는 C → 4C 여야 해요"
    assert (down.in_features, down.out_features) == (4 * C, C), "두 번째 Linear 는 4C → C 여야 해요"
    assert any(isinstance(m, nn.GELU) for m in ffwd.modules()), "두 Linear 사이에 GELU 가 없어요 (Day 4)"


# ---------------------------------------------------------------- Block
def test_block_shape():
    from minigpt.block import Block

    torch.manual_seed(0)
    block = Block(C, N_HEAD, BLOCK)
    out = block(torch.randn(2, BLOCK, C))
    assert out.shape == (2, BLOCK, C), f"Block 출력 shape 가 {tuple(out.shape)} 예요. 입력과 같아야 여러 개를 쌓을 수 있어요"


def test_block_has_parts():
    from minigpt.block import Block

    block = Block(C, N_HEAD, BLOCK)
    for name in ["ln1", "attn", "ln2", "ffwd"]:
        assert hasattr(block, name), f"Block 에 self.{name} 이 없어요"
    assert isinstance(block.ln1, nn.LayerNorm) and isinstance(block.ln2, nn.LayerNorm), "ln1, ln2 는 nn.LayerNorm 이어야 해요"


def test_block_residual():
    """attention 과 FFN 이 0 을 내면, Residual 덕분에 Block 은 입력을 그대로 통과시켜야 해요."""
    from minigpt.block import Block

    torch.manual_seed(0)
    block = Block(C, N_HEAD, BLOCK)
    with torch.no_grad():
        for p in list(block.attn.parameters()) + list(block.ffwd.parameters()):
            p.zero_()
    x = torch.randn(2, BLOCK, C)
    assert torch.allclose(block(x), x, atol=1e-6), (
        "attn 과 ffwd 출력이 0 인데 Block 출력이 입력과 달라요. "
        "x = x + ... 형태의 Residual 이 빠졌거나, LayerNorm 을 Residual 바깥에 적용했어요"
    )


def test_block_matches_pre_ln_formula():
    from minigpt.block import Block

    torch.manual_seed(0)
    block = Block(C, N_HEAD, BLOCK).eval()
    x = torch.randn(2, BLOCK, C)
    h = x + block.attn(block.ln1(x))
    expected = h + block.ffwd(block.ln2(h))
    assert torch.allclose(block(x), expected, atol=1e-5), "Pre-LN 순서가 달라요: LayerNorm 은 attn/ffwd '안쪽'에 들어가야 해요"


def test_block_is_causal():
    from minigpt.block import Block

    torch.manual_seed(0)
    block = Block(C, N_HEAD, BLOCK).eval()
    x = torch.randn(1, BLOCK, C)
    x2 = x.clone()
    x2[0, -1] = torch.randn(C)          # 마지막(미래) 위치만 바꾸기
    out1, out2 = block(x), block(x2)
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5), "미래 Token 을 바꿨더니 앞쪽 출력이 바뀌었어요. Causal Mask 를 확인하세요"


# ---------------------------------------------------------------- GPT
def test_gpt_logits_shape_and_no_loss():
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(_small_config())
    idx = torch.randint(65, (3, BLOCK))
    logits, loss = model(idx)
    assert logits.shape == (3, BLOCK, 65), f"logits shape 가 {tuple(logits.shape)} 예요. (B, T, vocab_size) 여야 해요"
    assert loss is None, "targets 가 없으면 loss 는 None 이어야 해요"


def test_gpt_shorter_input():
    from minigpt.model import GPT

    model = GPT(_small_config())
    logits, _ = model(torch.randint(65, (2, 3)))
    assert logits.shape == (2, 3, 65), "block_size 보다 짧은 입력도 처리할 수 있어야 해요"


def test_gpt_has_parts():
    from minigpt.model import GPT

    model = GPT(_small_config())
    for name in ["embedding", "blocks", "ln_f", "lm_head"]:
        assert hasattr(model, name), f"GPT 에 self.{name} 이 없어요"
    assert len(model.blocks) == 2, f"Block 이 {len(model.blocks)}개예요. n_layer 개만큼 있어야 해요"
    assert isinstance(model.ln_f, nn.LayerNorm), "ln_f 는 nn.LayerNorm 이어야 해요"
    assert model.num_params() > 0


def test_gpt_initial_loss_near_log_vocab():
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(_small_config())
    idx = torch.randint(65, (4, BLOCK))
    targets = torch.randint(65, (4, BLOCK))
    _, loss = model(idx, targets)
    assert loss is not None, "targets 를 주면 loss 를 계산해야 해요"
    assert loss.dim() == 0, "loss 는 숫자 하나(scalar)여야 해요"
    assert abs(loss.item() - math.log(65)) < 0.3, (
        f"학습 전 loss 가 {loss.item():.2f} 예요. 고르게 찍는 모델이면 log(65) ≈ 4.17 근처여야 해요"
    )


def test_gpt_loss_matches_cross_entropy():
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(_small_config()).eval()
    idx = torch.randint(65, (2, BLOCK))
    targets = torch.randint(65, (2, BLOCK))
    logits, loss = model(idx, targets)
    expected = F.cross_entropy(logits.reshape(-1, 65), targets.reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-5), "loss 가 모든 위치의 cross entropy 평균과 달라요"


def test_gpt_is_causal():
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(_small_config()).eval()
    idx = torch.randint(65, (1, BLOCK))
    idx2 = idx.clone()
    idx2[0, -1] = (idx[0, -1] + 1) % 65
    l1, _ = model(idx)
    l2, _ = model(idx2)
    assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5), "마지막 Token 을 바꿨더니 앞쪽 logits 가 바뀌었어요. 모델이 미래를 보고 있어요"
