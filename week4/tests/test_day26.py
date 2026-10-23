"""Day 26 테스트: Multi-Head Attention

    ../.venv/bin/python -m pytest tests/test_day26.py -v

※ Day 25의 Head(minigpt/attention.py)가 완성되어 있어야 통과할 수 있어요.
"""
import pytest
import torch
import torch.nn as nn

C, N_HEAD, BLOCK = 32, 4, 8


def make_mha(n_embd=C, n_head=N_HEAD, block_size=BLOCK):
    from minigpt.multihead import MultiHeadAttention
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd, n_head, block_size, dropout=0.0)
    mha.eval()
    return mha


def test_output_shape():
    mha = make_mha()
    x = torch.randn(2, BLOCK, C)
    out = mha(x)
    assert out.shape == (2, BLOCK, C), f"출력 shape가 (B, T, C) = (2, {BLOCK}, {C})이어야 하는데 {tuple(out.shape)}예요"


def test_shorter_sequence():
    mha = make_mha()
    x = torch.randn(3, 5, C)
    assert mha(x).shape == (3, 5, C), "T가 block_size보다 짧아도 (B, T, C)가 나와야 해요"


def test_heads_registered():
    mha = make_mha()
    assert isinstance(mha.heads, nn.ModuleList), "self.heads는 nn.ModuleList여야 파라미터가 모델에 등록돼요"
    assert len(mha.heads) == N_HEAD, f"Head가 {N_HEAD}개여야 하는데 {len(mha.heads)}개예요"
    n_head_params = sum(p.numel() for p in mha.heads.parameters())
    assert n_head_params > 0 and n_head_params < sum(p.numel() for p in mha.parameters()), \
        "Head 파라미터와 proj 파라미터가 모두 mha.parameters()에 들어 있어야 해요"


def test_each_head_size():
    mha = make_mha()
    x = torch.randn(2, BLOCK, C)
    for i, h in enumerate(mha.heads):
        out = h(x)
        assert out.shape == (2, BLOCK, C // N_HEAD), \
            f"{i}번 Head 출력의 마지막 축은 n_embd / n_head = {C // N_HEAD}이어야 하는데 {out.shape[-1]}예요"


def test_proj_is_linear_c_to_c():
    mha = make_mha()
    assert isinstance(mha.proj, nn.Linear), "self.proj는 nn.Linear여야 해요"
    assert mha.proj.in_features == C and mha.proj.out_features == C, \
        f"proj는 ({C} → {C})여야 하는데 ({mha.proj.in_features} → {mha.proj.out_features})예요"


def test_matches_concat_then_proj():
    mha = make_mha()
    x = torch.randn(2, BLOCK, C)
    expected = mha.proj(torch.cat([h(x) for h in mha.heads], dim=-1))
    assert torch.allclose(mha(x), expected, atol=1e-6), \
        "출력이 '모든 Head 결과를 마지막 축으로 이어 붙인 뒤 proj' 한 값과 달라요. 이어 붙이는 축과 순서를 확인하세요"


def test_causal():
    mha = make_mha()
    torch.manual_seed(1)
    x = torch.randn(1, BLOCK, C)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(C)          # 마지막(미래) Token만 바꿈
    out1, out2 = mha(x), mha(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6), \
        "마지막 Token만 바꿨는데 앞쪽 위치의 출력이 달라졌어요. 미래를 보고 있어요 (Head의 mask 확인)"
    assert not torch.allclose(out1[:, -1], out2[:, -1]), "바꾼 마지막 위치의 출력은 달라져야 해요"


def test_indivisible_raises():
    with pytest.raises(AssertionError):
        make_mha(n_embd=30, n_head=4)
