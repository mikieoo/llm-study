"""Day 25 테스트: Self-Attention Head + Causal Mask

    ../.venv/bin/python -m pytest tests/test_day25.py -v
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_EMBD, HEAD_SIZE, BLOCK_SIZE = 16, 8, 8


def make_head(dropout=0.0, block_size=BLOCK_SIZE):
    from minigpt.attention import Head

    torch.manual_seed(0)
    return Head(N_EMBD, HEAD_SIZE, block_size, dropout).eval()


def test_layers_exist():
    head = make_head()
    for name in ["key", "query", "value"]:
        layer = getattr(head, name, None)
        assert isinstance(layer, nn.Linear), f"self.{name}가 nn.Linear가 아니에요"
        assert layer.in_features == N_EMBD and layer.out_features == HEAD_SIZE, \
            f"self.{name}의 입력/출력 크기가 이상해요 (n_embd → head_size 여야 해요)"
        assert layer.bias is None, f"self.{name}에 bias가 있어요. Attention의 Q/K/V는 bias=False로 만들어요"


def test_tril_is_buffer_not_parameter():
    head = make_head()
    buffers = dict(head.named_buffers())
    assert "tril" in buffers, "tril이 buffer로 등록되지 않았어요 (register_buffer 사용)"
    assert "tril" not in dict(head.named_parameters()), "tril은 학습하는 값이 아니라서 parameter면 안 돼요"
    tril = buffers["tril"]
    assert tuple(tril.shape) == (BLOCK_SIZE, BLOCK_SIZE), "tril의 shape는 (block_size, block_size)여야 해요"
    assert torch.equal(tril != 0, torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)) != 0), \
        "tril이 아래 삼각형(대각선 포함) 모양이 아니에요"


def test_output_shape():
    head = make_head()
    x = torch.randn(2, BLOCK_SIZE, N_EMBD)
    out = head(x)
    assert tuple(out.shape) == (2, BLOCK_SIZE, HEAD_SIZE), \
        f"출력 shape가 {tuple(out.shape)}예요. (B, T, head_size)여야 해요"


def test_shorter_sequence_works():
    head = make_head()
    x = torch.randn(3, 5, N_EMBD)          # T=5 < block_size=8
    out = head(x)
    assert tuple(out.shape) == (3, 5, HEAD_SIZE), \
        "T가 block_size보다 짧을 때 shape가 이상해요. 마스크를 [:T, :T]로 잘라 썼나요?"


def test_matches_reference_attention():
    head = make_head()
    torch.manual_seed(1)
    x = torch.randn(2, 6, N_EMBD)
    with torch.no_grad():
        q = x @ head.query.weight.T
        k = x @ head.key.weight.T
        v = x @ head.value.weight.T
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = head(x)
    assert torch.allclose(out, expected, atol=1e-5), \
        "계산 결과가 softmax(QK^T / √dk + causal mask) V와 달라요. scale(√head_size), 마스크, softmax 축을 확인해보세요"


def test_causal_future_does_not_leak():
    head = make_head()
    torch.manual_seed(2)
    x = torch.randn(1, BLOCK_SIZE, N_EMBD)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(N_EMBD) * 5   # 마지막(미래) Token만 바꾸기
    with torch.no_grad():
        a, b = head(x), head(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), \
        "마지막 Token만 바꿨는데 앞쪽 위치 출력이 바뀌었어요. 미래를 보고 있어요 (마스크 확인)"
    assert not torch.allclose(a[:, -1], b[:, -1]), "마지막 위치 출력은 바뀌어야 해요"


def test_first_token_attends_only_to_itself():
    head = make_head()
    torch.manual_seed(3)
    x = torch.randn(1, 4, N_EMBD)
    with torch.no_grad():
        out = head(x)
        v0 = x[0, 0] @ head.value.weight.T
    assert torch.allclose(out[0, 0], v0, atol=1e-5), \
        "첫 번째 Token은 자기 자신만 볼 수 있어서 출력이 자기 Value와 같아야 해요"
