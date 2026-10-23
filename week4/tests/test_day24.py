"""Day 24. GPTEmbedding 테스트

    ../.venv/bin/python -m pytest tests/test_day24.py -v
"""
import pytest
import torch
import torch.nn as nn

V, C, BLOCK = 20, 16, 8


def make(dropout=0.0):
    from minigpt.embedding import GPTEmbedding
    torch.manual_seed(0)
    return GPTEmbedding(vocab_size=V, n_embd=C, block_size=BLOCK, dropout=dropout)


def test_tables_exist_with_right_shape():
    emb = make()
    assert isinstance(emb.token_embedding, nn.Embedding), "token_embedding은 nn.Embedding이어야 해요"
    assert isinstance(emb.position_embedding, nn.Embedding), "position_embedding은 nn.Embedding이어야 해요"
    assert tuple(emb.token_embedding.weight.shape) == (V, C), \
        "token_embedding 표 크기가 달라요. 행 개수 = Token 종류 수, 열 개수 = 벡터 길이"
    assert tuple(emb.position_embedding.weight.shape) == (BLOCK, C), \
        "position_embedding 표 크기가 달라요. 행 개수 = 위치 개수(block_size)"


def test_output_shape():
    emb = make()
    idx = torch.randint(V, (3, BLOCK))
    out = emb(idx)
    assert tuple(out.shape) == (3, BLOCK, C), f"출력 shape가 (B, T, C)여야 해요. 지금: {tuple(out.shape)}"


def test_equals_token_plus_position():
    emb = make()
    idx = torch.randint(V, (2, 5))
    out = emb(idx)
    expected = emb.token_embedding.weight[idx] + emb.position_embedding.weight[torch.arange(5)]
    assert torch.allclose(out, expected), "출력이 'Token 벡터 + 위치 벡터'와 달라요. 두 표에서 꺼낸 값을 더했는지 확인하세요"


def test_same_token_different_position_differs():
    emb = make()
    idx = torch.full((1, 4), 3)            # 같은 Token 4개
    out = emb(idx)[0]
    assert not torch.allclose(out[0], out[1]), \
        "같은 Token이라도 위치가 다르면 벡터가 달라야 해요. 위치 정보를 더했나요?"


def test_shorter_than_block_size_works():
    emb = make()
    for T in (1, 3, BLOCK):
        out = emb(torch.randint(V, (2, T)))
        assert tuple(out.shape) == (2, T, C), f"T={T}일 때도 동작해야 해요. 위치는 T개만 꺼내야 해요"


def test_longer_than_block_size_raises():
    emb = make()
    with pytest.raises(AssertionError):
        emb(torch.randint(V, (1, BLOCK + 1)))


def test_dropout_is_used_in_train_mode_only():
    emb = make(dropout=0.5)
    idx = torch.randint(V, (4, BLOCK))
    emb.eval()
    a, b = emb(idx), emb(idx)
    assert torch.allclose(a, b), "eval 모드에서는 결과가 매번 같아야 해요"
    emb.train()
    torch.manual_seed(1)
    out = emb(idx)
    assert (out == 0).float().mean() > 0.2, "train 모드에서 dropout이 적용되지 않았어요. 반환 직전에 self.dropout을 썼나요?"
