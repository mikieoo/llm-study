"""Day 24. Embedding (커리큘럼 6-3) — Token Embedding + Position Embedding"""
import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)        # Day 9: (V, C) 표
        self.position_embedding = nn.Embedding(block_size, n_embd)     # Day 10: (block_size, C) 표
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"입력 길이 {T}가 block_size {self.block_size}보다 깁니다"
        tok = self.token_embedding(idx)                                          # (B, T, C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))        # (T, C)
        return self.dropout(tok + pos)                                           # (B, T, C) broadcasting


if __name__ == "__main__":
    emb = GPTEmbedding(vocab_size=65, n_embd=32, block_size=8)
    idx = torch.randint(65, (2, 8))
    print("idx:", tuple(idx.shape), "→ out:", tuple(emb(idx).shape))
