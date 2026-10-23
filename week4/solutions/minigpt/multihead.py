"""Day 26. Multi-Head Attention (커리큘럼 6-6)"""
import torch
import torch.nn as nn

from .attention import Head


class MultiHeadAttention(nn.Module):
    """Head 여러 개를 병렬로 → Concat → Linear"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd는 n_head로 나누어떨어져야 합니다"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_head * hs) = (B, T, C)
        return self.dropout(self.proj(out))                     # (B, T, C)


if __name__ == "__main__":
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(mha(x).shape))
