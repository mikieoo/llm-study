"""Day 27. Feed Forward + Transformer Block (커리큘럼 6-7, 6-8)"""
import torch
import torch.nn as nn

from .multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """Linear(C → 4C) → GELU → Linear(4C → C)  (Day 17)"""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-LN Transformer Block (Day 15, 16)
    x = x + Attention(LN(x))   ← 소통
    x = x + FFN(LN(x))         ← 생각
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


if __name__ == "__main__":
    block = Block(n_embd=32, n_head=4, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(block(x).shape))
