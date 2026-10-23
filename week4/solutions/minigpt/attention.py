"""Day 25. Self-Attention + Causal Mask (커리큘럼 6-4, 6-5) — Head 하나"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Causal Self-Attention Head 하나: softmax(QK^T / √dk + mask) V"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)       # Day 12
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 학습하는 값이 아니라서 parameter가 아닌 buffer로 등록 (Day 14)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)                                                   # (B, T, hs)
        q = self.query(x)                                                 # (B, T, hs)
        v = self.value(x)                                                 # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5               # (B, T, T)  Day 13: / √dk
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))     # Day 14: 미래 가리기
        wei = F.softmax(wei, dim=-1)                                      # Day 5: 행마다 합 1
        wei = self.dropout(wei)
        return wei @ v                                                    # (B, T, hs)


if __name__ == "__main__":
    head = Head(n_embd=32, head_size=8, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(head(x).shape))
