"""Day 27. Feed Forward + Transformer Block (커리큘럼 6-7, 6-8)"""
import torch
import torch.nn as nn

from .multihead import MultiHeadAttention


class FeedForward(nn.Module):
    """Linear(C → 4C) → GELU → Linear(4C → C)  (Day 17)"""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        # TODO: self.net = nn.Sequential(...) 만들기
        #   1) C → 4C 로 늘리는 Linear
        #   2) GELU (Day 4: 이게 없으면 Linear 두 개가 하나로 합쳐져요)
        #   3) 4C → C 로 줄이는 Linear
        #   4) Dropout(dropout)
        raise NotImplementedError("TODO: FeedForward.__init__ 에서 self.net 만들기")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: (B, T, C) → (B, T, C)
        raise NotImplementedError("TODO: FeedForward.forward")


class Block(nn.Module):
    """Pre-LN Transformer Block (Day 15, 16)
    x = x + Attention(LN(x))   ← 소통
    x = x + FFN(LN(x))         ← 생각
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO: 부품 4개 만들기 (이름을 꼭 이렇게: 테스트와 GPT가 이 이름을 써요)
        #   self.ln1   : LayerNorm(n_embd)
        #   self.attn  : MultiHeadAttention(n_embd, n_head, block_size, dropout)   ← Day 26
        #   self.ln2   : LayerNorm(n_embd)
        #   self.ffwd  : FeedForward(n_embd, dropout)
        raise NotImplementedError("TODO: Block.__init__ 에서 ln1 / attn / ln2 / ffwd 만들기")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Pre-LN 두 줄 (week3/day15.ipynb)
        #   LayerNorm 은 '안쪽'에, Residual(x + ...) 은 '바깥'에
        raise NotImplementedError("TODO: Block.forward")


if __name__ == "__main__":
    block = Block(n_embd=32, n_head=4, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(block(x).shape))
