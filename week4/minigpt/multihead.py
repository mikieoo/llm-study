"""Day 26. Multi-Head Attention (커리큘럼 6-6)"""
import torch
import torch.nn as nn

from .attention import Head


class MultiHeadAttention(nn.Module):
    """Head 여러 개를 병렬로 → Concat → Linear"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd는 n_head로 나누어떨어져야 합니다"

        # TODO 1: Head 하나의 크기 head_size 정하기
        #   Head n_head개를 이어 붙였을 때 다시 n_embd가 되어야 해요 (week2/day14.ipynb 참고)

        # TODO 2: self.heads = Day 25에서 만든 Head를 n_head개 담은 nn.ModuleList
        #   Head(n_embd, head_size, block_size, dropout)
        #   주의: 그냥 파이썬 list에 담으면 파라미터가 모델에 등록되지 않아요

        # TODO 3: self.proj = 이어 붙인 결과를 한 번 더 섞는 Linear (C → C)

        # TODO 4: self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("TODO: MultiHeadAttention.__init__ 을 완성하세요")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # TODO 5: 모든 Head에 x를 넣고, 결과들을 마지막 축(dim=-1)으로 이어 붙이기 → (B, T, C)
        # TODO 6: proj → dropout 을 거쳐 반환 → (B, T, C)
        raise NotImplementedError("TODO: MultiHeadAttention.forward 를 완성하세요")


if __name__ == "__main__":
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(mha(x).shape))
