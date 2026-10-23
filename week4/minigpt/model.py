"""Day 27. GPT Model (커리큘럼 6-9)

Token Embedding + Position Embedding → Block × N → LayerNorm → Linear → Vocabulary Logits
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .embedding import GPTEmbedding


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # TODO: 부품 4개 만들기 (이름 그대로)
        #   self.embedding : GPTEmbedding(vocab_size, n_embd, block_size, dropout)   ← Day 24
        #   self.blocks    : Block 을 n_layer 개 이어 붙인 nn.Sequential            ← Day 27 block.py
        #   self.ln_f      : 마지막 LayerNorm(n_embd)  (Pre-LN 이라 끝에 한 번 더, week3/day19.ipynb)
        #   self.lm_head   : Linear(n_embd → vocab_size)
        raise NotImplementedError("TODO: GPT.__init__ 에서 embedding / blocks / ln_f / lm_head 만들기")
        self.apply(self._init_weights)   # 부품을 다 만든 뒤 초기화 (이 줄은 그대로 두세요)

    def _init_weights(self, module):
        # 작은 값으로 시작해야 처음 loss가 log(V) 근처에서 출발 (Day 19)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # TODO: idx (B, T) → logits (B, T, V)
        #   embedding → blocks → ln_f → lm_head   (각 단계 shape 를 주석으로 적어보세요)
        #
        # TODO: targets 가 있으면 loss 계산, 없으면 loss = None
        #   F.cross_entropy 는 (N, V) 와 (N,) 을 받아요 → view 로 펴기 (week1/day06.ipynb)
        #
        # return logits, loss
        raise NotImplementedError("TODO: GPT.forward")


if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    idx = torch.randint(65, (2, 8))
    logits, loss = model(idx, idx)
    print("params:", model.num_params())
    print("logits:", tuple(logits.shape), "| loss:", round(loss.item(), 3), "| log(V):", round(torch.log(torch.tensor(65.0)).item(), 3))
