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
        self.embedding = GPTEmbedding(config.vocab_size, config.n_embd, config.block_size, config.dropout)
        self.blocks = nn.Sequential(
            *[Block(config.n_embd, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)                          # Pre-LN이라 마지막에 한 번 더 (Day 19)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)       # (B, T, C) → (B, T, V)
        self.apply(self._init_weights)

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
        x = self.embedding(idx)          # (B, T, C)
        x = self.blocks(x)               # (B, T, C)
        x = self.ln_f(x)                 # (B, T, C)
        logits = self.lm_head(x)         # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))   # Day 6
        return logits, loss


if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    idx = torch.randint(65, (2, 8))
    logits, loss = model(idx, idx)
    print("params:", model.num_params())
    print("logits:", tuple(logits.shape), "| loss:", round(loss.item(), 3), "| log(V):", round(torch.log(torch.tensor(65.0)).item(), 3))
