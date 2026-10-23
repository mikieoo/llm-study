"""Day 24. Embedding (커리큘럼 6-3) — Token Embedding + Position Embedding"""
import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO 1: Token Embedding 표 만들기 → self.token_embedding
        #   Token ID 하나당 길이 n_embd 벡터 한 줄. 표 크기는 (vocab_size, n_embd)  (week2/day09.ipynb 참고)
        # TODO 2: Position Embedding 표 만들기 → self.position_embedding
        #   위치(0 ~ block_size-1) 하나당 벡터 한 줄. 표 크기는 (block_size, n_embd)  (week2/day10.ipynb 참고)
        raise NotImplementedError("TODO: token_embedding, position_embedding 만들기")
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"입력 길이 {T}가 block_size {self.block_size}보다 깁니다"
        # TODO 3: Token 벡터 꺼내기          → tok: (B, T, C)
        # TODO 4: 위치 0, 1, ..., T-1 벡터 꺼내기 → pos: (T, C)
        #   힌트: 위치 번호는 torch.arange(T, device=idx.device)  (idx와 같은 device여야 해요)
        # TODO 5: 둘을 더하고 dropout 적용해서 반환 → (B, T, C)  (broadcasting: (B,T,C) + (T,C))
        raise NotImplementedError("TODO: forward 완성하기")


if __name__ == "__main__":
    emb = GPTEmbedding(vocab_size=65, n_embd=32, block_size=8)
    idx = torch.randint(65, (2, 8))
    print("idx:", tuple(idx.shape), "→ out:", tuple(emb(idx).shape))
