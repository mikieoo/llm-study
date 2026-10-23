"""Day 28. Text Generation (커리큘럼 6-11)"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
    """idx (B, T)에 Token을 하나씩 max_new_tokens개 이어 붙여 반환 (Day 20)"""
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        # TODO 1: 최근 block_size개 Token만 잘라서 idx_cond 만들기 (Position Embedding 한계, week2/day10.ipynb)
        # TODO 2: model(idx_cond)로 logits (B, T, V) 얻기 — model은 (logits, loss)를 돌려줘요
        # TODO 3: 마지막 위치의 logits만 꺼내고 (B, V), temperature로 나누기 (week1/day05.ipynb 7절)
        # TODO 4: top_k가 None이 아니면 상위 k개보다 작은 logits를 -inf로 (힌트: torch.topk)
        # TODO 5: softmax로 확률 (B, V) → torch.multinomial로 1개 뽑기 (B, 1)
        # TODO 6: torch.cat으로 idx 뒤에 붙이기 → (B, T+1)
        raise NotImplementedError("TODO: generate의 반복문 안을 채워주세요")
    return idx
