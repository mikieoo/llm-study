"""Day 28. Text Generation (커리큘럼 6-11)"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
    """idx (B, T)에 Token을 하나씩 max_new_tokens개 이어 붙여 반환 (Day 20)"""
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                        # 최근 block_size개만 (Position Embedding 한계)
        logits, _ = model(idx_cond)                            # (B, T, V)
        logits = logits[:, -1, :] / temperature                # 마지막 위치만 (B, V), Day 5 Temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")       # 상위 k개 밖은 확률 0
        probs = F.softmax(logits, dim=-1)                      # (B, V)
        idx_next = torch.multinomial(probs, num_samples=1)     # (B, 1)
        idx = torch.cat([idx, idx_next], dim=1)                # (B, T+1)
    return idx
