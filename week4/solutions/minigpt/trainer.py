"""Day 28. 학습 루프 (커리큘럼 6-10)"""
import torch

from .data import get_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    """train/val 각각 batch 여러 개의 평균 loss (한 batch만 보면 너무 출렁여서)"""
    model.eval()                                     # Dropout 끄기
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_data, val_data, *, max_steps, batch_size, lr, eval_interval, eval_iters, device, log=print):
    """기본 학습 루프 (Day 7). 기록한 loss 목록을 반환"""
    block_size = model.config.block_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device)
            history.append({"step": step, **losses})
            log(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if step == max_steps:
            break

        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)            # 1. 예측 + loss
        optimizer.zero_grad()            # 2. 지난 gradient 지우기
        loss.backward()                  # 3. gradient 계산
        optimizer.step()                 # 4. Weight 수정
    return history
