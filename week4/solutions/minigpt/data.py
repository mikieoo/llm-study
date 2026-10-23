"""Day 23. Training Dataset 만들기 (커리큘럼 6-2)"""
import torch


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(data: torch.Tensor, train_ratio: float = 0.9):
    """앞 90%는 학습용, 뒤 10%는 검증용 (섞지 않음: 문장이 이어져 있으니까)"""
    n = int(len(data) * train_ratio)
    return data[:n], data[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device="cpu"):
    """data에서 길이 block_size 조각을 batch_size개 무작위로 뽑아 (x, y) 반환.
    y는 x를 한 칸 밀어놓은 것 (Day 20)."""
    ix = torch.randint(len(data) - block_size, (batch_size,))           # 각 조각의 시작 위치
    x = torch.stack([data[i : i + block_size] for i in ix])              # (B, T)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])      # (B, T)
    return x.to(device), y.to(device)


if __name__ == "__main__":
    data = torch.arange(100)
    train, val = train_val_split(data)
    print("train:", len(train), "| val:", len(val))
    torch.manual_seed(0)
    x, y = get_batch(train, block_size=8, batch_size=2)
    print("x:", x.tolist())
    print("y:", y.tolist())
