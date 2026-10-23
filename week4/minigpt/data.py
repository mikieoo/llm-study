"""Day 23. Training Dataset 만들기 (커리큘럼 6-2)"""
import torch


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(data: torch.Tensor, train_ratio: float = 0.9):
    """앞 90%는 학습용, 뒤 10%는 검증용 (섞지 않음: 문장이 이어져 있으니까)"""
    # TODO: 몇 번째 위치에서 자를지 계산하고 (int로!), 앞부분과 뒷부분을 슬라이싱해서 반환하세요.
    #       힌트: data[:n], data[n:]
    raise NotImplementedError("TODO: train_val_split 구현")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device="cpu"):
    """data에서 길이 block_size 조각을 batch_size개 무작위로 뽑아 (x, y) 반환.
    y는 x를 한 칸 밀어놓은 것 (Day 20)."""
    # TODO 1: 조각의 시작 위치 batch_size개를 무작위로 뽑기 (torch.randint)
    #         주의: y는 한 칸 더 뒤까지 필요하니까, 시작 위치의 최댓값을 잘 정해야 해요.
    # TODO 2: x = 각 시작 위치에서 block_size개 → torch.stack으로 (B, T)
    # TODO 3: y = x보다 한 칸 뒤에서 시작하는 block_size개 → (B, T)
    # TODO 4: 둘 다 device로 옮겨서 반환
    #         (week3/day20.ipynb의 get_batch 참고)
    raise NotImplementedError("TODO: get_batch 구현")


if __name__ == "__main__":
    data = torch.arange(100)
    train, val = train_val_split(data)
    print("train:", len(train), "| val:", len(val))
    torch.manual_seed(0)
    x, y = get_batch(train, block_size=8, batch_size=2)
    print("x:", x.tolist())
    print("y:", y.tolist())
