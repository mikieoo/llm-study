"""Day 23 테스트: train_val_split, get_batch

    ../.venv/bin/python -m pytest tests/test_day23.py -v
"""
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def test_load_text():
    from minigpt.data import load_text

    text = load_text(str(DATA_PATH))
    assert isinstance(text, str)
    assert len(text) > 1_000_000, "Tiny Shakespeare는 약 110만 글자예요"


def test_split_sizes():
    from minigpt.data import train_val_split

    data = torch.arange(1000)
    train, val = train_val_split(data)
    assert len(train) == 900, "기본값은 앞 90%가 train이어야 해요"
    assert len(val) == 100, "나머지 10%가 val이어야 해요"


def test_split_keeps_order_and_covers_all():
    from minigpt.data import train_val_split

    data = torch.arange(1000)
    train, val = train_val_split(data, train_ratio=0.8)
    assert len(train) == 800, "train_ratio 인자를 사용해야 해요"
    assert torch.equal(torch.cat([train, val]), data), "섞지 말고 앞/뒤로 잘라야 하고, 빠지거나 겹치는 부분이 없어야 해요"


def test_get_batch_shapes_and_dtype():
    from minigpt.data import get_batch

    data = torch.arange(500)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert tuple(x.shape) == (4, 8), f"x의 shape는 (batch_size, block_size)여야 해요. 지금: {tuple(x.shape)}"
    assert tuple(y.shape) == (4, 8), f"y의 shape는 (batch_size, block_size)여야 해요. 지금: {tuple(y.shape)}"
    assert x.dtype == data.dtype and y.dtype == data.dtype


def test_y_is_x_shifted_by_one():
    from minigpt.data import get_batch

    torch.manual_seed(0)
    data = torch.randint(0, 50, (300,))
    x, y = get_batch(data, block_size=10, batch_size=16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y는 x를 한 칸 왼쪽으로 민 것이어야 해요 (y[t] = x[t+1])"


def test_rows_are_contiguous_slices_of_data():
    from minigpt.data import get_batch

    data = torch.arange(1000)  # 값 = 위치라서 어디서 잘랐는지 바로 보여요
    x, y = get_batch(data, block_size=6, batch_size=32)
    for row_x, row_y in zip(x, y):
        start = row_x[0].item()
        assert torch.equal(row_x, data[start : start + 6]), "x의 각 행은 data에서 연속된 조각이어야 해요"
        assert torch.equal(row_y, data[start + 1 : start + 7]), "y의 각 행은 x보다 한 칸 뒤에서 시작하는 조각이어야 해요"


def test_never_goes_out_of_range():
    from minigpt.data import get_batch

    data = torch.arange(20)
    for seed in range(50):
        torch.manual_seed(seed)
        x, y = get_batch(data, block_size=8, batch_size=8)  # 범위를 넘으면 stack에서 에러가 나거나 길이가 짧아져요
        assert y.max().item() <= 19
    # 가능한 마지막 시작 위치(len - block_size - 1)도 실제로 뽑힐 수 있어야 해요
    starts = set()
    for seed in range(200):
        torch.manual_seed(seed)
        x, _ = get_batch(data, block_size=8, batch_size=8)
        starts.update(x[:, 0].tolist())
    assert 11 in starts, "마지막으로 가능한 시작 위치까지 뽑힐 수 있게 범위를 정해주세요"


def test_random_offsets():
    from minigpt.data import get_batch

    torch.manual_seed(0)
    data = torch.arange(10_000)
    x, _ = get_batch(data, block_size=8, batch_size=32)
    assert len(set(x[:, 0].tolist())) > 1, "시작 위치는 batch마다 무작위여야 해요"


def test_device_argument():
    from minigpt.data import get_batch

    data = torch.arange(100)
    x, y = get_batch(data, block_size=4, batch_size=2, device="cpu")
    assert x.device.type == "cpu" and y.device.type == "cpu"
    if torch.backends.mps.is_available():
        x, y = get_batch(data, block_size=4, batch_size=2, device="mps")
        assert x.device.type == "mps" and y.device.type == "mps", "device 인자로 받은 곳으로 옮겨야 해요"
