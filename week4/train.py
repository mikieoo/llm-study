"""mini GPT 학습 스크립트 (Day 28)

사용법 (week4 폴더에서):
    ../.venv/bin/python train.py                      # 내가 만든 minigpt/ 로 학습
    ../.venv/bin/python train.py --impl solutions     # 완성본으로 학습
    ../.venv/bin/python train.py --data data/내글.txt  # 다른 텍스트로 학습
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="mini GPT 학습")
    p.add_argument("--impl", choices=["mine", "solutions"], default="mine", help="mine = minigpt/, solutions = solutions/minigpt/")
    p.add_argument("--data", default=str(ROOT / "data" / "input.txt"))
    p.add_argument("--out", default=str(ROOT / "out"))
    # 모델 크기
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # 학습
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"                       # Apple Silicon GPU
    return "cpu"


def main():
    args = parse_args()
    if args.impl == "solutions":
        sys.path.insert(0, str(ROOT / "solutions"))
    else:
        sys.path.insert(0, str(ROOT))

    from minigpt.data import load_text, train_val_split
    from minigpt.model import GPT, GPTConfig
    from minigpt.sampling import generate
    from minigpt.tokenizer import CharTokenizer
    from minigpt.trainer import train

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # 1. 데이터 → Token ID
    text = load_text(args.data)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    print(f"[data] {len(text):,} 글자 | vocab {tok.vocab_size} | train {len(train_data):,} / val {len(val_data):,}")

    # 2. 모델
    config = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
        n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"[model] {model.num_params():,} params | device {device} | impl {args.impl}")

    # 3. 학습
    t0 = time.time()
    history = train(
        model, train_data, val_data,
        max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
        eval_interval=args.eval_interval, eval_iters=args.eval_iters, device=device,
    )
    print(f"[train] {time.time() - t0:.0f}초")

    # 4. 저장
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(config), "chars": tok.chars}, out / "ckpt.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
        steps = [h["step"] for h in history]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, [h["train"] for h in history], label="train")
        plt.plot(steps, [h["val"] for h in history], label="val")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("mini GPT 학습 곡선")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(out / "loss.png", dpi=120, bbox_inches="tight")
    except Exception as e:  # 그래프는 없어도 학습 결과에는 지장 없음
        print("[plot] 건너뜀:", e)
    print(f"[save] {out / 'ckpt.pt'}")

    # 5. 맛보기 생성
    model.eval()
    start = torch.tensor([tok.encode("\n")], device=device)
    print("\n[sample]")
    print(tok.decode(generate(model, start, max_new_tokens=300)[0].tolist()))


if __name__ == "__main__":
    main()
