"""학습된 mini GPT로 텍스트 생성 (Day 28)

사용법 (week4 폴더에서, train.py를 먼저 실행해야 out/ckpt.pt가 생김):
    ../.venv/bin/python generate.py --prompt "ROMEO:"
    ../.venv/bin/python generate.py --prompt "ROMEO:" --temperature 0.8 --top_k 20 --max_new_tokens 500
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description="mini GPT 텍스트 생성")
    p.add_argument("--impl", choices=["mine", "solutions"], default="mine")
    p.add_argument("--ckpt", default=str(ROOT / "out" / "ckpt.pt"))
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "solutions") if args.impl == "solutions" else str(ROOT))
    from minigpt.model import GPT, GPTConfig
    from minigpt.sampling import generate
    from minigpt.tokenizer import CharTokenizer

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    tok = CharTokenizer.from_chars(ckpt["chars"])
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    unknown = [ch for ch in args.prompt if ch not in tok.stoi]
    if unknown:
        sys.exit(f"학습 데이터에 없는 글자가 prompt에 있어요: {unknown}")

    idx = torch.tensor([tok.encode(args.prompt)])
    out = generate(model, idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
