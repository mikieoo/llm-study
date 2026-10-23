"""Day 22 테스트: Character Tokenizer

    ../.venv/bin/python -m pytest tests/test_day22.py -v
"""
from pathlib import Path

import torch

from minigpt.tokenizer import CharTokenizer

DATA = Path(__file__).resolve().parents[1] / "data" / "input.txt"


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer("banana")
    assert tok.chars == ["a", "b", "n"], "chars는 등장한 글자를 중복 없이 '정렬'한 리스트여야 해요"
    assert tok.vocab_size == 3, "vocab_size는 글자 종류의 개수예요"


def test_stoi_itos_are_inverse():
    tok = CharTokenizer("hello world")
    assert set(tok.stoi.values()) == set(range(tok.vocab_size)), "ID는 0부터 vocab_size-1까지 빠짐없이 써야 해요"
    for ch, i in tok.stoi.items():
        assert tok.itos[i] == ch, "itos는 stoi를 뒤집은 것이어야 해요"


def test_encode_order_follows_sorted_vocab():
    tok = CharTokenizer("cab")
    assert tok.encode("abc") == [0, 1, 2], "정렬된 순서대로 ID가 붙어야 해요 (a=0, b=1, c=2)"
    assert tok.encode("") == [], "빈 문자열은 빈 리스트로"


def test_round_trip():
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids), "encode는 int 리스트를 돌려줘야 해요"
    assert len(ids) == len(text), "글자 단위라 Token 수 = 글자 수예요"
    assert tok.decode(ids) == text, "decode(encode(s))는 원래 문자열로 돌아와야 해요"


def test_decode_accepts_tensor():
    tok = CharTokenizer("hello")
    ids = torch.tensor(tok.encode("hello"))
    assert tok.decode(ids) == "hello", "decode는 torch.Tensor도 받을 수 있어야 해요 (원소를 int로 바꾸기)"
    assert tok.decode(ids.tolist()) == "hello"


def test_korean():
    text = "나는 오늘 학교에 간다"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode("학교에 간다")) == "학교에 간다", "한글도 글자 하나가 Token 하나여야 해요"


def test_from_chars_restores_same_mapping():
    tok = CharTokenizer("hello world")
    tok2 = CharTokenizer.from_chars(tok.chars)
    assert tok2.stoi == tok.stoi and tok2.itos == tok.itos, "from_chars로 만든 Tokenizer가 원래와 같아야 해요"


def test_shakespeare_vocab():
    text = DATA.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    assert tok.vocab_size == 65, "Tiny Shakespeare의 글자 종류는 65개여야 해요"
    sample = text[:1000]
    assert tok.decode(tok.encode(sample)) == sample
