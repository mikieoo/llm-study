"""Day 22. Character Tokenizer (커리큘럼 6-1)"""


class CharTokenizer:
    """글자 하나 = Token 하나인 가장 단순한 Tokenizer"""

    def __init__(self, text: str):
        self.chars = sorted(set(text))                                # 등장한 글자 목록 = Vocabulary
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}        # 글자 → ID
        self.itos = {i: ch for i, ch in enumerate(self.chars)}        # ID → 글자

    @classmethod
    def from_chars(cls, chars):
        """저장해둔 글자 목록으로 Tokenizer 복원 (체크포인트 불러올 때 사용)"""
        return cls("".join(chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


if __name__ == "__main__":
    tok = CharTokenizer("hello world")
    print("vocab:", tok.chars, "| size:", tok.vocab_size)
    ids = tok.encode("hello")
    print("encode('hello') =", ids)
    print("decode(...)     =", tok.decode(ids))
