"""Day 22. Character Tokenizer (커리큘럼 6-1)"""


class CharTokenizer:
    """글자 하나 = Token 하나인 가장 단순한 Tokenizer"""

    def __init__(self, text: str):
        # TODO 1: text에 등장한 글자를 중복 없이 모아 정렬한 리스트 → self.chars  (= Vocabulary)
        # TODO 2: 글자 → ID 딕셔너리 → self.stoi
        # TODO 3: ID → 글자 딕셔너리 → self.itos
        # 힌트: week2/day08.ipynb 2절에서 만든 것과 같아요. set, sorted, enumerate
        raise NotImplementedError("TODO: chars / stoi / itos 만들기")

    @classmethod
    def from_chars(cls, chars):
        """저장해둔 글자 목록으로 Tokenizer 복원 (체크포인트 불러올 때 사용)"""
        return cls("".join(chars))

    @property
    def vocab_size(self) -> int:
        # TODO: Vocabulary 크기 (글자 종류 수)
        raise NotImplementedError("TODO: vocab_size")

    def encode(self, s: str) -> list[int]:
        # TODO: 문자열 → ID 리스트  (self.stoi 사용)
        raise NotImplementedError("TODO: encode")

    def decode(self, ids) -> str:
        # TODO: ID 리스트 → 문자열  (self.itos 사용)
        # 주의: ids에 torch.Tensor의 원소(0차원 텐서)가 들어와도 동작하도록 int(i)로 바꿔서 찾기
        raise NotImplementedError("TODO: decode")


if __name__ == "__main__":
    tok = CharTokenizer("hello world")
    print("vocab:", tok.chars, "| size:", tok.vocab_size)
    ids = tok.encode("hello")
    print("encode('hello') =", ids)
    print("decode(...)     =", tok.decode(ids))
