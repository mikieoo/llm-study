"""Day 25. Self-Attention + Causal Mask (커리큘럼 6-4, 6-5) — Head 하나"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """Causal Self-Attention Head 하나: softmax(QK^T / √dk + mask) V"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # TODO 1: Key / Query / Value를 만드는 Linear Layer 3개
        #   - 입력 길이 n_embd → 출력 길이 head_size
        #   - Attention에서는 bias를 쓰지 않아요 (Day 3 6절, Day 12 참고)
        #   - 이름은 꼭 self.key, self.query, self.value 로 (테스트가 이 이름을 씁니다)

        # TODO 2: 아래 삼각형 마스크 (block_size, block_size)를 "tril"이라는 이름의 buffer로 등록
        #   - 학습하는 값이 아니라서 parameter가 아닌 buffer로 (Day 14 참고)
        #   - 힌트: torch.tril, torch.ones, self.register_buffer

        self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("TODO: Head.__init__의 key/query/value와 tril buffer를 만들어주세요")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # TODO 3: x로 k, q, v 만들기 → 각각 (B, T, head_size)

        # TODO 4: Attention Score = q @ k^T / √head_size → (B, T, T)
        #   - batch가 있으니 .T 대신 transpose(-2, -1) (Day 2 5-3, Day 13 참고)

        # TODO 5: 미래 Token 가리기 — tril의 [:T, :T]가 0인 칸을 -inf로 (Day 14 참고)
        #   - T가 block_size보다 짧을 수도 있으니 꼭 잘라서 쓰기

        # TODO 6: 마지막 축으로 softmax → 행마다 합이 1 (Day 5 참고), 그다음 self.dropout 적용

        # TODO 7: weight @ v → (B, T, head_size) 반환
        raise NotImplementedError("TODO: Head.forward를 완성해주세요")


if __name__ == "__main__":
    head = Head(n_embd=32, head_size=8, block_size=8)
    x = torch.randn(2, 8, 32)
    print("x:", tuple(x.shape), "→ out:", tuple(head(x).shape))
