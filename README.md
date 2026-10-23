# llm-study

LLM / Transformer를 기초부터 공부하고, 마지막에 PyTorch로 mini GPT를 직접 구현하는 4주 학습 기록입니다.

- 커리큘럼: [LLM_Transformer_학습_커리큘럼.md](LLM_Transformer_학습_커리큘럼.md)

## 진행 상황

| 주차 | 내용 | 상태 |
|---|---|---|
| 1주 | 딥러닝 최소 기초 | 완료 |
| 2주 | NLP 기초 + Attention | 완료 |
| 3주 | Transformer + GPT | 완료 |
| 4주 | mini GPT 구현 | 완료 |

## 환경 세팅

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch numpy matplotlib tiktoken jupyter ipykernel pytest
```

## 4주차 mini GPT 실행

`week4/` 폴더에서 실행합니다. 날짜별 설명은 `week4/lessons/`에 있어요.

```bash
cd week4
../.venv/bin/python -m pytest tests/test_day22.py -v             # 그날 만든 코드 확인
../.venv/bin/python train.py                                     # 내 코드로 학습 (약 1~2분, Apple Silicon 기준)
../.venv/bin/python generate.py --prompt "ROMEO:" --top_k 20     # 학습된 모델로 생성
```

- `minigpt/`: 직접 채우는 코드 (TODO)
- `solutions/minigpt/`: 완성본. `--impl solutions` 또는 `MINIGPT_IMPL=solutions`로 실행
