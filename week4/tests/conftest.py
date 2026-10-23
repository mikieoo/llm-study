"""테스트가 어느 구현을 검사할지 정하는 설정

    ../.venv/bin/python -m pytest tests/test_day22.py                      # 내 코드 (minigpt/)
    MINIGPT_IMPL=solutions ../.venv/bin/python -m pytest tests/            # 완성본 (solutions/minigpt/)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPL = os.environ.get("MINIGPT_IMPL", "mine")
sys.path.insert(0, str(ROOT / "solutions") if IMPL == "solutions" else str(ROOT))
