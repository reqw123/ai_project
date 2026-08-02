"""Ensure `paper/`（config.py 所在目錄）在 sys.path 上。"""

import sys
from pathlib import Path

_paper_dir = Path(__file__).resolve().parents[1]
if str(_paper_dir) not in sys.path:
    sys.path.insert(0, str(_paper_dir))
