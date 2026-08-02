"""Ensure `cat_monitoring_system/` and `paper/` are importable as package roots.

Mirrors the other `*/tests/conftest.py` files in this project.
"""

import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
_paper_dir = _cat_monitoring_system_dir.parent

for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
