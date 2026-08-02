"""Ensure `cat_monitoring_system/` and `paper/` are importable as package roots.

Mirrors `analytics/tests/conftest.py` / `processors/tests/conftest.py`.
`behavior_tracker.py` does `from config import ...`, so we need `paper/`
(one level above `cat_monitoring_system/`) on sys.path as well.
"""

import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
_paper_dir = _cat_monitoring_system_dir.parent

for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
