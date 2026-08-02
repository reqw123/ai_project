"""Ensure `cat_monitoring_system/` and `paper/` are importable as package roots.

Mirrors `analytics/tests/conftest.py`. `frame_processor.py` additionally does
`from config import ...`, so unlike the analytics tests we also need `paper/`
(one level above `cat_monitoring_system/`) on sys.path for that import to
resolve, regardless of the cwd pytest is invoked from.
"""

import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
_paper_dir = _cat_monitoring_system_dir.parent

for _p in (_cat_monitoring_system_dir, _paper_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
