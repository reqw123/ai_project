"""Ensure `cat_monitoring_system/` is importable as a package root.

Mirrors `models/tests/conftest.py` / `processors/tests/conftest.py`.
"""

import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
if str(_cat_monitoring_system_dir) not in sys.path:
    sys.path.insert(0, str(_cat_monitoring_system_dir))
