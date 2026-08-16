"""Ensure `cat_monitoring_system/` is importable as a package root.

Mirrors `analytics/tests/conftest.py` / `trackers/tests/conftest.py`.
`models/keypoint_kalman.py` only needs numpy (no `config` import), but kept
consistent with the rest of the project's conftest convention.
"""

import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
if str(_cat_monitoring_system_dir) not in sys.path:
    sys.path.insert(0, str(_cat_monitoring_system_dir))
