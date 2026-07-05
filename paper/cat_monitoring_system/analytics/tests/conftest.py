"""Ensure `cat_monitoring_system/` is importable as the package root.

Mirrors what server/routes.py already does for `config` — the rest of the
codebase assumes `cat_monitoring_system/` (not `paper/`) is on sys.path so
sibling packages (`detectors`, `models`, ...) resolve as flat top-level
imports. This lets `pytest` be invoked from any cwd (repo root, `paper/`,
or `cat_monitoring_system/`) and still resolve `from analytics.baseline
import ...` the same way the running application does.
"""
import sys
from pathlib import Path

_cat_monitoring_system_dir = Path(__file__).resolve().parents[2]
if str(_cat_monitoring_system_dir) not in sys.path:
    sys.path.insert(0, str(_cat_monitoring_system_dir))
