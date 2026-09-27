"""讓 ``iot.voice`` 能以套件根匯入；每個測試各自一份 SQLite。不需要 paho-mqtt／cv2／torch。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_cat_monitoring_system_dir = Path(__file__).resolve().parents[3]
if str(_cat_monitoring_system_dir) not in sys.path:
    sys.path.insert(0, str(_cat_monitoring_system_dir))

from iot.voice import store  # noqa: E402


@pytest.fixture
def db_path(tmp_path) -> str:
    path = str(tmp_path / "owner_reports_test.db")
    yield path
    store.close_connection(path)
