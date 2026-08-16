# -*- coding: utf-8 -*-
"""一次性工具：把 Node-RED global.json 的 v2_daily_history 灌進 Python 自己
獨立的 analytics/daily_store.py（SQLite），讓 /dashboard/baseline 不用等
7 天真實時間累積才有基線可看。

**只讀 Node-RED 的資料，執行一次性搬遷，之後兩邊繼續各自獨立累積**——不是
把 daily_store 改成持續讀 global.json，那樣又會變回互相依賴。對應
docs/資料層架構現況與統一管理評估.md 第七節當初預告的「轉接層」：
「需要一個轉接層把 v2_daily_history 的欄位轉成 analytics.baseline.DailyRecord」。

用法：
    python _tools/2_backfill_daily_store.py

冪等：同一天重複跑會被 daily_store.save_day() 的 upsert 覆蓋，不會重複。
"""
import json
import sys
from datetime import date as _date
from pathlib import Path

_paper_dir = Path(__file__).resolve().parent.parent
_cat_monitoring_system_dir = _paper_dir / "cat_monitoring_system"
for _p in (str(_paper_dir), str(_cat_monitoring_system_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import NodeRedConfig  # noqa: E402

from analytics import daily_store  # noqa: E402
from analytics.baseline import DailyRecord  # noqa: E402

_FIELDS = (
    "monitoring_seconds",
    "walk_time",
    "walk_count",
    "stop_time",
    "stop_count",
    "lick_time",
    "lick_count",
    "scratch_time",
    "scratch_count",
    "shake_count",
    "active_time",
    "rest_time",
)


def _to_iso_date(raw):
    """對齊 analytics_deviation_bridge.json 舊版 toISODate() 的正規化邏輯：
    接受 ISO（YYYY-MM-DD）或 v2_daily_history 實際寫入的 YYYY/M/D 本地格式，
    無法辨識的回傳 None。"""
    if raw is None:
        return None
    s = str(raw).strip()
    try:
        return _date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        pass
    parts = s.split("/")
    if len(parts) == 3:
        try:
            y, mo, d = int(parts[0]), int(parts[1]), int(parts[2].split(" ")[0])
            return _date(y, mo, d).isoformat()
        except ValueError:
            return None
    return None


def main():
    path = NodeRedConfig.GLOBAL_CONTEXT_PATH
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            ctx = json.load(f)
    except FileNotFoundError:
        out.append(f"找不到 {path}，略過。")
        _flush(out)
        return

    history = ctx.get("v2_daily_history", [])
    written, skipped = 0, 0
    for d in history:
        iso = _to_iso_date(d.get("date"))
        if iso is None:
            skipped += 1
            out.append(f"  跳過（日期格式無法辨識）: {d.get('date')!r}")
            continue
        kwargs = {"day": _date.fromisoformat(iso)}
        for field_name in _FIELDS:
            if field_name in d:
                kwargs[field_name] = d[field_name]
        daily_store.save_day(DailyRecord(**kwargs))
        written += 1

    out.append(f"來源: {path}（v2_daily_history 共 {len(history)} 筆）")
    out.append(f"寫入 daily_store: {written} 筆，跳過: {skipped} 筆")
    out.append(f"daily_store 目前累積天數: {daily_store.record_count()}")
    _flush(out)


def _flush(lines):
    sys.stdout.buffer.write("\n".join(lines).encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
