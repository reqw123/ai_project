"""論文分析用：把飼主紀錄／每一句話匯出成 CSV（UTF-8 with BOM，Excel 直接開）。

    cd paper/cat_monitoring_system
    python -m iot.voice.export reports    --out reports.csv    [--from 2026-09-01] [--to 2026-09-30] [--include-deleted]
    python -m iot.voice.export utterances --out utterances.csv [--from 2026-09-01] [--to 2026-09-30]

* ``reports``：分類出來的紀錄，依「事件那一天」篩選；``deleted=1`` 的是飼主在儀表板刪掉的誤判
  （加 ``--include-deleted`` 才會出現，算實際使用的誤判率用）。
* ``utterances``：送進來的每一句話與沒分類的原因，人工標註後可以算召回率。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from typing import Optional

from iot.voice import store


def _day_start_ts(day: Optional[str]) -> Optional[float]:
    return datetime.fromisoformat(day).timestamp() if day else None


def _day_end_ts(day: Optional[str]) -> Optional[float]:
    return (datetime.fromisoformat(day) + timedelta(days=1)).timestamp() - 1e-6 if day else None


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def export(kind: str, out_path: str, day_from: Optional[str] = None, day_to: Optional[str] = None,
           include_deleted: bool = False, db_path: Optional[str] = None) -> int:
    if kind == "reports":
        rows = store.reports_between(day_from, day_to, include_deleted=include_deleted, db_path=db_path)
        cols = ["id", "day", "time", "kind", "category", "label", "text", "matched", "related",
                "day_offset", "device_id", "source", "deleted"]
    elif kind == "utterances":
        rows = store.utterances_between(_day_start_ts(day_from), _day_end_ts(day_to), db_path=db_path)
        cols = ["id", "time", "text", "normalized", "matched", "reason", "mode", "device_id", "source"]
    else:
        raise ValueError(f"未知的匯出種類：{kind!r}")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "time": _fmt_ts(r["ts"])})
    return len(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m iot.voice.export", description="匯出飼主語音紀錄（CSV）")
    p.add_argument("kind", choices=["reports", "utterances"])
    p.add_argument("--out", required=True)
    p.add_argument("--from", dest="day_from")
    p.add_argument("--to", dest="day_to")
    p.add_argument("--include-deleted", action="store_true")
    p.add_argument("--db", dest="db_path")
    a = p.parse_args(argv)
    n = export(a.kind, a.out, a.day_from, a.day_to, a.include_deleted, a.db_path)
    print(f"已匯出 {n} 筆 → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
