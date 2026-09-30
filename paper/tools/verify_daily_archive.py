"""
跨日封存／分母定義驗證（唯讀）
=======================================================
2026-09-29 修正「午夜寫入」與「分母統一」之後，用來確認實際跑過一次跨日的結果。
只讀、不寫任何檔案：
  - Python  tracker_state.json（今天的累積）
  - Python  daily_history.db（多天歷史；用 SQLite 唯讀模式開，不會觸發補欄位遷移）
  - Node-RED global.json（v2_today／v2_today_prev／v2_hourly_persist／v2_daily_history）

對每一天檢查定義是否成立：
  貓在畫面時間 ≤ 系統運行時間；系統運行 ≈ 貓在畫面 + 不在畫面；low_conf 三來源加總 ≈ low_conf
換版當天（2026-09-29）新舊算法混在一起，數字不合理是預期的，會標成「換版過渡日」。

用法：python verify_daily_archive.py            （列最近 5 天）
      python verify_daily_archive.py --days 10
待辦與判讀說明：docs/分母定義與跨日封存-驗證待辦.md
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LoggingConfig, NodeRedConfig  # noqa: E402

TRANSITION_DAYS = {"2026-09-29", "2026/9/29"}  # 換版過渡日：新舊算法混算
TOL = 2.0  # 秒；逐幀累加的浮點誤差與四捨五入容許值

OK, WARN, INFO = "✅", "⚠️ ", "ℹ️ "


def _global_json_path():
    """Node-RED 真正在寫的 global.json：設定值與 baseline_data 預設位置中，最近修改的那個。"""
    candidates = [
        Path(NodeRedConfig.GLOBAL_CONTEXT_PATH),
        Path(__file__).resolve().parent.parent / "baseline_data" / "global.json",
    ]
    existing = [p for p in candidates if p.exists()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def check_day(label, day, run, vis, nd, low=None, low_parts=None):
    """印出一天的三個時間與定義檢查結果。"""
    tag = f"{INFO}換版過渡日（數字不合理是預期的）" if str(day) in TRANSITION_DAYS else ""
    fmt = lambda v: "—" if v is None else f"{v:.1f}"  # noqa: E731
    print(f"  [{label}] {day}  運行 {fmt(run)}  在畫面 {fmt(vis)}  不在畫面 {fmt(nd)}  {tag}")
    if tag:
        return
    if run is None:
        print(f"    {WARN}沒有 run_seconds（修正前的舊資料？）")
        return
    ratio = (vis / run * 100) if run and vis is not None else None
    if vis is not None and vis <= run + TOL:
        print(f"    {OK}在畫面 ≤ 運行（在畫面比例 {ratio:.1f}%）" if ratio is not None else f"    {OK}在畫面 ≤ 運行")
    else:
        print(f"    {WARN}在畫面時間大於運行時間")
    if vis is not None and nd is not None:
        diff = run - (vis + nd)
        mark = OK if abs(diff) <= TOL else WARN
        print(f"    {mark}運行 − (在畫面 + 不在畫面) = {diff:.1f} 秒")
    if low is not None and low_parts:
        s = sum(v for v in low_parts.values() if v is not None)
        mark = OK if abs(s - low) <= TOL else WARN
        parts = "、".join(f"{k} {v:.1f}" for k, v in low_parts.items() if v is not None)
        print(f"    {mark}low_conf {low:.1f} 秒 = {parts}（加總 {s:.1f}）")


def print_period(p):
    """印出 18-24 時段的四個時間（缺的欄位顯示 —，代表舊版資料沒有這個欄位）。"""
    fmt = lambda k: "—" if _f(p.get(k)) is None else f"{_f(p.get(k)):.1f}"  # noqa: E731
    print(f"    {INFO}18-24 時段：運行 {fmt('monitoring_sec')}  在畫面 {fmt('visible_sec')}  "
          f"不在畫面 {fmt('not_detected_sec')}  無法判定 {fmt('low_conf_sec')}")


def section_tracker_state():
    print("\n① Python 今天的累積（tracker_state.json）")
    path = Path(LoggingConfig.TRACKER_STATE_PATH)
    if not path.exists():
        print(f"  {WARN}找不到 {path}")
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    bd = s.get("low_conf_breakdown") or {}
    check_day("今天", s.get("date"), _f(s.get("run_seconds")), _f(s.get("monitoring_seconds")),
              _f(s.get("not_detected_time")), _f(s.get("low_conf_time")),
              {k: _f(v) for k, v in bd.items()} if bd else None)


def section_daily_db(days):
    print(f"\n② Python 多天歷史（daily_history.db，最近 {days} 天）")
    path = Path(LoggingConfig.DAILY_HISTORY_DB_PATH)
    if not path.exists():
        print(f"  {WARN}找不到 {path}")
        return
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(daily_history)")}
        if "run_seconds" not in cols:
            print(f"  {WARN}資料庫還沒有新欄位——新版 Python 還沒開過這個資料庫（開啟時會自動補欄）")
            return
        rows = conn.execute(
            "SELECT day, run_seconds, monitoring_seconds, not_detected_time, low_conf_time, "
            "low_conf_warmup_time, low_conf_uncertain_time, low_conf_sqa_time, periods "
            "FROM daily_history ORDER BY day DESC LIMIT ?", (days,)
        ).fetchall()
    finally:
        conn.close()
    for day, run, vis, nd, low, lw, lu, ls, periods in rows:
        if run is None:
            print(f"  [DB] {day}  （修正前寫入的舊列，沒有新欄位）")
            continue
        check_day("DB", day, run, vis, nd, low, {"warmup": lw, "uncertain": lu, "sqa": ls})
        p = (json.loads(periods) if periods else {}).get("18-24")
        if p:
            print_period(p)


def section_nodered(days):
    path = _global_json_path()
    print(f"\n③ Node-RED（{path}）")
    if path is None:
        print(f"  {WARN}找不到 global.json")
        return
    g = json.loads(path.read_text(encoding="utf-8"))
    t = g.get("v2_today") or {}
    bd = t.get("low_conf_breakdown") or {}
    check_day("v2_today", t.get("date"), _f(t.get("run_seconds")), _f(t.get("monitoring_seconds")),
              _f(t.get("not_detected_time")), _f((t.get("low_conf") or {}).get("time")),
              {k: _f(v) for k, v in bd.items()} if bd else None)
    prev = g.get("v2_today_prev")
    print(f"  {INFO}v2_today_prev（跨日暫存，封存後應清空）= {prev.get('date') if prev else '空'}")
    hp = g.get("v2_hourly_persist")
    print(f"  {INFO}v2_hourly_persist（每小時持久化）= {hp.get('date') if hp else '空'}"
          f"{'，小時：' + ','.join(sorted((hp.get('hourly') or {}).keys())) if hp else ''}")
    hist = g.get("v2_daily_history") or []
    print(f"  v2_daily_history 共 {len(hist)} 天，最近 {min(days, len(hist))} 天：")
    for r in hist[-days:][::-1]:
        if "run_seconds" not in r:
            print(f"  [歷史] {r.get('date')}  （修正前封存的舊紀錄，沒有新欄位）")
            continue
        check_day("歷史", r.get("date"), _f(r.get("run_seconds")), _f(r.get("monitoring_seconds")),
                  _f(r.get("not_detected_time")), _f(r.get("low_conf_time")),
                  {"warmup": _f(r.get("low_conf_warmup_time")), "uncertain": _f(r.get("low_conf_uncertain_time")),
                   "sqa": _f(r.get("low_conf_sqa_time"))})
        if r.get("cat_visible_seconds") != r.get("monitoring_seconds"):
            print(f"    {WARN}cat_visible_seconds ≠ monitoring_seconds（扣兩次的 bug 還在？）")
        print_period((r.get("periods") or {}).get("18-24") or {})
    excluded = g.get("v2_excluded_dates") or []
    print(f"  {INFO}P4 不計入的日期：{', '.join(excluded) if excluded else '（無）'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=5, help="列出最近幾天（預設 5）")
    args = ap.parse_args()
    print("跨日封存／分母定義驗證（唯讀）")
    section_tracker_state()
    section_daily_db(args.days)
    section_nodered(args.days)
    print("\n判讀方式見 docs/分母定義與跨日封存-驗證待辦.md")


if __name__ == "__main__":
    main()
