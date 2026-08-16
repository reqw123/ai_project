# -*- coding: utf-8 -*-
"""唯讀診斷工具：印出 Node-RED global.json（對應 config.py 的
NodeRedConfig.GLOBAL_CONTEXT_PATH，預設 C:\\a\\global.json）目前累積的
個體化基線相關資料。不需要 main.py／Flask／Node-RED 任一個在跑，純粹讀檔案。

會印出：
  - v2_daily_history：每日監測記錄，含監測時數、是否達最短有效時數
    （OK/SHORT）、是否被標記排除（[EXCLUDED]）、當天 risk score（若已算過）
  - v2_excluded_dates：目前排除清單
  - v2_user_settings.baseline_days：使用者設定的基線天數視窗
  - v2_baseline：目前算出的基線快照（天數、計算時間）；尚未算過會明確印出提示

用途：確認 Node-RED 端實際累積了多少天歷史、資料品質如何（監測時數夠不夠、
有沒有被排除），常搭配 2_backfill_daily_store.py 使用——先用這支確認
Node-RED 那邊有資料，再用 2_backfill_daily_store.py 把資料搬進 Python 自己
獨立的 analytics/daily_store.py（SQLite）。

用法：
    python _tools/1_read_context.py

只讀不寫，執行多少次都安全，不會影響任何系統狀態。
"""
import json, sys

path = r'C:\a\global.json'
with open(path, encoding='utf-8') as f:
    ctx = json.load(f)

out = []
hist = ctx.get('v2_daily_history', [])
excl = ctx.get('v2_excluded_dates', [])
baseline = ctx.get('v2_baseline', None)
settings = ctx.get('v2_user_settings', {})

out.append(f"=== v2_daily_history: {len(hist)} ===")
for d in hist:
    secs = d.get('monitoring_seconds', 0)
    hrs  = round(secs/3600, 2)
    valid = 'OK' if secs >= 3600 else 'SHORT'
    excl_flag = '  [EXCLUDED]' if d.get('date') in excl else ''
    risk = d.get('risk') or {}
    rscore = risk.get('score', '--')
    out.append(f"  {d.get('date')}  {hrs}h ({secs}s)  {valid}{excl_flag}  risk={rscore}")

out.append(f"\nv2_excluded_dates: {excl if excl else 'none'}")
out.append(f"baseline_days setting: {settings.get('baseline_days',7)}")
if baseline:
    out.append(f"baseline days_count: {baseline.get('days_count','?')}")
    out.append(f"baseline computed_at: {baseline.get('computed_at','?')}")
else:
    out.append("no baseline computed yet")

sys.stdout.buffer.write('\n'.join(out).encode('utf-8'))
sys.stdout.buffer.write(b'\n')
