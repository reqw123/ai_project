# -*- coding: utf-8 -*-
import json, sys

path = r'C:\Users\homec\.node-red\context\global\global.json'
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
