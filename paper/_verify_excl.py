# -*- coding: utf-8 -*-
import json
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
p4  = nodes['e3809f1cf09346ad']
sm  = nodes['ff00000000000010']
eng = nodes['4b0c232763de1461']
wr  = nodes['c5ce881398ee6376']
bl  = nodes['66a22e0ec663d07d']

checks = [
  # 健康預警引擎
  ('Eng: v2_daily_history payload',   'v2_daily_history' in eng['func']),
  ('Eng: v2_excluded_dates payload',  'v2_excluded_dates' in eng['func']),
  ('Eng: flow.set v2_today_risk',     "flow.set('v2_today_risk'" in eng['func']),
  # 每日歷史記錄寫入
  ('Wr: risk field in rec',           'risk: flow.get' in wr['func']),
  # 基線計算器
  ('Bl: v2_excluded_dates filter',    'v2Excluded' in bl['func']),
  ('Bl: noExcl filter',               'noExcl' in bl['func']),
  # 使用者設定管理器
  ('SM: toggle_exclude_date',         'toggle_exclude_date' in sm['func']),
  ('SM: output 1 trigger',            'return [msg, null]' in sm['func']),
  # P4 UI
  ('P4: CSS p4-history',              '.p4-history{' in p4['format']),
  ('P4: CSS p4-hist-excl-btn',        'p4-hist-excl-btn' in p4['format']),
  ('P4: HTML history section',        'class="p4-history"' in p4['format']),
  ('P4: HTML toggleExclude',          'toggleExclude' in p4['format']),
  ('P4: JS historyDays',              'scope.historyDays' in p4['format']),
  ('P4: JS excludedDates',            'scope.excludedDates' in p4['format']),
  ('P4: JS watch historyDays update', 'evtDts' in p4['format']),
  ('P4: JS watch v2_excluded_dates',  'v2_excluded_dates' in p4['format']),
]
for lbl, found in checks:
    print(('OK  ' if found else 'MISS'), lbl)
