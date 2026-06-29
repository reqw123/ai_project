# -*- coding: utf-8 -*-
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data}
wr = nodes['c5ce881398ee6376']

old = (
    "let history = global.get('v2_daily_history', 'file') || [];\n"
    "history.push(rec);\n"
    "if (history.length > 90) history = history.slice(-90);\n"
    "global.set('v2_daily_history', history, 'file');"
)
new = (
    "let history = global.get('v2_daily_history', 'file') || [];\n"
    "history = history.filter(h => h.date !== date); // 同一天只保留最新一筆，避免重複污染基線\n"
    "history.push(rec);\n"
    "if (history.length > 90) history = history.slice(-90);\n"
    "global.set('v2_daily_history', history, 'file');"
)

found = old in wr['func']
print('anchor found:', found)
if found:
    wr['func'] = wr['func'].replace(old, new, 1)
    with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print('OK  WR dedup saved')
else:
    # 印出實際片段供檢查
    idx = wr['func'].find('history.push')
    print('context:', repr(wr['func'][max(0,idx-120):idx+60]))
