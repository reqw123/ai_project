# -*- coding: utf-8 -*-
"""
修正項目：
  1. 歷史資料管理 UI：加入 ✓ 計入 / 太短 / 不計入 資格標籤
  2. 每日歷史記錄寫入：同一天若已有紀錄，改為更新而非重複新增（避免兩筆 2026/6/16）
"""
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data}
p4 = nodes['e3809f1cf09346ad']   # P4 面板
wr = nodes['c5ce881398ee6376']   # 每日歷史記錄寫入

ok = []
fmt = p4['format']

# ══════════════════════════════════════════════════════════════════════
# 1-A. CSS：加入 badge 樣式
# ══════════════════════════════════════════════════════════════════════
old_css_anchor = '.p4-hist-empty{font-size:12px;color:rgba(255,255,255,.3);text-align:center;padding:10px 0}'
new_css_anchor = (old_css_anchor +
    '\n.p4-hist-badge{font-size:10px;padding:2px 5px;border-radius:3px;font-weight:600;flex-shrink:0}'
    '\n.p4-hist-badge.ok{background:rgba(76,175,80,.18);color:#66bb6a}'
    '\n.p4-hist-badge.short{background:rgba(255,255,255,.06);color:rgba(255,255,255,.3)}'
    '\n.p4-hist-na{font-size:10px;color:rgba(255,255,255,.25);flex-shrink:0}'
)
ok.append(('CSS badge', old_css_anchor in fmt))
fmt = fmt.replace(old_css_anchor, new_css_anchor, 1)

# ══════════════════════════════════════════════════════════════════════
# 1-B. HTML row：加資格標籤、太短的天不顯示排除按鈕
# ══════════════════════════════════════════════════════════════════════
old_row_html = (
    '        <div class="p4-hist-info">\n'
    '          <span class="p4-hist-hrs">{{d.hrs}}h</span>\n'
    '          <span ng-if="d.risk" class="p4-hist-risk {{d.risk.cls}}">{{d.risk.short}}</span>\n'
    '          <span ng-if="d.hasEvt" class="p4-hist-evt">\U0001f4cc</span>\n'
    '        </div>\n'
    '        <button class="p4-hist-excl-btn {{excludedDates.indexOf(d.date)>=0?\'active\':\'\'}}"'
    ' ng-click="toggleExclude(d.date)">\n'
    '          {{excludedDates.indexOf(d.date)>=0?\'↩ 取消排除\':\'✕ 排除此天\'}}\n'
    '        </button>'
)
new_row_html = (
    '        <div class="p4-hist-info">\n'
    '          <span class="p4-hist-hrs">{{d.hrs}}h</span>\n'
    # ✓ 計入：有效 且 未被排除
    '          <span ng-if="d.valid && excludedDates.indexOf(d.date)<0"'
    ' class="p4-hist-badge ok">✓ 計入</span>\n'
    # 太短：監控不足 1 小時
    '          <span ng-if="!d.valid" class="p4-hist-badge short">⚠ 太短</span>\n'
    '          <span ng-if="d.risk" class="p4-hist-risk {{d.risk.cls}}">{{d.risk.short}}</span>\n'
    '          <span ng-if="d.hasEvt" class="p4-hist-evt">\U0001f4cc</span>\n'
    '        </div>\n'
    # 有效天：顯示排除按鈕
    '        <button ng-if="d.valid" class="p4-hist-excl-btn {{excludedDates.indexOf(d.date)>=0?\'active\':\'\'}}"'
    ' ng-click="toggleExclude(d.date)">\n'
    '          {{excludedDates.indexOf(d.date)>=0?\'↩ 取消排除\':\'✕ 排除此天\'}}\n'
    '        </button>\n'
    # 太短：顯示「不計入」文字，無法點擊
    '        <span ng-if="!d.valid" class="p4-hist-na">不計入</span>'
)
ok.append(('HTML row', old_row_html in fmt))
fmt = fmt.replace(old_row_html, new_row_html, 1)

# ══════════════════════════════════════════════════════════════════════
# 1-C. $watch：historyDays map 加入 valid 欄位（monitoring_seconds >= 3600）
# ══════════════════════════════════════════════════════════════════════
old_map = (
    '        return {date:d.date,hrs:d.monitoring_hours||0,risk:rObj,hasEvt:evtDts.indexOf(d.date)>=0};'
)
new_map = (
    '        return {date:d.date,hrs:d.monitoring_hours||0,'
    'valid:(d.monitoring_seconds||0)>=3600,'
    'risk:rObj,hasEvt:evtDts.indexOf(d.date)>=0};'
)
ok.append(('$watch map valid', old_map in fmt))
fmt = fmt.replace(old_map, new_map, 1)

p4['format'] = fmt

# ══════════════════════════════════════════════════════════════════════
# 2. 每日歷史記錄寫入：同一天改為 upsert（避免重複）
# ══════════════════════════════════════════════════════════════════════
old_wr = (
    'let hist = global.get(\'v2_daily_history\', \'file\') || [];\n'
    'hist.push(rec);\n'
    'if (hist.length > 365) hist = hist.slice(-365);\n'
    'global.set(\'v2_daily_history\', hist, \'file\');'
)
new_wr = (
    'let hist = global.get(\'v2_daily_history\', \'file\') || [];\n'
    '// 同一天只保留最新一筆（避免重複影響基線計算）\n'
    'hist = hist.filter(h => h.date !== date);\n'
    'hist.push(rec);\n'
    'hist.sort((a,b)=>a.date<b.date?-1:a.date>b.date?1:0);\n'
    'if (hist.length > 365) hist = hist.slice(-365);\n'
    'global.set(\'v2_daily_history\', hist, \'file\');'
)
ok.append(('WR upsert', old_wr in wr['func']))
wr['func'] = wr['func'].replace(old_wr, new_wr, 1)

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

for lbl, found in ok:
    print(('OK  ' if found else 'MISS'), lbl)
print('nodes total:', len(data))
