# -*- coding: utf-8 -*-
"""
功能：排除特定天出基線
  1. 健康預警引擎     → 儲存今日風險到 flow context + 傳遞 v2_daily_history / v2_excluded_dates
  2. 每日歷史記錄寫入 → 紀錄結束當天附上 risk 欄位
  3. 個體化基線計算器 → validDays 再加一層排除過濾
  4. 使用者設定管理器 → toggle_exclude_date handler → 觸發基線重算（output 1）
  5. P4 UI          → 歷史資料管理區塊
"""
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data}
p4  = nodes['e3809f1cf09346ad']  # P4 面板
sm  = nodes['ff00000000000010']  # 使用者設定管理器
eng = nodes['4b0c232763de1461']  # 健康預警引擎
wr  = nodes['c5ce881398ee6376']  # 每日歷史記錄寫入
bl  = nodes['66a22e0ec663d07d']  # 個體化基線計算器

ok = []

# ══════════════════════════════════════════════════════════════════════
# 1. 健康預警引擎：儲存今日風險 + 傳遞歷史資料到前端
# ══════════════════════════════════════════════════════════════════════
old = ("msg.payload.user_settings  = cfg;\n"
       "msg.payload.v2_event_tags  = (global.get('v2_event_tags','file')||[]).slice(-14);")
new = ("msg.payload.user_settings   = cfg;\n"
       "msg.payload.v2_event_tags   = (global.get('v2_event_tags','file')||[]).slice(-14);\n"
       "msg.payload.v2_daily_history  = (global.get('v2_daily_history','file')||[]).slice(-30);\n"
       "msg.payload.v2_excluded_dates = global.get('v2_excluded_dates','file') || [];\n"
       "flow.set('v2_today_risk', {level: level, score: riskScore});")
ok.append(('Eng: payload + risk store', old in eng['func']))
eng['func'] = eng['func'].replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 2. 每日歷史記錄寫入：rec 加 risk 欄位
# ══════════════════════════════════════════════════════════════════════
old = '    periods\n};'
new = '    periods,\n    risk: flow.get(\'v2_today_risk\') || null\n};'
ok.append(('Wr: rec.risk field', old in wr['func']))
wr['func'] = wr['func'].replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 3. 個體化基線計算器：排除指定日期
# ══════════════════════════════════════════════════════════════════════
old_bl = 'let days = validDays.slice(-30);'
cnt = bl['func'].count(old_bl)
ok.append(('Bl: anchor unique', cnt == 1))
new_bl = (
    "// ── 排除飼主手動標記的日期 ─────────────────────────────────────\n"
    "let v2Excluded = global.get('v2_excluded_dates', 'file') || [];\n"
    "if (v2Excluded.length > 0) {\n"
    "    let noExcl = validDays.filter(d => !v2Excluded.includes(d.date));\n"
    "    if (noExcl.length >= MIN_BASELINE_DAYS) {\n"
    "        validDays = noExcl;\n"
    "        node.warn('排除 ' + v2Excluded.length + ' 天後基線有效天數：' + validDays.length);\n"
    "    } else {\n"
    "        node.warn('排除後有效天不足（' + noExcl.length + ' 天），維持原有天數');\n"
    "    }\n"
    "}\n"
    "let days = validDays.slice(-30);"
)
bl['func'] = bl['func'].replace(old_bl, new_bl, 1)

# ══════════════════════════════════════════════════════════════════════
# 4. 使用者設定管理器：toggle_exclude_date handler
# ══════════════════════════════════════════════════════════════════════
old_sm_end = ("    node.warn('事件標記已儲存：' + date + ' [' + tags.join(',') + ']');\n"
              "    return [null, null];\n"
              "}\n"
              "}")
new_sm_end = (
    "    node.warn('事件標記已儲存：' + date + ' [' + tags.join(',') + ']');\n"
    "    return [null, null];\n"
    "}\n"
    "\n"
    "// ── 排除日期切換（立即觸發基線重算）────────────────────────────────\n"
    "if (action === 'toggle_exclude_date') {\n"
    "    let exDate = (msg.payload.date || '').trim();\n"
    "    if (!exDate) return [null, null];\n"
    "    let excluded = global.get('v2_excluded_dates', 'file') || [];\n"
    "    let ei = excluded.indexOf(exDate);\n"
    "    if (ei >= 0) excluded.splice(ei, 1);  // 已排除 → 取消排除\n"
    "    else excluded.push(exDate);             // 未排除 → 加入排除\n"
    "    global.set('v2_excluded_dates', excluded, 'file');\n"
    "    node.warn('排除日期更新：' + JSON.stringify(excluded));\n"
    "    msg.payload = {};\n"
    "    return [msg, null];  // output 1 → 觸發基線重算\n"
    "}\n"
    "}"
)
ok.append(('SM: toggle_exclude handler', old_sm_end in sm['func']))
sm['func'] = sm['func'].replace(old_sm_end, new_sm_end, 1)

# ══════════════════════════════════════════════════════════════════════
# 5-A. P4 CSS：歷史資料管理樣式
# ══════════════════════════════════════════════════════════════════════
fmt = p4['format']
old_css = '.p4-evt-hist-note{color:rgba(255,255,255,.35);font-style:italic}'
new_css = old_css + '''
.p4-history{margin-top:12px;border-top:1px solid rgba(255,255,255,.08)}
.p4-hist-header{padding:12px 0;font-size:13px;font-weight:700;color:rgba(255,255,255,.6);cursor:pointer;user-select:none;display:flex;justify-content:space-between;align-items:center}
.p4-hist-header:hover{color:rgba(255,255,255,.9)}
.p4-hist-hint{font-size:11px;color:rgba(255,255,255,.35);margin-bottom:10px;line-height:1.5}
.p4-hist-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.p4-hist-date{font-size:11px;color:rgba(255,255,255,.6);min-width:76px;flex-shrink:0}
.p4-hist-info{flex:1;display:flex;gap:6px;align-items:center;min-width:0}
.p4-hist-hrs{font-size:10px;color:rgba(255,255,255,.35)}
.p4-hist-risk{font-size:10px;padding:2px 6px;border-radius:4px;font-weight:700}
.p4-hist-risk.hi{background:rgba(244,67,54,.2);color:#ef5350}
.p4-hist-risk.warn{background:rgba(255,112,67,.2);color:#ff7043}
.p4-hist-risk.attn{background:rgba(255,167,38,.2);color:#ffa726}
.p4-hist-risk.ok{background:rgba(76,175,80,.2);color:#66bb6a}
.p4-hist-evt{font-size:11px;opacity:.7}
.p4-hist-excl-btn{padding:4px 10px;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer;border:1px solid rgba(255,255,255,.2);color:rgba(255,255,255,.5);background:rgba(255,255,255,.05);font-family:inherit;white-space:nowrap;transition:all .2s;flex-shrink:0}
.p4-hist-excl-btn.active{background:rgba(239,83,80,.15);border-color:#ef5350;color:#ef5350}
.p4-hist-empty{font-size:12px;color:rgba(255,255,255,.3);text-align:center;padding:10px 0}'''
ok.append(('CSS history styles', old_css in fmt))
fmt = fmt.replace(old_css, new_css, 1)

# ══════════════════════════════════════════════════════════════════════
# 5-B. P4 HTML：歷史資料管理區塊（插入 events 後、</div><script> 前）
# ══════════════════════════════════════════════════════════════════════
old_html_end = '\n</div>\n<script>'
new_html_end = (
    '\n'
    '  <div class="p4-history">\n'
    '    <div class="p4-hist-header" ng-click="histOpen=!histOpen">'
    '📋 歷史資料管理 <span>{{histOpen?\'▲\':\'▼\'}}</span></div>\n'
    '    <div ng-if="histOpen">\n'
    '      <div class="p4-hist-hint">標記異常天（帶出門、看醫生）→ 排除後立即重算個體基線</div>\n'
    '      <div class="p4-hist-row" ng-repeat="d in historyDays">\n'
    '        <div class="p4-hist-date">{{d.date}}</div>\n'
    '        <div class="p4-hist-info">\n'
    '          <span class="p4-hist-hrs">{{d.hrs}}h</span>\n'
    '          <span ng-if="d.risk" class="p4-hist-risk {{d.risk.cls}}">{{d.risk.short}}</span>\n'
    '          <span ng-if="d.hasEvt" class="p4-hist-evt">📌</span>\n'
    '        </div>\n'
    '        <button class="p4-hist-excl-btn {{excludedDates.indexOf(d.date)>=0?\'active\':\'\'}}"'
    ' ng-click="toggleExclude(d.date)">\n'
    '          {{excludedDates.indexOf(d.date)>=0?\'✕ 已排除\':\'納入基線\'}}\n'
    '        </button>\n'
    '      </div>\n'
    '      <div ng-if="historyDays.length===0" class="p4-hist-empty">尚無歷史資料</div>\n'
    '    </div>\n'
    '  </div>\n'
    '\n</div>\n<script>'
)
cnt2 = fmt.count(old_html_end)
ok.append(('HTML history anchor unique', cnt2 == 1))
fmt = fmt.replace(old_html_end, new_html_end, 1)

# ══════════════════════════════════════════════════════════════════════
# 5-C. P4 JS：histOpen / historyDays / excludedDates / toggleExclude
#             （加在 evtOpen 那一行附近）
# ══════════════════════════════════════════════════════════════════════
old_js = '  scope.evtOpen = false; scope.selTags = []; scope.evtNote = \'\'; scope.evtMsg = \'\';\n  scope.recentEvents = [];'
new_js = (
    '  scope.evtOpen = false; scope.selTags = []; scope.evtNote = \'\'; scope.evtMsg = \'\';\n'
    '  scope.recentEvents = [];\n'
    '  scope.histOpen = false;\n'
    '  scope.historyDays = [];\n'
    '  scope.excludedDates = [];\n'
    '  scope.toggleExclude = function(date){\n'
    '    scope.send({payload:{action:\'toggle_exclude_date\',date:date}});\n'
    '    let i=scope.excludedDates.indexOf(date);\n'
    '    if(i>=0) scope.excludedDates.splice(i,1); else scope.excludedDates.push(date);\n'
    '  };'
)
ok.append(('JS histOpen / toggleExclude', old_js in fmt))
fmt = fmt.replace(old_js, new_js, 1)

# ══════════════════════════════════════════════════════════════════════
# 5-D. P4 JS $watch：更新 historyDays + excludedDates
# ══════════════════════════════════════════════════════════════════════
old_watch = ('    if(p.v2_event_tags) scope.recentEvents=(p.v2_event_tags||[]).slice().reverse().slice(0,7);\n'
             '  });')
new_watch = (
    '    if(p.v2_event_tags) scope.recentEvents=(p.v2_event_tags||[]).slice().reverse().slice(0,7);\n'
    '    if(p.v2_daily_history){\n'
    '      let evtDts=(p.v2_event_tags||[]).map(function(e){return e.date;});\n'
    '      scope.historyDays=(p.v2_daily_history||[]).slice().reverse().slice(0,14).map(function(d){\n'
    '        let r=d.risk||null;\n'
    '        let rObj=r?{cls:r.level===\'High Risk\'?\'hi\':r.level===\'Warning\'?\'warn\':r.level===\'Attention\'?\'attn\':\'ok\',\n'
    '                    short:r.level===\'High Risk\'?\'高風\':r.level===\'Warning\'?\'警告\':r.level===\'Attention\'?\'留意\':\'正常\'}:null;\n'
    '        return {date:d.date,hrs:d.monitoring_hours||0,risk:rObj,hasEvt:evtDts.indexOf(d.date)>=0};\n'
    '      });\n'
    '    }\n'
    '    if(p.v2_excluded_dates) scope.excludedDates=(p.v2_excluded_dates||[]).slice();\n'
    '  });'
)
ok.append(('JS $watch history', old_watch in fmt))
fmt = fmt.replace(old_watch, new_watch, 1)

p4['format'] = fmt

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

for lbl, found in ok:
    print(('OK  ' if found else 'MISS'), lbl)
print('nodes total:', len(data))
