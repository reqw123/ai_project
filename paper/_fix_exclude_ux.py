# -*- coding: utf-8 -*-
"""
修正「排除特定天出基線」UX 問題：
  1. 按鈕文字改為動作語意（按下去會做什麼），不再是狀態語意
  2. 加入 _excludePending 旗標，防止 Python 端週期訊息在按下後立即覆蓋 UI 狀態
  3. 誤觸保護：排除後 3 秒內收到的舊訊息不會覆蓋，超過後才同步
"""
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data}
p4 = nodes['e3809f1cf09346ad']  # P4 面板
fmt = p4['format']

ok = []

# ══════════════════════════════════════════════════════════════════════
# 1. 按鈕標籤：改為動作語意
#    舊：'✕ 已排除' / '納入基線'  ← 描述「現在的狀態」，容易誤解
#    新：'↩ 取消排除' / '✕ 排除此天'  ← 描述「按下去會做什麼」
# ══════════════════════════════════════════════════════════════════════
old_btn_label = "{{excludedDates.indexOf(d.date)>=0?'✕ 已排除':'納入基線'}}"
new_btn_label = "{{excludedDates.indexOf(d.date)>=0?'↩ 取消排除':'✕ 排除此天'}}"
ok.append(('btn label anchor', old_btn_label in fmt))
fmt = fmt.replace(old_btn_label, new_btn_label, 1)

# ══════════════════════════════════════════════════════════════════════
# 2. _excludePending 旗標 + toggleExclude 更新
#    舊 toggleExclude：只做樂觀更新，但沒有防止 $watch 覆蓋
#    新：設旗標 3 秒，期間 $watch 不覆寫 excludedDates
# ══════════════════════════════════════════════════════════════════════
old_toggle = (
    "  scope.histOpen = false;\n"
    "  scope.historyDays = [];\n"
    "  scope.excludedDates = [];\n"
    "  scope.toggleExclude = function(date){\n"
    "    scope.send({payload:{action:'toggle_exclude_date',date:date}});\n"
    "    let i=scope.excludedDates.indexOf(date);\n"
    "    if(i>=0) scope.excludedDates.splice(i,1); else scope.excludedDates.push(date);\n"
    "  };"
)
new_toggle = (
    "  scope.histOpen = false;\n"
    "  scope.historyDays = [];\n"
    "  scope.excludedDates = [];\n"
    "  scope._excludePending = false;  // 防止 $watch 在 toggle 後立即覆蓋 UI 狀態\n"
    "  scope.toggleExclude = function(date){\n"
    "    // 先樂觀更新 UI\n"
    "    let i=scope.excludedDates.indexOf(date);\n"
    "    if(i>=0) scope.excludedDates.splice(i,1); else scope.excludedDates.push(date);\n"
    "    // 設旗標：3 秒內 $watch 不覆寫（給 file store 寫入 + 引擎回送時間）\n"
    "    scope._excludePending = true;\n"
    "    setTimeout(function(){ scope.$apply(function(){ scope._excludePending = false; }); }, 3000);\n"
    "    scope.send({payload:{action:'toggle_exclude_date',date:date}});\n"
    "  };"
)
ok.append(('toggleExclude anchor', old_toggle in fmt))
fmt = fmt.replace(old_toggle, new_toggle, 1)

# ══════════════════════════════════════════════════════════════════════
# 3. $watch 中更新 excludedDates 時加入 pending 防護
#    舊：if(p.v2_excluded_dates) scope.excludedDates=(p.v2_excluded_dates||[]).slice();
#    新：只有在沒有 pending 時才覆寫
# ══════════════════════════════════════════════════════════════════════
old_watch_excl = "    if(p.v2_excluded_dates) scope.excludedDates=(p.v2_excluded_dates||[]).slice();\n"
new_watch_excl = (
    "    // 若剛按了 toggle，3 秒內不以舊訊息覆蓋本地狀態\n"
    "    if(p.v2_excluded_dates && !scope._excludePending)\n"
    "      scope.excludedDates=(p.v2_excluded_dates||[]).slice();\n"
)
ok.append(('$watch excludedDates anchor', old_watch_excl in fmt))
fmt = fmt.replace(old_watch_excl, new_watch_excl, 1)

p4['format'] = fmt

with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

for lbl, found in ok:
    print(('OK  ' if found else 'MISS'), lbl)
print('done, nodes total:', len(data))
