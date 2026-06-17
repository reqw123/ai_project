# -*- coding: utf-8 -*-
"""
修正兩個 Bug 的根本原因：
  Bug 1（排除按鈕/刪除基線無效）+ Bug 2（設定儲存後還原）
  → 共同根源：使用者設定管理器 func 末尾有一個多餘的 }，
    造成 JavaScript 語法錯誤，整個 SM 函式無法執行，
    save_settings / toggle_exclude_date 全部失效。

修正項目：
  1. 移除 SM 末尾多餘的 }
  2. P4 saveSettings() 加入 _settingsSaving 旗標（3秒），
     防止儲存後伺服器舊訊息把設定值蓋回去
"""
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
sm = nodes['ff00000000000010']
p4 = nodes['e3809f1cf09346ad']

ok = []

# ══════════════════════════════════════════════════════════════════════
# 1. 移除 SM 末尾多餘的 }
#    舊：  return [msg, null];  // output 1 → 觸發基線重算\n}\n}\n\nreturn [null, null];
#    新：  return [msg, null];  // output 1 → 觸發基線重算\n}\n\nreturn [null, null];
# ══════════════════════════════════════════════════════════════════════
old_sm_tail = (
    "    return [msg, null];  // output 1 → 觸發基線重算\n"
    "}\n"
    "}\n"
    "\n"
    "return [null, null];"
)
new_sm_tail = (
    "    return [msg, null];  // output 1 → 觸發基線重算\n"
    "}\n"
    "\n"
    "return [null, null];"
)
ok.append(('SM stray brace', old_sm_tail in sm['func']))
sm['func'] = sm['func'].replace(old_sm_tail, new_sm_tail, 1)

# 驗證修正後的大括號平衡
opens  = sm['func'].count('{')
closes = sm['func'].count('}')
ok.append(('SM brace balanced', opens == closes))

# ══════════════════════════════════════════════════════════════════════
# 2. P4：saveSettings() 加入 _settingsSaving 旗標
#    防止 SM 處理完成前，引擎舊訊息把 scope.settings 蓋回去
# ══════════════════════════════════════════════════════════════════════
fmt = p4['format']

# 2-A. 初始化旗標（加在 scope.settingsOpen 那行旁邊）
old_init = "  scope.settingsOpen = false;\n  scope.saveMsg = '';"
new_init = "  scope.settingsOpen = false;\n  scope._settingsSaving = false;\n  scope.saveMsg = '';"
ok.append(('P4 _settingsSaving init', old_init in fmt))
fmt = fmt.replace(old_init, new_init, 1)

# 2-B. saveSettings() 儲存時設旗標 3 秒
old_save_fn = (
    "    scope.saveMsg='✅ 設定已儲存，閾值秒速生效';\n"
    "    scope.settingsOpen = false;\n"
    "    setTimeout(function(){ scope.$apply(function(){ scope.saveMsg=''; }); },3000);\n"
    "  };"
)
new_save_fn = (
    "    scope.saveMsg='✅ 設定已儲存，閾值秒速生效';\n"
    "    scope.settingsOpen = false;\n"
    "    scope._settingsSaving = true;  // 3 秒內不讓舊訊息把設定值蓋回去\n"
    "    setTimeout(function(){ scope.$apply(function(){ scope._settingsSaving = false; }); }, 3000);\n"
    "    setTimeout(function(){ scope.$apply(function(){ scope.saveMsg=''; }); },3000);\n"
    "  };"
)
ok.append(('P4 saveSettings flag', old_save_fn in fmt))
fmt = fmt.replace(old_save_fn, new_save_fn, 1)

# 2-C. $watch 中加入 _settingsSaving 防護
old_watch_cfg = "    if(p.user_settings && !scope.settingsOpen) Object.assign(scope.settings,p.user_settings);"
new_watch_cfg = "    if(p.user_settings && !scope.settingsOpen && !scope._settingsSaving) Object.assign(scope.settings,p.user_settings);"
ok.append(('P4 $watch settings guard', old_watch_cfg in fmt))
fmt = fmt.replace(old_watch_cfg, new_watch_cfg, 1)

p4['format'] = fmt

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

for lbl, found in ok:
    print(('OK  ' if found else 'MISS'), lbl)
