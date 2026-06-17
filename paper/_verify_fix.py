# -*- coding: utf-8 -*-
import json
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
sm = nodes['ff00000000000010']
p4 = nodes['e3809f1cf09346ad']

opens  = sm['func'].count('{')
closes = sm['func'].count('}')
print(f"SM brace balance: {{ {opens} }} {closes} diff={opens-closes}  -> {'OK' if opens==closes else 'STILL BROKEN'}")
print("SM has stray }:", ('}\n}\n\nreturn' in sm['func']))
print("SM save_settings handler:", ('save_settings' in sm['func']))
print("P4 _settingsSaving init:", ('scope._settingsSaving = false' in p4['format']))
print("P4 saveSettings flag:", ('_settingsSaving = true' in p4['format']))
print("P4 $watch guard:", ('!scope._settingsSaving' in p4['format']))
