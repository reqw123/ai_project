# -*- coding: utf-8 -*-
import json
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

# Print all function nodes with their IDs and first 80 chars of func
for n in data:
    if n.get('type') == 'function':
        func = n.get('func','')
        print(f"ID:{n['id']}  name:{n.get('name','?')[:40]}")
        print(f"  func[0:200]: {func[:200]}")
        print()
