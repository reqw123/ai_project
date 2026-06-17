# -*- coding: utf-8 -*-
import json
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}

# Key nodes
targets = ['66a22e0ec663d07d', '341f4eef4167c6ec', 'c5ce881398ee6376', '7b161e3492a5cd6a']
for tid in targets:
    n = nodes.get(tid, {})
    print(f"=== ID:{tid} ===")
    print(f"name: {n.get('name','?')}")
    print(n.get('func','')[:2000])
    print()
