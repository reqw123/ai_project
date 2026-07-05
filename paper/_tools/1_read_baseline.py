# -*- coding: utf-8 -*-
import json, sys
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
bl = nodes['66a22e0ec663d07d']  # 個體化基線計算器
sys.stdout.buffer.write(bl.get('func','').encode('utf-8'))
