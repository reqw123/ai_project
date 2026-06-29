# -*- coding: utf-8 -*-
import json, sys
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
eng = nodes['4b0c232763de1461']  # 健康預警引擎
sys.stdout.buffer.write(eng.get('func','').encode('utf-8'))
