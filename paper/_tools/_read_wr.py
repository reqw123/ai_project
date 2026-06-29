# -*- coding: utf-8 -*-
import json, sys
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
wr = nodes['c5ce881398ee6376']
sys.stdout.buffer.write(wr.get('func','').encode('utf-8'))
