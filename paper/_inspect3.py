# -*- coding: utf-8 -*-
import json, sys
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}

# Health alert engine + baseline writer
for tid in ['4b0c232763de1461', 'c5ce881398ee6376', '66a22e0ec663d07d']:
    n = nodes.get(tid, {})
    sys.stdout.buffer.write(('=== ' + tid + ' ===\n').encode('utf-8'))
    sys.stdout.buffer.write(n.get('func','').encode('utf-8'))
    sys.stdout.buffer.write(b'\n\n')
