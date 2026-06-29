# -*- coding: utf-8 -*-
import json, sys

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}

p4 = nodes['e3809f1cf09346ad']
fmt = p4['format']
script_start = fmt.find('<script>')
sys.stdout.buffer.write(fmt[script_start:].encode('utf-8'))

sys.stdout.buffer.write(b'\n\n### SM ###\n')
sm = nodes['ff00000000000010']
sys.stdout.buffer.write(sm.get('func','').encode('utf-8'))
