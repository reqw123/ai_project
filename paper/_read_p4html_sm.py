# -*- coding: utf-8 -*-
import json, sys

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}

# P4 HTML (script 이전 부분만)
p4 = nodes['e3809f1cf09346ad']
fmt = p4['format']
script_start = fmt.find('<script>')
sys.stdout.buffer.write(b'=== P4 HTML ===\n')
sys.stdout.buffer.write(fmt[:script_start].encode('utf-8'))

# SM 마지막 300자
sm = nodes['ff00000000000010']
func = sm.get('func','')
sys.stdout.buffer.write(b'\n\n=== SM LAST 500 chars ===\n')
sys.stdout.buffer.write(func[-500:].encode('utf-8'))

# SM 전체 줄 수 및 브레이스 균형 확인
opens = func.count('{')
closes = func.count('}')
sys.stdout.buffer.write(f'\n\n=== SM brace balance: {{ {opens} }} {closes} (diff={opens-closes}) ===\n'.encode('utf-8'))
