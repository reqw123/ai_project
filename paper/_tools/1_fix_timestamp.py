# -*- coding: utf-8 -*-
import json

PATH = r'c:\ai_project\paper\貓咪主控.json'

OLD_TS = 'timestamp:         new Date().toISOString(),'
NEW_TS = (
    'timestamp:         (function(){'
    'var d=new Date(),p=function(n){return String(n).padStart(2,"0")};'
    'return d.getFullYear()+"-"+p(d.getMonth()+1)+"-"+p(d.getDate())+" "'
    '+p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds());'
    '})(),'
)

with open(PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') == '89b182d351391d5f':
        old = n['func']
        new = old.replace(OLD_TS, NEW_TS)
        if new != old:
            n['func'] = new
            print('Fixed: timestamp UTC -> local (YYYY-MM-DD HH:MM:SS)')
        else:
            print('ERROR: pattern not matched, check manually')
        break

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
