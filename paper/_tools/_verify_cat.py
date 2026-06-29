# -*- coding: utf-8 -*-
import json
with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)
nodes = {n['id']: n for n in data}
p4  = nodes['e3809f1cf09346ad']
sm  = nodes['ff00000000000010']
eng = nodes['4b0c232763de1461']

checks = [
  ('CSS .p4-set-text',        '.p4-set-text{flex:1' in p4['format']),
  ('CSS .p4-events',          '.p4-events{margin-top' in p4['format']),
  ('HTML cat divider',        'p4-set-divider' in p4['format'] and 'cat_name' in p4['format']),
  ('HTML cat_birth_year',     'cat_birth_year' in p4['format']),
  ('HTML catAgeStr',          'catAgeStr' in p4['format']),
  ('HTML events section',     'p4-events' in p4['format']),
  ('HTML evtTagDefs ng-rep',  'ng-repeat="t in evtTagDefs"' in p4['format']),
  ('JS catAgeStr fn',         'scope.catAgeStr' in p4['format']),
  ('JS evtTagDefs',           'scope.evtTagDefs' in p4['format']),
  ('JS saveEvent fn',         'scope.saveEvent' in p4['format']),
  ('JS watch v2_event_tags',  'v2_event_tags' in p4['format']),
  ('JS notify_level in s',    'notify_level:parseInt' in p4['format']),
  ('JS cat_name in s',        'cat_name:' in p4['format']),
  ('SM cat_name field',       'cat_name:' in sm['func']),
  ('SM save_event_tag',       'save_event_tag' in sm['func']),
  ('Eng pass events',         'v2_event_tags' in eng['func']),
]
for lbl, found in checks:
    print(('OK  ' if found else 'MISS'), lbl)
print('nodes total:', len(data))
