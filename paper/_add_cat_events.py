# -*- coding: utf-8 -*-
import json

with open('c:/ai_project/paper/cat_health_v2_flow.json', encoding='utf-8') as f:
    data = json.load(f)

nodes = {n['id']: n for n in data}
p4  = nodes['e3809f1cf09346ad']
sm  = nodes['ff00000000000010']
eng = nodes['4b0c232763de1461']
fmt = p4['format']

ok = []

# ══════════════════════════════════════════════════════════════════════
# 1.  P4 CSS
# ══════════════════════════════════════════════════════════════════════
old = '.p4-set-test:active{opacity:.85}'
new = old + '''
.p4-set-text{flex:1;padding:6px 10px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);border-radius:6px;color:#fff;font-size:12px;font-family:inherit;box-sizing:border-box}
.p4-set-text:focus{outline:none;border-color:#ffa726}
.p4-set-year{width:58px;padding:6px 6px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);border-radius:6px;color:#fff;font-size:12px;text-align:center}
.p4-set-month{width:42px;padding:6px 6px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);border-radius:6px;color:#fff;font-size:12px;text-align:center}
.p4-set-year:focus,.p4-set-month:focus{outline:none;border-color:#ffa726}
.p4-set-age{font-size:11px;color:#ffa726;margin:-4px 0 10px 122px;font-weight:600;line-height:1.4}
.p4-events{margin-top:12px;border-top:1px solid rgba(255,255,255,.08)}
.p4-evt-header{padding:12px 0;font-size:13px;font-weight:700;color:rgba(255,255,255,.6);cursor:pointer;user-select:none;display:flex;justify-content:space-between;align-items:center}
.p4-evt-header:hover{color:rgba(255,255,255,.9)}
.p4-evt-body{padding-bottom:12px}
.p4-evt-date-row{font-size:12px;color:rgba(255,255,255,.4);margin-bottom:10px}
.p4-evt-tags{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.p4-evt-tag{padding:6px 12px;border-radius:99px;font-size:12px;cursor:pointer;border:1px solid rgba(255,255,255,.15);color:rgba(255,255,255,.5);background:rgba(255,255,255,.04);user-select:none;transition:all .2s}
.p4-evt-tag.active{background:rgba(255,167,38,.2);border-color:#ffa726;color:#ffa726}
.p4-evt-save{width:100%;padding:9px;background:linear-gradient(135deg,#5c6bc0,#3949ab);border:none;border-radius:8px;color:#fff;font-size:13px;font-weight:700;cursor:pointer;font-family:inherit;margin-bottom:6px}
.p4-evt-save:active{opacity:.85}
.p4-evt-hist{margin-top:12px;border-top:1px solid rgba(255,255,255,.06);padding-top:10px}
.p4-evt-hist-title{font-size:10px;color:rgba(255,255,255,.3);letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px}
.p4-evt-hist-row{font-size:11px;color:rgba(255,255,255,.5);padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.p4-evt-hist-date{color:rgba(255,255,255,.35);margin-right:8px}
.p4-evt-hist-tags{color:#ffa726}
.p4-evt-hist-note{color:rgba(255,255,255,.35);font-style:italic}'''
ok.append(('CSS block', old in fmt))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 2.  P4 HTML – 貓咪資料 before baseline_days row
# ══════════════════════════════════════════════════════════════════════
old = '<div class="p4-set-row"><div class="p4-set-lbl">📅 基線天數</div>'
new = (
    '<div class="p4-set-divider">🐱 貓咪資料</div>\n'
    '      <div class="p4-set-row"><div class="p4-set-lbl">名字</div>'
    '<div class="p4-set-ctrl">'
    '<input type="text" class="p4-set-text" ng-model="settings.cat_name" placeholder="例：小花">'
    '</div></div>\n'
    '      <div class="p4-set-row"><div class="p4-set-lbl">品種</div>'
    '<div class="p4-set-ctrl">'
    '<input type="text" class="p4-set-text" ng-model="settings.cat_breed" placeholder="例：混種貓（選填）">'
    '</div></div>\n'
    '      <div class="p4-set-row"><div class="p4-set-lbl">出生年月</div>'
    '<div class="p4-set-ctrl" style="gap:6px">'
    '<input type="number" class="p4-set-year" ng-model="settings.cat_birth_year" min="2000" max="2030" placeholder="年">'
    '<span style="color:rgba(255,255,255,.4);font-size:12px">年</span>'
    '<input type="number" class="p4-set-month" ng-model="settings.cat_birth_month" min="1" max="12" placeholder="月">'
    '<span style="color:rgba(255,255,255,.4);font-size:12px">月</span>'
    '</div></div>\n'
    '      <div class="p4-set-age">{{catAgeStr()}}</div>\n'
    '      <div class="p4-set-divider">行為監控設定</div>\n'
    '      ' + old
)
ok.append(('HTML cat profile', old in fmt))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 3.  P4 HTML – 今日事件標記 section (before </div><script>)
# ══════════════════════════════════════════════════════════════════════
old = '\n</div>\n<script>'
new = (
    '\n'
    '  <div class="p4-events">\n'
    '    <div class="p4-evt-header" ng-click="evtOpen=!evtOpen">'
    '📌 今日事件標記 <span>{{evtOpen?\'▲\':\'▼\'}}</span></div>\n'
    '    <div ng-if="evtOpen" class="p4-evt-body">\n'
    '      <div class="p4-evt-date-row">📅 今天：{{todayDateStr()}}</div>\n'
    '      <div class="p4-evt-tags">\n'
    '        <span class="p4-evt-tag" ng-repeat="t in evtTagDefs"\n'
    '              ng-class="{active:selTags.indexOf(t.key)>=0}"\n'
    '              ng-click="toggleTag(t.key)">{{t.ico}} {{t.lbl}}</span>\n'
    '      </div>\n'
    '      <input class="p4-set-url-input" type="text" ng-model="evtNote"\n'
    '             placeholder="自訂備註（選填）" style="margin-bottom:10px">\n'
    '      <button class="p4-evt-save" ng-click="saveEvent()">📌 記錄今日事件</button>\n'
    '      <span class="p4-set-msg" ng-if="evtMsg">{{evtMsg}}</span>\n'
    '      <div ng-if="recentEvents.length>0" class="p4-evt-hist">\n'
    '        <div class="p4-evt-hist-title">近期事件記錄</div>\n'
    '        <div class="p4-evt-hist-row" ng-repeat="e in recentEvents">\n'
    '          <span class="p4-evt-hist-date">{{e.date}}</span>\n'
    '          <span class="p4-evt-hist-tags">{{e.tags.join(\' · \')}}</span>\n'
    '          <span ng-if="e.note" class="p4-evt-hist-note"> — {{e.note}}</span>\n'
    '        </div>\n'
    '      </div>\n'
    '    </div>\n'
    '  </div>\n'
    '\n</div>\n<script>'
)
ok.append(('HTML events section', fmt.count(old) == 1))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 4.  P4 JS – scope.settings 預設值加貓咪欄位
# ══════════════════════════════════════════════════════════════════════
old = "scope.settings = {baseline_days:7,lick_pct:20,scratch_pct:15,shake_pct:10,stop_pct:55,lsc_count:10,notify_level:2,discord_webhook:''};"
new = "scope.settings = {baseline_days:7,lick_pct:20,scratch_pct:15,shake_pct:10,stop_pct:55,lsc_count:10,notify_level:2,discord_webhook:'',cat_name:'',cat_breed:'',cat_birth_year:0,cat_birth_month:1};"
ok.append(('JS settings defaults', old in fmt))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 5.  P4 JS – saveSettings() 加貓咪欄位 + notify_level（之前遺漏）
# ══════════════════════════════════════════════════════════════════════
old = ("let s={baseline_days:parseInt(scope.settings.baseline_days)||7,"
       "lick_pct:parseInt(scope.settings.lick_pct)||20,"
       "scratch_pct:parseInt(scope.settings.scratch_pct)||15,"
       "shake_pct:parseInt(scope.settings.shake_pct)||10,"
       "stop_pct:parseInt(scope.settings.stop_pct)||55,"
       "lsc_count:parseInt(scope.settings.lsc_count)||10,"
       "discord_webhook:(scope.settings.discord_webhook||'').trim()};")
new = ("let s={baseline_days:parseInt(scope.settings.baseline_days)||7,"
       "lick_pct:parseInt(scope.settings.lick_pct)||20,"
       "scratch_pct:parseInt(scope.settings.scratch_pct)||15,"
       "shake_pct:parseInt(scope.settings.shake_pct)||10,"
       "stop_pct:parseInt(scope.settings.stop_pct)||55,"
       "lsc_count:parseInt(scope.settings.lsc_count)||10,"
       "notify_level:parseInt(scope.settings.notify_level)||2,"
       "discord_webhook:(scope.settings.discord_webhook||'').trim(),"
       "cat_name:(scope.settings.cat_name||'').trim(),"
       "cat_breed:(scope.settings.cat_breed||'').trim(),"
       "cat_birth_year:parseInt(scope.settings.cat_birth_year)||0,"
       "cat_birth_month:parseInt(scope.settings.cat_birth_month)||1};")
ok.append(('JS saveSettings fields', old in fmt))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 6.  P4 JS – 新增 catAgeStr / evtTagDefs / toggleTag / saveEvent
# ══════════════════════════════════════════════════════════════════════
old = '  scope.saveSettings = function(){'
new = '''  scope.catAgeStr = function(){
    let y=parseInt(scope.settings.cat_birth_year)||0,m=parseInt(scope.settings.cat_birth_month)||0;
    if(!y||!m)return '';
    let now=new Date(),mos=(now.getFullYear()-y)*12+(now.getMonth()+1-m);
    if(mos<0)return '⚠️ 出生年月超過今天';
    let yrs=Math.floor(mos/12),rem=mos%12;
    let s2=yrs>0?yrs+'歲':''; if(rem>0)s2+=(s2?' ':'')+rem+'個月'; if(!s2)s2='未滿1個月';
    let g=mos<24?'幼貓':mos<120?'成貓':'老貓';
    return '🐱 年齡：'+s2+'（'+g+'）';
  };
  scope.evtTagDefs = [
    {key:'vet',      ico:'🏥', lbl:'帶去看醫生'},
    {key:'food',     ico:'🍽', lbl:'換了貓糧'},
    {key:'stranger', ico:'👤', lbl:'陌生人來訪'},
    {key:'stress',   ico:'😿', lbl:'情緒緊張'},
    {key:'medicine', ico:'💊', lbl:'服藥/治療'}
  ];
  scope.evtOpen = false; scope.selTags = []; scope.evtNote = ''; scope.evtMsg = '';
  scope.recentEvents = [];
  scope.toggleTag = function(key){
    let i=scope.selTags.indexOf(key);
    if(i>=0) scope.selTags.splice(i,1); else scope.selTags.push(key);
  };
  scope.todayDateStr = function(){ return new Date().toLocaleDateString('zh-TW'); };
  scope.saveEvent = function(){
    if(scope.selTags.length===0&&!scope.evtNote.trim()){
      scope.evtMsg='⚠️ 請選擇標記或填寫備註';
      setTimeout(function(){ scope.$apply(function(){ scope.evtMsg=''; }); },3000); return;
    }
    let lbl=scope.selTags.map(function(k){ let t=scope.evtTagDefs.find(function(d){return d.key===k;}); return t?t.lbl:k; });
    scope.send({payload:{action:'save_event_tag',date:new Date().toLocaleDateString('zh-TW'),tags:lbl,note:scope.evtNote.trim()}});
    scope.selTags=[]; scope.evtNote='';
    scope.evtMsg='📌 事件已記錄';
    setTimeout(function(){ scope.$apply(function(){ scope.evtMsg=''; }); },3000);
  };
  scope.saveSettings = function(){'''
ok.append(('JS new functions', old in fmt))
fmt = fmt.replace(old, new, 1)

# ══════════════════════════════════════════════════════════════════════
# 7.  P4 JS – $watch 加 recentEvents
# ══════════════════════════════════════════════════════════════════════
old = '    if(p.user_settings && !scope.settingsOpen) Object.assign(scope.settings,p.user_settings);\n  });'
new = ('    if(p.user_settings && !scope.settingsOpen) Object.assign(scope.settings,p.user_settings);\n'
       '    if(p.v2_event_tags) scope.recentEvents=(p.v2_event_tags||[]).slice().reverse().slice(0,7);\n'
       '  });')
ok.append(('JS $watch events', old in fmt))
fmt = fmt.replace(old, new, 1)

p4['format'] = fmt

# ══════════════════════════════════════════════════════════════════════
# 8.  使用者設定管理器 – 加 cat 欄位 + save_event_tag handler
# ══════════════════════════════════════════════════════════════════════
old_cat = "        discord_webhook: (s.discord_webhook || '').trim(),\n        notify_level:    Math.min(3, Math.max(1, parseInt(s.notify_level) || 2))\n    };"
new_cat = ("        discord_webhook: (s.discord_webhook || '').trim(),\n"
           "        notify_level:    Math.min(3, Math.max(1, parseInt(s.notify_level) || 2)),\n"
           "        cat_name:        (s.cat_name  || '').trim().slice(0,20),\n"
           "        cat_breed:       (s.cat_breed || '').trim().slice(0,30),\n"
           "        cat_birth_year:  parseInt(s.cat_birth_year)  || 0,\n"
           "        cat_birth_month: Math.min(12, Math.max(1, parseInt(s.cat_birth_month) || 1))\n"
           "    };")
ok.append(('SM cat fields', old_cat in sm['func']))
sm['func'] = sm['func'].replace(old_cat, new_cat, 1)

old_evt = 'return [null, null];\n}'
new_evt = (
    'return [null, null];\n'
    '}\n\n'
    '// ── 事件標記儲存 ──────────────────────────────────────────────────\n'
    "if (action === 'save_event_tag') {\n"
    "    let tags = msg.payload.tags || [];\n"
    "    let note = (msg.payload.note || '').trim();\n"
    "    let date = msg.payload.date || new Date().toLocaleDateString('zh-TW');\n"
    "    if (tags.length === 0 && !note) return [null, null];\n"
    "    let events = global.get('v2_event_tags', 'file') || [];\n"
    "    events = events.filter(function(e){ return e.date !== date; });\n"
    "    events.push({date: date, tags: tags, note: note});\n"
    "    if (events.length > 90) events = events.slice(-90);\n"
    "    global.set('v2_event_tags', events, 'file');\n"
    "    node.warn('事件標記已儲存：' + date + ' [' + tags.join(',') + ']');\n"
    "    return [null, null];\n"
    "}\n}"
)
ok.append(('SM event handler', old_evt in sm['func']))
sm['func'] = sm['func'].replace(old_evt, new_evt, 1)

# ══════════════════════════════════════════════════════════════════════
# 9.  健康預警引擎 – 傳遞 v2_event_tags 給前端（唯讀，不影響任何邏輯）
# ══════════════════════════════════════════════════════════════════════
old_tag_pass = "msg.payload.user_settings = cfg;"
new_tag_pass = ("msg.payload.user_settings  = cfg;\n"
                "msg.payload.v2_event_tags  = (global.get('v2_event_tags','file')||[]).slice(-14);")
ok.append(('Eng pass event tags', old_tag_pass in eng['func']))
eng['func'] = eng['func'].replace(old_tag_pass, new_tag_pass, 1)

# ══════════════════════════════════════════════════════════════════════
# Save & report
# ══════════════════════════════════════════════════════════════════════
with open('c:/ai_project/paper/cat_health_v2_flow.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('Results:')
for label, found in ok:
    print(f'  {"✅" if found else "❌"} {label}')
print(f'\nTotal nodes: {len(data)}')
