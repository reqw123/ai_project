# -*- coding: utf-8 -*-
"""
把 v2 四維風險算法整合進主控 /yolo_result，同時移除 v2 的 P5 面板。

變更清單：
1. 貓咪主控.json  - 健康引擎分析 (257d6083ac06afed)
   ‣ 直接從 today_stats 自算 distribution / rhythm / deviation，不再依賴 v2 中間節點
   ‣ 同步將資料寫入 v2 global context，讓 v2 基線引擎（定時任務）繼續正常運作
2. 貓咪主控.json  - 健康警示 (d349c9a9a2509e65)
   ‣ 解除節律/轉移卡的灰化（opacity:.32 → 動態綁定）
   ‣ 補全色彩 CSS
3. cat_health_v2_flow.json  - 移除 P5 健康預警面板及相關節點
   ‣ b2a8610d74979717 (P5 面板)
   ‣ 88627ebf3b74045d (← P5入口 link_in)
   ‣ 4d01766fcc276f06 (→ P5健康預警 link_out)
   ‣ 更新健康預警引擎 wires，移除指向 P5 的輸出
"""
import json, re

# ══════════════════════════════════════════════════════════════════
# 1 & 2. 貓咪主控.json
# ══════════════════════════════════════════════════════════════════
MAIN_PATH     = r'c:\ai_project\paper\貓咪主控.json'
ENGINE_ID     = '257d6083ac06afed'
ALERT_UI_ID   = 'd349c9a9a2509e65'

# ── 1. 健康引擎分析：用自計算版取代舊的讀取語句 ──
OLD_HEADER = (
    "let data  = msg.payload;\n"
    "let stats = (data && data.today_stats) ? data.today_stats : {};\n"
    "let dev   = data.deviation;\n"
    "let dist  = data.distribution || flow.get('v2_distribution') || {};\n"
    "let rhythm= data.rhythm       || flow.get('v2_rhythm')       || {};\n"
    "let mx    = data.v2_matrix    || global.get('v2_transition_matrix') || {};\n"
    "let td    = data.v2_today     || {};"
)

NEW_HEADER = (
    "let data  = msg.payload;\n"
    "let stats = (data && data.today_stats) ? data.today_stats : {};\n"
    "\n"
    "// ── 自行計算 distribution ──\n"
    "let walkT=stats.walk_time||0,lickT=stats.lick_time||0,scratchT=stats.scratch_time||0,shakeT=stats.shake_time||0,stopT=stats.stop_time||0;\n"
    "let totalT=walkT+lickT+scratchT+shakeT+stopT;\n"
    "let dist=totalT>0?{walk:parseFloat((walkT/totalT*100).toFixed(2)),lick:parseFloat((lickT/totalT*100).toFixed(2)),scratch:parseFloat((scratchT/totalT*100).toFixed(2)),shake:parseFloat((shakeT/totalT*100).toFixed(2)),stop:parseFloat((stopT/totalT*100).toFixed(2))}:{walk:0,lick:0,scratch:0,shake:0,stop:0};\n"
    "\n"
    "// ── 自行計算 rhythm（from hourly_distribution）──\n"
    "let hourly=stats.hourly_distribution||{};\n"
    "let _pds={'00-06':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},'06-12':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},'12-18':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0},'18-24':{walk:0,lick:0,scratch:0,shake:0,stop:0,total:0}};\n"
    "for(let h=0;h<24;h++){let k=String(h).padStart(2,'0'),hd=hourly[k]||{},pd=h<6?'00-06':h<12?'06-12':h<18?'12-18':'18-24';for(let b of['walk','lick','scratch','shake','stop']){_pds[pd][b]+=(hd[b]||0);_pds[pd].total+=(hd[b]||0);}}\n"
    "let rhythm={};\n"
    "for(let[pd,d]of Object.entries(_pds)){if(d.total===0){rhythm[pd]={dominant:'none',pct:{},total:0};continue;}let bs=['walk','lick','scratch','shake','stop'],dom=bs.reduce((a,b)=>d[a]>=d[b]?a:b,'stop'),pct={};for(let b of bs)pct[b]=parseFloat((d[b]/d.total*100).toFixed(2));rhythm[pd]={dominant:dom,pct,total:parseFloat(d.total.toFixed(1))};}\n"
    "\n"
    "// ── 偏差計算（vs 個體基線）──\n"
    "let bl=null;try{bl=global.get('v2_baseline','file')||global.get('v2_baseline');}catch(e){bl=global.get('v2_baseline');}\n"
    "let dev=null;\n"
    "if(bl&&bl.metrics){let m=bl.metrics;function pctDev(cur,key){let b=m[key];if(!b||b.mean===0)return null;return parseFloat(((cur-b.mean)/b.mean*100).toFixed(1));}dev={lick_time:pctDev(lickT,'lick_time'),scratch_time:pctDev(scratchT,'scratch_time'),shake_count:pctDev(stats.shake||0,'shake_count'),stop_time:pctDev(stopT,'stop_time'),lick_count:pctDev(stats.lick||0,'lick_count'),scratch_count:pctDev(stats.scratch||0,'scratch_count'),walk_time:pctDev(walkT,'walk_time'),walk_count:pctDev(stats.walk||0,'walk_count')};}\n"
    "\n"
    "// ── 同步寫入 v2 global context（讓 v2 基線引擎定時任務繼續運作）──\n"
    "let _td2=new Date().toLocaleDateString('zh-TW');\n"
    "global.set('v2_today',{date:_td2,walk:{count:stats.walk||0,time:walkT},lick:{count:stats.lick||0,time:lickT},scratch:{count:stats.scratch||0,time:scratchT},shake:{count:stats.shake||0,time:shakeT},stop:{count:stats.stop||0,time:stopT},active_time:stats.active_time||0,rest_time:stats.rest_time||0,not_detected_time:stats.not_detected_time||0,monitoring_seconds:stats.monitoring_seconds||0},'file');\n"
    "global.set('v2_hourly',stats.hourly_distribution||{});\n"
    "global.set('v2_transition_matrix',stats.transition_matrix||{});\n"
    "\n"
    "let mx=stats.transition_matrix||{};\n"
    "let td={active_time:stats.active_time||0};"
)

# ── 2. 健康警示：節律/轉移卡解除灰化 ──
OLD_GRAYED = (
    "    <div class=\"hw-comp-card\" style=\"opacity:.32\">\n"
    "      <div class=\"hw-comp-title\">🌙 節律偏離（需每小時資料）</div>\n"
    "      <div class=\"hw-comp-score comp-na\">0</div>\n"
    "      <div class=\"hw-comp-bar\"><div class=\"hw-comp-fill fill-na\" style=\"width:0%\"></div></div>\n"
    "    </div>\n"
    "    <div class=\"hw-comp-card\" style=\"opacity:.32\">\n"
    "      <div class=\"hw-comp-title\">🔀 轉移偏離（需轉移矩陣）</div>\n"
    "      <div class=\"hw-comp-score comp-na\">0</div>\n"
    "      <div class=\"hw-comp-bar\"><div class=\"hw-comp-fill fill-na\" style=\"width:0%\"></div></div>\n"
    "    </div>"
)

NEW_ACTIVE = (
    "    <div class=\"hw-comp-card\">\n"
    "      <div class=\"hw-comp-title\">🌙 節律偏離</div>\n"
    "      <div class=\"hw-comp-score comp-rhy\">{{comp('rhythm')}}</div>\n"
    "      <div class=\"hw-comp-bar\"><div class=\"hw-comp-fill fill-rhy\" ng-style=\"{width:comp('rhythm')+'%'}\"></div></div>\n"
    "    </div>\n"
    "    <div class=\"hw-comp-card\">\n"
    "      <div class=\"hw-comp-title\">🔀 轉移偏離</div>\n"
    "      <div class=\"hw-comp-score comp-trans\">{{comp('transition')}}</div>\n"
    "      <div class=\"hw-comp-bar\"><div class=\"hw-comp-fill fill-trans\" ng-style=\"{width:comp('transition')+'%'}\"></div></div>\n"
    "    </div>"
)

OLD_CSS_TAIL = (
    ".comp-dist{color:#85B7EB}.comp-freq{color:#F09595}.comp-na{color:rgba(255,255,255,.2)}\n"
    ".fill-dist{background:linear-gradient(90deg,#1e88e5,#64b5f6)}\n"
    ".fill-freq{background:linear-gradient(90deg,#e53935,#ef9a9a)}\n"
    ".fill-na{background:rgba(255,255,255,.12)}"
)

NEW_CSS_TAIL = (
    ".comp-dist{color:#85B7EB}.comp-freq{color:#F09595}.comp-rhy{color:#EF9F27}.comp-trans{color:#9FE1CB}\n"
    ".fill-dist{background:linear-gradient(90deg,#1e88e5,#64b5f6)}\n"
    ".fill-freq{background:linear-gradient(90deg,#e53935,#ef9a9a)}\n"
    ".fill-rhy{background:linear-gradient(90deg,#f9a825,#ffe082)}\n"
    ".fill-trans{background:linear-gradient(90deg,#1d9e75,#5dcaa5)}"
)

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    nid = n.get('id','')

    if nid == ENGINE_ID:
        func = n['func']
        if OLD_HEADER in func:
            n['func'] = func.replace(OLD_HEADER, NEW_HEADER)
            print('Fixed: 健康引擎分析 → 自算 distribution/rhythm/deviation + v2 global sync')
        else:
            print('WARNING: 健康引擎分析 header pattern not matched')

    elif nid == ALERT_UI_ID:
        fmt = n['format']
        changed = False
        if OLD_GRAYED in fmt:
            fmt = fmt.replace(OLD_GRAYED, NEW_ACTIVE)
            changed = True
            print('Fixed: 健康警示 → 節律/轉移卡解除灰化（動態綁定）')
        else:
            print('WARNING: 健康警示 grayed-card pattern not matched')
        if OLD_CSS_TAIL in fmt:
            fmt = fmt.replace(OLD_CSS_TAIL, NEW_CSS_TAIL)
            changed = True
            print('Fixed: 健康警示 → 新增 fill-rhy / fill-trans CSS')
        else:
            print('WARNING: 健康警示 CSS tail pattern not matched')
        if changed:
            n['format'] = fmt

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')

# ══════════════════════════════════════════════════════════════════
# 3. cat_health_v2_flow.json  ── 移除 P5 面板
# ══════════════════════════════════════════════════════════════════
V2_PATH        = r'c:\ai_project\paper\cat_health_v2_flow.json'
REMOVE_IDS     = {'b2a8610d74979717', '88627ebf3b74045d', '4d01766fcc276f06'}
ENGINE_V2_ID   = '4b0c232763de1461'   # 健康預警引擎
P5_LINK_OUT    = '4d01766fcc276f06'   # → P5健康預警 link_out

with open(V2_PATH, 'r', encoding='utf-8') as f:
    v2nodes = json.load(f)

kept = []
for n in v2nodes:
    nid = n.get('id','')
    if nid in REMOVE_IDS:
        print(f'Removed: v2 node {nid} ({n.get("name",n.get("type",""))})')
        continue

    if nid == ENGINE_V2_ID:
        # wires[0] 中移除指向 P5 的 link_out
        wires = n.get('wires', [])
        if wires and P5_LINK_OUT in wires[0]:
            wires[0] = [w for w in wires[0] if w != P5_LINK_OUT]
            n['wires'] = wires
            print(f'Fixed: v2 健康預警引擎 wires → 移除 P5 link_out')

    kept.append(n)

with open(V2_PATH, 'w', encoding='utf-8') as f:
    json.dump(kept, f, ensure_ascii=False, indent=4)
print('Written: cat_health_v2_flow.json')
print('\nAll done.')
