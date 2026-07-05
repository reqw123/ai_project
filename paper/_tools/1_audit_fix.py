# -*- coding: utf-8 -*-
"""
三檔審計修復：
1. 貓咪主控.json  - 數據分發器 (89b182d351391d5f) 語法錯誤 + zombie 變數
2. 貓咪分析.json  - 建立GPT分析請求 (77ce08ade825bf7a) zombie health_score 欄位與分析邏輯
3. cat_health_v2_flow.json - 4 處 health_score zombie：行為統計累積器、P1面板、P2面板、建立CSV
"""
import json, re

# ══════════════════════════════════════════════════════════════════
# 1. 貓咪主控.json  ── 修復 數據分發器
# ══════════════════════════════════════════════════════════════════
MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'
DISTRIB_ID = '89b182d351391d5f'

NEW_DISTRIB_FUNC = (
    "let data = msg.payload;\n"
    "if (!data) return null;\n"
    "\n"
    "if (!data.current)     data.current     = {};\n"
    "if (!data.today_stats) data.today_stats = {};\n"
    "\n"
    "let stats = data.today_stats;\n"
    "stats.walk             = stats.walk             || 0;\n"
    "stats.walk_time        = stats.walk_time        || 0;\n"
    "stats.lick             = stats.lick             || 0;\n"
    "stats.lick_time        = stats.lick_time        || 0;\n"
    "stats.scratch          = stats.scratch          || 0;\n"
    "stats.scratch_time     = stats.scratch_time     || 0;\n"
    "stats.shake            = stats.shake            || 0;\n"
    "stats.shake_time       = stats.shake_time       || 0;\n"
    "stats.stop             = stats.stop             || 0;\n"
    "stats.stop_time        = stats.stop_time        || 0;\n"
    "stats.not_detected_time = stats.not_detected_time || 0;\n"
    "stats.active_time      = stats.active_time      || 0;\n"
    "stats.rest_time        = stats.rest_time        || 0;\n"
    "\n"
    "if (Array.isArray(data.behavior_log)) {\n"
    "    data.behavior_log = data.behavior_log.map(function(item) {\n"
    "        return {\n"
    "            behavior: item.behavior || 'unknown',\n"
    "            time:     item.time     || '--:--',\n"
    "            duration: item.duration || 0,\n"
    "            emoji:    getEmoji(item.behavior)\n"
    "        };\n"
    "    });\n"
    "} else {\n"
    "    data.behavior_log = [];\n"
    "}\n"
    "\n"
    "if (!data.current.emoji)     data.current.emoji     = getEmoji(data.current.behavior);\n"
    "if (!data.current.timestamp) data.current.timestamp = new Date().toLocaleTimeString();\n"
    "\n"
    "msg.payload = data;\n"
    "\n"
    "// CSV：每 5 分鐘寫一次（與 v2 flow 一致）\n"
    "let now     = Date.now();\n"
    "let lastCsv = flow.get('last_csv_write') || 0;\n"
    "let doCsv   = (now - lastCsv) >= 300000;\n"
    "if (doCsv) flow.set('last_csv_write', now);\n"
    "\n"
    "let csvMsg = doCsv ? { payload: {\n"
    "    timestamp:         new Date().toISOString(),\n"
    "    walk:              stats.walk,              walk_time:         stats.walk_time,\n"
    "    lick:              stats.lick,              lick_time:         stats.lick_time,\n"
    "    scratch:           stats.scratch,           scratch_time:      stats.scratch_time,\n"
    "    shake:             stats.shake,             shake_time:        stats.shake_time,\n"
    "    stop:              stats.stop,              stop_time:         stats.stop_time,\n"
    "    active_time:       stats.active_time,\n"
    "    rest_time:         stats.rest_time,\n"
    "    not_detected_time: stats.not_detected_time,\n"
    "    activity_score:    data.activity_score || 0\n"
    "}} : null;\n"
    "\n"
    "return [\n"
    "    msg,\n"
    "    msg,\n"
    "    msg,\n"
    "    msg,\n"
    "    { payload: data.activity_score || 0 },\n"
    "    csvMsg\n"
    "];\n"
    "\n"
    "function getEmoji(behavior) {\n"
    "    switch (behavior) {\n"
    "        case 'scratch': return '🐾';\n"
    "        case 'lick':    return '🧼';\n"
    "        case 'walk':    return '🚶';\n"
    "        case 'shake':   return '🔄';\n"
    "        case 'stop':    return '⏹';\n"
    "        case 'not':     return '📷';\n"
    "        default:        return '🐱';\n"
    "    }\n"
    "}"
)

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)
for n in nodes:
    if n.get('id') == DISTRIB_ID:
        n['func'] = NEW_DISTRIB_FUNC
        print('Fixed: 數據分發器 (語法錯誤 + zombie shouldWriteCsv/csvPayload)')
with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)

# ══════════════════════════════════════════════════════════════════
# 2. 貓咪分析.json  ── 移除 GPT system prompt 中的 health_score 殭屍參照
# ══════════════════════════════════════════════════════════════════
ANALYSIS_PATH = r'c:\ai_project\paper\貓咪分析.json'
GPT_NODE_ID   = '77ce08ade825bf7a'

with open(ANALYSIS_PATH, 'r', encoding='utf-8') as f:
    anodes = json.load(f)

for n in anodes:
    if n.get('id') != GPT_NODE_ID:
        continue
    func = n['func']

    # (a) 移除 CSV 欄位說明中的 health_score 行
    func = re.sub(
        r'\nhealth_score：健康分數（0–100）\n',
        '\n',
        func
    )

    # (b) 重寫第四分析項：移除 health_score 依賴，改為 activity_score + 異常行為
    OLD_SECTION4 = (
        "四、整體健康風險評估\n"
        "請綜合 health_score、activity_score、lick、scratch、shake 進行判斷。\n"
        "若 health_score ≥ 80，視為低風險；\n"
        "50–79 視為中風險；\n"
        "< 50 視為高風險。\n"
        "若單一異常明顯超標或多項異常同時偏高，也應提高風險等級。"
    )
    NEW_SECTION4 = (
        "四、整體行為風險評估\n"
        "請綜合 activity_score、lick_time、scratch_time、shake_time、stop_time 進行判斷。\n"
        "計算 abnormal_ratio = (lick_time + scratch_time + shake_time) / (active_time + rest_time)；\n"
        "若 activity_score ≥ 70 且 abnormal_ratio < 0.2，視為低風險；\n"
        "若 activity_score 40–69 或 abnormal_ratio 0.2–0.4，視為中風險；\n"
        "若 activity_score < 40 或 abnormal_ratio > 0.4，視為高風險；\n"
        "若單一異常明顯超標或多項異常同時偏高，也應提高風險等級。"
    )
    if OLD_SECTION4 in func:
        func = func.replace(OLD_SECTION4, NEW_SECTION4)
        print('Fixed: 建立GPT分析請求 - 移除 health_score 欄位說明 + 重寫第四分析項')
    else:
        print('WARNING: 建立GPT分析請求 section4 pattern not matched - check manually')

    n['func'] = func

with open(ANALYSIS_PATH, 'w', encoding='utf-8') as f:
    json.dump(anodes, f, ensure_ascii=False, indent=4)

# ══════════════════════════════════════════════════════════════════
# 3. cat_health_v2_flow.json  ── 4 處 health_score 殭屍
# ══════════════════════════════════════════════════════════════════
V2_PATH         = r'c:\ai_project\paper\cat_health_v2_flow.json'
V2_ACCUM_ID     = '8feb3527c9bca065'   # 行為統計累積器
V2_P1_ID        = '11eb0cbc1df49089'   # P1 即時監控面板
V2_P2_ID        = 'c90243a7c1a9407d'   # P2 今日統計面板
V2_CSV_ID       = 'b4352a0355dd4f8e'   # 建立CSV (daily_stats.csv)

with open(V2_PATH, 'r', encoding='utf-8') as f:
    v2nodes = json.load(f)

for n in v2nodes:
    nid = n.get('id', '')

    # 3a. 行為統計累積器：移除 csvMsg 中的 health_score 欄位
    if nid == V2_ACCUM_ID:
        func = n['func']
        func = re.sub(
            r',?\s*\n\s*health_score:\s*data\.health_score\s*\|\|\s*0\n',
            '\n',
            func
        )
        # 處理 inline 格式的情況
        func = re.sub(
            r',\s*health_score:\s*data\.health_score\s*\|\|\s*0',
            '',
            func
        )
        n['func'] = func
        print('Fixed: v2 行為統計累積器 - 移除 csvMsg.health_score')

    # 3b. P1 即時監控面板：移除 健康分 score-box
    elif nid == V2_P1_ID:
        fmt = n['format']
        OLD_HS_BOX_P1 = (
            "\n      <div class=\"p1-score-box\">\n"
            "        <div class=\"p1-score-val\">{{msg.payload.health_score || '--'}}</div>\n"
            "        <div class=\"p1-score-lbl\">健康分</div>\n"
            "      </div>"
        )
        if OLD_HS_BOX_P1 in fmt:
            fmt = fmt.replace(OLD_HS_BOX_P1, '')
            n['format'] = fmt
            print('Fixed: v2 P1 面板 - 移除健康分 score-box')
        else:
            print('WARNING: v2 P1 健康分 box not matched')

    # 3c. P2 今日統計面板：健康分 sum-box → 風險分 sum-box；移除 scoreColor 函式
    elif nid == V2_P2_ID:
        fmt = n['format']
        # 替換 Health Score 顯示方塊
        OLD_HS_BOX_P2 = (
            "\n    <div class=\"p2-sum-box\">\n"
            "      <div class=\"p2-sum-val\" ng-style=\"{color: scoreColor()}\">{{msg.payload.health_score || '--'}}</div>\n"
            "      <div class=\"p2-sum-lbl\">Health Score</div>\n"
            "    </div>"
        )
        NEW_RISK_BOX_P2 = (
            "\n    <div class=\"p2-sum-box\">\n"
            "      <div class=\"p2-sum-val\" ng-style=\"{color: riskColor()}\">{{riskScore()}}</div>\n"
            "      <div class=\"p2-sum-lbl\">Risk Score</div>\n"
            "    </div>"
        )
        if OLD_HS_BOX_P2 in fmt:
            fmt = fmt.replace(OLD_HS_BOX_P2, NEW_RISK_BOX_P2)
        else:
            print('WARNING: v2 P2 Health Score box not matched')

        # 移除 scoreColor 函式，新增 riskColor + riskScore 函式
        OLD_SCORE_COLOR = (
            "scope.scoreColor = function(){ "
            "let s=scope.msg&&scope.msg.payload&&scope.msg.payload.health_score||0; "
            "return s>=80?'#66bb6a':s>=60?'#ffa726':'#ef5350'; };"
        )
        NEW_RISK_FUNCS = (
            "scope.riskScore = function(){ "
            "let r=scope.msg&&scope.msg.payload&&scope.msg.payload.risk; "
            "return r?r.score:'--'; };\n"
            "  scope.riskColor = function(){ "
            "let r=scope.msg&&scope.msg.payload&&scope.msg.payload.risk; "
            "return r&&r.color?r.color:'#888888'; };"
        )
        if OLD_SCORE_COLOR in fmt:
            fmt = fmt.replace(OLD_SCORE_COLOR, NEW_RISK_FUNCS)
            n['format'] = fmt
            print('Fixed: v2 P2 面板 - Health Score→Risk Score + 替換 scoreColor→riskColor/riskScore')
        else:
            print('WARNING: v2 P2 scoreColor function not matched')
            n['format'] = fmt  # save partial fix

    # 3d. 建立CSV (daily_stats.csv)：移除 health_score 欄
    elif nid == V2_CSV_ID:
        temp = n.get('temp', '')
        new_temp = re.sub(r',?health_score', '', temp)
        if new_temp != temp:
            n['temp'] = new_temp
            print('Fixed: v2 建立CSV - 移除 health_score 欄位')
        else:
            print('WARNING: v2 CSV health_score not found in temp')

with open(V2_PATH, 'w', encoding='utf-8') as f:
    json.dump(v2nodes, f, ensure_ascii=False, indent=4)

print('\nAll done.')
