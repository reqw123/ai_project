# -*- coding: utf-8 -*-
"""
讓 影像串流 widget 預設自動進入浮動小視窗模式（360×240）
1. 縮小 toggleFloating() 浮動尺寸 640×420 → 360×240
2. 頁面載入後自動觸發浮動（setTimeout 200ms）
3. 浮動時靠右上角，不與按鈕重疊
4. grid height 8 → 3（縮小 grid 佔位，避免空白佔版）
"""
import json, re

MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') != '0b47210816be0ad4':
        continue

    fmt = n['format']

    # ── 1. 縮小浮動尺寸 640→360, 420→240, top/left→top/right ──────
    OLD_TOGGLE = (
        '        if (panel.classList.contains("floating")) {\n'
        '            panel.style.top = "80px";\n'
        '            panel.style.left = "80px";\n'
        '            panel.style.width = "640px";\n'
        '            panel.style.height = "420px";\n'
        '        } else {\n'
        '            panel.style.top = "";\n'
        '            panel.style.left = "";\n'
        '            panel.style.width = "100%";\n'
        '            panel.style.height = "";\n'
        '        }'
    )
    NEW_TOGGLE = (
        '        if (panel.classList.contains("floating")) {\n'
        '            panel.style.top = "70px";\n'
        '            panel.style.right = "20px";\n'
        '            panel.style.left = "auto";\n'
        '            panel.style.bottom = "auto";\n'
        '            panel.style.width = "360px";\n'
        '            panel.style.height = "240px";\n'
        '        } else {\n'
        '            panel.style.top = "";\n'
        '            panel.style.right = "";\n'
        '            panel.style.left = "";\n'
        '            panel.style.bottom = "";\n'
        '            panel.style.width = "100%";\n'
        '            panel.style.height = "";\n'
        '        }'
    )
    if OLD_TOGGLE in fmt:
        fmt = fmt.replace(OLD_TOGGLE, NEW_TOGGLE)
        print('✓ 浮動尺寸改為 360×240，位置靠右上角')
    else:
        print('⚠ toggleFloating 尺寸段落未找到，請手動確認')

    # ── 2. 在 scope.$watch 之前插入「預設自動浮動」──────────────────
    AUTO_FLOAT_SNIPPET = (
        '\n'
        '    // 預設自動進入小視窗模式\n'
        '    setTimeout(function() {\n'
        '        if (!panel.classList.contains("floating")) {\n'
        '            toggleFloating();\n'
        '        }\n'
        '    }, 200);\n\n'
    )
    BEFORE_WATCH = '    scope.$watch("msg", function(msg) {'
    if BEFORE_WATCH in fmt:
        fmt = fmt.replace(BEFORE_WATCH, AUTO_FLOAT_SNIPPET + '    ' + 'scope.$watch("msg", function(msg) {', 1)
        print('✓ 插入預設自動浮動（setTimeout 200ms）')
    else:
        print('⚠ scope.$watch 錨點未找到')

    # ── 3. 調整 .stream-floating-panel 浮動狀態的 CSS ──────────────
    # 原: top:80px;left:80px;width:640px;height:420px
    # 改: top:70px;right:20px（JS 動態設定，CSS 只留基礎）
    fmt = re.sub(
        r'\.stream-floating-panel\.floating \{[^}]*\}',
        (
            '.stream-floating-panel.floating {\n'
            '        position: fixed;\n'
            '        z-index: 9999;\n'
            '        resize: both;\n'
            '    }'
        ),
        fmt,
        count=1
    )
    print('✓ 更新 .stream-floating-panel.floating CSS')

    # ── 4. 縮小 grid 高度：height "8" → "3" ────────────────────────
    n['height'] = '3'
    print('✓ grid height 8 → 3')

    n['format'] = fmt
    break

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
