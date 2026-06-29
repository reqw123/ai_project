"""
修正全域 ui_template 拖曳抖動問題

問題根因：
  拖曳流程為 iframe.pointermove → postMessage → 父頁面 message event → rAF → 更新 transform
  每次移動多一次 async postMessage 往返，約延遲 1-2 幀 (16-32ms)，造成抖動感。

修正方式：
  拖曳「開始」時：iframe 傳送一次 gfv-drag-start（帶起始座標）給父頁面
  拖曳「進行中」：父頁面把 iframe 的 pointer-events 設為 none，
                  讓 document.pointermove 直接在父頁面觸發，
                  不再需要 postMessage，直接呼叫 liveMoveBy()
  拖曳「結束」時：父頁面的 document.pointerup 呼叫 commitMove()，
                  恢復 pointer-events，並傳 gfv-drag-end 給 iframe 更新 UI
"""

import json, sys

SRC = r'c:\cat_paper_code-main\paper\貓咪主控.json'

with open(SRC, 'r', encoding='utf-8') as f:
    data = json.load(f)

patched = False
for node in data:
    if node.get('type') == 'ui_template' and node.get('templateScope') == 'global':
        fmt = node['format']

        # ── 修正 1：父頁面 message handler ────────────────────────────────
        # 加入 gfv-drag-start 處理，並新增 document.pointermove / pointerup 接管
        OLD_PARENT_MSG = (
            "  window.addEventListener('message', function(e){\n"
            "    if(!e.data) return;\n"
            "    if(e.data.type === 'gfv-move'){            // 拖曳進行中：位移增量 dx, dy\n"
            "      liveMoveBy(e.data.dx, e.data.dy);\n"
            "    }\n"
            "    if(e.data.type === 'gfv-move-end'){         // 拖曳結束：烘焙座標\n"
            "      commitMove();\n"
            "    }\n"
            "    if(e.data.type === 'gfv-resize'){           // 縮放進行中：目標寬高（錨定左上角）\n"
            "      liveResizeTo(e.data.width, e.data.height);\n"
            "    }\n"
            "    if(e.data.type === 'gfv-resize-end'){       // 縮放結束：烘焙座標\n"
            "      commitResize();\n"
            "    }\n"
            "    if(e.data.type === 'gfv-visible'){\n"
            "      fr.style.setProperty('pointer-events', e.data.on ? 'auto' : 'none', 'important');\n"
            "    }\n"
            "  });"
        )
        NEW_PARENT_MSG = (
            "  /* ── 父頁面直接接管拖曳，消除 iframe postMessage 往返延遲 ──\n"
            "     拖曳中 iframe 的 pointer-events 設為 none，\n"
            "     父頁面 document.pointermove 直接觸發，零跨 frame 延遲。 */\n"
            "  var _pdrag=false,_pdragPX=0,_pdragPY=0;\n"
            "\n"
            "  window.addEventListener('message', function(e){\n"
            "    if(!e.data) return;\n"
            "    if(e.data.type==='gfv-drag-start'){              // 拖曳開始：接管 pointermove\n"
            "      _pdrag=true; _pdragPX=e.data.x; _pdragPY=e.data.y;\n"
            "      fr.style.setProperty('pointer-events','none','important');\n"
            "    }\n"
            "    if(e.data.type==='gfv-move'){                    // 相容舊版（保留）\n"
            "      liveMoveBy(e.data.dx, e.data.dy);\n"
            "    }\n"
            "    if(e.data.type==='gfv-move-end'){ commitMove(); }\n"
            "    if(e.data.type==='gfv-resize'){\n"
            "      liveResizeTo(e.data.width, e.data.height);\n"
            "    }\n"
            "    if(e.data.type==='gfv-resize-end'){ commitResize(); }\n"
            "    if(e.data.type==='gfv-visible'){\n"
            "      if(!_pdrag)  // 拖曳中不覆蓋 pointer-events:none\n"
            "        fr.style.setProperty('pointer-events', e.data.on ? 'auto' : 'none', 'important');\n"
            "    }\n"
            "  });\n"
            "\n"
            "  /* 父頁面直接接收 pointermove（零 postMessage 延遲）*/\n"
            "  document.addEventListener('pointermove',function(e){\n"
            "    if(!_pdrag) return;\n"
            "    var dx=e.clientX-_pdragPX, dy=e.clientY-_pdragPY;\n"
            "    _pdragPX=e.clientX; _pdragPY=e.clientY;\n"
            "    liveMoveBy(dx, dy);\n"
            "  });\n"
            "\n"
            "  /* 父頁面接收 pointerup，提交座標並恢復 iframe 事件 */\n"
            "  document.addEventListener('pointerup',function(){\n"
            "    if(!_pdrag) return;\n"
            "    commitMove();\n"
            "    _pdrag=false;\n"
            "    fr.style.setProperty('pointer-events','auto','important');\n"
            "    try{ fr.contentWindow.postMessage({type:'gfv-drag-end'},'*'); }catch(ex){}\n"
            "  });"
        )

        if OLD_PARENT_MSG not in fmt:
            print('[WARN] 找不到父頁面 message handler，跳過修正 1', file=sys.stderr)
        else:
            fmt = fmt.replace(OLD_PARENT_MSG, NEW_PARENT_MSG, 1)
            print('[OK] 修正 1：父頁面 message handler + document.pointermove/pointerup')

        # ── 修正 2：iframe dragStart ────────────────────────────────────────
        # 移除 setPointerCapture（父頁面接管後無需）
        # 改送 gfv-drag-start 給父頁面，只送一次
        OLD_DRAG_START = (
            "function dragStart(e){\n"
            "  if(e.pointerType==='mouse'&&e.button!==0)return;\n"
            "  if(e.target===rhE||e.target===rhS)return;\n"
            "  if(!isDragTarget(e.target))return;\n"
            "  da=true;pid=e.pointerId;\n"
            "  ox=e.clientX;oy=e.clientY;\n"
            "  pnl.setPointerCapture(e.pointerId);\n"
            "  bar.classList.add('dragging');dragIcon.classList.add('on');\n"
            "  e.preventDefault();e.stopPropagation();\n"
            "}"
        )
        NEW_DRAG_START = (
            "function dragStart(e){\n"
            "  if(e.pointerType==='mouse'&&e.button!==0)return;\n"
            "  if(e.target===rhE||e.target===rhS)return;\n"
            "  if(!isDragTarget(e.target))return;\n"
            "  da=true;pid=e.pointerId;\n"
            "  ox=e.clientX;oy=e.clientY;\n"
            "  bar.classList.add('dragging');dragIcon.classList.add('on');\n"
            "  // 通知父頁面接管 pointermove（消除 postMessage 往返延遲）\n"
            "  parent.postMessage({type:'gfv-drag-start',x:e.clientX,y:e.clientY},'*');\n"
            "  e.preventDefault();e.stopPropagation();\n"
            "}"
        )
        if OLD_DRAG_START not in fmt:
            print('[WARN] 找不到 dragStart，跳過修正 2', file=sys.stderr)
        else:
            fmt = fmt.replace(OLD_DRAG_START, NEW_DRAG_START, 1)
            print('[OK] 修正 2：iframe dragStart → 改送 gfv-drag-start')

        # ── 修正 3：iframe dragMove ─────────────────────────────────────────
        # 父頁面已接管 pointermove，這裡不再需要 postMessage
        OLD_DRAG_MOVE = (
            "function dragMove(e){\n"
            "  if(!da||e.pointerId!==pid)return;\n"
            "  var dx=e.clientX-ox, dy=e.clientY-oy;\n"
            "  ox=e.clientX; oy=e.clientY;\n"
            "  parent.postMessage({type:'gfv-move', dx:dx, dy:dy}, '*');\n"
            "  e.preventDefault();\n"
            "}"
        )
        NEW_DRAG_MOVE = (
            "function dragMove(e){\n"
            "  if(!da||e.pointerId!==pid)return;\n"
            "  e.preventDefault(); // 父頁面直接處理 pointermove，不再需要 postMessage\n"
            "}"
        )
        if OLD_DRAG_MOVE not in fmt:
            print('[WARN] 找不到 dragMove，跳過修正 3', file=sys.stderr)
        else:
            fmt = fmt.replace(OLD_DRAG_MOVE, NEW_DRAG_MOVE, 1)
            print('[OK] 修正 3：iframe dragMove → 移除 postMessage')

        # ── 修正 4：iframe dragEnd ──────────────────────────────────────────
        # pointer-events:none 時 pointerup 不會在 iframe 觸發，
        # 父頁面的 document.pointerup 會處理 commitMove 和恢復
        OLD_DRAG_END = (
            "function dragEnd(e){\n"
            "  if(!da||e.pointerId!==pid)return;\n"
            "  da=false;pid=null;\n"
            "  pnl.releasePointerCapture(e.pointerId);\n"
            "  bar.classList.remove('dragging');dragIcon.classList.remove('on');\n"
            "  parent.postMessage({type:'gfv-move-end'}, '*');\n"
            "}"
        )
        NEW_DRAG_END = (
            "function dragEnd(e){\n"
            "  // pointer-events:none 時此函式通常不被呼叫；父頁面 pointerup 會處理\n"
            "  if(!da) return;\n"
            "  da=false;pid=null;\n"
            "  try{ pnl.releasePointerCapture(e.pointerId); }catch(ex){}\n"
            "}"
        )
        if OLD_DRAG_END not in fmt:
            print('[WARN] 找不到 dragEnd，跳過修正 4', file=sys.stderr)
        else:
            fmt = fmt.replace(OLD_DRAG_END, NEW_DRAG_END, 1)
            print('[OK] 修正 4：iframe dragEnd → 簡化')

        # ── 修正 5：iframe window.addEventListener('message') ──────────────
        # 加入 gfv-drag-end 的 UI 恢復處理（父頁面發回）
        OLD_IFRAME_MSG = (
            "window.addEventListener('message',function(e){\n"
            "  if(!e.data) return;\n"
            "  if(e.data.type==='gfv-url'){"
        )
        NEW_IFRAME_MSG = (
            "window.addEventListener('message',function(e){\n"
            "  if(!e.data) return;\n"
            "  if(e.data.type==='gfv-drag-end'){  // 父頁面通知：拖曳結束，恢復 UI\n"
            "    bar.classList.remove('dragging');\n"
            "    dragIcon.classList.remove('on');\n"
            "    da=false;\n"
            "  }\n"
            "  if(e.data.type==='gfv-url'){"
        )
        if OLD_IFRAME_MSG not in fmt:
            print('[WARN] 找不到 iframe message handler，跳過修正 5', file=sys.stderr)
        else:
            fmt = fmt.replace(OLD_IFRAME_MSG, NEW_IFRAME_MSG, 1)
            print('[OK] 修正 5：iframe message handler 加入 gfv-drag-end 處理')

        node['format'] = fmt
        patched = True
        break

if not patched:
    print('[ERROR] 找不到全域 ui_template 節點', file=sys.stderr)
    sys.exit(1)

with open(SRC, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

print('[完成] 貓咪主控.json 已更新，拖曳抖動修正完成')
