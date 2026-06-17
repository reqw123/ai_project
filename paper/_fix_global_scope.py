# -*- coding: utf-8 -*-
"""
修正 global_float_video_01：移除 scope 依賴，改用 window._catIP 橋接
修正 影像串流：在 scope.$watch 內設置 window._catIP 並呼叫 window.gfvSetUrl
"""
import json

MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'

# ── 新的 global float video template（完全不用 scope）────────────────
NEW_GFV_FORMAT = r"""<script>
(function(){
  var BTN_ID   = 'gfv_float_btn';
  var PANEL_ID = 'gfv_float_panel';
  var IMG_ID   = 'gfv_float_img';
  var _url     = '';

  // 讓 local template 可以呼叫此函式推送串流 URL
  window.gfvSetUrl = function(url){
    if (!url || url === _url) return;
    _url = url;
    var img = document.getElementById(IMG_ID);
    if (img){ img.style.opacity = 0; img.src = url; }
  };

  // 防止頁面切換時重複建立
  if (document.getElementById(BTN_ID)) return;

  // ── 注入 CSS ──────────────────────────────────────────────────────
  var st = document.createElement('style');
  st.textContent =
    '#gfv_float_btn{position:fixed;bottom:20px;right:20px;' +
    'width:46px;height:46px;border-radius:50%;' +
    'background:rgba(13,17,23,.92);' +
    'border:2px solid rgba(0,255,136,.5);' +
    'color:#00ff88;font-size:22px;line-height:44px;' +
    'text-align:center;cursor:pointer;' +
    'z-index:2147483647;' +
    'box-shadow:0 4px 16px rgba(0,0,0,.7);' +
    'user-select:none;transition:transform .15s,border-color .15s;}' +
    '#gfv_float_btn:hover{transform:scale(1.12);border-color:#00ff88;}' +
    '#gfv_float_panel{position:fixed;bottom:76px;right:20px;width:300px;' +
    'background:#0d1117;border:1px solid rgba(0,255,136,.28);' +
    'border-radius:12px;overflow:hidden;z-index:2147483646;' +
    'box-shadow:0 8px 32px rgba(0,0,0,.85);' +
    'display:none;flex-direction:column;}' +
    '#gfv_bar{background:rgba(0,0,0,.65);padding:8px 12px;' +
    'font-size:12px;color:#00ff88;' +
    'display:flex;justify-content:space-between;align-items:center;' +
    'cursor:move;user-select:none;' +
    'border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;}' +
    '#gfv_close{cursor:pointer;opacity:.55;font-size:14px;}' +
    '#gfv_close:hover{opacity:1;}' +
    '#gfv_float_img{width:100%;display:block;background:#000;min-height:168px;object-fit:contain;}' +
    '#gfv_offline{text-align:center;padding:30px 0;color:rgba(255,255,255,.3);font-size:12px;}';
  document.head.appendChild(st);

  // ── 建立按鈕 ──────────────────────────────────────────────────────
  var btn = document.createElement('div');
  btn.id = BTN_ID;
  btn.title = '懸浮影像視窗';
  btn.textContent = '📷';
  document.body.appendChild(btn);

  // ── 建立面板 ──────────────────────────────────────────────────────
  var pnl = document.createElement('div');
  pnl.id = PANEL_ID;
  pnl.innerHTML =
    '<div id="gfv_bar">' +
      '<span>📷 即時串流</span>' +
      '<span id="gfv_close">&#10005;</span>' +
    '</div>' +
    '<img id="gfv_float_img" src="" alt="" style="opacity:0;" ' +
      'onload="this.style.opacity=1;var o=document.getElementById(\'gfv_offline\');if(o)o.style.display=\'none\';" ' +
      'onerror="this.style.opacity=0;var o=document.getElementById(\'gfv_offline\');if(o)o.style.display=\'block\';">' +
    '<div id="gfv_offline">📹 串流離線</div>';
  document.body.appendChild(pnl);

  // ── 開關（點擊時嘗試讀 window._catIP）────────────────────────────
  function gfvToggle(){
    var p = document.getElementById(PANEL_ID);
    if (!p) return;
    var opening = p.style.display !== 'flex';
    p.style.display = opening ? 'flex' : 'none';
    // 開啟時若還沒有 URL，嘗試從 window._catIP 補上
    if (opening && !_url && window._catIP) {
      window.gfvSetUrl('http://' + window._catIP + ':5000/stream');
    }
  }
  btn.addEventListener('click', gfvToggle);
  setTimeout(function(){
    var c = document.getElementById('gfv_close');
    if (c) c.addEventListener('click', gfvToggle);
  }, 0);

  // ── 拖曳 ──────────────────────────────────────────────────────────
  var drag = false, ox = 0, oy = 0;
  document.getElementById('gfv_bar').addEventListener('mousedown', function(e){
    drag = true;
    var r = pnl.getBoundingClientRect();
    ox = e.clientX - r.left; oy = e.clientY - r.top;
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    if (!drag) return;
    pnl.style.left = (e.clientX - ox) + 'px';
    pnl.style.top  = (e.clientY - oy) + 'px';
    pnl.style.right = 'auto'; pnl.style.bottom = 'auto';
  });
  document.addEventListener('mouseup', function(){ drag = false; });

})();
</script>"""

# ── 影像串流 scope.$watch 末尾追加的橋接代碼 ─────────────────────────
# 在 scope.$watch 裡，finalUrl 計算完後，設置 window._catIP 並呼叫 gfvSetUrl
BRIDGE_SNIPPET = """
        // 橋接全域懸浮按鈕
        if (ip) {
            window._catIP = ip;
            if (typeof window.gfvSetUrl === 'function') {
                window.gfvSetUrl('http://' + ip + ':5000/stream');
            }
        }"""

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    nid = n.get('id', '')

    # 1. 更新 global_float_video_01
    if nid == 'global_float_video_01':
        n['format'] = NEW_GFV_FORMAT
        print('Updated: global_float_video_01 → scope-free DOM approach')

    # 2. 更新 影像串流（0b47210816be0ad4）的 format
    if nid == '0b47210816be0ad4':
        fmt = n.get('format', '')
        # 找到 scope.$watch 結尾（closing });  ) 之前插入橋接代碼
        # 目標位置：在 }); (scope.$watch 結尾) 之前
        OLD_WATCH_END = '    });\n\n})(scope);\n</script>'
        NEW_WATCH_END = (
            BRIDGE_SNIPPET +
            '\n    });\n\n})(scope);\n</script>'
        )
        if OLD_WATCH_END in fmt:
            n['format'] = fmt.replace(OLD_WATCH_END, NEW_WATCH_END)
            print('Updated: 影像串流 → added window._catIP bridge')
        else:
            print('WARNING: 影像串流 scope.$watch ending pattern not found, skip')

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
