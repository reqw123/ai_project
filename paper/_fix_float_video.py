# -*- coding: utf-8 -*-
"""重寫 global_float_video_01：改用 JS 直接 appendChild 到 body，繞過 Angular 容器限制"""
import json

MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'
NODE_ID   = 'global_float_video_01'

NEW_FORMAT = """<script>
(function(scope){
  var BTN_ID   = 'gfv_float_btn';
  var PANEL_ID = 'gfv_float_panel';
  var IMG_ID   = 'gfv_float_img';
  var _url     = '';

  // 防止頁面切換時重複建立
  if (!document.getElementById(BTN_ID)) {

    // ── 注入 CSS ──────────────────────────────────────────────
    var st = document.createElement('style');
    st.id  = 'gfv-css';
    st.textContent =
      '#gfv_float_btn{' +
        'position:fixed;bottom:20px;right:20px;' +
        'width:46px;height:46px;border-radius:50%;' +
        'background:rgba(13,17,23,.92);' +
        'border:2px solid rgba(0,255,136,.5);' +
        'color:#00ff88;font-size:22px;line-height:44px;' +
        'text-align:center;cursor:pointer;' +
        'z-index:2147483647;' +
        'box-shadow:0 4px 16px rgba(0,0,0,.7);' +
        'user-select:none;transition:transform .15s,border-color .15s;}' +
      '#gfv_float_btn:hover{transform:scale(1.12);border-color:#00ff88;}' +
      '#gfv_float_panel{' +
        'position:fixed;bottom:76px;right:20px;width:300px;' +
        'background:#0d1117;border:1px solid rgba(0,255,136,.28);' +
        'border-radius:12px;overflow:hidden;' +
        'z-index:2147483646;' +
        'box-shadow:0 8px 32px rgba(0,0,0,.85);' +
        'display:none;flex-direction:column;}' +
      '#gfv_bar{' +
        'background:rgba(0,0,0,.65);padding:8px 12px;' +
        'font-size:12px;color:#00ff88;' +
        'display:flex;justify-content:space-between;align-items:center;' +
        'cursor:move;user-select:none;' +
        'border-bottom:1px solid rgba(255,255,255,.07);flex-shrink:0;}' +
      '#gfv_close{cursor:pointer;opacity:.55;font-size:14px;}' +
      '#gfv_close:hover{opacity:1;}' +
      '#gfv_float_img{width:100%;display:block;background:#000;min-height:168px;object-fit:contain;}' +
      '#gfv_offline{text-align:center;padding:30px 0;color:rgba(255,255,255,.3);font-size:12px;}';
    document.head.appendChild(st);

    // ── 建立按鈕 ──────────────────────────────────────────────
    var btn = document.createElement('div');
    btn.id    = BTN_ID;
    btn.title = '懸浮影像視窗';
    btn.textContent = '\\uD83D\\uDCF7'; // 📷
    document.body.appendChild(btn);

    // ── 建立面板 ──────────────────────────────────────────────
    var pnl = document.createElement('div');
    pnl.id = PANEL_ID;
    pnl.innerHTML =
      '<div id="gfv_bar">' +
        '<span>\\uD83D\\uDCF7 即時串流</span>' +
        '<span id="gfv_close">&#10005;</span>' +
      '</div>' +
      '<img id="' + IMG_ID + '" src="" alt="" ' +
        'onload="' +
          'this.style.opacity=1;' +
          'var o=document.getElementById(\\'gfv_offline\\');if(o)o.style.display=\\'none\\';" ' +
        'onerror="' +
          'this.style.opacity=0;' +
          'var o=document.getElementById(\\'gfv_offline\\');if(o)o.style.display=\\'block\\';">' +
      '<div id="gfv_offline" style="display:none;">\\uD83D\\uDCF9 串流離線</div>';
    document.body.appendChild(pnl);

    // ── 開關 ──────────────────────────────────────────────────
    function gfvToggle(){
      var p = document.getElementById(PANEL_ID);
      if (!p) return;
      p.style.display = (p.style.display === 'flex') ? 'none' : 'flex';
    }
    btn.addEventListener('click', gfvToggle);
    var closeBtn = document.getElementById('gfv_close');
    if (closeBtn) closeBtn.addEventListener('click', gfvToggle);

    // ── 拖曳 ──────────────────────────────────────────────────
    var bar  = document.getElementById('gfv_bar');
    var drag = false, ox = 0, oy = 0;
    if (bar) {
      bar.addEventListener('mousedown', function(e){
        drag = true;
        var r = pnl.getBoundingClientRect();
        ox = e.clientX - r.left;
        oy = e.clientY - r.top;
        e.preventDefault();
      });
    }
    document.addEventListener('mousemove', function(e){
      if (!drag) return;
      pnl.style.left   = (e.clientX - ox) + 'px';
      pnl.style.top    = (e.clientY - oy) + 'px';
      pnl.style.right  = 'auto';
      pnl.style.bottom = 'auto';
    });
    document.addEventListener('mouseup', function(){ drag = false; });
  }

  // ── 設定串流 URL ──────────────────────────────────────────────
  function gfvSetUrl(url){
    if (!url || url === _url) return;
    _url = url;
    var img = document.getElementById(IMG_ID);
    if (img){ img.style.opacity = 0; img.src = url; }
  }

  // ── 接收 Node-RED 訊息 ────────────────────────────────────────
  scope.$watch('msg', function(msg){
    if (!msg || !msg.payload) return;
    var ip = msg.payload.ip || null;
    if (ip) {
      gfvSetUrl('http://' + ip + ':5000/stream');
    } else if (msg.stream_url) {
      gfvSetUrl(msg.stream_url);
    }
  });

})(scope);
</script>"""

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') == NODE_ID:
        n['format'] = NEW_FORMAT
        print('Updated: global_float_video_01 → DOM-based body append')
        break

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
