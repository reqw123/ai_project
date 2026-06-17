# -*- coding: utf-8 -*-
"""
修正 global_float_video_01：加入容器塌縮邏輯，避免佔用 layout 空間
"""
import json

MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'

NEW_GFV_FORMAT = r"""<div id="gfv-nr-anchor" style="display:none;height:0;width:0;overflow:hidden;position:absolute;"></div>
<script>
(function(){

  // ── 1. 塌縮自己的容器（讓它不佔 layout 空間）──────────────────
  function collapseContainer(){
    var anchor = document.getElementById('gfv-nr-anchor');
    if (!anchor) return;
    var p = anchor.parentElement;
    while (p && p !== document.body && p !== document.documentElement) {
      var kids = p.children, allOurs = true;
      for (var i = 0; i < kids.length; i++) {
        var id = kids[i].id, tag = kids[i].tagName;
        if (id !== 'gfv-nr-anchor' &&
            id !== 'gfv_float_btn' &&
            id !== 'gfv_float_panel' &&
            tag !== 'SCRIPT' && tag !== 'STYLE') {
          allOurs = false; break;
        }
      }
      if (!allOurs) break;
      p.style.height    = '0';
      p.style.minHeight = '0';
      p.style.maxHeight = '0';
      p.style.overflow  = 'hidden';
      p.style.padding   = '0';
      p.style.margin    = '0';
      p.style.flexShrink = '0';
      p.style.flexGrow   = '0';
      p = p.parentElement;
    }
  }
  collapseContainer();
  setTimeout(collapseContainer, 100); // 補一次保底

  // ── 2. 讓 local template 推送串流 URL ────────────────────────
  var _url = '';
  window.gfvSetUrl = function(url){
    if (!url || url === _url) return;
    _url = url;
    var img = document.getElementById('gfv_float_img');
    if (img){ img.style.opacity = 0; img.src = url; }
  };

  // ── 3. 防重複建立 ────────────────────────────────────────────
  if (document.getElementById('gfv_float_btn')) return;

  // ── 4. 注入 CSS ──────────────────────────────────────────────
  var st = document.createElement('style');
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
      'border-radius:12px;overflow:hidden;z-index:2147483646;' +
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
    '#gfv_offline{text-align:center;padding:20px 0;color:rgba(255,255,255,.3);font-size:12px;}';
  document.head.appendChild(st);

  // ── 5. 建立按鈕 ──────────────────────────────────────────────
  var btn = document.createElement('div');
  btn.id = 'gfv_float_btn';
  btn.title = '懸浮影像視窗';
  btn.textContent = '📷';
  document.body.appendChild(btn);

  // ── 6. 建立面板 ──────────────────────────────────────────────
  var pnl = document.createElement('div');
  pnl.id = 'gfv_float_panel';
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

  // ── 7. 開關 ──────────────────────────────────────────────────
  function gfvToggle(){
    var p = document.getElementById('gfv_float_panel');
    if (!p) return;
    var opening = p.style.display !== 'flex';
    p.style.display = opening ? 'flex' : 'none';
    if (opening && !_url && window._catIP) {
      window.gfvSetUrl('http://' + window._catIP + ':5000/stream');
    }
  }
  btn.addEventListener('click', gfvToggle);
  setTimeout(function(){
    var c = document.getElementById('gfv_close');
    if (c) c.addEventListener('click', gfvToggle);
  }, 0);

  // ── 8. 拖曳 ──────────────────────────────────────────────────
  var drag = false, ox = 0, oy = 0;
  var bar = document.getElementById('gfv_bar');
  if (bar) {
    bar.addEventListener('mousedown', function(e){
      drag = true;
      var r = pnl.getBoundingClientRect();
      ox = e.clientX - r.left; oy = e.clientY - r.top;
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

  // ── 9. 全螢幕 ────────────────────────────────────────────────
  pnl.addEventListener('dblclick', function(){
    if (!document.fullscreenElement) {
      pnl.requestFullscreen && pnl.requestFullscreen();
    } else {
      document.exitFullscreen && document.exitFullscreen();
    }
  });

})();
</script>"""

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

for n in nodes:
    if n.get('id') == 'global_float_video_01':
        n['format'] = NEW_GFV_FORMAT
        print('Updated: global_float_video_01 → container collapse + drag + fullscreen')
        break

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written.')
