# -*- coding: utf-8 -*-
"""
将 影像串流 改為純橋接（不佔 layout 空間），所有 UI 移至 global_float_video_01。
1. 影像串流 → 橋接腳本（CSS 隱藏 group + scope.$watch 橋接 IP）
2. group_stream_cat → className=gfv-hidden-bridge-group, disp=false
3. global_float_video_01 → 360x240 懸浮視窗 + 重載/全螢幕/拖曳
"""
import json

MAIN_PATH = r'c:\ai_project\paper\貓咪主控.json'

# ── 1. 影像串流 純橋接格式 ─────────────────────────────────────────────
BRIDGE_FORMAT = (
    '<style>.gfv-hidden-bridge-group{display:none!important;'
    'height:0!important;overflow:hidden!important;'
    'margin:0!important;padding:0!important;}</style>\n'
    '<script>\n'
    '(function(scope){\n'
    '    scope.$watch(\'msg\', function(msg){\n'
    '        if (!msg) return;\n'
    '        var ip = (msg.payload && msg.payload.ip) ? msg.payload.ip : null;\n'
    '        var url = \'\';\n'
    '        if (ip) {\n'
    '            url = \'http://\' + ip + \':5000/stream\';\n'
    '            window._catIP = ip;\n'
    '        } else if (msg.stream_url) {\n'
    '            url = msg.stream_url;\n'
    '        }\n'
    '        if (!url) return;\n'
    '        if (typeof window.gfvSetUrl === \'function\') {\n'
    '            window.gfvSetUrl(url);\n'
    '        } else {\n'
    '            setTimeout(function(){\n'
    '                if (typeof window.gfvSetUrl === \'function\') window.gfvSetUrl(url);\n'
    '            }, 1000);\n'
    '        }\n'
    '    });\n'
    '})(scope);\n'
    '</script>'
)

# ── 2. global_float_video_01 完整格式（360x240，重載/全螢幕/拖曳）──────────
GFV_FORMAT = r"""<div id="gfv-nr-anchor" style="display:none;height:0;width:0;overflow:hidden;position:absolute;"></div>
<script>
(function(){

  // ── 塌縮自己的容器（讓它不佔 layout 空間）────────────────────────────
  function collapseContainer(){
    var anchor = document.getElementById('gfv-nr-anchor');
    if (!anchor) return;
    var p = anchor.parentElement;
    while (p && p !== document.body && p !== document.documentElement) {
      var kids = p.children, allOurs = true;
      for (var i = 0; i < kids.length; i++) {
        var id = kids[i].id, tag = kids[i].tagName;
        if (id !== 'gfv-nr-anchor' && id !== 'gfv_float_btn' && id !== 'gfv_float_panel' &&
            tag !== 'SCRIPT' && tag !== 'STYLE') { allOurs = false; break; }
      }
      if (!allOurs) break;
      p.style.height='0'; p.style.minHeight='0'; p.style.maxHeight='0';
      p.style.overflow='hidden'; p.style.padding='0'; p.style.margin='0';
      p.style.flexShrink='0'; p.style.flexGrow='0';
      p = p.parentElement;
    }
  }
  collapseContainer();
  setTimeout(collapseContainer, 100);

  // ── URL 橋接（供 local template 呼叫）──────────────────────────────
  var _url = '';
  window.gfvSetUrl = function(url){
    if (!url || url === _url) return;
    _url = url;
    var img = document.getElementById('gfv_float_img');
    if (img){ img.style.opacity=0; img.src=url; }
  };

  if (document.getElementById('gfv_float_btn')) return;

  // ── CSS ──────────────────────────────────────────────────────────
  var st = document.createElement('style');
  st.textContent =
    /* 📷 圓形按鈕 */
    '#gfv_float_btn{position:fixed;bottom:20px;right:20px;' +
      'width:46px;height:46px;border-radius:50%;' +
      'background:rgba(13,17,23,.92);border:2px solid rgba(0,255,136,.5);' +
      'color:#00ff88;font-size:22px;line-height:44px;text-align:center;' +
      'cursor:pointer;z-index:2147483647;' +
      'box-shadow:0 4px 16px rgba(0,0,0,.7);' +
      'user-select:none;transition:transform .15s,border-color .15s;}' +
    '#gfv_float_btn:hover{transform:scale(1.12);border-color:#00ff88;}' +
    /* 浮動面板 */
    '#gfv_float_panel{position:fixed;bottom:76px;right:20px;' +
      'width:360px;height:240px;' +
      'background:#0d1117;border:1px solid rgba(0,255,136,.28);' +
      'border-radius:12px;overflow:hidden;z-index:2147483646;' +
      'box-shadow:0 8px 32px rgba(0,0,0,.85);' +
      'display:none;flex-direction:column;}' +
    /* 標題列 */
    '#gfv_bar{height:34px;flex-shrink:0;background:rgba(0,0,0,.72);' +
      'padding:0 10px;font-size:12px;font-weight:600;color:#00ff88;' +
      'display:flex;justify-content:space-between;align-items:center;' +
      'cursor:move;user-select:none;' +
      'border-bottom:1px solid rgba(255,255,255,.07);}' +
    '#gfv_btns{display:flex;gap:5px;align-items:center;}' +
    /* 圖示按鈕 */
    '.gfv_ib{width:22px;height:22px;border-radius:4px;' +
      'display:flex;align-items:center;justify-content:center;' +
      'cursor:pointer;opacity:.55;font-size:13px;' +
      'border:1px solid rgba(255,255,255,.12);' +
      'background:rgba(255,255,255,.06);color:#fff;' +
      'transition:opacity .15s,background .15s;flex-shrink:0;}' +
    '.gfv_ib:hover{opacity:1;background:rgba(255,255,255,.15);}' +
    /* 影像 */
    '#gfv_float_img{flex:1;width:100%;min-height:0;' +
      'display:block;background:#000;object-fit:contain;' +
      'transition:opacity .25s;}' +
    /* 離線提示 */
    '#gfv_offline{flex:1;display:none;align-items:center;justify-content:center;' +
      'flex-direction:column;gap:6px;' +
      'color:rgba(255,255,255,.3);font-size:12px;}' +
    /* 全螢幕 */
    '#gfv_float_panel:-webkit-full-screen,' +
    '#gfv_float_panel:fullscreen{' +
      'width:100vw!important;height:100vh!important;' +
      'top:0!important;left:0!important;' +
      'right:auto!important;bottom:auto!important;border-radius:0;}';
  document.head.appendChild(st);

  // ── 📷 按鈕 ──────────────────────────────────────────────────────
  var btn = document.createElement('div');
  btn.id='gfv_float_btn'; btn.title='串流視窗'; btn.textContent='📷';
  document.body.appendChild(btn);

  // ── 面板 HTML ────────────────────────────────────────────────────
  var pnl = document.createElement('div');
  pnl.id = 'gfv_float_panel';
  pnl.innerHTML =
    '<div id="gfv_bar">' +
      '<span>📷 即時串流</span>' +
      '<div id="gfv_btns">' +
        '<div class="gfv_ib" id="gfv_rl" title="重新載入">↺</div>' +
        '<div class="gfv_ib" id="gfv_fs" title="全螢幕">⛶</div>' +
        '<div class="gfv_ib" id="gfv_cl" title="關閉">✕</div>' +
      '</div>' +
    '</div>' +
    '<img id="gfv_float_img" src="" alt="" style="opacity:0;" ' +
      'onload="this.style.opacity=1;' +
             'var o=document.getElementById(\'gfv_offline\');if(o)o.style.display=\'none\';" ' +
      'onerror="this.style.opacity=0;' +
              'var o=document.getElementById(\'gfv_offline\');if(o)o.style.display=\'flex\';">' +
    '<div id="gfv_offline">📹<br>串流離線</div>';
  document.body.appendChild(pnl);

  // ── 開關 ─────────────────────────────────────────────────────────
  function gfvOpen(){
    pnl.style.display='flex';
    if (!_url && window._catIP) window.gfvSetUrl('http://'+window._catIP+':5000/stream');
  }
  function gfvClose(){ pnl.style.display='none'; }
  btn.addEventListener('click', function(){ pnl.style.display==='flex'?gfvClose():gfvOpen(); });

  // ── 面板按鈕事件 ─────────────────────────────────────────────────
  setTimeout(function(){
    /* 關閉 */
    var cl=document.getElementById('gfv_cl');
    if (cl) cl.addEventListener('click', function(e){ e.stopPropagation(); gfvClose(); });

    /* 全螢幕 */
    var fs=document.getElementById('gfv_fs');
    if (fs) fs.addEventListener('click', function(e){
      e.stopPropagation();
      if (!document.fullscreenElement){ pnl.requestFullscreen&&pnl.requestFullscreen(); }
      else { document.exitFullscreen&&document.exitFullscreen(); }
    });

    /* 重新載入 */
    var rl=document.getElementById('gfv_rl');
    if (rl) rl.addEventListener('click', function(e){
      e.stopPropagation();
      if (!_url) return;
      var img=document.getElementById('gfv_float_img');
      if (img){ img.style.opacity=0; img.src='';
        setTimeout(function(){ img.src=_url+'?t='+Date.now(); }, 200); }
    });
  }, 0);

  // ── 拖曳（只允許拖標題列，不含按鈕）────────────────────────────────
  var drag=false, ox=0, oy=0;
  var bar=document.getElementById('gfv_bar');
  if (bar) {
    bar.addEventListener('mousedown', function(e){
      if (e.target.classList.contains('gfv_ib')) return;
      drag=true;
      var r=pnl.getBoundingClientRect();
      ox=e.clientX-r.left; oy=e.clientY-r.top;
      e.preventDefault();
    });
  }
  document.addEventListener('mousemove', function(e){
    if (!drag) return;
    pnl.style.left=(e.clientX-ox)+'px';
    pnl.style.top=(e.clientY-oy)+'px';
    pnl.style.right='auto'; pnl.style.bottom='auto';
  });
  document.addEventListener('mouseup', function(){ drag=false; });

  // ── 雙擊影像 → 全螢幕 ────────────────────────────────────────────
  pnl.addEventListener('dblclick', function(e){
    if (e.target.classList.contains('gfv_ib')) return;
    if (!document.fullscreenElement){ pnl.requestFullscreen&&pnl.requestFullscreen(); }
    else { document.exitFullscreen&&document.exitFullscreen(); }
  });

})();
</script>"""


with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

changes = []
for n in nodes:
    nid = n.get('id', '')

    # 影像串流 → 純橋接
    if nid == '0b47210816be0ad4':
        n['format'] = BRIDGE_FORMAT
        n['height'] = '1'
        changes.append('stream widget -> bridge only, height=1')

    # group_stream_cat → 隱藏
    if nid == 'group_stream_cat':
        n['className'] = 'gfv-hidden-bridge-group'
        n['disp'] = False
        changes.append('group_stream_cat -> hidden via CSS class')

    # global_float_video_01 → 完整 360x240 浮動視窗
    if nid == 'global_float_video_01':
        n['format'] = GFV_FORMAT
        changes.append('global_float_video_01 -> 360x240 + reload/fullscreen/drag')

print('Changes:')
for c in changes:
    print(' ', c)

if len(changes) != 3:
    print('WARNING: expected 3 changes, got', len(changes))

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
