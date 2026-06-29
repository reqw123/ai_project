"""
update_flow4.py — Round 4 fixes for 貓咪主控.json
1. Remove _firstOpened auto-mini logic (panel opens at default medium size)
2. Fix gfvMiniToggle dimension capture (use ||520/||340 fallback)
3. Remove ↺ reload button from panel title bar (keep ⊡ ⛶ ✕ only)
4. Simplify hint bar to single 📷 button (remove ⊡ ⛶ ↺ buttons)
"""
import json, sys, os

FLOW = os.path.join(os.path.dirname(__file__), '..', '貓咪主控.json')

# ── New global template ──────────────────────────────────────────────────────
GLOBAL_FMT = r"""<div id="gfv-nr-anchor" style="display:none;height:0;width:0;overflow:hidden;position:absolute;pointer-events:none;"></div>
<script>
(function(){

  /* 確保父容器不佔空間 */
  function collapseContainer(){
    var anchor=document.getElementById('gfv-nr-anchor');
    if(!anchor)return;
    var p=anchor.parentElement;
    while(p&&p!==document.body&&p!==document.documentElement){
      var kids=p.children,allOurs=true;
      for(var i=0;i<kids.length;i++){
        var id=kids[i].id,tag=kids[i].tagName;
        if(id!=='gfv-nr-anchor'&&id!=='gfv_float_btn'&&id!=='gfv_float_panel'&&
           tag!=='SCRIPT'&&tag!=='STYLE'){allOurs=false;break;}
      }
      if(!allOurs)break;
      p.style.cssText+='height:0!important;min-height:0!important;max-height:0!important;overflow:hidden!important;padding:0!important;margin:0!important;';
      p=p.parentElement;
    }
  }
  collapseContainer();
  setTimeout(collapseContainer,200);

  var _url='';

  window.gfvSetUrl=function(url){
    if(!url||url===_url)return;
    _url=url;
    var btn=document.getElementById('gfv_float_btn');
    if(btn)btn.classList.add('gfv-connected');
    var dot=document.getElementById('gfv_status_dot');
    if(dot){dot.style.background='#00ff88';dot.style.boxShadow='0 0 6px #00ff88';}
    var bdot=document.getElementById('gfv_bar_dot');
    if(bdot){bdot.style.background='#00ff88';bdot.style.boxShadow='0 0 5px #00ff88';}
    gfvShowToast();
  };

  /* 通知顯示在頂部（黑色導覽列下方）*/
  function gfvShowToast(){
    var old=document.getElementById('gfv_toast');
    if(old)old.remove();
    var t=document.createElement('div');
    t.id='gfv_toast';
    t.innerHTML=
      '<span style="font-size:15px">📷</span>'+
      '<b style="color:#00ff88;margin:0 8px">串流已就緒</b>'+
      '<span style="font-size:11px;color:rgba(255,255,255,.4)">點擊右下角開啟</span>';
    t.style.cssText=
      'position:fixed!important;top:64px!important;right:20px!important;'+
      'z-index:2147483645;background:rgba(4,14,9,.96);'+
      'border:1px solid rgba(0,255,136,.38);border-radius:8px;'+
      'padding:10px 18px;color:#e0ffe8;font-size:13px;'+
      'box-shadow:0 6px 24px rgba(0,0,0,.75);white-space:nowrap;'+
      'display:flex;align-items:center;gap:4px;'+
      'animation:gfv-toast-top .35s cubic-bezier(.22,.68,0,1.2);';
    document.body.appendChild(t);
    setTimeout(function(){
      t.style.transition='opacity .45s,transform .45s';
      t.style.opacity='0';t.style.transform='translateY(-12px)';
      setTimeout(function(){if(t.parentNode)t.remove();},450);
    },3500);
  }

  if(document.getElementById('gfv_float_btn'))return;

  /* ── CSS ── */
  var st=document.createElement('style');
  st.textContent=
    /* 圓形純圖示按鈕 */
    '#gfv_float_btn{'+
      'position:fixed!important;bottom:24px!important;right:20px!important;'+
      'width:58px!important;height:58px!important;border-radius:50%!important;'+
      'background:linear-gradient(135deg,rgba(6,10,16,.98),rgba(8,18,12,.98));'+
      'border:2px solid rgba(0,255,136,.45);'+
      'color:#fff;cursor:pointer;z-index:2147483647;'+
      'box-shadow:0 4px 18px rgba(0,0,0,.8);'+
      'user-select:none;display:flex!important;align-items:center;justify-content:center;'+
      'font-size:26px;transition:transform .2s,border-color .2s,box-shadow .2s;'+
      'animation:gfv-idle 2.8s ease-in-out infinite;}'+
    '#gfv_float_btn:hover{transform:scale(1.1)!important;border-color:#00ff88;animation:none;'+
      'box-shadow:0 6px 28px rgba(0,0,0,.85),0 0 0 4px rgba(0,255,136,.12);}'+
    '#gfv_float_btn.gfv-connected{animation:gfv-conn 3s ease-in-out infinite;}'+
    '#gfv_float_btn.gfv-connected:hover{animation:none;}'+
    '#gfv_status_dot{'+
      'position:absolute;top:4px;right:4px;'+
      'width:12px;height:12px;border-radius:50%;'+
      'background:#333;border:2px solid rgba(6,10,16,.9);'+
      'transition:background .4s,box-shadow .4s;}'+

    /* 浮動面板 - 預設中等大小 */
    '#gfv_float_panel{position:fixed!important;bottom:92px!important;right:20px!important;'+
      'width:520px;height:340px;min-width:180px;min-height:110px;'+
      'background:#070b0f;border:1px solid rgba(0,255,136,.25);border-radius:14px;'+
      'z-index:2147483646;overflow:hidden;'+
      'box-shadow:0 20px 60px rgba(0,0,0,.95),0 0 0 1px rgba(0,255,136,.07);'+
      'display:none;flex-direction:column;cursor:grab;}'+
    '#gfv_float_panel:active{cursor:grabbing;}'+
    '#gfv_float_panel.gfv-dragging{cursor:grabbing!important;}'+

    /* 標題列 */
    '#gfv_bar{height:40px;flex-shrink:0;display:flex;align-items:center;'+
      'justify-content:space-between;padding:0 6px 0 12px;user-select:none;'+
      'background:linear-gradient(90deg,rgba(0,16,9,.96),rgba(0,22,12,.96));'+
      'border-bottom:1px solid rgba(0,255,136,.1);transition:height .15s;}'+
    '#gfv_bar_l{display:flex;align-items:center;gap:8px;}'+
    '#gfv_bar_dot{width:8px;height:8px;border-radius:50%;background:#333;flex-shrink:0;'+
      'transition:background .4s,box-shadow .4s;}'+
    '#gfv_bar_title{font-size:11px;font-weight:700;color:rgba(0,255,136,.75);letter-spacing:.8px;}'+
    '#gfv_bar_r{display:flex;align-items:center;gap:4px;}'+
    '.gfv_ib{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;'+
      'justify-content:center;cursor:pointer!important;font-size:13px;'+
      'border:1px solid rgba(255,255,255,.07);background:rgba(255,255,255,.04);'+
      'color:rgba(255,255,255,.5);transition:all .15s;flex-shrink:0;user-select:none;}'+
    '.gfv_ib:hover{background:rgba(255,255,255,.14);color:#fff;border-color:rgba(255,255,255,.22);}'+
    '#gfv_cl:hover{background:rgba(220,50,50,.2);color:#ff6060;border-color:rgba(220,50,50,.3);}'+
    '#gfv_mn:hover{background:rgba(0,255,136,.14);color:#00ff88;border-color:rgba(0,255,136,.35);}'+
    '#gfv_fs:hover{background:rgba(100,180,255,.14);color:#64b4ff;border-color:rgba(100,180,255,.35);}'+

    /* 影像區：pointer-events:none → 拖曳可穿透影像 */
    '#gfv_float_img{flex:1;width:100%;min-height:0;display:block;background:#000;'+
      'object-fit:contain;transition:opacity .3s;pointer-events:none;user-select:none;draggable:false;}'+
    '#gfv_offline{flex:1;display:none;flex-direction:column;align-items:center;'+
      'justify-content:center;gap:10px;color:rgba(255,255,255,.2);font-size:12px;pointer-events:none;}'+
    '#gfv_off_ico{font-size:40px;opacity:.28;}'+

    /* SE 角縮放把手 */
    '#gfv_rz{position:absolute;bottom:0;right:0;width:20px;height:20px;'+
      'cursor:se-resize!important;z-index:6;display:flex;align-items:flex-end;'+
      'justify-content:flex-end;padding:3px;color:rgba(0,255,136,.18);font-size:12px;'+
      'line-height:1;transition:color .15s;user-select:none;}'+
    '#gfv_rz:hover{color:rgba(0,255,136,.7);}'+

    /* 各邊縮放把手（4px 細邊，不干擾拖曳）*/
    '.gfv-rh{position:absolute;z-index:5;user-select:none;}'+
    '.gfv-rh-n{top:0;left:16px;right:16px;height:4px;cursor:n-resize!important;}'+
    '.gfv-rh-s{bottom:0;left:16px;right:20px;height:4px;cursor:s-resize!important;}'+
    '.gfv-rh-w{left:0;top:16px;bottom:16px;width:4px;cursor:w-resize!important;}'+
    '.gfv-rh-e{right:0;top:40px;bottom:16px;width:4px;cursor:e-resize!important;}'+
    '.gfv-rh-nw{top:0;left:0;width:14px;height:14px;cursor:nw-resize!important;}'+
    '.gfv-rh-ne{top:0;right:0;width:14px;height:14px;cursor:ne-resize!important;}'+
    '.gfv-rh-sw{bottom:0;left:0;width:14px;height:14px;cursor:sw-resize!important;}'+

    /* 迷你模式 */
    '#gfv_float_panel.gfv-mini{width:220px!important;height:152px!important;}'+
    '#gfv_float_panel.gfv-mini #gfv_bar{height:26px;}'+
    '#gfv_float_panel.gfv-mini #gfv_bar_title{font-size:9px;}'+
    '#gfv_float_panel.gfv-mini .gfv_ib{width:22px;height:22px;font-size:11px;}'+

    /* 全螢幕 */
    '#gfv_float_panel:-webkit-full-screen,#gfv_float_panel:fullscreen{'+
      'width:100vw!important;height:100vh!important;border-radius:0;'+
      'top:0!important;left:0!important;right:auto!important;bottom:auto!important;cursor:default;}'+

    /* 動畫 */
    '@keyframes gfv-idle{0%,100%{box-shadow:0 4px 18px rgba(0,0,0,.8);}'+
      '50%{box-shadow:0 4px 18px rgba(0,0,0,.8),0 0 0 5px rgba(0,255,136,.09);}}'+
    '@keyframes gfv-conn{0%,100%{box-shadow:0 4px 18px rgba(0,0,0,.8),0 0 10px rgba(0,255,136,.25);}'+
      '50%{box-shadow:0 4px 18px rgba(0,0,0,.8),0 0 20px rgba(0,255,136,.5),0 0 0 5px rgba(0,255,136,.12);}}'+
    '@keyframes gfv-toast-top{from{opacity:0;transform:translateY(-10px);}to{opacity:1;transform:none;}}';
  document.head.appendChild(st);

  /* ── 圓形純圖示按鈕（無文字）── */
  var btn=document.createElement('div');
  btn.id='gfv_float_btn';
  btn.title='點擊開啟即時串流';
  btn.innerHTML='📷<div id="gfv_status_dot"></div>';
  btn.style.cssText='position:fixed!important;bottom:24px!important;right:20px!important;';
  document.body.appendChild(btn);

  /* ── 浮動面板 ── */
  var pnl=document.createElement('div');
  pnl.id='gfv_float_panel';
  pnl.innerHTML=
    '<div id="gfv_bar">'+
      '<div id="gfv_bar_l"><div id="gfv_bar_dot"></div>'+
        '<span id="gfv_bar_title">📷 LIVE STREAM</span></div>'+
      '<div id="gfv_bar_r">'+
        '<div class="gfv_ib" id="gfv_mn" title="迷你模式">⊡</div>'+
        '<div class="gfv_ib" id="gfv_fs" title="全螢幕（雙擊影像）">⛶</div>'+
        '<div class="gfv_ib" id="gfv_cl" title="關閉">✕</div>'+
      '</div>'+
    '</div>'+
    '<img id="gfv_float_img" src="" alt="" style="opacity:0;" draggable="false">'+
    '<div id="gfv_offline"><div id="gfv_off_ico">📹</div><div>串流尚未連線</div></div>'+
    '<div class="gfv-rh gfv-rh-n" id="gfv_rz_n"></div>'+
    '<div class="gfv-rh gfv-rh-s" id="gfv_rz_s"></div>'+
    '<div class="gfv-rh gfv-rh-w" id="gfv_rz_w"></div>'+
    '<div class="gfv-rh gfv-rh-e" id="gfv_rz_e"></div>'+
    '<div class="gfv-rh gfv-rh-nw" id="gfv_rz_nw"></div>'+
    '<div class="gfv-rh gfv-rh-ne" id="gfv_rz_ne"></div>'+
    '<div class="gfv-rh gfv-rh-sw" id="gfv_rz_sw"></div>'+
    '<div id="gfv_rz" title="縮放">⌟</div>';

  var _img=pnl.querySelector('#gfv_float_img');
  if(_img){
    _img.addEventListener('load',function(){
      this.style.opacity=1;
      var o=pnl.querySelector('#gfv_offline');if(o)o.style.display='none';
      var d=pnl.querySelector('#gfv_bar_dot');
      if(d){d.style.background='#00ff88';d.style.boxShadow='0 0 5px #00ff88';}
    });
    _img.addEventListener('error',function(){
      this.style.opacity=0;
      var o=pnl.querySelector('#gfv_offline');if(o)o.style.display='flex';
    });
  }
  document.body.appendChild(pnl);

  /* ── 開 / 關 ── */
  function gfvOpen(){
    pnl.style.display='flex';
    pnl.style.cursor='grab';
    var url=_url||(window._catIP?'http://'+window._catIP+':5000/stream':'');
    if(url){
      if(!_url)_url=url;
      var img=document.getElementById('gfv_float_img');
      if(img){img.style.opacity=0;img.src=url;}
    }
  }
  function gfvClose(){
    pnl.style.display='none';
    var img=document.getElementById('gfv_float_img');
    if(img){img.src='';img.style.opacity=0;}
    var d=document.getElementById('gfv_bar_dot');
    if(d){d.style.background='#333';d.style.boxShadow='none';}
  }
  window.gfvToggle=function(){pnl.style.display==='flex'?gfvClose():gfvOpen();};
  btn.addEventListener('click',function(){window.gfvToggle();});

  /* ── 迷你模式 ── */
  var _mini=false,_mnW,_mnH,_mnL,_mnT,_mnR,_mnB;
  function gfvMiniToggle(){
    if(!_mini){
      var r=pnl.getBoundingClientRect();
      /* 若面板尚未完全渲染 getBoundingClientRect 回傳 0，使用預設尺寸 */
      _mnW=r.width||520;_mnH=r.height||340;
      _mnL=pnl.style.left;_mnT=pnl.style.top;
      _mnR=pnl.style.right;_mnB=pnl.style.bottom;
      pnl.classList.add('gfv-mini');
      _mini=true;
      var mn=document.getElementById('gfv_mn');
      if(mn){mn.textContent='⬜';mn.title='還原視窗';}
    } else {
      pnl.classList.remove('gfv-mini');
      pnl.style.width=(_mnW||520)+'px';
      pnl.style.height=(_mnH||340)+'px';
      pnl.style.left=_mnL||'auto';
      pnl.style.top=_mnT||'auto';
      pnl.style.right=_mnR||'20px';
      pnl.style.bottom=_mnB||'92px';
      _mini=false;
      var mn=document.getElementById('gfv_mn');
      if(mn){mn.textContent='⊡';mn.title='迷你模式';}
    }
  }

  /* ── 對外暴露（hint bar 按鈕使用）── */
  window.gfvMiniOpen=function(){
    if(pnl.style.display!=='flex'){
      gfvOpen();
      /* 給面板足夠時間渲染後再切換迷你模式 */
      setTimeout(function(){if(!_mini)gfvMiniToggle();},200);
    } else {
      gfvMiniToggle();
    }
  };
  window.gfvOpenFs=function(){
    if(pnl.style.display!=='flex')gfvOpen();
    setTimeout(function(){pnl.requestFullscreen&&pnl.requestFullscreen();},80);
  };
  window.gfvReload=function(){
    if(!_url)return;
    var img=document.getElementById('gfv_float_img');
    if(img){img.style.opacity=0;img.src='';setTimeout(function(){img.src=_url+'?t='+Date.now();},150);}
  };

  /* ── 面板按鈕綁定 ── */
  setTimeout(function(){
    var el;
    el=document.getElementById('gfv_cl');
    if(el)el.addEventListener('click',function(e){e.stopPropagation();gfvClose();});
    el=document.getElementById('gfv_fs');
    if(el)el.addEventListener('click',function(e){
      e.stopPropagation();
      if(!document.fullscreenElement)pnl.requestFullscreen&&pnl.requestFullscreen();
      else document.exitFullscreen&&document.exitFullscreen();
    });
    el=document.getElementById('gfv_mn');
    if(el)el.addEventListener('click',function(e){e.stopPropagation();gfvMiniToggle();});
  },0);

  /* ── 全區域拖曳（移動視窗，不改變大小）── */
  var drag=false,ox=0,oy=0;
  pnl.addEventListener('mousedown',function(e){
    if(e.target.closest&&e.target.closest('.gfv_ib'))return;  /* 按鈕：跳過 */
    if(e.target.closest&&e.target.closest('.gfv-rh'))return;  /* 縮放把手：跳過 */
    if(e.target.id==='gfv_rz')return;                         /* SE角：跳過 */
    drag=true;
    var r=pnl.getBoundingClientRect();
    ox=e.clientX-r.left;oy=e.clientY-r.top;
    pnl.classList.add('gfv-dragging');
    e.preventDefault();
  });

  /* ── 縮放（SE角）── */
  var rzDrag=false,rzW0,rzH0,rzX0,rzY0;
  var rzEl=document.getElementById('gfv_rz');
  if(rzEl){
    rzEl.addEventListener('mousedown',function(e){
      rzDrag=true;
      var r=pnl.getBoundingClientRect();
      rzW0=r.width;rzH0=r.height;rzX0=e.clientX;rzY0=e.clientY;
      e.preventDefault();e.stopPropagation();
    });
  }

  /* ── 各邊縮放 ── */
  var rzEdge=null,rzR0;
  function bindRzEdge(id,edge){
    var el=document.getElementById(id);
    if(!el)return;
    el.addEventListener('mousedown',function(e){
      rzEdge=edge;rzR0=pnl.getBoundingClientRect();
      rzW0=rzR0.width;rzH0=rzR0.height;rzX0=e.clientX;rzY0=e.clientY;
      e.preventDefault();e.stopPropagation();
    });
  }
  setTimeout(function(){
    bindRzEdge('gfv_rz_n','n');bindRzEdge('gfv_rz_s','s');
    bindRzEdge('gfv_rz_w','w');bindRzEdge('gfv_rz_e','e');
    bindRzEdge('gfv_rz_nw','nw');bindRzEdge('gfv_rz_ne','ne');
    bindRzEdge('gfv_rz_sw','sw');
  },0);

  document.addEventListener('mousemove',function(e){
    if(drag){
      /* 移動視窗（保持原本大小不變）*/
      pnl.style.left=(e.clientX-ox)+'px';
      pnl.style.top=(e.clientY-oy)+'px';
      pnl.style.right='auto';pnl.style.bottom='auto';
    }
    if(rzDrag){
      pnl.style.width=Math.max(180,rzW0+(e.clientX-rzX0))+'px';
      pnl.style.height=Math.max(110,rzH0+(e.clientY-rzY0))+'px';
    }
    if(rzEdge){
      var dx=e.clientX-rzX0,dy=e.clientY-rzY0;
      if(rzEdge==='s'||rzEdge==='sw'){pnl.style.height=Math.max(110,rzH0+dy)+'px';}
      if(rzEdge==='n'||rzEdge==='nw'||rzEdge==='ne'){
        var newH=Math.max(110,rzH0-dy);
        pnl.style.height=newH+'px';
        pnl.style.top=(rzR0.top+rzH0-newH)+'px';
        pnl.style.bottom='auto';
      }
      if(rzEdge==='e'||rzEdge==='ne'){pnl.style.width=Math.max(180,rzW0+dx)+'px';}
      if(rzEdge==='w'||rzEdge==='nw'||rzEdge==='sw'){
        var newW=Math.max(180,rzW0-dx);
        pnl.style.width=newW+'px';
        pnl.style.left=(rzR0.left+rzW0-newW)+'px';
        pnl.style.right='auto';
      }
    }
  });

  document.addEventListener('mouseup',function(){
    drag=false;rzDrag=false;rzEdge=null;
    pnl.classList.remove('gfv-dragging');
    if(pnl.style.display==='flex')pnl.style.cursor='grab';
  });

  /* ── 雙擊切換全螢幕 ── */
  pnl.addEventListener('dblclick',function(e){
    if(e.target.closest&&e.target.closest('.gfv_ib'))return;
    if(e.target.id==='gfv_rz')return;
    if(!document.fullscreenElement)pnl.requestFullscreen&&pnl.requestFullscreen();
    else document.exitFullscreen&&document.exitFullscreen();
  });

  /* ── 分頁切換：暫停/恢復串流 ── */
  document.addEventListener('visibilitychange',function(){
    var img=document.getElementById('gfv_float_img');
    var p2=document.getElementById('gfv_float_panel');
    if(document.hidden){if(img)img.src='';}
    else if(p2&&p2.style.display==='flex'&&_url){
      if(img){img.style.opacity=0;img.src=_url;}
    }
  });

})();
</script>"""

# ── New hint bar template (single button) ───────────────────────────────────
LOCAL_FMT = r"""<style>
.sv-bar{height:100%;display:flex;align-items:center;gap:8px;padding:0 16px;background:linear-gradient(135deg,rgba(0,14,8,.95),rgba(2,20,12,.95));border:1px solid rgba(0,255,136,.16);border-radius:12px;overflow:hidden;position:relative;cursor:pointer;transition:border-color .22s,background .22s;box-sizing:border-box;}
.sv-bar:hover{border-color:rgba(0,255,136,.38);background:linear-gradient(135deg,rgba(0,18,10,.97),rgba(3,24,14,.97));}
.sv-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(0,255,136,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,.04) 1px,transparent 1px);background-size:28px 28px;pointer-events:none;}
.sv-dot{width:9px;height:9px;border-radius:50%;background:#00ff88;box-shadow:0 0 6px #00ff88;flex-shrink:0;animation:sv-blink 1.8s ease-in-out infinite;position:relative;z-index:1;}
@keyframes sv-blink{0%,100%{opacity:1;box-shadow:0 0 6px #00ff88;}50%{opacity:.22;box-shadow:none;}}
.sv-spacer{flex:1;}
.sv-ib{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;cursor:pointer;flex-shrink:0;background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.18);color:#fff;transition:all .18s;position:relative;z-index:1;}
.sv-ib:hover{background:rgba(0,255,136,.2);border-color:rgba(0,255,136,.7);transform:scale(1.08);box-shadow:0 0 12px rgba(0,255,136,.2);}
.sv-ib.sv-main::after{content:'';position:absolute;inset:-6px;border-radius:15px;border:1.5px solid rgba(0,255,136,.24);animation:sv-ring 2.4s ease-in-out infinite;pointer-events:none;}
@keyframes sv-ring{0%,100%{transform:scale(1);opacity:.5;}50%{transform:scale(1.15);opacity:0;}}
</style>
<div class="sv-bar" onclick="window.gfvToggle&&window.gfvToggle()">
  <div class="sv-grid"></div>
  <div class="sv-dot"></div>
  <div class="sv-spacer"></div>
  <div class="sv-ib sv-main" title="開啟 / 關閉串流視窗" onclick="event.stopPropagation();window.gfvToggle&&window.gfvToggle()">📷</div>
</div>
<script>
(function(scope){
  scope.$watch('msg',function(msg){
    if(!msg)return;
    var ip=(msg.payload&&msg.payload.ip)?msg.payload.ip:null;
    var url='';
    if(ip){url='http://'+ip+':5000/stream';window._catIP=ip;}
    else if(msg.stream_url){url=msg.stream_url;}
    if(!url)return;
    if(typeof window.gfvSetUrl==='function'){window.gfvSetUrl(url);}
    else{setTimeout(function(){if(typeof window.gfvSetUrl==='function')window.gfvSetUrl(url);},1000);}
  });
})(scope);
</script>"""

# ── Apply changes ─────────────────────────────────────────────────────────────
with open(FLOW, 'r', encoding='utf-8') as f:
    data = json.load(f)

changed = 0
for node in data:
    if not isinstance(node, dict):
        continue
    if node.get('id') == '49a0135623787e51':
        node['format'] = GLOBAL_FMT
        changed += 1
        print('Updated global template node 49a0135623787e51')
    elif node.get('id') == '6e03afc2800b13c3':
        node['format'] = LOCAL_FMT
        changed += 1
        print('Updated hint bar node 6e03afc2800b13c3')

if changed != 2:
    print(f'ERROR: expected 2 changes, got {changed}', file=sys.stderr)
    sys.exit(1)

with open(FLOW, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f'Done. {FLOW} updated successfully.')
