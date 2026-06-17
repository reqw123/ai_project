# -*- coding: utf-8 -*-
"""
新增一個 templateScope=global 的迷你懸浮影像節點，
讓影像可以跨 tab 顯示在右下角。
"""
import json

MAIN_PATH   = r'c:\ai_project\paper\貓咪主控.json'
NEW_ID      = 'global_float_video_01'
SENDER_IDS  = {'a8424ee7758559af', 'c60345daba38a7bd'}   # build_response + 從持久化context恢復串流

GLOBAL_FLOAT_FORMAT = """<style>
.gfv-toggle{
  position:fixed;bottom:20px;right:20px;width:46px;height:46px;
  border-radius:50%;background:rgba(13,17,23,.85);
  border:2px solid rgba(0,255,136,.45);color:#00ff88;
  font-size:22px;line-height:46px;text-align:center;
  cursor:pointer;z-index:10000;
  box-shadow:0 4px 16px rgba(0,0,0,.6);
  transition:transform .2s,border-color .2s;
  user-select:none;
}
.gfv-toggle:hover{transform:scale(1.1);border-color:rgba(0,255,136,.9);}
.gfv-panel{
  position:fixed;bottom:76px;right:20px;
  width:300px;
  background:#0d1117;
  border:1px solid rgba(0,255,136,.25);
  border-radius:12px;overflow:hidden;
  z-index:9999;
  box-shadow:0 8px 32px rgba(0,0,0,.8);
  display:none;
  flex-direction:column;
}
.gfv-panel.open{display:flex;}
.gfv-bar{
  background:rgba(0,0,0,.6);
  padding:8px 12px;
  font-size:12px;color:#00ff88;
  display:flex;justify-content:space-between;align-items:center;
  cursor:move;user-select:none;
  border-bottom:1px solid rgba(255,255,255,.06);
  flex-shrink:0;
}
.gfv-close{cursor:pointer;opacity:.6;font-size:14px;}
.gfv-close:hover{opacity:1;}
.gfv-img{width:100%;display:block;background:#000;min-height:160px;object-fit:contain;}
.gfv-offline{text-align:center;padding:28px 0;color:rgba(255,255,255,.3);font-size:12px;}
</style>

<div class="gfv-toggle" onclick="gfvToggle()" title="懸浮影像視窗">📷</div>

<div class="gfv-panel" id="gfvPanel">
  <div class="gfv-bar" id="gfvBar">
    <span>📷 即時串流</span>
    <span class="gfv-close" onclick="gfvToggle()">✕</span>
  </div>
  <img class="gfv-img" id="gfvImg"
       src="" alt=""
       onload="document.getElementById('gfvOffline').style.display='none';this.style.display='block';"
       onerror="document.getElementById('gfvOffline').style.display='block';this.style.display='none';">
  <div class="gfv-offline" id="gfvOffline" style="display:none;">📹 串流離線</div>
</div>

<script>
(function(scope){
  var _streamUrl = '';

  window.gfvToggle = function(){
    var p = document.getElementById('gfvPanel');
    p.classList.toggle('open');
  };

  // 拖曳
  var bar   = document.getElementById('gfvBar');
  var panel = document.getElementById('gfvPanel');
  var drag  = false, ox = 0, oy = 0;
  bar.addEventListener('mousedown', function(e){
    drag = true;
    ox = e.clientX - panel.getBoundingClientRect().left;
    oy = e.clientY - panel.getBoundingClientRect().top;
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    if (!drag) return;
    panel.style.left   = (e.clientX - ox) + 'px';
    panel.style.top    = (e.clientY - oy) + 'px';
    panel.style.right  = 'auto';
    panel.style.bottom = 'auto';
  });
  document.addEventListener('mouseup', function(){ drag = false; });

  function setStream(url){
    if (!url || url === _streamUrl) return;
    _streamUrl = url;
    var img = document.getElementById('gfvImg');
    if (img) img.src = url;
  }

  scope.$watch('msg', function(msg){
    if (!msg || !msg.payload) return;
    var ip = msg.payload.ip || null;
    if (ip) {
      setStream('http://' + ip + ':5000/stream');
    } else if (msg.stream_url) {
      setStream(msg.stream_url);
    }
  });
})(scope);
</script>"""

NEW_NODE = {
    "id":             NEW_ID,
    "type":           "ui_template",
    "name":           "懸浮影像（全域）",
    "order":          0,
    "templateScope":  "global",
    "format":         GLOBAL_FLOAT_FORMAT,
    "storeOutMessages": False,
    "fwdInMessages":  True,
    "resendOnRefresh": True,
    "className":      "",
    "x":              200,
    "y":              900,
    "wires":          [[]]
}

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    nodes = json.load(f)

# 確認還沒有這個節點
existing_ids = {n.get('id') for n in nodes}
if NEW_ID in existing_ids:
    print('Node already exists, skipping add.')
else:
    nodes.append(NEW_NODE)
    print('Added: global float video node', NEW_ID)

# 把 NEW_ID 加進兩個 sender 的 wires[0]
for n in nodes:
    if n.get('id') in SENDER_IDS:
        wires = n.get('wires', [[]])
        if wires and NEW_ID not in wires[0]:
            wires[0].append(NEW_ID)
            n['wires'] = wires
            print(f'Wired: {n.get("name","?")} ({n["id"]}) → {NEW_ID}')

with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    json.dump(nodes, f, ensure_ascii=False, indent=4)
print('Written: 貓咪主控.json')
