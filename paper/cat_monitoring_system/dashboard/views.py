"""
個體化基線儀表板：Flask Blueprint。

兩個路由：
  GET /dashboard/baseline    唯讀展示頁（HTML，原生字串，比照 routes.py 既有慣例，
                             不引入 Jinja2）
  GET /api/deviation/latest  回傳 dashboard/cache.py 存的最新一次計算結果

只讀 cache.py 的資料，不直接呼叫 analytics/ 的計算函式（計算仍然只在
routes.py 的 POST /api/deviation 發生），也不讀 Node-RED 的 global.json——
資料完全來自「行為統計累積器」既有的自動觸發鏈路，見
analytics/README.md 與 analytics_deviation_bridge.json。
"""

from __future__ import annotations

from flask import Blueprint, Response, jsonify

from config import BaselineDashboardConfig

from . import cache

bp = Blueprint("baseline_dashboard", __name__)


@bp.route("/api/deviation/latest", methods=["GET"])
def api_deviation_latest():
    """回傳快取的最新結果；尚未有資料時回傳 {"status":"not_yet_computed"}，
    HTTP 一律 200——前端不需要另外處理 fetch 的錯誤分支。"""
    return jsonify(cache.get_latest())


@bp.route("/dashboard/baseline", methods=["GET"])
def baseline_dashboard_page():
    return Response(_render_page(), mimetype="text/html")


def _render_page() -> str:
    poll_ms = int(BaselineDashboardConfig.POLL_INTERVAL_SEC * 1000)
    return _PAGE_TEMPLATE.replace("__POLL_MS__", str(poll_ms))


# 視覺風格沿用 cat_health_v3_flow.json「P4 個體基線面板」「健康警示」的深色
# 主題與配色邏輯（見 貓咪個體化基線.md、NODE_RED_FUNCTIONS.md）。內容範圍是
# analytics/ 實際會回傳的東西——刻意不畫「時段涵蓋率」這類欄位，因為那是
# Node-RED 舊引擎才有算的 periods 資料，Python 版 baseline.py 沒有這個概念
# （見 analytics/README.md「還沒做的事」），畫出來會是騙人的假資料。
_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>個體化基線儀表板（Python 分析引擎）</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{font-family:'Microsoft JhengHei',sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:20px}
  h1{font-size:18px;font-weight:700;margin:0 0 4px}
  .sub{font-size:12px;color:rgba(255,255,255,.4);margin-bottom:18px}
  .wrap{max-width:720px;margin:0 auto}
  .card{background:#161b22;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:18px;margin-bottom:16px}
  .risk-card{text-align:center;padding:24px}
  .risk-emoji{font-size:40px;line-height:1;margin-bottom:8px}
  .risk-score{font-size:52px;font-weight:900;line-height:1;margin-bottom:4px}
  .risk-level{font-size:16px;font-weight:700}
  .row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.06);font-size:13px}
  .row:last-child{border-bottom:none}
  .lbl{color:rgba(255,255,255,.55)}
  .val{font-weight:700}
  .warn{color:#ffa726;font-size:12px;padding:6px 0}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:rgba(255,255,255,.4);font-weight:600;padding:6px 4px;border-bottom:1px solid rgba(255,255,255,.1)}
  td{padding:6px 4px;border-bottom:1px solid rgba(255,255,255,.05)}
  .sec-title{font-size:12px;font-weight:700;color:rgba(255,255,255,.55);margin-bottom:10px}
  .placeholder{text-align:center;color:rgba(255,255,255,.35);padding:20px;font-size:13px}
  .updated{font-size:10px;color:rgba(255,255,255,.3);text-align:right;margin-top:-8px;margin-bottom:16px}
</style>
</head>
<body>
<div class="wrap">
  <h1>🐱 個體化基線儀表板</h1>
  <div class="sub">Python 分析引擎（cat_monitoring_system/analytics）· 唯讀展示，不影響 Node-RED 舊引擎</div>
  <div class="updated" id="updated">尚未取得資料</div>

  <div class="card risk-card" id="risk-card">
    <div class="risk-emoji" id="risk-emoji">⏳</div>
    <div class="risk-score" id="risk-score">--</div>
    <div class="risk-level" id="risk-level">等待第一筆資料...</div>
  </div>

  <div class="card">
    <div class="sec-title">📋 基線狀態</div>
    <div id="baseline-status">
      <div class="placeholder">尚無資料</div>
    </div>
  </div>

  <div class="card">
    <div class="sec-title">⚠️ 偏差指標（|σ| ≥ 2.0）</div>
    <div id="deviation-alerts">
      <div class="placeholder">尚無資料</div>
    </div>
  </div>

  <div class="card">
    <div class="sec-title">📊 各指標基線統計（Mean / Median / MAD）</div>
    <table id="metrics-table">
      <thead><tr><th>指標</th><th>Mean</th><th>Median</th><th>MAD</th><th>今日</th><th>σ</th></tr></thead>
      <tbody><tr><td colspan="6" class="placeholder">尚無資料</td></tr></tbody>
    </table>
  </div>
</div>

<script>
const LEVEL_STYLE = {
  'Normal':                        {color:'#4caf50', emoji:'✅'},
  'Mild Behavioral Deviation':     {color:'#ffa726', emoji:'⚠️'},
  'Moderate Behavioral Deviation': {color:'#ff7043', emoji:'🚨'},
  'Severe Behavioral Deviation':   {color:'#f44336', emoji:'🆘'},
  'Insufficient Data':             {color:'#9e9e9e', emoji:'⏳'}
};

function fmtAgo(iso){
  if(!iso) return '';
  var sec = Math.max(0, Math.round((Date.now() - new Date(iso).getTime())/1000));
  return sec + ' 秒前更新';
}

function render(data){
  if(data.status === 'not_yet_computed'){
    document.getElementById('updated').textContent = '尚未取得資料（等待 Node-RED 觸發第一次計算）';
    return;
  }
  document.getElementById('updated').textContent = fmtAgo(data.cached_at);

  var fusion = data.fusion || {};
  var style = LEVEL_STYLE[fusion.level] || LEVEL_STYLE['Insufficient Data'];
  document.getElementById('risk-emoji').textContent = style.emoji;
  document.getElementById('risk-score').textContent = (fusion.score != null ? fusion.score : '--');
  document.getElementById('risk-score').style.color = style.color;
  document.getElementById('risk-level').textContent = fusion.level || '計算中...';
  document.getElementById('risk-level').style.color = style.color;
  document.getElementById('risk-card').style.border = '1px solid ' + style.color;

  var bl = data.baseline || {};
  var blHtml = '';
  blHtml += '<div class="row"><span class="lbl">基線使用天數</span><span class="val">' + (bl.days_count||0) + ' / ' + (bl.required_days||'--') + '</span></div>';
  blHtml += '<div class="row"><span class="lbl">信心等級</span><span class="val">' + (bl.confidence||'--') + '</span></div>';
  blHtml += '<div class="row"><span class="lbl">計算時間</span><span class="val">' + (bl.computed_at||'--') + '</span></div>';
  if(bl.sanity_warnings && bl.sanity_warnings.length > 0){
    bl.sanity_warnings.forEach(function(w){ blHtml += '<div class="warn">⚠️ ' + w + '</div>'; });
  } else if (bl.sanity_ok) {
    blHtml += '<div class="row"><span class="lbl">Sanity Check</span><span class="val" style="color:#4caf50">OK</span></div>';
  }
  document.getElementById('baseline-status').innerHTML = blHtml || '<div class="placeholder">尚無資料</div>';

  var devMetrics = (data.deviation && data.deviation.metrics) || {};
  var alertsHtml = '';
  Object.keys(devMetrics).forEach(function(k){
    var m = devMetrics[k];
    if(m.sigma_equivalent != null && Math.abs(m.sigma_equivalent) >= 2.0){
      alertsHtml += '<div class="row"><span class="lbl">' + k + '</span><span class="val">' + m.sigma_equivalent.toFixed(2) + 'σ</span></div>';
    }
  });
  document.getElementById('deviation-alerts').innerHTML = alertsHtml || '<div class="placeholder">✅ 目前無顯著偏差</div>';

  var blMetrics = (data.baseline && data.baseline.metrics) || {};
  var rows = '';
  Object.keys(blMetrics).sort().forEach(function(k){
    var stat = blMetrics[k];
    var dev = devMetrics[k] || {};
    var sigma = dev.sigma_equivalent != null ? dev.sigma_equivalent.toFixed(2) : '--';
    rows += '<tr><td>' + k + '</td><td>' + stat.mean + '</td><td>' + stat.median + '</td><td>' + stat.mad + '</td><td>' + (dev.today != null ? dev.today : '--') + '</td><td>' + sigma + '</td></tr>';
  });
  document.querySelector('#metrics-table tbody').innerHTML = rows || '<tr><td colspan="6" class="placeholder">尚無資料</td></tr>';
}

function poll(){
  fetch('/api/deviation/latest')
    .then(function(r){ return r.json(); })
    .then(render)
    .catch(function(e){ console.error('poll failed', e); });
}

poll();
setInterval(poll, __POLL_MS__);
</script>
</body>
</html>
"""
