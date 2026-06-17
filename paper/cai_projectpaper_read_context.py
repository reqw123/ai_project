<style>
.hw{font-family:'Microsoft JhengHei',sans-serif;padding:14px;background:#0d1117;border-radius:14px}
.hw-score-card{border-radius:16px;padding:24px;text-align:center;margin-bottom:16px}
.hw-score-emoji{font-size:44px;line-height:1;margin-bottom:10px}
.hw-score-num{font-size:60px;font-weight:900;line-height:1;margin-bottom:4px}
.hw-score-level{font-size:18px;font-weight:700;letter-spacing:1px}
.hw-score-sub{font-size:10px;color:rgba(255,255,255,.4);margin-top:8px}
.hw-comps{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:16px}
.hw-comp-card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px}
.hw-comp-title{font-size:10px;color:rgba(255,255,255,.45);margin-bottom:6px;font-weight:600}
.hw-comp-score{font-size:24px;font-weight:800;margin-bottom:6px}
.hw-comp-bar{height:5px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden}
.hw-comp-fill{height:100%;border-radius:99px;transition:width .6s ease}
.comp-dist{color:#85B7EB}.comp-freq{color:#F09595}.comp-na{color:rgba(255,255,255,.2)}
.fill-dist{background:linear-gradient(90deg,#1e88e5,#64b5f6)}
.fill-freq{background:linear-gradient(90deg,#e53935,#ef9a9a)}
.fill-na{background:rgba(255,255,255,.12)}
.hw-alerts-title{font-size:12px;font-weight:700;color:rgba(255,255,255,.55);margin-bottom:10px}
.hw-alert-item{display:flex;align-items:flex-start;gap:8px;padding:9px 12px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:10px;margin-bottom:6px}
.hw-alert-ico{font-size:15px;flex-shrink:0}
.hw-alert-txt{font-size:12px;color:rgba(255,255,255,.75);line-height:1.5}
.hw-no-alerts{text-align:center;padding:14px;color:rgba(255,255,255,.3);font-size:12px}
</style>
<div class="hw" ng-init="init()">
  <div class="hw-score-card" ng-style="{border:'1px solid '+riskColor(),background:'rgba('+hexRgb(riskColor())+',0.08)'}">
    <div class="hw-score-emoji">{{risk().emoji||'⏳'}}</div>
    <div class="hw-score-num" ng-style="{color:riskColor()}">{{risk().score||0}}</div>
    <div class="hw-score-level" ng-style="{color:riskColor()}">{{risk().level||'計算中...'}}</div>
    <div class="hw-score-sub">Behavior Risk Score（0=最佳 · 100=最高風險）</div>
  </div>
  <div class="hw-comps">
    <div class="hw-comp-card">
      <div class="hw-comp-title">📊 行為占比偏離</div>
      <div class="hw-comp-score comp-dist">{{comp('distribution')}}</div>
      <div class="hw-comp-bar"><div class="hw-comp-fill fill-dist" ng-style="{width:comp('distribution')+'%'}"></div></div>
    </div>
    <div class="hw-comp-card">
      <div class="hw-comp-title">📉 頻率偏離{{comp('frequency')>0?'':'(需基線)'}}</div>
      <div class="hw-comp-score comp-freq">{{comp('frequency')}}</div>
      <div class="hw-comp-bar"><div class="hw-comp-fill" ng-class="comp('frequency')>0?'fill-freq':'fill-na'" ng-style="{width:comp('frequency')+'%'}"></div></div>
    </div>
    <div class="hw-comp-card" style="opacity:.32">
      <div class="hw-comp-title">🌙 節律偏離（需
