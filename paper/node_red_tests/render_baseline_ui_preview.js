'use strict';

/**
 * 產生一個「自帶互動控制項」的靜態 HTML 頁面：可以在瀏覽器裡直接調整參數、
 * 或切換成「複製真實資料 + 倍率竄改」模式，按「重新計算」即時看到套用
 * baseline_calculator.js（跟真實 Node-RED 節點邏輯一致）之後、長得像
 * 真實 P4 面板的結果，不用每次改參數都重新跑 node 指令。
 *
 * 做法：把 baseline_calculator.js / synthetic_data.js 的原始碼整段讀進來，
 * 拿掉 Node.js 專屬的 module.exports（瀏覽器沒有 module 物件），直接內嵌
 * 進 <script> 標籤，讓這個 HTML 檔案本身就是一個完全自足、不用連線、
 * 不用啟動任何 server 的小工具（雙擊打開即可，file:// 也能跑，因為沒有
 * 任何 fetch()／XHR 去讀外部檔案，真實資料快照也是產生當下就內嵌進去的
 * 靜態複本）。
 *
 * 用法：
 *   （可選）python fetch_real_history.py   → 產生 real_history_snapshot.json
 *   node render_baseline_ui_preview.js     → 產生 baseline_ui_preview.html
 *   用瀏覽器打開 baseline_ui_preview.html，表單預設值就是目前的假資料預設
 *   參數，改完按「重新計算」即可，全部在瀏覽器裡完成，不用再重跑這支腳本。
 *
 * ⚠️ 唯讀：real_history_snapshot.json 是 fetch_real_history.py 唯讀抓取的
 * 複本，這支腳本只讀它，不會寫回任何 Node-RED 的真實 context 檔案。
 */

const fs = require('fs');
const path = require('path');

const CALC_JS_PATH = path.join(__dirname, 'baseline_calculator.js');
const SYNTH_JS_PATH = path.join(__dirname, 'synthetic_data.js');
const REAL_SNAPSHOT_PATH = path.join(__dirname, 'real_history_snapshot.json');

function stripModuleExports(src) {
  return src.replace(/^module\.exports.*$/m, '');
}

const calcSrc = stripModuleExports(fs.readFileSync(CALC_JS_PATH, 'utf-8'));
const synthSrc = stripModuleExports(fs.readFileSync(SYNTH_JS_PATH, 'utf-8'));

let realSnapshotJson = 'null';
let realSnapshotNote = '尚未抓取真實資料 —— 執行 python fetch_real_history.py 後重跑這支腳本即可啟用「複製真實資料」模式。';
let realDayOptionsHtml = '';
if (fs.existsSync(REAL_SNAPSHOT_PATH)) {
  const snap = JSON.parse(fs.readFileSync(REAL_SNAPSHOT_PATH, 'utf-8'));
  realSnapshotJson = JSON.stringify(snap);
  realSnapshotNote = `已內嵌 ${snap.history.length} 天真實資料快照（來源：${snap.source_path}，抓取當下的唯讀複本，不會即時同步）。`;

  const excludedSet = new Set(snap.excluded_dates || []);
  const sortedDays = snap.history.slice().sort((a, b) => new Date(b.date) - new Date(a.date));
  realDayOptionsHtml = sortedDays.map((d) => {
    const hours = d.monitoring_seconds ? (d.monitoring_seconds / 3600).toFixed(1) : '?';
    const isExcluded = excludedSet.has(d.date);
    const label = `${d.date}　監控${hours}h${isExcluded ? '　[原本已排除]' : ''}`;
    // 預設勾選：原本沒被標記排除的日期
    const selectedAttr = isExcluded ? '' : 'selected';
    return `<option value="${d.date}" ${selectedAttr}>${label}</option>`;
  }).join('\n        ');
}

// 跟真實 P4 面板 scope.rows 完全一致（見 cat_health_v3_flow.json「P4 個體
// 基線面板」節點 format 欄位裡的 scope.rows 定義）
const ROWS = [
  { ico: '😸', lbl: '舔舐次數',        key: 'lick_count' },
  { ico: '😸', lbl: '舔舐時間',        key: 'lick_time' },
  { ico: '🐾', lbl: '搔抓次數',        key: 'scratch_count' },
  { ico: '🐾', lbl: '搔抓時間',        key: 'scratch_time' },
  { ico: '🔄', lbl: '甩頭次數',        key: 'shake_count' },
  { ico: '🚶', lbl: '行走次數',        key: 'walk_count' },
  { ico: '🚶', lbl: '行走時間',        key: 'walk_time' },
  { ico: '😴', lbl: '靜止時間',        key: 'stop_time' },
  { ico: '😸', lbl: '舔舐佔活動時間%', key: 'lick_pct_active' },
  { ico: '🐾', lbl: '搔抓佔活動時間%', key: 'scratch_pct_active' },
  { ico: '🏃', lbl: '活動時間佔比%',   key: 'active_ratio' },
];

// 照抄自 cat_health_v3_flow.json「P4 個體基線面板」節點的 <style>（只留這裡用得到的 class）
const P4_CSS = `
.p4{font-family:'Microsoft JhengHei',sans-serif;padding:14px;background:#0d1117;border-radius:14px}
.p4-hdr{display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap}
.p4-hdr-box{flex:1;min-width:110px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:14px;text-align:center}
.p4-hdr-val{font-size:22px;font-weight:800;color:#fff;margin-bottom:4px}
.p4-hdr-lbl{font-size:11px;color:rgba(255,255,255,.4)}
.p4-table-title{font-size:13px;font-weight:700;color:rgba(255,255,255,.55);margin-bottom:10px;letter-spacing:.5px}
.p4-table{width:100%;border-collapse:collapse;margin-bottom:20px}
.p4-table th{background:rgba(255,255,255,.06);color:rgba(255,255,255,.5);font-size:11px;font-weight:600;padding:8px 10px;text-align:right;border:1px solid rgba(255,255,255,.07)}
.p4-table th:first-child{text-align:left}
.p4-table td{padding:8px 10px;border:1px solid rgba(255,255,255,.05);color:rgba(255,255,255,.8);font-size:12px;text-align:right}
.p4-table td:first-child{color:#fff;font-weight:600;text-align:left}
.p4-table tr:nth-child(even) td{background:rgba(255,255,255,.02)}
.p4-no-bl{text-align:center;padding:40px;color:rgba(255,255,255,.3);font-size:13px}
.p4-warn-box{background:rgba(255,167,38,.1);border:1px solid rgba(255,167,38,.3);border-radius:8px;padding:10px 12px;margin-bottom:16px;font-size:12px;color:#ffa726;line-height:1.5}
.p4-ok-box{font-size:12px;color:rgba(76,175,80,.8);margin-bottom:16px}
`;

const FORM_CSS = `
body{background:#161b22;margin:0;padding:24px;font-family:sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{color:#fff;font-size:18px}
.hint{color:#888;font-size:12px;line-height:1.6;margin-bottom:18px}
.panel{background:#1c2128;border:1px solid #30363d;border-radius:12px;padding:16px;margin-bottom:20px}
.panel h2{color:#e6edf3;font-size:14px;margin:0 0 12px}
.row{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:12px;align-items:flex-end}
.field{display:flex;flex-direction:column;gap:4px;min-width:120px}
.field label{color:#9198a1;font-size:11px}
.field input[type=number]{background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:6px 8px;font-size:13px;width:100px}
.field.wide input[type=number]{width:100%}
.radio-group{display:flex;gap:16px;align-items:center}
.radio-group label{color:#e6edf3;font-size:13px;display:flex;align-items:center;gap:6px;cursor:pointer}
button{cursor:pointer;font-family:inherit;border:none;border-radius:8px;padding:9px 18px;font-size:13px;font-weight:700}
.btn-primary{background:linear-gradient(135deg,#42a5f5,#1565c0);color:#fff}
.btn-preset{background:rgba(255,255,255,.08);color:#e6edf3;margin-right:8px;font-weight:600}
.btn-preset:hover{background:rgba(255,255,255,.15)}
.btn-small{background:rgba(255,255,255,.08);color:#e6edf3;padding:5px 10px;font-size:11px;font-weight:600;margin-right:6px}
.btn-small:hover{background:rgba(255,255,255,.15)}
.section-note{color:#6e7681;font-size:11px;margin-top:8px;line-height:1.5}
.field select[multiple]{background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:4px;font-size:12px;width:100%;min-width:280px}
.field select[multiple] option{padding:3px 6px}
.field select[multiple] option:checked{background:#1565c0;color:#fff}
.day-count-hint{color:#9198a1;font-size:11px;margin-top:4px}
`;

const html = `<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>個體化基線 — 互動式假資料 UI 預覽</title>
<style>${P4_CSS}${FORM_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>個體化基線計算器 — 互動式假資料 UI 預覽</h1>
  <p class="hint">
    全部在瀏覽器裡執行（唯一個 HTML 檔，沒有伺服器、沒有網路連線），改參數後按「重新計算」即可，不用重跑任何指令。<br>
    CSS 樣式照抄自 cat_health_v3_flow.json 的「P4 個體基線面板」節點，方便跟真實 Dashboard 開兩個分頁肉眼比對。<br>
    ${realSnapshotNote}
  </p>

  <div class="panel">
    <h2>1. 資料來源</h2>
    <div class="row">
      <div class="radio-group">
        <label><input type="radio" name="source" value="synthetic" checked> 純虛構資料</label>
        <label><input type="radio" name="source" value="real" id="realRadio"> 複製真實資料 + 倍率竄改</label>
      </div>
    </div>

    <div id="synthFields">
      <div class="row">
        <div class="field"><label>天數 day_count</label><input type="number" id="dayCount" value="15" min="1"></div>
        <div class="field"><label>亂數種子 seed</label><input type="number" id="seed" value="20260718"></div>
        <div class="field"><label>起始日期</label><input type="text" id="startDate" value="2026-06-01" style="width:130px"></div>
      </div>
      <div class="row">
        <div class="field"><label>walk_min_base（分鐘）</label><input type="number" id="walkMinBase" value="90"></div>
        <div class="field"><label>lick_min_base（分鐘）</label><input type="number" id="lickMinBase" value="35"></div>
        <div class="field"><label>scratch_sec_base（秒）</label><input type="number" id="scratchSecBase" value="25"></div>
        <div class="field"><label>shake_sec_base（秒）</label><input type="number" id="shakeSecBase" value="15"></div>
      </div>
      <div class="row">
        <button class="btn-preset" type="button" onclick="applyPreset('normal')">快速套用：情境1 正常資料</button>
        <button class="btn-preset" type="button" onclick="applyPreset('excess_lick')">快速套用：情境2 過度舔舐</button>
        <button class="btn-preset" type="button" onclick="applyPreset('insufficient')">快速套用：情境3 天數不足</button>
      </div>
    </div>

    <div id="realFields" style="display:none">
      <div class="row">
        <div class="field wide" style="flex:1">
          <label>個別挑選要納入基線計算的日期（Ctrl/Cmd + 點選可複選；「原本已排除」的日期預設不勾選）</label>
          <select multiple id="realDaySelect" size="8" onchange="updateDaySelectHint()">
        ${realDayOptionsHtml}
          </select>
          <div class="day-count-hint" id="daySelectHint"></div>
        </div>
      </div>
      <div class="row">
        <button class="btn-small" type="button" onclick="selectRealDays('all')">全選</button>
        <button class="btn-small" type="button" onclick="selectRealDays('none')">全不選</button>
        <button class="btn-small" type="button" onclick="selectRealDays('non-excluded')">只選未排除日期</button>
      </div>
      <div class="row">
        <div class="field"><label>行走時間倍率 ×</label><input type="number" id="walkMult" value="1.0" step="0.1"></div>
        <div class="field"><label>舔舐時間倍率 ×</label><input type="number" id="lickMult" value="1.0" step="0.1"></div>
        <div class="field"><label>搔抓時間倍率 ×</label><input type="number" id="scratchMult" value="1.0" step="0.1"></div>
        <div class="field"><label>甩頭時間倍率 ×</label><input type="number" id="shakeMult" value="1.0" step="0.1"></div>
      </div>
      <p class="section-note">倍率預設 1.0 = 完全不竄改、原封不動顯示真實資料。想模擬「舔舐突然變 5 倍」這種異常，把「舔舐時間倍率」改成 5 即可（會同步等比例調整次數）。</p>
    </div>

    <div class="row" style="margin-top:6px">
      <div class="field"><label>baseline_days 門檻</label><input type="number" id="baselineDays" value="7" min="1"></div>
      <button class="btn-primary" type="button" onclick="recompute()">重新計算 ▶</button>
    </div>
  </div>

  <div class="panel">
    <h2>2. 計算結果</h2>
    <div id="result"></div>
  </div>
</div>

<script>
${calcSrc}
${synthSrc}

const REAL_SNAPSHOT = ${realSnapshotJson};

const ROWS = ${JSON.stringify(ROWS)};

const PRESETS = {
  normal:       { walkMinBase: 90, lickMinBase: 35,  scratchSecBase: 25, shakeSecBase: 15, dayCount: 15, seed: 20260718, startDate: '2026-06-01' },
  excess_lick:  { walkMinBase: 90, lickMinBase: 480, scratchSecBase: 25, shakeSecBase: 15, dayCount: 7,  seed: 999,      startDate: '2026-07-01' },
  insufficient: { walkMinBase: 90, lickMinBase: 35,  scratchSecBase: 25, shakeSecBase: 15, dayCount: 4,  seed: 42,       startDate: '2026-08-01' },
};

function applyPreset(name) {
  const p = PRESETS[name];
  document.getElementById('dayCount').value = p.dayCount;
  document.getElementById('seed').value = p.seed;
  document.getElementById('startDate').value = p.startDate;
  document.getElementById('walkMinBase').value = p.walkMinBase;
  document.getElementById('lickMinBase').value = p.lickMinBase;
  document.getElementById('scratchSecBase').value = p.scratchSecBase;
  document.getElementById('shakeSecBase').value = p.shakeSecBase;
  document.querySelector('input[name=source][value=synthetic]').checked = true;
  toggleSourceFields();
  recompute();
}

function toggleSourceFields() {
  const isReal = document.querySelector('input[name=source]:checked').value === 'real';
  document.getElementById('synthFields').style.display = isReal ? 'none' : '';
  document.getElementById('realFields').style.display = isReal ? '' : 'none';
}
document.querySelectorAll('input[name=source]').forEach((el) => el.addEventListener('change', toggleSourceFields));

function tamperDay(day, mult) {
  const d = Object.assign({}, day);
  d.walk_time = Math.round((d.walk_time || 0) * mult.walk);
  d.walk_count = Math.round((d.walk_count || 0) * mult.walk);
  d.lick_time = Math.round((d.lick_time || 0) * mult.lick);
  d.lick_count = Math.round((d.lick_count || 0) * mult.lick);
  d.scratch_time = Math.round((d.scratch_time || 0) * mult.scratch);
  d.scratch_count = Math.round((d.scratch_count || 0) * mult.scratch);
  d.shake_time = Math.round((d.shake_time || 0) * mult.shake);
  d.shake_count = Math.round((d.shake_count || 0) * mult.shake);
  return d;
}

function selectRealDays(mode) {
  const select = document.getElementById('realDaySelect');
  if (!select || !REAL_SNAPSHOT) return;
  const excludedSet = new Set(REAL_SNAPSHOT.excluded_dates || []);
  Array.from(select.options).forEach((opt) => {
    if (mode === 'all') opt.selected = true;
    else if (mode === 'none') opt.selected = false;
    else if (mode === 'non-excluded') opt.selected = !excludedSet.has(opt.value);
  });
  updateDaySelectHint();
  recompute();
}

function updateDaySelectHint() {
  const select = document.getElementById('realDaySelect');
  const hint = document.getElementById('daySelectHint');
  if (!select || !hint) return;
  const n = Array.from(select.selectedOptions).length;
  hint.textContent = '目前已勾選 ' + n + ' / ' + select.options.length + ' 天';
}

function buildHistory() {
  const source = document.querySelector('input[name=source]:checked').value;
  const baselineDays = parseInt(document.getElementById('baselineDays').value, 10) || 7;

  if (source === 'real') {
    if (!REAL_SNAPSHOT) {
      alert('尚未內嵌真實資料快照，請先在終端機執行 python fetch_real_history.py 再重新產生這個 HTML。');
      return null;
    }
    const mult = {
      walk: parseFloat(document.getElementById('walkMult').value) || 1,
      lick: parseFloat(document.getElementById('lickMult').value) || 1,
      scratch: parseFloat(document.getElementById('scratchMult').value) || 1,
      shake: parseFloat(document.getElementById('shakeMult').value) || 1,
    };
    const selectedDates = new Set(Array.from(document.getElementById('realDaySelect').selectedOptions).map((o) => o.value));
    const picked = REAL_SNAPSHOT.history.filter((d) => selectedDates.has(d.date));
    const history = picked.map((d) => tamperDay(d, mult));
    return { history, settings: { baseline_days: baselineDays }, excludedDates: [] };
  }

  const dayCount = parseInt(document.getElementById('dayCount').value, 10) || 15;
  const seed = parseInt(document.getElementById('seed').value, 10) || 1;
  const startDate = document.getElementById('startDate').value || '2026-06-01';
  const history = makeSyntheticHistory(seed, startDate, dayCount, {
    walkMinBase: parseFloat(document.getElementById('walkMinBase').value),
    lickMinBase: parseFloat(document.getElementById('lickMinBase').value),
    scratchSecBase: parseFloat(document.getElementById('scratchSecBase').value),
    shakeSecBase: parseFloat(document.getElementById('shakeSecBase').value),
  });
  return { history, settings: { baseline_days: baselineDays }, excludedDates: [] };
}

function fmt(v) { return (v === null || v === undefined) ? '--' : v; }

function renderPanel(result) {
  if (!result.ok) {
    return '<div class="p4"><div class="p4-no-bl">❌ ' + result.error + '（現有 ' + result.current_days + ' 天 / 需要 ' + result.required_days + ' 天）</div></div>';
  }
  const b = result.baseline;
  const daysColor = b.days_count >= 14 ? '#81c784' : (b.days_count >= 7 ? '#ffa726' : '#ef9a9a');
  const rowsHtml = ROWS.map((r) => {
    const m = b.metrics[r.key];
    if (!m) return '';
    return '<tr><td>' + r.ico + ' ' + r.lbl + '</td><td>' + fmt(m.mean) + '</td><td>' + fmt(m.std) + '</td><td>' + fmt(m.median) + '</td><td>' + fmt(m.iqr) + '</td><td>' + (m.sample_count != null ? m.sample_count : b.days_count) + '</td></tr>';
  }).join('');
  const sanityHtml = b.sanity_warnings.length > 0
    ? '<div class="p4-warn-box">⚠️ 基線合理性警告：<br>' + b.sanity_warnings.map((w) => '• ' + w).join('<br>') + '</div>'
    : '<div class="p4-ok-box">✓ 合理性檢查通過，基線數值落在文獻參考範圍內</div>';

  return '<div class="p4">' +
    '<div class="p4-hdr">' +
      '<div class="p4-hdr-box"><div class="p4-hdr-val" style="color:' + daysColor + '">' + b.days_count + '</div><div class="p4-hdr-lbl">基線使用天數</div></div>' +
      '<div class="p4-hdr-box"><div class="p4-hdr-val" style="font-size:14px">' + b.computed_at.slice(0, 10) + '</div><div class="p4-hdr-lbl">計算日期</div></div>' +
      '<div class="p4-hdr-box"><div class="p4-hdr-val">' + b.confidence + '</div><div class="p4-hdr-lbl">信心等級</div></div>' +
      '<div class="p4-hdr-box"><div class="p4-hdr-val">✅ 已建立</div><div class="p4-hdr-lbl">基線狀態</div></div>' +
    '</div>' +
    sanityHtml +
    '<div class="p4-table-title">📐 個體化基線指標（Mean / Std / Median / IQR）</div>' +
    '<table class="p4-table"><tr><th>指標</th><th>Mean</th><th>Std</th><th>Median</th><th>IQR</th><th>樣本數</th></tr>' + rowsHtml + '</table>' +
    '<p style="color:#666;font-size:11px;margin-top:-10px">註：真實 P4 面板還會多顯示「今日值」與「Z-score/σ」兩欄，那是「行為偏差融合引擎」拿今天的即時資料去跟這份基線比較才算得出來的，不屬於 computeBaseline() 本身，這裡先略過。</p>' +
  '</div>';
}

function recompute() {
  const built = buildHistory();
  if (!built) return;
  const result = computeBaseline({ history: built.history, settings: built.settings, excludedDates: built.excludedDates || [] });
  document.getElementById('result').innerHTML = renderPanel(result);
}

if (!REAL_SNAPSHOT) {
  document.getElementById('realRadio').disabled = true;
} else {
  updateDaySelectHint();
}

window.addEventListener('DOMContentLoaded', recompute);
</script>
</body>
</html>`;

const outPath = path.join(__dirname, 'baseline_ui_preview.html');
fs.writeFileSync(outPath, html, 'utf-8');
console.log('已產生: ' + outPath);
console.log(realSnapshotNote);
console.log('用瀏覽器打開這個檔案即可互動調整參數，改完按頁面上的「重新計算」按鈕。');
