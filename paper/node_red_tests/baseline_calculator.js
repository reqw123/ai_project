'use strict';

/**
 * 個體化基線計算器 — 純函式版，供單元測試用。
 *
 * 對應 paper/cat_health_v3_flow.json 節點 id "66a22e0ec663d07d"
 * 「個體化基線計算器」（第3層 基線引擎）。Node-RED function node 裡的原始
 * 程式碼直接寫 global.get/global.set/node.warn，沒辦法脫離 Node-RED runtime
 * 單獨執行或測試。這裡把同一套計算邏輯（門檻、公式、排除日期自動遞補、
 * sanity check）逐行對照搬出來，只是把「讀 global context」換成「讀函式
 * 參數」、把「global.set / node.warn」換成「回傳值」，數學與判斷邏輯完全
 *沒有更動。
 *
 * 用法：改這裡 → 跑 baseline_calculator.test.js 驗證 → 測試過了之後，把
 * 對應的段落複製回 cat_health_v3_flow.json 節點 66a22e0ec663d07d 的 "func"
 * 欄位（Node-RED function node 目前不支援直接 require 本地檔案，所以是手動
 * 同步，不是自動載入）。
 */

const MIN_DAILY_SEC = 3600;
const REQUIRED_PERIODS = [];
const EWMA_ALPHA = 0.15;

// 來源：Eckstein & Hart (2000), VAScat Feline Grooming Study
const POPULATION_REFERENCE = {
  lick_duration:    { mean_sec: 3600, mean_display: '~60 min/day',  ci: '(SD 未明確報告)', source: 'Eckstein & Hart (2000)', note: '健康室內成貓，口腔梳理（Oral Grooming）' },
  scratch_duration: { mean_sec: 60,   mean_display: '~1 min/day',   ci: '(SD 未明確報告)', source: 'Eckstein & Hart (2000)', note: '健康室內成貓，搔抓梳理（Scratch Grooming）' },
  head_shake:       { mean_display: 'Rare',   source: 'Clinical observation', note: '偶爾甩頭屬正常範圍，無固定參考值' },
  walk_duration:    { mean_display: 'Varies', source: 'Individual-dependent', note: '活動量因個體、年齡、環境差異大' },
  lick_pct_day:       { mean_pct: 4,   mean_display: '~4% of full day',             source: 'Eckstein & Hart (2000)', note: '口腔梳理佔全天時間預算比例' },
  lick_pct_active:    { mean_pct: 8,   mean_display: '~8% of non-rest active time', source: 'Eckstein & Hart (2000)', note: '口腔梳理佔「非睡眠/休息時間」比例' },
  scratch_pct_day:    { mean_pct: 0.1, mean_display: '~0.1% of full day',             source: 'Eckstein & Hart (2000)', note: '搔抓梳理佔全天時間預算比例' },
  scratch_pct_active: { mean_pct: 0.2, mean_display: '~0.2% of non-rest active time', source: 'Eckstein & Hart (2000)', note: '搔抓梳理佔「非睡眠/休息時間」比例' },
  active_ratio:       { mean_pct: 43,  mean_display: '~43% of full day', source: 'Eckstein & Hart (2000)', note: '一般活動（含行走等非靜止清醒行為）佔全天比例' },
  rest_ratio:         { mean_pct: 50,  mean_display: '~50% of full day', source: 'Eckstein & Hart (2000)', note: '睡眠/休息佔全天比例' },
};

function calcConfidence(n) {
  if (n < 7) return 'Low';
  if (n < 30) return 'Medium';
  return 'High';
}

function calcFullStats(vals) {
  const arr = vals.filter((v) => v !== null && v !== undefined).map(Number);
  const n = arr.length;
  if (n === 0) {
    return { mean: 0, median: 0, std: 0, iqr: 0, q1: 0, q3: 0, sample_count: 0, rolling_mean: 0, rolling_std: 0, last_update: new Date().toISOString() };
  }
  const sorted = [...arr].sort((a, b) => a - b);
  const mean = arr.reduce((s, v) => s + v, 0) / n;
  const std = Math.sqrt(arr.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / n);
  const median = n % 2 === 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[Math.floor(n / 2)];
  const q1_idx = (n - 1) * 0.25, q3_idx = (n - 1) * 0.75;
  const q1 = sorted[Math.floor(q1_idx)] + (q1_idx % 1) * (sorted[Math.ceil(q1_idx)] - sorted[Math.floor(q1_idx)]);
  const q3 = sorted[Math.floor(q3_idx)] + (q3_idx % 1) * (sorted[Math.ceil(q3_idx)] - sorted[Math.floor(q3_idx)]);
  let ewma = arr[0];
  for (let i = 1; i < n; i++) ewma = EWMA_ALPHA * arr[i] + (1 - EWMA_ALPHA) * ewma;
  const ewmaArr = [arr[0]];
  let cur = arr[0];
  for (let i = 1; i < n; i++) { cur = EWMA_ALPHA * arr[i] + (1 - EWMA_ALPHA) * cur; ewmaArr.push(cur); }
  const rollVar = arr.map((v, i) => Math.pow(v - ewmaArr[i], 2)).reduce((s, v) => s + v, 0) / n;
  return {
    mean: parseFloat(mean.toFixed(2)),
    median: parseFloat(median.toFixed(2)),
    std: parseFloat(std.toFixed(2)),
    iqr: parseFloat((q3 - q1).toFixed(2)),
    q1: parseFloat(q1.toFixed(2)),
    q3: parseFloat(q3.toFixed(2)),
    sample_count: n,
    rolling_mean: parseFloat(ewma.toFixed(2)),
    rolling_std: parseFloat(Math.sqrt(rollVar).toFixed(2)),
    last_update: new Date().toISOString(),
  };
}

/**
 * @param {Object} input
 * @param {Array}  input.history       global.v2_daily_history 的內容
 * @param {Object} input.settings      global.v2_user_settings 的內容（baseline_days）
 * @param {Array}  input.excludedDates global.v2_excluded_dates 的內容
 * @returns 成功：{ ok:true, baseline, excludedDatesAfterAutoRestore, autoRestoredDates, warnings }
 *          失敗：{ ok:false, error, current_days, required_days, population_reference?, warnings }
 */
function computeBaseline({ history = [], settings = {}, excludedDates = [] } = {}) {
  const MIN_BASELINE_DAYS = settings.baseline_days != null ? +settings.baseline_days : 7;
  const MAX_BASELINE_DAYS = Math.max(MIN_BASELINE_DAYS, 30);
  const warnings = [];

  if (history.length < MIN_BASELINE_DAYS) {
    const alertMsg = `歷史資料不足（${history.length} 天），尚需 ${MIN_BASELINE_DAYS - history.length} 天才能建立基線。`;
    warnings.push(alertMsg);
    return {
      ok: false, error: '基線資料不足',
      current_days: history.length, required_days: MIN_BASELINE_DAYS,
      population_reference: POPULATION_REFERENCE, warnings,
    };
  }

  let validDays = history.filter((d) => {
    if ((d.monitoring_seconds || 0) < MIN_DAILY_SEC) return false;
    if (!d.periods) return true;
    return REQUIRED_PERIODS.every((p) => d.periods[p] && d.periods[p].monitoring_sec > 300);
  });

  let v2Excluded = excludedDates.slice();
  let autoRestored = [];
  if (v2Excluded.length > 0) {
    let noExcl = validDays.filter((d) => !v2Excluded.includes(d.date));
    if (noExcl.length < MIN_BASELINE_DAYS) {
      const excludedValid = validDays
        .filter((d) => v2Excluded.includes(d.date))
        .sort((a, b) => new Date(b.date) - new Date(a.date));
      let stillExcluded = v2Excluded.slice();
      for (const d of excludedValid) {
        if (noExcl.length >= MIN_BASELINE_DAYS) break;
        const idx = stillExcluded.indexOf(d.date);
        if (idx >= 0) stillExcluded.splice(idx, 1);
        noExcl.push(d);
        autoRestored.push(d.date);
      }
      if (autoRestored.length > 0) {
        v2Excluded = stillExcluded;
        warnings.push('基線天數不足新門檻，自動取消排除（由近到遠遞補）：' + autoRestored.join(', '));
      }
    }
    validDays = noExcl;
    warnings.push('排除 ' + (excludedDates.length - autoRestored.length) + ' 天後基線有效天數：' + validDays.length);
  }

  if (validDays.length < MIN_BASELINE_DAYS) {
    const alertMsg = `有效資料（${validDays.length} 天，已含自動遞補）少於基線設定天數（${MIN_BASELINE_DAYS} 天）。\n請降低基線最少天數設定，或等待更多有效資料累積。`;
    warnings.push(alertMsg);
    return {
      ok: false, error: '基線有效天不足',
      current_days: validDays.length, required_days: MIN_BASELINE_DAYS, warnings,
    };
  }

  validDays.sort((a, b) => new Date(a.date) - new Date(b.date));
  const days = validDays.slice(-MAX_BASELINE_DAYS);

  const layer1 = {
    walk_duration:     calcFullStats(days.map((d) => d.walk_time || 0)),
    walk_count:        calcFullStats(days.map((d) => d.walk_count || 0)),
    low_conf_duration: calcFullStats(days.map((d) => d.low_conf_time || 0)),
    inactive_duration: calcFullStats(days.map((d) => d.stop_time || 0)),
    stop_count:        calcFullStats(days.map((d) => d.stop_count || 0)),
    active_ratio:      calcFullStats(days.map((d) => { const mon = d.monitoring_seconds || 1; return parseFloat(((d.active_time || 0) / mon * 100).toFixed(2)); })),
    daily_monitoring:  calcFullStats(days.map((d) => d.monitoring_seconds || 0)),
  };
  const layer2 = {
    lick_frequency:       calcFullStats(days.map((d) => d.lick_count || 0)),
    lick_duration:        calcFullStats(days.map((d) => d.lick_time || 0)),
    scratch_frequency:    calcFullStats(days.map((d) => d.scratch_count || 0)),
    scratch_duration:     calcFullStats(days.map((d) => d.scratch_time || 0)),
    head_shake_frequency: calcFullStats(days.map((d) => d.shake_count || 0)),
  };
  // 刻意不補 ||0：舊格式歷史天數沒有這些欄位時，讓 calcFullStats 的
  // null/undefined 過濾自動排除，避免用假的 0 污染百分比基線。
  const layer2_pct = {
    lick_pct_of_day:       calcFullStats(days.map((d) => d.lick_pct_of_day)),
    lick_pct_of_active:    calcFullStats(days.map((d) => d.lick_pct_of_active)),
    scratch_pct_of_day:    calcFullStats(days.map((d) => d.scratch_pct_of_day)),
    scratch_pct_of_active: calcFullStats(days.map((d) => d.scratch_pct_of_active)),
    active_ratio:          calcFullStats(days.map((d) => d.active_ratio)),
    rest_ratio:            calcFullStats(days.map((d) => d.rest_ratio)),
  };
  const layer3_hourly = {};
  for (const p of ['00-06', '06-12', '12-18', '18-24']) {
    layer3_hourly[p] = calcFullStats(days.map((d) => (d.periods && d.periods[p] ? (d.periods[p].monitoring_sec || 0) : 0)));
  }
  const periodCoverage = {};
  for (const p of ['00-06', '06-12', '12-18', '18-24']) {
    const covered = days.filter((d) => d.periods && d.periods[p] && d.periods[p].monitoring_sec > 300).length;
    periodCoverage[p] = { covered, total: days.length };
  }

  const metrics = {
    lick_count:    { mean: layer2.lick_frequency.mean,       std: layer2.lick_frequency.std,       median: layer2.lick_frequency.median,       iqr: layer2.lick_frequency.iqr },
    lick_time:     { mean: layer2.lick_duration.mean,        std: layer2.lick_duration.std,        median: layer2.lick_duration.median,        iqr: layer2.lick_duration.iqr },
    scratch_count: { mean: layer2.scratch_frequency.mean,    std: layer2.scratch_frequency.std,    median: layer2.scratch_frequency.median,    iqr: layer2.scratch_frequency.iqr },
    scratch_time:  { mean: layer2.scratch_duration.mean,     std: layer2.scratch_duration.std,     median: layer2.scratch_duration.median,     iqr: layer2.scratch_duration.iqr },
    shake_count:   { mean: layer2.head_shake_frequency.mean, std: layer2.head_shake_frequency.std, median: layer2.head_shake_frequency.median, iqr: layer2.head_shake_frequency.iqr },
    shake_time:    { mean: 0, std: 0, median: 0, iqr: 0 },
    walk_time:     { mean: layer1.walk_duration.mean, std: layer1.walk_duration.std, median: layer1.walk_duration.median, iqr: layer1.walk_duration.iqr },
    walk_count:    { mean: layer1.walk_count.mean,    std: layer1.walk_count.std,    median: layer1.walk_count.median,    iqr: layer1.walk_count.iqr },
    stop_time:     { mean: layer1.inactive_duration.mean, std: layer1.inactive_duration.std, median: layer1.inactive_duration.median, iqr: layer1.inactive_duration.iqr },
    stop_count:    { mean: layer1.stop_count.mean,    std: layer1.stop_count.std,    median: layer1.stop_count.median,    iqr: layer1.stop_count.iqr },
    lick_pct_day:       { mean: layer2_pct.lick_pct_of_day.mean,       std: layer2_pct.lick_pct_of_day.std,       median: layer2_pct.lick_pct_of_day.median,       iqr: layer2_pct.lick_pct_of_day.iqr,       sample_count: layer2_pct.lick_pct_of_day.sample_count },
    lick_pct_active:    { mean: layer2_pct.lick_pct_of_active.mean,    std: layer2_pct.lick_pct_of_active.std,    median: layer2_pct.lick_pct_of_active.median,    iqr: layer2_pct.lick_pct_of_active.iqr,    sample_count: layer2_pct.lick_pct_of_active.sample_count },
    scratch_pct_day:    { mean: layer2_pct.scratch_pct_of_day.mean,    std: layer2_pct.scratch_pct_of_day.std,    median: layer2_pct.scratch_pct_of_day.median,    iqr: layer2_pct.scratch_pct_of_day.iqr,    sample_count: layer2_pct.scratch_pct_of_day.sample_count },
    scratch_pct_active: { mean: layer2_pct.scratch_pct_of_active.mean, std: layer2_pct.scratch_pct_of_active.std, median: layer2_pct.scratch_pct_of_active.median, iqr: layer2_pct.scratch_pct_of_active.iqr, sample_count: layer2_pct.scratch_pct_of_active.sample_count },
    active_ratio:       { mean: layer2_pct.active_ratio.mean,          std: layer2_pct.active_ratio.std,          median: layer2_pct.active_ratio.median,          iqr: layer2_pct.active_ratio.iqr,          sample_count: layer2_pct.active_ratio.sample_count },
    rest_ratio:         { mean: layer2_pct.rest_ratio.mean,            std: layer2_pct.rest_ratio.std,            median: layer2_pct.rest_ratio.median,            iqr: layer2_pct.rest_ratio.iqr,            sample_count: layer2_pct.rest_ratio.sample_count },
  };

  // ── Sanity Check（合理性驗證：個體基線 vs 群體文獻值，僅供提示，不影響異常判斷）──
  const sanityWarnings = [];
  const lickMean = layer2.lick_duration.mean;
  const scrMean = layer2.scratch_duration.mean;

  if (lickMean > 6 * 3600) {
    sanityWarnings.push('Lick Duration 個體基線（' + (lickMean / 60).toFixed(0) + ' min/day）遠超群體參考值（~60 min/day）。請確認錄影時間、辨識模型精度及資料品質。');
    warnings.push('[Sanity] Lick baseline too high: ' + (lickMean / 60).toFixed(0) + ' min/day vs ~60 min/day pop ref');
  }
  if (lickMean > 0 && lickMean < 0.5 * 3600) {
    sanityWarnings.push('Lick Duration 個體基線（' + (lickMean / 60).toFixed(1) + ' min/day）遠低於群體參考值（~60 min/day）。監控時間可能不足或模型漏偵。');
    warnings.push('[Sanity] Lick baseline too low: ' + (lickMean / 60).toFixed(1) + ' min/day vs ~60 min/day pop ref');
  }
  if (scrMean > 10 * 60) {
    sanityWarnings.push('Scratch Duration 個體基線（' + (scrMean / 60).toFixed(0) + ' min/day）遠超群體參考值（~1 min/day）。請確認辨識精度。');
    warnings.push('[Sanity] Scratch baseline too high: ' + (scrMean / 60).toFixed(1) + ' min/day vs ~1 min/day pop ref');
  }

  const lickPctMean = layer2_pct.lick_pct_of_active.mean;
  const scrPctMean = layer2_pct.scratch_pct_of_active.mean;
  if (layer2_pct.lick_pct_of_active.sample_count >= 3 && lickPctMean > 8 * 4) {
    sanityWarnings.push('Lick % of Active Time 個體基線（' + lickPctMean.toFixed(1) + '%）遠超群體參考值（~8%）。請確認錄影時間、辨識模型精度及資料品質。');
    warnings.push('[Sanity] Lick pct-of-active too high: ' + lickPctMean.toFixed(1) + '% vs ~8% pop ref');
  }
  if (layer2_pct.scratch_pct_of_active.sample_count >= 3 && scrPctMean > 0.2 * 10) {
    sanityWarnings.push('Scratch % of Active Time 個體基線（' + scrPctMean.toFixed(2) + '%）遠超群體參考值（~0.2%）。請確認辨識精度。');
    warnings.push('[Sanity] Scratch pct-of-active too high: ' + scrPctMean.toFixed(2) + '% vs ~0.2% pop ref');
  }

  const confidence = calcConfidence(days.length);

  const baseline = {
    computed_at: new Date().toISOString(),
    days_count: days.length,
    required_days: MIN_BASELINE_DAYS,
    confidence,
    required_periods: REQUIRED_PERIODS,
    period_coverage: periodCoverage,
    layer1, layer2, layer3_hourly, metrics,
    population_reference: POPULATION_REFERENCE,
    sanity_warnings: sanityWarnings,
    sanity_ok: sanityWarnings.length === 0,
  };

  warnings.push('基線計算完成：' + days.length + ' 天資料，信心等級：' + confidence + (sanityWarnings.length > 0 ? ' ⚠️ Sanity Warning' : ''));

  return {
    ok: true,
    baseline,
    excludedDatesAfterAutoRestore: v2Excluded,
    autoRestoredDates: autoRestored,
    warnings,
  };
}

module.exports = { computeBaseline, calcFullStats, calcConfidence, POPULATION_REFERENCE };
