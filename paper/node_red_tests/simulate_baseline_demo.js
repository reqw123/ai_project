'use strict';

/**
 * 模擬情境腳本（console 版）—— 純粹拿來「肉眼看數字」，不是斷言測試。
 *
 * 捏造一份跟真實每日彙整（Python behavior_tracker.get_today_stats）欄位形狀
 * 一致、但數值虛構的多天歷史資料，跑過 computeBaseline() 之後把結果印出來，
 * 方便跟 P4「個體基線面板」上會顯示的數字互相對照，確認算出來的東西合理。
 *
 * 不碰 Node-RED、不碰 global context、不寫檔案，跑完就結束，對正式系統
 * 零影響。要看「長得像 P4 面板」的版本，跑 render_baseline_ui_preview.js。
 *
 * 用法：node simulate_baseline_demo.js
 */

const { computeBaseline } = require('./baseline_calculator');
const { makeSyntheticHistory } = require('./synthetic_data');

function printBaselineResult(title, result) {
  console.log('\n' + '='.repeat(70));
  console.log(title);
  console.log('='.repeat(70));

  if (!result.ok) {
    console.log(`[FAIL] ${result.error}  現有天數=${result.current_days}  需要天數=${result.required_days}`);
    return;
  }

  const b = result.baseline;
  console.log(`天數: ${b.days_count}  信心等級: ${b.confidence}  sanity_ok: ${b.sanity_ok}`);
  if (b.sanity_warnings.length > 0) {
    console.log('Sanity Warnings:');
    for (const w of b.sanity_warnings) console.log('  ⚠ ' + w);
  }

  console.log('\n指標統計（mean / std / median / iqr）:');
  const rows = Object.entries(b.metrics);
  const nameW = Math.max(...rows.map(([k]) => k.length)) + 2;
  for (const [key, m] of rows) {
    const sc = m.sample_count != null ? `  (n=${m.sample_count})` : '';
    console.log(
      '  ' + key.padEnd(nameW) +
      `mean=${String(m.mean).padStart(8)}  std=${String(m.std).padStart(8)}  ` +
      `median=${String(m.median).padStart(8)}  iqr=${String(m.iqr).padStart(8)}${sc}`
    );
  }

  if (result.warnings.length > 0) {
    console.log('\n[node.warn 訊息紀錄]');
    for (const w of result.warnings) console.log('  · ' + w.split('\n')[0]);
  }
}

// ── 情境 1：21 天「正常」貓咪行為資料 ──────────────────────────────────
{
  const days = makeSyntheticHistory(20260718, '2026-06-01', 21);
  const result = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  printBaselineResult('情境 1：21 天正常資料（predicted confidence=Medium, sanity_ok=true）', result);
}

// ── 情境 2：7 天「過度舔舐」異常資料，應該觸發 sanity warning ─────────────
{
  const days = makeSyntheticHistory(999, '2026-07-01', 7, { lickMinBase: 8 * 60 }); // 遠超正常的 50 分鐘
  const result = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  printBaselineResult('情境 2：7 天過度舔舐異常資料（predicted sanity_ok=false）', result);
}

// ── 情境 3：只有 4 天資料，天數不足 ────────────────────────────────────
{
  const days = makeSyntheticHistory(42, '2026-08-01', 4);
  const result = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  printBaselineResult('情境 3：只有 4 天資料（predicted ok=false）', result);
}
