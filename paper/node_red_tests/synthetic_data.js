'use strict';

/**
 * 捏造假的每日彙整資料（欄位形狀跟 Python behavior_tracker.get_today_stats 對齊），
 * 給 simulate_baseline_demo.js（console 輸出）與 render_baseline_ui_preview.js
 * （UI 面板預覽）共用，避免同一份假資料產生邏輯維護兩份。
 */

// ── 可重現的簡易 PRNG（同一組 seed 每次跑出來的假資料都一樣，方便比對前後差異）──
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 產生一天內部一致的假資料：時長先決定，monitoring_seconds/active_time/
 * 各百分比欄位都從同一組時長反推，避免像獨立亂數那樣互相矛盾。
 */
function makeSyntheticDay(date, rng, opts = {}) {
  const monitoringHours = opts.monitoringHours ?? (6 + rng() * 4); // 6~10 小時
  const monitoring_seconds = Math.round(monitoringHours * 3600);

  // 預設值刻意調整成「舔舐佔活動時間比例落在文獻合理範圍內」（walk 遠大於 lick，
  // 讓 lick_pct_of_active 落在 sanity check 門檻（<32%）以內），純粹是為了讓
  // 情境1「正常資料」demo 真的能展示 sanity_ok=true 的情況，不是要精確重現
  // Eckstein & Hart (2000) 的 24 小時全天觀察數字（本系統只監控部分時段，
  // 跟文獻的全天觀察時間尺度本來就對不齊，這是已知落差，不是這裡要解決的）。
  const walk_time    = Math.round((opts.walkMinBase    ?? 90) * 60 * (0.7 + rng() * 0.6));
  const lick_time     = Math.round((opts.lickMinBase    ?? 35) * 60 * (0.7 + rng() * 0.6));
  const scratch_time  = Math.round((opts.scratchSecBase ?? 25) * (0.6 + rng() * 0.8));
  const shake_time    = Math.round((opts.shakeSecBase   ?? 15) * (0.5 + rng() * 1.0));
  const active_time   = walk_time + lick_time + scratch_time + shake_time;
  const stop_time     = Math.max(0, monitoring_seconds - active_time);

  const walk_count    = Math.max(1, Math.round(walk_time / 90));
  const lick_count    = Math.max(1, Math.round(lick_time / 120));
  const scratch_count = Math.max(0, Math.round(scratch_time / 30));
  const shake_count   = Math.max(0, Math.round(shake_time / 8));
  const stop_count    = Math.max(1, Math.round(stop_time / 600));

  const active_non_rest_time = walk_time + lick_time + scratch_time + shake_time;

  return {
    date,
    monitoring_seconds,
    active_time,
    walk_time, walk_count,
    lick_time, lick_count,
    scratch_time, scratch_count,
    shake_time, shake_count,
    stop_time, stop_count,
    active_ratio:          parseFloat((active_non_rest_time / monitoring_seconds * 100).toFixed(2)),
    rest_ratio:             parseFloat((stop_time / monitoring_seconds * 100).toFixed(2)),
    lick_pct_of_day:        parseFloat((lick_time / monitoring_seconds * 100).toFixed(3)),
    scratch_pct_of_day:     parseFloat((scratch_time / monitoring_seconds * 100).toFixed(3)),
    lick_pct_of_active:     active_non_rest_time > 0 ? parseFloat((lick_time / active_non_rest_time * 100).toFixed(3)) : 0,
    scratch_pct_of_active:  active_non_rest_time > 0 ? parseFloat((scratch_time / active_non_rest_time * 100).toFixed(3)) : 0,
  };
}

function makeSyntheticHistory(seed, startYmd, dayCount, opts = {}) {
  const rng = mulberry32(seed);
  const [y, m, d] = startYmd.split('-').map(Number);
  const start = new Date(y, m - 1, d);
  const days = [];
  for (let i = 0; i < dayCount; i++) {
    const dt = new Date(start.getTime() + i * 86400000);
    days.push(makeSyntheticDay(dt.toISOString().slice(0, 10), rng, opts));
  }
  return days;
}

module.exports = { mulberry32, makeSyntheticDay, makeSyntheticHistory };
