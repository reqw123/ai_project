'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { computeBaseline, calcFullStats, calcConfidence } = require('./baseline_calculator');

function mkDay(date, overrides = {}) {
  return Object.assign(
    {
      date,
      monitoring_seconds: 3600 * 8,
      walk_time: 0, walk_count: 0,
      lick_time: 0, lick_count: 0,
      scratch_time: 0, scratch_count: 0,
      shake_time: 0, shake_count: 0,
      stop_time: 0, stop_count: 0,
      active_time: 0,
    },
    overrides,
  );
}

// ── calcFullStats：純數學公式，用手算得出的答案對照 ─────────────────────
test('calcFullStats：mean/median/std 與手算結果一致', () => {
  const s = calcFullStats([10, 20, 30, 40, 50, 60, 70]);
  assert.equal(s.mean, 40);
  assert.equal(s.median, 40);
  assert.equal(s.sample_count, 7);
  // population std of [10..70 step10] = sqrt(2800/7) = 20
  assert.equal(s.std, 20);
});

test('calcFullStats：null/undefined 會被過濾掉，不會當成 0 污染統計', () => {
  const s = calcFullStats([10, null, 20, undefined, 30]);
  assert.equal(s.sample_count, 3);
  assert.equal(s.mean, 20);
});

test('calcFullStats：空陣列回傳全 0，不丟例外', () => {
  const s = calcFullStats([]);
  assert.equal(s.sample_count, 0);
  assert.equal(s.mean, 0);
  assert.equal(s.std, 0);
});

// ── calcConfidence：三段門檻的邊界值 ─────────────────────────────────────
test('calcConfidence：Low/Medium/High 邊界（6/7/29/30 天）', () => {
  assert.equal(calcConfidence(6), 'Low');
  assert.equal(calcConfidence(7), 'Medium');
  assert.equal(calcConfidence(29), 'Medium');
  assert.equal(calcConfidence(30), 'High');
});

// ── computeBaseline：天數不足時要直接回傳 error，不能算出假基線 ──────────
test('computeBaseline：history 天數 < baseline_days 時回傳「基線資料不足」', () => {
  const history = [mkDay('2026-01-01'), mkDay('2026-01-02')];
  const r = computeBaseline({ history, settings: { baseline_days: 7 } });
  assert.equal(r.ok, false);
  assert.equal(r.error, '基線資料不足');
  assert.equal(r.current_days, 2);
  assert.equal(r.required_days, 7);
  assert.ok(r.population_reference); // 這個分支要附群體參考值供 UI 顯示
});

test('computeBaseline：monitoring_seconds 不足 1 小時的天數會被排除在有效天數外', () => {
  const days = [];
  for (let i = 1; i <= 7; i++) days.push(mkDay(`2026-02-0${i}`));
  days.push(mkDay('2026-02-08', { monitoring_seconds: 600 })); // 只監控10分鐘，應被濾掉
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.days_count, 7); // 8 天存進 history，但只有 7 天算進基線
});

test('computeBaseline：恰好 7 天有效資料 → 成功、confidence=Medium、數值精確', () => {
  const days = [];
  for (let i = 1; i <= 7; i++) days.push(mkDay(`2026-01-0${i}`, { lick_time: 3000, lick_count: 5 }));
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.days_count, 7);
  assert.equal(r.baseline.confidence, 'Medium'); // calcConfidence(7) === 'Medium'
  assert.equal(r.baseline.metrics.lick_time.mean, 3000);
  assert.equal(r.baseline.metrics.lick_time.std, 0);
  assert.equal(r.baseline.sanity_ok, true);
});

test('computeBaseline：超過 30 天有效資料時只取最近 30 天，confidence=High', () => {
  const days = [];
  for (let i = 0; i < 35; i++) {
    const d = new Date(2026, 0, 1 + i);
    days.push(mkDay(d.toISOString().slice(0, 10)));
  }
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.days_count, 30); // MAX_BASELINE_DAYS = max(7,30) = 30
  assert.equal(r.baseline.confidence, 'High');
});

test('computeBaseline：sanity check — lick 基線遠超群體參考值時要標記 sanity_ok=false', () => {
  const days = [];
  for (let i = 1; i <= 7; i++) days.push(mkDay(`2026-03-0${i}`, { lick_time: 8 * 3600 })); // 8hr/day，遠超群體 ~1hr
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.sanity_ok, false);
  assert.ok(r.baseline.sanity_warnings.some((w) => w.includes('Lick Duration')));
});

test('computeBaseline：舊格式歷史天數缺百分比欄位時，layer2_pct 的 sample_count 只算新格式天數', () => {
  const days = [];
  for (let i = 1; i <= 5; i++) days.push(mkDay(`2026-05-0${i}`)); // 舊格式：沒有 lick_pct_of_day 欄位
  for (let i = 6; i <= 8; i++) days.push(mkDay(`2026-05-0${i}`, { lick_pct_of_day: 3.5 })); // 新格式
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 } });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.metrics.lick_pct_day.sample_count, 3);
});

test('computeBaseline：排除日期使有效天數不足時，由近到遠自動遞補取消排除', () => {
  const days = [];
  for (let i = 1; i <= 9; i++) days.push(mkDay(`2026-04-0${i}`));
  // 排除 3 天（04-07~09），剩 6 天 < baseline_days=7 → 應自動從「最近的被排除日」開始遞補 1 天
  const excluded = ['2026-04-07', '2026-04-08', '2026-04-09'];
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 }, excludedDates: excluded });
  assert.equal(r.ok, true);
  assert.equal(r.baseline.days_count, 7);
  assert.deepEqual(r.autoRestoredDates, ['2026-04-09']);
  assert.deepEqual(r.excludedDatesAfterAutoRestore, ['2026-04-07', '2026-04-08']);
});

test('computeBaseline：排除日期遞補後仍不足門檻時回傳「基線有效天不足」', () => {
  // 7 天存進 history（剛好過第一關的天數門檻），但第7天監控時數不足 1 小時會被濾掉，
  // 實際只剩 6 天「有效」；再排除其中 2 天，就算把排除的都遞補回來也只有 6 天 < 7，
  // 驗證「有效天數的天花板本來就不夠」而非「排除造成」的情境會被正確擋下。
  const days = [];
  for (let i = 1; i <= 6; i++) days.push(mkDay(`2026-06-0${i}`));
  days.push(mkDay('2026-06-07', { monitoring_seconds: 600 }));
  const excluded = ['2026-06-01', '2026-06-02'];
  const r = computeBaseline({ history: days, settings: { baseline_days: 7 }, excludedDates: excluded });
  assert.equal(r.ok, false);
  assert.equal(r.error, '基線有效天不足');
  assert.equal(r.current_days, 6);
});
