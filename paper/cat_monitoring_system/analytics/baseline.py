"""Individual baseline computation.

Python port of the 個體化基線計算器 function node in ``cat_health_v3_flow.json``.
Formulas (mean/std/median/quartile interpolation/EWMA) are kept numerically
identical to the Node-RED original so a migration does not silently change
existing baselines; only the *consumer* (deviation.py) changes how the
computed statistics get turned into an alert.

std here is the population standard deviation (divide by n, not n-1),
matching the original JS (``Math.sqrt(sum(sq)/n)``) — the daily-history
sample is the entire population of "this cat's days we know about", not a
sample drawn from a larger population, so ddof=0 is intentional and is kept
for parity with the numbers already shown on the existing dashboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

EWMA_ALPHA = 0.15
MIN_BASELINE_DAYS_DEFAULT = 7
MAX_BASELINE_DAYS_DEFAULT = 30

# Behaviors that are genuinely continuous durations (seconds/day) vs. sparse
# non-negative event counts. This classification is what deviation.py uses
# to pick a statistical model — it is the central fix in this redesign
# (see analytics/README.md "為什麼要分兩種模型").
CONTINUOUS_METRICS = frozenset({
    "walk_time", "stop_time", "lick_time", "scratch_time",
})
COUNT_METRICS = frozenset({
    "walk_count", "stop_count", "lick_count", "scratch_count", "shake_count",
})


@dataclass
class DailyRecord:
    """One day of aggregated behavior stats, equivalent to one entry in
    Node-RED's ``v2_daily_history`` global array."""

    day: date
    monitoring_seconds: float = 0.0
    walk_time: float = 0.0
    walk_count: int = 0
    stop_time: float = 0.0
    stop_count: int = 0
    lick_time: float = 0.0
    lick_count: int = 0
    scratch_time: float = 0.0
    scratch_count: int = 0
    shake_count: int = 0
    active_time: float = 0.0
    rest_time: float = 0.0


@dataclass
class MetricStats:
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    mad: float               # median absolute deviation (raw, unscaled)
    ewma: float
    rolling_std: float
    sample_count: int


@dataclass
class Baseline:
    computed_at: str
    days_count: int
    required_days: int
    confidence: str          # "Low" / "Medium" / "High"
    excluded_dates: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)   # name -> MetricStats
    sanity_warnings: list = field(default_factory=list)
    sanity_ok: bool = True
    # Per-day raw values for COUNT_METRICS, in the exact same day-window
    # (after excluded-dates filtering + max_days truncation) that produced
    # `metrics`. deviation.py's tail-probability model needs the raw daily
    # counts, not just mean/std/median — this guarantees it always fits on
    # the same window the baseline stats came from, instead of requiring
    # every caller to re-derive a matching window by hand.
    count_histories: dict = field(default_factory=dict)   # name -> list[int]


class InsufficientDataError(Exception):
    """Raised when there are not enough valid days to compute a baseline."""

    def __init__(self, current_days: int, required_days: int):
        self.current_days = current_days
        self.required_days = required_days
        super().__init__(
            f"歷史資料不足（{current_days} 天），尚需 "
            f"{max(0, required_days - current_days)} 天才能建立基線。"
        )


def _confidence_for(n: int) -> str:
    if n < 7:
        return "Low"
    if n < 30:
        return "Medium"
    return "High"


def _quantile(sorted_vals: list, q: float) -> float:
    """Linear-interpolated quantile, matching the Node-RED implementation's
    ``sorted[floor(idx)] + frac * (sorted[ceil(idx)] - sorted[floor(idx)])``."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    idx = (n - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    frac = idx - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def compute_metric_stats(values: list) -> MetricStats:
    """Compute the full statistics bundle for one metric's daily history.

    ``mad`` is returned *unscaled* (raw median absolute deviation); scaling
    for use as a robust-z denominator happens in deviation.py, where the
    scale factor differs by intended use (see deviation.py docstring).
    """
    arr = [float(v) for v in values if v is not None]
    n = len(arr)
    if n == 0:
        now = ""
        return MetricStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    sorted_arr = sorted(arr)
    mean = sum(arr) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in arr) / n)
    median = _quantile(sorted_arr, 0.5)
    q1 = _quantile(sorted_arr, 0.25)
    q3 = _quantile(sorted_arr, 0.75)
    iqr = q3 - q1
    mad = _quantile(sorted(abs(v - median) for v in arr), 0.5)

    ewma = arr[0]
    ewma_series = [ewma]
    for v in arr[1:]:
        ewma = EWMA_ALPHA * v + (1 - EWMA_ALPHA) * ewma
        ewma_series.append(ewma)
    roll_var = sum((v - e) ** 2 for v, e in zip(arr, ewma_series)) / n
    rolling_std = math.sqrt(roll_var)

    return MetricStats(
        mean=round(mean, 2), std=round(std, 2), median=round(median, 2),
        q1=round(q1, 2), q3=round(q3, 2), iqr=round(iqr, 2),
        mad=round(mad, 2), ewma=round(ewma, 2),
        rolling_std=round(rolling_std, 2), sample_count=n,
    )


def compute_baseline(
    history: list,
    min_days: int = MIN_BASELINE_DAYS_DEFAULT,
    max_days: int = MAX_BASELINE_DAYS_DEFAULT,
    excluded_dates: Optional[list] = None,
    min_daily_monitoring_sec: float = 3600.0,
) -> Baseline:
    """Compute an individualized baseline from a list of ``DailyRecord``.

    Raises ``InsufficientDataError`` if there are fewer than ``min_days``
    valid (monitored-enough, non-excluded) days — mirrors the Node-RED
    early-return behavior, but as an explicit exception instead of a
    magic ``{"error": ...}`` payload, so callers can't accidentally treat
    a failure as a real baseline.
    """
    excluded = set(excluded_dates or [])

    valid = [
        d for d in history
        if d.monitoring_seconds >= min_daily_monitoring_sec
        and d.day.isoformat() not in excluded
    ]
    if len(valid) < min_days:
        # Fall back to including excluded days if that's the only way to
        # reach min_days (matches Node-RED: exclusion only applies if
        # enough data remains afterward).
        valid_incl_excluded = [
            d for d in history if d.monitoring_seconds >= min_daily_monitoring_sec
        ]
        if len(valid_incl_excluded) >= min_days:
            valid = valid_incl_excluded
        else:
            raise InsufficientDataError(len(valid_incl_excluded), min_days)

    valid.sort(key=lambda d: d.day)
    days = valid[-max_days:]

    metrics = {}
    for name in sorted(CONTINUOUS_METRICS | COUNT_METRICS):
        metrics[name] = compute_metric_stats([getattr(d, name) for d in days])

    count_histories = {name: [getattr(d, name) for d in days] for name in COUNT_METRICS}

    sanity_warnings = []
    lick_mean = metrics["lick_time"].mean
    scratch_mean = metrics["scratch_time"].mean
    if lick_mean > 6 * 3600:
        sanity_warnings.append(
            f"Lick Duration 個體基線（{lick_mean/60:.0f} min/day）遠超群體參考值（~60 min/day）。"
            "請確認錄影時間、辨識模型精度及資料品質。"
        )
    if 0 < lick_mean < 0.5 * 3600:
        sanity_warnings.append(
            f"Lick Duration 個體基線（{lick_mean/60:.1f} min/day）遠低於群體參考值（~60 min/day）。"
            "監控時間可能不足或模型漏偵。"
        )
    if scratch_mean > 10 * 60:
        sanity_warnings.append(
            f"Scratch Duration 個體基線（{scratch_mean/60:.0f} min/day）遠超群體參考值（~1 min/day）。"
            "請確認辨識精度。"
        )

    from datetime import datetime as _dt
    return Baseline(
        computed_at=_dt.now().isoformat(),
        days_count=len(days),
        required_days=min_days,
        confidence=_confidence_for(len(days)),
        excluded_dates=sorted(excluded),
        metrics=metrics,
        sanity_warnings=sanity_warnings,
        sanity_ok=len(sanity_warnings) == 0,
        count_histories=count_histories,
    )
