"""Today-vs-baseline deviation scoring.

This is the module that actually changes behavior relative to the Node-RED
original (偏差分析引擎). See ``analytics/README.md`` for the full writeup;
short version:

Problem
-------
The existing 偏差分析引擎 computes ``z = (today - mean) / std`` for every
metric, continuous durations (walk_time, lick_time, ...) *and* sparse event
counts (scratch_count, shake_count, ...) alike. For a rare, low-count
behavior — e.g. a healthy cat scratches ~0-1 times/day — ``std`` over a
7-30 day baseline is often small (many days tied at 0 or 1), so one extra
scratch on an otherwise unremarkable day can push ``z`` past the 2.5σ "Mild"
threshold purely from a small denominator, not from an actually unusual
count. The engine *does* also compute ``robust_z`` (IQR-based), but nothing
downstream (行為偏差融合引擎) ever reads it — it's dead code today.

Fix
---
Two different statistical models, chosen by what kind of quantity the
metric actually is (see ``baseline.CONTINUOUS_METRICS`` /
``baseline.COUNT_METRICS``):

* **Continuous durations** (seconds/day): robust z-score using the Median
  Absolute Deviation (MAD), scaled by 1.4826 so it's consistent with σ
  under normality. MAD has a 50% breakdown point (vs. 25% for IQR, 0% for
  std), so a couple of unusually long grooming days in the baseline window
  can't drag the "normal" spread up the way a mean/std baseline can.

* **Sparse event counts** (integers, mean well under ~5/day): a Poisson (or
  Negative Binomial, if the count history is overdispersed) tail
  probability. Instead of asking "how many σ away is today", we ask "if
  this cat's true daily rate is what the baseline says, how surprising is
  seeing *at least* today's count?" This gives an honest answer for rare
  events without needing a non-degenerate std/MAD/IQR — the concrete
  scenario this fixes: baseline scratch_count = [0,1,1,1,0,1] (mean=0.67),
  today=2 → old z ≈ +2.83 (flags "Mild"); Poisson tail
  P(X≥2 | λ≈0.71) ≈ 0.16 → correctly not remarkable (see
  ``analytics/tests/test_deviation.py::test_sparse_count_no_false_alarm``
  for the exact reproduction of this case).

Both paths report a ``sigma_equivalent`` so the downstream fusion engine
(fusion.py) can keep using a single σ-shaped scale (2.5 / 3.0 / 4.0
mild/moderate/severe) without needing to know which model produced it —
for the Poisson path, ``sigma_equivalent`` is the z-score of a standard
normal that would have the same one-sided tail probability
(``scipy``-free implementation, see ``_norm_isf`` below).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from analytics.baseline import Baseline, CONTINUOUS_METRICS, COUNT_METRICS

MAD_SCALE = 1.4826          # consistency constant so MAD ≈ std under normal
MIN_VARIABILITY_MAD = 1e-9  # below this, treat MAD as "degenerate"
COUNT_OVERDISPERSION_RATIO = 1.5  # variance/mean above this → use NB not Poisson
RATE_SMOOTHING_PRIOR = 0.25       # Bayesian-ish additive smoothing for λ, avoids
                                   # λ=0 producing an infinite/undefined tail prob
                                   # after an all-zero baseline window


@dataclass
class MetricDeviation:
    metric: str
    today: float
    model: str                       # "robust_z" | "poisson_tail" | "nbinom_tail" | "insufficient_variability"
    sigma_equivalent: Optional[float]  # signed; None if not computable
    tail_p: Optional[float] = None     # only set for count models
    deviation_score: Optional[float] = None  # 0-100, same scale as before
    note: str = ""


@dataclass
class DeviationResult:
    baseline_days: int
    confidence: str
    metrics: dict = field(default_factory=dict)   # name -> MetricDeviation


def _norm_isf(p: float) -> float:
    """Inverse survival function of the standard normal, i.e. the z such
    that P(Z >= z) = p. Pure-stdlib Acklam-style rational approximation
    (no scipy dependency) — accurate to ~1e-9, more than enough for a
    display-facing "sigma equivalent".
    """
    p = min(max(p, 1e-300), 1 - 1e-16)
    # symmetry: isf(p) = -ppf(p) for p<0.5 branch handled via ppf(1-p)
    q = 1.0 - p
    return _norm_ppf(q)


def _norm_ppf(p: float) -> float:
    """Percent-point function (inverse CDF) of the standard normal."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Acklam's algorithm coefficients.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _log_poisson_pmf(k: int, lam: float) -> float:
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def _poisson_upper_tail(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam), computed by summing PMF terms
    upward from k until the remaining tail is negligible. Fine for the
    small daily counts (<< 1000) this module deals with."""
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    total = 0.0
    i = k
    term = math.exp(_log_poisson_pmf(i, lam))
    while term > 1e-15 and i < k + 5000:
        total += term
        i += 1
        term = math.exp(_log_poisson_pmf(i, lam))
    return min(1.0, total)


def _poisson_lower_tail(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    if lam <= 0:
        return 1.0
    return max(0.0, 1.0 - _poisson_upper_tail(k + 1, lam))


def _log_nbinom_pmf(k: int, r: float, p: float) -> float:
    # r = size (number of "successes"), p = P(success); mean = r(1-p)/p
    return (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p) + k * math.log(1 - p)
    )


def _nbinom_upper_tail(k: int, r: float, p: float) -> float:
    total = 0.0
    i = k
    term = math.exp(_log_nbinom_pmf(i, r, p))
    while term > 1e-15 and i < k + 5000:
        total += term
        i += 1
        term = math.exp(_log_nbinom_pmf(i, r, p))
    return min(1.0, total)


def _nbinom_lower_tail(k: int, r: float, p: float) -> float:
    return max(0.0, 1.0 - _nbinom_upper_tail(k + 1, r, p))


def _fit_count_model(daily_counts: list) -> tuple:
    """Return ('poisson', lam) or ('nbinom', r, p) fitted on the baseline
    window, with additive rate smoothing to avoid a degenerate all-zero
    history producing an infinitely significant first non-zero count."""
    n = len(daily_counts)
    mean = sum(daily_counts) / n if n else 0.0
    mean_smoothed = mean + RATE_SMOOTHING_PRIOR / max(n, 1)
    if n < 2:
        return ("poisson", max(mean_smoothed, RATE_SMOOTHING_PRIOR))
    var = sum((c - mean) ** 2 for c in daily_counts) / n
    if mean <= 0 or var / max(mean, 1e-9) < COUNT_OVERDISPERSION_RATIO:
        return ("poisson", max(mean_smoothed, RATE_SMOOTHING_PRIOR))
    # Method-of-moments NB fit: var = mean + mean^2 / r  =>  r = mean^2/(var-mean)
    r = mean_smoothed ** 2 / max(var - mean, 1e-6)
    p = r / (r + mean_smoothed)
    return ("nbinom", r, p)


def _deviation_score(sigma_eq: Optional[float]) -> Optional[float]:
    """Same 0-100 mapping as the Node-RED original's calcDeviationScore,
    applied to the sigma-equivalent instead of the raw z."""
    if sigma_eq is None:
        return None
    az = abs(sigma_eq)
    if az < 1.0:
        return round(az * 20, 1)
    if az < 2.0:
        return round(20 + (az - 1.0) * 30, 1)
    if az < 3.0:
        return round(50 + (az - 2.0) * 30, 1)
    return round(min(100, 80 + (az - 3.0) * 10), 1)


def _deviate_continuous(metric: str, today: float, stats) -> MetricDeviation:
    scaled_mad = stats.mad * MAD_SCALE
    if scaled_mad < MIN_VARIABILITY_MAD:
        return MetricDeviation(
            metric=metric, today=today, model="insufficient_variability",
            sigma_equivalent=None,
            note=(
                "基線在此指標上幾乎沒有波動（MAD≈0），無法計算穩健 z-score；"
                "需要更多天數的資料，暫不納入偏差判斷。"
            ),
        )
    z = (today - stats.median) / scaled_mad
    return MetricDeviation(
        metric=metric, today=today, model="robust_z",
        sigma_equivalent=round(z, 3),
        deviation_score=_deviation_score(z),
    )


def _deviate_count(metric: str, today: float, stats, history_counts: list) -> MetricDeviation:
    """Two-sided: unusually *high* counts (e.g. excess scratching) and
    unusually *low* counts (e.g. a normally-active cat suddenly not
    grooming at all, which can itself be a health signal) both produce a
    non-zero deviation, matching the original engine's symmetric
    ``abs(z)`` behavior — only the statistical model underneath changes.
    """
    k = max(0, round(today))
    model = _fit_count_model(history_counts)
    if model[0] == "poisson":
        _, lam = model
        upper_p = _poisson_upper_tail(k, lam)
        lower_p = _poisson_lower_tail(k, lam)
        note = f"Poisson(λ={lam:.2f}) 尾機率模型"
    else:
        _, r, p_param = model
        upper_p = _nbinom_upper_tail(k, r, p_param)
        lower_p = _nbinom_lower_tail(k, r, p_param)
        note = f"Negative-Binomial(r={r:.2f}) 尾機率模型（歷史計數過度離散）"

    if k >= stats.median:
        p, sign = upper_p, 1.0
    else:
        p, sign = lower_p, -1.0
    sigma_eq = sign * _norm_isf(max(min(p, 1.0), 1e-300))

    return MetricDeviation(
        metric=metric, today=today, model=f"{model[0]}_tail",
        sigma_equivalent=round(sigma_eq, 3), tail_p=round(p, 5),
        deviation_score=_deviation_score(sigma_eq), note=note,
    )


def compute_deviation(
    today: dict,
    baseline: Baseline,
    history_counts_by_metric: Optional[dict] = None,
) -> DeviationResult:
    """Score today's behavior counts/durations against ``baseline``.

    Parameters
    ----------
    today
        e.g. ``{"walk_time": 1800, "lick_time": 900, "scratch_count": 2, ...}``
    baseline
        Output of ``baseline.compute_baseline``. For COUNT_METRICS, the
        per-day counts needed to fit the tail-probability model default to
        ``baseline.count_histories`` (the exact day-window the baseline's
        own stats were computed from) — you normally don't need to pass
        ``history_counts_by_metric`` at all.
    history_counts_by_metric
        Optional override/supplement, e.g. ``{"scratch_count": [0,1,1,1,0,1]}``.
        Only needed if you're scoring against a different count history
        than the one baked into ``baseline`` (e.g. in tests). Merged on
        top of ``baseline.count_histories``, taking precedence per-metric.
    """
    history_counts_by_metric = {**baseline.count_histories, **(history_counts_by_metric or {})}
    result = DeviationResult(baseline_days=baseline.days_count, confidence=baseline.confidence)

    for metric, stats in baseline.metrics.items():
        cur = float(today.get(metric, 0) or 0)
        if metric in CONTINUOUS_METRICS:
            result.metrics[metric] = _deviate_continuous(metric, cur, stats)
        elif metric in COUNT_METRICS:
            counts = history_counts_by_metric.get(metric)
            if not counts:
                result.metrics[metric] = MetricDeviation(
                    metric=metric, today=cur, model="insufficient_variability",
                    sigma_equivalent=None,
                    note="缺少每日計數歷史，無法擬合尾機率模型。",
                )
            else:
                result.metrics[metric] = _deviate_count(metric, cur, stats, counts)
    return result
