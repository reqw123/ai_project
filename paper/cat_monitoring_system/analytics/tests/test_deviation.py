import math
from datetime import date, timedelta

import pytest

from analytics.baseline import DailyRecord, compute_baseline
from analytics.deviation import compute_deviation, _poisson_upper_tail, _norm_isf


def _mk_days(n, **overrides):
    days = []
    for i in range(n):
        kwargs = dict(day=date(2026, 1, 1) + timedelta(days=i), monitoring_seconds=7200.0)
        for key, values in overrides.items():
            kwargs[key] = values[i]
        days.append(DailyRecord(**kwargs))
    return days


def test_count_history_defaults_from_baseline_no_explicit_arg_needed():
    """compute_deviation must be usable with just `today` + `baseline` —
    the count history for the tail-probability fit should come from
    baseline.count_histories automatically, not require the caller to
    separately reconstruct the same day-window by hand."""
    scratch_counts = [0, 1, 1, 1, 0, 1]
    days = _mk_days(6, scratch_count=scratch_counts)
    baseline = compute_baseline(days, min_days=6)
    assert baseline.count_histories["scratch_count"] == scratch_counts

    dev = compute_deviation(today={"scratch_count": 2}, baseline=baseline)
    m = dev.metrics["scratch_count"]
    assert m.model == "poisson_tail"
    assert m.sigma_equivalent < 2.5


def test_sparse_count_no_false_alarm():
    """Reproduces the exact failure mode this redesign targets: a rare,
    low-count behavior (scratch_count) where the population std over the
    baseline window is small enough that one extra event on an otherwise
    unremarkable day pushes the *old* (x-mean)/std z-score past the 2.5σ
    "Mild Behavioral Deviation" threshold, purely from a small denominator
    — not because the count is actually statistically surprising.
    """
    scratch_counts = [0, 1, 1, 1, 0, 1]
    days = _mk_days(6, scratch_count=scratch_counts)
    baseline = compute_baseline(days, min_days=6)

    mean = sum(scratch_counts) / len(scratch_counts)
    std = math.sqrt(sum((c - mean) ** 2 for c in scratch_counts) / len(scratch_counts))
    today = 2
    old_z = (today - mean) / std
    assert old_z > 2.5, "sanity check: old z-score formula does flag this as ≥Mild"

    dev = compute_deviation(
        today={"scratch_count": today},
        baseline=baseline,
        history_counts_by_metric={"scratch_count": scratch_counts},
    )
    m = dev.metrics["scratch_count"]
    assert m.model == "poisson_tail"
    assert m.sigma_equivalent < 2.5, (
        f"new model should NOT flag today={today} vs history={scratch_counts} as "
        f"even Mild deviation, got sigma_equivalent={m.sigma_equivalent}"
    )
    assert m.tail_p > 0.05


def test_sparse_count_still_flags_a_genuine_outbreak():
    """The fix must not become blind to real anomalies: a cat that never
    scratches suddenly scratching 8 times in a day should still register
    as a strong deviation."""
    scratch_counts = [0, 0, 1, 0, 0, 1, 0]
    days = _mk_days(7, scratch_count=scratch_counts)
    baseline = compute_baseline(days, min_days=7)

    dev = compute_deviation(
        today={"scratch_count": 8},
        baseline=baseline,
        history_counts_by_metric={"scratch_count": scratch_counts},
    )
    m = dev.metrics["scratch_count"]
    assert m.sigma_equivalent >= 3.0, "a genuine 8x jump from a near-zero baseline should still flag"


def test_count_deviation_is_two_sided_not_just_high():
    """A normally-active cat that suddenly stops grooming/scratching
    entirely should also register as a deviation (negative sigma), not
    be silently zeroed out — reduced grooming can itself be a health
    signal, and the original Node-RED engine's abs(z) was symmetric."""
    lick_counts = [12, 11, 13, 12, 14, 12, 13]  # consistently active groomer
    days = _mk_days(7, lick_count=lick_counts)
    baseline = compute_baseline(days, min_days=7)

    dev = compute_deviation(
        today={"lick_count": 0},  # groomed zero times today
        baseline=baseline,
        history_counts_by_metric={"lick_count": lick_counts},
    )
    m = dev.metrics["lick_count"]
    assert m.sigma_equivalent < 0, "an unusually low count should report a negative sigma, not be zeroed out"
    assert abs(m.sigma_equivalent) >= 2.5, "going from ~12/day to 0 should register as at least a Mild deviation"


def test_continuous_robust_z_resists_a_single_contaminating_day():
    """One freak long-grooming day sitting in the baseline window inflates
    mean/std enough to *mask* a genuinely anomalous day that comes after
    it — the median/MAD baseline barely moves, so it stays sensitive.

    This is the "masking" failure mode of mean/std baselines: a single
    contaminated history day doesn't just risk one false positive, it can
    silently blind the detector to real anomalies for as long as that day
    stays in the rolling window.
    """
    lick_times = [3000, 3100, 2900, 3050, 2950, 3000, 30000]  # last day: contaminant
    days = _mk_days(7, lick_time=lick_times)
    baseline = compute_baseline(days, min_days=7)
    stats = baseline.metrics["lick_time"]

    # median barely moves...
    assert 2950 <= stats.median <= 3100
    # ...while the mean is dragged way up by the single outlier.
    assert stats.mean > 5000

    today = 9000  # a real 3x jump from the ~3000s/day typical value
    old_mean = sum(lick_times) / len(lick_times)
    old_std = (sum((v - old_mean) ** 2 for v in lick_times) / len(lick_times)) ** 0.5
    old_z = (today - old_mean) / old_std
    assert abs(old_z) < 1.0, "sanity check: the contaminated mean/std masks this real anomaly"

    dev = compute_deviation(today={"lick_time": today}, baseline=baseline)
    m = dev.metrics["lick_time"]
    assert m.model == "robust_z"
    assert m.sigma_equivalent > 5.0, (
        "robust (median/MAD) z-score should still clearly flag a real 3x jump "
        "even though one contaminated day is sitting in the baseline window"
    )


def test_continuous_insufficient_variability_flagged_not_silently_wrong():
    lick_times = [3000] * 7  # zero spread
    days = _mk_days(7, lick_time=lick_times)
    baseline = compute_baseline(days, min_days=7)

    dev = compute_deviation(today={"lick_time": 5000}, baseline=baseline)
    m = dev.metrics["lick_time"]
    assert m.model == "insufficient_variability"
    assert m.sigma_equivalent is None


def test_poisson_upper_tail_matches_hand_computed_value():
    # P(X>=2 | Poisson(lambda=0.708)) = 1 - e^-λ(1+λ)
    lam = 0.708
    expected = 1 - math.exp(-lam) * (1 + lam)
    assert _poisson_upper_tail(2, lam) == pytest.approx(expected, rel=1e-9)


def test_norm_isf_roundtrips_known_quantiles():
    # P(Z >= 1.645) ≈ 0.05 (one-sided 95th percentile)
    assert _norm_isf(0.05) == pytest.approx(1.645, abs=0.01)
    # P(Z >= 2.5) ≈ 0.0062
    assert _norm_isf(0.0062) == pytest.approx(2.5, abs=0.02)
