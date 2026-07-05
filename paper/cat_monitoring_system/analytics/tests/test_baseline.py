from datetime import date, timedelta

import pytest

from analytics.baseline import (
    DailyRecord, compute_baseline, compute_metric_stats, InsufficientDataError,
)


def _mk_days(n, **overrides):
    """n days starting 2026-01-01, each with monitoring_seconds=7200 and
    zeroed metrics unless overridden by lists (index-aligned to day)."""
    days = []
    for i in range(n):
        kwargs = dict(day=date(2026, 1, 1) + timedelta(days=i), monitoring_seconds=7200.0)
        for key, values in overrides.items():
            kwargs[key] = values[i]
        days.append(DailyRecord(**kwargs))
    return days


def test_compute_metric_stats_matches_hand_computed_values():
    # mean=3, population std = sqrt(mean((x-3)^2)) for [1,2,3,4,5] = sqrt(2) ≈ 1.414
    stats = compute_metric_stats([1, 2, 3, 4, 5])
    assert stats.mean == 3.0
    assert stats.median == 3.0
    assert stats.std == pytest.approx(1.414, abs=0.01)
    assert stats.sample_count == 5
    # quartiles via linear interpolation: q1=2, q3=4 -> iqr=2
    assert stats.q1 == 2.0
    assert stats.q3 == 4.0
    assert stats.iqr == 2.0


def test_compute_metric_stats_empty_is_all_zero_not_a_crash():
    stats = compute_metric_stats([])
    assert stats.sample_count == 0
    assert stats.mean == 0 and stats.std == 0


def test_insufficient_days_raises_explicit_error():
    days = _mk_days(3, walk_time=[100, 200, 300])
    with pytest.raises(InsufficientDataError) as exc:
        compute_baseline(days, min_days=7)
    assert exc.value.current_days == 3
    assert exc.value.required_days == 7


def test_excluded_dates_removed_when_enough_data_remains():
    days = _mk_days(10, walk_time=[100] * 9 + [999999])  # last day is an outlier
    excluded = [days[-1].day.isoformat()]
    bl = compute_baseline(days, min_days=7, excluded_dates=excluded)
    assert bl.days_count == 9
    assert bl.metrics["walk_time"].mean == 100.0


def test_excluded_dates_ignored_if_would_drop_below_min_days():
    days = _mk_days(7, walk_time=[100] * 6 + [999999])
    excluded = [d.day.isoformat() for d in days[:2]]  # would leave only 5 valid days
    bl = compute_baseline(days, min_days=7, excluded_dates=excluded)
    # exclusion not applied because it would drop below min_days -> all 7 kept
    assert bl.days_count == 7


def test_low_medium_high_confidence_bands():
    # thresholds (matches Node-RED calcConfidence): n<7 Low, n<30 Medium, else High
    assert compute_baseline(_mk_days(6, walk_time=[1] * 6), min_days=3).confidence == "Low"
    assert compute_baseline(_mk_days(10, walk_time=[1] * 10), min_days=7).confidence == "Medium"
    assert compute_baseline(_mk_days(30, walk_time=[1] * 30), min_days=7).confidence == "High"


def test_sanity_warning_fires_when_lick_baseline_absurdly_high():
    # lick_time way above 6x the ~3600s/day population reference
    days = _mk_days(7, lick_time=[8 * 3600] * 7)
    bl = compute_baseline(days, min_days=7)
    assert not bl.sanity_ok
    assert any("Lick Duration" in w for w in bl.sanity_warnings)


def test_sanity_ok_for_a_plausible_baseline():
    days = _mk_days(10, lick_time=[3000] * 10, scratch_time=[45] * 10)
    bl = compute_baseline(days, min_days=7)
    assert bl.sanity_ok
    assert bl.sanity_warnings == []
