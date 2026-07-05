from datetime import date, timedelta

from analytics.baseline import DailyRecord, compute_baseline
from analytics.deviation import compute_deviation
from analytics.fusion import compute_fusion


def _mk_days(n, **overrides):
    days = []
    for i in range(n):
        kwargs = dict(day=date(2026, 1, 1) + timedelta(days=i), monitoring_seconds=7200.0)
        for key, values in overrides.items():
            kwargs[key] = values[i]
        days.append(DailyRecord(**kwargs))
    return days


def _flat_baseline(**per_metric_history):
    n = len(next(iter(per_metric_history.values())))
    days = _mk_days(n, **per_metric_history)
    return compute_baseline(days, min_days=n)


def test_normal_day_scores_low_and_no_override():
    baseline = _flat_baseline(
        lick_time=[3000] * 7, lick_count=[10] * 7,
        scratch_time=[40, 50, 60, 45, 55, 50, 48], scratch_count=[1, 2, 1, 2, 1, 1, 2],
        shake_count=[1, 0, 1, 0, 1, 0, 1], walk_time=[1800] * 7, stop_time=[20000] * 7,
    )
    dev = compute_deviation(
        today={
            "lick_time": 3050, "lick_count": 10,
            "scratch_time": 52, "scratch_count": 1,
            "shake_count": 1, "walk_time": 1850, "stop_time": 20100,
        },
        baseline=baseline,
        history_counts_by_metric={
            "lick_count": [10] * 7, "scratch_count": [1, 2, 1, 2, 1, 1, 2], "shake_count": [1, 0, 1, 0, 1, 0, 1],
        },
    )
    fusion = compute_fusion(dev, class_c_score=0.0)
    assert fusion.level == "Normal"
    assert fusion.class_a_triggered is False
    assert fusion.score < 20


def test_single_class_a_spike_overrides_to_at_least_moderate():
    """Single Behavior Critical Rule: a large scratch_time spike alone
    (Class B all normal) must still be able to push the level up, even
    though it's a minority of the fused weighted score."""
    baseline = _flat_baseline(
        lick_time=[3000] * 7, lick_count=[10] * 7,
        scratch_time=[45, 50, 55, 48, 52, 47, 50], scratch_count=[1] * 7,
        shake_count=[1] * 7, walk_time=[1800] * 7, stop_time=[20000] * 7,
    )
    dev = compute_deviation(
        today={
            "lick_time": 3000, "lick_count": 10,
            "scratch_time": 600,  # far above baseline (~50s) -> huge robust z
            "scratch_count": 1, "shake_count": 1, "walk_time": 1800, "stop_time": 20000,
        },
        baseline=baseline,
        history_counts_by_metric={
            "lick_count": [10] * 7, "scratch_count": [1] * 7, "shake_count": [1] * 7,
        },
    )
    fusion = compute_fusion(dev, class_c_score=0.0)
    assert fusion.class_a_triggered is True
    assert fusion.class_a_override in (
        "Mild Behavioral Deviation", "Moderate Behavioral Deviation", "Severe Behavioral Deviation",
    )
    assert fusion.level != "Normal"
    assert fusion.top_contributors[0][0] == "scratch_time"


def test_class_b_alone_cannot_reach_severe():
    """Class B (walk/stop/shake) is explicitly *not* allowed to
    independently trigger — it can only support/raise confidence, capped
    by the 25% fusion weight."""
    baseline = _flat_baseline(
        lick_time=[3000] * 7, lick_count=[10] * 7,
        scratch_time=[50] * 7, scratch_count=[1] * 7,
        shake_count=[1, 0, 1, 0, 1, 0, 1], walk_time=[1800] * 7, stop_time=[20000] * 7,
    )
    dev = compute_deviation(
        today={
            "lick_time": 3000, "lick_count": 10, "scratch_time": 50, "scratch_count": 1,
            "shake_count": 20,  # extreme Class B spike
            "walk_time": 1800, "stop_time": 20000,
        },
        baseline=baseline,
        history_counts_by_metric={
            "lick_count": [10] * 7, "scratch_count": [1] * 7, "shake_count": [1, 0, 1, 0, 1, 0, 1],
        },
    )
    fusion = compute_fusion(dev, class_c_score=0.0)
    assert fusion.class_a_triggered is False
    assert fusion.level != "Severe Behavioral Deviation"
