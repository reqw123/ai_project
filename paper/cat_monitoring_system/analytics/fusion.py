"""Class A/B/C evidence fusion — port of 行為偏差融合引擎 (fusion_score,
level, Class A override rule) from ``cat_health_v3_flow.json``.

Weights, thresholds and the "Single Behavior Critical Rule" override are
kept numerically identical to the Node-RED original; the only substantive
change is that every ``sigma`` value fed in here can now come from either
the robust-z (continuous) or Poisson/NB-tail (count) model in
``deviation.py`` instead of always being a plain ``(x-mean)/std`` — see
``deviation.py`` module docstring for why that matters for the sparse
count metrics (scratch_frequency, lick_frequency, head_shake).

This module does not compute rhythm/transition (Class C) scores itself —
those come from the existing rhythm-analysis / transition-matrix nodes in
Node-RED (行為節律分析), which are already pure aggregation with no z-score
problem and are out of scope for this redesign. ``compute_fusion`` accepts
pre-computed Class C sub-scores as plain floats so it can be called either
from a full Python pipeline or bridged from Node-RED during migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from analytics.deviation import DeviationResult

LEVELS = [
    "Normal",
    "Mild Behavioral Deviation",
    "Moderate Behavioral Deviation",
    "Severe Behavioral Deviation",
]
LEVEL_MIN_SCORE = {
    "Normal": 0,
    "Mild Behavioral Deviation": 20,
    "Moderate Behavioral Deviation": 45,
    "Severe Behavioral Deviation": 70,
}

# metric name (as produced by deviation.py) -> (weight, threshold_mild, threshold_mod, threshold_severe)
#
# Duration features (lick_time, scratch_time) are weighted above their paired count
# features (lick_count, scratch_count) not because duration is assumed to be more
# clinically severe than frequency -- Colombo et al. (2022, Veterinary Dermatology
# 33(5), "VAScat") found lick and scratch VAS scores correlate with *different*
# clinical signs (hair loss vs.
# head/neck pruritus respectively, r=0.26 between the two scales) with no evidence
# either axis is inherently more severe, which is why lick/scratch stay separate
# features here rather than being merged. The duration-over-count weighting instead
# reflects measurement stability: bout *count* depends on an arbitrary segmentation
# rule (Eckstein & Hart, 2000, used a 60s gap to end a bout and merged brief <5s
# pauses into the same bout), making it more sensitive to detection noise than
# cumulative *duration*, which is a continuous sum less perturbed by where bout
# boundaries fall.
CLASS_A_FEATURES = {
    "lick_time": {"weight": 0.30, "label": "Lick Duration"},
    "lick_count": {"weight": 0.15, "label": "Lick Frequency"},
    "scratch_time": {"weight": 0.30, "label": "Scratch Duration"},
    "scratch_count": {"weight": 0.10, "label": "Scratch Frequency"},
}
# Sigma cutoffs (2.5 / 3.0 / 4.0) are currently expert-set, not data-derived. Colombo
# et al. (2022) calibrated an analogous cat pruritus severity cutoff via ROC analysis
# against owner-scored VAS in 153 diseased + 108 healthy cats (cutoff=2 on a 0-10
# scale gave 93% sensitivity / 92% specificity) -- once real diagnosed-anomaly
# labels exist for this system, the same ROC-optimization approach should replace
# these fixed thresholds.
CLASS_A_THRESHOLDS = {"mild": 2.5, "moderate": 3.0, "severe": 4.0}
CLASS_A_WEIGHT_SUM = sum(f["weight"] for f in CLASS_A_FEATURES.values())  # 0.85

# shake_count's 0.40 (highest of the three Class B features) has NO literature
# support: a targeted lit scan (2026-08) found no quantified/validated link between
# head-shake frequency and ear-disease diagnosis in cats, and veterinary consensus
# (Brame & Cain, 2021, JFMS) describes head-shaking as sensitive but not
# diagnostically specific to ear disease -- multiple unrelated causes present with
# the same behavior. Kept as-is (expert judgment) pending real data; documented
# here as an explicit, unvalidated design choice rather than left silent.
CLASS_B_FEATURES = {
    "shake_count": {"weight": 0.40, "label": "Head Shake"},
    "walk_time": {"weight": 0.30, "label": "Walk Duration"},
    "stop_time": {"weight": 0.30, "label": "Inactive Duration"},
}

# Class A > Class C > Class B (45/30/25) implicitly assumes self-care behavior
# (Class A) is a more specific health signal than general activity (Class B). A
# targeted lit scan (2026-08) found this assumption is NOT supported by the sickness
# -behavior literature -- Lopes et al. (2021, J Experimental Biology) and the Merck
# Veterinary Manual's pain-assessment guidance both treat reduced grooming and
# reduced activity as co-occurring, equally non-specific components of the same
# cytokine-driven "sickness behavior" syndrome, not as a hierarchy of specificity.
# Class C's weight has indirect support (Wagner et al., 2021, Methods: circadian-
# rhythm deviation detected disease in dairy cattle with 95% recall) but that study
# did not compare against other evidence classes, so it cannot justify the specific
# 30% figure either. All three weights remain expert-set; see
# ``paper/docs/個體化基線與異常偵測文獻來源.md`` §3 for the full literature picture
# and the honest limitation this implies for the thesis methodology section.
FUSION_WEIGHTS = {"class_a": 0.45, "class_b": 0.25, "class_c": 0.30}


@dataclass
class FusionResult:
    """跨類別（A/B/C）證據融合後的最終健康風險評估結果。"""

    score: float
    level: str
    class_a_score: float
    class_b_score: float
    class_c_score: float
    class_a_triggered: bool
    class_a_override: Optional[str]
    top_contributors: list = field(
        default_factory=list
    )  # [(metric, sigma), ...] desc by |sigma|


def _sigma(dev: DeviationResult, metric: str) -> float:
    m = dev.metrics.get(metric)
    if m is None or m.sigma_equivalent is None:
        return 0.0
    return m.sigma_equivalent


def _score_to_level(score: float) -> str:
    if score < 20:
        return "Normal"
    if score < 45:
        return "Mild Behavioral Deviation"
    if score < 70:
        return "Moderate Behavioral Deviation"
    return "Severe Behavioral Deviation"


def compute_fusion(
    deviation: DeviationResult,
    class_c_score: float = 0.0,
) -> FusionResult:
    """Fuse Class A (critical self-care behaviors), Class B (supporting
    activity behaviors) and a pre-computed Class C (rhythm/transition
    pattern) score into one overall level.

    ``class_c_score`` (0-100) is supplied by the caller — rhythm/transition
    analysis is unchanged from the Node-RED original and out of scope here.
    """
    sigmas_a = {k: _sigma(deviation, k) for k in CLASS_A_FEATURES}
    class_a_weighted = (
        sum(abs(sigmas_a[k]) * f["weight"] for k, f in CLASS_A_FEATURES.items())
        / CLASS_A_WEIGHT_SUM
    )
    class_a_score = min(100.0, class_a_weighted * 25)

    max_abs_a = max((abs(v) for v in sigmas_a.values()), default=0.0)
    override = None
    if max_abs_a >= CLASS_A_THRESHOLDS["severe"]:
        override = "Severe Behavioral Deviation"
    elif max_abs_a >= CLASS_A_THRESHOLDS["moderate"]:
        override = "Moderate Behavioral Deviation"
    elif max_abs_a >= CLASS_A_THRESHOLDS["mild"]:
        override = "Mild Behavioral Deviation"
    class_a_triggered = override is not None

    sigmas_b = {k: _sigma(deviation, k) for k in CLASS_B_FEATURES}
    class_b_weighted = sum(
        abs(sigmas_b[k]) * f["weight"] for k, f in CLASS_B_FEATURES.items()
    )
    class_b_score = min(100.0, class_b_weighted * 25)

    class_c_score = max(0.0, min(100.0, class_c_score))

    fusion_score = (
        class_a_score * FUSION_WEIGHTS["class_a"]
        + class_b_score * FUSION_WEIGHTS["class_b"]
        + class_c_score * FUSION_WEIGHTS["class_c"]
    )
    fusion_level = _score_to_level(fusion_score)

    final_level = fusion_level
    if override is not None and LEVELS.index(override) > LEVELS.index(fusion_level):
        final_level = override

    final_score = fusion_score
    if final_level != fusion_level:
        final_score = max(fusion_score, LEVEL_MIN_SCORE[final_level])

    all_sigmas = {**sigmas_a, **sigmas_b}
    top = sorted(all_sigmas.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return FusionResult(
        score=round(final_score, 1),
        level=final_level,
        class_a_score=round(class_a_score, 1),
        class_b_score=round(class_b_score, 1),
        class_c_score=round(class_c_score, 1),
        class_a_triggered=class_a_triggered,
        class_a_override=override,
        top_contributors=top,
    )
