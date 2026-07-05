"""Individual-baseline / behavioral-deviation statistics engine.

Python re-implementation of the statistics currently living inside the
Node-RED function nodes of ``cat_health_v3_flow.json``
(個體化基線計算器 / 偏差分析引擎 / 行為偏差融合引擎). See
``analytics/README.md`` for the design rationale and migration plan.
"""
from analytics.baseline import (
    MetricStats,
    Baseline,
    compute_baseline,
)
from analytics.deviation import (
    MetricDeviation,
    DeviationResult,
    compute_deviation,
)
from analytics.fusion import (
    FusionResult,
    compute_fusion,
)

__all__ = [
    "MetricStats",
    "Baseline",
    "compute_baseline",
    "MetricDeviation",
    "DeviationResult",
    "compute_deviation",
    "FusionResult",
    "compute_fusion",
]
