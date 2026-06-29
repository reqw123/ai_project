"""Ear-to-ear distance and related body-scale metrics."""
from typing import Tuple
import math
import numpy as np
from plugins.lick_stage.config import LickConfig as _C


def compute_ear_distance(kpts, kpt_conf) -> Tuple[float, float, bool, float, float]:
    """
    Returns (dist_px, dist_norm, valid, body_scale, body_ear_ratio).

    dist_px        — pixel distance between left and right ears
    dist_norm      — dist_px / body_scale
    valid          — True when both ears are above confidence threshold
    body_scale     — chest-to-hip distance in pixels
    body_ear_ratio — body_scale / dist_px  (used by front-view guard)
    All numeric outputs are NaN when the corresponding keypoints are missing.
    """
    left_ok  = float(kpt_conf[_C.KP_LEFT_EAR])  >= _C.EAR_CONF_THRESHOLD
    right_ok = float(kpt_conf[_C.KP_RIGHT_EAR]) >= _C.EAR_CONF_THRESHOLD
    chest_ok = float(kpt_conf[_C.KP_CHEST])     >= _C.BODY_KP_CONF
    hip_ok   = float(kpt_conf[_C.KP_HIP])       >= _C.BODY_KP_CONF

    if left_ok and right_ok:
        left_pt  = np.asarray(kpts[_C.KP_LEFT_EAR],  dtype=np.float64)
        right_pt = np.asarray(kpts[_C.KP_RIGHT_EAR], dtype=np.float64)
        dist_px  = float(np.linalg.norm(left_pt - right_pt))
        valid    = True
    else:
        dist_px = float("nan")
        valid   = False

    if chest_ok and hip_ok:
        chest_pt   = np.asarray(kpts[_C.KP_CHEST], dtype=np.float64)
        hip_pt     = np.asarray(kpts[_C.KP_HIP],   dtype=np.float64)
        body_scale = float(np.linalg.norm(chest_pt - hip_pt))
    else:
        body_scale = float("nan")

    if math.isfinite(body_scale) and math.isfinite(dist_px) and dist_px > 1e-6:
        dist_norm      = dist_px / body_scale
        body_ear_ratio = body_scale / dist_px
    else:
        dist_norm      = float("nan")
        body_ear_ratio = float("nan")

    return dist_px, dist_norm, valid, body_scale, body_ear_ratio
