"""Head / face direction inference and state smoothing."""
import math
from collections import Counter, deque
from typing import Tuple
import numpy as np
from plugins.lick_stage.config import LickConfig as _C


def compute_head_ear_angle(kpts, kpt_conf) -> float:
    """Angle in degrees at nose between left-ear and right-ear vectors. NaN if unavailable."""
    nose_ok  = float(kpt_conf[_C.KP_NOSE])      >= _C.NOSE_CONF_THRESHOLD
    left_ok  = float(kpt_conf[_C.KP_LEFT_EAR])  >= _C.EAR_CONF_THRESHOLD
    right_ok = float(kpt_conf[_C.KP_RIGHT_EAR]) >= _C.EAR_CONF_THRESHOLD
    if not (nose_ok and left_ok and right_ok):
        return float("nan")
    nose_pt  = np.asarray(kpts[_C.KP_NOSE],      dtype=np.float64)
    left_pt  = np.asarray(kpts[_C.KP_LEFT_EAR],  dtype=np.float64)
    right_pt = np.asarray(kpts[_C.KP_RIGHT_EAR], dtype=np.float64)
    v_l = left_pt  - nose_pt
    v_r = right_pt - nose_pt
    na = math.hypot(float(v_l[0]), float(v_l[1]))
    nb = math.hypot(float(v_r[0]), float(v_r[1]))
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    cos_val = max(-1.0, min(1.0, float(np.dot(v_l, v_r)) / (na * nb)))
    return math.degrees(math.acos(cos_val))


def infer_face_state_cat_centric(target_geom, nose_ok: bool) -> Tuple[str, float, float, float]:
    """
    Determine face direction using cat-body coordinate system.

    Returns (state, forward_norm, lateral_norm, gaze_angle_deg).
    forward_norm > 0 means nose points toward chest (front-facing).
    """
    if target_geom is None or not nose_ok:
        return _C.STATE_UNKNOWN, float("nan"), float("nan"), float("nan")

    nose          = np.asarray(target_geom["nose"],          dtype=np.float64)
    body_center   = np.asarray(target_geom["body_center"],   dtype=np.float64)
    body_axis     = np.asarray(target_geom["body_axis_unit"],dtype=np.float64)
    body_normal   = np.asarray(target_geom["body_normal"],   dtype=np.float64)
    body_len      = float(target_geom.get("body_len", 0.0))

    if body_len < 1e-6:
        return _C.STATE_UNKNOWN, float("nan"), float("nan"), float("nan")

    rel = nose - body_center
    # forward: dot with -body_axis (chest direction is −body_axis since axis = hip−chest)
    forward_norm = float(np.dot(rel, -body_axis) / body_len)
    lateral_norm = float(np.dot(rel, body_normal) / body_len) * float(_C.CAT_LR_SIGN)
    gaze_angle_deg = float(np.degrees(np.arctan2(lateral_norm, forward_norm)))

    if forward_norm >= _C.CAT_FRONT_FORWARD_MIN:
        if lateral_norm <= -_C.CAT_LR_MARGIN:
            return _C.STATE_FRONT_LEFT,  forward_norm, lateral_norm, gaze_angle_deg
        if lateral_norm >= _C.CAT_LR_MARGIN:
            return _C.STATE_FRONT_RIGHT, forward_norm, lateral_norm, gaze_angle_deg
        return _C.STATE_FRONT, forward_norm, lateral_norm, gaze_angle_deg

    if forward_norm <= -_C.CAT_BACK_FORWARD_MIN:
        return _C.STATE_BACK, forward_norm, lateral_norm, gaze_angle_deg

    return _C.STATE_UNKNOWN, forward_norm, lateral_norm, gaze_angle_deg


def infer_face_state_user_rules(
    head_ear_angle_deg: float,
    dist_norm: float,
    dist_px: float,
    nose_conf: float,
) -> Tuple[str, bool]:
    """
    Apply hard geometric rules that override cat-centric state.

    Returns (state, rule_applied).
    Priority: BACK rule fires before FRONT rule (same as standalone script).
    """
    if _C.BACK_VIEW_REQUIRE_LOW_NOSE and nose_conf <= _C.BACK_CAMERA_NOSE_CONF_MAX:
        if (not math.isfinite(dist_px)) or dist_px > _C.BACK_CAMERA_DIST_MIN_PX:
            return _C.STATE_BACK, True

    if math.isfinite(head_ear_angle_deg) and math.isfinite(dist_norm):
        if head_ear_angle_deg > _C.FRONT_CAMERA_ANGLE_MIN_DEG and dist_norm > _C.FRONT_CAMERA_NORM_MIN:
            return _C.STATE_FRONT, True

    if (not _C.BACK_VIEW_REQUIRE_LOW_NOSE) and nose_conf <= _C.BACK_CAMERA_NOSE_CONF_MAX:
        if math.isfinite(dist_px) and dist_px > _C.BACK_CAMERA_DIST_MIN_PX:
            return _C.STATE_BACK, True

    return _C.STATE_UNKNOWN, False


def stabilize_direction_vector(new_vec, prev_vec, alpha: float, flip_margin: float):
    """Flip-aware EMA for a unit direction vector across frames.

    Plain EMA on a vector that can legitimately point in either of two
    opposite directions frame-to-frame (e.g. derived from an axis with no
    inherent sign, like the ear-to-ear line) is unsafe: if it flips ~180°,
    naively averaging with the previous vector cancels toward zero instead
    of tracking the true orientation, which is exactly the "梯形亂轉"
    instability. Fix: if the new vector points roughly opposite the
    previous one, flip it first so both point the same way, *then* blend.

    flip_margin adds a deadband around the perpendicular (dot ~ 0) case so
    per-frame keypoint jitter right at the flip boundary doesn't toggle the
    sign back and forth — only a clearly opposite reading (dot < -flip_margin)
    is treated as a genuine flip.

    Returns a re-normalized unit vector. Pure function — caller owns state.
    """
    new_vec = np.asarray(new_vec, dtype=np.float64)
    norm = math.hypot(float(new_vec[0]), float(new_vec[1]))
    if norm < 1e-9:
        return np.asarray(prev_vec, dtype=np.float64) if prev_vec is not None else new_vec
    new_vec = new_vec / norm

    if prev_vec is None:
        return new_vec

    prev_vec = np.asarray(prev_vec, dtype=np.float64)
    dot = float(np.dot(new_vec, prev_vec))
    if dot < -flip_margin:
        new_vec = -new_vec
        dot = -dot

    blended = alpha * new_vec + (1.0 - alpha) * prev_vec
    b_norm = math.hypot(float(blended[0]), float(blended[1]))
    return blended / b_norm if b_norm > 1e-9 else new_vec


def smooth_state(history: deque) -> Tuple[str, float]:
    """Majority-vote smoothing over recent history. Returns (dominant_state, stability)."""
    if not history:
        return _C.STATE_UNKNOWN, 0.0
    counts = Counter(history)
    dominant, votes = counts.most_common(1)[0]
    return dominant, float(votes) / len(history)


def check_front_view_guard(kpt_conf, dist_px: float, body_ear_ratio: float) -> bool:
    """
    Return True when the cat is in a face-on position that makes body-axis
    analysis unreliable (body foreshortened toward camera).

    Note: body_scale_norm guard (requires frame diagonal) is intentionally
    omitted here; only the body_ear_ratio criterion is applied.
    """
    if not _C.FRONT_VIEW_GUARD_ENABLED:
        return False
    left_ok  = float(kpt_conf[_C.KP_LEFT_EAR])  >= _C.EAR_CONF_THRESHOLD
    right_ok = float(kpt_conf[_C.KP_RIGHT_EAR]) >= _C.EAR_CONF_THRESHOLD
    if not (left_ok and right_ok):
        return False
    if not math.isfinite(body_ear_ratio):
        return False
    return body_ear_ratio <= _C.FRONT_VIEW_BODY_EAR_RATIO_MAX
