"""Pure geometry: builds the 7 body-zone targets and classifies a nose point.

No side effects, no I/O, no overlay drawing — this module only computes.
"""
import math
from typing import Optional, Tuple
import numpy as np

from .config import ExtZoneConfig as _C


def _perp(v) -> np.ndarray:
    return np.array([-float(v[1]), float(v[0])], dtype=np.float64)


def _norm(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _conf_ok(kpt_conf, idx: int, threshold: float = _C.CONF_THRESHOLD) -> bool:
    return float(kpt_conf[idx]) > threshold


def _point_in_circle(pt, center, radius: float) -> Tuple[bool, float]:
    d = _norm(np.asarray(pt, dtype=np.float64) - np.asarray(center, dtype=np.float64))
    return d <= radius, d


def _point_on_strip(pt, p0, p1, half_width: float) -> Tuple[bool, float]:
    """Rectangular strip around segment p0-p1. Returns (hit, perp_dist)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    seg_len = _norm(seg)
    if seg_len < 1e-6:
        return False, float("inf")
    axis = seg / seg_len
    rel  = np.asarray(pt, dtype=np.float64) - p0
    t    = float(np.dot(rel, axis))
    perp = float(np.dot(rel, _perp(axis)))
    hit  = (0.0 <= t <= seg_len) and (abs(perp) <= half_width)
    return hit, abs(perp)


def build_zone_targets(kpts, kpt_conf) -> Optional[dict]:
    """Build all 7 zone target shapes from one frame of keypoints.

    Returns None when the minimum required keypoints (chest, hip) are
    missing/low-confidence — callers must treat that as "no zone".
    """
    if kpts is None or kpt_conf is None:
        return None
    if not (_conf_ok(kpt_conf, _C.KP_CHEST) and _conf_ok(kpt_conf, _C.KP_HIP)):
        return None

    chest = np.asarray(kpts[_C.KP_CHEST], dtype=np.float64)
    hip   = np.asarray(kpts[_C.KP_HIP],   dtype=np.float64)
    body_axis = hip - chest
    body_len  = _norm(body_axis)
    if body_len < 1e-6:
        return None
    body_axis_unit = body_axis / body_len
    body_normal    = _perp(body_axis_unit)
    eff_len = max(_C.BODY_LEN_MIN_PX, min(_C.BODY_LEN_MAX_PX, body_len))

    if _conf_ok(kpt_conf, _C.KP_MID_BACK):
        torso_center = np.asarray(kpts[_C.KP_MID_BACK], dtype=np.float64)
    else:
        torso_center = 0.5 * (chest + hip)

    # Head target
    left_ok  = _conf_ok(kpt_conf, _C.KP_LEFT_EAR)
    right_ok = _conf_ok(kpt_conf, _C.KP_RIGHT_EAR)
    if left_ok and right_ok:
        head_center = 0.5 * (np.asarray(kpts[_C.KP_LEFT_EAR], dtype=np.float64)
                              + np.asarray(kpts[_C.KP_RIGHT_EAR], dtype=np.float64))
    elif _conf_ok(kpt_conf, _C.KP_NOSE):
        head_center = np.asarray(kpts[_C.KP_NOSE], dtype=np.float64)
    else:
        head_center = chest - body_axis_unit * eff_len * 0.5
    head_radius = eff_len * _C.HEAD_RADIUS_RATIO
    neck_radius = eff_len * _C.NECK_RADIUS_RATIO

    # Ventral-side sign: legs hang on the belly side regardless of camera
    # angle, so the average confident-knee side of the spine axis is
    # "abdomen"; the opposite side is "side/back".
    knee_idxs = (_C.KP_FL_KNEE, _C.KP_FR_KNEE, _C.KP_HL_KNEE, _C.KP_HR_KNEE)
    knee_pts = [np.asarray(kpts[i], dtype=np.float64)
                for i in knee_idxs if _conf_ok(kpt_conf, i, _C.LIMB_CONF_THRESHOLD)]
    if knee_pts:
        avg_knee = np.mean(knee_pts, axis=0)
        ventral_sign = 1.0 if float(np.dot(avg_knee - torso_center, body_normal)) >= 0.0 else -1.0
    else:
        ventral_sign = 1.0

    torso_ru = max(1e-6, eff_len * _C.TORSO_HALF_LEN_RATIO)
    torso_rv = max(1e-6, eff_len * _C.TORSO_HALF_WIDTH_RATIO)

    # Forelimb / hindlimb (left and right merged into one zone each)
    limb_groups = {
        "FORELIMB": ((_C.KP_FL_KNEE, _C.KP_FL_PAW), (_C.KP_FR_KNEE, _C.KP_FR_PAW)),
        "HINDLIMB": ((_C.KP_HL_KNEE, _C.KP_HL_PAW), (_C.KP_HR_KNEE, _C.KP_HR_PAW)),
    }
    limb_strip_hw = eff_len * _C.LIMB_STRIP_HW_RATIO
    paw_radius    = eff_len * _C.LIMB_PAW_RADIUS_RATIO

    limbs = {}
    for group, pairs in limb_groups.items():
        segments, paws = [], []
        for knee_idx, paw_idx in pairs:
            if (_conf_ok(kpt_conf, knee_idx, _C.LIMB_CONF_THRESHOLD)
                    and _conf_ok(kpt_conf, paw_idx, _C.LIMB_CONF_THRESHOLD)):
                knee = np.asarray(kpts[knee_idx], dtype=np.float64)
                paw  = np.asarray(kpts[paw_idx],  dtype=np.float64)
                segments.append((knee, paw))
                paws.append(paw)
        limbs[group] = {"segments": segments, "paws": paws}

    # Tail: single shared strip through Root -> Mid -> Tip (no left/right split)
    tail_segs = []
    tail_idxs = (_C.KP_TAIL_ROOT, _C.KP_TAIL_MID, _C.KP_TAIL_TIP)
    if all(_conf_ok(kpt_conf, i, _C.LIMB_CONF_THRESHOLD) for i in tail_idxs):
        root = np.asarray(kpts[_C.KP_TAIL_ROOT], dtype=np.float64)
        mid  = np.asarray(kpts[_C.KP_TAIL_MID],  dtype=np.float64)
        tip  = np.asarray(kpts[_C.KP_TAIL_TIP],  dtype=np.float64)
        tail_segs = [(root, mid), (mid, tip)]
    tail_strip_hw = eff_len * _C.TAIL_STRIP_HW_RATIO

    return {
        "body_axis_unit": body_axis_unit,
        "body_normal":    body_normal,
        "torso_center":   torso_center,
        "torso_ru":       torso_ru,
        "torso_rv":       torso_rv,
        "ventral_sign":   ventral_sign,
        "head_center":    head_center,
        "head_radius":    head_radius,
        "neck_center":    chest,
        "neck_radius":    neck_radius,
        "limbs":          limbs,
        "limb_strip_hw":  limb_strip_hw,
        "paw_radius":     paw_radius,
        "tail_segs":      tail_segs,
        "tail_strip_hw":  tail_strip_hw,
    }


def classify_zone(nose_pt, targets: Optional[dict]) -> Tuple[int, str, float]:
    """
    Test the nose point against the zone targets.

    Priority (most specific first): limb paw circles > limb strips >
    tail strip > head (disabled) > neck/chest (disabled) >
    torso half (side/back vs abdomen).

    Returns (zone_id, zone_name, confidence).
    """
    if targets is None or nose_pt is None:
        return _C.ZONE_NO_TARGET, _C.ZONE_NAMES[_C.ZONE_NO_TARGET], 0.0

    pt = np.asarray(nose_pt, dtype=np.float64)

    for group, zone_id in (("FORELIMB", _C.ZONE_FORELIMB), ("HINDLIMB", _C.ZONE_HINDLIMB)):
        for paw in targets["limbs"][group]["paws"]:
            hit, d = _point_in_circle(pt, paw, targets["paw_radius"])
            if hit:
                conf = max(0.0, min(1.0, 1.0 - d / max(targets["paw_radius"], 1e-6)))
                return zone_id, _C.ZONE_NAMES[zone_id], conf

    for group, zone_id in (("FORELIMB", _C.ZONE_FORELIMB), ("HINDLIMB", _C.ZONE_HINDLIMB)):
        for p0, p1 in targets["limbs"][group]["segments"]:
            hit, perp = _point_on_strip(pt, p0, p1, targets["limb_strip_hw"])
            if hit:
                conf = max(0.0, min(1.0, 1.0 - perp / max(targets["limb_strip_hw"], 1e-6)))
                return zone_id, _C.ZONE_NAMES[zone_id], conf

    for p0, p1 in targets["tail_segs"]:
        hit, perp = _point_on_strip(pt, p0, p1, targets["tail_strip_hw"])
        if hit:
            conf = max(0.0, min(1.0, 1.0 - perp / max(targets["tail_strip_hw"], 1e-6)))
            return _C.ZONE_TAIL, _C.ZONE_NAMES[_C.ZONE_TAIL], conf

    # 頭部區域的判定刻意停用：head_center/head_radius 以耳朵中點（或鼻子本身）
    # 為圓心，鼻子幾乎必然落在自己頭部的圓圈內——不管貓有沒有在舔頭部，只要
    # 沒有明顯把頭伸向其他部位，這裡都會誤判命中。這不是「舔頭部的時間」，
    # 而是「頭沒有轉向其他部位的時間」，統計上沒有意義，故直接跳過此判定，
    # 讓鼻子落在頭部圓圈內、又不在四肢/尾巴範圍內時歸類為 NO_TARGET。
    # （head_center/head_radius 仍保留在 targets 內，供未來需要時使用。）

    # 胸口區域的判定同樣刻意停用：本系統的關鍵點設計裡沒有獨立的「頸部」點，
    # 鼻子在骨架連結上直接接到胸口（BODY_LINKS: nose(0)->chest(3)），neck_center
    # 就是 KP_CHEST 本身，跟鼻子只隔一節骨架連結、距離天生就很近。只要貓咪
    # 低頭理毛——不管實際舔的是四肢以外的哪個部位（軀幹、腹部、側背都需要
    # 頭部前傾）——鼻子在移動路徑上幾乎必然會先經過胸口附近，導致這裡誤判
    # 命中、把本該算在軀幹（ABDOMEN/SIDE_BACK）的接觸時間搶走。跟上面 HEAD
    # 停用是同一種「判定點天生緊貼參考點」的結構性偏誤，故一併跳過此判定，
    # 讓鼻子落在胸口圓圈內、又不在四肢/尾巴範圍內時改落到下方軀幹橢圓判定
    # （neck_center/neck_radius 仍保留在 targets 內，供未來需要時使用）。

    rel = pt - targets["torso_center"]
    u = float(np.dot(rel, targets["body_axis_unit"]))
    v = float(np.dot(rel, targets["body_normal"]))
    ru, rv = targets["torso_ru"], targets["torso_rv"]
    norm_d = math.sqrt((u / max(ru, 1e-6)) ** 2 + (v / max(rv, 1e-6)) ** 2)
    if norm_d <= 1.0:
        conf = max(0.0, min(1.0, 1.0 - norm_d))
        is_ventral = (v >= 0.0) == (targets["ventral_sign"] >= 0.0)
        if is_ventral:
            return _C.ZONE_ABDOMEN, _C.ZONE_NAMES[_C.ZONE_ABDOMEN], conf
        return _C.ZONE_SIDE_BACK, _C.ZONE_NAMES[_C.ZONE_SIDE_BACK], conf

    return _C.ZONE_NO_TARGET, _C.ZONE_NAMES[_C.ZONE_NO_TARGET], 0.0


def _xy(p) -> list:
    return [round(float(p[0]), 1), round(float(p[1]), 1)]


def targets_to_geometry_payload(targets: Optional[dict]) -> dict:
    """Convert already-computed zone shapes into JSON-safe raw pixel
    coordinates for client-side (Node-RED) drawing. No new geometry is
    computed here — this only re-packages `targets` from build_zone_targets().
    """
    if targets is None:
        return {}

    def _strip(p0, p1) -> dict:
        return {"p0": _xy(p0), "p1": _xy(p1)}

    def _circle(center, radius) -> dict:
        return {"cx": round(float(center[0]), 1), "cy": round(float(center[1]), 1), "r": round(float(radius), 1)}

    forelimb = targets["limbs"]["FORELIMB"]
    hindlimb = targets["limbs"]["HINDLIMB"]

    return {
        "head": _circle(targets["head_center"], targets["head_radius"]),
        "neck": _circle(targets["neck_center"], targets["neck_radius"]),
        "torso": {
            "cx": round(float(targets["torso_center"][0]), 1),
            "cy": round(float(targets["torso_center"][1]), 1),
            "ux": round(float(targets["body_axis_unit"][0]), 4),
            "uy": round(float(targets["body_axis_unit"][1]), 4),
            "vx": round(float(targets["body_normal"][0]), 4),
            "vy": round(float(targets["body_normal"][1]), 4),
            "ru": round(float(targets["torso_ru"]), 1),
            "rv": round(float(targets["torso_rv"]), 1),
            "ventral_sign": targets["ventral_sign"],
        },
        "forelimb_segs": [_strip(p0, p1) for p0, p1 in forelimb["segments"]],
        "forelimb_paws": [_circle(p, targets["paw_radius"]) for p in forelimb["paws"]],
        "hindlimb_segs": [_strip(p0, p1) for p0, p1 in hindlimb["segments"]],
        "hindlimb_paws": [_circle(p, targets["paw_radius"]) for p in hindlimb["paws"]],
        "tail_segs":     [_strip(p0, p1) for p0, p1 in targets["tail_segs"]],
        "limb_hw":       round(float(targets["limb_strip_hw"]), 1),
        "tail_hw":       round(float(targets["tail_strip_hw"]), 1),
    }
