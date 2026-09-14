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
    rel = np.asarray(pt, dtype=np.float64) - p0
    t = float(np.dot(rel, axis))
    perp = float(np.dot(rel, _perp(axis)))
    hit = (0.0 <= t <= seg_len) and (abs(perp) <= half_width)
    return hit, abs(perp)


def _bbox_diagonal(kpts, kpt_conf) -> float:
    """Bbox diagonal of every keypoint above LIMB_CONF_THRESHOLD (the lowest
    existing bar) — only used as the last-resort scale fallback, so it makes
    no assumption about which specific keypoints are visible."""
    pts = [
        np.asarray(kpts[i], dtype=np.float64)
        for i in range(len(kpt_conf))
        if float(kpt_conf[i]) > _C.LIMB_CONF_THRESHOLD
    ]
    if len(pts) < 2:
        return float("nan")
    arr = np.asarray(pts, dtype=np.float64)
    span = arr.max(axis=0) - arr.min(axis=0)
    return math.hypot(float(span[0]), float(span[1]))


def _compute_body_scale(
    kpts, kpt_conf, chest, hip, mid_back_ok: bool, chest_hip_len: float
) -> Tuple[float, str]:
    """Hybrid scale (M5), replacing the old absolute BODY_LEN_MIN/MAX_PX clamp.

    Priority: chest-midback-hip path length (resists curl-compression of the
    straight-line distance) -> straight chest-hip distance -> bbox diagonal *
    BBOX_TO_BODY_LEN_RATIO. See config.py for the rationale.
    """
    if mid_back_ok:
        mid_back = np.asarray(kpts[_C.KP_MID_BACK], dtype=np.float64)
        spine_len = _norm(chest - mid_back) + _norm(mid_back - hip)
        if spine_len > _C.SCALE_DEGENERATE_LEN_PX:
            return spine_len, "spine_path"

    if chest_hip_len > _C.SCALE_DEGENERATE_LEN_PX:
        return chest_hip_len, "chest_hip"

    bbox_diag = _bbox_diagonal(kpts, kpt_conf)
    if math.isfinite(bbox_diag) and bbox_diag > 1e-6:
        return bbox_diag * _C.BBOX_TO_BODY_LEN_RATIO, "bbox_fallback"

    return max(chest_hip_len, 1e-6), "degenerate"


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
    hip = np.asarray(kpts[_C.KP_HIP], dtype=np.float64)
    body_axis = hip - chest
    body_len = _norm(body_axis)
    if body_len < 1e-6:
        return None
    body_axis_unit = body_axis / body_len
    body_normal = _perp(body_axis_unit)
    mid_back_ok = _conf_ok(kpt_conf, _C.KP_MID_BACK)
    eff_len, scale_source = _compute_body_scale(
        kpts, kpt_conf, chest, hip, mid_back_ok, body_len
    )

    if mid_back_ok:
        torso_center = np.asarray(kpts[_C.KP_MID_BACK], dtype=np.float64)
    else:
        torso_center = 0.5 * (chest + hip)

    # Head target
    left_ok = _conf_ok(kpt_conf, _C.KP_LEFT_EAR)
    right_ok = _conf_ok(kpt_conf, _C.KP_RIGHT_EAR)
    if left_ok and right_ok:
        head_center = 0.5 * (
            np.asarray(kpts[_C.KP_LEFT_EAR], dtype=np.float64)
            + np.asarray(kpts[_C.KP_RIGHT_EAR], dtype=np.float64)
        )
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
    knee_pts = [
        np.asarray(kpts[i], dtype=np.float64)
        for i in knee_idxs
        if _conf_ok(kpt_conf, i, _C.LIMB_CONF_THRESHOLD)
    ]
    if knee_pts:
        avg_knee = np.mean(knee_pts, axis=0)
        ventral_sign = (
            1.0 if float(np.dot(avg_knee - torso_center, body_normal)) >= 0.0 else -1.0
        )
        ventral_sign_known = True
    else:
        # M5：沒有任何信心足夠的膝蓋關鍵點可用時，+1.0 只是任意預設值，不是
        # 真的證據——ventral_sign_known=False 讓 classify_zone() 知道這件事，
        # 命中軀幹橢圓時改回傳 TORSO_UNSPECIFIED，不要拿這個假的預設值去猜
        # ABDOMEN/SIDE_BACK。
        ventral_sign = 1.0
        ventral_sign_known = False

    torso_ru = max(1e-6, eff_len * _C.TORSO_HALF_LEN_RATIO)
    torso_rv = max(1e-6, eff_len * _C.TORSO_HALF_WIDTH_RATIO)

    # Forelimb / hindlimb (left and right merged into one zone each)
    limb_groups = {
        "FORELIMB": ((_C.KP_FL_KNEE, _C.KP_FL_PAW), (_C.KP_FR_KNEE, _C.KP_FR_PAW)),
        "HINDLIMB": ((_C.KP_HL_KNEE, _C.KP_HL_PAW), (_C.KP_HR_KNEE, _C.KP_HR_PAW)),
    }
    limb_strip_hw = eff_len * _C.LIMB_STRIP_HW_RATIO
    paw_radius = eff_len * _C.LIMB_PAW_RADIUS_RATIO

    limbs = {}
    for group, pairs in limb_groups.items():
        segments, paws = [], []
        for knee_idx, paw_idx in pairs:
            if _conf_ok(kpt_conf, knee_idx, _C.LIMB_CONF_THRESHOLD) and _conf_ok(
                kpt_conf, paw_idx, _C.LIMB_CONF_THRESHOLD
            ):
                knee = np.asarray(kpts[knee_idx], dtype=np.float64)
                paw = np.asarray(kpts[paw_idx], dtype=np.float64)
                segments.append((knee, paw))
                paws.append(paw)
        limbs[group] = {"segments": segments, "paws": paws}

    # Tail: single shared strip through Root -> Mid -> Tip (no left/right split)
    tail_segs = []
    tail_idxs = (_C.KP_TAIL_ROOT, _C.KP_TAIL_MID, _C.KP_TAIL_TIP)
    if all(_conf_ok(kpt_conf, i, _C.LIMB_CONF_THRESHOLD) for i in tail_idxs):
        root = np.asarray(kpts[_C.KP_TAIL_ROOT], dtype=np.float64)
        mid = np.asarray(kpts[_C.KP_TAIL_MID], dtype=np.float64)
        tip = np.asarray(kpts[_C.KP_TAIL_TIP], dtype=np.float64)
        tail_segs = [(root, mid), (mid, tip)]
    tail_strip_hw = eff_len * _C.TAIL_STRIP_HW_RATIO

    return {
        "body_axis_unit": body_axis_unit,
        "body_normal": body_normal,
        "eff_len": eff_len,
        "scale_source": scale_source,  # "spine_path" / "chest_hip" / "bbox_fallback" / "degenerate"
        "torso_center": torso_center,
        "torso_ru": torso_ru,
        "torso_rv": torso_rv,
        "ventral_sign": ventral_sign,
        "ventral_sign_known": ventral_sign_known,
        "head_center": head_center,
        "head_radius": head_radius,
        "neck_center": chest,
        "neck_radius": neck_radius,
        "limbs": limbs,
        "limb_strip_hw": limb_strip_hw,
        "paw_radius": paw_radius,
        "tail_segs": tail_segs,
        "tail_strip_hw": tail_strip_hw,
    }


def classify_zone(nose_pt, targets: Optional[dict]) -> Tuple[int, str, float]:
    """
    Test the nose point against every zone shape, scoring every intersecting
    candidate instead of returning on the first priority-ordered hit (M5：
    候選評分 + AMBIGUOUS，取代舊版「四肢腳掌圓 > 四肢長條 > 尾巴 > 軀幹」固定
    優先序判定——跟 lick_stage/contact_regions.py 的 find_nearest_zone() 同一套
    設計，讓兩個姊妹外掛的候選評分機制一致）。

    每個命中候選都換算成正規化分數 ∈ [0,1]（1 - 距離/該候選區域的特徵尺度），
    分數越接近 1 代表鼻尖越貼近該區域中心/骨架。前肢/後肢的 paw 圓跟 strip
    長條算同一個 zone，取兩者裡「命中時距離最短」的當該 zone 的候選分數。
    當最高分跟次高分（不同 zone）差距小於 AMBIGUITY_MARGIN 時，回傳
    ZONE_AMBIGUOUS 而不是武斷選一個。

    頭部/胸口判定仍然停用（理由見下方），軀幹橢圓命中時可能是 ABDOMEN/
    SIDE_BACK，或 `targets["ventral_sign_known"]` 為 False 時的
    TORSO_UNSPECIFIED（見 build_zone_targets() 的說明）。

    Returns (zone_id, zone_name, geometry_score)。
    """
    if targets is None or nose_pt is None:
        return _C.ZONE_NO_TARGET, _C.ZONE_NAMES[_C.ZONE_NO_TARGET], 0.0

    pt = np.asarray(nose_pt, dtype=np.float64)
    candidates = []  # (score, zone_id)

    # Forelimb / hindlimb：paw 圓跟 strip 長條共用同一個 zone_id，兩者裡命中
    # 時距離最短的當這個 zone 的候選（跟 find_nearest_zone() 的 limb_dist
    # 同一套邏輯）。特徵尺度統一用 paw_radius 當分母——跟 strip 半寬同一比例
    # 級數，足夠當正規化分母（見 find_nearest_zone() 的 limb_scale 說明）。
    for group, zone_id in (
        ("FORELIMB", _C.ZONE_FORELIMB),
        ("HINDLIMB", _C.ZONE_HINDLIMB),
    ):
        best_d = float("inf")
        for paw in targets["limbs"][group]["paws"]:
            hit, d = _point_in_circle(pt, paw, targets["paw_radius"])
            if hit and d < best_d:
                best_d = d
        for p0, p1 in targets["limbs"][group]["segments"]:
            hit, perp = _point_on_strip(pt, p0, p1, targets["limb_strip_hw"])
            if hit and perp < best_d:
                best_d = perp
        if math.isfinite(best_d):
            scale = max(targets["paw_radius"], 1e-6)
            score = max(0.0, min(1.0, 1.0 - best_d / scale))
            candidates.append((score, zone_id))

    # Tail
    best_tail_d = float("inf")
    for p0, p1 in targets["tail_segs"]:
        hit, perp = _point_on_strip(pt, p0, p1, targets["tail_strip_hw"])
        if hit and perp < best_tail_d:
            best_tail_d = perp
    if math.isfinite(best_tail_d):
        scale = max(targets["tail_strip_hw"], 1e-6)
        score = max(0.0, min(1.0, 1.0 - best_tail_d / scale))
        candidates.append((score, _C.ZONE_TAIL))

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

    # Torso ellipse
    rel = pt - targets["torso_center"]
    u = float(np.dot(rel, targets["body_axis_unit"]))
    v = float(np.dot(rel, targets["body_normal"]))
    ru, rv = targets["torso_ru"], targets["torso_rv"]
    norm_d = math.sqrt((u / max(ru, 1e-6)) ** 2 + (v / max(rv, 1e-6)) ** 2)
    if norm_d <= 1.0:
        score = max(0.0, min(1.0, 1.0 - norm_d))
        if not targets.get("ventral_sign_known", True):
            # M5：鼻子確實碰到軀幹，但沒有膝蓋關鍵點可以判斷哪一側是腹側——
            # 誠實回傳「軀幹，腹/背未定」，不要拿 ventral_sign 的任意預設值
            # 硬猜 ABDOMEN 或 SIDE_BACK（見 build_zone_targets() 的說明）。
            torso_zone_id = _C.ZONE_TORSO_UNSPECIFIED
        else:
            is_ventral = (v >= 0.0) == (targets["ventral_sign"] >= 0.0)
            torso_zone_id = _C.ZONE_ABDOMEN if is_ventral else _C.ZONE_SIDE_BACK
        candidates.append((score, torso_zone_id))

    if not candidates:
        return _C.ZONE_NO_TARGET, _C.ZONE_NAMES[_C.ZONE_NO_TARGET], 0.0

    # 同一個 zone 可能同時被多個候選命中；每個 zone 只留最高分，才能正確比較
    # 「不同 zone 之間」是否構成 AMBIGUOUS。
    best_per_zone: dict = {}
    for score, zone in candidates:
        if zone not in best_per_zone or score > best_per_zone[zone]:
            best_per_zone[zone] = score
    ranked = sorted(best_per_zone.items(), key=lambda kv: kv[1], reverse=True)

    best_zone, best_score = ranked[0]
    if len(ranked) >= 2 and (best_score - ranked[1][1]) < _C.AMBIGUITY_MARGIN:
        return _C.ZONE_AMBIGUOUS, _C.ZONE_NAMES[_C.ZONE_AMBIGUOUS], best_score
    return best_zone, _C.ZONE_NAMES[best_zone], best_score


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
        return {
            "cx": round(float(center[0]), 1),
            "cy": round(float(center[1]), 1),
            "r": round(float(radius), 1),
        }

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
        "tail_segs": [_strip(p0, p1) for p0, p1 in targets["tail_segs"]],
        "limb_hw": round(float(targets["limb_strip_hw"]), 1),
        "tail_hw": round(float(targets["tail_strip_hw"]), 1),
    }
