"""
Geometric helpers and lick-zone contact detection.

All functions are pure (no side effects, no I/O).
"""
import math
from typing import Optional, Tuple
import numpy as np
from plugins.lick_stage.config import LickConfig as _C

# Precomputed ellipse boundary angles (module-level for performance)
_ANGLES = np.linspace(0.0, 2.0 * np.pi, _C.ELLIPSE_SAMPLES, endpoint=False)
_ECOS   = np.cos(_ANGLES)
_ESIN   = np.sin(_ANGLES)


# ── Private geometry primitives ───────────────────────────────────────────────

def _point_in_oriented_ellipse(point, center, axis_u, axis_v, radius_u, radius_v) -> bool:
    rel = point - center
    u   = float(np.dot(rel, axis_u))
    v   = float(np.dot(rel, axis_v))
    ru2 = max(float(radius_u) ** 2, 1e-9)
    rv2 = max(float(radius_v) ** 2, 1e-9)
    return (u * u) / ru2 + (v * v) / rv2 <= 1.0 + 1e-9


def _sample_ellipse_boundary(center, axis_u, axis_v, radius_u, radius_v):
    c  = np.asarray(center, dtype=np.float64)
    eu = np.asarray(axis_u, dtype=np.float64)
    ev = np.asarray(axis_v, dtype=np.float64)
    return c + np.outer(_ECOS * float(radius_u), eu) + np.outer(_ESIN * float(radius_v), ev)


def _distance_point_to_segment(point, seg_start, seg_end) -> float:
    p  = np.asarray(point,     dtype=np.float64)
    a  = np.asarray(seg_start, dtype=np.float64)
    b  = np.asarray(seg_end,   dtype=np.float64)
    ab = b - a
    ab2 = float(ab[0]) ** 2 + float(ab[1]) ** 2
    if ab2 < 1e-12:
        return math.hypot(float(p[0] - a[0]), float(p[1] - a[1]))
    pa = p - a
    t  = max(0.0, min(1.0, (float(pa[0]) * float(ab[0]) + float(pa[1]) * float(ab[1])) / ab2))
    return math.hypot(float(pa[0]) - t * float(ab[0]), float(pa[1]) - t * float(ab[1]))


def _compute_strip_corners(p0, p1, half_width):
    """Return 4 corners (CCW) of a rectangle around segment p0-p1, or None."""
    seg = p1 - p0
    length = math.hypot(float(seg[0]), float(seg[1]))
    if length < 1e-6:
        return None
    axis   = seg / length
    normal = np.array([-float(axis[1]), float(axis[0])], dtype=np.float64)
    off    = normal * half_width
    return [p0 + off, p1 + off, p1 - off, p0 - off]


def _point_in_polygon(point, polygon) -> bool:
    """Ray-casting point-in-polygon test with boundary tolerance."""
    p    = np.asarray(point,   dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False
    x, y = float(p[0]), float(p[1])
    n      = poly.shape[0]
    inside = False
    for i in range(n):
        j        = (i - 1) % n
        xi, yi   = float(poly[i, 0]), float(poly[i, 1])
        xj, yj   = float(poly[j, 0]), float(poly[j, 1])
        sx, sy   = xj - xi, yj - yi
        rx, ry   = x - xi, y - yi
        seg_norm = math.hypot(sx, sy)
        if seg_norm > 1e-9:
            area2 = abs(sx * ry - sy * rx)
            dotv  = rx * sx + ry * sy
            if area2 / seg_norm <= 1e-6 and -1e-9 <= dotv <= sx * sx + sy * sy + 1e-9:
                return True
        dy = yj - yi
        if abs(dy) < 1e-12:
            dy = 1e-12
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / dy + xi):
            inside = not inside
    return inside


def _segments_intersect(p1, p2, q1, q2) -> bool:
    def _orient(a, b, c):
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _on_seg(a, b, c):
        return (
            min(float(a[0]), float(b[0])) - 1e-9 <= float(c[0]) <= max(float(a[0]), float(b[0])) + 1e-9
            and min(float(a[1]), float(b[1])) - 1e-9 <= float(c[1]) <= max(float(a[1]), float(b[1])) + 1e-9
        )

    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    o1, o2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    o3, o4 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True
    if abs(o1) <= 1e-9 and _on_seg(p1, p2, q1): return True
    if abs(o2) <= 1e-9 and _on_seg(p1, p2, q2): return True
    if abs(o3) <= 1e-9 and _on_seg(q1, q2, p1): return True
    if abs(o4) <= 1e-9 and _on_seg(q1, q2, p2): return True
    return False


def _polygons_intersect(poly_a, poly_b) -> bool:
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 3 or b.shape[0] < 3:
        return False
    for pa in a:
        if _point_in_polygon(pa, b): return True
    for pb in b:
        if _point_in_polygon(pb, a): return True
    na, nb = a.shape[0], b.shape[0]
    for i in range(na):
        a0, a1 = a[i], a[(i + 1) % na]
        for j in range(nb):
            if _segments_intersect(a0, a1, b[j], b[(j + 1) % nb]):
                return True
    return False


def _polygon_aabb(poly) -> Tuple[float, float, float, float]:
    p = np.asarray(poly, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (float(np.min(p[:, 0])), float(np.min(p[:, 1])),
            float(np.max(p[:, 0])), float(np.max(p[:, 1])))


def _aabb_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _polygon_contacts_circle(poly_pts, center, radius) -> bool:
    poly = np.asarray(poly_pts, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False
    c = np.asarray(center, dtype=np.float64)
    r = max(float(radius), 0.0)
    if r <= 0.0:
        return False
    if not _aabb_overlap(_polygon_aabb(poly),
                         (float(c[0] - r), float(c[1] - r),
                          float(c[0] + r), float(c[1] + r))):
        return False
    if _point_in_polygon(c, poly):
        return True
    for p in poly:
        if math.hypot(float(p[0]) - float(c[0]), float(p[1]) - float(c[1])) <= r + 1e-9:
            return True
    n = poly.shape[0]
    for i in range(n):
        if _distance_point_to_segment(c, poly[i], poly[(i + 1) % n]) <= r + 1e-9:
            return True
    return False


def _polygon_contacts_oriented_ellipse(poly_pts, center, axis_u, axis_v, radius_u, radius_v) -> bool:
    poly = np.asarray(poly_pts, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False
    c  = np.asarray(center, dtype=np.float64)
    eu = np.asarray(axis_u, dtype=np.float64)
    ev = np.asarray(axis_v, dtype=np.float64)
    ru, rv = float(radius_u), float(radius_v)
    ex = abs(float(eu[0])) * ru + abs(float(ev[0])) * rv
    ey = abs(float(eu[1])) * ru + abs(float(ev[1])) * rv
    if not _aabb_overlap(_polygon_aabb(poly),
                         (float(c[0] - ex), float(c[1] - ey),
                          float(c[0] + ex), float(c[1] + ey))):
        return False
    if _point_in_polygon(c, poly):
        return True
    for p in poly:
        if _point_in_oriented_ellipse(p, c, eu, ev, ru, rv):
            return True
    for p in _sample_ellipse_boundary(c, eu, ev, ru, rv):
        if _point_in_polygon(p, poly):
            return True
    return False


# ── Limb region builders ──────────────────────────────────────────────────────

def _build_limb_joint_targets(kpts, kpt_conf, body_len: float) -> list:
    """Circle regions at each paw keypoint."""
    eff_len    = max(_C.CONTACT_BODY_LEN_MIN_PX, min(_C.CONTACT_BODY_LEN_MAX_PX, body_len))
    paw_radius = max(1e-6, eff_len * _C.LIMB_PAW_CIRCLE_R_RATIO * _C.LIMB_CONTACT_SCALE)
    targets    = []
    for group, _knee_idx, paw_idx in _C.LIMB_SEGMENTS:
        if float(kpt_conf[paw_idx]) > _C.LIMB_CONF_THRESHOLD:
            targets.append({
                "group":  group,
                "center": np.asarray(kpts[paw_idx], dtype=np.float64),
                "radius": paw_radius,
            })
    return targets


def _build_limb_strip_targets(kpts, kpt_conf, body_len: float) -> list:
    """Rectangle strips along knee-to-paw segments."""
    eff_len    = max(_C.CONTACT_BODY_LEN_MIN_PX, min(_C.CONTACT_BODY_LEN_MAX_PX, body_len))
    half_w     = max(1e-6, eff_len * _C.LIMB_STRIP_HW_RATIO * _C.LIMB_CONTACT_SCALE)
    edge_gap   = max(0.0,  eff_len * _C.LIMB_STRIP_EDGE_GAP  * _C.LIMB_CONTACT_SCALE)
    paw_radius = max(1e-6, eff_len * _C.LIMB_PAW_CIRCLE_R_RATIO * _C.LIMB_CONTACT_SCALE)
    strips     = []
    for group, knee_idx, paw_idx in _C.LIMB_SEGMENTS:
        if not (float(kpt_conf[knee_idx]) > _C.LIMB_CONF_THRESHOLD
                and float(kpt_conf[paw_idx]) > _C.LIMB_CONF_THRESHOLD):
            continue
        knee = np.asarray(kpts[knee_idx], dtype=np.float64)
        paw  = np.asarray(kpts[paw_idx],  dtype=np.float64)
        seg  = paw - knee
        seg_len = math.hypot(float(seg[0]), float(seg[1]))
        if seg_len < 1e-6:
            continue
        axis   = seg / seg_len
        s_start = knee + axis * edge_gap
        s_end   = paw  - axis * (paw_radius + edge_gap)
        if math.hypot(float((s_end - s_start)[0]), float((s_end - s_start)[1])) < 1e-3:
            continue
        corners = _compute_strip_corners(s_start, s_end, half_w)
        if corners is None:
            continue
        strips.append({
            "group":   group,
            "p0":      s_start,
            "p1":      s_end,
            "corners": corners,
        })
    return strips


# ── Public API ────────────────────────────────────────────────────────────────

def compute_geometry(kpts, kpt_conf) -> Optional[dict]:
    """
    Build head/body geometry dict from keypoints.

    Returns None when required keypoints (nose, chest, hip) are missing.
    """
    if kpts is None or kpt_conf is None:
        return None

    left_ok  = float(kpt_conf[_C.KP_LEFT_EAR])  > _C.EAR_CONF_THRESHOLD
    right_ok = float(kpt_conf[_C.KP_RIGHT_EAR]) > _C.EAR_CONF_THRESHOLD
    nose_ok  = float(kpt_conf[_C.KP_NOSE])      >= _C.NOSE_CONF_THRESHOLD
    chest_ok = float(kpt_conf[_C.KP_CHEST])     > _C.EAR_CONF_THRESHOLD
    hip_ok   = float(kpt_conf[_C.KP_HIP])       > _C.EAR_CONF_THRESHOLD

    if not (nose_ok and chest_ok and hip_ok):
        return None

    nose  = np.asarray(kpts[_C.KP_NOSE],  dtype=np.float64)
    chest = np.asarray(kpts[_C.KP_CHEST], dtype=np.float64)
    hip   = np.asarray(kpts[_C.KP_HIP],   dtype=np.float64)

    if left_ok and right_ok:
        left_ear  = np.asarray(kpts[_C.KP_LEFT_EAR],  dtype=np.float64)
        right_ear = np.asarray(kpts[_C.KP_RIGHT_EAR], dtype=np.float64)
        ear_center = 0.5 * (left_ear + right_ear)
        head_vec   = nose - ear_center
        head_norm  = math.hypot(float(head_vec[0]), float(head_vec[1]))
        head_dir   = head_vec / head_norm if head_norm > 1e-6 else np.zeros(2, dtype=np.float64)
    else:
        left_ear = right_ear = None
        ear_center = nose.copy()
        head_vec   = np.zeros(2, dtype=np.float64)
        head_dir   = np.zeros(2, dtype=np.float64)
        head_norm  = 0.0

    body_axis = hip - chest
    body_len  = math.hypot(float(body_axis[0]), float(body_axis[1]))
    if body_len < 1e-6:
        return None
    body_axis_unit = body_axis / body_len
    body_normal    = np.array([-float(body_axis_unit[1]), float(body_axis_unit[0])], dtype=np.float64)
    body_center    = 0.5 * (chest + hip)

    region_rx = max(1e-6, 0.5 * _C.BODY_ELLIPSE_W_RATIO * body_len)
    region_ry = max(1e-6, 0.5 * _C.BODY_ELLIPSE_H_RATIO * body_len)

    # Nose contact trapezoid
    eff_len       = max(_C.CONTACT_BODY_LEN_MIN_PX, min(_C.CONTACT_BODY_LEN_MAX_PX, body_len))
    px_per_cm     = max(eff_len / max(_C.CAT_BODY_LENGTH_CM, 1e-6), 1e-6)
    trap_height   = max(1e-6, _C.NOSE_TRAP_THICKNESS_CM * _C.NOSE_TRAP_THICKNESS_SCALE * px_per_cm)
    trap_top_half = max(1e-6, 0.5 * _C.NOSE_TRAP_TOP_W_RATIO * _C.NOSE_TRAP_W_SCALE * eff_len)
    trap_bot_half = max(1e-6, 0.5 * _C.NOSE_TRAP_BOT_W_RATIO * _C.NOSE_TRAP_W_SCALE * eff_len)

    if left_ok and right_ok:
        ear_line      = right_ear - left_ear
        ear_line_norm = math.hypot(float(ear_line[0]), float(ear_line[1]))
        trap_perp = ear_line / ear_line_norm if ear_line_norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
    else:
        bn = math.hypot(float(body_normal[0]), float(body_normal[1]))
        trap_perp = body_normal / bn if bn > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)

    trap_dir = np.array([-float(trap_perp[1]), float(trap_perp[0])], dtype=np.float64)
    if float(np.dot(trap_dir, body_center - nose)) < 0.0:
        trap_dir = -trap_dir

    trap_bottom = nose + trap_dir * trap_height
    if not math.isfinite(float(trap_bottom[1])) or float(trap_bottom[1]) <= float(nose[1]):
        trap_dir    = -trap_dir
        trap_bottom = nose + trap_dir * trap_height

    nose_trap = np.asarray([
        nose        - trap_perp * trap_top_half,
        nose        + trap_perp * trap_top_half,
        trap_bottom + trap_perp * trap_bot_half,
        trap_bottom - trap_perp * trap_bot_half,
    ], dtype=np.float64)

    limb_targets       = _build_limb_joint_targets(kpts, kpt_conf, body_len)
    limb_strip_targets = _build_limb_strip_targets(kpts, kpt_conf, body_len)

    return {
        "nose":                   nose,
        "ear_center":             ear_center,
        "body_center":            body_center,
        "body_normal":            body_normal,
        "body_axis_unit":         body_axis_unit,
        "body_len":               body_len,
        "region_rx":              region_rx,
        "region_ry":              region_ry,
        "nose_contact_trapezoid": nose_trap,
        "limb_targets":           limb_targets,
        "limb_strip_targets":     limb_strip_targets,
    }


def find_nearest_zone(target_geom) -> Tuple[str, float, bool]:
    """
    Test nose-contact trapezoid against all contact regions.

    Returns (zone_label, distance, hit).
    zone_label is BODY_CENTER, FL, FR, HL, HR, or NO_TARGET.
    """
    if target_geom is None:
        return _C.ZONE_NO_TARGET, float("nan"), False

    nose_pt  = target_geom.get("nose")
    nose_trap = np.asarray(target_geom.get("nose_contact_trapezoid", []), dtype=np.float64)
    if nose_pt is None or nose_trap.ndim != 2 or nose_trap.shape[0] != 4:
        return _C.ZONE_NO_TARGET, float("nan"), False

    candidates = []

    # Body center ellipse
    if _polygon_contacts_oriented_ellipse(
        nose_trap,
        target_geom["body_center"],
        np.asarray(target_geom["body_normal"],    dtype=np.float64),
        np.asarray(target_geom["body_axis_unit"], dtype=np.float64),
        float(target_geom["region_rx"]),
        float(target_geom["region_ry"]),
    ):
        d_body = _distance_point_to_segment(
            nose_pt,
            target_geom["body_center"] - 0.5 * float(target_geom["body_len"]) * np.asarray(target_geom["body_axis_unit"], dtype=np.float64),
            target_geom["body_center"] + 0.5 * float(target_geom["body_len"]) * np.asarray(target_geom["body_axis_unit"], dtype=np.float64),
        )
        candidates.append((d_body, _C.ZONE_BODY))

    # Limb paw circles (group → minimum distance)
    limb_dist = {g: float("inf") for g in ("FL", "FR", "HL", "HR")}
    for limb in target_geom.get("limb_targets", []):
        center = np.asarray(limb["center"], dtype=np.float64)
        if _polygon_contacts_circle(nose_trap, center, float(limb["radius"])):
            g = str(limb.get("group", ""))
            d = math.hypot(float(nose_pt[0]) - float(center[0]),
                           float(nose_pt[1]) - float(center[1]))
            if g in limb_dist and d < limb_dist[g]:
                limb_dist[g] = d

    # Limb strip rectangles
    for strip in target_geom.get("limb_strip_targets", []):
        corners = strip.get("corners")
        if corners is None or len(corners) != 4:
            continue
        if _polygons_intersect(nose_trap, np.asarray(corners, dtype=np.float64)):
            g = str(strip.get("group", ""))
            d = _distance_point_to_segment(nose_pt, strip["p0"], strip["p1"])
            if g in limb_dist and d < limb_dist[g]:
                limb_dist[g] = d

    for group, dist in limb_dist.items():
        if math.isfinite(dist):
            candidates.append((dist, group))

    if not candidates:
        return _C.ZONE_NO_TARGET, float("nan"), False

    d_min, label_min = min(candidates, key=lambda x: x[0])
    return label_min, float(d_min), True
