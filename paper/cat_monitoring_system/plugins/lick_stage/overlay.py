"""Body/zone/nose overlay for lick-stage plugin."""
import math
import cv2
import numpy as np


# LICK_ZONE_CENTER label — must match contact_regions.py
_ZONE_CENTER = "BODY"

# Per-limb vivid colors (BGR): (hit_fill, hit_border, idle_fill, idle_border)
_PAW_COLORS: dict[str, tuple] = {
    "FL": ((0,  165, 255), (80,  220, 255), (0,   80, 140), (60, 140, 180)),  # orange
    "FR": ((0,  255, 100), (120, 255, 180), (0,  110,  45), (60, 160,  90)),  # lime
    "HL": ((220,  50, 200), (255, 120, 240), (120, 20, 120), (160, 60, 170)), # magenta
    "HR": ((255, 230,   0), (255, 255, 100), (140, 120,  0), (180, 160, 60)), # cyan-yellow
}
_PAW_DEFAULT = ((180, 180, 180), (230, 230, 230), (90, 90, 90), (140, 140, 140))


def draw_all_overlays(
    frame: np.ndarray,
    geom,
    trap_pts,
    hit: bool,
    zone_label: str,
    nearest_label: str,
    nose_xy: tuple,
    frame_idx: int = 0,
    show: bool = True,
) -> None:
    """Draw all lick-stage overlays onto *frame* in-place.

    Parameters
    ----------
    frame         : BGR image (modified in-place).
    geom          : Full target_geom dict from compute_geometry(), or None.
    trap_pts      : (4, 2) float64 trapezoid corners in pixel coords, or None.
    hit           : True when nose is inside a lick zone.
    zone_label    : Contacted zone label (e.g. "FL", "BODY", "NO_TARGET").
    nearest_label : Nearest zone label regardless of hit (used for highlight).
    nose_xy       : (x, y) pixel position of the nose keypoint.
    frame_idx     : Current frame index for pulse animation.
    show          : Master switch; no-op when False.
    """
    if not show:
        return

    render_h = frame.shape[0]
    _ov = max(0.6, render_h / 720.0)
    pulse = 0.5 + 0.5 * math.sin(frame_idx * 0.28)

    if geom is not None:
        _draw_body_region(frame, geom, nearest_label, hit, _ov)

    _draw_nose_trapezoid(frame, trap_pts, hit, zone_label, nose_xy, pulse, _ov)


# ── Legacy single-function alias (kept for back-compat) ──────────────────────

def draw_nose_trapezoid(
    frame: np.ndarray,
    trap_pts,
    hit: bool,
    zone_label: str,
    nose_xy: tuple,
    frame_idx: int = 0,
    show: bool = True,
) -> None:
    """Deprecated shim — prefer draw_all_overlays()."""
    if not show:
        return
    render_h = frame.shape[0]
    _ov = max(0.6, render_h / 720.0)
    pulse = 0.5 + 0.5 * math.sin(frame_idx * 0.28)
    _draw_nose_trapezoid(frame, trap_pts, hit, zone_label, nose_xy, pulse, _ov)


# ── Private drawing helpers ──────────────────────────────────────────────────

def _draw_body_region(
    frame: np.ndarray,
    geom: dict,
    nearest_label: str,
    hit: bool,
    _ov: float,
) -> None:
    """Draw body center, body ellipse, limb strips and paw circles.

    Uses a single region_overlay copy so all semi-transparent fills are
    composited in one cv2.addWeighted call (mirrors standalone script).
    """
    body_c = geom.get("body_center")
    if body_c is None:
        return

    body_pt = (int(body_c[0]), int(body_c[1]))

    # --- Body center dot (drawn directly — always fully opaque) ---
    cv2.circle(frame, body_pt, max(3, int(4 * _ov)), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, body_pt, max(2, int(2 * _ov)), (0, 180, 255),   -1, cv2.LINE_AA)

    # --- Prepare region_overlay for semi-transparent fills ---
    region_overlay = frame.copy()

    # Body ellipse fill
    region_rx = float(geom.get("region_rx", 0.0))
    region_ry = float(geom.get("region_ry", 0.0))
    axis_unit = geom.get("body_axis_unit")
    if region_rx > 0 and region_ry > 0 and axis_unit is not None:
        axes = (max(2, int(region_rx)), max(2, int(region_ry)))
        angle_deg = float(np.degrees(np.arctan2(axis_unit[1], axis_unit[0])))
        body_is_nearest = hit and nearest_label == _ZONE_CENTER
        fill_color = (30, 210, 80) if body_is_nearest else (80, 120, 240)
        cv2.ellipse(region_overlay, body_pt, axes, angle_deg, 0, 360,
                    fill_color, -1, cv2.LINE_AA)
    else:
        axes = None
        angle_deg = 0.0

    # Limb strips fill (collect border data for later polylines)
    strip_draw_data = []
    for strip in geom.get("limb_strip_targets", []):
        corners = strip.get("corners")
        if corners is None or len(corners) != 4:
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in corners], dtype=np.int32)
        zone_group = str(strip.get("group", ""))
        is_nearest = hit and nearest_label == zone_group
        fill   = (70, 180, 220) if is_nearest else (95, 95, 95)
        border = (40, 220, 255) if is_nearest else (155, 155, 155)
        cv2.fillConvexPoly(region_overlay, pts, fill, cv2.LINE_AA)
        strip_draw_data.append((pts, border))

    # Collect paw data (NOT drawn into region_overlay — separate vivid pass below)
    limb_draw_data = []
    for limb in geom.get("limb_targets", []):
        center = np.asarray(limb.get("center"), dtype=np.float64)
        radius = float(limb.get("radius", 0.0))
        if not np.all(np.isfinite(center)) or radius <= 0.0:
            continue
        cx, cy = int(center[0]), int(center[1])
        rr = max(1, int(round(radius * 0.5 * 2.0)))  # sx=sy=1 → radius*0.5*(1+1)
        zone_group = str(limb.get("group", ""))
        is_nearest = hit and nearest_label == zone_group
        cols = _PAW_COLORS.get(zone_group, _PAW_DEFAULT)
        fill   = cols[0] if is_nearest else cols[2]
        border = cols[1] if is_nearest else cols[3]
        limb_draw_data.append(((cx, cy), rr, fill, border, is_nearest))

    # Single composite blend for body ellipse + limb strips
    cv2.addWeighted(region_overlay, 0.35, frame, 0.76, 0, frame)

    # Body ellipse border (drawn after blend, fully opaque)
    if axes is not None:
        cv2.ellipse(frame, body_pt, axes, angle_deg, 0, 360,
                    (230, 230, 230), max(1, int(1 * _ov)), cv2.LINE_AA)

    # Limb strip borders
    for pts, border in strip_draw_data:
        cv2.polylines(frame, [pts], True, border, max(1, int(1 * _ov)), cv2.LINE_AA)

    # Paw circles — separate high-alpha blend so vivid colors are clearly visible
    if limb_draw_data:
        paw_ov = frame.copy()
        for (cx, cy), rr, fill, _border, _is_nearest in limb_draw_data:
            cv2.circle(paw_ov, (cx, cy), rr, fill, -1, cv2.LINE_AA)
        cv2.addWeighted(paw_ov, 0.72, frame, 0.28, 0, frame)
        for (cx, cy), rr, _fill, border, is_nearest in limb_draw_data:
            thick = max(2, int(2 * _ov)) if is_nearest else max(1, int(1 * _ov))
            cv2.circle(frame, (cx, cy), rr, border, thick, cv2.LINE_AA)
            # Thin white ring for contrast against any background
            cv2.circle(frame, (cx, cy), rr + max(1, int(1 * _ov)),
                       (255, 255, 255), 1, cv2.LINE_AA)


def _draw_nose_trapezoid(
    frame: np.ndarray,
    trap_pts,
    hit: bool,
    zone_label: str,
    nose_xy: tuple,
    pulse: float,
    _ov: float,
) -> None:
    """Draw nose-contact trapezoid with pulsing border and contact ripple."""
    if trap_pts is None or np.asarray(trap_pts).ndim != 2 or np.asarray(trap_pts).shape[0] != 4:
        return

    trap_draw = np.array([[int(p[0]), int(p[1])] for p in trap_pts], dtype=np.int32)

    fill_color = (80, 255, 170) if hit else (60, 170, 255)
    glow_color = (40, 255, 220) if hit else (255, 210, 90)
    alpha      = 0.65 if hit else 0.36

    # Semi-transparent fill
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, trap_draw, fill_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Pulsing border
    glow_thick = max(2, int((4.0 + 2.0 * pulse) * _ov))
    edge_thick = max(1, int((2.0 + 1.0 * pulse) * _ov))
    cv2.polylines(frame, [trap_draw], True, glow_color,      glow_thick, cv2.LINE_AA)
    cv2.polylines(frame, [trap_draw], True, (255, 255, 255), edge_thick, cv2.LINE_AA)

    # Corner dots
    for idx, pt in enumerate(trap_draw):
        dot_r = max(2, int((3 + idx % 2) * _ov))
        cv2.circle(frame, (int(pt[0]), int(pt[1])), dot_r + 1, (0, 0, 0),      -1, cv2.LINE_AA)
        cv2.circle(frame, (int(pt[0]), int(pt[1])), dot_r,     (255, 255, 255), -1, cv2.LINE_AA)

    # Label text
    label = f"CONTACT -> {zone_label}" if hit else "NOSE CONTACT ZONE"
    tx = int(nose_xy[0]) + int(14 * _ov)
    ty = int(nose_xy[1]) + int(24 * _ov)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.40 * _ov, (0, 0, 0),   max(2, int(2 * _ov)), cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.40 * _ov, glow_color,  max(1, int(1 * _ov)), cv2.LINE_AA)

    # Contact ripple
    if hit:
        nose_pt  = (int(nose_xy[0]), int(nose_xy[1]))
        ripple_r = max(5, int((8 + 10 * pulse) * _ov))
        cv2.circle(frame, nose_pt, ripple_r, (80, 255, 255),  max(2, int(3 * _ov)), cv2.LINE_AA)
        cv2.circle(frame, nose_pt, max(2, int(2 * _ov)), (255, 255, 255), -1, cv2.LINE_AA)
