"""Per-frame lick analysis orchestrator."""
from collections import deque
from typing import Optional
import numpy as np

from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.models import LickResult, ZoneStats
from plugins.lick_stage.statistics import LickStatistics
from plugins.lick_stage.ear_distance import compute_ear_distance
from plugins.lick_stage.head_direction import (
    compute_head_ear_angle,
    infer_face_state_cat_centric,
    infer_face_state_user_rules,
    smooth_state,
    check_front_view_guard,
)
from plugins.lick_stage.contact_regions import compute_geometry, find_nearest_zone


class LickAnalyzer:
    """
    Stateful per-frame lick zone and face direction analyzer.

    Call analyze() once per frame in order.
    """

    def __init__(self):
        self._stats         = LickStatistics()
        self._state_history = deque(maxlen=_C.STATE_SMOOTH_WINDOW)
        self._ema_kpts: Optional[np.ndarray] = None
        # Last-frame overlay state (consumed by manager.draw_overlay)
        self.last_trap_pts:    Optional[np.ndarray] = None  # (4,2) float64 or None
        self.last_hit:         bool = False
        self.last_zone_label:  str  = "NO_TARGET"
        self.last_nose_xy:     tuple = (0.0, 0.0)
        self.last_geom:        Optional[dict] = None        # full target_geom dict
        self.last_nearest_label: str = "NO_TARGET"

    def analyze(
        self,
        kpts,
        kpt_conf,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
    ) -> LickResult:
        if kpts is None or kpt_conf is None:
            return self._handle_no_cat(frame_idx, elapsed_sec, dt_sec)
        return self._handle_cat(kpts, kpt_conf, frame_idx, elapsed_sec, dt_sec)

    def reset(self) -> None:
        self._stats.reset()
        self._state_history.clear()
        self._ema_kpts = None

    # ── Private helpers ───────────────────────────────────────────────

    def _handle_no_cat(self, frame_idx: int, elapsed_sec: float, dt_sec: float) -> LickResult:
        self._ema_kpts    = None
        self.last_trap_pts = None
        self.last_hit      = False
        self.last_geom     = None
        self.last_nearest_label = "NO_TARGET"
        self._state_history.append(_C.STATE_NO_CAT)
        state_sm, stability = smooth_state(self._state_history)
        self._stats.update("NO_TARGET", dt_sec)
        return self._build_result("NO_TARGET", state_sm, stability, False, frame_idx, elapsed_sec)

    def _handle_cat(self, kpts, kpt_conf, frame_idx: int, elapsed_sec: float, dt_sec: float) -> LickResult:
        _nan = float("nan")

        # Optional keypoint EMA (default alpha=1.0 = bypass)
        if _C.EMA_ALPHA < 1.0 - 1e-9:
            if self._ema_kpts is None:
                self._ema_kpts = np.asarray(kpts, dtype=np.float64).copy()
            else:
                self._ema_kpts = _C.EMA_ALPHA * np.asarray(kpts, dtype=np.float64) + (1.0 - _C.EMA_ALPHA) * self._ema_kpts
            smooth_kpts = self._ema_kpts
        else:
            smooth_kpts = kpts

        dist_px, dist_norm, valid, _body_scale, body_ear_ratio = compute_ear_distance(smooth_kpts, kpt_conf)
        front_guard  = check_front_view_guard(kpt_conf, dist_px, body_ear_ratio)
        nose_conf    = float(kpt_conf[_C.KP_NOSE])
        nose_ok      = nose_conf >= _C.NOSE_CONF_THRESHOLD
        angle_deg    = compute_head_ear_angle(smooth_kpts, kpt_conf)

        gaze_fwd = gaze_lat = gaze_angle = _nan

        if front_guard:
            if _C.BACK_VIEW_REQUIRE_LOW_NOSE and nose_conf <= _C.BACK_CAMERA_NOSE_CONF_MAX:
                state_now = _C.STATE_BACK
            else:
                state_now = _C.STATE_FRONT_VIEW
            self._state_history.append(state_now)
            state_sm  = state_now
            stability = 1.0
            zone_label = "NO_TARGET"
            self.last_trap_pts     = None
            self.last_hit          = False
            self.last_geom         = None
            self.last_nearest_label = "NO_TARGET"
        else:
            target_geom = compute_geometry(smooth_kpts, kpt_conf)
            cat_state, gaze_fwd, gaze_lat, gaze_angle = infer_face_state_cat_centric(target_geom, nose_ok)
            state_now, rule_applied = infer_face_state_user_rules(angle_deg, dist_norm, dist_px, nose_conf)
            if state_now == _C.STATE_UNKNOWN:
                state_now = cat_state
            self._state_history.append(state_now)
            if rule_applied and state_now in (
                _C.STATE_FRONT, _C.STATE_FRONT_LEFT, _C.STATE_FRONT_RIGHT, _C.STATE_BACK
            ):
                state_sm  = state_now
                stability = 1.0
            else:
                state_sm, stability = smooth_state(self._state_history)

            nearest_label, _dist, hit = find_nearest_zone(target_geom)
            zone_label = nearest_label if hit else "NO_TARGET"

            # Store overlay state for draw_overlay()
            trap_raw = target_geom.get("nose_contact_trapezoid")
            self.last_trap_pts      = np.asarray(trap_raw, dtype=np.float64) if trap_raw is not None else None
            self.last_hit           = bool(hit)
            self.last_zone_label    = nearest_label
            self.last_nearest_label = nearest_label
            self.last_geom          = target_geom
            nose_kp = smooth_kpts[_C.KP_NOSE]
            self.last_nose_xy       = (float(nose_kp[0]), float(nose_kp[1]))

        self._stats.update(zone_label, dt_sec)
        return self._build_result(
            zone_label, state_sm, stability, valid, frame_idx, elapsed_sec,
            dist_px, dist_norm, gaze_fwd, gaze_lat, gaze_angle,
        )

    def _build_result(
        self,
        zone_label: str,
        state_sm: str,
        stability: float,
        valid: bool,
        frame_idx: int,
        elapsed_sec: float,
        dist_px:    float = float("nan"),
        dist_norm:  float = float("nan"),
        gaze_fwd:   float = float("nan"),
        gaze_lat:   float = float("nan"),
        gaze_angle: float = float("nan"),
    ) -> LickResult:
        def _zs(key: str) -> ZoneStats:
            hits, t = self._stats.zone_stats(key)
            return ZoneStats(hits=hits, time_sec=t)

        return LickResult(
            current_zone    = zone_label,
            best_zone       = self._stats.best_zone(),
            body            = _zs("BODY"),
            fl              = _zs("FL"),
            fr              = _zs("FR"),
            hl              = _zs("HL"),
            hr              = _zs("HR"),
            face_state      = state_sm,
            state_stability = stability,
            valid           = valid,
            frame           = frame_idx,
            time_sec        = elapsed_sec,
            dist_px         = dist_px,
            dist_norm       = dist_norm,
            gaze_fwd        = gaze_fwd,
            gaze_lat        = gaze_lat,
            gaze_angle      = gaze_angle,
        )
