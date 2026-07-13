import math
from dataclasses import dataclass, field


def _jf(v, digits: int = 3):
    """Return None for NaN/inf (JSON-safe), otherwise round to digits."""
    if not isinstance(v, float) or not math.isfinite(v):
        return None
    return round(v, digits)


@dataclass
class ZoneStats:
    hits: int = 0
    time_sec: float = 0.0


@dataclass
class LickResult:
    current_zone: str = "NO_TARGET"
    best_zone: str = "NO_TARGET"
    body: ZoneStats = field(default_factory=ZoneStats)
    fl: ZoneStats = field(default_factory=ZoneStats)
    fr: ZoneStats = field(default_factory=ZoneStats)
    hl: ZoneStats = field(default_factory=ZoneStats)
    hr: ZoneStats = field(default_factory=ZoneStats)
    face_state: str = "UNKNOWN"
    state_stability: float = 0.0
    valid: bool = False
    frame: int = 0
    time_sec: float = 0.0
    # Extended metrics used by Node-RED P3 panel
    dist_px: float = float("nan")
    dist_norm: float = float("nan")
    gaze_fwd: float = float("nan")
    gaze_lat: float = float("nan")
    gaze_angle: float = float("nan")
    # Raw geometry for client-side (Node-RED) visualization only — never
    # consumed by the core pipeline itself.
    trap_pts: list = field(default_factory=list)   # [[x,y]*4] or [] when no cat
    nose_xy: list = field(default_factory=list)    # [x, y] or []

    def to_payload(self) -> dict:
        # 佔比（%）：以 5 區累積時間總和為分母，在 Python 端算好直接送給
        # Node-RED，Node 端只負責顯示，不重複做這個算術（模組化原則）。
        total_time = (
            self.body.time_sec + self.fl.time_sec + self.fr.time_sec
            + self.hl.time_sec + self.hr.time_sec
        )

        def _pct(t: float) -> float:
            return round(t / total_time * 100, 2) if total_time > 1e-9 else 0.0

        _zone_map = {"BODY": self.body, "FL": self.fl, "FR": self.fr, "HL": self.hl, "HR": self.hr}
        _best_stat = _zone_map.get(self.best_zone)

        return {
            "current_zone":    self.current_zone,
            "best_zone":       self.best_zone,
            "best_pct":        _pct(_best_stat.time_sec) if _best_stat is not None else None,
            "body_time":       round(self.body.time_sec, 2),
            "fl_time":         round(self.fl.time_sec,   2),
            "fr_time":         round(self.fr.time_sec,   2),
            "hl_time":         round(self.hl.time_sec,   2),
            "hr_time":         round(self.hr.time_sec,   2),
            "body_hits":       self.body.hits,
            "fl_hits":         self.fl.hits,
            "fr_hits":         self.fr.hits,
            "hl_hits":         self.hl.hits,
            "hr_hits":         self.hr.hits,
            "body_pct":        _pct(self.body.time_sec),
            "fl_pct":          _pct(self.fl.time_sec),
            "fr_pct":          _pct(self.fr.time_sec),
            "hl_pct":          _pct(self.hl.time_sec),
            "hr_pct":          _pct(self.hr.time_sec),
            "total_lick_time": round(total_time, 2),
            "face_state":      self.face_state,
            "state_stability": round(self.state_stability, 3),
            "valid":           self.valid,
            "frame":           self.frame,
            "time_sec":        round(self.time_sec, 2),
            # Extended — null when keypoints unavailable
            "dist_px":    _jf(self.dist_px,   1),
            "dist_norm":  _jf(self.dist_norm,  4),
            "gaze_fwd":   _jf(self.gaze_fwd,   3),
            "gaze_lat":   _jf(self.gaze_lat,   3),
            "gaze_angle": _jf(self.gaze_angle, 1),
            # Visualization-only geometry (Node-RED draws this; core never reads it)
            "trap_pts": self.trap_pts,
            "nose_xy":  self.nose_xy,
        }
