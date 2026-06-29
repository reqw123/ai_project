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

    def to_payload(self) -> dict:
        return {
            "current_zone":    self.current_zone,
            "best_zone":       self.best_zone,
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
        }
