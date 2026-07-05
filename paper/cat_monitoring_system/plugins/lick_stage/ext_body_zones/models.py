import math
from dataclasses import dataclass, field


def _jf(v, digits: int = 3):
    """Return None for NaN/inf (JSON-safe), otherwise round to digits."""
    if not isinstance(v, float) or not math.isfinite(v):
        return None
    return round(v, digits)


@dataclass
class ZoneStat:
    hits: int = 0
    time_sec: float = 0.0


@dataclass
class ExtZoneResult:
    current_zone: int = 0          # 0=NO_TARGET, 1..7 per ExtZoneConfig.ZONE_*
    zone_name: str = "NO_TARGET"
    confidence: float = 0.0
    valid: bool = False
    frame: int = 0
    time_sec: float = 0.0
    hits: int = 0                  # cumulative hits for current_zone
    zone_time_sec: float = 0.0     # cumulative time_sec for current_zone
    # Per-zone breakdown for all 7 zones — {zone_name: ZoneStat}, for the
    # Node-RED "部位時長統計" table (mirrors plugins/lick_stage's per-zone fields)
    zone_breakdown: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "current_zone":  self.current_zone,
            "zone_name":     self.zone_name,
            "confidence":    _jf(self.confidence, 3),
            "valid":         self.valid,
            "frame":         self.frame,
            "time_sec":      round(self.time_sec, 2),
            "hits":          self.hits,
            "zone_time_sec": round(self.zone_time_sec, 2),
            "zones": {
                name: {"hits": st.hits, "time_sec": round(st.time_sec, 2)}
                for name, st in self.zone_breakdown.items()
            },
        }
