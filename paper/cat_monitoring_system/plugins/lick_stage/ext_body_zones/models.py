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
        # 佔比（%）與「今日理毛之最」都在 Python 端算好，Node-RED 只負責顯示，
        # 不重複做這個算術（模組化原則；分母＝7 區累積時間總和）。
        total = sum(st.time_sec for st in self.zone_breakdown.values())

        def _pct(t: float) -> float:
            return round(t / total * 100, 2) if total > 1e-9 else 0.0

        dominant_zone, dominant_pct = None, 0.0
        if total > 1e-9:
            dominant_zone = max(self.zone_breakdown, key=lambda k: self.zone_breakdown[k].time_sec)
            dominant_pct  = _pct(self.zone_breakdown[dominant_zone].time_sec)

        # Face(Wash) 對照指標：文獻 Eckstein & Hart (2000) 的 Face(Wash) zone
        # 定義是「頭部＋前肢」合計 31%。但本系統的頭部判定（鼻子點是否落在
        # 頭部圓圈內）先天不可靠——頭部圓心就在鼻子附近，貓只要沒有明顯把頭
        # 轉向別處，鼻子幾乎必然落在頭部圓圈內，不能代表「正在舔頭部」，故
        # regions.py 的 classify_zone() 已停用頭部判定，HEAD 區恒為 0。
        # 這裡改用判定較可靠的前肢（需要鼻子明確伸向膝蓋/爪子附近才會命中）
        # 佔比，近似對照文獻的 31%，非嚴格對應文獻定義，僅供合理性參考。
        face_wash_proxy_pct = _pct(self.zone_breakdown.get("FORELIMB", ZoneStat()).time_sec)

        return {
            "current_zone":  self.current_zone,
            "zone_name":     self.zone_name,
            "confidence":    _jf(self.confidence, 3),
            "valid":         self.valid,
            "frame":         self.frame,
            "time_sec":      round(self.time_sec, 2),
            "hits":          self.hits,
            "zone_time_sec": round(self.zone_time_sec, 2),
            "total_lick_time":         round(total, 2),
            "dominant_zone":           dominant_zone,
            "dominant_pct":            dominant_pct,
            "face_wash_proxy_pct":     face_wash_proxy_pct,
            "zones": {
                name: {"hits": st.hits, "time_sec": round(st.time_sec, 2), "pct": _pct(st.time_sec)}
                for name, st in self.zone_breakdown.items()
            },
        }
