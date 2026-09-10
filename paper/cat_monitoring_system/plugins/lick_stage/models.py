"""LickStagePlugin 的資料結構：單一舔舐區域統計與每幀完整分析結果。"""

import math
from dataclasses import dataclass, field


def _jf(v, digits: int = 3):
    """Return None for NaN/inf (JSON-safe), otherwise round to digits."""
    if not isinstance(v, float) or not math.isfinite(v):
        return None
    return round(v, digits)


@dataclass
class ZoneStats:
    """單一舔舐區域的累積命中次數與時長。"""

    hits: int = 0
    time_sec: float = 0.0


@dataclass
class LickResult:
    """單一影格的完整舔舐分析結果：各區域統計、臉部朝向與視覺化用幾何資訊。"""

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
    trap_pts: list = field(default_factory=list)  # [[x,y]*4] or [] when no cat
    nose_xy: list = field(default_factory=list)  # [x, y] or []

    # ── 第一階段 v2 契約欄位（說明書「狀態與原因碼重構」/「統計分母」）──────
    # 目前與 v1 欄位並存（shadow 模式）：v1 欄位維持原本語意不動，Node-RED
    # 舊面板照常運作；新面板改讀下列欄位。切換完成後 v1 欄位才會移除。
    schema_version: str = "1.0"
    session_id: str = ""
    source_timestamp: float = None  # 來源媒體時間（秒）；None 代表呼叫端未提供
    frame_state: str = "NO_CAT"  # 五種互斥狀態之一，見 analysis_context.FrameState
    reason_code: str = None  # LICK_UNASSIGNED / 無效幀時的原因碼
    observed_sec: float = 0.0  # 所有幀 dt 總和（含 NO_CAT）
    valid_observed_sec: float = 0.0  # 畫面裡有貓的 dt 總和
    no_cat_sec: float = 0.0
    stgcn_lick_sec: float = 0.0  # 動作辨識輸出的舔毛時間
    assigned_zone_sec: float = 0.0  # 可定位到部位的舔毛時間
    unassigned_lick_sec: float = 0.0  # 幾何模組失敗的舔毛時間
    zone_coverage_ratio: float = 0.0  # assigned ÷ stgcn
    unknown_rate: float = 0.0  # unassigned ÷ stgcn

    def to_payload(self) -> dict:
        """組成可直接 POST 給 Node-RED 的 JSON-safe payload dict。"""
        # 佔比（%）：以 5 區累積時間總和為分母，在 Python 端算好直接送給
        # Node-RED，Node 端只負責顯示，不重複做這個算術（模組化原則）。
        total_time = (
            self.body.time_sec
            + self.fl.time_sec
            + self.fr.time_sec
            + self.hl.time_sec
            + self.hr.time_sec
        )

        def _pct(t: float) -> float:
            return round(t / total_time * 100, 2) if total_time > 1e-9 else 0.0

        _zone_map = {
            "BODY": self.body,
            "FL": self.fl,
            "FR": self.fr,
            "HL": self.hl,
            "HR": self.hr,
        }
        _best_stat = _zone_map.get(self.best_zone)

        return {
            "current_zone": self.current_zone,
            "best_zone": self.best_zone,
            "best_pct": _pct(_best_stat.time_sec) if _best_stat is not None else None,
            "body_time": round(self.body.time_sec, 2),
            "fl_time": round(self.fl.time_sec, 2),
            "fr_time": round(self.fr.time_sec, 2),
            "hl_time": round(self.hl.time_sec, 2),
            "hr_time": round(self.hr.time_sec, 2),
            "body_hits": self.body.hits,
            "fl_hits": self.fl.hits,
            "fr_hits": self.fr.hits,
            "hl_hits": self.hl.hits,
            "hr_hits": self.hr.hits,
            "body_pct": _pct(self.body.time_sec),
            "fl_pct": _pct(self.fl.time_sec),
            "fr_pct": _pct(self.fr.time_sec),
            "hl_pct": _pct(self.hl.time_sec),
            "hr_pct": _pct(self.hr.time_sec),
            "total_lick_time": round(total_time, 2),
            "face_state": self.face_state,
            "state_stability": round(self.state_stability, 3),
            "valid": self.valid,
            "frame": self.frame,
            "time_sec": round(self.time_sec, 2),
            # Extended — null when keypoints unavailable
            "dist_px": _jf(self.dist_px, 1),
            "dist_norm": _jf(self.dist_norm, 4),
            "gaze_fwd": _jf(self.gaze_fwd, 3),
            "gaze_lat": _jf(self.gaze_lat, 3),
            "gaze_angle": _jf(self.gaze_angle, 1),
            # Visualization-only geometry (Node-RED draws this; core never reads it)
            "trap_pts": self.trap_pts,
            "nose_xy": self.nose_xy,
            # ── 第一階段 v2 契約欄位（與上方 v1 欄位並存）──────────────────
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "source_timestamp": _jf(self.source_timestamp, 3),
            "frame_state": self.frame_state,
            "reason_code": self.reason_code,
            "observed_sec": round(self.observed_sec, 2),
            "valid_observed_sec": round(self.valid_observed_sec, 2),
            "no_cat_sec": round(self.no_cat_sec, 2),
            "stgcn_lick_sec": round(self.stgcn_lick_sec, 2),
            "assigned_zone_sec": round(self.assigned_zone_sec, 2),
            "unassigned_lick_sec": round(self.unassigned_lick_sec, 2),
            "zone_coverage_ratio": round(self.zone_coverage_ratio, 4),
            "unknown_rate": round(self.unknown_rate, 4),
        }
