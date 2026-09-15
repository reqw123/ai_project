"""第一階段共用契約：來源時間、Session 身份、互斥狀態碼與原因碼。

對應《家貓舔毛部位辨識與健康分析模組三階段工程說明書》第一階段
「資料可信度與介面重構」。本模組刻意只放**純資料結構與純函式**：

  * 不做任何 I/O、不匯入 numpy / cv2 / requests；
  * 不匯入本 plugin 的其他子模組（analyzer / statistics / models …），
    以免形成迴圈相依；
  * 兩個外掛（LickStagePlugin 與 ext_body_zones/ExtBodyZonePlugin）都可
    直接 `from plugins.lick_stage.analysis_context import ...` 共用同一份
    狀態定義，不必各自維護一份。

核心工程動機（見說明書「來源時間規格」）：現行 `manager.py` /
`ext_body_zones/plugin.py` 以 `time.monotonic()` 計算相鄰 update 的間隔，
代表「網路等待、GPU 卡頓、離線推論速度、暫停」都會被誤算成貓咪舔毛時間。
外掛只能接受**來源媒體時間**（影片 PTS，或 `frame_idx / source_fps`），
不得再自行產生主要統計時間。`SourceClock` 就是這個規則的落地點。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional

# payload / 資料表的 schema 版本；任何欄位語意變更都必須提高這個號碼
# （見說明書「向後相容策略」）。
SCHEMA_VERSION = "2.0"

# 相鄰兩個來源時間戳的差若超過這個秒數，視為時間不連續（例如影片被 seek、
# 排程暫停後恢復、串流大幅掉幀），該 dt **不累加**進任何統計。
DEFAULT_MAX_VALID_GAP_SEC = 1.0


class FrameState:
    """互斥的單幀狀態（說明書「狀態與原因碼重構」）。

    `kpts` 為空只代表分析器沒有收到可用姿態，**不能**代表沒有貓。舊介面把
    「沒有貓 / 非舔毛 / 舔毛信心不足 / 舔毛但無法定位」全部混成同一種輸入，
    是後續統計偏差的主因。這五種狀態互斥，任一幀必為其一。
    """

    NO_CAT = "NO_CAT"  # cat_present=False
    NOT_LICK = "NOT_LICK"  # 有貓，但 ST-GCN 當幀非 lick
    LOW_LICK_CONF = "LOW_LICK_CONF"  # lick 候選但未過信心門檻
    LICK_ASSIGNED = "LICK_ASSIGNED"  # 過 gate 且 canonical zone 成立
    LICK_UNASSIGNED = "LICK_UNASSIGNED"  # 過 gate 但無可靠部位

    ALL = (NO_CAT, NOT_LICK, LOW_LICK_CONF, LICK_ASSIGNED, LICK_UNASSIGNED)

    # 會累加進「已觀測時間」（valid_observed_sec）的狀態 —— 亦即「畫面裡有貓」
    CAT_PRESENT = (NOT_LICK, LOW_LICK_CONF, LICK_ASSIGNED, LICK_UNASSIGNED)
    # 會累加進 stgcn_lick_sec 的狀態 —— 亦即「ST-GCN 認定正在舔毛」
    LICK = (LICK_ASSIGNED, LICK_UNASSIGNED)


class ReasonCode:
    """`LICK_UNASSIGNED` 或無效幀的原因碼（說明書「必要原因碼」）。

    最後一欄「是否計入未指派舔毛」：True 代表這段 dt 會進 unassigned_lick_sec，
    False 代表這段 dt 連 stgcn_lick_sec 都不進（屬候選 / 無效資料）。
    """

    POSE_INVALID = "POSE_INVALID"  # 必要關鍵點不足或品質過低 —— 計入未指派
    FRONT_VIEW_UNOBSERVABLE = "FRONT_VIEW_UNOBSERVABLE"  # 正面姿態 2D 幾何無法分辨 —— 計入未指派
    NO_REGION_HIT = "NO_REGION_HIT"  # 姿態有效但鼻端未命中任何區域 —— 計入未指派
    AMBIGUOUS = "AMBIGUOUS"  # 前兩名候選差距低於 margin —— 計入未指派
    TRACK_SWITCH = "TRACK_SWITCH"  # 追蹤 ID 切換或身份不確定 —— 計入未指派
    TIMESTAMP_INVALID = "TIMESTAMP_INVALID"  # 來源時間倒退 / 跳躍 / 缺失 —— 該 dt 不累加
    MODEL_GATE_REJECTED = "MODEL_GATE_REJECTED"  # ST-GCN 未形成有效 lick bout —— 屬候選資料

    # 這些原因碼對應的 dt 仍計入 stgcn_lick_sec（只是無法指派部位）
    COUNTS_AS_UNASSIGNED_LICK = (
        POSE_INVALID,
        FRONT_VIEW_UNOBSERVABLE,
        NO_REGION_HIT,
        AMBIGUOUS,
        TRACK_SWITCH,
    )


@dataclass(frozen=True)
class AnalysisContext:
    """一段 Session 的不可變 metadata。由呼叫端（FrameProcessor）在
    `start_session()` 時建立，之後每一筆事件 / 視窗資料都帶著它，讓資料
    可追溯到影片、貓、模型與設定版本（說明書「目標架構與資料責任」）。
    """

    session_id: str
    video_id: str = ""
    cat_id: Optional[str] = None
    period: str = ""  # AM / PM / ... 或明確命名的時段
    model_version: str = ""
    config_hash: str = ""
    source_fps: float = 0.0


class SourceClock:
    """把來源時間戳轉成「單調遞增、跨處理速度不變」的 dt。

    使用方式：每幀呼叫一次 `tick(source_timestamp)`，拿回 `(dt_sec,
    discontinuity, timestamp_invalid)`：

      * `dt_sec`            —— 這一幀應累加的秒數（來源時間差，已夾鉗）
      * `discontinuity`     —— 這一幀與前一幀時間不連續（dt 不累加）
      * `timestamp_invalid` —— 時間戳本身無效 / 倒退（對應 TIMESTAMP_INVALID）

    驗收目標（說明書）：同一支影片以 0.5×、1×、2× 處理，各累計秒數的差異
    應小於一個來源影格間隔 —— 因為 dt 只由 source_timestamp 決定，與這一幀
    實際花多少 wall-clock 時間處理完全無關。
    """

    def __init__(self, max_valid_gap_sec: float = DEFAULT_MAX_VALID_GAP_SEC):
        self.max_valid_gap_sec = float(max_valid_gap_sec)
        self._prev_ts: Optional[float] = None

    def reset(self) -> None:
        """清空時間基準（新 Session / 明確來源切換時呼叫）。"""
        self._prev_ts = None

    def tick(self, source_timestamp) -> tuple[float, bool, bool]:
        if source_timestamp is None:
            return 0.0, False, True
        try:
            ts = float(source_timestamp)
        except (TypeError, ValueError):
            return 0.0, False, True
        if math.isnan(ts) or math.isinf(ts) or ts < 0.0:
            return 0.0, False, True

        if self._prev_ts is None:
            self._prev_ts = ts
            return 0.0, False, False

        raw = ts - self._prev_ts
        if raw < 0.0:
            # 時間倒退：重設基準，這一幀不累加（說明書：TIMESTAMP_INVALID → dt 不累加）
            self._prev_ts = ts
            return 0.0, True, True

        self._prev_ts = ts
        if raw > self.max_valid_gap_sec:
            # 跳躍過大：標記不連續，dt 不累加
            return 0.0, True, False
        return raw, False, False

    def clamp_external_dt(self, dt_sec, source_timestamp=None) -> tuple[float, bool]:
        """呼叫端已自行提供 dt_sec（例如 FrameProcessor 直接給 source_dt）時，
        仍套用同一套 gap 夾鉗規則，並同步更新內部時間基準。

        回傳 `(dt_sec, discontinuity)`。
        """
        if source_timestamp is not None:
            try:
                self._prev_ts = float(source_timestamp)
            except (TypeError, ValueError):
                pass
        try:
            dt = float(dt_sec)
        except (TypeError, ValueError):
            return 0.0, True
        if math.isnan(dt) or math.isinf(dt) or dt < 0.0:
            return 0.0, True
        if dt > self.max_valid_gap_sec:
            return 0.0, True
        return dt, False


def derive_frame_state(
    cat_present: bool,
    is_lick: bool,
    zone_assigned: bool,
    *,
    lick_confidence: Optional[float] = None,
    low_conf_threshold: Optional[float] = None,
) -> str:
    """由布林條件推導互斥的 `FrameState`（說明書「狀態與原因碼重構」表格）。

    * `zone_assigned` —— 幾何模組已產生可靠的 canonical zone。
    * `lick_confidence` / `low_conf_threshold` —— 兩者皆給定且信心低於門檻時，
      即使呼叫端說 `is_lick=True` 也降級為 `LOW_LICK_CONF`（候選，不進主統計）。
      目前 FrameProcessor 端已先用 0.80 門檻過濾，所以這條分支平常不會觸發，
      但保留給第二階段把 gate 邏輯搬進來後使用。
    """
    if not cat_present:
        return FrameState.NO_CAT
    if not is_lick:
        return FrameState.NOT_LICK
    if (
        lick_confidence is not None
        and low_conf_threshold is not None
        and float(lick_confidence) < float(low_conf_threshold)
    ):
        return FrameState.LOW_LICK_CONF
    return FrameState.LICK_ASSIGNED if zone_assigned else FrameState.LICK_UNASSIGNED


class ZoneL1:
    """統一部位本體的上層類別（說明書「統一部位本體」）。

    `HEAD` 只會由 `canonical_ext_zone()`（`ext_body_zones` 的 9 個 zone
    映射，見下方）產生——`lick_stage` 自己的 `canonical_zone()` 永遠不會
    回傳 `HEAD`（鼻部接觸梯形從鼻子往外延伸，天生無法命中貓咪自己的頭），
    是 M6 融合 ext_body_zones 才新增的類別，不影響既有 `canonical_zone()`
    呼叫端（`bout_aggregator.py` 的 zone hysteresis 只吃 `canonical_zone()`
    的輸出驅動切分邏輯，不吃 `canonical_ext_zone()`，見該檔案模組說明）。
    """

    TORSO = "TORSO"
    FORELIMB = "FORELIMB"
    HINDLIMB = "HINDLIMB"
    TAIL = "TAIL"
    HEAD = "HEAD"
    UNKNOWN = "UNKNOWN"


# LickStage 原始標籤（BODY_CENTER / FL / FR / HL / HR）→ (上層, 下層) 對照。
# ext_body_zones 另有自己的 7 區標籤，映射在該子模組處理。
_LICKSTAGE_ZONE_MAP = {
    "BODY_CENTER": (ZoneL1.TORSO, None),
    "BODY": (ZoneL1.TORSO, None),
    "FL": (ZoneL1.FORELIMB, "LEFT"),
    "FR": (ZoneL1.FORELIMB, "RIGHT"),
    "HL": (ZoneL1.HINDLIMB, "LEFT"),
    "HR": (ZoneL1.HINDLIMB, "RIGHT"),
}


def canonical_zone(raw_label) -> tuple[str, Optional[str]]:
    """把 LickStage 原始 zone 標籤轉成 (zone_l1, zone_l2)。

    未知 / NO_TARGET → (UNKNOWN, None)。下層無法細分時 zone_l2 為 None
    （例如 TORSO 的左右 / 腹背由第二階段 ext 幾何補上）。
    """
    return _LICKSTAGE_ZONE_MAP.get(str(raw_label), (ZoneL1.UNKNOWN, None))


# ext_body_zones 的 9 個 zone 名稱（`ext_body_zones/config.py` 的
# `ExtZoneConfig.ZONE_NAMES`）→ `ZoneL1`。刻意不在這裡 import
# `ExtZoneConfig`（`ext_body_zones` 是零依賴的獨立姊妹插件，見該模組
# docstring「Independent of plugins/lick_stage/config.py」；反過來
# `analysis_context.py` 也不該對它建立 import 依賴），改用字串常值——
# `ext_body_zones/config.py` 若改了這些名稱，這裡要跟著手動同步（跟
# `canonical_zone()` 的 `_LICKSTAGE_ZONE_MAP` 用同一種「用字串對照、不
# import 對方模組」的既有慣例一致）。
# NECK_CHEST/SIDE_BACK/ABDOMEN/TORSO_UNSPECIFIED 全部併入 TORSO：
# 這幾個都是軀幹次分區，`ZoneL1` 這一層只要粗粒度的軀幹類別即可，細節
# 保留在 `ext_zone_mode`（見 bout_aggregator.py 的 `to_event()`）裡的
# 原始 ext 標籤，不會遺失。
# 附註：`ext_body_zones/regions.py::classify_zone()` 目前刻意停用 HEAD／
# NECK_CHEST 判定本身（鼻子天生緊貼自己頭部/胸口，命中不代表真的在舔那裡，
# 見該函式內的說明），所以這兩個 key 實務上永遠不會被 `classify_zone()`
# 觸發——這裡先補齊映射純粹是面向未來（那天判定重新啟用時這裡不用跟著
# 改），不代表目前系統會產生 `ZoneL1.HEAD`。
_EXT_ZONE_MAP = {
    "HEAD": ZoneL1.HEAD,
    "NECK_CHEST": ZoneL1.TORSO,
    "SIDE_BACK": ZoneL1.TORSO,
    "ABDOMEN": ZoneL1.TORSO,
    "TORSO_UNSPECIFIED": ZoneL1.TORSO,
    "FORELIMB": ZoneL1.FORELIMB,
    "HINDLIMB": ZoneL1.HINDLIMB,
    "TAIL": ZoneL1.TAIL,
}


def canonical_ext_zone(raw_ext_label) -> str:
    """把 `ext_body_zones` 的原始 zone 名稱（`ZONE_NAMES` 的 value，例如
    `"ABDOMEN"`）映射成 `ZoneL1`。`None`／`"NO_TARGET"`／`"AMBIGUOUS"`／
    未知字串一律回傳 `ZoneL1.UNKNOWN`（跟 `canonical_zone()` 對
    `NO_TARGET` 的既有處理一致；`AMBIGUOUS` 語意上就是「無法可靠分辨」，
    歸類到任何一個具體 zone 都不誠實，所以也算 `UNKNOWN`，不是新發明的
    特例）。這個函式只用來產生**補充**欄位（`bout_aggregator.py` 的
    `ext_zone_l1_mode`），不會影響 `bout_aggregator.py` 既有的 zone
    hysteresis 切分邏輯——那個邏輯只吃 `canonical_zone()` 的輸出，見
    `bout_aggregator.py` 模組開頭說明與 M6 融合的設計決定。
    """
    return _EXT_ZONE_MAP.get(str(raw_ext_label), ZoneL1.UNKNOWN)


def stable_config_hash(payload) -> str:
    """對任意可序列化物件產生穩定短雜湊，用於 `config_hash` 欄位。

    刻意不追求密碼學強度：只是要讓「這批資料是用哪一組設定跑的」可比對、
    可分層（說明書「向後相容策略」：跨版本資料不得未經分層直接合併）。
    """
    try:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        blob = repr(payload)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
