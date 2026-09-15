"""ExtBodyZonePlugin — independent supplementary 7-zone body classifier.

Design contract:
  - Fully independent of the core pipeline and of plugins/lick_stage's
    existing analyzer/manager/overlay — the core program only feeds
    keypoints in, and this module never returns anything for it to read.
  - Never draws any overlay and never mutates the input frame.
  - Never raises: every public entry point is wrapped in try/except and
    fails silently, so a bug here can never crash the main system.
  - Persists results only via file / MQTT (both best-effort, optional).

Optional drop-in integration (not required for this module to exist):
    processor.register_plugin(ExtBodyZonePlugin())
mirrors the existing plugins/lick_stage registration in server/routes.py —
frame_processor.py already calls plugin.update(kpts, kpt_conf) and
plugin.close() on every registered plugin without reading a return value.

第一階段（說明書「來源時間規格」）：update() 新增可選 keyword 契約參數
（source_timestamp / dt_sec / cat_present / is_lick …），全部省略時退回舊
行為（wall-clock 計時），與 plugins/lick_stage/manager.py 同一套設計。
"""

import logging
import time

from plugins.lick_stage.analysis_context import FrameState, ReasonCode, SourceClock

from .config import ExtZoneConfig as _C
from .models import ExtZoneResult, ZoneStat
from .output import ZoneCsvWriter, ZoneHttpPublisher, ZoneMqttPublisher
from .regions import build_zone_targets, classify_zone, targets_to_geometry_payload

_log = logging.getLogger(__name__)


class ExtBodyZonePlugin:
    """獨立插件：7 區身體分區偵測（HEAD/NECK_CHEST/SIDE_BACK/ABDOMEN/
    FORELIMB/HINDLIMB/TAIL），結果可選擇性寫入 CSV、發布至 MQTT 或
    Node-RED。與 plugins/lick_stage 的既有插件並行、互不依賴。"""

    def __init__(
        self,
        csv_path: str = _C.OUTPUT_CSV_PATH,
        mqtt_enabled: bool = _C.MQTT_ENABLED,
        nodered_enabled: bool = _C.NODERED_ENABLED,
    ):
        self._frame_count = 0
        # 舊路徑（wall-clock）
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        # 新路徑（來源時間）
        self._clock = SourceClock()
        self._source_elapsed = 0.0
        self._session_id = ""
        self._session_active = False
        # 觀測分母（說明書「統計分母」）
        self._observed_sec = 0.0
        self._no_cat_sec = 0.0
        self._valid_observed_sec = 0.0
        self._stgcn_lick_sec = 0.0
        self._assigned_zone_sec = 0.0
        self._unassigned_lick_sec = 0.0

        self._zone_stats = {
            zid: ZoneStat() for zid in _C.ZONE_NAMES if zid != _C.ZONE_NO_TARGET
        }
        self._prev_zone = _C.ZONE_NO_TARGET
        self._last_log_t = -1e9
        self._last_geo_t = -1e9

        self._csv = None
        self._mqtt = None
        self._nodered = None
        try:
            if _C.OUTPUT_ENABLED:
                self._csv = ZoneCsvWriter(csv_path)
            if mqtt_enabled:
                self._mqtt = ZoneMqttPublisher()
            if nodered_enabled:
                self._nodered = ZoneHttpPublisher()
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin output init failed: %s", exc)

    # ── Session 生命週期 ────────────────────────────────────────────────
    def start_session(self, session_id: str, **_meta) -> None:
        self._session_id = session_id
        self._session_active = True
        self._clock.reset()
        self._source_elapsed = 0.0
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        self._frame_count = 0
        self._prev_zone = _C.ZONE_NO_TARGET

    def finish_session(self, end_source_timestamp=None) -> None:
        self._session_active = False

    def reset_session(self, reason: str = "") -> None:
        self._clock.reset()
        self._source_elapsed = 0.0
        self._prev_zone = _C.ZONE_NO_TARGET

    # ── Drop-in hook matching frame_processor's existing plugin protocol ──
    def update(
        self,
        kpts,
        kpt_conf,
        *,
        source_timestamp=None,
        dt_sec=None,
        frame_idx=None,
        cat_present=None,
        is_lick=None,
        lick_confidence=None,
        track_id=None,
        session_id=None,
        pose_quality=None,
    ) -> None:
        """Fail-safe 進入點，符合 FrameProcessor 既有的 plugin 呼叫慣例。

        pose_quality：M4 共用 PoseFilter（frame_processor.py）算出的骨長
        品質分數。本外掛目前不使用，僅接受此參數避免呼叫端新增此 keyword
        後在 TypeError 分支整組退化成舊版兩參數呼叫（連 source_timestamp/
        session_id 等既有契約參數都會一併遺失）。kpts 也預期已經是同一個
        共用 PoseFilter 平滑過的結果（本外掛過去完全不做平滑，直接吃
        raw kpts；M4 接線後改吃平滑後的座標）。
        """
        try:
            self._run(
                kpts,
                kpt_conf,
                None,
                source_timestamp=source_timestamp,
                dt_sec=dt_sec,
                frame_idx=frame_idx,
                cat_present=cat_present,
                is_lick=is_lick,
                session_id=session_id,
            )
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin.update error: %s", exc)

    def close(self) -> None:
        """關閉所有已啟用的輸出端（CSV/MQTT/Node-RED）。"""
        try:
            if self._csv is not None:
                self._csv.close()
            if self._mqtt is not None:
                self._mqtt.close()
            if self._nodered is not None:
                self._nodered.close()
        except Exception:
            pass

    # ── Internal ────────────────────────────────────────────────────────
    def _run(
        self,
        kpts,
        kpt_conf,
        nose_pt_override,
        *,
        source_timestamp=None,
        dt_sec=None,
        frame_idx=None,
        cat_present=None,
        is_lick=None,
        session_id=None,
    ) -> None:
        use_source_time = source_timestamp is not None or dt_sec is not None
        if use_source_time:
            if not self._session_active:
                self.start_session(session_id or self._session_id or "S_auto")
            if dt_sec is not None:
                dt, discontinuity = self._clock.clamp_external_dt(
                    dt_sec, source_timestamp
                )
            else:
                dt, discontinuity, _ = self._clock.tick(source_timestamp)
            self._source_elapsed += dt
            self._elapsed_sec = self._source_elapsed
        else:
            now = time.monotonic()
            dt = max(0.0, now - self._last_wall_t)
            self._last_wall_t = now
            self._elapsed_sec += dt
            discontinuity = False

        self._frame_count += 1

        cp = cat_present
        il = is_lick
        if cp is None:
            cp = kpts is not None and kpt_conf is not None
            il = cp

        # 時間不連續：不累加任何分母
        accum_dt = 0.0 if discontinuity else max(0.0, dt)
        self._observed_sec += accum_dt

        if not cp:
            self._no_cat_sec += accum_dt
            self._prev_zone = _C.ZONE_NO_TARGET
            # 說明書 M2 才把「事件表 / 視窗表」接上；M1 shadow 模式下 CSV/MQTT
            # 的輸出節奏維持舊行為（只在舔毛幀寫），非舔毛幀只更新內部分母。
            return

        self._valid_observed_sec += accum_dt

        if not il:
            self._prev_zone = _C.ZONE_NO_TARGET
            return

        if kpts is None or kpt_conf is None:
            # 舔毛但無姿態
            self._prev_zone = _C.ZONE_NO_TARGET
            self._stgcn_lick_sec += accum_dt
            self._unassigned_lick_sec += accum_dt
            self._emit(
                FrameState.LICK_UNASSIGNED,
                ReasonCode.POSE_INVALID,
                _C.ZONE_NO_TARGET,
                0.0,
                None,
                None,
            )
            return

        targets = build_zone_targets(kpts, kpt_conf)
        nose_pt = nose_pt_override if nose_pt_override is not None else kpts[_C.KP_NOSE]
        zone_id, zone_name, confidence = classify_zone(nose_pt, targets)

        self._stgcn_lick_sec += accum_dt
        if zone_id != _C.ZONE_NO_TARGET:
            self._assigned_zone_sec += accum_dt
            stat = self._zone_stats[zone_id]
            stat.time_sec += accum_dt
            if self._prev_zone != zone_id:
                stat.hits += 1
            frame_state = FrameState.LICK_ASSIGNED
            reason = None
        else:
            self._unassigned_lick_sec += accum_dt
            frame_state = FrameState.LICK_UNASSIGNED
            reason = (
                ReasonCode.POSE_INVALID
                if targets is None
                else ReasonCode.NO_REGION_HIT
            )
        self._prev_zone = zone_id

        self._emit(frame_state, reason, zone_id, confidence, targets, nose_pt)

    def _emit(self, frame_state, reason_code, zone_id, confidence, targets, nose_pt):
        zone_name = _C.ZONE_NAMES.get(zone_id, "NO_TARGET")
        stat = self._zone_stats.get(zone_id)
        result = ExtZoneResult(
            current_zone=zone_id,
            zone_name=zone_name,
            confidence=confidence,
            valid=targets is not None,
            frame=self._frame_count,
            time_sec=self._elapsed_sec,
            hits=stat.hits if stat is not None else 0,
            zone_time_sec=stat.time_sec if stat is not None else 0.0,
            zone_breakdown={
                _C.ZONE_NAMES[zid]: st for zid, st in self._zone_stats.items()
            },
            schema_version="2.0",
            session_id=self._session_id,
            frame_state=frame_state,
            reason_code=reason_code,
            observed_sec=self._observed_sec,
            valid_observed_sec=self._valid_observed_sec,
            no_cat_sec=self._no_cat_sec,
            stgcn_lick_sec=self._stgcn_lick_sec,
            assigned_zone_sec=self._assigned_zone_sec,
            unassigned_lick_sec=self._unassigned_lick_sec,
        )
        self._persist(result)
        self._publish_geometry(result, targets, nose_pt)

    def _persist(self, result: ExtZoneResult) -> None:
        if self._elapsed_sec - self._last_log_t < _C.LOG_INTERVAL_SEC:
            return
        self._last_log_t = self._elapsed_sec
        try:
            if self._csv is not None:
                self._csv.write(result.current_zone, result.zone_time_sec, result.hits)
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin csv write failed: %s", exc)
        try:
            if self._mqtt is not None:
                self._mqtt.publish(result.to_payload())
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin mqtt publish failed: %s", exc)

    def _publish_geometry(self, result: ExtZoneResult, targets, nose_pt) -> None:
        """Send raw pixel geometry to Node-RED for client-side drawing only.
        Re-packages shapes already computed in _run() — no new geometry math."""
        if self._nodered is None:
            return
        if self._elapsed_sec - self._last_geo_t < _C.GEO_PUBLISH_INTERVAL_SEC:
            return
        self._last_geo_t = self._elapsed_sec
        try:
            payload = result.to_payload()
            payload["nose_xy"] = (
                [round(float(nose_pt[0]), 1), round(float(nose_pt[1]), 1)]
                if nose_pt is not None
                else []
            )
            payload["shapes"] = targets_to_geometry_payload(targets)
            self._nodered.publish(payload)
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin geometry publish failed: %s", exc)
