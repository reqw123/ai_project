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
"""
import logging
import time

from .config import ExtZoneConfig as _C
from .models import ExtZoneResult, ZoneStat
from .regions import build_zone_targets, classify_zone, targets_to_geometry_payload
from .output import ZoneCsvWriter, ZoneMqttPublisher, ZoneHttpPublisher

_log = logging.getLogger(__name__)


class ExtBodyZonePlugin:
    def __init__(
        self,
        csv_path: str = _C.OUTPUT_CSV_PATH,
        mqtt_enabled: bool = _C.MQTT_ENABLED,
        nodered_enabled: bool = _C.NODERED_ENABLED,
    ):
        self._frame_count = 0
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        self._zone_stats = {zid: ZoneStat() for zid in _C.ZONE_NAMES if zid != _C.ZONE_NO_TARGET}
        self._prev_zone  = _C.ZONE_NO_TARGET
        self._last_log_t = -1e9
        self._last_geo_t = -1e9

        self._csv     = None
        self._mqtt    = None
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

    # ── Public API (spec naming: process(frame, skeleton, nose_pt)) ────────
    def process(self, frame=None, keypoints=None, nose_pt=None) -> None:
        """Fail-safe entry point. Always returns None by design — results
        are persisted internally, never handed back to the caller."""
        try:
            self._run(keypoints, None, nose_pt)
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin.process error: %s", exc)
        return None

    # ── Drop-in hook matching frame_processor's existing plugin protocol ──
    def update(self, kpts, kpt_conf) -> None:
        try:
            self._run(kpts, kpt_conf, None)
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin.update error: %s", exc)

    def close(self) -> None:
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
    def _run(self, kpts, kpt_conf, nose_pt_override) -> None:
        now    = time.monotonic()
        dt_sec = max(0.0, now - self._last_wall_t)
        self._last_wall_t  = now
        self._frame_count += 1
        self._elapsed_sec += dt_sec

        if kpts is None or kpt_conf is None:
            self._prev_zone = _C.ZONE_NO_TARGET
            return

        targets = build_zone_targets(kpts, kpt_conf)
        nose_pt = nose_pt_override if nose_pt_override is not None else kpts[_C.KP_NOSE]
        zone_id, zone_name, confidence = classify_zone(nose_pt, targets)

        if zone_id != _C.ZONE_NO_TARGET:
            stat = self._zone_stats[zone_id]
            stat.time_sec += dt_sec
            if self._prev_zone != zone_id:
                stat.hits += 1
        self._prev_zone = zone_id

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
            zone_breakdown={_C.ZONE_NAMES[zid]: st for zid, st in self._zone_stats.items()},
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
            payload["nose_xy"] = [round(float(nose_pt[0]), 1), round(float(nose_pt[1]), 1)] if nose_pt is not None else []
            payload["shapes"]  = targets_to_geometry_payload(targets)
            self._nodered.publish(payload)
        except Exception as exc:
            _log.debug("ExtBodyZonePlugin geometry publish failed: %s", exc)
