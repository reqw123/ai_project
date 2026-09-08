"""把 ingest → router → store → alert → publish 串起來的協調層。

``Hub`` 是純邏輯（不碰 MQTT / signal），``handle_message()`` 吃「topic + payload
bytes」跑完整條處理鏈，方便 ``test_runner_smoke.py`` 直接灌假資料驗證。

``run()`` 才建立真正的 ``MqttIngestor``、裝 SIGINT/SIGBREAK、啟動週期性檢查
執行緒，並 block 到收到中斷訊號。
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

from iot.config import IotHubConfig as _C
from iot.output.alert_engine import AlertEngine
from iot.output.publisher import MqttPublisher, PublishFn
from iot.sensors.base import (
    BodyTempReading,
    EnvironmentReading,
    MotionEvent,
    WeightReading,
)
from iot.sensors.environment import EnvironmentSmoother
from iot.sensors.router import SensorRouter
from iot.sensors.weight import WeightEventDetector
from iot.storage import store

_log = logging.getLogger(__name__)


class Hub:
    """一筆感測資料的完整處理鏈。所有外部副作用（DB、MQTT publish）都可注入。"""

    def __init__(
        self,
        publisher: MqttPublisher,
        db_path: Optional[str] = None,
        router: Optional[SensorRouter] = None,
        alert_engine: Optional[AlertEngine] = None,
        smoother: Optional[EnvironmentSmoother] = None,
        weight_detector: Optional[WeightEventDetector] = None,
    ):
        self._pub = publisher
        self._db_path = db_path
        self._router = router or SensorRouter()
        self._alerts = alert_engine or AlertEngine()
        self._smoother = smoother or EnvironmentSmoother()
        self._weight = weight_detector or WeightEventDetector()

    # ── 主入口 ───────────────────────────────────────────────────────────
    def handle_message(self, topic: str, payload: bytes | str) -> None:
        reading = self._router.route(topic, payload)
        if reading is None:
            return
        try:
            if isinstance(reading, EnvironmentReading):
                self._handle_env(reading)
            elif isinstance(reading, MotionEvent):
                self._handle_motion(reading)
            elif isinstance(reading, WeightReading):
                self._handle_weight(reading)
            elif isinstance(reading, BodyTempReading):
                self._handle_bodytemp(reading)
        except Exception:  # noqa: BLE001 — fail-safe：一筆壞資料不該中斷 hub
            _log.exception("處理讀數時發生未預期例外（topic=%s）", topic)

    # ── 各類型 ───────────────────────────────────────────────────────────
    def _persist(self, fn, *args) -> None:
        try:
            fn(*args, db_path=self._db_path)
        except Exception as exc:  # noqa: BLE001 — 寫入失敗只警告，繼續跑
            _log.warning("SQLite 寫入失敗（%s）：%s", getattr(fn, "__name__", fn), exc)

    def _emit_alerts(self, alerts) -> None:
        for alert in alerts:
            self._persist(
                store.insert_alert,
                alert.key,
                alert.severity,
                alert.message,
                alert.ts,
                alert.value,
                alert.threshold,
            )
            self._pub.publish_alert(alert)
            _log.info("告警：%s", alert.message)

    def _handle_env(self, reading: EnvironmentReading) -> None:
        self._persist(store.insert_env, reading)
        smoothed = self._smoother.smooth(reading)
        self._pub.publish_derived(
            "env",
            reading.source_id,
            {"source_id": reading.source_id, "ts": reading.ts, **smoothed},
        )
        self._emit_alerts(self._alerts.check_environment(reading))

    def _handle_bodytemp(self, reading: BodyTempReading) -> None:
        self._persist(store.insert_bodytemp, reading)
        self._pub.publish_derived(
            "bodytemp", reading.source_id, reading.to_payload()
        )
        self._emit_alerts(self._alerts.check_bodytemp(reading))

    def _handle_motion(self, event: MotionEvent) -> None:
        self._persist(store.insert_motion, event)
        self._pub.publish_derived("motion", event.source_id, event.to_payload())

    def _handle_weight(self, reading: WeightReading) -> None:
        self._persist(store.insert_weight, reading)
        self._pub.publish_derived("weight", reading.source_id, reading.to_payload())
        event = self._weight.update(reading)
        if event is not None:
            self._persist(store.insert_weight_event, event)
            # 進食事件重要，不走 derived 節流，直接發一筆。
            self._pub.publish_event("weight_change", event.to_payload())
            _log.info(
                "重量變化事件（%s）：%+.1f g（%s）",
                event.scale_id, event.delta_g, event.direction,
            )

    # ── 週期性（時間型）檢查 ─────────────────────────────────────────────
    def run_periodic_checks(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        last_feeding = {
            scale_id: self._safe_last_feeding_ts(scale_id)
            for scale_id in _C.FOOD_SCALE_IDS
        }
        self._emit_alerts(self._alerts.check_feeding_silence(last_feeding, now=now))

    def _safe_last_feeding_ts(self, scale_id: str):
        try:
            return store.last_weight_event_ts(
                scale_id, direction="decrease", db_path=self._db_path
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("查詢最後進食時間失敗（%s）：%s", scale_id, exc)
            return None


# ── 正式執行 ─────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, _C.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run() -> None:
    _configure_logging()
    _log.info("IoT Sensor Hub 啟動中……（broker %s:%s，DB %s）",
              _C.MQTT_HOST, _C.MQTT_PORT, _C.DB_PATH)

    from iot.ingest.mqtt_ingestor import MqttIngestor

    stop_event = threading.Event()
    ingestor_box: dict[str, MqttIngestor] = {}

    def publish_fn(topic: str, payload: str) -> None:
        ing = ingestor_box.get("ingestor")
        if ing is not None:
            ing.publish(topic, payload)

    publisher = MqttPublisher(publish_fn)  # type: ignore[arg-type]
    hub = Hub(publisher)

    try:
        ingestor = MqttIngestor(hub.handle_message)
    except ImportError:
        _log.error(
            "缺少 paho-mqtt，IoT Sensor Hub 無法啟動。請 `pip install paho-mqtt`"
            "（或用 iot/requirements.txt）。主系統不受影響。"
        )
        return
    ingestor_box["ingestor"] = ingestor
    ingestor.start()

    def _periodic_loop() -> None:
        while not stop_event.wait(_C.PERIODIC_CHECK_INTERVAL_SEC):
            try:
                hub.run_periodic_checks()
            except Exception:  # noqa: BLE001
                _log.exception("週期性檢查發生未預期例外")

    threading.Thread(target=_periodic_loop, name="iot-periodic", daemon=True).start()

    def _handle_signal(signum, frame):
        _log.info("收到中斷訊號（%s），關閉中……", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGBREAK"):  # Windows
        signal.signal(signal.SIGBREAK, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.wait(1.0):
            pass
    finally:
        ingestor.stop()
        store.close_connection()
        _log.info("IoT Sensor Hub 已停止。")
