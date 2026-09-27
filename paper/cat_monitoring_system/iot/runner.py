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
from iot.ingest.no_data_watchdog import NoDataWatchdog
from iot.output.alert_engine import AlertEngine
from iot.output.publisher import MqttPublisher, PublishFn
from iot.repeat_limiter import RepeatLimiter
from iot.sensors.base import (
    BodyTempReading,
    EnvironmentReading,
    MotionEvent,
    WeightReading,
)
from iot.sensors.environment import EnvironmentSmoother
from iot.sensors.health import SensorHealthMonitor
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
        health: Optional[SensorHealthMonitor] = None,
    ):
        self._pub = publisher
        self._db_path = db_path
        self._router = router or SensorRouter()
        self._alerts = alert_engine or AlertEngine()
        self._smoother = smoother or EnvironmentSmoother()
        self._weight = weight_detector or WeightEventDetector()
        self._alert_log = RepeatLimiter(lambda: _C.WARN_REPEAT_SEC)   # 門檻告警的終端「持續中」重印
        self._reading_logged: dict[str, float] = {}  # 節點 → 上次印讀數的時間（除錯用）
        self._health = health or SensorHealthMonitor(
            _C.DATA_TIMEOUT_SEC, _C.SENSOR_RAIL_CONSECUTIVE, _C.SENSOR_RAILS, _C.SENSOR_EXPECTED_FIELDS,
            repeat_sec=lambda: _C.WARN_REPEAT_SEC)

    # ── 主入口 ───────────────────────────────────────────────────────────
    def handle_message(self, topic: str, payload: bytes | str) -> None:
        reading = self._router.route(topic, payload)
        if reading is None:
            nan_node = self._router.last_nan_node
            if nan_node:   # 有送、但必填欄位是 nan（例：MLX90614 沒接）：算在線，欄位消失照樣會報
                self._log_health(self._health.touch(*nan_node, time.monotonic()))
            return
        self._log_health(self._health.observe(reading, time.monotonic()))
        self._log_reading(reading)
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

    # ── 資料保留：定時清理舊的原始讀數 ─────────────────────────────────────
    def purge_old_data(self, now: Optional[float] = None) -> dict[str, int]:
        """刪掉超過 DATA_RETENTION_DAYS 天的原始讀數（只清這個行程處理的感測器那幾張表）；0＝不清理。"""
        days = _C.DATA_RETENTION_DAYS
        if days <= 0:
            return {}
        now = time.time() if now is None else now
        tables = [store.RAW_TABLES[k] for k in _C.active_kinds() if k in store.RAW_TABLES]
        try:
            deleted = store.purge_older_than(tables, now - days * 86400.0, db_path=self._db_path)
        except Exception as exc:  # noqa: BLE001 — 清不掉只警告，下次再試
            _log.warning("清理舊資料失敗：%s", exc)
            return {}
        done = {t: n for t, n in deleted.items() if n}
        if done:
            _log.info("資料保留 %g 天：刪掉舊的原始讀數 %s",
                      days, "，".join(f"{t} {n} 筆" for t, n in done.items()))
        return deleted

    # ── 除錯：讀數列印 ──────────────────────────────────────────────────
    def _log_reading(self, reading) -> None:
        """每個節點最新讀數每 READING_LOG_INTERVAL_SEC 秒印一行（0＝不印），看數值有沒有進來、對不對。"""
        every = _C.READING_LOG_INTERVAL_SEC
        if every <= 0:
            return
        node = f"{reading.kind}/{reading.source_id}"
        now = time.monotonic()
        if now - self._reading_logged.get(node, -1e9) < every:
            return
        self._reading_logged[node] = now
        fields = {k: v for k, v in reading.to_payload().items() if k not in ("kind", "source_id", "ts")}
        _log.info("讀數 %s：%s", node, "，".join(f"{k}={v}" for k, v in fields.items()))

    # ── 感測器健康（ESP32 在線、感測器沒接好）───────────────────────────────
    @staticmethod
    def _log_health(msgs) -> None:
        for level, msg in msgs:
            (_log.warning if level == "warning" else _log.info)(msg)

    def check_sensor_health(self, now: Optional[float] = None) -> None:
        """runner 主迴圈每秒呼叫：某欄位太久沒值就記警告。"""
        self._log_health(self._health.tick(time.monotonic() if now is None else now))

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
            _log.warning("告警：%s", alert.message)   # WARNING：設定視窗紀錄框顯示成紅字
            self._alert_log.clear(alert.key)
            self._alert_log.hit(alert.key, time.monotonic())
        self._log_alert_status()

    def _log_alert_status(self) -> None:
        """超標但在冷卻中（Discord 不重發）→ 日誌每 WARN_REPEAT_SEC 秒提醒一次；回到正常 → 記一行。
        09-27 使用者要求：數值一直超標時終端要一直印，不是 15 分鐘冷卻內只看到一次。"""
        suppressed, recovered = self._alerts.drain()
        now = time.monotonic()
        for key, message in suppressed:
            if self._alert_log.hit(key, now) is not None:
                _log.warning("告警（持續中，Discord 冷卻中不重發）：%s", message)
        for key, text in recovered:
            self._alert_log.clear(key)
            _log.info("恢復正常：%s", text)

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
        # 只有這個行程有處理秤重時才檢查「多久沒進食」（只開環境感測的行程不能誤報）
        if "weight" not in _C.active_kinds():
            return
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
    _log.info("IoT Sensor Hub 啟動中……（broker %s:%s，DB %s，感測器 %s，client id %s）",
              _C.MQTT_HOST, _C.MQTT_PORT, _C.DB_PATH, ",".join(_C.active_kinds()), _C.MQTT_CLIENT_ID)

    from iot.ingest.mqtt_ingestor import MqttIngestor

    stop_event = threading.Event()
    ingestor_box: dict[str, MqttIngestor] = {}

    def publish_fn(topic: str, payload: str) -> None:
        ing = ingestor_box.get("ingestor")
        if ing is not None:
            ing.publish(topic, payload)

    publisher = MqttPublisher(publish_fn)  # type: ignore[arg-type]
    hub = Hub(publisher)
    watchdog = NoDataWatchdog(
        _C.active_kinds(), _C.TOPIC_PREFIX, f"{_C.MQTT_HOST}:{_C.MQTT_PORT}",
        first_check_sec=_C.NO_DATA_FIRST_CHECK_SEC, timeout_sec=_C.DATA_TIMEOUT_SEC,
        repeat_sec=_C.WARN_REPEAT_SEC,
    )

    def on_message(topic: str, payload: bytes) -> None:
        recovered = watchdog.on_message(topic)
        if recovered:
            _log.info(recovered)
        hub.handle_message(topic, payload)

    try:
        ingestor = MqttIngestor(
            on_message,
            on_connected=lambda: watchdog.on_connected(time.monotonic()),
            on_disconnected=watchdog.on_disconnected,
        )
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

    def _purge_loop() -> None:   # 啟動時清一次，之後每 DATA_PURGE_EVERY_HOURS 小時一次
        while True:
            hub.purge_old_data()
            if stop_event.wait(_C.DATA_PURGE_EVERY_HOURS * 3600.0):
                return

    threading.Thread(target=_purge_loop, name="iot-purge", daemon=True).start()

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
            for msg in watchdog.tick(time.monotonic()):
                _log.warning(msg)
            hub.check_sensor_health()
    finally:
        ingestor.stop()
        store.close_connection()
        _log.info("IoT Sensor Hub 已停止。")
