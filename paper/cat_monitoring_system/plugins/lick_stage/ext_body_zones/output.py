"""File / MQTT output sinks for the extended body-zone module.

Pure side-effect sinks — never raise, never touch the caller's frame, and
never return a value the main program is expected to read.
"""

import csv
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

from .config import ExtZoneConfig as _C

_log = logging.getLogger(__name__)

_HTTP_WARN_INTERVAL_SEC = 30.0  # ZoneHttpPublisher 同一個實例失敗時最多多久警告一次


class ZoneCsvWriter:
    """Appends one row per persisted snapshot to a CSV file. Fail-safe."""

    _FIELDS = ["timestamp", "zone", "time_sec", "hits"]

    def __init__(self, path: str = _C.OUTPUT_CSV_PATH):
        self._ready = False
        self._fh = None
        self._writer = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            is_new = not os.path.exists(path)
            self._fh = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fh)
            if is_new:
                self._writer.writerow(self._FIELDS)
                self._fh.flush()
            self._ready = True
        except Exception as exc:
            _log.debug("ZoneCsvWriter init failed: %s", exc)

    def write(self, zone: int, time_sec: float, hits: int) -> None:
        """附加一列快照紀錄並立即 flush；初始化失敗時安靜地不做任何事。"""
        if not self._ready:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._writer.writerow([ts, zone, round(float(time_sec), 2), int(hits)])
            self._fh.flush()
        except Exception as exc:
            _log.debug("ZoneCsvWriter.write failed: %s", exc)

    def close(self) -> None:
        """關閉底層檔案（若已開啟）。"""
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class ZoneMqttPublisher:
    """Optional MQTT publisher. Silently disabled if paho-mqtt is unavailable."""

    def __init__(
        self,
        host: str = _C.MQTT_HOST,
        port: int = _C.MQTT_PORT,
        topic: str = _C.MQTT_TOPIC,
    ):
        self._topic = topic
        self._client = None
        self._lock = threading.Lock()
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client()
            self._client.connect_async(host, port)
            self._client.loop_start()
        except Exception as exc:
            _log.debug("ZoneMqttPublisher disabled: %s", exc)
            self._client = None

    def publish(self, payload: dict) -> None:
        """發布一筆 JSON payload 到 MQTT topic；未連線時安靜地不做任何事。"""
        if self._client is None:
            return
        try:
            with self._lock:
                self._client.publish(self._topic, json.dumps(payload), qos=0)
        except Exception as exc:
            _log.debug("ZoneMqttPublisher.publish failed: %s", exc)

    def close(self) -> None:
        """停止 MQTT 迴圈並中斷連線。"""
        try:
            if self._client is not None:
                self._client.loop_stop()
                self._client.disconnect()
        except Exception:
            pass


class ZoneHttpPublisher:
    """有界 latest-only 背景 HTTP 發布器，對齊 plugins/lick_stage 的
    NodeRedPublisher（說明書「傳輸與 Node-RED 可靠性」）：publish() 立即返回，
    只保留最新 payload（舊的未送出就覆蓋並計入 dropped），單一 worker 用
    requests.Session POST 並 raise_for_status，指標由 stats() 取出。"""

    def __init__(self, url: str = _C.NODERED_URL, timeout: float = _C.NODERED_TIMEOUT):
        self._url = url
        self._timeout = timeout
        try:
            self._session = _requests.Session() if _HAS_REQUESTS else None
        except Exception:
            self._session = None

        self._pending = None
        self._cond = threading.Condition()
        self._running = True

        self._lock = threading.Lock()
        self._success = 0
        self._failure = 0
        self._dropped = 0
        self._last_success_ts = 0.0
        self._latencies = deque(maxlen=64)

        self._last_warn_time = 0.0  # 上次記警告的時間，用來節流避免洗版
        self._warn_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run, name="ext_zone_nr", daemon=True
        )
        self._worker.start()

    def publish(self, payload: dict) -> None:
        """把最新 payload 放進 slot；有未送出的舊值就覆蓋並計入 dropped。"""
        if not _HAS_REQUESTS or not self._url:
            return
        with self._cond:
            if self._pending is not None:
                with self._lock:
                    self._dropped += 1
            self._pending = payload
            self._cond.notify()

    def stats(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
            return {
                "publish_ok": self._success,
                "publish_fail": self._failure,
                "dropped_payload": self._dropped,
                "last_success_ts": round(self._last_success_ts, 3),
                "p95_latency_ms": round(p95 * 1000.0, 1),
            }

    def close(self) -> None:
        """停止背景 worker（不等待進行中的請求完成）。"""
        with self._cond:
            self._running = False
            self._cond.notify()
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait()
                if not self._running:
                    return
                payload = self._pending
                self._pending = None
            self._post(payload)

    def _post(self, payload: dict) -> None:
        t0 = time.time()
        try:
            client = self._session if self._session is not None else _requests
            resp = client.post(self._url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            with self._lock:
                self._success += 1
                self._last_success_ts = time.time()
                self._latencies.append(time.time() - t0)
        except Exception as e:
            with self._lock:
                self._failure += 1
            # 過去這裡是完全靜默的 except: pass，跟同檔案其他 sink（至少都
            # 有 _log.debug）不一致，也是本模組唯一完全無聲的失敗路徑。這條
            # 路徑呼叫頻率高，逐筆都記警告會洗版，節流成同一個 publisher
            # 實例最多每 _HTTP_WARN_INTERVAL_SEC 秒警告一次。check-then-set
            # 用 _warn_lock 保護（背景 worker 與呼叫 _post 的測試路徑可能並行）。
            should_warn = False
            with self._warn_lock:
                now = time.time()
                if now - self._last_warn_time >= _HTTP_WARN_INTERVAL_SEC:
                    self._last_warn_time = now
                    should_warn = True
            if should_warn:
                _log.warning(
                    "推送到 Node-RED 失敗（%s）：%s"
                    "（此類錯誤 %d 秒內只警告一次，避免洗版）",
                    self._url,
                    e,
                    int(_HTTP_WARN_INTERVAL_SEC),
                )
