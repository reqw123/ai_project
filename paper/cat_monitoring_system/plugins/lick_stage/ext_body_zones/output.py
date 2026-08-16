"""File / MQTT output sinks for the extended body-zone module.

Pure side-effect sinks — never raise, never touch the caller's frame, and
never return a value the main program is expected to read.
"""

import concurrent.futures
import csv
import json
import logging
import os
import threading
import time
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
    """Non-blocking HTTP publisher to Node-RED, mirroring plugins/lick_stage's
    NodeRedPublisher. publish() returns immediately; the POST happens in a
    background thread so it can never stall the frame loop."""

    def __init__(self, url: str = _C.NODERED_URL, timeout: float = _C.NODERED_TIMEOUT):
        self._url = url
        self._timeout = timeout
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ext_zone_nr"
        )
        self._last_warn_time = 0.0  # 上次記警告的時間，用來節流避免洗版
        self._warn_lock = threading.Lock()  # 保護上面的 check-then-set，避免 pool 內多執行緒同時通過節流判斷

    def publish(self, payload: dict) -> None:
        """提交一筆 JSON payload 到背景執行緒發送，立即返回。"""
        if not _HAS_REQUESTS or not self._url:
            return
        self._pool.submit(self._post, payload)

    def close(self) -> None:
        """關閉背景執行緒池（不等待進行中的請求完成）。"""
        self._pool.shutdown(wait=False)

    def _post(self, payload: dict) -> None:
        try:
            _requests.post(self._url, json=payload, timeout=self._timeout)
        except Exception as e:
            # 過去這裡是完全靜默的 except: pass，跟同檔案其他 sink（至少都
            # 有 _log.debug）不一致，也是本模組唯一完全無聲的失敗路徑。這條
            # 路徑呼叫頻率高，逐筆都記警告會洗版，節流成同一個 publisher
            # 實例最多每 _HTTP_WARN_INTERVAL_SEC 秒警告一次。
            #
            # check-then-set 用 _warn_lock 保護（跟 plugins/lick_stage/
            # publisher.py 同樣的修正）：_post() 是在
            # ThreadPoolExecutor(max_workers=2) 裡跑的，兩個 worker 幾乎同時
            # 送出的請求都失敗時，若沒有鎖，兩者都可能在對方更新
            # _last_warn_time 之前判斷「超過節流間隔」，導致同一次故障重複
            # 印出兩則警告，節流形同虛設。
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
