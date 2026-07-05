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
from datetime import datetime

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

from .config import ExtZoneConfig as _C

_log = logging.getLogger(__name__)


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
        if not self._ready:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._writer.writerow([ts, zone, round(float(time_sec), 2), int(hits)])
            self._fh.flush()
        except Exception as exc:
            _log.debug("ZoneCsvWriter.write failed: %s", exc)

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class ZoneMqttPublisher:
    """Optional MQTT publisher. Silently disabled if paho-mqtt is unavailable."""

    def __init__(self, host: str = _C.MQTT_HOST, port: int = _C.MQTT_PORT, topic: str = _C.MQTT_TOPIC):
        self._topic  = topic
        self._client = None
        self._lock   = threading.Lock()
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client()
            self._client.connect_async(host, port)
            self._client.loop_start()
        except Exception as exc:
            _log.debug("ZoneMqttPublisher disabled: %s", exc)
            self._client = None

    def publish(self, payload: dict) -> None:
        if self._client is None:
            return
        try:
            with self._lock:
                self._client.publish(self._topic, json.dumps(payload), qos=0)
        except Exception as exc:
            _log.debug("ZoneMqttPublisher.publish failed: %s", exc)

    def close(self) -> None:
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
        self._url     = url
        self._timeout = timeout
        self._pool    = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ext_zone_nr"
        )

    def publish(self, payload: dict) -> None:
        if not _HAS_REQUESTS or not self._url:
            return
        self._pool.submit(self._post, payload)

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def _post(self, payload: dict) -> None:
        try:
            _requests.post(self._url, json=payload, timeout=self._timeout)
        except Exception:
            pass
