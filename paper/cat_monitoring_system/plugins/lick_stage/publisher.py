"""Non-blocking, bounded latest-only HTTP publisher to Node-RED.

說明書第一階段「傳輸與 Node-RED 可靠性」：即時儀表板只需要最新狀態，
應採**有界 latest-only 佇列**（長度 1，新的覆蓋舊的），而不是無上限地把
HTTP 工作丟進 ThreadPoolExecutor —— Node-RED 離線或延遲時，無界佇列會
持續成長、吃光記憶體。

行為：
  * publish() 立即返回，只把最新 payload 放進單一 slot；有舊的未送出就直接
    覆蓋並計入 dropped。
  * 單一背景 worker 迴圈取出最新 payload、用 requests.Session（連線重用）
    POST、raise_for_status。
  * 指標：success / failure / dropped / last_success_ts / p95 latency，
    由 stats() 取出，供 Node-RED 監控傳輸健康度。
  * 失敗警告仍節流（同一實例每 _WARN_INTERVAL_SEC 秒最多一次），避免洗版。
"""

import logging
import threading
import time
from collections import deque

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

from plugins.lick_stage.config import LickConfig as _C

_log = logging.getLogger(__name__)

_WARN_INTERVAL_SEC = 30.0  # 同一個 publisher 實例失敗時最多多久警告一次
_LATENCY_WINDOW = 64  # 保留最近幾筆 latency 供 p95 估計


class NodeRedPublisher:
    """有界 latest-only 背景發布器。publish() 立即返回，永不拋例外。"""

    def __init__(self, url: str, timeout: float = _C.NODERED_TIMEOUT):
        self._url = url
        self._timeout = timeout
        try:
            self._session = _requests.Session() if _HAS_REQUESTS else None
        except Exception:
            # 測試會 monkeypatch _requests 成只有 post() 的假物件
            self._session = None

        # 單一 slot：只保留最新 payload；worker 由 condition 喚醒
        self._pending = None
        self._cond = threading.Condition()
        self._running = True

        # 指標
        self._lock = threading.Lock()
        self._success = 0
        self._failure = 0
        self._dropped = 0
        self._last_success_ts = 0.0
        self._latencies = deque(maxlen=_LATENCY_WINDOW)

        self._last_warn_time = 0.0
        self._warn_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run, name="lick_nr", daemon=True
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
        """傳輸健康度快照（success / failure / dropped / p95 latency / last_success）。"""
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

    # ── 背景 worker ─────────────────────────────────────────────────────
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
            # 過去這裡是完全靜默的 except: pass——URL 設錯/Node-RED 掛掉時，
            # 這個 plugin 的資料會悄悄停止送達，沒有任何跡象。逐筆都記警告會
            # 洗版，所以節流成同一個 publisher 實例最多每 _WARN_INTERVAL_SEC
            # 秒警告一次，但至少確保問題持續發生時最終看得到。
            should_warn = False
            with self._warn_lock:
                now = time.time()
                if now - self._last_warn_time >= _WARN_INTERVAL_SEC:
                    self._last_warn_time = now
                    should_warn = True
            if should_warn:
                _log.warning(
                    "推送到 Node-RED 失敗（%s）：%s"
                    "（此類錯誤 %d 秒內只警告一次，避免洗版）",
                    self._url,
                    e,
                    int(_WARN_INTERVAL_SEC),
                )
