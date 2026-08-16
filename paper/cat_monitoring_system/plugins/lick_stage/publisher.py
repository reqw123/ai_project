"""Non-blocking HTTP publisher to Node-RED."""

import concurrent.futures
import logging
import threading
import time

try:
    import requests as _requests

    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

from plugins.lick_stage.config import LickConfig as _C

_log = logging.getLogger(__name__)

_WARN_INTERVAL_SEC = 30.0  # 同一個 publisher 實例失敗時最多多久警告一次


class NodeRedPublisher:
    """
    Sends JSON payloads to a Node-RED endpoint in a background thread pool.

    publish() returns immediately; the HTTP POST happens asynchronously.
    """

    def __init__(self, url: str, timeout: float = _C.NODERED_TIMEOUT):
        self._url = url
        self._timeout = timeout
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lick_nr"
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
            # 過去這裡是完全靜默的 except: pass——URL 設錯/Node-RED 掛掉時，
            # 這個 plugin 的資料會悄悄停止送達，沒有任何跡象。這條路徑呼叫
            # 頻率高（每個 process 循環都可能推送一次），逐筆都記警告會洗版，
            # 所以節流成同一個 publisher 實例最多每 _WARN_INTERVAL_SEC 秒警告
            # 一次，但至少確保問題持續發生時最終看得到。
            #
            # check-then-set 用 _warn_lock 保護：_post() 是在
            # ThreadPoolExecutor(max_workers=2) 裡跑的，兩個 worker 幾乎同時
            # 送出的請求都失敗時，若沒有鎖，兩者都可能在對方更新
            # _last_warn_time 之前判斷「超過節流間隔」，導致同一次故障重複
            # 印出兩則警告，節流形同虛設。
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
