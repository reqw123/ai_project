"""
Node-RED 通訊 — 非阻塞雙端點背景發送

設計原則：
  - send_data() 立即返回，絕不阻塞幀處理迴圈
  - v1 / v2 各自獨立 worker thread，互不影響
  - 佇列容量 = 1（drop-on-full）：永遠傳最新資料，不積壓舊訊息
"""
import logging
import queue
import threading
import requests
from datetime import datetime
from config import NodeRedConfig


class _EndpointWorker:
    """單一端點的背景發送者。佇列滿時捨棄舊資料，保留最新一筆。"""

    def __init__(self, url: str, label: str):
        self.url = url
        self.label = label
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"nodered-{label}"
        )
        self._thread.start()

    def put(self, data: dict) -> None:
        """非阻塞投入。若佇列已滿則捨棄舊值，投入新值。"""
        try:
            self._q.put_nowait(data)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(data)
            except queue.Full:
                pass

    def close(self) -> None:
        """送出停止訊號（None），讓 worker 結束迴圈。"""
        self.put(None)  # type: ignore[arg-type]

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                requests.post(self.url, json=item, timeout=NodeRedConfig.TIMEOUT)
            except Exception as exc:
                logging.warning("NodeRedClient [%s] POST failed: %s", self.label, exc)


class NodeRedClient:
    """
    管理 /yolo_result（v1）和 /yolo_result_v2（v2）的非阻塞推送。

    每個端點各有一條獨立 daemon thread；send_data() 呼叫後立即返回，
    背景 worker 自行處理網路延遲與重試，主迴圈零等待。
    """

    def __init__(self, url_result: str, url_notify: str | None = None, local_ip: str | None = None):
        self.url_notify = url_notify
        self.local_ip = local_ip

        v2_url = getattr(NodeRedConfig, "ENDPOINT_RESULT_V2", None)
        if v2_url == url_result:
            v2_url = None

        self._workers: list[_EndpointWorker] = []
        if url_result:
            self._workers.append(_EndpointWorker(url_result, "v1"))
        if v2_url:
            self._workers.append(_EndpointWorker(v2_url, "v2"))

    def send_data(self, data: dict) -> bool:
        """將資料投入所有端點的佇列，立即返回 True。"""
        for worker in self._workers:
            worker.put(data)
        return True

    def close(self) -> None:
        """通知所有 worker 結束（程式關閉時呼叫）。"""
        for worker in self._workers:
            worker.close()
