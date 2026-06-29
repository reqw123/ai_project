"""Non-blocking HTTP publisher to Node-RED."""
import concurrent.futures
import logging

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _requests = None
    _HAS_REQUESTS = False

from plugins.lick_stage.config import LickConfig as _C

_log = logging.getLogger(__name__)


class NodeRedPublisher:
    """
    Sends JSON payloads to a Node-RED endpoint in a background thread pool.

    publish() returns immediately; the HTTP POST happens asynchronously.
    """

    def __init__(self, url: str, timeout: float = _C.NODERED_TIMEOUT):
        self._url     = url
        self._timeout = timeout
        self._pool    = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lick_nr"
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
