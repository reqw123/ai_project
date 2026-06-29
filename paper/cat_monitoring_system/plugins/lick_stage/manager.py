"""LickStagePlugin — public facade for the lick stage plugin."""
import time
import logging

from plugins.lick_stage.analyzer import LickAnalyzer
from plugins.lick_stage.publisher import NodeRedPublisher
from plugins.lick_stage.config import LickConfig as _C

_log = logging.getLogger(__name__)


class LickStagePlugin:
    """
    Pluggable lick-stage analysis module.

    Integration contract
    ────────────────────
    • Call update(kpts, kpt_conf) once per processed frame.
    • kpts     — (17, 2) float32/float64 numpy array, or None when no cat.
    • kpt_conf — (17,)   float32/float64 numpy array, or None when no cat.
    • Any exception raised inside update() is caught and logged at DEBUG
      level so it can never crash the main system.

    The plugin can be removed entirely (delete the plugins/lick_stage/
    directory) without affecting the main system — the registration in
    routes.py uses a try/except import.
    """

    def __init__(self, nodered_url: str = _C.NODERED_URL):
        self._analyzer  = LickAnalyzer()
        self._publisher = NodeRedPublisher(nodered_url) if nodered_url else None
        self._frame_count  = 0
        self._elapsed_sec  = 0.0
        self._last_wall_t  = time.monotonic()

    def update(self, kpts, kpt_conf) -> None:
        """Fail-safe entry point. Never raises."""
        try:
            now    = time.monotonic()
            dt_sec = max(0.0, now - self._last_wall_t)
            self._last_wall_t   = now
            self._frame_count  += 1
            self._elapsed_sec  += dt_sec

            result = self._analyzer.analyze(
                kpts, kpt_conf,
                self._frame_count,
                self._elapsed_sec,
                dt_sec,
            )

            if self._publisher is not None:
                self._publisher.publish(result.to_payload())

        except Exception as exc:
            _log.debug("LickStagePlugin.update error: %s", exc)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
