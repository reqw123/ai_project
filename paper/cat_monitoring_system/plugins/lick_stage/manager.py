"""LickStagePlugin — public facade for the lick stage plugin."""

import logging
import time

from plugins.lick_stage.analysis_context import AnalysisContext, SourceClock
from plugins.lick_stage.analyzer import LickAnalyzer
from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.event_aggregator import EventAggregator
from plugins.lick_stage.overlay import draw_all_overlays
from plugins.lick_stage.publisher import NodeRedPublisher
from plugins.lick_stage.storage import LickStorage

_log = logging.getLogger(__name__)


class LickStagePlugin:
    """
    Pluggable lick-stage analysis module.

    Integration contract
    ────────────────────
    • Call update(kpts, kpt_conf, **context) once per processed frame.
    • kpts     — (17, 2) float32/float64 numpy array, or None when no cat.
    • kpt_conf — (17,)   float32/float64 numpy array, or None when no cat.
    • Optional keyword context (第一階段新增，見說明書「來源時間規格」)：
        source_timestamp — 來源媒體時間（秒）；影片 PTS 或 frame_idx/source_fps。
        dt_sec           — 呼叫端已算好的來源時間差；未給則由 source_timestamp 推導。
        frame_idx        — 單一影片內單調遞增的影格序號。
        cat_present      — 畫面裡是否有貓（與 is_lick 分離）。
        is_lick          — ST-GCN 當幀是否判定為 lick。
        lick_confidence  — ST-GCN lick 信心值。
        track_id / session_id — 追蹤與 Session 身份。
      全部省略時退回舊行為（wall-clock 計時、kpts 有值 ⟺ 通過 lick gate），
      現有呼叫端不受影響。
    • Any exception raised inside update() is caught and logged at DEBUG
      level so it can never crash the main system.

    The plugin can be removed entirely (delete the plugins/lick_stage/
    directory) without affecting the main system — the registration in
    routes.py uses a try/except import.
    """

    def __init__(
        self,
        nodered_url: str = _C.NODERED_URL,
        storage_db_path: str = _C.STORAGE_DB_PATH,
    ):
        self._session_id = ""
        self._context = None  # AnalysisContext once a session starts
        self._session_active = False
        self._frame_count = 0
        # 舊路徑（wall-clock）計時狀態
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        # 新路徑（來源時間）計時狀態
        self._clock = SourceClock()
        self._source_elapsed = 0.0

        self._publisher = NodeRedPublisher(nodered_url) if nodered_url else None
        self._storage = LickStorage(storage_db_path)
        # analyzer 建構時會呼叫一次工廠，需先備妥 _context / _storage
        self._analyzer = LickAnalyzer(
            event_aggregator_factory=self._make_event_aggregator
        )

    def _make_event_aggregator(self) -> EventAggregator:
        """analyzer.reset() 每次呼叫都會透過這個工廠建立乾淨的聚合器，
        並把 sink 接到 storage（storage 停用時 sink 仍安全 no-op）。"""
        period = self._context.period if self._context is not None else ""
        return EventAggregator(
            window_sec=_C.WINDOW_SUMMARY_SEC,
            period=period,
            on_event=self._storage.write_event,
            on_window=self._storage.write_window,
        )

    # ── Session 生命週期（說明書「Session 生命週期」）──────────────────────
    def start_session(
        self,
        session_id: str,
        *,
        video_id: str = "",
        cat_id=None,
        period: str = "",
        model_version: str = "",
        config_hash: str = "",
        source_fps: float = 0.0,
    ) -> None:
        """清空前一段的 EMA / 方向 / 候選 / bout 狀態，建立不可變 metadata。"""
        self._session_id = session_id
        self._context = AnalysisContext(
            session_id=session_id,
            video_id=video_id,
            cat_id=cat_id,
            period=period,
            model_version=model_version,
            config_hash=config_hash,
            source_fps=float(source_fps or 0.0),
        )
        self._clock.reset()
        self._source_elapsed = 0.0
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        self._frame_count = 0
        self._analyzer.reset()  # 透過工廠建立接上 storage sink 的新聚合器
        try:
            self._storage.open_session(self._context)
        except Exception as exc:  # fail-safe
            _log.debug("LickStagePlugin storage.open_session error: %s", exc)
        self._session_active = True

    def finish_session(self, end_source_timestamp=None) -> None:
        """結算尚未結束的 active bout（避免最後一段舔毛事件遺失），
        並關閉 storage（結算 session row、匯出 CSV）。"""
        try:
            self._analyzer.finalize(end_source_timestamp)
        except Exception as exc:  # fail-safe
            _log.debug("LickStagePlugin.finish_session error: %s", exc)
        try:
            self._storage.close_session()
        except Exception as exc:
            _log.debug("LickStagePlugin storage.close_session error: %s", exc)
        self._session_active = False

    def reset_session(self, reason: str = "") -> None:
        """只在明確來源切換 / 錯誤恢復 / 人工重跑時使用。"""
        _log.debug("LickStagePlugin.reset_session (%s)", reason)
        self._clock.reset()
        self._source_elapsed = 0.0
        self._elapsed_sec = 0.0
        self._last_wall_t = time.monotonic()
        self._analyzer.reset()

    def update(
        self,
        kpts,
        kpt_conf,
        *,
        source_timestamp=None,
        dt_sec=None,
        frame_idx=None,
        cat_present=None,
        is_lick=None,
        lick_confidence=None,
        track_id=None,
        session_id=None,
    ) -> None:
        """Fail-safe entry point. Never raises."""
        try:
            use_source_time = source_timestamp is not None or dt_sec is not None

            if use_source_time:
                # 首次呼叫但尚未 start_session：以呼叫端傳的 session_id 自動開場，
                # 讓 FrameProcessor 不必額外協調生命週期（M2 再正式接管）。
                if not self._session_active:
                    self.start_session(session_id or self._session_id or "S_auto")

                if dt_sec is not None:
                    dt, discontinuity = self._clock.clamp_external_dt(
                        dt_sec, source_timestamp
                    )
                else:
                    dt, discontinuity, _ts_invalid = self._clock.tick(source_timestamp)
                self._source_elapsed += dt
                elapsed = self._source_elapsed
            else:
                # 舊路徑：wall-clock 計時（現有呼叫端與既有測試維持不變）
                now = time.monotonic()
                dt = max(0.0, now - self._last_wall_t)
                self._last_wall_t = now
                self._elapsed_sec += dt
                elapsed = self._elapsed_sec
                discontinuity = False

            self._frame_count += 1
            fidx = int(frame_idx) if frame_idx is not None else self._frame_count

            cp = cat_present
            il = is_lick
            if cp is None:
                # 舊語意：kpts 有值 ⟺ 已通過 ST-GCN lick gate
                cp = kpts is not None and kpt_conf is not None
                il = cp

            result = self._analyzer.analyze(
                kpts,
                kpt_conf,
                fidx,
                elapsed,
                dt,
                cat_present=cp,
                is_lick=il,
                lick_confidence=lick_confidence,
                source_timestamp=source_timestamp,
                session_id=session_id or self._session_id,
                discontinuity=discontinuity,
            )

            if self._publisher is not None:
                self._publisher.publish(result.to_payload())

        except Exception as exc:
            _log.debug("LickStagePlugin.update error: %s", exc)

    def draw_overlay(self, frame, frame_idx: int = 0, show: bool = True) -> None:
        """Draw all lick-stage overlays onto *frame* in-place. Fail-safe."""
        try:
            draw_all_overlays(
                frame,
                self._analyzer.last_geom,
                self._analyzer.last_trap_pts,
                self._analyzer.last_hit,
                self._analyzer.last_zone_label,
                self._analyzer.last_nearest_label,
                self._analyzer.last_nose_xy,
                frame_idx=frame_idx,
                show=show,
            )
        except Exception as exc:
            _log.debug("LickStagePlugin.draw_overlay error: %s", exc)

    def close(self) -> None:
        """關閉底層 Node-RED 發送器與 storage（若有啟用）。"""
        try:
            self.finish_session()
        except Exception:
            pass
        try:
            self._storage.close()
        except Exception:
            pass
        if self._publisher is not None:
            self._publisher.close()
