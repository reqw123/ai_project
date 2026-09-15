"""Per-frame lick analysis orchestrator."""

from collections import deque
from typing import Optional

import numpy as np

from plugins.lick_stage.analysis_context import (
    FrameState,
    ReasonCode,
    SCHEMA_VERSION,
)
from plugins.lick_stage.event_aggregator import EventAggregator
from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.contact_regions import (
    build_nose_trapezoid,
    compute_geometry,
    find_nearest_zone,
    trap_dir_from_perp,
)
from plugins.lick_stage.ear_distance import compute_ear_distance
from plugins.lick_stage.head_direction import (
    check_front_view_guard,
    compute_head_ear_angle,
    infer_face_state_cat_centric,
    infer_face_state_user_rules,
    smooth_state,
    stabilize_direction_vector,
)
from plugins.lick_stage.models import LickResult, ZoneStats
from plugins.lick_stage.statistics import LickStatistics


class LickAnalyzer:
    """
    Stateful per-frame lick zone and face direction analyzer.

    Call analyze() once per frame in order.
    """

    def __init__(self, event_aggregator_factory=None):
        # event_aggregator_factory：呼叫端（manager）提供的 () -> EventAggregator
        # 工廠，讓 reset() 能在新 Session 建立乾淨的聚合器並接上 storage sink。
        # 未提供時用無 sink 的預設聚合器（仍會逐幀計算，只是不落地）。
        self._events_factory = event_aggregator_factory or (lambda: EventAggregator())
        self._events = self._events_factory()
        self._stats = LickStatistics()
        self._state_history = deque(maxlen=_C.STATE_SMOOTH_WINDOW)
        # 上一幀「已穩定」的方向向量，供翻轉感知 EMA 使用。
        # trap_dir 的「候選值」由 trap_dir_from_perp() 從穩定後的 trap_perp
        # 決定性推導（不再依賴容易被雜訊干擾的 body_center 判斷）；trap_perp
        # 用「翻轉感知 EMA + 連續反向確認」穩定，trap_dir 只用純翻轉感知 EMA
        # （見 config.py 的 TRAP_DIR_EMA_ALPHA 註解說明為何不套用確認幀數）。
        self._prev_trap_perp: Optional[np.ndarray] = None
        self._prev_trap_dir: Optional[np.ndarray] = None
        # 連續「反向」讀數的計數（僅 trap_perp 使用），用來分辨單幀雜訊 vs. 真正的方向改變
        self._trap_perp_flip_streak = 0
        # 連續「trap_dir 偏離 y>=0（物理上不該發生）」的計數，用來分辨臨界水平
        # 抖動 vs. 真的卡在錯誤方向；見 _stabilize_nose_trapezoid() 說明
        self._trap_dir_wrong_streak = 0
        # Last-frame overlay state (consumed by manager.draw_overlay)
        self.last_trap_pts: Optional[np.ndarray] = None  # (4,2) float64 or None
        self.last_hit: bool = False
        self.last_zone_label: str = "NO_TARGET"
        self.last_nose_xy: tuple = (0.0, 0.0)
        self.last_geom: Optional[dict] = None  # full target_geom dict
        self.last_nearest_label: str = "NO_TARGET"
        # 第一階段：本幀契約資訊，供 _build_result 帶進 payload
        self._ctx_session_id: str = ""
        self._ctx_source_ts = None

    def reset(self) -> None:
        """清空所有跨幀狀態與累計統計（新 Session / 明確來源切換時呼叫）。"""
        self._events = self._events_factory()
        self._stats = LickStatistics()
        self._state_history.clear()
        self._prev_trap_perp = None
        self._prev_trap_dir = None
        self._trap_perp_flip_streak = 0
        self._trap_dir_wrong_streak = 0
        self.last_trap_pts = None
        self.last_hit = False
        self.last_zone_label = "NO_TARGET"
        self.last_nose_xy = (0.0, 0.0)
        self.last_geom = None
        self.last_nearest_label = "NO_TARGET"

    def finalize(self, end_source_ts=None) -> None:
        """Session 結束：結算最後一段 active bout 與最後一個未滿視窗。"""
        self._stats.finalize()
        try:
            self._events.finalize(end_source_ts)
        except Exception:
            pass

    def analyze(
        self,
        kpts,
        kpt_conf,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
        *,
        cat_present: Optional[bool] = None,
        is_lick: Optional[bool] = None,
        lick_confidence: Optional[float] = None,
        source_timestamp=None,
        session_id: str = "",
        discontinuity: bool = False,
        pose_quality: Optional[float] = None,
    ) -> LickResult:
        """分析單一影格的舔舐區域與臉部朝向，回傳本幀分析結果。

        向後相容：舊呼叫端只傳位置參數 (kpts, kpt_conf, frame_idx,
        elapsed_sec, dt_sec)，語意是「kpts 有值 ⟺ 已通過 ST-GCN lick gate」。
        新呼叫端（FrameProcessor 第一階段改動後）額外傳 cat_present / is_lick
        等契約欄位，讓 NO_CAT 與 NOT_LICK 可被區分（說明書「狀態與原因碼重構」）。

        pose_quality：M4 由共用 PoseFilter（frame_processor.py 建構、經
        LickStagePlugin.update() 傳入）提供的骨長品質分數，`None` 代表呼叫端
        未接上共用 PoseFilter 或該幀尚無法計算，純轉送給事件聚合器，這裡
        不做任何判斷。kpts 本身也預期已經是共用 PoseFilter 平滑過的結果——
        本方法自 M4 第二階段起不再自行對 kpts 做 EMA（見 _handle_cat()）。
        """
        if cat_present is None:
            cat_present = kpts is not None and kpt_conf is not None
        if is_lick is None:
            is_lick = kpts is not None and kpt_conf is not None

        self._ctx_session_id = session_id
        self._ctx_source_ts = source_timestamp

        if not cat_present:
            result = self._handle_no_cat(
                frame_idx, elapsed_sec, dt_sec, discontinuity
            )
        elif not is_lick:
            result = self._handle_not_lick(
                frame_idx, elapsed_sec, dt_sec, discontinuity
            )
        elif kpts is None or kpt_conf is None:
            result = self._handle_lick_no_pose(
                frame_idx, elapsed_sec, dt_sec, lick_confidence, discontinuity
            )
        else:
            result = self._handle_cat(
                kpts,
                kpt_conf,
                frame_idx,
                elapsed_sec,
                dt_sec,
                lick_confidence,
                discontinuity,
            )

        # 逐幀餵事件 / 視窗聚合器（說明書「單一事件聚合原則」）。純資料處理，
        # 失敗不影響本幀結果。
        try:
            self._events.feed(
                source_ts=(
                    source_timestamp
                    if source_timestamp is not None
                    else elapsed_sec
                ),
                frame_idx=frame_idx,
                dt_sec=dt_sec,
                frame_state=result.frame_state,
                zone_label=result.current_zone,
                action_score=lick_confidence,
                pose_quality=pose_quality,
                reason_code=result.reason_code,
                discontinuity=discontinuity,
            )
        except Exception:
            pass
        return result

    # ── Private helpers ───────────────────────────────────────────────

    def _reset_transient_state(self) -> None:
        """清空跨幀平滑狀態（EMA / 梯形方向 / 翻轉計數）與 overlay 快取。

        「貓離開畫面」與「非舔毛」都走這條路：兩者都不該讓陳舊姿態繼續拉動
        下一段真正的舔毛判定（說明書「update：track 切換不得沿用前一隻貓的
        狀態」的同一種考量）。M4 接線後關鍵點平滑狀態由呼叫端的共用
        PoseFilter 持有，這裡不再需要清空自己的 EMA；frame_processor.py 的
        _notify_plugins() 在同樣的時機點呼叫 PoseFilter.reset()，維持跟這裡
        一致的重置政策。
        """
        self._prev_trap_perp = None
        self._prev_trap_dir = None
        self._trap_perp_flip_streak = 0
        self._trap_dir_wrong_streak = 0
        self.last_trap_pts = None
        self.last_hit = False
        self.last_geom = None
        self.last_nearest_label = "NO_TARGET"

    def _handle_no_cat(
        self,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
        discontinuity: bool = False,
    ) -> LickResult:
        self._reset_transient_state()
        self._state_history.append(_C.STATE_NO_CAT)
        state_sm, stability = smooth_state(self._state_history)
        self._stats.update(
            "NO_TARGET", dt_sec, FrameState.NO_CAT, discontinuity=discontinuity
        )
        return self._build_result(
            "NO_TARGET",
            state_sm,
            stability,
            False,
            frame_idx,
            elapsed_sec,
            frame_state=FrameState.NO_CAT,
        )

    def _handle_not_lick(
        self,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
        discontinuity: bool = False,
    ) -> LickResult:
        """有貓、但 ST-GCN 當幀非 lick。時間計入 valid_observed_sec，不計舔毛。"""
        self._reset_transient_state()
        self._state_history.append(_C.STATE_NO_CAT)
        state_sm, stability = smooth_state(self._state_history)
        self._stats.update(
            "NO_TARGET", dt_sec, FrameState.NOT_LICK, discontinuity=discontinuity
        )
        return self._build_result(
            "NO_TARGET",
            state_sm,
            stability,
            False,
            frame_idx,
            elapsed_sec,
            frame_state=FrameState.NOT_LICK,
        )

    def _handle_lick_no_pose(
        self,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
        lick_confidence,
        discontinuity: bool = False,
    ) -> LickResult:
        """呼叫端判定正在舔毛，但沒有可用姿態 → LICK_UNASSIGNED / POSE_INVALID。"""
        self._reset_transient_state()
        self._state_history.append(_C.STATE_NO_CAT)
        state_sm, stability = smooth_state(self._state_history)
        self._stats.update(
            "NO_TARGET",
            dt_sec,
            FrameState.LICK_UNASSIGNED,
            discontinuity=discontinuity,
        )
        return self._build_result(
            "NO_TARGET",
            state_sm,
            stability,
            False,
            frame_idx,
            elapsed_sec,
            frame_state=FrameState.LICK_UNASSIGNED,
            reason_code=ReasonCode.POSE_INVALID,
        )

    def _stabilize_vector(self, new_vec, prev_vec, flip_streak: int):
        """單一方向向量（trap_perp）的翻轉感知穩定化，外加「連續反向才接受」的安全閥。

        單幀雜訊造成的反向讀數：flip_streak 累加但未達門檻時，直接沿用前一個
        穩定方向（不理會這幀雜訊，也不 blend 進去，避免被拖著慢慢偏移）。
        連續 TRAP_PERP_FLIP_CONFIRM_FRAMES 幀都反向：視為真正的方向改變
        （例如貓整個轉身），直接採用新方向，不強行沿用舊方向造成永久卡死。

        Returns (stable_vec, new_flip_streak)。
        """
        new_vec = np.asarray(new_vec, dtype=np.float64)
        norm = float(np.hypot(new_vec[0], new_vec[1]))
        if norm < 1e-9:
            return (prev_vec if prev_vec is not None else new_vec), 0
        unit = new_vec / norm

        if prev_vec is None:
            return unit, 0

        dot = float(np.dot(unit, prev_vec))
        if dot < -_C.TRAP_PERP_FLIP_MARGIN:
            flip_streak += 1
            if flip_streak >= _C.TRAP_PERP_FLIP_CONFIRM_FRAMES:
                return unit, 0  # 真正的方向改變：直接採用新方向，重置計數
            return prev_vec, flip_streak  # 尚未確認：視為雜訊，沿用前一穩定方向

        # 非反向讀數：正常翻轉感知 EMA，並清空反向計數
        stable = stabilize_direction_vector(
            unit, prev_vec, _C.TRAP_PERP_EMA_ALPHA, _C.TRAP_PERP_FLIP_MARGIN
        )
        return stable, 0

    def _stabilize_nose_trapezoid(self, target_geom: Optional[dict]) -> None:
        """跨幀穩定鼻子接觸梯形，就地更新 target_geom['nose_contact_trapezoid']。

        compute_geometry() 每幀從當前關鍵點重新算 trap_perp，對耳間距過短
        （貓側躺、頭部縮短）時的雜訊很敏感，容易讓梯形角度frame-to-frame跳動。
        trap_perp 用「翻轉感知 EMA + 連續反向確認」跨幀穩定。

        trap_dir 只在「第一次出現」（_prev_trap_dir 尚未建立）時，用
        trap_dir_from_perp() 強制指向影像下方一次，之後每一幀改成單純對
        [-perp.y, perp.x] 這個旋轉結果做翻轉感知 EMA，**不再每幀重新
        強制 y>=0**：這個「每幀都強制」的做法試過了，反而在 trap_dir 本身
        接近水平（耳朵連線接近垂直，例如貓側躺頭部縮短時）時，會在一個已經
        被 EMA 穩定收斂、y 分量微小的結果上又做一次無緩衝的硬性翻轉，等於
        把不穩定性從 trap_perp 轉移到自己身上。只在初始化時定調一次方向、
        後續單純平滑追蹤，才能真正繼承 trap_perp 的穩定性，不引入新的
        雜訊來源。「短邊保證在上面」因此是初始化時就決定好、且在正常小幅
        抖動下會一路保持的強穩定狀態，而非每幀都重新驗證的絕對數學保證
        ——真要讓貓整個轉一圈頭部持續轉向的極端情況，才可能讓它跟著轉。

        但純 EMA 追蹤完全信任 _prev_trap_dir 這個歷史錨點，若初始化那一刻
        剛好定出不符合直覺的方向，後續會一路「穩定地」錯下去、沒有機制
        自我修正。因此在 EMA 之後另外加一道獨立安全網：連續好幾幀
        （TRAP_DIR_WRONG_SIDE_CONFIRM_FRAMES）都偏離 y>=0 時才強制翻轉拉回，
        平常的臨界水平抖動只會讓計數器歸零、不會觸發翻轉，不影響上述
        「不每幀強制」想保留的穩定性。

        2026-09-11：兩次試過拿掉這裡的 y>=0（先改 body_axis_unit，再改
        head_axis_unit），三支影片實測都測不贏這個版本，已撤回，見
        contact_regions.py 的 trap_dir_from_perp() docstring 完整沿革。
        """
        if target_geom is None:
            return
        trap_perp = target_geom.get("trap_perp")
        if trap_perp is None:
            return

        stable_perp, self._trap_perp_flip_streak = self._stabilize_vector(
            trap_perp,
            self._prev_trap_perp,
            self._trap_perp_flip_streak,
        )
        self._prev_trap_perp = stable_perp

        if self._prev_trap_dir is None:
            stable_dir = trap_dir_from_perp(stable_perp)
            self._trap_dir_wrong_streak = 0
        else:
            dir_candidate = np.array(
                [-float(stable_perp[1]), float(stable_perp[0])], dtype=np.float64
            )
            stable_dir = stabilize_direction_vector(
                dir_candidate,
                self._prev_trap_dir,
                _C.TRAP_DIR_EMA_ALPHA,
                _C.TRAP_DIR_FLIP_MARGIN,
            )
            # 安全網：純 EMA 追蹤不會主動驗證 y>=0，只有連續多幀持續偏離
            # （非臨界水平抖動，是真的卡在錯誤方向）才強制翻轉拉回正確半球
            if float(stable_dir[1]) < 0.0:
                self._trap_dir_wrong_streak += 1
                if self._trap_dir_wrong_streak >= _C.TRAP_DIR_WRONG_SIDE_CONFIRM_FRAMES:
                    stable_dir = -stable_dir
                    self._trap_dir_wrong_streak = 0
            else:
                self._trap_dir_wrong_streak = 0
        self._prev_trap_dir = stable_dir

        target_geom["trap_perp"] = stable_perp
        target_geom["trap_dir"] = stable_dir
        target_geom["nose_contact_trapezoid"] = build_nose_trapezoid(
            target_geom["nose"],
            stable_perp,
            stable_dir,
            target_geom["trap_top_half"],
            target_geom["trap_bot_half"],
            target_geom["trap_height"],
        )

    def _handle_cat(
        self,
        kpts,
        kpt_conf,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
        lick_confidence=None,
        discontinuity: bool = False,
    ) -> LickResult:
        _nan = float("nan")

        # M4 第二階段：關鍵點平滑已由呼叫端的共用 PoseFilter（見
        # frame_processor.py）完成，這裡收到的 kpts 視為已平滑，不再自己套
        # 一層 EMA（避免跟共用 PoseFilter 的 alpha 疊加變成雙重平滑）。直接
        # 呼叫 analyze()（未經共用 PoseFilter，例如測試/獨立工具）的呼叫端
        # 會拿到未平滑的原始關鍵點，跟共用 PoseFilter 停用時（alpha>=1.0）
        # 的行為一致。
        smooth_kpts = np.asarray(kpts, dtype=np.float64)

        dist_px, dist_norm, valid, _body_scale, body_ear_ratio = compute_ear_distance(
            smooth_kpts, kpt_conf
        )
        front_guard = check_front_view_guard(kpt_conf, dist_px, body_ear_ratio)
        nose_conf = float(kpt_conf[_C.KP_NOSE])
        nose_ok = nose_conf >= _C.NOSE_CONF_THRESHOLD
        angle_deg = compute_head_ear_angle(smooth_kpts, kpt_conf)

        gaze_fwd = gaze_lat = gaze_angle = _nan
        geometry_score = _nan

        if front_guard:
            if (
                _C.BACK_VIEW_REQUIRE_LOW_NOSE
                and nose_conf <= _C.BACK_CAMERA_NOSE_CONF_MAX
            ):
                state_now = _C.STATE_BACK
            else:
                state_now = _C.STATE_FRONT_VIEW
            self._state_history.append(state_now)
            state_sm = state_now
            stability = 1.0
            zone_label = "NO_TARGET"
            # 正面姿態下 2D 幾何無法分辨部位 → 未指派，原因 FRONT_VIEW_UNOBSERVABLE
            frame_state = FrameState.LICK_UNASSIGNED
            reason_code = ReasonCode.FRONT_VIEW_UNOBSERVABLE
            self.last_trap_pts = None
            self.last_hit = False
            self.last_geom = None
            self.last_nearest_label = "NO_TARGET"
        else:
            target_geom = compute_geometry(smooth_kpts, kpt_conf)
            self._stabilize_nose_trapezoid(target_geom)
            cat_state, gaze_fwd, gaze_lat, gaze_angle = infer_face_state_cat_centric(
                target_geom, nose_ok
            )
            state_now, rule_applied = infer_face_state_user_rules(
                angle_deg, dist_norm, dist_px, nose_conf
            )
            if state_now == _C.STATE_UNKNOWN:
                state_now = cat_state
            self._state_history.append(state_now)
            if rule_applied and state_now in (
                _C.STATE_FRONT,
                _C.STATE_FRONT_LEFT,
                _C.STATE_FRONT_RIGHT,
                _C.STATE_BACK,
            ):
                state_sm = state_now
                stability = 1.0
            else:
                state_sm, stability = smooth_state(self._state_history)

            nearest_label, geometry_score, hit = find_nearest_zone(target_geom)
            zone_label = nearest_label if hit else "NO_TARGET"
            if hit:
                frame_state = FrameState.LICK_ASSIGNED
                reason_code = None
            else:
                # 姿態有效但鼻端梯形未命中任何區域 → 未指派，原因 NO_REGION_HIT
                frame_state = FrameState.LICK_UNASSIGNED
                reason_code = ReasonCode.NO_REGION_HIT

            # Store overlay state for draw_overlay()
            trap_raw = target_geom.get("nose_contact_trapezoid")
            self.last_trap_pts = (
                np.asarray(trap_raw, dtype=np.float64) if trap_raw is not None else None
            )
            self.last_hit = bool(hit)
            self.last_zone_label = nearest_label
            self.last_nearest_label = nearest_label
            self.last_geom = target_geom
            nose_kp = smooth_kpts[_C.KP_NOSE]
            self.last_nose_xy = (float(nose_kp[0]), float(nose_kp[1]))

        self._stats.update(
            zone_label, dt_sec, frame_state, discontinuity=discontinuity
        )
        return self._build_result(
            zone_label,
            state_sm,
            stability,
            valid,
            frame_idx,
            elapsed_sec,
            dist_px,
            dist_norm,
            gaze_fwd,
            gaze_lat,
            gaze_angle,
            geometry_score,
            frame_state=frame_state,
            reason_code=reason_code,
        )

    def _build_result(
        self,
        zone_label: str,
        state_sm: str,
        stability: float,
        valid: bool,
        frame_idx: int,
        elapsed_sec: float,
        dist_px: float = float("nan"),
        dist_norm: float = float("nan"),
        gaze_fwd: float = float("nan"),
        gaze_lat: float = float("nan"),
        gaze_angle: float = float("nan"),
        geometry_score: float = float("nan"),
        frame_state: str = FrameState.NO_CAT,
        reason_code=None,
    ) -> LickResult:
        trap_pts = self.last_trap_pts.tolist() if self.last_trap_pts is not None else []
        nose_xy = list(self.last_nose_xy) if trap_pts else []

        def _zs(key: str) -> ZoneStats:
            hits, t = self._stats.zone_stats(key)
            return ZoneStats(hits=hits, time_sec=t)

        st = self._stats
        return LickResult(
            current_zone=zone_label,
            best_zone=st.best_zone(),
            body=_zs("BODY"),
            fl=_zs("FL"),
            fr=_zs("FR"),
            hl=_zs("HL"),
            hr=_zs("HR"),
            ambiguous=_zs("AMBIGUOUS"),
            geometry_score=geometry_score,
            face_state=state_sm,
            state_stability=stability,
            valid=valid,
            frame=frame_idx,
            time_sec=elapsed_sec,
            dist_px=dist_px,
            dist_norm=dist_norm,
            gaze_fwd=gaze_fwd,
            gaze_lat=gaze_lat,
            gaze_angle=gaze_angle,
            trap_pts=trap_pts,
            nose_xy=nose_xy,
            # ── 第一階段 v2 契約欄位（說明書「統計分母與計算口徑」）──────
            schema_version=SCHEMA_VERSION,
            session_id=self._ctx_session_id,
            source_timestamp=(
                float(self._ctx_source_ts)
                if isinstance(self._ctx_source_ts, (int, float))
                else None
            ),
            frame_state=frame_state,
            reason_code=reason_code,
            observed_sec=st.observed_sec,
            valid_observed_sec=st.valid_observed_sec,
            no_cat_sec=st.no_cat_sec,
            stgcn_lick_sec=st.stgcn_lick_sec,
            assigned_zone_sec=st.assigned_zone_sec,
            unassigned_lick_sec=st.unassigned_lick_sec,
            zone_coverage_ratio=st.zone_coverage_ratio(),
            unknown_rate=st.unknown_rate(),
        )
