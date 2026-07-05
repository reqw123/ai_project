"""Per-frame lick analysis orchestrator."""
from collections import deque
from typing import Optional
import numpy as np

from plugins.lick_stage.config import LickConfig as _C
from plugins.lick_stage.models import LickResult, ZoneStats
from plugins.lick_stage.statistics import LickStatistics
from plugins.lick_stage.ear_distance import compute_ear_distance
from plugins.lick_stage.head_direction import (
    compute_head_ear_angle,
    infer_face_state_cat_centric,
    infer_face_state_user_rules,
    smooth_state,
    check_front_view_guard,
    stabilize_direction_vector,
)
from plugins.lick_stage.contact_regions import (
    compute_geometry,
    find_nearest_zone,
    build_nose_trapezoid,
    trap_dir_from_perp,
)


class LickAnalyzer:
    """
    Stateful per-frame lick zone and face direction analyzer.

    Call analyze() once per frame in order.
    """

    def __init__(self):
        self._stats         = LickStatistics()
        self._state_history = deque(maxlen=_C.STATE_SMOOTH_WINDOW)
        self._ema_kpts: Optional[np.ndarray] = None
        # 上一幀「已穩定」的方向向量，供翻轉感知 EMA 使用。
        # trap_dir 的「候選值」由 trap_dir_from_perp() 從穩定後的 trap_perp
        # 決定性推導（不再依賴容易被雜訊干擾的 body_center 判斷）；trap_perp
        # 用「翻轉感知 EMA + 連續反向確認」穩定，trap_dir 只用純翻轉感知 EMA
        # （見 config.py 的 TRAP_DIR_EMA_ALPHA 註解說明為何不套用確認幀數）。
        self._prev_trap_perp: Optional[np.ndarray] = None
        self._prev_trap_dir:  Optional[np.ndarray] = None
        # 連續「反向」讀數的計數（僅 trap_perp 使用），用來分辨單幀雜訊 vs. 真正的方向改變
        self._trap_perp_flip_streak = 0
        # Last-frame overlay state (consumed by manager.draw_overlay)
        self.last_trap_pts:    Optional[np.ndarray] = None  # (4,2) float64 or None
        self.last_hit:         bool = False
        self.last_zone_label:  str  = "NO_TARGET"
        self.last_nose_xy:     tuple = (0.0, 0.0)
        self.last_geom:        Optional[dict] = None        # full target_geom dict
        self.last_nearest_label: str = "NO_TARGET"

    def analyze(
        self,
        kpts,
        kpt_conf,
        frame_idx: int,
        elapsed_sec: float,
        dt_sec: float,
    ) -> LickResult:
        if kpts is None or kpt_conf is None:
            return self._handle_no_cat(frame_idx, elapsed_sec, dt_sec)
        return self._handle_cat(kpts, kpt_conf, frame_idx, elapsed_sec, dt_sec)

    def reset(self) -> None:
        self._stats.reset()
        self._state_history.clear()
        self._ema_kpts = None
        self._prev_trap_perp = None
        self._prev_trap_dir  = None
        self._trap_perp_flip_streak = 0

    # ── Private helpers ───────────────────────────────────────────────

    def _handle_no_cat(self, frame_idx: int, elapsed_sec: float, dt_sec: float) -> LickResult:
        self._ema_kpts    = None
        self._prev_trap_perp = None
        self._prev_trap_dir  = None
        self._trap_perp_flip_streak = 0
        self.last_trap_pts = None
        self.last_hit      = False
        self.last_geom     = None
        self.last_nearest_label = "NO_TARGET"
        self._state_history.append(_C.STATE_NO_CAT)
        state_sm, stability = smooth_state(self._state_history)
        self._stats.update("NO_TARGET", dt_sec)
        return self._build_result("NO_TARGET", state_sm, stability, False, frame_idx, elapsed_sec)

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
        stable = stabilize_direction_vector(unit, prev_vec, _C.TRAP_PERP_EMA_ALPHA, _C.TRAP_PERP_FLIP_MARGIN)
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
        """
        if target_geom is None:
            return
        trap_perp = target_geom.get("trap_perp")
        if trap_perp is None:
            return

        stable_perp, self._trap_perp_flip_streak = self._stabilize_vector(
            trap_perp, self._prev_trap_perp, self._trap_perp_flip_streak,
        )
        self._prev_trap_perp = stable_perp

        if self._prev_trap_dir is None:
            stable_dir = trap_dir_from_perp(stable_perp)
        else:
            dir_candidate = np.array([-float(stable_perp[1]), float(stable_perp[0])], dtype=np.float64)
            stable_dir = stabilize_direction_vector(
                dir_candidate, self._prev_trap_dir, _C.TRAP_DIR_EMA_ALPHA, _C.TRAP_DIR_FLIP_MARGIN,
            )
        self._prev_trap_dir = stable_dir

        target_geom["trap_perp"] = stable_perp
        target_geom["trap_dir"]  = stable_dir
        target_geom["nose_contact_trapezoid"] = build_nose_trapezoid(
            target_geom["nose"], stable_perp, stable_dir,
            target_geom["trap_top_half"], target_geom["trap_bot_half"], target_geom["trap_height"],
        )

    def _handle_cat(self, kpts, kpt_conf, frame_idx: int, elapsed_sec: float, dt_sec: float) -> LickResult:
        _nan = float("nan")

        # Optional keypoint EMA (default alpha=1.0 = bypass)
        if _C.EMA_ALPHA < 1.0 - 1e-9:
            if self._ema_kpts is None:
                self._ema_kpts = np.asarray(kpts, dtype=np.float64).copy()
            else:
                self._ema_kpts = _C.EMA_ALPHA * np.asarray(kpts, dtype=np.float64) + (1.0 - _C.EMA_ALPHA) * self._ema_kpts
            smooth_kpts = self._ema_kpts
        else:
            smooth_kpts = kpts

        dist_px, dist_norm, valid, _body_scale, body_ear_ratio = compute_ear_distance(smooth_kpts, kpt_conf)
        front_guard  = check_front_view_guard(kpt_conf, dist_px, body_ear_ratio)
        nose_conf    = float(kpt_conf[_C.KP_NOSE])
        nose_ok      = nose_conf >= _C.NOSE_CONF_THRESHOLD
        angle_deg    = compute_head_ear_angle(smooth_kpts, kpt_conf)

        gaze_fwd = gaze_lat = gaze_angle = _nan

        if front_guard:
            if _C.BACK_VIEW_REQUIRE_LOW_NOSE and nose_conf <= _C.BACK_CAMERA_NOSE_CONF_MAX:
                state_now = _C.STATE_BACK
            else:
                state_now = _C.STATE_FRONT_VIEW
            self._state_history.append(state_now)
            state_sm  = state_now
            stability = 1.0
            zone_label = "NO_TARGET"
            self.last_trap_pts     = None
            self.last_hit          = False
            self.last_geom         = None
            self.last_nearest_label = "NO_TARGET"
        else:
            target_geom = compute_geometry(smooth_kpts, kpt_conf)
            self._stabilize_nose_trapezoid(target_geom)
            cat_state, gaze_fwd, gaze_lat, gaze_angle = infer_face_state_cat_centric(target_geom, nose_ok)
            state_now, rule_applied = infer_face_state_user_rules(angle_deg, dist_norm, dist_px, nose_conf)
            if state_now == _C.STATE_UNKNOWN:
                state_now = cat_state
            self._state_history.append(state_now)
            if rule_applied and state_now in (
                _C.STATE_FRONT, _C.STATE_FRONT_LEFT, _C.STATE_FRONT_RIGHT, _C.STATE_BACK
            ):
                state_sm  = state_now
                stability = 1.0
            else:
                state_sm, stability = smooth_state(self._state_history)

            nearest_label, _dist, hit = find_nearest_zone(target_geom)
            zone_label = nearest_label if hit else "NO_TARGET"

            # Store overlay state for draw_overlay()
            trap_raw = target_geom.get("nose_contact_trapezoid")
            self.last_trap_pts      = np.asarray(trap_raw, dtype=np.float64) if trap_raw is not None else None
            self.last_hit           = bool(hit)
            self.last_zone_label    = nearest_label
            self.last_nearest_label = nearest_label
            self.last_geom          = target_geom
            nose_kp = smooth_kpts[_C.KP_NOSE]
            self.last_nose_xy       = (float(nose_kp[0]), float(nose_kp[1]))

        self._stats.update(zone_label, dt_sec)
        return self._build_result(
            zone_label, state_sm, stability, valid, frame_idx, elapsed_sec,
            dist_px, dist_norm, gaze_fwd, gaze_lat, gaze_angle,
        )

    def _build_result(
        self,
        zone_label: str,
        state_sm: str,
        stability: float,
        valid: bool,
        frame_idx: int,
        elapsed_sec: float,
        dist_px:    float = float("nan"),
        dist_norm:  float = float("nan"),
        gaze_fwd:   float = float("nan"),
        gaze_lat:   float = float("nan"),
        gaze_angle: float = float("nan"),
    ) -> LickResult:
        trap_pts = self.last_trap_pts.tolist() if self.last_trap_pts is not None else []
        nose_xy  = list(self.last_nose_xy) if trap_pts else []

        def _zs(key: str) -> ZoneStats:
            hits, t = self._stats.zone_stats(key)
            return ZoneStats(hits=hits, time_sec=t)

        return LickResult(
            current_zone    = zone_label,
            best_zone       = self._stats.best_zone(),
            body            = _zs("BODY"),
            fl              = _zs("FL"),
            fr              = _zs("FR"),
            hl              = _zs("HL"),
            hr              = _zs("HR"),
            face_state      = state_sm,
            state_stability = stability,
            valid           = valid,
            frame           = frame_idx,
            time_sec        = elapsed_sec,
            dist_px         = dist_px,
            dist_norm       = dist_norm,
            gaze_fwd        = gaze_fwd,
            gaze_lat        = gaze_lat,
            gaze_angle      = gaze_angle,
            trap_pts        = trap_pts,
            nose_xy         = nose_xy,
        )
