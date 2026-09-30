"""
行為追蹤和統計
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime

from config import BehaviorTrackingConfig, LoggingConfig

# 真正的 datetime 類別：影片時鐘模式要用 fromtimestamp() 把影片時間換成日期時間
# （測試會把模組層級的 `datetime` 名稱換成假時鐘，這個別名不受影響）
_RealDatetime = datetime

# low_conf（貓在畫面但行為無法判定）的三種來源，由 frame_processor 傳入：
#   warmup    — 貓剛出現／剛啟動，關鍵點窗口還沒滿 SEQUENCE_LENGTH 幀，ST-GCN 還不能推論
#   uncertain — ST-GCN 有推論，但最高類別機率低於門檻（姿態可能清楚，只是行為不確定，
#               例如動作轉換、五類以外的行為）
#   sqa       — 骨架品質檢查（SQA）判定這個窗口的骨架幾何不可信（遮擋、關鍵點亂跳）
LOW_CONF_REASONS = ("warmup", "uncertain", "sqa")

# 四個 6 小時時段，跟 Node-RED「每日統計彙整」的 PERIOD_HOURS 一致
PERIOD_HOURS = {
    "00-06": ("00", "01", "02", "03", "04", "05"),
    "06-12": ("06", "07", "08", "09", "10", "11"),
    "12-18": ("12", "13", "14", "15", "16", "17"),
    "18-24": ("18", "19", "20", "21", "22", "23"),
}
_PERIOD_TIME_KEYS = ("monitoring_sec", "visible_sec", "not_detected_sec", "low_conf_sec")


def summarize_periods(hourly_distribution):
    """把每小時分布加總成四個時段；每時段含五類行為秒數、total（五類加總）與
    monitoring_sec（系統運行）/visible_sec（貓在畫面）/not_detected_sec/low_conf_sec。"""
    behaviors = list(BehaviorTrackingConfig.BEHAVIOR_CATEGORIES.values())
    periods = {}
    for name, hours in PERIOD_HOURS.items():
        acc = {k: 0.0 for k in behaviors + list(_PERIOD_TIME_KEYS)}
        for h in hours:
            bucket = hourly_distribution.get(h) or {}
            for k in acc:
                acc[k] += float(bucket.get(k, 0.0) or 0.0)
        acc = {k: round(v, 1) for k, v in acc.items()}
        acc["total"] = round(sum(acc[b] for b in behaviors), 1)
        periods[name] = acc
    return periods


class ImprovedBehaviorTracker:
    """累積每日行為時長/次數統計、行為轉移矩陣、活動力分數與警報判斷；
    狀態可持久化到磁碟，同一天內重啟程式會自動還原累積進度。"""

    def __init__(self):
        _behaviors = list(BehaviorTrackingConfig.BEHAVIOR_CATEGORIES.values())
        self._lock = threading.RLock()
        # 時鐘：None＝電腦時鐘（即時來源）；enable_media_clock() 換成影片時間（本機錄影）
        self._clock = None
        # 是否讀寫 tracker_state.json；影片時鐘模式關閉（錄影推論可重跑，不跟即時監測混）
        self._persist_state = True
        self.behavior_time = {b: 0.0 for b in _behaviors}
        self.behavior_count = {b: 0 for b in _behaviors}
        self.low_conf_time = (
            0.0  # YOLO 有偵測到貓但 ST-GCN 信心不足（獨立追蹤，不歸入任何行為）
        )
        self.low_conf_count = (
            0  # 低信心事件次數（連續低信心視為同一事件，離開低信心狀態才算 1 次）
        )
        self._in_low_conf = False  # 是否正處於連續低信心區間，用於事件計數的邊界判斷
        self.not_detected_time = 0.0  # YOLO 未偵測到貓（貓不在畫面中）
        self.behavior_history = deque(maxlen=BehaviorTrackingConfig.MAX_HISTORY_SIZE)
        self.current_behavior = None
        self.current_gcn_id = None  # 正在進行的行為對應的 GCN ID
        self.behavior_start_time = self._now_ts()
        self.current_event_start_time = (
            self._now_ts()
        )  # 真實事件開始時間，只在行為切換時重置
        self.last_update_time = (
            self._now_ts()
        )  # 用於計算逐幀時間差；同時也是「監測結束時間」
        # （最後一次收到有效幀的時刻）給 Dashboard 顯示用
        self.today_start_time = (
            self._now_ts()
        )  # 今日監測開始時間（第一次啟動或跨日重置的當下）；
        # 同一天內重啟程式會從 load_state() 還原，不會被重置
        self.last_reset = self._now_dt().date()
        self.activity_window = deque(maxlen=BehaviorTrackingConfig.ACTIVITY_WINDOW_SIZE)
        self.transition_matrix = {}  # {"walk->lick": 3, ...}
        # {"08": {"walk": 120.0, ..., "monitoring_sec": 系統運行, "visible_sec": 貓在畫面,
        #         "not_detected_sec": 不在畫面, "low_conf_sec": 無法判定}, ...}
        self.hourly_distribution = {}
        # 三個逐幀累加的計時器（同一個 dt，恆等式：run = 在畫面 + 不在畫面）：
        #   run_seconds        系統運行時間（有在處理畫面的時間；中斷見 MAX_FRAME_GAP_SECONDS）
        #   monitoring_seconds 貓在畫面時間（五類行為 + low_conf）——歷史欄位名稱沿用，
        #                      比例指標的分母、基線收錄門檻都用這個
        #   not_detected_time  貓不在畫面時間（見上方宣告）
        self.run_seconds = 0.0
        self.monitoring_seconds = 0.0
        self.low_conf_breakdown = {r: 0.0 for r in LOW_CONF_REASONS}
        self._last_valid_behavior = None
        self._last_save_time = 0.0
        self.behavior_min_duration = {b: None for b in _behaviors}
        self.behavior_max_duration = {b: None for b in _behaviors}
        self.load_state()

    # ── 時鐘 ────────────────────────────────────────────────────────────
    def _now_ts(self):
        """目前時間（epoch 秒）：預設電腦時鐘；影片時鐘模式＝錄影開始時間＋影片位置。"""
        return self._clock() if self._clock is not None else time.time()

    def _now_dt(self):
        """目前時間（datetime，本機時區），跟 _now_ts() 同一個時鐘。"""
        if self._clock is not None:
            return _RealDatetime.fromtimestamp(self._clock())
        return datetime.now()

    def enable_media_clock(self, clock_fn):
        """本機錄影改用影片時間：之後所有時長、每小時分桶、日期、中斷判定都照 clock_fn()
        （epoch 秒＝錄影開始時間＋影片播放到的位置）計算，跟處理快慢、哪天處理無關。

        從乾淨狀態開始：__init__ 的 load_state() 可能還原了今天的即時監測資料，那份
        留在 tracker_state.json 不動（本模式不讀寫存檔），記憶體裡的統計全部清空，
        免得錄影與即時監測混在同一天。錄影推論可以整段重跑，所以不需要續算。"""
        with self._lock:
            self._clock = clock_fn
            self._persist_state = False
            now = self._now_ts()
            self._clear_counters_locked()
            self.behavior_history.clear()
            self.activity_window.clear()
            self.current_behavior = None
            self.current_gcn_id = None
            self.behavior_start_time = now
            self.current_event_start_time = now
            self.last_update_time = now
            self.today_start_time = now
            self.last_reset = self._now_dt().date()

    def finish_media_day(self):
        """本機錄影播完：結算進行中的事件，把這一天寫進多天歷史（daily_history.db）。

        影片錄到 24:00 就結束，之後沒有下一幀能觸發跨日，所以播完當下就寫。統計不
        歸零（Dashboard 繼續顯示最後結果；這個行程要處理下一支影片得重開）。
        回傳寫入的日期；沒有任何時間可寫或寫入失敗回傳 None。"""
        with self._lock:
            self._settle_current_behavior(self.last_update_time, 0.0)
            day = self.last_reset
            snapshot = self._snapshot_locked()
        return day if self._persist_daily_record(day, snapshot) else None

    def save_state(self):
        """將當日累積統計寫入 LoggingConfig.TRACKER_STATE_PATH，供重啟後還原。
        影片時鐘模式不寫（見 enable_media_clock）。"""
        if not self._persist_state:
            return
        try:
            with self._lock:
                state = {
                    "date": str(self.last_reset),
                    "behavior_time": dict(self.behavior_time),
                    "behavior_count": dict(self.behavior_count),
                    "low_conf_time": self.low_conf_time,
                    "low_conf_count": self.low_conf_count,
                    "not_detected_time": self.not_detected_time,
                    "transition_matrix": dict(self.transition_matrix),
                    "hourly_distribution": {
                        h: dict(v) for h, v in self.hourly_distribution.items()
                    },
                    "monitoring_seconds": self.monitoring_seconds,
                    "run_seconds": self.run_seconds,
                    "low_conf_breakdown": dict(self.low_conf_breakdown),
                    "_last_valid_behavior": self._last_valid_behavior,
                    "behavior_min_duration": dict(self.behavior_min_duration),
                    "behavior_max_duration": dict(self.behavior_max_duration),
                    "today_start_time": self.today_start_time,
                }
            path = LoggingConfig.TRACKER_STATE_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # 原子寫入：先寫到同目錄下的暫存檔，成功寫完+flush 後再用
            # os.replace() 原子取代正式檔案。避免寫到一半被中斷（例如
            # Ctrl+C 觸發 main.py 的 os._exit()）時，正式檔案被截斷成
            # 損毀的 JSON，導致 load_state() 讀取失敗、整天累積的統計遺失
            # （os.replace 在同一個檔案系統內是單一原子系統呼叫，不會有
            # 「寫一半」的中間狀態）。
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            logging.warning("TrackerState save failed: %s", e)

    def load_state(self):
        """還原當日累積統計；若存檔日期非今天則不還原（避免跨日資料污染），
        但會先把那一天補寫進多天歷史——程式在午夜前就關掉時，check_daily_reset()
        沒機會觸發，這是那一天唯一能進 daily_history.db 的時機。"""
        try:
            path = LoggingConfig.TRACKER_STATE_PATH
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("date") != str(self._now_dt().date()):
                if self._persist_stale_state(state):
                    # 補寫完就把舊存檔改名封存，之後不會再被補寫第二次。不改名的話，
                    # 影片時鐘模式（不寫存檔）下這份舊檔會一直留著，每次啟動都重新
                    # 補寫一次，把同一天後來寫進 DB 的錄影紀錄蓋回舊資料（upsert）。
                    archived = f"{path}.persisted-{state.get('date')}"
                    os.replace(path, archived)
                    logging.info("TrackerState: 舊存檔已封存為 %s", archived)
                return  # 不同天的存檔，不還原
            with self._lock:
                self.behavior_time = {
                    k: float(v) for k, v in state.get("behavior_time", {}).items()
                }
                self.behavior_count = {
                    k: int(v) for k, v in state.get("behavior_count", {}).items()
                }
                self.low_conf_time = float(state.get("low_conf_time", 0.0))
                self.low_conf_count = int(state.get("low_conf_count", 0))
                self.not_detected_time = float(state.get("not_detected_time", 0.0))
                self.transition_matrix = {
                    k: int(v) for k, v in state.get("transition_matrix", {}).items()
                }
                self.hourly_distribution = {
                    h: {b: float(t) for b, t in v.items()}
                    for h, v in state.get("hourly_distribution", {}).items()
                }
                self.monitoring_seconds = float(state.get("monitoring_seconds", 0.0))
                self.run_seconds = float(state.get("run_seconds", 0.0))
                for r, v in (state.get("low_conf_breakdown") or {}).items():
                    if r in self.low_conf_breakdown:
                        self.low_conf_breakdown[r] = float(v)
                self._last_valid_behavior = state.get("_last_valid_behavior", None)
                for b in self.behavior_min_duration:
                    v = state.get("behavior_min_duration", {}).get(b)
                    if v is not None:
                        self.behavior_min_duration[b] = float(v)
                for b in self.behavior_max_duration:
                    v = state.get("behavior_max_duration", {}).get(b)
                    if v is not None:
                        self.behavior_max_duration[b] = float(v)
                # 還原「今日監測開始時間」，讓同一天內重啟程式不會把開始時間
                # 誤植成重啟當下——沒有這個欄位的舊存檔（升級前）就維持
                # __init__ 預設的目前時間，只影響升級當下那一次顯示。
                saved_start = state.get("today_start_time")
                if saved_start is not None:
                    self.today_start_time = float(saved_start)
            logging.info("TrackerState restored from %s", path)
        except Exception as e:
            logging.warning("TrackerState load failed: %s", e)

    def _persist_daily_record(self, day, snapshot):
        """把 `_reset_if_new_day_locked()` 傳入的「歸零前」當日累積統計快照，
        彙整成一筆 DailyRecord 寫入 Python 端獨立的多天歷史（analytics/daily_store.py）。

        這份持久化跟 Node-RED 自己的 v2_daily_history 完全獨立——不讀、不寫
        Node-RED 的 global.json，兩邊各自累積、互不相干，任一邊資料被寫壞
        都不會波及另一邊（見 docs/資料層架構現況與統一管理評估.md 第九節）。

        2026-08-11 修正：改吃呼叫端傳入的 `snapshot`，不再直接讀
        `self.behavior_time`/`self.behavior_count`/`self.monitoring_seconds`。
        這支方法現在特意在 `self._lock` 之外呼叫（見 `check_daily_reset()`），
        呼叫當下這些欄位可能已經被歸零、甚至已經被其他執行緒的 `update()`
        呼叫進一步累積新一天的資料，讀 `self.*` 會拿到錯誤的資料，一定要
        用重置當下鎖內拍下的快照。失敗時只記警告、不拋出，避免持久化問題
        影響即時監測本身（跟 save_state() 的容錯原則一致）。

        2026-09-30：那天完全沒有累積任何時間（系統沒運行，例如只開著 Flask、管線整天
        暫停就跨日）不寫——跟 Node-RED「有運行過就封存」同一條規則，兩本日記才一致；
        要不要進基線由 baseline.compute_baseline() 的貓在畫面門檻決定。
        回傳是否真的寫入。
        """
        if (
            snapshot.get("run_seconds", 0.0) + snapshot.get("monitoring_seconds", 0.0)
            + snapshot.get("not_detected_time", 0.0) + sum(snapshot["behavior_time"].values())
        ) <= 0:
            return False
        try:
            from analytics import daily_store
            from analytics.baseline import DailyRecord

            behavior_time = snapshot["behavior_time"]
            behavior_count = snapshot["behavior_count"]
            total_active = (
                behavior_time["walk"]
                + behavior_time["scratch"]
                + behavior_time["lick"]
                + behavior_time["shake"]
                + behavior_time["stop"]
            )
            breakdown = snapshot.get("low_conf_breakdown") or {}
            record = DailyRecord(
                day=day,
                monitoring_seconds=snapshot["monitoring_seconds"],
                run_seconds=snapshot.get("run_seconds", 0.0),
                not_detected_time=snapshot.get("not_detected_time", 0.0),
                low_conf_time=snapshot.get("low_conf_time", 0.0),
                low_conf_warmup_time=breakdown.get("warmup", 0.0),
                low_conf_uncertain_time=breakdown.get("uncertain", 0.0),
                low_conf_sqa_time=breakdown.get("sqa", 0.0),
                periods=summarize_periods(snapshot.get("hourly_distribution") or {}),
                walk_time=behavior_time["walk"],
                walk_count=behavior_count["walk"],
                stop_time=behavior_time["stop"],
                stop_count=behavior_count["stop"],
                lick_time=behavior_time["lick"],
                lick_count=behavior_count["lick"],
                scratch_time=behavior_time["scratch"],
                scratch_count=behavior_count["scratch"],
                shake_count=behavior_count["shake"],
                active_time=total_active,
                # rest_time 目前不影響 analytics 的評分模型（不在
                # baseline.CONTINUOUS_METRICS/COUNT_METRICS 內），沿用
                # stop_time 只是保留語意對照，不代表額外計算。
                rest_time=behavior_time["stop"],
            )
            daily_store.save_day(record)
            return True
        except Exception as e:
            logging.warning("Daily history persist failed (day=%s): %s", day, e)
            return False

    def _snapshot_locked(self):
        """呼叫端必須已持有 self._lock。拍下寫入多天歷史所需的當日累積統計。"""
        return {
            "behavior_time": dict(self.behavior_time),
            "behavior_count": dict(self.behavior_count),
            "monitoring_seconds": self.monitoring_seconds,
            "run_seconds": self.run_seconds,
            "not_detected_time": self.not_detected_time,
            "low_conf_time": self.low_conf_time,
            "low_conf_breakdown": dict(self.low_conf_breakdown),
            "hourly_distribution": {
                h: dict(v) for h, v in self.hourly_distribution.items()
            },
        }

    def _persist_stale_state(self, state):
        """把「不是今天」的 tracker_state 存檔補寫成那一天的多天歷史紀錄。

        程式在午夜前關掉（例如只錄 18:00~23:xx）時 check_daily_reset() 沒機會觸發，
        下次啟動 load_state() 讀到舊日期的存檔，原本直接丟掉 → 那一天永遠不會進
        daily_history.db。save_day 是 upsert，同一天重複補寫只會覆蓋成同樣的數字。
        完全沒有累積任何時間的存檔（剛啟動就關）不寫，避免產生空白天。"""
        try:
            from datetime import date as _date

            day = _date.fromisoformat(str(state.get("date")))
        except (TypeError, ValueError):
            return
        behaviors = list(BehaviorTrackingConfig.BEHAVIOR_CATEGORIES.values())
        snapshot = {
            "behavior_time": {
                b: float((state.get("behavior_time") or {}).get(b, 0.0)) for b in behaviors
            },
            "behavior_count": {
                b: int((state.get("behavior_count") or {}).get(b, 0)) for b in behaviors
            },
            "monitoring_seconds": float(state.get("monitoring_seconds", 0.0)),
            "run_seconds": float(state.get("run_seconds", 0.0)),
            "not_detected_time": float(state.get("not_detected_time", 0.0)),
            "low_conf_time": float(state.get("low_conf_time", 0.0)),
            "low_conf_breakdown": dict(state.get("low_conf_breakdown") or {}),
            "hourly_distribution": dict(state.get("hourly_distribution") or {}),
        }
        if self._persist_daily_record(day, snapshot):
            logging.info("TrackerState: 前次存檔日期 %s 已補寫進多天歷史", day)
            return True
        return False

    def _reset_if_new_day_locked(self):
        """呼叫端必須已持有 self._lock。若日期已跨天，重置所有當日累積統計
        （純記憶體操作，不做任何 I/O），並回傳 `(prev_date, snapshot)` 供
        呼叫端在鎖外呼叫 `_persist_daily_record()`；沒跨天則回傳
        `(None, None)`。

        2026-08-11 從 check_daily_reset() 拆出來：原本 SQLite 寫入
        （_persist_daily_record）是在這裡、也就是在 self._lock 內執行的，
        因為 `update()`（即時推論熱路徑）跟 `get_today_stats()`（HTTP 請求）
        都會在持鎖狀態下呼叫 check_daily_reset()，一旦跨日那一刻剛好撞上
        SQLite 連線競爭（例如 analytics/manage_baseline_history.py 同時在
        寫同一個 db），`daily_store._connect()` 的 `timeout=10.0` 可以讓
        整條即時影像管線被卡住長達 10 秒。拆開後，這裡只做快速的記憶體
        重置，真正可能慢的 I/O 交給呼叫端在鎖外處理。
        """
        today = self._now_dt().date()
        if today == self.last_reset:
            return None, None
        prev_date = self.last_reset
        # 先更新 last_reset 防止兩個執行緒同時通過 != 判斷造成雙重 reset
        self.last_reset = today
        snapshot = self._snapshot_locked()
        self._clear_counters_locked()
        self.today_start_time = self._now_ts()  # 跨日重置：新的一天，重新起算監測開始時間
        return prev_date, snapshot

    def _clear_counters_locked(self):
        """呼叫端必須已持有 self._lock。把當日累積統計全部歸零（跨日重置與
        enable_media_clock 共用）。"""
        self.behavior_time = {k: 0.0 for k in self.behavior_time}
        self.behavior_count = {k: 0 for k in self.behavior_count}
        self.low_conf_time = 0.0
        self.low_conf_count = 0
        self._in_low_conf = False
        self.not_detected_time = 0.0
        self.transition_matrix = {}
        self.hourly_distribution = {}
        self.monitoring_seconds = 0.0
        self.run_seconds = 0.0
        self.low_conf_breakdown = {r: 0.0 for r in LOW_CONF_REASONS}
        self._last_valid_behavior = None
        self.behavior_min_duration = {k: None for k in self.behavior_min_duration}
        self.behavior_max_duration = {k: None for k in self.behavior_max_duration}

    def check_daily_reset(self):
        """公開介面：若日期已跨天，重置所有當日累積統計並把前一天的資料
        持久化到 Python 端獨立的多天歷史。

        2026-08-11 修正：呼叫端（`update()`/`get_today_stats()`）不能已經
        持有 `self._lock` 才呼叫這支方法——記憶體重置在鎖內做（快），
        `_persist_daily_record()` 的 SQLite 寫入刻意留到鎖外做（可能慢），
        避免即時管線被卡住，見 `_reset_if_new_day_locked()` 說明。

        2026-08-11 稍晚再修正（code review 發現的效能迴歸）：`update()`
        是每一幀都會呼叫的即時熱路徑，這支方法被獨立出來後，等於每幀都
        多付一次鎖的 acquire/release（跟原本「已經在 `update()` 的鎖裡」
        相比變成兩次）。這裡加一個無鎖的快速路徑：先不加鎖比較一次日期，
        沒跨天（絕大多數幀）直接返回，完全不碰鎖；只有日期真的不一樣時
        才進鎖裡用 `_reset_if_new_day_locked()` 二次確認並執行重置——這是
        標準的 double-checked locking，這裡的無鎖讀取只是「要不要進鎖」的
        提示，正確性仍然由鎖內的二次確認保證，不會因為讀到瞬間髒值而重複
        重置或漏掉重置（CPython 的 GIL 保證 `date` 物件屬性讀取不會撕裂）。
        """
        if self._now_dt().date() == self.last_reset:
            return
        with self._lock:
            prev_date, snapshot = self._reset_if_new_day_locked()
        if prev_date is not None:
            self._persist_daily_record(prev_date, snapshot)

    def map_gcn_to_tracker(self, behavior_id):
        """把 ST-GCN 的行為 ID 轉成統計用的行為名稱字串；未知 ID 預設為 'walk'。"""
        mapping = BehaviorTrackingConfig.BEHAVIOR_CATEGORIES
        return mapping.get(behavior_id, "walk")

    def _settle_current_behavior(self, now, next_activity_value=0.0):
        """結算目前正在進行的行為事件到 now 這一刻，並清空 current_behavior。

        無論是「切換到新行為」還是「貓消失/信心不足導致事件被迫中斷」都要呼叫
        這支方法，確保空窗（not_detected/low_conf）期間的時間只會算進
        not_detected_time/low_conf_time，不會被歸到消失前或恢復後的行為時長裡
        （修正：先前貓咪消失/低信心時 behavior_start_time 不會推進，等貓咪重新
        出現時那整段空窗時間會被灌進 behavior_time，即使剛好是同一個行為）。

        呼叫端必須已持有 self._lock。回傳是否真的結算了一個事件（current_behavior
        原本不是 None），供呼叫端決定要不要立即 save_state()。
        """
        if self.current_behavior is None:
            return False
        duration = now - self.behavior_start_time
        total_event_duration = now - self.current_event_start_time
        if self.current_behavior in self.behavior_time:
            self.behavior_time[self.current_behavior] += max(
                0.0, duration
            )  # 補上最後一段
        if self.current_behavior in self.behavior_count:
            self.behavior_count[self.current_behavior] += 1  # 每個完整事件只算 1 次
        beh = self.current_behavior
        dur_r = round(total_event_duration, 1)  # 完整事件時長
        if beh in self.behavior_min_duration:
            if (
                self.behavior_min_duration[beh] is None
                or dur_r < self.behavior_min_duration[beh]
            ):
                self.behavior_min_duration[beh] = dur_r
            if (
                self.behavior_max_duration[beh] is None
                or dur_r > self.behavior_max_duration[beh]
            ):
                self.behavior_max_duration[beh] = dur_r
        self.behavior_history.append(
            {
                "behavior": self.current_behavior,
                "gcn_behavior_id": self.current_gcn_id,
                "timestamp": (  # 事件結束時刻（中斷時是上一幀，不是現在）
                    _RealDatetime.fromtimestamp(now) if self._clock is not None else datetime.now()
                ),
                "duration": dur_r,
                "activity": next_activity_value,
            }
        )
        self.current_behavior = None
        self.current_gcn_id = None
        return True

    def update(self, behavior_id, activity_value, low_conf_reason=None):
        """以本幀的行為 ID 與活動力數值更新累積統計、轉移矩陣與活動視窗。

        low_conf_reason：behavior_id == -1（貓在畫面但行為無法判定）時的來源，
        LOW_CONF_REASONS 其中之一；沒給或不認得的值歸到 "uncertain"。"""
        # check_daily_reset() 刻意放在下面的主鎖之外呼叫：它內部的
        # _persist_daily_record()（可能慢的 SQLite 寫入）本來就設計成鎖外
        # 執行，如果在這裡先進了 `with self._lock:` 才呼叫，self._lock 是
        # RLock、重入不會擋住同一執行緒，但也不會釋放鎖給其他執行緒，等於
        # 白白拆了還是卡住即時管線——見 check_daily_reset() 說明。
        self.check_daily_reset()
        _event_completed = False
        with self._lock:
            now = self._now_ts()
            prev_time = self.last_update_time
            dt = now - prev_time
            self.last_update_time = now

            if dt < 0 or dt > BehaviorTrackingConfig.MAX_FRAME_GAP_SECONDS:
                # 中斷（排程暫停、串流卡住、時鐘倒退…）：這段空白不算系統運行，也不算
                # 任何狀態；進行中的行為事件在上一幀就結束，不把空白灌進它的時長。
                if self._settle_current_behavior(prev_time, 0.0):
                    _event_completed = True
                self._in_low_conf = False
                dt = 0.0

            # 系統運行時間：所有狀態都累積（每小時的 monitoring_sec 同義，用於判斷時段是否有被監控）
            self.run_seconds += dt
            hour_key = self._now_dt().strftime("%H")
            if hour_key not in self.hourly_distribution:
                self.hourly_distribution[hour_key] = {
                    b: 0.0 for b in BehaviorTrackingConfig.BEHAVIOR_CATEGORIES.values()
                }
            bucket = self.hourly_distribution[hour_key]
            bucket["monitoring_sec"] = bucket.get("monitoring_sec", 0.0) + dt
            if behavior_id == -2:
                bucket["not_detected_sec"] = bucket.get("not_detected_sec", 0.0) + dt
            else:
                # 貓在畫面時間（五類行為 + low_conf）
                self.monitoring_seconds += dt
                bucket["visible_sec"] = bucket.get("visible_sec", 0.0) + dt

            if behavior_id == -2:
                # YOLO 未偵測到貓（behavior_id == -2）：累積到 not_detected_time，
                # 並結算正在進行的行為事件，避免消失期間的時間灌進行為時長
                self.not_detected_time += dt
                self._in_low_conf = (
                    False  # 貓消失視為離開低信心區間，下次進入低信心重新計 1 次事件
                )
                if self._settle_current_behavior(now, 0.0):
                    _event_completed = True
            elif behavior_id == -1:
                # 信心不足時（behavior_id == -1）：貓在畫面，但行為無法判定（來源見
                # LOW_CONF_REASONS）。獨立累積到 low_conf_time/low_conf_count，不歸入任何
                # 行為統計（也不算「休息」——「休息」由 stop 行為本身的次數/時長全權代表）
                self.low_conf_time += dt
                reason = low_conf_reason if low_conf_reason in LOW_CONF_REASONS else "uncertain"
                self.low_conf_breakdown[reason] += dt
                bucket["low_conf_sec"] = bucket.get("low_conf_sec", 0.0) + dt
                if not self._in_low_conf:
                    self.low_conf_count += 1
                    self._in_low_conf = True
                if self._settle_current_behavior(now, activity_value):
                    _event_completed = True
                self.activity_window.append(
                    {"time": now, "activity": activity_value, "weight": 1.0}
                )
            else:
                # 有效行為（0~4）：累積到對應 behavior_time
                self._in_low_conf = False  # 離開低信心區間
                behavior = self.map_gcn_to_tracker(behavior_id)

                # 每小時分布（dt 為幀間隔，即時累積）
                self.hourly_distribution[hour_key][behavior] = (
                    self.hourly_distribution[hour_key].get(behavior, 0.0) + dt
                )

                # 轉移矩陣（只在行為切換時記錄）
                if self._last_valid_behavior and self._last_valid_behavior != behavior:
                    key = self._last_valid_behavior + "->" + behavior
                    self.transition_matrix[key] = self.transition_matrix.get(key, 0) + 1
                self._last_valid_behavior = behavior

                duration = now - self.behavior_start_time  # 距上次累積時間點的間隔

                if behavior != self.current_behavior:
                    # ── 行為切換（或貓消失/低信心後恢復）：結算前一個事件，開始新事件 ──
                    self._settle_current_behavior(now, activity_value)
                    self.current_behavior = behavior
                    self.current_gcn_id = behavior_id
                    self.behavior_start_time = now
                    self.current_event_start_time = now  # 真實事件起點重置
                    _event_completed = True
                elif (
                    self.current_behavior is not None
                    and duration >= BehaviorTrackingConfig.MIN_RECORD_DURATION_SECONDS
                ):
                    # ── 相同行為的定期累積：只更新時間，不建立新事件 ──
                    if self.current_behavior in self.behavior_time:
                        self.behavior_time[self.current_behavior] += duration
                    self.behavior_start_time = (
                        now  # 重置部分計時器；current_event_start_time 保持不動
                    )
                # 均勻權重：每幀貢獻相等，使 get_activity_score() 為純粹的時間視窗平均
                self.activity_window.append(
                    {"time": now, "activity": activity_value, "weight": 1.0}
                )
        now_t = time.time()
        if _event_completed or now_t - self._last_save_time >= 30.0:
            self._last_save_time = now_t
            self.save_state()

    def get_activity_score(self):
        """回傳近期活動視窗（ACTIVITY_SCORE_WINDOW_SECONDS 秒內）的平均活動分數（0-100）。"""
        with self._lock:
            if len(self.activity_window) == 0:
                return 0
            now = self._now_ts()
            recent = [
                r
                for r in self.activity_window
                if (now - r["time"])
                < BehaviorTrackingConfig.ACTIVITY_SCORE_WINDOW_SECONDS
            ]
            if len(recent) == 0:
                return 0  # 視窗內無資料 = 貓不在畫面或無運動
            n = len(recent)
            score = round(sum(r["activity"] for r in recent) / n)
            return max(0, min(100, score))

    def get_today_stats(self):
        """組出供 Dashboard 顯示的當日完整統計字典（各行為時長/次數、活動時長等）。"""
        # 同 update()：check_daily_reset() 要在主鎖之外呼叫，避免它內部可能
        # 慢的 SQLite 寫入卡住這支方法（HTTP 請求路徑）的呼叫端。
        self.check_daily_reset()
        with self._lock:
            total_active = (
                self.behavior_time["walk"]
                + self.behavior_time["scratch"]
                + self.behavior_time["lick"]
                + self.behavior_time["shake"]
                + self.behavior_time["stop"]
            )
            stats = {
                "walk": self.behavior_count["walk"],
                "walk_time": round(self.behavior_time["walk"], 1),
                "scratch": self.behavior_count["scratch"],
                "scratch_time": round(self.behavior_time["scratch"], 1),
                "lick": self.behavior_count["lick"],
                "lick_time": round(self.behavior_time["lick"], 1),
                "shake": self.behavior_count["shake"],
                "shake_time": round(self.behavior_time["shake"], 1),
                "stop": self.behavior_count["stop"],
                "stop_time": round(self.behavior_time["stop"], 1),
                "active_time": round(total_active, 1),
                "low_conf": self.low_conf_count,
                "low_conf_time": round(self.low_conf_time, 1),
                "low_conf_breakdown": {
                    r: round(v, 1) for r, v in self.low_conf_breakdown.items()
                },
                "not_detected_time": round(self.not_detected_time, 1),
                # 分母定義（逐幀累加，run_seconds = monitoring_seconds + not_detected_time）：
                # monitoring_seconds = 貓在畫面時間（比例指標分母、基線收錄門檻）
                # run_seconds        = 系統運行時間
                "monitoring_seconds": round(self.monitoring_seconds, 1),
                "run_seconds": round(self.run_seconds, 1),
                # 供 Dashboard 直接顯示，不在前端 JS 重算（跟上面兩個同值，保留舊鍵名）：
                # cat_visible_time = 貓在畫面時長（五類行為 + low_conf）
                # total_uptime     = 系統運行時長（含貓不在畫面的時間）
                "cat_visible_time": round(self.monitoring_seconds, 1),
                "total_uptime": round(self.run_seconds, 1),
                # 統計所屬日期（ISO），Node-RED 用它判斷跨日，不再靠收到資料的當下時間
                "date": self.last_reset.isoformat(),
                "transition_matrix": dict(self.transition_matrix),
                "hourly_distribution": {
                    h: dict(v) for h, v in self.hourly_distribution.items()
                },
                "behavior_min_duration": dict(self.behavior_min_duration),
                "behavior_max_duration": dict(self.behavior_max_duration),
                # 供 Dashboard 顯示「監測開始時間 ~ 監測結束時間」：Unix 秒數
                # （前端 new Date(ts*1000) 換算），不在這裡格式化成字串，避免
                # 綁死時區/格式，前端可自行決定顯示方式。
                # today_start_time = 今日第一次開始監測的時刻（同一天內重啟
                #   程式不會改變，見 load_state()）。
                # last_update_time = 最後一次收到有效幀、實際更新統計的時刻
                #   ——系統持續運行時等同「現在」；管線暫停/影片播畢/排程
                #   結束後就停在最後一次更新的時間點，視覺上代表監測結束。
                "today_start_time": self.today_start_time,
                "last_update_time": self.last_update_time,
            }
        return stats

    def get_alerts(self):
        """依累積統計比對各項警戒門檻，回傳需要提醒使用者的警報清單。"""
        with self._lock:
            scratch_time = self.behavior_time.get("scratch", 0)
            scratch_count = self.behavior_count.get("scratch", 0)
            lick_time = self.behavior_time.get("lick", 0)
            lick_count = self.behavior_count.get("lick", 0)
            shake_time = self.behavior_time.get("shake", 0)
            shake_count = self.behavior_count.get("shake", 0)
            walk_time = self.behavior_time.get("walk", 0)
            stop_time = self.behavior_time.get("stop", 0)
            stop_count = self.behavior_count.get("stop", 0)

        alerts = []
        if scratch_time > BehaviorTrackingConfig.SCRATCH_ALERT_TIME_SECONDS:
            alerts.append(
                {
                    "level": "high",
                    "icon": "🚨",
                    "title": "搔抓時間異常",
                    "message": f"今日累積搔抓 {scratch_time:.1f} 秒（{scratch_count}次）",
                    "suggestion": "請檢查皮膚是否有紅腫、掉毛、傷口",
                    "action": "聯絡獸醫",
                }
            )
        elif scratch_count >= BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD:
            alerts.append(
                {
                    "level": "medium",
                    "icon": "⚠️",
                    "title": "搔抓頻率偏高",
                    "message": f"今日已搔抓 {scratch_count} 次（累積{scratch_time:.1f}秒）",
                    "suggestion": "建議觀察是否有皮膚不適症狀",
                    "action": "持續觀察",
                }
            )
        if lick_time > BehaviorTrackingConfig.LICK_ALERT_TIME_SECONDS:
            alerts.append(
                {
                    "level": "medium",
                    "icon": "🧼",
                    "title": "舔舐時間較長",
                    "message": f"今日舔舐 {lick_time:.1f} 秒（{lick_count}次）",
                    "suggestion": "可能有壓力或皮膚問題",
                    "action": "觀察精神狀態",
                }
            )
        if shake_count >= BehaviorTrackingConfig.SHAKE_ALERT_COUNT_THRESHOLD:
            alerts.append(
                {
                    "level": "medium",
                    "icon": "🔄",
                    "title": "甩頭動作頻繁",
                    "message": f"今日甩頭 {shake_count} 次（累積{shake_time:.1f}秒）",
                    "action": "檢查耳朵",
                }
            )
        if stop_time > BehaviorTrackingConfig.STOP_ALERT_TIME_SECONDS:
            alerts.append(
                {
                    "level": "medium",
                    "icon": "⏹",
                    "title": "長時間靜止不動",
                    "message": f"今日累積靜止 {stop_time:.1f} 秒（{stop_count}次）",
                    "suggestion": "貓咪長時間靜止，可能有身體不適",
                    "action": "觀察精神與食慾",
                }
            )
        total_time = lick_time + scratch_time + walk_time + shake_time + stop_time
        if (
            total_time > 0
            and walk_time < BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS
        ):
            alerts.append(
                {
                    "level": "medium",
                    "icon": "😴",
                    "title": "活動度過低",
                    "message": f"今日走動時間僅 {walk_time:.1f} 秒（低於門檻 {BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS:.0f} 秒）",
                    "suggestion": "貓咪活動不足，可能有身體不適",
                    "action": "嘗試互動或遊玩",
                }
            )
        return alerts
