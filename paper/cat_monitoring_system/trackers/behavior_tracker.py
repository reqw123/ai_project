"""
行為追蹤和統計
"""
from collections import deque
from datetime import datetime
import json
import logging
import os
import threading
import time

from config import BehaviorTrackingConfig, LoggingConfig

class ImprovedBehaviorTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self.behavior_time = {"walk": 0.0, "scratch": 0.0, "lick": 0.0, "shake": 0.0, "stop": 0.0}
        self.behavior_count = {"walk": 0, "scratch": 0, "lick": 0, "shake": 0, "stop": 0}
        self.rest_time = 0.0          # YOLO 有偵測到貓但 ST-GCN 信心不足
        self.not_detected_time = 0.0  # YOLO 未偵測到貓（貓不在畫面中）
        self.behavior_history = deque(maxlen=BehaviorTrackingConfig.MAX_HISTORY_SIZE)
        self.current_behavior = None
        self.current_gcn_id = None  # 正在進行的行為對應的 GCN ID
        self.behavior_start_time = time.time()
        self.last_update_time = time.time()  # 用於計算逐幀時間差
        self.last_reset = datetime.now().date()
        self.activity_window = deque(maxlen=BehaviorTrackingConfig.ACTIVITY_WINDOW_SIZE)
        self.alerts = deque(maxlen=BehaviorTrackingConfig.MAX_ALERTS_SIZE)
        self.transition_matrix = {}   # {"walk->lick": 3, ...}
        self.hourly_distribution = {} # {"08": {"walk": 120.0, ...}, ...}
        self.monitoring_seconds = 0.0
        self._last_valid_behavior = None
        self._last_save_time = 0.0
        self.load_state()
    def save_state(self):
        try:
            with self._lock:
                state = {
                    "date":               str(self.last_reset),
                    "behavior_time":      dict(self.behavior_time),
                    "behavior_count":     dict(self.behavior_count),
                    "rest_time":          self.rest_time,
                    "not_detected_time":  self.not_detected_time,
                    "transition_matrix":   dict(self.transition_matrix),
                    "hourly_distribution": {h: dict(v) for h, v in self.hourly_distribution.items()},
                    "monitoring_seconds":  self.monitoring_seconds,
                    "_last_valid_behavior": self._last_valid_behavior,
                }
            path = LoggingConfig.TRACKER_STATE_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception as e:
            logging.warning("TrackerState save failed: %s", e)

    def load_state(self):
        try:
            path = LoggingConfig.TRACKER_STATE_PATH
            if not os.path.exists(path):
                return
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("date") != str(datetime.now().date()):
                return  # 不同天的存檔，不還原
            with self._lock:
                self.behavior_time     = {k: float(v) for k, v in state.get("behavior_time",  {}).items()}
                self.behavior_count    = {k: int(v)   for k, v in state.get("behavior_count", {}).items()}
                self.rest_time         = float(state.get("rest_time",         0.0))
                self.not_detected_time = float(state.get("not_detected_time", 0.0))
                self.transition_matrix   = {k: int(v) for k, v in state.get("transition_matrix", {}).items()}
                self.hourly_distribution = {h: {b: float(t) for b, t in v.items()}
                                            for h, v in state.get("hourly_distribution", {}).items()}
                self.monitoring_seconds  = float(state.get("monitoring_seconds", 0.0))
                self._last_valid_behavior = state.get("_last_valid_behavior", None)
            logging.info("TrackerState restored from %s", path)
        except Exception as e:
            logging.warning("TrackerState load failed: %s", e)

    def check_daily_reset(self):
        today = datetime.now().date()
        if today != self.last_reset:
            # 先更新 last_reset 防止兩個執行緒同時通過 != 判斷造成雙重 reset
            self.last_reset = today
            self.behavior_time = {k: 0.0 for k in self.behavior_time}
            self.behavior_count = {k: 0 for k in self.behavior_count}
            self.rest_time = 0.0
            self.not_detected_time = 0.0
            self.transition_matrix = {}
            self.hourly_distribution = {}
            self.monitoring_seconds = 0.0
            self._last_valid_behavior = None
    def map_gcn_to_tracker(self, behavior_id):
        mapping = BehaviorTrackingConfig.BEHAVIOR_CATEGORIES
        return mapping.get(behavior_id, "walk")
    def update(self, behavior_id, activity_value):
        with self._lock:
            self.check_daily_reset()
            now = time.time()
            dt = now - self.last_update_time
            self.last_update_time = now

            # 每小時監控時間（所有狀態都累積，用於判斷時段是否有被監控）
            hour_key = datetime.now().strftime("%H")
            if hour_key not in self.hourly_distribution:
                self.hourly_distribution[hour_key] = {b: 0.0 for b in ["walk", "lick", "scratch", "shake", "stop"]}
            self.hourly_distribution[hour_key]["monitoring_sec"] = (
                self.hourly_distribution[hour_key].get("monitoring_sec", 0.0) + dt
            )

            # YOLO 未偵測到貓（behavior_id == -2）：累積到 not_detected_time，不算休息
            if behavior_id == -2:
                self.not_detected_time += dt
                return

            # 信心不足時（behavior_id == -1）：YOLO 有偵測到但 ST-GCN 信心未達門檻，累積到 rest_time
            if behavior_id == -1:
                self.rest_time += dt
                self.activity_window.append({"time": now, "activity": activity_value, "weight": 1.0})
                return

            # 有效行為（0~4）：累積到對應 behavior_time
            behavior = self.map_gcn_to_tracker(behavior_id)

            # 每小時分布（dt 為幀間隔，即時累積）
            hour_key = datetime.now().strftime("%H")
            if hour_key not in self.hourly_distribution:
                self.hourly_distribution[hour_key] = {b: 0.0 for b in ["walk", "lick", "scratch", "shake", "stop"]}
            self.hourly_distribution[hour_key][behavior] = self.hourly_distribution[hour_key].get(behavior, 0.0) + dt

            # 轉移矩陣（只在行為切換時記錄）
            if self._last_valid_behavior and self._last_valid_behavior != behavior:
                key = self._last_valid_behavior + "->" + behavior
                self.transition_matrix[key] = self.transition_matrix.get(key, 0) + 1
            self._last_valid_behavior = behavior

            duration = now - self.behavior_start_time
            record_this = False
            if behavior != self.current_behavior:
                record_this = True
            elif self.current_behavior is not None and duration >= BehaviorTrackingConfig.MIN_RECORD_DURATION_SECONDS:
                record_this = True
            if record_this and self.current_behavior is not None:
                if self.current_behavior in self.behavior_time:
                    self.behavior_time[self.current_behavior] += duration
                if self.current_behavior in self.behavior_count:
                    self.behavior_count[self.current_behavior] += 1
                record = {
                    "behavior": self.current_behavior,
                    "gcn_behavior_id": self.current_gcn_id,  # 記錄剛結束的舊行為 ID
                    "timestamp": datetime.now(),
                    "duration": round(duration, 1),
                    "activity": activity_value
                }
                self.behavior_history.append(record)
                self.current_behavior = behavior
                self.current_gcn_id = behavior_id
                self.behavior_start_time = now
            elif behavior != self.current_behavior:
                self.current_behavior = behavior
                self.current_gcn_id = behavior_id
                self.behavior_start_time = now
            # 均勻權重：每幀貢獻相等，使 get_activity_score() 為純粹的時間視窗平均
            self.activity_window.append({"time": now, "activity": activity_value, "weight": 1.0})
        now_t = time.time()
        if now_t - self._last_save_time >= 30.0:
            self._last_save_time = now_t
            self.save_state()
    def get_activity_score(self):
        with self._lock:
            if len(self.activity_window) == 0:
                return 0
            now = time.time()
            recent = [r for r in self.activity_window if (now - r["time"]) < BehaviorTrackingConfig.ACTIVITY_SCORE_WINDOW_SECONDS]
            if len(recent) == 0:
                return 0  # 視窗內無資料 = 貓不在畫面或無運動
            n = len(recent)
            score = round(sum(r["activity"] for r in recent) / n)
            return max(0, min(100, score))
    def get_today_stats(self):
        with self._lock:
            self.check_daily_reset()
            total_active = (self.behavior_time["walk"] + self.behavior_time["scratch"]
                            + self.behavior_time["lick"] + self.behavior_time["shake"]
                            + self.behavior_time["stop"])
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
                "rest_time": round(self.rest_time, 1),
                "not_detected_time": round(self.not_detected_time, 1),
                "monitoring_seconds": round(self.monitoring_seconds, 1),
                "transition_matrix": dict(self.transition_matrix),
                "hourly_distribution": {h: dict(v) for h, v in self.hourly_distribution.items()},
            }
        return stats
    def add_monitoring_seconds(self, seconds: float) -> None:
        with self._lock:
            self.monitoring_seconds += seconds

    def add_alert(self, alert_type, message):
        with self._lock:
            self.alerts.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": alert_type, "message": message})

    def get_alerts(self):
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
            alerts.append({"level": "high", "icon": "🚨", "title": "搔抓時間異常", "message": f"今日累積搔抓 {scratch_time:.1f} 秒（{scratch_count}次）", "suggestion": "請檢查皮膚是否有紅腫、掉毛、傷口", "action": "聯絡獸醫"})
        elif scratch_count >= BehaviorTrackingConfig.SCRATCH_ALERT_COUNT_THRESHOLD:
            alerts.append({"level": "medium", "icon": "⚠️", "title": "搔抓頻率偏高", "message": f"今日已搔抓 {scratch_count} 次（累積{scratch_time:.1f}秒）", "suggestion": "建議觀察是否有皮膚不適症狀", "action": "持續觀察"})
        if lick_time > BehaviorTrackingConfig.LICK_ALERT_TIME_SECONDS:
            alerts.append({"level": "medium", "icon": "🧼", "title": "舔舐時間較長", "message": f"今日舔舐 {lick_time:.1f} 秒（{lick_count}次）", "suggestion": "可能有壓力或皮膚問題", "action": "觀察精神狀態"})
        if shake_count >= BehaviorTrackingConfig.SHAKE_ALERT_COUNT_THRESHOLD:
            alerts.append({"level": "medium", "icon": "🔄", "title": "甩頭動作頻繁", "message": f"今日甩頭 {shake_count} 次（累積{shake_time:.1f}秒）", "action": "檢查耳朵"})
        if stop_time > BehaviorTrackingConfig.STOP_ALERT_TIME_SECONDS:
            alerts.append({"level": "medium", "icon": "⏹", "title": "長時間靜止不動", "message": f"今日累積靜止 {stop_time:.1f} 秒（{stop_count}次）", "suggestion": "貓咪長時間靜止，可能有身體不適", "action": "觀察精神與食慾"})
        total_time = lick_time + scratch_time + walk_time + shake_time + stop_time
        if total_time > 0 and walk_time < BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS:
            alerts.append({"level": "medium", "icon": "😴", "title": "活動度過低", "message": f"今日走動時間僅 {walk_time:.1f} 秒（低於門檻 {BehaviorTrackingConfig.LOW_ACTIVITY_TIME_THRESHOLD_SECONDS:.0f} 秒）", "suggestion": "貓咪活動不足，可能有身體不適", "action": "嘗試互動或遊玩"})
        return alerts
