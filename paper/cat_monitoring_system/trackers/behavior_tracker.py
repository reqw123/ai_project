"""
行為追蹤和統計
"""
from collections import deque
from datetime import datetime
import time

class ImprovedBehaviorTracker:
    def __init__(self):
        self.behavior_time = {"walk": 0.0, "scratch": 0.0, "lick": 0.0, "shake": 0.0}
        self.behavior_count = {"walk": 0, "scratch": 0, "lick": 0, "shake": 0}
        self.rest_time = 0.0  # 非行為時長（behavior_id == -1）
        self.behavior_history = deque(maxlen=500)
        self.current_behavior = None
        self.behavior_start_time = time.time()
        self.last_update_time = time.time()  # 用於計算逐幀時間差
        self.last_reset = datetime.now().date()
        self.activity_window = deque(maxlen=60)
        self.alerts = deque(maxlen=50)
    def check_daily_reset(self):
        today = datetime.now().date()
        if today != self.last_reset:
            self.behavior_time = {k: 0.0 for k in self.behavior_time}
            self.behavior_count = {k: 0 for k in self.behavior_count}
            self.rest_time = 0.0  # 重置非行為時長
            self.last_reset = today
    def map_gcn_to_tracker(self, behavior_id):
        mapping = {0: "walk", 1: "lick", 2: "scratch", 3: "shake"}
        return mapping.get(behavior_id, "walk")
    def update(self, behavior_id, activity_value):
        self.check_daily_reset()
        now = time.time()
        dt = now - self.last_update_time  # 逐幀時間差
        self.last_update_time = now
        
        # 信心不足時（behavior_id == -1）：累積到 rest_time
        if behavior_id == -1:
            self.rest_time += dt
            self.activity_window.append({"time": now, "activity": activity_value, "weight": 0.5})
            return
        
        # 有效行為（0~3）：累積到對應 behavior_time
        behavior = self.map_gcn_to_tracker(behavior_id)
        duration = now - self.behavior_start_time
        record_this = False
        if behavior != self.current_behavior:
            record_this = True
        elif self.current_behavior is not None and duration >= 2.0:
            record_this = True
        if record_this and self.current_behavior is not None:
            if self.current_behavior in self.behavior_time:
                self.behavior_time[self.current_behavior] += duration
            if self.current_behavior in self.behavior_count:
                self.behavior_count[self.current_behavior] += 1
            record = {
                "behavior": self.current_behavior,
                "gcn_behavior_id": behavior_id,
                "timestamp": datetime.now(),
                "duration": round(duration, 1),
                "activity": activity_value
            }
            self.behavior_history.append(record)
            self.current_behavior = behavior
            self.behavior_start_time = now
        elif behavior != self.current_behavior:
            self.current_behavior = behavior
            self.behavior_start_time = now
        self.activity_window.append({"time": now, "activity": activity_value, "weight": duration if duration > 0 else 0.5})
    def get_activity_score(self):
        if len(self.activity_window) == 0:
            return 50
        now = time.time()
        # 取最近 3 秒的活動量，讓分數更平滑
        recent = [r for r in self.activity_window if (now - r["time"]) < 3]
        if len(recent) == 0:
            return 50
        total_weight = sum(r["weight"] for r in recent)
        weighted_sum = sum(r["activity"] * r["weight"] for r in recent)
        score = round(weighted_sum / total_weight) if total_weight > 0 else 50
        return max(0, min(100, score))
    def get_today_stats(self):
        self.check_daily_reset()
        total_active = self.behavior_time["walk"] + self.behavior_time["scratch"] + self.behavior_time["lick"] + self.behavior_time["shake"]
        stats = {
            "walk": self.behavior_count["walk"],
            "walk_time": round(self.behavior_time["walk"], 1),
            "scratch": self.behavior_count["scratch"],
            "scratch_time": round(self.behavior_time["scratch"], 1),
            "lick": self.behavior_count["lick"],
            "lick_time": round(self.behavior_time["lick"], 1),
            "shake": self.behavior_count["shake"],
            "shake_time": round(self.behavior_time["shake"], 1),
            "active_time": round(total_active, 1),
            "rest_time": round(self.rest_time, 1)  # 直接使用累積的非行為時長
        }

        # Node-RED 相容欄位：沿用舊模板命名（normal/groom）
        stats["normal"] = stats["walk"]
        stats["normal_time"] = stats["walk_time"]
        stats["groom"] = stats["lick"]
        stats["groom_time"] = stats["lick_time"]
        return stats
    def add_alert(self, alert_type, message):
        self.alerts.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "type": alert_type, "message": message})
    def get_alerts(self):
        alerts = []
        scratch_time = self.behavior_time.get("scratch", 0)
        scratch_count = self.behavior_count.get("scratch", 0)
        if scratch_time > 10:
            alerts.append({"level": "high", "icon": "🚨", "title": "搔抓時間異常", "message": f"今日累積搔抓 {scratch_time:.1f} 秒（{scratch_count}次）", "suggestion": "請檢查皮膚是否有紅腫、掉毛、傷口", "action": "聯絡獸醫"})
        elif scratch_count >= 5:
            alerts.append({"level": "medium", "icon": "⚠️", "title": "搔抓頻率偏高", "message": f"今日已搔抓 {scratch_count} 次（累積{scratch_time:.1f}秒）", "suggestion": "建議觀察是否有皮膚不適症狀", "action": "持續觀察"})
        lick_time = self.behavior_time.get("lick", 0)
        lick_count = self.behavior_count.get("lick", 0)
        if lick_time > 10:
            alerts.append({"level": "medium", "icon": "🧼", "title": "舔舐時間較長", "message": f"今日舔舐 {lick_time:.1f} 秒（{lick_count}次）", "suggestion": "可能有壓力或皮膚問題", "action": "觀察精神狀態"})
        shake_time = self.behavior_time.get("shake", 0)
        shake_count = self.behavior_count.get("shake", 0)
        if shake_count >= 10:
            alerts.append({
                "level": "medium",
                "icon": "🔄",
                "title": "甩頭動作頻繁",
                "message": f"今日甩頭 {shake_count} 次（累積{shake_time:.1f}秒）",
                "action": "檢查耳朵"
            })
        total_time = lick_time + scratch_time + self.behavior_time.get("walk", 0) + self.behavior_time.get("shake", 0)
        # 四類行為獨立時，normal 基準即 walk，不再使用不存在的 normal bucket
        normal_time = self.behavior_time.get("walk", 0)
        if total_time > 0 and normal_time < 20:
            alerts.append({"level": "medium", "icon": "😴", "title": "活動度過低", "message": f"今日活動時間 {(total_time - normal_time):.1f} 秒，靜止 {normal_time:.1f} 秒", "suggestion": "貓咪活動不足，可能有身體不適", "action": "嘗試互動或遊玩"})
        return alerts
