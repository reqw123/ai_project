"""
CSV 日誌記錄
"""
import csv
from datetime import datetime
from pathlib import Path

class CSVLogger:
    def __init__(self, csv_path):
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame", "Timestamp", "Behavior", "GCN_Confidence", "Abnormal", "Motion_Score", "Stability"])
    def log(self, frame_idx, behavior, confidence, abnormal, motion_score, stability):
        self.csv_writer.writerow([
            frame_idx,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            behavior,
            f"{confidence:.4f}",
            "YES" if abnormal else "NO",
            f"{motion_score:.6f}",
            f"{stability:.6f}"
        ])
    def close(self):
        self.csv_file.close()


class BehaviorSegmentLogger:
    """每段行為結束時寫入一筆 CSV，供行為趨勢分析使用。"""
    def __init__(self, csv_path):
        self.path = Path(csv_path)
        self._ensure_header()
    def _ensure_header(self):
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    "date", "time", "behavior_id", "behavior_name", "duration_sec", "activity"
                ])
    def log_segment(self, behavior_id, behavior_name, duration_sec, activity=0):
        now = datetime.now()
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                behavior_id,
                behavior_name,
                f"{duration_sec:.1f}",
                activity,
            ])
    def close(self):
        pass
