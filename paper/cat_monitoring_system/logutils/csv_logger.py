"""
CSV 日誌記錄
"""
import csv
import threading
import atexit
from datetime import datetime
from pathlib import Path
from config import LoggingConfig, CatIdentityConfig

class CSVLogger:
    HEADER = ["Frame", "Timestamp", "Behavior", "GCN_Confidence", "Is_Still", "Motion_Score", "Cat_ID"]

    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = LoggingConfig.CSV_PATH
        self._lock = threading.Lock()
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 附加模式（跟 BehaviorSegmentLogger 一致）：先前用 'w' 截斷模式，伺服器
        # 每次重啟（crash/redeploy）都會把先前累積的逐幀記錄整個清空重寫。
        write_header = not path.exists() or path.stat().st_size == 0
        self.csv_file = open(path, 'a', newline='', buffering=1)  # line-buffered
        try:
            self.csv_writer = csv.writer(self.csv_file)
            if write_header:
                self.csv_writer.writerow(self.HEADER)
                self.csv_file.flush()
        except Exception:
            self.csv_file.close()
            raise
        atexit.register(self._atexit_close)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def log(self, frame_idx, behavior, confidence, is_still, motion_score):
        with self._lock:
            self.csv_writer.writerow([
                frame_idx,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                behavior,
                f"{confidence:.4f}",
                "YES" if is_still else "NO",
                f"{motion_score:.6f}",
                CatIdentityConfig.CAT_ID,
            ])
            self.csv_file.flush()

    def close(self):
        with self._lock:
            if not self.csv_file.closed:
                self.csv_file.flush()
                self.csv_file.close()

    def _atexit_close(self):
        try:
            self.close()
        except Exception:
            pass


class BehaviorSegmentLogger:
    """每段行為結束時寫入一筆 CSV，供行為趨勢分析使用。"""

    HEADER = ["date", "time", "behavior_id", "behavior_name", "duration_sec", "activity", "cat_id"]

    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = LoggingConfig.SEGMENTS_CSV_PATH
        self.path = Path(csv_path)
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._open()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        self._file = open(self.path, 'a', newline='', encoding='utf-8', buffering=1)
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self.HEADER)
            self._file.flush()
        atexit.register(self._atexit_close)

    def log_segment(self, behavior_id, behavior_name, duration_sec, activity=0):
        now = datetime.now()
        with self._lock:
            self._writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                behavior_id,
                behavior_name,
                f"{duration_sec:.1f}",
                activity,
                CatIdentityConfig.CAT_ID,
            ])
            self._file.flush()

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()

    def _atexit_close(self):
        try:
            self.close()
        except Exception:
            pass
