"""
異常檢測（EMA）
"""
import numpy as np

class AnomalyDetector:
    def __init__(self, alpha=0.7, abnormal_thres=0.2):
        self.ema_motion = 0.0
        self.alpha = alpha
        self.abnormal_thres = abnormal_thres
        self.prev_kpts = None
    def detect(self, kpts, kpt_conf):
        abnormal = False
        activity_value = 0.0
        if self.prev_kpts is not None:
            disp = np.linalg.norm(kpts - self.prev_kpts, axis=1)
            valid = kpt_conf > 0.5
            if np.any(valid):
                motion_score = float(np.mean(disp[valid]))
                self.ema_motion = self.alpha * motion_score + (1 - self.alpha) * self.ema_motion
                abnormal = self.ema_motion > self.abnormal_thres
                MAX_MOTION = 20.0
                norm_motion = min(motion_score / MAX_MOTION, 1.0)
                activity_value = int(norm_motion * 100)
                if motion_score < 0.5:
                    activity_value = 0
        self.prev_kpts = kpts.copy()
        return abnormal, activity_value
