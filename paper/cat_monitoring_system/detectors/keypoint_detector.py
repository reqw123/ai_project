"""
YOLO Pose 關鍵點檢測封裝
"""
from ultralytics import YOLO
import numpy as np

class KeypointDetector:
    def __init__(self, model_path, device='cuda', imgsz=640, conf_thres=0.5):
        self.model = YOLO(model_path)
        self._use_half = False
        try:
            self.model.to(device)
            # FP16 只在 CUDA 上有效；若 to() 成功且 device 是 CUDA，啟用 half
            if str(device).lower().startswith('cuda'):
                self._use_half = True
        except Exception:
            pass
        self.imgsz = imgsz
        self.conf_thres = conf_thres

    def detect(self, frame):
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf_thres,
            half=self._use_half, verbose=False
        )[0]
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kpts = results.keypoints.xy[0].cpu().numpy()
            # conf 在部分 YOLO 版本可能為 None（非 pose 模型），安全地 fallback 為全 1
            if results.keypoints.conf is not None:
                kpt_conf = results.keypoints.conf[0].cpu().numpy()
            else:
                kpt_conf = np.ones(kpts.shape[0], dtype=np.float32)
            bbox = results.boxes.xyxy[0].cpu().numpy() if results.boxes is not None else None
            # 明確轉為 Python float，避免 0-d ndarray 流入 JSON 序列化
            conf = float(results.boxes.conf[0].cpu().numpy()) if results.boxes is not None else None
            return kpts, kpt_conf, bbox, conf
        return None, None, None, None
