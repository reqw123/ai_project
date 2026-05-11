"""
YOLO Pose 關鍵點檢測封裝
"""
from ultralytics import YOLO
import numpy as np

class KeypointDetector:
    def __init__(self, model_path, device='cuda', imgsz=640, conf_thres=0.5):
        self.model = YOLO(model_path)
        try:
            self.model.to(device)
        except:
            pass
        self.imgsz = imgsz
        self.conf_thres = conf_thres
    def detect(self, frame):
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_thres, verbose=False)[0]
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            kpts = results.keypoints.xy[0].cpu().numpy()
            kpt_conf = results.keypoints.conf[0].cpu().numpy()
            bbox = results.boxes.xyxy[0].cpu().numpy() if results.boxes is not None else None
            conf = results.boxes.conf[0].cpu().numpy() if results.boxes is not None else None
            return kpts, kpt_conf, bbox, conf
        return None, None, None, None
