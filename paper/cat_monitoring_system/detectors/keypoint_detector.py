"""
YOLO Pose 關鍵點檢測封裝
"""
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class KeypointDetector:
    def __init__(self, model_path, device='cuda', imgsz=640, conf_thres=0.5,
                 track_iou_thres=0.3, track_max_missed=10):
        """
        track_iou_thres:  多隻貓同框時，與上一幀鎖定 bbox 的 IoU 需 ≥ 此值才視為同一隻貓延續追蹤；
                          否則放棄追蹤，改選信心最高的偵測（等同重新鎖定目標）。
        track_max_missed: 連續幾幀完全沒偵測到（貓消失/被遮擋）後，才放棄先前鎖定的目標。
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO 模型檔案不存在: {model_path}")
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
        self.track_iou_thres = track_iou_thres
        self.track_max_missed = track_max_missed
        self._prev_bbox = None
        self._missed_count = 0

    def reset_track(self):
        """開始處理新影片/新串流時呼叫，避免把上一段影片鎖定的貓誤帶到新的一段。"""
        self._prev_bbox = None
        self._missed_count = 0

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 1e-9 else 0.0

    def detect(self, frame):
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf_thres,
            quantize=16 if self._use_half else None, verbose=False
        )[0]
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            n = len(results.keypoints.xy)
            has_boxes = results.boxes is not None and len(results.boxes) > 0

            # 預設：選信心最高的偵測（單貓情境下這就是唯一解）
            best = 0
            if has_boxes:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                best = int(np.argmax(confs))

                # 多隻貓同框時，優先延續「上一幀鎖定的同一隻貓」而非重新比信心值，
                # 避免兩隻貓信心值來回互換時，骨架序列在不同貓之間跳動
                if len(boxes_xyxy) > 1 and self._prev_bbox is not None:
                    ious = np.array([self._iou(self._prev_bbox, b) for b in boxes_xyxy])
                    track_idx = int(np.argmax(ious))
                    if ious[track_idx] >= self.track_iou_thres:
                        best = track_idx

                if best >= n:
                    best = 0

            kpts = results.keypoints.xy[best].cpu().numpy()
            # conf 在部分 YOLO 版本可能為 None（非 pose 模型），安全地 fallback 為全 1
            if results.keypoints.conf is not None:
                kpt_conf = results.keypoints.conf[best].cpu().numpy()
            else:
                kpt_conf = np.ones(kpts.shape[0], dtype=np.float32)
            bbox = results.boxes.xyxy[best].cpu().numpy() if has_boxes else None
            # 明確轉為 Python float，避免 0-d ndarray 流入 JSON 序列化
            conf = float(results.boxes.conf[best].cpu().numpy()) if has_boxes else None

            if bbox is not None:
                self._prev_bbox = bbox
                self._missed_count = 0

            return kpts, kpt_conf, bbox, conf

        # 這幀沒偵測到任何目標：累積遺失幀數，超過門檻才放棄先前鎖定的目標
        # （容忍短暫遮擋，避免遮擋一結束就被當成新目標重新選）
        self._missed_count += 1
        if self._missed_count > self.track_max_missed:
            self._prev_bbox = None
        return None, None, None, None
