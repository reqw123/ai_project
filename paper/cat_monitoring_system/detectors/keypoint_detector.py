"""
YOLO Pose 關鍵點檢測封裝
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class KeypointDetector:
    """封裝 YOLO-Pose 推論，並在多隻貓同框時以 IoU 追蹤延續同一隻貓的鎖定目標。"""

    def __init__(
        self,
        model_path,
        device="cuda",
        imgsz=640,
        bbox_conf_thres=0.5,
        track_iou_thres=0.3,
        track_max_missed=10,
    ):
        """
        bbox_conf_thres:  YOLO **bbox 偵測**信心門檻（傳給 model.predict(conf=)）——低於此值的
                          偵測框整個丟棄。跟「每個關鍵點的信心」(kpt_conf) 是兩回事：這裡管
                          「這個框算不算一隻貓」，kpt_conf 管「這個關節座標可不可信」。
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
            if str(device).lower().startswith("cuda"):
                self._use_half = True
        except Exception:
            pass
        self.imgsz = imgsz
        self.bbox_conf_thres = bbox_conf_thres
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

    def detect(self, frame, return_all_instances=False, single_cat=False):
        """對單一影格跑 YOLO-Pose 推論，回傳 (keypoints, keypoint_conf, bbox, bbox_conf)；
        沒有偵測到目標時回傳全 None。

        return_all_instances=True 時，額外多回傳第 5 個值：這一幀所有偵測到的貓
        （含被追蹤鎖定機制篩掉、沒被選為前 4 個回傳值的那些），每筆為
        (kpts, kpt_conf, bbox, bbox_conf)；沒有任何偵測時為空 list。這個參數
        只是把同一次 YOLO 推論裡本來就算出來、但預設會被丟棄的其餘偵測結果
        一併回傳，不會多跑一次推論，也完全不影響前 4 個值的追蹤/選取邏輯——
        單純給呼叫端（例如多貓同框時的畫面視覺化）用，預設 False 時行為與
        既有呼叫端完全一致。

        single_cat=True 時：直接讓 YOLO 只回傳「信心值最高」的那一隻
        （model.predict(max_det=1)），完全不套用跨幀 IoU 追蹤延續，也不會有
        「上一幀鎖定的貓」這回事——RunModeConfig.SYSTEM_MODE=="single"（單貓
        系統模式）專用：假設畫面全程只有一隻貓，每一幀都獨立鎖信心最高的偵測。
        跟 return_all_instances 互斥沒有意義（只會有一筆），呼叫端不會同時給。
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.bbox_conf_thres,
            max_det=1 if single_cat else 300,
            quantize=16 if self._use_half else None,
            verbose=False,
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
                # 避免兩隻貓信心值來回互換時，骨架序列在不同貓之間跳動。
                # single_cat=True（單貓系統模式）時完全不做這件事——永遠鎖信心最高的。
                if (
                    not single_cat
                    and len(boxes_xyxy) > 1
                    and self._prev_bbox is not None
                ):
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
            bbox_conf = float(results.boxes.conf[best].cpu().numpy()) if has_boxes else None

            all_instances = None
            if return_all_instances:
                all_instances = []
                for i in range(n):
                    _kpts_i = results.keypoints.xy[i].cpu().numpy()
                    if results.keypoints.conf is not None:
                        _kconf_i = results.keypoints.conf[i].cpu().numpy()
                    else:
                        _kconf_i = np.ones(_kpts_i.shape[0], dtype=np.float32)
                    _bbox_i = results.boxes.xyxy[i].cpu().numpy() if has_boxes and i < len(results.boxes) else None
                    _bconf_i = float(results.boxes.conf[i].cpu().numpy()) if has_boxes and i < len(results.boxes) else None
                    all_instances.append((_kpts_i, _kconf_i, _bbox_i, _bconf_i))

            if bbox is not None:
                self._prev_bbox = bbox
                self._missed_count = 0

            if return_all_instances:
                return kpts, kpt_conf, bbox, bbox_conf, all_instances
            return kpts, kpt_conf, bbox, bbox_conf

        # 這幀沒偵測到任何目標：累積遺失幀數，超過門檻才放棄先前鎖定的目標
        # （容忍短暫遮擋，避免遮擋一結束就被當成新目標重新選）
        self._missed_count += 1
        if self._missed_count > self.track_max_missed:
            self._prev_bbox = None
        if return_all_instances:
            return None, None, None, None, []
        return None, None, None, None
