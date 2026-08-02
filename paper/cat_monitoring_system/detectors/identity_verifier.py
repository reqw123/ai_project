"""
貓咪身分驗證器（多貓辨識）—— 只用顏色特徵，不做行為辨識。

背景：個體化基線假設全程只有目標貓一隻入鏡。FrameProcessor 原本挑選
「這一幀該用哪隻貓的姿態」只靠 KeypointDetector 內部的 IoU 追蹤延續 +
信心值最高退回（見 keypoint_detector.py），那只保證空間連續性，不保證
身分——目標貓暫時離開、另一隻貓剛好在附近位置出現時，追蹤可能誤鎖到
錯的貓，行為資料就會被另一隻貓污染。這個模組補上身分這一層：給一幀
畫面 + YOLO 給的 bbox，回答「這是不是目標貓」。

方法（跟 tools/3_cat_identity_verification_test.py 反覆測試/還原過的版本
一致，數值門檻沿用該腳本驗證過的設定）：從 bbox 裁切區域算 HSV 色相/
飽和度 2D 直方圖，當成不需要額外模型的顏色指紋，跟 enroll 階段收集好的
參考樣本比最近鄰 Bhattacharyya 距離。純幾何/姿態比例特徵（第一版做法）
已經驗證過對姿勢/視角太敏感、單幀雜訊太高，故不採用；也沒有用「把距離
除以每隻貓自己基準鬆緊度」的正規化（同樣試過，會矯枉過正讓基準較鬆散
的貓搶走大部分曖昧樣本），純粹是最近鄰原始距離比較。

低耦合、fail-safe 設計（比照 config.py SQAConfig 的既有慣例）：
  - 建構子（__init__）在基準檔不存在/載入失敗時會拋例外，由呼叫端
    （FrameProcessor）負責 catch 並整個停用這個模組，回退成「偵測到的
    貓一律視為目標貓」的原本行為，不影響其餘偵測/分類流程。
  - verify() 本身不吞例外（維持單純），呼叫端一樣要包 try/except。
  - 這個模組被整個刪除，KeypointDetector/BehaviorClassifier 等其餘
    偵測/分類流程不受任何影響（沒有其他模組 import 這個檔案）。

要更新/重新調整基準，請用 tools/3_cat_identity_verification_test.py 的
enroll（跟 diagnose 驗證分離度）——這個模組只負責「載入已經驗證過的基準
檔案來做即時判斷」，不包含 enroll 或參數調校邏輯，那些留在測試腳本裡。
"""
import json
from pathlib import Path

import cv2
import numpy as np


class IdentityVerifier:
    # 這幾個門檻是 tools/3_cat_identity_verification_test.py 實測驗證過的
    # 版本，修改前請先用該測試腳本的 diagnose 模式確認效果，不要憑感覺調。
    H_BINS = 30
    S_BINS = 32
    HSV_S_MIN = 60
    HSV_V_MIN = 32
    HSV_V_MAX = 255
    BBOX_INSET_RATIO = 0.12
    UNKNOWN_DISTANCE_CEILING = 0.55

    def __init__(self, target_profile_path, other_profile_path=None, target_key="target"):
        """target_profile_path 必須是有效的基準檔，載入失敗會直接拋例外
        （FileNotFoundError/json 解析錯誤等），由呼叫端決定要不要整個停用
        這個模組。other_profile_path 可以是 None 或不存在的路徑，代表沒有
        第二隻貓的基準，這時會退化成單純「離目標貓基準夠不夠近」的門檻
        判斷，不做兩貓最近鄰比較。"""
        self.target_key = target_key
        self._profiles = {target_key: self._load_profile(target_profile_path)}
        if other_profile_path:
            other_path = Path(other_profile_path)
            if other_path.exists():
                self._profiles["other"] = self._load_profile(other_path)

    @staticmethod
    def _load_profile(path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        hists = [np.array(s, dtype=np.float32) for s in data["samples"]]
        if not hists:
            raise ValueError(f"基準檔沒有任何樣本，無法使用: {path}")
        return hists

    def _extract_histogram(self, frame, bbox):
        """從 bbox 區域裁切出 HSV 顏色直方圖特徵，跟 enroll 階段用的是同一套
        邏輯（見 tools/3_cat_identity_verification_test.py 的
        extract_color_histogram）。回傳 None 代表 bbox 太小或裁切失敗。"""
        if bbox is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None

        ix1 = int(np.clip(x1 + bw * self.BBOX_INSET_RATIO, 0, w - 1))
        iy1 = int(np.clip(y1 + bh * self.BBOX_INSET_RATIO, 0, h - 1))
        ix2 = int(np.clip(x2 - bw * self.BBOX_INSET_RATIO, 0, w))
        iy2 = int(np.clip(y2 - bh * self.BBOX_INSET_RATIO, 0, h))
        if ix2 - ix1 < 5 or iy2 - iy1 < 5:
            ix1, iy1 = int(max(0, x1)), int(max(0, y1))
            ix2, iy2 = int(min(w, x2)), int(min(h, y2))
            if ix2 - ix1 < 5 or iy2 - iy1 < 5:
                return None

        crop = frame[iy1:iy2, ix1:ix2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, self.HSV_S_MIN, self.HSV_V_MIN), (180, 255, self.HSV_V_MAX))
        if cv2.countNonZero(mask) < 20:
            mask = None

        hist = cv2.calcHist([hsv], [0, 1], mask, [self.H_BINS, self.S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
        return hist.flatten().astype(np.float32)

    def _hist_distance(self, a, b):
        """Bhattacharyya 距離：0=完全相同，1=完全不同。"""
        ha = a.reshape(self.H_BINS, self.S_BINS)
        hb = b.reshape(self.H_BINS, self.S_BINS)
        return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))

    def _best_match_distance(self, hist, profile_hists):
        return min(self._hist_distance(hist, p) for p in profile_hists)

    def verify(self, frame, bbox):
        """判定這個 bbox 是不是目標貓。

        回傳 (is_target: bool, matched_key: str|None, distance: float|None)。
        matched_key/distance 目前只供除錯/記錄用，FrameProcessor 只看
        is_target。bbox 太小/裁切失敗等無法判斷的情況，保守回傳
        is_target=False——不確定就不採信，避免把不確定的資料誤算進目標貓
        的統計裡（寧可少算幾幀，不要污染基線）。"""
        hist = self._extract_histogram(frame, bbox)
        if hist is None:
            return False, None, None

        dists = {key: self._best_match_distance(hist, hists) for key, hists in self._profiles.items()}
        best_key = min(dists, key=dists.get)
        best_dist = dists[best_key]
        if best_dist > self.UNKNOWN_DISTANCE_CEILING:
            return False, None, best_dist
        return best_key == self.target_key, best_key, best_dist
