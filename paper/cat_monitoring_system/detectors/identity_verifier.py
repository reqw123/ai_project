"""
貓咪身分驗證器（多貓辨識）—— MobileNetV3-Small CNN，不做行為辨識。

背景：個體化基線假設全程只有目標貓一隻入鏡。FrameProcessor 原本挑選
「這一幀該用哪隻貓的姿態」只靠 KeypointDetector 內部的 IoU 追蹤延續 +
信心值最高退回（見 keypoint_detector.py），那只保證空間連續性，不保證
身分——目標貓暫時離開、另一隻貓剛好在附近位置出現時，追蹤可能誤鎖到
錯的貓，行為資料就會被另一隻貓污染。這個模組補上身分這一層：給一幀
畫面 + YOLO 給的 bbox，回答「這是不是目標貓」。

方法：ImageNet 預訓練的 MobileNetV3-Small 微調成 N 類貓咪身分分類器，
從 bbox 裁切區域（往外擴 CROP_PADDING_RATIO，與訓練資料一致）跑一次
前向 → softmax。最高信心 < IDENTITY_CONF_THRESHOLD 視為「未知」。用最近
smooth_window 幀的多數決平滑，稀釋單幀雜訊/瞬間角度造成的偶發誤判。
（第一版的 HSV 顏色直方圖已完全汰換——顏色特徵丟掉紋理/斑紋，貓數變多
或兩隻毛色相近時分不開。）

訓練與模型檔：
  - 資料集：tools/cat_identity/1_build_dataset.py（YOLO 抽 bbox 裁切圖）
  - 訓練：  tools/cat_identity/2_train.py → C:\\ai_project\\identity_models\\
            run_<時間>_<流水號>/ + latest.pt（權重檔自帶 class_names /
            image_size / normalize 參數，本模組直接讀出來用）
  - 影片端驗證：tools/cat_identity/3_infer_video.py

低耦合、fail-safe 設計（比照 config.py SQAConfig 的既有慣例）：
  - 建構子在模型檔不存在/載入失敗/target_class 不在 class_names 裡時會拋
    例外，由呼叫端（FrameProcessor）catch 並整個停用這個模組，回退成
    「偵測到的貓一律視為目標貓」的原本行為，不影響其餘偵測/分類流程。
  - torch/torchvision 缺失時，frame_processor.py 的 guarded import
    （except Exception: _IdentityVerifier = None）會接住，同樣自動停用。
  - verify() 本身不吞例外（維持單純），呼叫端一樣包 try/except。
  - 這個模組被整個刪除，KeypointDetector/BehaviorClassifier 等其餘流程
    不受任何影響（只有 frame_processor.py 以 guarded import 引用它）。
"""
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class IdentityVerifier:
    # 這幾個門檻沿用訓練/驗證時的設定；改之前先看 cat_identity/2_train.py 產出的
    # confusion_matrix.png / test_metrics.json 確認效果，不要憑感覺調。
    IDENTITY_CONF_THRESHOLD = 0.80          # 單幀 softmax 最高值低於此 → 該幀判「未知」（預設；可由建構子覆蓋）
    CROP_PADDING_RATIO = 0.04      # bbox 往外擴的比例，須與 cat_identity/1_build_dataset.py 訓練時一致
    DEFAULT_SMOOTH_WINDOW = 5      # 取最近幾幀的原始判定做多數決

    def __init__(
        self, model_path, target_class, device="cuda", smooth_window=None,
        identity_conf_threshold=None,
    ):
        """model_path 必須是 cat_identity/2_train.py 產出的有效權重檔（.pt），
        載入失敗會直接拋例外（FileNotFoundError / KeyError / RuntimeError 等），
        由呼叫端決定要不要整個停用這個模組。

        target_class 是「目標貓」在權重檔 class_names 裡的名稱；不在裡面會
        拋 ValueError。其餘所有類別（含信心不足判為「未知」）都會被視為
        「不是目標貓」。

        identity_conf_threshold：單幀 softmax 信心門檻，None＝用 IDENTITY_CONF_THRESHOLD 預設。
        由 FrameProcessor 傳入 CatIdentityConfig.IDENTITY_CONF_THRESHOLD（可在
        設定視窗調整）。
        """
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.class_names = list(ckpt["class_names"])
        if target_class not in self.class_names:
            raise ValueError(
                f"target_class={target_class!r} 不在模型 class_names={self.class_names} 裡"
            )
        self.target_class = target_class
        self.image_size = int(ckpt.get("image_size", 128))
        mean = ckpt.get("norm_mean", _IMAGENET_MEAN)
        std = ckpt.get("norm_std", _IMAGENET_STD)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, len(self.class_names))
        model.load_state_dict(ckpt["model_state"])
        self.model = model.to(self.device).eval()

        resize_to = int(round(self.image_size * 1.25))
        self._transform = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        win = int(smooth_window) if smooth_window else self.DEFAULT_SMOOTH_WINDOW
        self._recent = deque(maxlen=max(1, win))

        self.identity_conf_threshold = (
            float(identity_conf_threshold)
            if identity_conf_threshold is not None
            else self.IDENTITY_CONF_THRESHOLD
        )

    def _crop_bbox(self, frame, bbox):
        """依 bbox 裁切、往外擴 CROP_PADDING_RATIO 後夾在畫面內。回傳 BGR ndarray 或 None。
        與 tools/cat_identity/1_build_dataset.py 的 crop_bbox 邏輯一致（訓練資料同一套裁法）。"""
        if bbox is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return None
        px, py = bw * self.CROP_PADDING_RATIO, bh * self.CROP_PADDING_RATIO
        cx1 = int(np.clip(x1 - px, 0, w - 1))
        cy1 = int(np.clip(y1 - py, 0, h - 1))
        cx2 = int(np.clip(x2 + px, 0, w))
        cy2 = int(np.clip(y2 + py, 0, h))
        if cx2 - cx1 < 10 or cy2 - cy1 < 10:
            return None
        return frame[cy1:cy2, cx1:cx2]

    @torch.no_grad()
    def _classify(self, crop_bgr):
        """一張 BGR 裁切圖 → (pred_class_name_or_None, top_conf)。
        信心低於 IDENTITY_CONF_THRESHOLD 時 pred 回 None（未知），top_conf 仍照實回傳。"""
        rgb = crop_bgr[:, :, ::-1]  # BGR → RGB
        x = self._transform(Image.fromarray(np.ascontiguousarray(rgb))).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        top_i = int(probs.argmax())
        top_p = float(probs[top_i])
        pred = self.class_names[top_i] if top_p >= self.identity_conf_threshold else None
        return pred, top_p

    @torch.no_grad()
    def target_probability(self, frame, bbox):
        """單一 bbox 是「目標貓」的 softmax 機率（0.0–1.0），裁切失敗回傳 None。

        跟 verify() 不同：**不動平滑佇列**、不套用信心門檻，只回傳這一個 crop
        當下的機率。多隻貓同框時 FrameProcessor 用它逐一評分、挑出最像目標貓的
        那隻，再對挑到的那隻呼叫一次 verify() 做跨幀平滑（維持 verify() 一幀一次
        呼叫的既有契約）。"""
        crop = self._crop_bbox(frame, bbox)
        if crop is None:
            return None
        rgb = crop[:, :, ::-1]  # BGR → RGB
        x = self._transform(
            Image.fromarray(np.ascontiguousarray(rgb))
        ).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return float(probs[self.class_names.index(self.target_class)])

    def verify(self, frame, bbox):
        """判定這個 bbox 是不是目標貓。

        回傳 (is_target: bool, matched_key: str|None, score: float|None)。
          - matched_key：平滑後的類別名稱（含 None=未知），供除錯/畫徽章用；
            FrameProcessor 只看 is_target。
          - score：本幀 CNN 最高信心（未平滑）。
        bbox 太小/裁切失敗等無法判斷的情況：本幀原始判定記為「未知」，一樣
        納入多數決（連續幾幀都判不出來 → 平滑結果變未知 → is_target=False，
        寧可少算幾幀，不要污染基線）。
        """
        crop = self._crop_bbox(frame, bbox)
        if crop is None:
            raw, score = None, None
        else:
            raw, score = self._classify(crop)

        self._recent.append(raw)
        smoothed = Counter(self._recent).most_common(1)[0][0]
        return smoothed == self.target_class, smoothed, score
