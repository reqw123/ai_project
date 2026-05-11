#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
貓咪姿態標記品質自動檢測系統 - 完整版
整合四種檢測方法: Teacher-Student, Augmentation Consistency, Loss Tracking, Pose Embedding

【重要】評分權重設計:
────────────────────────────────────────────────────────────
品質檢測的核心是"抓到真正的標記錯誤", 因此權重分配如下:

✅ 核心檢測方法(高權重):
        • Teacher-Student 一致性: 45% (最重要, 兩模型都錯的機率低)
        • Pose Embedding 離群: 30% (次重要, 能抓到姿態異常)

⚠️ 輔助診斷信號(低權重):
    • Augmentation Consistency: 10%（降低！只作為輔助）
    • Loss Tracking: 15%（輔助，需要 GT 標記）

原理：
1. Teacher-Student 最可靠：兩個不同模型都預測錯，標記很可能有問題
2. Embedding 離群很重要：姿態特徵異常通常是標記問題
3. Augmentation 只是輔助：外觀變化不應主導判定
4. Loss Tracking 輔助參考：需要 GT，且較主觀

可疑判定標準：
- 綜合評分 < 60 分
- 或有 2+ 個嚴重問題（不含 Augmentation 問題）
────────────────────────────────────────────────────────────

【重要】Augmentation Consistency 設計原則：
────────────────────────────────────────────────────────────
對於 Pose 任務，資料增強必須遵循「不改變關鍵點語意」原則：

✅ 使用的增強（不影響關鍵點位置關係）：
    • 亮度/對比度調整
    • Gamma 調整
    • JPEG 壓縮
    • 高斯噪聲（小幅）
    • 高斯模糊（小 kernel）
    • 色調/飽和度調整

❌ 不使用的增強（會改變關鍵點語意）：
    • 水平/垂直翻轉（左右對稱會混淆）
    • 旋轉（角度變化影響姿態判斷）
    • 裁剪（可能丟失關鍵點）
    • 縮放/仿射變換（改變比例關係）

判定閾值（v2.2 已調整為輔助信號）：
    • 一致性分數 < 0.5 才提示（寬鬆，只作提醒）
    • 變異度 > 70 像素才提示（很寬鬆）
    • 參考尺度：80 像素
    • 權重：僅 10%（不主導判定）

原理：
只使用外觀類增強，如果在這些溫和的增強下預測仍不穩定，
可能是品質問題，但**僅作為輔助信號，不單獨判定可疑**。
────────────────────────────────────────────────────────────

使用方法:
        python label_quality_checker_complete.py

作者: Auto Quality Checker
版本: v2.2
日期: 2026-01-21
"""

import os
import cv2
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import albumentations as A
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置常數 ====================
TOTAL_KPTS = 17
CONF_THRESHOLD = 0.3
# OKS_THRESHOLD = 0.5  # (未使用，保留說明)

# 各關鍵點的標準差 (用於計算 OKS)
KEYPOINT_SIGMAS = np.array([
    0.026, 0.025, 0.025,  # 0-2: nose, eyes
    0.035, 0.079, 0.072,  # 3-5: neck, body_mid, tail_base
    0.062, 0.107,         # 6-7: left_shoulder, left_front_paw
    0.062, 0.107,         # 8-9: right_shoulder, right_front_paw
    0.087, 0.089,         # 10-11: left_hip, left_hind_paw
    0.087, 0.089,         # 12-13: right_hip, right_hind_paw
    0.079, 0.079, 0.079   # 14-16: tail parts
])


# ==================== 數據結構 ====================
@dataclass
class PoseQualityMetrics:
    """單張圖片的品質評估指標"""
    image_path: str
    ts_consistency: float
    ts_oks_score: float
    aug_consistency: float
    aug_variance: float
    initial_loss: float
    final_loss: float
    loss_reduction: float
    loss_stability: float
    embedding_distance: float
    lof_score: float
    overall_score: float
    is_suspicious: bool
    suspicion_reasons: list
    yolo_detected: bool = False

class LabelQualityChecker:
    def _to_native(self, obj):
        """將 numpy 型態遞迴轉成 Python 原生型態，方便 JSON 序列化"""
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_native(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def _calculate_oks(self, kpts1: np.ndarray, kpts2: np.ndarray, img_shape: Tuple[int, int, int]) -> float:
        """計算 Object Keypoint Similarity (OKS)"""
        h, w = img_shape[:2]
        # 用影像尺度當作 scale（簡化版 OKS）
        scale = np.sqrt(h * w) + 1e-6

        d = np.linalg.norm(kpts1[:, :2] - kpts2[:, :2], axis=1)
        oks_per_kpt = np.exp(-(d ** 2) / (2 * (scale ** 2) * (KEYPOINT_SIGMAS ** 2) + 1e-6))

        vis = (kpts1[:, 2] > 0) & (kpts2[:, 2] > 0)
        if vis.sum() == 0:
            return 0.0
        return float(oks_per_kpt[vis].mean())

    def __init__(self, teacher_model_path, student_model_path=None, device='cuda', enable_teacher_student=True, enable_augmentation=True, enable_loss_tracking=True, enable_embedding=True):
        self.device = device
        if self.device == 'cuda' and (not torch.cuda.is_available()):
            print("⚠️ CUDA 不可用，已自動切換為 CPU")
            self.device = 'cpu'
        self.teacher_model = YOLO(teacher_model_path)
        self.student_model = YOLO(student_model_path) if student_model_path else None
        self.enable_teacher_student = enable_teacher_student
        self.enable_augmentation = enable_augmentation
        self.enable_loss_tracking = enable_loss_tracking
        self.enable_embedding = enable_embedding
        self.augmentation_pipeline = self._create_augmentation_pipeline() if enable_augmentation else []
        self.results = []
        self.embeddings = []
        self.image_paths = []
        self.embedding_detected_mask = []  # 新增: 標記哪些有偵測到
        dummy_kpts = np.ones((17, 3), dtype=np.float32)
        self.embedding_dim = len(self.extract_pose_embedding(dummy_kpts))

    def _create_augmentation_pipeline(self):
        """
        創建對 Pose 安全的增強管道
        原則：只做「不改變關鍵點語意」的增強
        - ✅ 外觀類增強（亮度、對比度、噪聲、模糊）
        - ❌ 空間變換（旋轉、翻轉、裁剪、縮放）
        """
        pipelines = []
        import albumentations as A
        pipelines.append(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.ImageCompression(quality_lower=75, quality_upper=95, p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.GaussNoise(var_limit=(5.0, 15.0), p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        ]))
        pipelines.append(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.5),
        ]))
        return pipelines

    # ==================== 方法 2: Augmentation Consistency ====================
    def check_augmentation_consistency(self, image: np.ndarray, n_augments: int = 7, teacher_kpts: np.ndarray = None) -> Tuple[float, float]:
        """
        檢查在不同增強下的預測一致性（使用 Pose 安全的增強）
        支援傳入 teacher_kpts 以避免重複推論。
        """
        predictions = []
        # 原始圖片預測
        if teacher_kpts is None:
            original_result = self.teacher_model.predict(image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
            teacher_kpts = self._extract_keypoints(original_result)
        if teacher_kpts is None:
            return 0.0, 1.0
        predictions.append(teacher_kpts)
        # 增強版本預測
        for i in range(min(n_augments, len(self.augmentation_pipeline))):
            augmented = self.augmentation_pipeline[i](image=image)['image']
            aug_result = self.teacher_model.predict(augmented, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
            aug_kpts = self._extract_keypoints(aug_result)
            if aug_kpts is not None:
                predictions.append(aug_kpts)
        if len(predictions) < 2:
            return 0.5, 0.5
        predictions_array = np.array([p[:, :2] for p in predictions])
        var_xy = np.var(predictions_array, axis=0)  # shape: (17,2), 單位 px^2
        std_xy = np.sqrt(var_xy)                    # 單位 px
        variance = float(np.mean(std_xy))           # 平均像素級抖動
        consistency_score = float(np.exp(-variance / 80.0))
        return consistency_score, variance
    
    # ==================== 方法 3: Loss Tracking ====================
    def track_training_loss(self, 
                           image: np.ndarray, 
                           gt_keypoints: np.ndarray,
                           n_iterations: int = 10) -> Tuple[float, float, float, float]:
        """追蹤訓練損失的變化"""
        losses = []
        for i in range(n_iterations):
            noise = np.random.normal(0, 5, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
            result = self.teacher_model.predict(noisy_image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
            pred_kpts = self._extract_keypoints(result)
            if pred_kpts is None:
                loss = 1000.0
            else:
                distances = np.linalg.norm(pred_kpts[:, :2] - gt_keypoints[:, :2], axis=1)
                loss = np.mean(distances)
            losses.append(loss)
        losses = np.array(losses)
        initial_loss = float(losses[0])
        final_loss = float(losses[-1])
        loss_reduction = float((initial_loss - final_loss) / (initial_loss + 1e-6))
        loss_stability = float(np.std(losses))
        return initial_loss, final_loss, loss_reduction, loss_stability
    
    # ==================== 方法 4: Pose Embedding ====================
    def extract_pose_embedding(self, keypoints: np.ndarray) -> np.ndarray:
        """從關鍵點提取姿態特徵向量"""
        features = []
        
        # 1. 歸一化座標
        if keypoints.shape[0] > 0:
            valid_kpts = keypoints[keypoints[:, 2] > 0][:, :2]
            if len(valid_kpts) > 0:
                bbox_min = valid_kpts.min(axis=0)
                bbox_max = valid_kpts.max(axis=0)
                bbox_size = bbox_max - bbox_min + 1e-6
                
                normalized_kpts = (keypoints[:, :2] - bbox_min) / bbox_size
                features.extend(normalized_kpts.flatten())
            else:
                features.extend(np.zeros(TOTAL_KPTS * 2))
        else:
            features.extend(np.zeros(TOTAL_KPTS * 2))
        
        # 2. 骨架長度
        skeleton_links = [
            (0, 1), (0, 2), (1, 2),
            (0, 3), (3, 4), (4, 5),
            (3, 6), (6, 7), (3, 8), (8, 9),
            (5, 10), (10, 11), (5, 12), (12, 13),
            (5, 14), (14, 15), (15, 16)
        ]
        
        for i, j in skeleton_links:
            if keypoints[i, 2] > 0 and keypoints[j, 2] > 0:
                length = np.linalg.norm(keypoints[i, :2] - keypoints[j, :2])
                features.append(length)
            else:
                features.append(0.0)
        
        # 3. 角度特徵
        angle_triplets = [(1, 0, 2), (3, 4, 5), (6, 3, 8)]
        
        for i, j, k in angle_triplets:
            if keypoints[i, 2] > 0 and keypoints[j, 2] > 0 and keypoints[k, 2] > 0:
                v1 = keypoints[i, :2] - keypoints[j, :2]
                v2 = keypoints[k, :2] - keypoints[j, :2]
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                features.append(angle)
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def detect_outliers(self) -> Dict[str, list]:
        """(已棄用) 使用 Local Outlier Factor 檢測離群樣本 (僅供參考)"""
        valid_indices = [i for i, e in enumerate(self.embeddings) if e is not None]
        valid_embeddings = [self.embeddings[i] for i in valid_indices]
        if len(valid_embeddings) < 10:
            print("⚠️ 樣本數量太少，跳過離群檢測")
            return {'outliers': [], 'lof_scores': []}
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(valid_embeddings)
        lof = LocalOutlierFactor(n_neighbors=min(20, len(valid_embeddings) - 1), contamination=0.1)
        outlier_labels = lof.fit_predict(embeddings_scaled)
        lof_scores = -lof.negative_outlier_factor_
        outlier_indices = [valid_indices[j] for j, lab in enumerate(outlier_labels) if lab == -1]
        return {
            'outliers': outlier_indices,
            'lof_scores': lof_scores.tolist()
        }
    
    # ==================== 輔助函數 ====================
    def _extract_keypoints(self, result) -> Optional[np.ndarray]:
        """從 YOLO 結果提取關鍵點"""
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None
        
        kpts_xy = result.keypoints.xy[0].cpu().numpy()
        kpts_conf = result.keypoints.conf[0].cpu().numpy()
        keypoints = np.column_stack([kpts_xy, kpts_conf])
        
        return keypoints
    
    def _load_ground_truth(self, label_path: str, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """從標記檔案載入 Ground Truth 關鍵點"""
        if not os.path.exists(label_path):
            return None
        
        h, w = img_shape[:2]
        
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None
            
            parts = line.split()
            if len(parts) < 1 + 4 + TOTAL_KPTS * 3:
                return None
            
            kpt_data = parts[5:]
            
            keypoints = []
            for i in range(TOTAL_KPTS):
                x = float(kpt_data[i * 3]) * w
                y = float(kpt_data[i * 3 + 1]) * h
                v = float(kpt_data[i * 3 + 2])
                keypoints.append([x, y, v])
            
            return np.array(keypoints, dtype=np.float32)
    
    # ==================== 主處理流程 ====================
    def process_dataset(self, 
                       data_dir: str,
                       output_path: str,
                       visualize: bool = True,
                       clear_output: bool = False) -> List[PoseQualityMetrics]:
        """
        處理整個資料集並生成品質報告
        
        Args:
            data_dir: 資料集目錄 (包含 images/ 和 labels/ 子目錄)
            output_path: 輸出報告路徑
            visualize: 是否生成視覺化圖表
            clear_output: 是否清空輸出目錄(預設 False, 會保留舊檔案)
        """
        data_dir = Path(data_dir)
        images_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        
        if not images_dir.exists():
            raise ValueError(f"找不到圖片目錄: {images_dir}")
        
        image_files = sorted(list(images_dir.glob("*.jpg")) + 
                           list(images_dir.glob("*.jpeg")) + 
                           list(images_dir.glob("*.png")))
        
        if len(image_files) == 0:
            raise ValueError(f"在 {images_dir} 中找不到圖片")
        
        # 處理輸出路徑（已由 resolve_output_path 統一處理）
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理輸出目錄清空選項
        if clear_output and output_dir.exists():
            import shutil
            print(f"🗑️  清空輸出目錄: {output_dir}")
            
            # 只清空特定檔案，不刪除整個目錄
            patterns_to_remove = [
                'quality_report.json',
                'suspicious_images.txt',
                'score_distribution.png',
                'metrics_scatter.png',
                'embedding_pca.png',
                'suspicious_samples'
            ]
            
            for pattern in patterns_to_remove:
                path = output_dir / pattern
                if path.is_file():
                    path.unlink()
                    print(f"   刪除: {pattern}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"   刪除目錄: {pattern}/")
        
        print(f"\n{'='*70}")
        print(f"🔍 開始檢測標記品質")
        print(f"{'='*70}")
        print(f"📂 圖片數量: {len(image_files)}")
        print(f"📂 資料目錄: {data_dir}")
        print(f"📂 輸出目錄: {output_dir}")
        if clear_output:
            print(f"🗑️  清空模式: 已清空舊檔案")
        else:
            print(f"📝 覆蓋模式: 會覆蓋同名檔案")
        print(f"{'='*70}\n")
        
        # 處理每張圖片
        for img_path in tqdm(image_files, desc="處理圖片"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_keypoints = self._load_ground_truth(str(label_path), image.shape)
            # Teacher predict cache
            teacher_result = self.teacher_model.predict(image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
            teacher_kpts = self._extract_keypoints(teacher_result)
            # 方法 1: Teacher-Student
            if self.enable_teacher_student:
                if self.student_model is not None:
                    student_result = self.student_model.predict(image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
                    student_kpts = self._extract_keypoints(student_result)
                    if teacher_kpts is None or student_kpts is None:
                        ts_consistency, ts_oks = 0.0, 0.0
                    else:
                        oks_score = self._calculate_oks(teacher_kpts, student_kpts, image.shape)
                        distances = np.linalg.norm(teacher_kpts[:, :2] - student_kpts[:, :2], axis=1)
                        avg_distance = np.mean(distances)
                        ts_consistency = float(np.exp(-avg_distance / 50))
                        ts_oks = float(oks_score)
                else:
                    ts_consistency, ts_oks = 1.0, 1.0
            else:
                ts_consistency, ts_oks = 1.0, 1.0
            # 方法 2: Augmentation Consistency
            if self.enable_augmentation:
                aug_consistency, aug_variance = self.check_augmentation_consistency(image, n_augments=7, teacher_kpts=teacher_kpts)
            else:
                aug_consistency, aug_variance = 1.0, 0.0
            # 方法 3: Loss Tracking
            if self.enable_loss_tracking:
                if gt_keypoints is not None:
                    init_loss, final_loss, loss_red, loss_stab = self.track_training_loss(image, gt_keypoints, n_iterations=5)
                else:
                    init_loss, final_loss, loss_red, loss_stab = 0, 0, 0.0, 0.0
            else:
                init_loss, final_loss, loss_red, loss_stab = 0, 0, 1.0, 0.0
            # 方法 4: Embedding
            yolo_detected = False
            if self.enable_embedding:
                if teacher_kpts is not None:
                    embedding = self.extract_pose_embedding(teacher_kpts)
                    self.embeddings.append(embedding)
                    self.embedding_detected_mask.append(True)
                    self.image_paths.append(str(img_path))
                    yolo_detected = True
                else:
                    # 不塞 0 向量，塞 None，後續不參與 LOF
                    self.embeddings.append(None)
                    self.embedding_detected_mask.append(False)
                    self.image_paths.append(str(img_path))
                    yolo_detected = False
            else:
                yolo_detected = teacher_kpts is not None
            # suspicion 判斷
            suspicion_reasons = []
            severe_flags = 0
            if teacher_kpts is None:
                suspicion_reasons.append("YOLO 未偵測到貓/關鍵點")
                severe_flags += 1
            if self.enable_teacher_student and self.student_model is not None and ts_consistency < 0.5:
                suspicion_reasons.append("Teacher-Student 不一致")
                severe_flags += 1
            if self.enable_augmentation and aug_consistency < 0.5:
                suspicion_reasons.append("Augmentation 不穩定")
            if self.enable_loss_tracking and loss_red < 0.3:
                suspicion_reasons.append("Loss 收斂不良")
                severe_flags += 1
            if self.enable_embedding and teacher_kpts is not None:
                embedding_score = 0.5  # 給一半分，等 LOF 再修正
            else:
                embedding_score = 0.0
            overall_score = (
                ts_consistency * 45 +
                embedding_score * 30 +
                aug_consistency * 10 +
                loss_red * 15
            )
            is_suspicious = (overall_score < 60) or (severe_flags >= 2)
            metrics = PoseQualityMetrics(
                image_path=str(img_path),
                ts_consistency=ts_consistency,
                ts_oks_score=ts_oks,
                aug_consistency=aug_consistency,
                aug_variance=aug_variance,
                initial_loss=init_loss,
                final_loss=final_loss,
                loss_reduction=loss_red,
                loss_stability=loss_stab,
                embedding_distance=0.0,  # 先佔位
                lof_score=0.0,           # 先佔位
                overall_score=overall_score,
                is_suspicious=is_suspicious,
                suspicion_reasons=suspicion_reasons,
                yolo_detected=yolo_detected
            )
            self.results.append(metrics)
        # LOF/embedding 統計
        if self.enable_embedding:
            # 只針對有偵測到的樣本做 LOF
            valid_indices = [i for i, flag in enumerate(self.embedding_detected_mask) if flag]
            valid_embeddings = [self.embeddings[i] for i in valid_indices if self.embeddings[i] is not None]
            if len(valid_embeddings) >= 10:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                embeddings_scaled = scaler.fit_transform(valid_embeddings)
                from sklearn.neighbors import LocalOutlierFactor
                lof = LocalOutlierFactor(n_neighbors=min(20, len(valid_embeddings) - 1), contamination=0.1)
                outlier_labels = lof.fit_predict(embeddings_scaled)
                lof_scores = -lof.negative_outlier_factor_
                # 將分數寫回原本的 self.results
                for idx, emb_idx in enumerate(valid_indices):
                    r = self.results[emb_idx]
                    r.lof_score = lof_scores[idx]
                    r.embedding_distance = lof_scores[idx]
                    embedding_score = max(0.0, 1.0 - min(1.0, lof_scores[idx] / 2.0))
                    r.overall_score = (
                        r.ts_consistency * 45 +
                        embedding_score * 30 +
                        r.aug_consistency * 10 +
                        r.loss_reduction * 15
                    )
                    if lof_scores[idx] > 1.5 and "Pose Embedding 離群" not in r.suspicion_reasons:
                        r.suspicion_reasons.append("Pose Embedding 離群")
                # 重新判斷 is_suspicious（可升可降）
                core_flags = ["Teacher-Student 不一致", "Loss 收斂不良", "Pose Embedding 離群", "YOLO 未偵測到貓/關鍵點"]
                for idx, emb_idx in enumerate(valid_indices):
                    r = self.results[emb_idx]
                    severe_cnt = sum(flag in r.suspicion_reasons for flag in core_flags)
                    # 只允許升級，不允許降級
                    r.is_suspicious = r.is_suspicious or (r.overall_score < 60) or (severe_cnt >= 2)
            # 沒偵測到的樣本，lof_score/embedding_score 維持預設 0，不加分也不扣分
        return self.results

    def save_report(self, json_path):
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'summary': {
                'total_images': len(self.results),
                'suspicious_images': sum(1 for r in self.results if r.is_suspicious),
                'suspicious_rate': sum(1 for r in self.results if r.is_suspicious) / len(self.results) if self.results else 0,
                'avg_overall_score': np.mean([r.overall_score for r in self.results]) if self.results else 0,
            },
            'detailed_results': [self._to_native(asdict(r)) for r in self.results],
            'suspicious_images': [self._to_native(asdict(r)) for r in self.results if r.is_suspicious]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON 報告已儲存: {json_path}")
        suspicious_list_path = json_path.parent / 'suspicious_images.txt'
        suspicious_count = 0
        with open(suspicious_list_path, 'w', encoding='utf-8') as f:
            f.write("可疑圖片列表 (建議人工複查)\n")
            f.write("="*70 + "\n\n")
            for r in self.results:
                if r.is_suspicious:
                    suspicious_count += 1
                    f.write(f"圖片: {Path(r.image_path).name}\n")
                    f.write(f"  評分: {r.overall_score:.1f}/100\n")
                    f.write(f"  問題:\n")
                    for reason in r.suspicion_reasons:
                        f.write(f"    - {reason}\n")
                    f.write("\n")
            if suspicious_count == 0:
                f.write("✅ 沒有發現可疑的標記！所有圖片品質良好。\n")
        print(f"✅ 可疑清單已儲存: {suspicious_list_path}")
        if suspicious_count > 0:
            self._copy_suspicious_images(json_path.parent)
        return suspicious_count
    
    def _copy_suspicious_images(self, output_dir: Path):
        """複製可疑圖片和對應的標記檔案到專門的目錄，並生成視覺化標註"""
        import shutil
        
        # 創建可疑圖片目錄結構
        suspicious_dir = output_dir / 'suspicious_samples'
        suspicious_images_dir = suspicious_dir / 'images'
        suspicious_labels_dir = suspicious_dir / 'labels'
        suspicious_vis_dir = suspicious_dir / 'visualizations'
        
        suspicious_images_dir.mkdir(parents=True, exist_ok=True)
        suspicious_labels_dir.mkdir(parents=True, exist_ok=True)
        suspicious_vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 複製可疑圖片到專門目錄...")
        
        copied_count = 0
        details_list = []
        
        for result in self.results:
            if result.is_suspicious:
                img_path = Path(result.image_path)
                
                # 複製圖片
                try:
                    dest_img = suspicious_images_dir / img_path.name
                    shutil.copy2(img_path, dest_img)
                    
                    # 複製對應的標記檔案
                    label_name = img_path.stem + '.txt'
                    
                    # 嘗試從多個可能的位置找標記檔案
                    possible_label_paths = [
                        img_path.parent.parent / 'labels' / label_name,  # dataset/labels/
                        img_path.parent / 'labels' / label_name,         # images/labels/
                        img_path.with_suffix('.txt'),                     # 同目錄
                    ]
                    
                    label_copied = False
                    for label_path in possible_label_paths:
                        if label_path.exists():
                            dest_label = suspicious_labels_dir / label_name
                            shutil.copy2(label_path, dest_label)
                            label_copied = True
                            break
                    
                    # 生成視覺化標註圖片
                    self._create_visualization(img_path, result, suspicious_vis_dir)
                    
                    # 記錄詳細資訊
                    details_list.append({
                        'filename': img_path.name,
                        'score': result.overall_score,
                        'reasons': result.suspicion_reasons,
                        'metrics': {
                            'ts_consistency': result.ts_consistency,
                            'ts_oks': result.ts_oks_score,
                            'aug_consistency': result.aug_consistency,
                            'loss_reduction': result.loss_reduction,
                            'lof_score': result.lof_score
                        }
                    })
                    
                    copied_count += 1
                    
                    if not label_copied:
                        print(f"  ⚠️  未找到標記: {img_path.name}")
                
                except Exception as e:
                    print(f"  ❌ 複製失敗 {img_path.name}: {e}")
        
        # 創建詳細說明檔案
        readme_path = suspicious_dir / 'README.txt'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("可疑樣本目錄說明\n")
            f.write("="*70 + "\n\n")
            f.write("此目錄包含所有被標記為可疑的圖片和標記檔案。\n\n")
            f.write("目錄結構:\n")
            f.write("  suspicious_samples/\n")
            f.write("  ├── images/          # 原始可疑圖片\n")
            f.write("  ├── labels/          # 對應的標記檔案 (YOLOv8 格式)\n")
            f.write("  ├── visualizations/  # 視覺化標註圖片\n")
            f.write("  ├── details.txt      # 詳細問題說明\n")
            f.write("  └── README.txt       # 本說明檔案\n\n")
            f.write("檢測方法說明:\n")
            f.write("1. Teacher-Student 不一致性\n")
            f.write("   - 使用兩個不同的模型預測同一張圖片\n")
            f.write("   - 如果預測結果差異過大，表示標記可能有問題\n\n")
            f.write("2. Augmentation Consistency\n")
            f.write("   - 對圖片進行外觀類增強（亮度、對比度、噪聲、模糊）\n")
            f.write("   - 注意：只使用不改變關鍵點語意的增強！\n")
            f.write("   - 不使用翻轉、旋轉、裁剪等空間變換\n")
            f.write("   - 如果在溫和的外觀增強下預測仍不穩定，才是真正的問題\n\n")
            f.write("3. Loss Tracking\n")
            f.write("   - 模擬訓練過程，追蹤損失變化\n")
            f.write("   - 好的標記應該能讓損失穩定下降\n\n")
            f.write("4. Pose Embedding 離群檢測\n")
            f.write("   - 使用姿態特徵檢測異常樣本\n")
            f.write("   - 找出姿態不自然的標記\n\n")
            f.write("建議處理流程:\n")
            f.write("1. 查看 details.txt 了解每張圖片的具體問題\n")
            f.write("2. 打開 visualizations/ 查看視覺化標註\n")
            f.write("3. 使用標註工具逐一檢查並修正\n")
            f.write("4. 將修正後的檔案複製回原始資料集\n")
            f.write("5. 重新執行品質檢測驗證\n\n")
            f.write(f"共 {copied_count} 個可疑樣本\n")
        
        # 創建詳細問題清單
        details_path = suspicious_dir / 'details.txt'
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write("可疑樣本詳細資訊\n")
            f.write("="*70 + "\n\n")
            
            # 按評分排序（最差的在前）
            details_list.sort(key=lambda x: x['score'])
            
            for idx, detail in enumerate(details_list, 1):
                f.write(f"【{idx}】 {detail['filename']}\n")
                f.write(f"{'─'*70}\n")
                f.write(f"綜合評分: {detail['score']:.1f}/100\n\n")
                
                f.write("問題診斷:\n")
                for reason in detail['reasons']:
                    f.write(f"  ❌ {reason}\n")
                f.write("\n")
                
                f.write("詳細指標:\n")
                metrics = detail['metrics']
                f.write(f"  • Teacher-Student 一致性: {metrics['ts_consistency']:.3f}\n")
                f.write(f"  • OKS 分數: {metrics['ts_oks']:.3f}\n")
                f.write(f"  • 增強一致性: {metrics['aug_consistency']:.3f}\n")
                f.write(f"  • 損失下降比例: {metrics['loss_reduction']:.3f}\n")
                f.write(f"  • LOF 離群分數: {metrics['lof_score']:.3f}\n")
                f.write("\n")
                
                # 給出具體建議
                f.write("建議處理:\n")
                if metrics['ts_consistency'] < 0.5:
                    f.write("  → 檢查關鍵點位置是否標記正確\n")
                if metrics['aug_consistency'] < 0.6:
                    f.write("  → 檢查是否存在遮擋或模糊區域\n")
                if metrics['loss_reduction'] < 0.3:
                    f.write("  → 標記可能與實際姿態差異較大\n")
                if metrics['lof_score'] > 1.5:
                    f.write("  → 姿態異常，可能是標記錯誤或罕見姿態\n")
                f.write("\n" + "="*70 + "\n\n")
        
        print(f"✅ 已複製 {copied_count} 個可疑樣本到: {suspicious_dir}")
        print(f"   📁 圖片: {suspicious_images_dir}")
        print(f"   📁 標記: {suspicious_labels_dir}")
        print(f"   📁 視覺化: {suspicious_vis_dir}")
        print(f"   📄 詳細說明: {details_path}")
    
    def _create_visualization(self, img_path: Path, result: PoseQualityMetrics, output_dir: Path):
        """為可疑圖片創建視覺化標註"""
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                return
            vis_image = image.copy()
            h, w = vis_image.shape[:2]
            teacher_results = self.teacher_model.predict(image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
            teacher_kpts = self._extract_keypoints(teacher_results)
            student_kpts = None
            if self.student_model is not None:
                student_results = self.student_model.predict(image, verbose=False, conf=CONF_THRESHOLD, device=self.device)[0]
                student_kpts = self._extract_keypoints(student_results)
            # 繪製 Teacher 預測（綠色）
            if teacher_kpts is not None:
                for i, (x, y, conf) in enumerate(teacher_kpts):
                    if conf > 0.3:
                        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(vis_image, f'T{i}', (int(x)+8, int(y)-8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # 繪製 Student 預測（紅色）
            if student_kpts is not None:
                for i, (x, y, conf) in enumerate(student_kpts):
                    if conf > 0.3:
                        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.putText(vis_image, f'S{i}', (int(x)+8, int(y)+8),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # 繪製差異連線（黃色虛線）
            if teacher_kpts is not None and student_kpts is not None:
                for i in range(min(len(teacher_kpts), len(student_kpts))):
                    if teacher_kpts[i, 2] > 0.3 and student_kpts[i, 2] > 0.3:
                        t_pt = (int(teacher_kpts[i, 0]), int(teacher_kpts[i, 1]))
                        s_pt = (int(student_kpts[i, 0]), int(student_kpts[i, 1]))
                        
                        # 計算距離
                        dist = np.linalg.norm(teacher_kpts[i, :2] - student_kpts[i, :2])
                        
                        # 如果差異大於閾值，用黃色標記
                        if dist > 20:
                            cv2.line(vis_image, t_pt, s_pt, (0, 255, 255), 2)
            # 添加資訊標籤
            info_bg = np.zeros((120, w, 3), dtype=np.uint8)
            
            # 繪製文字
            y_offset = 25
            cv2.putText(info_bg, f"Score: {result.overall_score:.1f}/100", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            for reason in result.suspicion_reasons[:3]:  # 最多顯示3個原因
                cv2.putText(info_bg, f"- {reason[:50]}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y_offset += 20
            
            # 添加圖例
            legend_y = 25
            cv2.circle(info_bg, (w-150, legend_y), 5, (0, 255, 0), -1)
            cv2.putText(info_bg, "Teacher", (w-135, legend_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.circle(info_bg, (w-150, legend_y+25), 5, (0, 0, 255), -1)
            cv2.putText(info_bg, "Student", (w-135, legend_y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 合併圖片
            final_image = np.vstack([vis_image, info_bg])
            
            # 儲存
            output_path = output_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(output_path), final_image)
            
        except Exception as e:
            print(f"  ⚠️  視覺化失敗 {img_path.name}: {e}")
    
    def visualize_results(self, output_dir: Path):
        """生成視覺化圖表"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 評分分布
        plt.figure(figsize=(12, 6))
        scores = [r.overall_score for r in self.results]
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=60, color='red', linestyle='--', label='可疑閾值 (60)')
        plt.xlabel('綜合評分', fontsize=12)
        plt.ylabel('圖片數量', fontsize=12)
        plt.title('標記品質評分分布', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distribution.png', dpi=150)
        plt.close()
        
        # 2. 各項指標散點圖
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = [
            ('ts_consistency', 'Teacher-Student 一致性'),
            ('aug_consistency', 'Augmentation 一致性'),
            ('loss_reduction', '損失下降比例'),
            ('lof_score', 'LOF 離群分數')
        ]
        
        for idx, (metric_name, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [getattr(r, metric_name) for r in self.results]
            colors = ['red' if r.is_suspicious else 'green' for r in self.results]
            
            ax.scatter(range(len(values)), values, c=colors, alpha=0.6, s=30)
            ax.set_xlabel('圖片索引', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.grid(alpha=0.3)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label='正常'),
                Patch(facecolor='red', label='可疑')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_scatter.png', dpi=150)
        plt.close()
        
        # 3. PCA 降維
        valid_embeddings = [e for e in self.embeddings if e is not None]
        if len(valid_embeddings) >= 10:
            try:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(valid_embeddings)
                plt.figure(figsize=(10, 8))
                # 只標註有 embedding 的樣本
                valid_results = [self.results[i] for i, e in enumerate(self.embeddings) if e is not None]
                colors = ['red' if r.is_suspicious else 'green' for r in valid_results]
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=colors, alpha=0.6, s=50)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
                plt.title('姿態 Embedding PCA 視覺化', fontsize=14)
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', label='正常'),
                    Patch(facecolor='red', label='可疑')
                ]
                plt.legend(handles=legend_elements)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'embedding_pca.png', dpi=150)
                plt.close()
            except Exception as e:
                print(f"⚠️ PCA 視覺化失敗: {e}")
        print(f"✅ 視覺化圖表已儲存至: {output_dir}")


# ==================== 互動式介面 ====================
def interactive_mode():
    """互動式模式"""
    print("\n" + "="*70)
    print("🐱 貓咪姿態標記品質自動檢測系統")
    print("="*70)
    print("\n請提供以下資訊:\n")
    
    # 獲取用戶輸入
    model_path = input("1. Teacher 模型路徑 (.pt): ").strip()
    if not os.path.exists(model_path):
        print(f"❌ 錯誤: 找不到模型 {model_path}")
        return
    
    student_path = input("2. Student 模型路徑 (選填，直接Enter跳過): ").strip()
    if student_path and not os.path.exists(student_path):
        print(f"⚠️ 警告: Student 模型不存在，將使用預設模型")
        student_path = None
    
    data_dir = input("3. 資料集目錄 (需包含 images/ 和 labels/): ").strip()
    if not os.path.isdir(data_dir):
        print(f"❌ 錯誤: 找不到目錄 {data_dir}")
        return
    
    output_path = input("4. 輸出報告路徑 (可以是目錄或 .json 檔案，預設: quality_report.json): ").strip()
    if not output_path:
        output_path = "quality_report.json"
    
    visualize = input("5. 是否生成視覺化圖表? (y/N): ").strip().lower() == 'y'
    
    device = input("6. 運算裝置 (cuda/cpu，預設: cuda): ").strip().lower()
    if device not in ['cuda', 'cpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 檢測方法選擇
    print("\n7. 選擇要啟用的檢測方法 (y/N):")
    enable_ts = input("   - Teacher-Student 不一致性檢測 (推薦): ").strip().lower() in ['y', 'yes', '']
    enable_aug = input("   - Augmentation Consistency 檢測 (較慢): ").strip().lower() in ['y', 'yes']
    enable_loss = input("   - Loss Tracking 檢測 (需要標記檔案): ").strip().lower() in ['y', 'yes']
    enable_emb = input("   - Pose Embedding 離群檢測: ").strip().lower() in ['y', 'yes', '']
    
    # 如果全部都不選，至少啟用 Teacher-Student
    if not any([enable_ts, enable_aug, enable_loss, enable_emb]):
        print("   ⚠️  未選擇任何方法，將啟用 Teacher-Student 檢測")
        enable_ts = True
    
    # 輸出目錄處理
    print("\n8. 輸出目錄處理:")
    print("   a) 覆蓋模式 (預設) - 只覆蓋同名檔案，保留其他檔案")
    print("   b) 清空模式 - 刪除所有舊的檢測結果")
    print("   c) 自動命名 - 使用時間戳記建立新目錄")
    
    output_mode = input("   選擇模式 (a/b/c，預設 a): ").strip().lower()
    
    clear_output = False
    if output_mode == 'b':
        clear_output = True
        print("   ✅ 將清空輸出目錄")
    elif output_mode == 'c':
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_output = Path(output_path)
        if original_output.suffix:
            # 是檔案路徑
            output_path = str(original_output.parent / timestamp / original_output.name)
        else:
            # 是目錄路徑
            output_path = str(original_output / timestamp)
        print(f"   ✅ 使用新目錄: {output_path}")
    else:
        print("   ✅ 將覆蓋同名檔案")
    
    print("\n" + "="*70)
    print("開始執行檢測...")
    print("="*70 + "\n")
    
    # 執行檢測
    checker = LabelQualityChecker(
        teacher_model_path=model_path,
        student_model_path=student_path if student_path else None,
        device=device,
        enable_teacher_student=enable_ts,
        enable_augmentation=enable_aug,
        enable_loss_tracking=enable_loss,
        enable_embedding=enable_emb
    )
    
    json_path, vis_dir = resolve_output_path(output_path)
    results = checker.process_dataset(
        data_dir=data_dir,
        output_path=json_path,
        visualize=visualize,
        clear_output=clear_output
    )
    # 主動存報告與圖表
    checker.save_report(json_path)
    if visualize:
        checker.visualize_results(vis_dir)
    # 顯示摘要
    print(f"\n{'='*70}")
    print("📊 檢測完成摘要")
    print(f"{'='*70}")
    print(f"✅ 總圖片數: {len(results)}")
    print(f"⚠️  可疑圖片: {sum(1 for r in results if r.is_suspicious)}")
    print(f"📈 可疑比例: {sum(1 for r in results if r.is_suspicious) / len(results) * 100:.1f}%")
    print(f"📊 平均評分: {np.mean([r.overall_score for r in results]):.1f}/100")
    print(f"\n💾 完整報告: {json_path}")
    print(f"📝 可疑列表: {vis_dir / 'suspicious_images.txt'}")
    if visualize:
        print(f"📈 視覺化圖表: {vis_dir}")
    print(f"{'='*70}\n")


# ==================== 命令列介面 ====================
def command_line_mode():
    """命令列模式"""
    parser = argparse.ArgumentParser(
        description='貓咪姿態標記品質自動檢測系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 基本用法（覆蓋模式，保留其他舊檔案）
  python label_quality_checker_complete.py --model adam.pt --data_dir dataset --output report.json
  
  # 清空模式（刪除所有舊結果）
  python label_quality_checker_complete.py --model adam.pt --data_dir dataset --output results/ --clear-output
  
  # 自動命名模式（使用時間戳記建立新目錄）
  python label_quality_checker_complete.py --model adam.pt --data_dir dataset --output results/ --auto-naming
  
  # 只使用 Teacher-Student 檢測（速度最快）
  python label_quality_checker_complete.py --model adam.pt --data_dir dataset --output report.json \
      --disable-augmentation --disable-loss-tracking --disable-embedding
  
  # 停用 Augmentation（如果覺得太慢）
  python label_quality_checker_complete.py --model adam.pt --data_dir dataset --output report.json \
      --disable-augmentation --visualize --clear-output
  
  # 完整參數範例
  python label_quality_checker_complete.py --model adam.pt --student yolo8s-pose.pt \
      --data_dir dataset --output results/ --visualize --device cuda --auto-naming
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Teacher 模型路徑 (.pt 檔案)')
    parser.add_argument('--student', type=str, default=None,
                       help='Student 模型路徑 (可選)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='資料集目錄 (需包含 images/ 和 labels/)')
    parser.add_argument('--output', type=str, default='quality_report.json',
                       help='輸出報告路徑')
    parser.add_argument('--visualize', action='store_true',
                       help='生成視覺化圖表')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='運算裝置 (cuda/cpu)')
    
    # 檢測方法開關
    parser.add_argument('--disable-teacher-student', action='store_true',
                       help='停用 Teacher-Student 檢測')
    parser.add_argument('--disable-augmentation', action='store_true',
                       help='停用 Augmentation Consistency 檢測')
    parser.add_argument('--disable-loss-tracking', action='store_true',
                       help='停用 Loss Tracking 檢測')
    parser.add_argument('--disable-embedding', action='store_true',
                       help='停用 Embedding 離群檢測')
    
    # 輸出目錄處理選項
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--clear-output', action='store_true',
                             help='清空輸出目錄（刪除舊檢測結果）')
    output_group.add_argument('--auto-naming', action='store_true',
                             help='自動命名輸出目錄（使用時間戳記）')
    
    args = parser.parse_args()
    
    # 檢查路徑
    if not os.path.exists(args.model):
        print(f"❌ 錯誤: 找不到模型檔案 {args.model}")
        return
    
    if not os.path.isdir(args.data_dir):
        print(f"❌ 錯誤: 找不到資料目錄 {args.data_dir}")
        return
    
    # 處理輸出路徑（自動命名）
    output_path = args.output
    if args.auto_naming:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_output = Path(output_path)
        if original_output.suffix:
            # 是檔案路徑
            output_path = str(original_output.parent / timestamp / original_output.name)
        else:
            # 是目錄路徑
            output_path = str(original_output / timestamp)
        print(f"📁 使用時間戳記目錄: {output_path}\n")
    
    # 執行檢測
    checker = LabelQualityChecker(
        teacher_model_path=args.model,
        student_model_path=args.student,
        device=args.device,
        enable_teacher_student=not args.disable_teacher_student,
        enable_augmentation=not args.disable_augmentation,
        enable_loss_tracking=not args.disable_loss_tracking,
        enable_embedding=not args.disable_embedding
    )
    json_path, vis_dir = resolve_output_path(output_path)
    results = checker.process_dataset(
        data_dir=args.data_dir,
        output_path=json_path,
        visualize=args.visualize,
        clear_output=args.clear_output
    )
    # 主動存報告與圖表
    checker.save_report(json_path)
    if args.visualize:
        checker.visualize_results(vis_dir)
    # 顯示摘要
    print(f"\n{'='*70}")
    print("📊 檢測完成摘要")
    print(f"{'='*70}")
    print(f"✅ 總圖片數: {len(results)}")
    print(f"⚠️  可疑圖片: {sum(1 for r in results if r.is_suspicious)}")
    print(f"📈 可疑比例: {sum(1 for r in results if r.is_suspicious) / len(results) * 100:.1f}%")
    print(f"📊 平均評分: {np.mean([r.overall_score for r in results]):.1f}/100")
    print(f"\n💾 完整報告: {json_path}")
    print(f"📝 可疑列表: {vis_dir / 'suspicious_images.txt'}")
    if args.visualize:
        print(f"📈 視覺化圖表: {vis_dir}")
    print(f"{'='*70}\n")


# ==================== 唯一來源：路徑解析 ====================
def resolve_output_path(output_path: str):
    """
    統一輸出路徑規則（唯一真相來源)

    - output_path 是 .json 檔案 → json_path=該檔案；vis_dir=該檔案 parent
    - output_path 是資料夾      → json_path=<dir>/quality_report.json；vis_dir=<dir>
    """
    p = Path(output_path)
    if p.suffix.lower() == ".json":
        json_path = p
        vis_dir = p.parent
    else:
        vis_dir = p
        json_path = p / "quality_report.json"
    vis_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return str(json_path), vis_dir


# ==================== 主程式 ====================
def main():
    """主程式入口"""
    import sys
    
    if len(sys.argv) > 1:
        # 有命令列參數，使用命令列模式
        command_line_mode()
    else:
        # 無參數，使用互動式模式
        interactive_mode()


if __name__ == "__main__":
    main()