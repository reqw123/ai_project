#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Pose 无监督自动标注工具
基于 labeling.py 改造，自动提取特征、聚类分析并生成训练数据集
"""

import os
import cv2
import yaml
import shutil
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 机器学习库
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 配置参数
# =====================
TOTAL_KPTS = 17
CLASS_ID = 0
CLASS_NAME = "cat"

# 骨架连接定义
HEAD = [(0,1), (0,2), (1,2)]
BODY = [(0,3), (3,4), (4,5)]
FRONT = [(3,6), (6,7), (3,8), (8,9)]
HIND = [(5,10), (10,11), (5,12), (12,13)]
TAIL = [(5,14), (14,15), (15,16)]
SKELETON = HEAD + BODY + FRONT + HIND + TAIL

# 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye",
    "neck", "body_mid", "tail_base",
    "left_shoulder", "left_front_paw",
    "right_shoulder", "right_front_paw",
    "left_hip", "left_hind_paw",
    "right_hip", "right_hind_paw",
    "tail_mid", "tail_end1", "tail_end2"
]

# 可视化颜色
COLORS = {
    'skeleton': (255, 0, 255),
    'visible': (0, 255, 0),
    'occluded': (0, 255, 255),
    'invisible': (128, 128, 128),
    'text': (255, 255, 255)
}


class PoseFeatureExtractor:
    """姿态特征提取器"""
    
    @staticmethod
    def extract_geometric_features(keypoints, visibility):
        """提取几何特征（50维）"""
        features = []
        
        # 1. 关键点坐标（可见点）
        visible_mask = visibility > 0
        if np.sum(visible_mask) > 0:
            visible_kpts = keypoints[visible_mask]
            center = visible_kpts.mean(axis=0)
            
            # 归一化坐标（相对于中心点）
            for i in range(TOTAL_KPTS):
                if visibility[i] > 0:
                    rel_x = keypoints[i, 0] - center[0]
                    rel_y = keypoints[i, 1] - center[1]
                    features.extend([rel_x, rel_y])
                else:
                    features.extend([0, 0])
        else:
            features.extend([0] * (TOTAL_KPTS * 2))
        
        # 2. 骨骼长度（15条）
        for a, b in SKELETON:
            if visibility[a] > 0 and visibility[b] > 0:
                length = np.linalg.norm(keypoints[a] - keypoints[b])
                features.append(length)
            else:
                features.append(0)
        
        # 3. 角度特征（3个主要角度）
        # 颈部-身体-尾巴角度
        if all(visibility[[3,4,5]] > 0):
            v1 = keypoints[3] - keypoints[4]
            v2 = keypoints[5] - keypoints[4]
            angle = np.arccos(np.clip(np.dot(v1, v2) / 
                             (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
            features.append(angle)
        else:
            features.append(0)
        
        return np.array(features[:50])  # 确保50维
    
    @staticmethod
    def extract_motion_features(keypoints_sequence, visibility_sequence):
        """提取运动特征（30维）"""
        if len(keypoints_sequence) < 2:
            return np.zeros(30)
        
        features = []
        
        # 计算位移
        displacements = []
        for i in range(len(keypoints_sequence) - 1):
            disp = keypoints_sequence[i+1] - keypoints_sequence[i]
            visible_mask = (visibility_sequence[i] > 0) & (visibility_sequence[i+1] > 0)
            if np.sum(visible_mask) > 0:
                avg_disp = np.linalg.norm(disp[visible_mask], axis=1).mean()
                displacements.append(avg_disp)
        
        if displacements:
            features.extend([
                np.mean(displacements),
                np.std(displacements),
                np.max(displacements),
                np.min(displacements)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # 填充到30维
        while len(features) < 30:
            features.append(0)
        
        return np.array(features[:30])


class UnsupervisedPoseLabeler:
    """无监督姿态标注器"""
    
    def __init__(self, model_path, output_dir):
        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.yolo_dir = self.output_dir / "yolo_dataset"
        self.images_dir = self.yolo_dir / "images"
        self.labels_dir = self.yolo_dir / "labels"
        self.vis_dir = self.output_dir / "visualizations"
        self.analysis_dir = self.output_dir / "analysis"
        
        for d in [self.images_dir, self.labels_dir, self.vis_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.frames_data = []
        self.keypoints_history = []
        self.visibility_history = []
        self.box_history = []  # 存储检测框
        self.features = []
        
        # 分析结果
        self.clusters = None
        self.anomaly_scores = None
        self.scaler = StandardScaler()
        
        print(f"✓ 输出目录: {self.output_dir}")
    
    def process_video(self, video_path, sample_rate=5, max_frames=None):
        """处理视频提取姿态数据"""
        print(f"\n处理视频: {video_path}")
        print("="*70)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"总帧数: {total_frames}, FPS: {fps:.2f}")
        print(f"采样率: 每 {sample_rate} 帧提取一次")
        
        frame_idx = 0
        processed = 0
        
        # 用于运动特征的滑动窗口
        window_size = 5
        keypoints_window = []
        visibility_window = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # 运行推理
                results = self.model.predict(frame, verbose=False)[0]
                
                if results.keypoints is not None and len(results.keypoints.xy) > 0:
                    keypoints = results.keypoints.xy[0].cpu().numpy()
                    conf = results.keypoints.conf[0].cpu().numpy()
                    
                    # 确保17个关键点
                    if len(keypoints) >= TOTAL_KPTS:
                        keypoints = keypoints[:TOTAL_KPTS]
                        conf = conf[:TOTAL_KPTS]
                        
                        # 计算可见性
                        visibility = np.zeros(TOTAL_KPTS, dtype=np.int32)
                        for i in range(TOTAL_KPTS):
                            if conf[i] > 0.6:
                                visibility[i] = 2
                            elif conf[i] > 0.3:
                                visibility[i] = 1
                            else:
                                visibility[i] = 0
                        
                        # 更新滑动窗口
                        keypoints_window.append(keypoints)
                        visibility_window.append(visibility)
                        if len(keypoints_window) > window_size:
                            keypoints_window.pop(0)
                            visibility_window.pop(0)
                        
                        # 提取特征
                        geo_features = PoseFeatureExtractor.extract_geometric_features(
                            keypoints, visibility)
                        motion_features = PoseFeatureExtractor.extract_motion_features(
                            keypoints_window, visibility_window)
                        
                        combined_features = np.concatenate([geo_features, motion_features])
                        
                        # 存储
                        self.frames_data.append({
                            'frame_idx': frame_idx,
                            'frame': frame.copy()
                        })
                        self.keypoints_history.append(keypoints)
                        self.visibility_history.append(visibility)
                        self.features.append(combined_features)
                        
                        processed += 1
                        
                        if processed % 100 == 0:
                            print(f"  已处理: {processed} 帧")
                        
                        if max_frames and processed >= max_frames:
                            break
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\n✓ 视频处理完成")
        print(f"✓ 提取帧数: {processed}")
        print(f"✓ 特征维度: {len(self.features[0]) if self.features else 0}")
    
    def process_images(self, images_dir):
        """处理图片文件夹"""
        print(f"\n处理图片: {images_dir}")
        print("="*70)
        
        img_path = Path(images_dir)
        image_files = sorted([
            p for p in img_path.glob("*.*")
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        if not image_files:
            raise ValueError(f"未找到图片文件")
        
        print(f"找到 {len(image_files)} 张图片")
        
        keypoints_window = []
        visibility_window = []
        window_size = 5
        
        for idx, img_file in enumerate(image_files):
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
            
            results = self.model.predict(frame, verbose=False)[0]
            
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                keypoints = results.keypoints.xy[0].cpu().numpy()
                conf = results.keypoints.conf[0].cpu().numpy()
                
                # 获取检测框
                box = None
                if results.boxes is not None and len(results.boxes) > 0:
                    box = results.boxes.xyxy[0].cpu().numpy()
                
                if len(keypoints) >= TOTAL_KPTS:
                    keypoints = keypoints[:TOTAL_KPTS]
                    conf = conf[:TOTAL_KPTS]
                    
                    visibility = np.zeros(TOTAL_KPTS, dtype=np.int32)
                    for i in range(TOTAL_KPTS):
                        if conf[i] > 0.6:
                            visibility[i] = 2
                        elif conf[i] > 0.3:
                            visibility[i] = 1
                    
                    keypoints_window.append(keypoints)
                    visibility_window.append(visibility)
                    if len(keypoints_window) > window_size:
                        keypoints_window.pop(0)
                        visibility_window.pop(0)
                    
                    geo_features = PoseFeatureExtractor.extract_geometric_features(
                        keypoints, visibility)
                    motion_features = PoseFeatureExtractor.extract_motion_features(
                        keypoints_window, visibility_window)
                    
                    combined_features = np.concatenate([geo_features, motion_features])
                    
                    self.frames_data.append({
                        'filename': img_file.name,
                        'frame': frame.copy()
                    })
                    self.keypoints_history.append(keypoints)
                    self.visibility_history.append(visibility)
                    self.box_history.append(box)
                    self.features.append(combined_features)
            
            if (idx + 1) % 100 == 0:
                print(f"  已处理: {idx + 1} 张图片")
        
        print(f"\n✓ 图片处理完成")
        print(f"✓ 有效图片: {len(self.frames_data)}")
    
    def perform_clustering(self, n_clusters=5, use_dbscan=False, eps=0.5, min_samples=5):
        """执行聚类分析"""
        print(f"\n执行聚类分析...")
        print("="*70)
        
        if len(self.features) == 0:
            raise ValueError("没有特征数据，请先处理视频或图片")
        
        n_samples = len(self.features)
        print(f"样本数: {n_samples}")
        
        # 检查样本数是否足够
        if n_samples < n_clusters:
            print(f"⚠️  警告: 样本数 ({n_samples}) 少于聚类数 ({n_clusters})")
            n_clusters = max(1, n_samples)
            print(f"✓ 自动调整聚类数为: {n_clusters}")
        
        X = np.array(self.features)
        X_scaled = self.scaler.fit_transform(X)
        
        if use_dbscan:
            print(f"使用 DBSCAN (eps={eps}, min_samples={min_samples})")
            # 调整min_samples
            min_samples = min(min_samples, n_samples)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.clusters = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            n_noise = list(self.clusters).count(-1)
            print(f"✓ 发现 {n_clusters} 个聚类, {n_noise} 个噪声点")
        else:
            print(f"使用 K-means (k={n_clusters})")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = kmeans.fit_predict(X_scaled)
            print(f"✓ 聚类完成: {n_clusters} 个簇")
        
        # 打印聚类分布
        max_cluster = max(self.clusters) if len(self.clusters) > 0 else 0
        min_cluster = min(self.clusters) if len(self.clusters) > 0 else 0
        
        for i in range(min_cluster, max_cluster + 1):
            count = np.sum(self.clusters == i)
            if count > 0:  # 只显示有样本的聚类
                percent = count / len(self.clusters) * 100
                print(f"  聚类 {i}: {count} 样本 ({percent:.1f}%)")
        
        return self.clusters
    
    def detect_anomalies(self, contamination=0.1):
        """异常检测"""
        print(f"\n执行异常检测...")
        print("="*70)
        
        X = np.array(self.features)
        X_scaled = self.scaler.fit_transform(X)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        self.anomaly_scores = iso_forest.score_samples(X_scaled)
        
        n_anomalies = np.sum(anomaly_labels == -1)
        print(f"✓ 检测到 {n_anomalies} 个异常样本 ({n_anomalies/len(X)*100:.1f}%)")
        
        return anomaly_labels
    
    def visualize_analysis(self):
        """可视化分析结果"""
        print(f"\n生成分析图表...")
        print("="*70)
        
        X = np.array(self.features)
        n_samples = len(X)
        
        # 检查样本数
        if n_samples < 2:
            print(f"⚠️  样本数过少 ({n_samples})，跳过可视化分析")
            return
        
        X_scaled = self.scaler.fit_transform(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. PCA降维可视化
        n_pca_components = min(2, n_samples)
        pca = PCA(n_components=n_pca_components)
        X_pca = pca.fit_transform(X_scaled)
        
        axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1] if n_pca_components > 1 else 0, 
                          c=self.clusters, cmap='tab10', s=10, alpha=0.6)
        axes[0, 0].set_title('PCA Visualization (Colored by Cluster)', fontsize=12)
        if n_pca_components > 1:
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        else:
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('PC1 Score')
        
        # 2. t-SNE降维可视化
        if n_samples > 1 and n_samples <= 1000 and n_samples >= 5:
            perplexity = min(30, max(5, n_samples // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X_scaled)
            
            axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                             c=self.clusters, cmap='tab10', s=10, alpha=0.6)
            axes[0, 1].set_title(f't-SNE Visualization (perplexity={perplexity})', fontsize=12)
        else:
            reason = f'样本数太少 (n={n_samples})' if n_samples < 5 else f'样本数太多 (n={n_samples})'
            axes[0, 1].text(0.5, 0.5, f't-SNE 跳过\n{reason}',
                           ha='center', va='center', fontsize=14)
            axes[0, 1].set_title('t-SNE (Skipped)', fontsize=12)
        
        # 3. 聚类分布
        unique, counts = np.unique(self.clusters, return_counts=True)
        axes[1, 0].bar(unique, counts, color='steelblue')
        axes[1, 0].set_title('Cluster Distribution', fontsize=12)
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Sample Count')
        
        # 4. 异常分数分布
        if self.anomaly_scores is not None:
            axes[1, 1].hist(self.anomaly_scores, bins=50, color='coral', alpha=0.7)
            axes[1, 1].set_title('Anomaly Score Distribution', fontsize=12)
            axes[1, 1].set_xlabel('Anomaly Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        save_path = self.analysis_dir / "clustering_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 分析图表已保存: {save_path}")
    
    def export_yolo_dataset(self):
        """导出YOLOv8格式数据集"""
        print(f"\n导出YOLOv8格式数据集...")
        print("="*70)
        
        if self.clusters is None:
            print("警告: 未执行聚类分析，将所有样本标记为类别0")
            self.clusters = np.zeros(len(self.frames_data), dtype=int)
        
        saved_count = 0
        
        for idx, frame_info in enumerate(self.frames_data):
            cluster_id = self.clusters[idx]
            frame = frame_info['frame']
            h, w = frame.shape[:2]
            
            # 保存图片
            if 'frame_idx' in frame_info:
                filename = f"frame_{frame_info['frame_idx']:06d}_c{cluster_id}.jpg"
            else:
                filename = f"{Path(frame_info['filename']).stem}_c{cluster_id}.jpg"
            
            img_path = self.images_dir / filename
            cv2.imwrite(str(img_path), frame)
            
            # 获取关键点和检测框
            keypoints = self.keypoints_history[idx]
            visibility = self.visibility_history[idx]
            box = self.box_history[idx]
            
            # 使用模型检测框计算 bbox
            if box is not None:
                x1, y1, x2, y2 = box
                bbox_center_x = np.clip(((x1 + x2) / 2) / w, 0, 1)
                bbox_center_y = np.clip(((y1 + y2) / 2) / h, 0, 1)
                bbox_width = np.clip((x2 - x1) / w, 0, 1)
                bbox_height = np.clip((y2 - y1) / h, 0, 1)
            else:
                # 备选：从关键点计算
                visible_kpts = keypoints[visibility > 0]
                if len(visible_kpts) == 0:
                    visible_kpts = keypoints
                x_min = visible_kpts[:, 0].min()
                y_min = visible_kpts[:, 1].min()
                x_max = visible_kpts[:, 0].max()
                y_max = visible_kpts[:, 1].max()
                bbox_center_x = np.clip(((x_min + x_max) / 2) / w, 0, 1)
                bbox_center_y = np.clip(((y_min + y_max) / 2) / h, 0, 1)
                bbox_width = np.clip((x_max - x_min) / w, 0, 1)
                bbox_height = np.clip((y_max - y_min) / h, 0, 1)
            
            # 构建YOLO标注
            label_parts = [
                str(CLASS_ID),
                f"{bbox_center_x:.16f}",
                f"{bbox_center_y:.16f}",
                f"{bbox_width:.16f}",
                f"{bbox_height:.16f}"
            ]
            
            # 添加关键点
            for i in range(TOTAL_KPTS):
                kpt_x = np.clip(keypoints[i, 0] / w, 0, 1)
                kpt_y = np.clip(keypoints[i, 1] / h, 0, 1)
                label_parts.extend([
                    f"{kpt_x:.16f}",
                    f"{kpt_y:.16f}",
                    str(int(visibility[i]))
                ])
            
            # 保存标注
            label_path = self.labels_dir / f"{Path(filename).stem}.txt"
            with open(label_path, 'w') as f:
                f.write(" ".join(label_parts))
            
            # 保存可视化
            vis_frame = self.draw_pose_visualization(frame, keypoints, visibility, cluster_id)
            vis_path = self.vis_dir / filename
            cv2.imwrite(str(vis_path), vis_frame)
            
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"  已保存: {saved_count} 样本")
        
        print(f"\n✓ 数据集导出完成")
        print(f"✓ 总样本数: {saved_count}")
        print(f"✓ 图片路径: {self.images_dir}")
        print(f"✓ 标注路径: {self.labels_dir}")
        
        # 创建配置文件
        self._create_dataset_yaml()
    
    def draw_pose_visualization(self, frame, keypoints, visibility, cluster_id):
        """绘制姿态可视化"""
        vis_frame = frame.copy()
        
        # 绘制骨架
        for a, b in SKELETON:
            if visibility[a] > 0 and visibility[b] > 0:
                pt1 = tuple(keypoints[a].astype(int))
                pt2 = tuple(keypoints[b].astype(int))
                cv2.line(vis_frame, pt1, pt2, COLORS['skeleton'], 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            if visibility[i] == 2:
                color = COLORS['visible']
            elif visibility[i] == 1:
                color = COLORS['occluded']
            else:
                color = COLORS['invisible']
            
            cv2.circle(vis_frame, (int(x), int(y)), 3, color, -1)
        
        # 添加聚类标签
        cv2.putText(vis_frame, f"Cluster: {cluster_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis_frame
    
    def _create_dataset_yaml(self):
        """创建数据集配置文件"""
        yaml_content = {
            'path': str(self.yolo_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'names': {0: CLASS_NAME},
            'kpt_shape': [TOTAL_KPTS, 3],
            'flip_idx': [
                0, 2, 1, 3, 4, 5,
                8, 9, 6, 7,
                12, 13, 10, 11,
                14, 15, 16
            ]
        }
        
        yaml_path = self.yolo_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        # 创建README
        readme_path = self.yolo_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# YOLOv8 Pose 无监督学习数据集

## 数据集信息
- 类别: {CLASS_NAME}
- 关键点数: {TOTAL_KPTS}
- 总样本数: {len(self.frames_data)}
- 聚类数: {len(set(self.clusters)) if self.clusters is not None else 0}

## 关键点定义
""")
            for i, name in enumerate(KEYPOINT_NAMES):
                f.write(f"{i}: {name}\n")
            
            if self.clusters is not None:
                f.write(f"\n## 聚类分布\n")
                for i in range(max(self.clusters) + 1):
                    count = np.sum(self.clusters == i)
                    f.write(f"- 聚类 {i}: {count} 样本 ({count/len(self.clusters)*100:.1f}%)\n")
            
            f.write(f"""
## 训练命令
```python
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```
""")
        
        print(f"✓ 配置文件已保存: {yaml_path}")
        print(f"✓ 说明文档已保存: {readme_path}")


class AutoLabelingGUI:
    """GUI界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 Pose 训练数据生成工具")
        self.root.geometry("900x750")
        
        # 样式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        
        # 标题
        title = ttk.Label(self.root, text="YOLOv8 Pose 训练数据生成工具", 
                         style='Title.TLabel')
        title.pack(pady=15)
        
        subtitle = ttk.Label(self.root, text="自动生成可直接训练 YOLOv8 Pose 模型的数据集", 
                            font=('Arial', 10), foreground='gray')
        subtitle.pack()
        
        # 输入选择
        input_frame = ttk.LabelFrame(self.root, text="📁 输入源", padding=15)
        input_frame.pack(fill='x', padx=20, pady=10)
        
        # 模型选择
        model_frame = ttk.Frame(input_frame)
        model_frame.pack(fill='x', pady=5)
        ttk.Label(model_frame, text="模型文件:", width=12).pack(side='left')
        self.model_var = tk.StringVar(value="C:/cat_pose/640_best2.pt")
        ttk.Entry(model_frame, textvariable=self.model_var, width=50).pack(side='left', padx=5)
        ttk.Button(model_frame, text="浏览", command=self.choose_model).pack(side='left')
        
        # 输入类型选择
        type_frame = ttk.Frame(input_frame)
        type_frame.pack(fill='x', pady=5)
        ttk.Label(type_frame, text="输入类型:", width=12).pack(side='left')
        self.input_type = tk.StringVar(value="images")
        ttk.Radiobutton(type_frame, text="视频文件", variable=self.input_type, 
                       value="video").pack(side='left', padx=10)
        ttk.Radiobutton(type_frame, text="图片文件夹", variable=self.input_type,
                       value="images").pack(side='left')
        
        # 输入路径
        input_path_frame = ttk.Frame(input_frame)
        input_path_frame.pack(fill='x', pady=5)
        ttk.Label(input_path_frame, text="输入路径:", width=12).pack(side='left')
        self.input_var = tk.StringVar(value="C:/cat_pose/cat_images")
        ttk.Entry(input_path_frame, textvariable=self.input_var, width=50).pack(side='left', padx=5)
        ttk.Button(input_path_frame, text="浏览", command=self.choose_input).pack(side='left')
        
        # 输出路径
        output_frame = ttk.Frame(input_frame)
        output_frame.pack(fill='x', pady=5)
        ttk.Label(output_frame, text="输出路径:", width=12).pack(side='left')
        self.output_var = tk.StringVar(value="C:/Users/homec/auto_dataset")
        ttk.Entry(output_frame, textvariable=self.output_var, width=50).pack(side='left', padx=5)
        ttk.Button(output_frame, text="浏览", command=self.choose_output).pack(side='left')
        
        # 分析参数
        param_frame = ttk.LabelFrame(self.root, text="⚙️ 分析参数", padding=15)
        param_frame.pack(fill='x', padx=20, pady=10)
        
        # 聚类参数
        cluster_frame = ttk.Frame(param_frame)
        cluster_frame.pack(fill='x', pady=5)
        
        ttk.Label(cluster_frame, text="聚类方法:").pack(side='left', padx=5)
        self.cluster_method = tk.StringVar(value="kmeans")
        ttk.Radiobutton(cluster_frame, text="K-means", variable=self.cluster_method,
                       value="kmeans").pack(side='left', padx=5)
        ttk.Radiobutton(cluster_frame, text="DBSCAN", variable=self.cluster_method,
                       value="dbscan").pack(side='left', padx=5)
        
        ttk.Label(cluster_frame, text="聚类数:").pack(side='left', padx=(20,5))
        self.n_clusters_var = tk.IntVar(value=5)
        ttk.Spinbox(cluster_frame, from_=2, to=20, textvariable=self.n_clusters_var,
                   width=8).pack(side='left')
        
        # 采样参数
        sample_frame = ttk.Frame(param_frame)
        sample_frame.pack(fill='x', pady=5)
        
        ttk.Label(sample_frame, text="视频采样率:").pack(side='left', padx=5)
        self.sample_rate_var = tk.IntVar(value=5)
        ttk.Spinbox(sample_frame, from_=1, to=30, textvariable=self.sample_rate_var,
                   width=8).pack(side='left', padx=5)
        ttk.Label(sample_frame, text="帧 (每N帧提取一次)").pack(side='left')
        
        ttk.Label(sample_frame, text="最大帧数:").pack(side='left', padx=(20,5))
        self.max_frames_var = tk.IntVar(value=1000)
        ttk.Spinbox(sample_frame, from_=100, to=10000, increment=100,
                   textvariable=self.max_frames_var, width=10).pack(side='left', padx=5)
        
        # 功能选项
        option_frame = ttk.LabelFrame(self.root, text="📊 分析选项", padding=15)
        option_frame.pack(fill='x', padx=20, pady=10)
        
        self.perform_clustering_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="执行聚类分析", 
                       variable=self.perform_clustering_var).pack(anchor='w')
        
        self.detect_anomalies_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="执行异常检测",
                       variable=self.detect_anomalies_var).pack(anchor='w')
        
        self.visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="生成可视化图表",
                       variable=self.visualize_var).pack(anchor='w')
        
        self.export_yolo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="导出YOLOv8数据集",
                       variable=self.export_yolo_var).pack(anchor='w')
        
        # 说明
        info_frame = ttk.Frame(self.root, padding=10)
        info_frame.pack(fill='x', padx=20)
        
        info_text = """💡 工作流程:
1. 使用YOLOv8模型自动提取所有姿态关键点和可见性
2. 可选：执行聚类分析自动发现姿态模式并分类标记
3. 自动生成符合YOLOv8标准的训练数据集 (images + labels)
4. 输出 dataset.yaml 配置文件，可直接用于模型训练
5. 生成可视化图表帮助理解数据分布"""
        
        ttk.Label(info_frame, text=info_text, justify='left',
                 foreground='#555', font=('Arial', 9)).pack()
        
        # 开始按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame,
                                   text="🚀 生成训练数据集",
                                   command=self.start_analysis,
                                   font=('Arial', 14, 'bold'),
                                   bg='#4CAF50',
                                   fg='white',
                                   activebackground='#45a049',
                                   activeforeground='white',
                                   padx=50,
                                   pady=15,
                                   cursor='hand2')
        self.start_btn.pack()
    
    def choose_model(self):
        path = filedialog.askopenfilename(
            title="选择YOLOv8模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if path:
            self.model_var.set(path)
    
    def choose_input(self):
        if self.input_type.get() == "video":
            path = filedialog.askopenfilename(
                title="选择视频文件",
                filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
            )
        else:
            path = filedialog.askdirectory(title="选择图片文件夹")
        
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                # 自动设置输出路径
                if self.input_type.get() == "video":
                    base_name = Path(path).stem
                else:
                    base_name = Path(path).name
                output_path = Path(path).parent / f"{base_name}_auto_labeled"
                self.output_var.set(str(output_path))
    
    def choose_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_var.set(path)
    
    def start_analysis(self):
        """开始分析"""
        model_path = self.model_var.get()
        input_path = self.input_var.get()
        output_path = self.output_var.get()
        
        # 验证输入
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件!")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("错误", "请选择有效的输入路径!")
            return
        
        if not output_path:
            messagebox.showerror("错误", "请选择输出文件夹!")
            return
        
        # 禁用按钮
        self.start_btn.config(state='disabled', text="生成中...")
        self.root.update()
        
        try:
            # 创建标注器
            labeler = UnsupervisedPoseLabeler(model_path, output_path)
            
            # 处理输入
            if self.input_type.get() == "video":
                labeler.process_video(
                    input_path,
                    sample_rate=self.sample_rate_var.get(),
                    max_frames=self.max_frames_var.get()
                )
            else:
                labeler.process_images(input_path)
            
            # 执行分析
            if self.perform_clustering_var.get():
                try:
                    if self.cluster_method.get() == "kmeans":
                        labeler.perform_clustering(n_clusters=self.n_clusters_var.get())
                    else:
                        labeler.perform_clustering(use_dbscan=True, eps=0.5, min_samples=5)
                except Exception as e:
                    print(f"⚠️  聚类分析失败: {e}")
            
            if self.detect_anomalies_var.get():
                try:
                    labeler.detect_anomalies(contamination=0.1)
                except Exception as e:
                    print(f"⚠️  异常检测失败: {e}")
            
            if self.visualize_var.get():
                try:
                    labeler.visualize_analysis()
                except Exception as e:
                    print(f"⚠️  可视化生成失败: {e}")
            
            if self.export_yolo_var.get():
                labeler.export_yolo_dataset()
            
            messagebox.showinfo("完成", 
                              f"训练数据集生成完成!\n\n"
                              f"总样本数: {len(labeler.frames_data)}\n"
                              f"输出路径: {output_path}\n\n"
                              f"可直接使用 dataset.yaml 训练 YOLOv8 Pose 模型！")
            
        except Exception as e:
            messagebox.showerror("错误", f"数据集生成失败:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.start_btn.config(state='normal', text="🚀 生成训练数据集")
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = AutoLabelingGUI()
    gui.run()
