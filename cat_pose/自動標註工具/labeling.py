#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Pose Annotator Pro - 完整修正版
严格遵循YOLOv8 Pose和Roboflow标注格式
"""

import os
import shutil
import yaml
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

 
# =====================
# 用戶可在此處直接設置模型路徑、圖片資料夾、輸出資料夾
# =====================
MODEL_PATH = r"C:/cat_pose/yo.pt"  # 修改為你的模型路徑
IMG_DIR = r"C:/cat_pose/cat_images"       # 修改為你的圖片資料夾
OUTPUT_DIR = r"C:\cat_pose\自動標註工具\human_labeling"  # 修改為你的輸出資料夾

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
    "nose", "left_eye", "right_eye",          # 0-2: 头部
    "neck", "body_mid", "tail_base",          # 3-5: 身体
    "left_shoulder", "left_front_paw",        # 6-7: 左前肢
    "right_shoulder", "right_front_paw",      # 8-9: 右前肢
    "left_hip", "left_hind_paw",              # 10-11: 左后肢
    "right_hip", "right_hind_paw",            # 12-13: 右后肢
    "tail_mid", "tail_end1", "tail_end2"      # 14-16: 尾巴
]

# 可视化参数
COLORS = {
    'skeleton': (255, 0, 255),
    'visible': (0, 255, 0),
    'occluded': (0, 255, 255),
    'invisible': (128, 128, 128),
    'selected': (0, 0, 255),
    'text': (255, 255, 255),
    'bg': (0, 0, 0)
}

POINT_RADIUS = 3
SELECTED_RADIUS = 5
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1


# =====================
# 数据标注主类
# =====================
class PoseAnnotator:
    def __init__(self, model_path, img_dir, output_dir):
        """初始化標註器"""
        self.model = YOLO(model_path)
        self.img_dir = Path(img_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录
        self.labels_dir = self.output_dir / "labels"
        self.images_dir = self.output_dir / "images"
        self.vis_dir = self.output_dir / "visualizations"
        
        for d in [self.labels_dir, self.images_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片
        self.images = sorted([
            p for p in self.img_dir.glob("*.*") 
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        if not self.images:
            raise ValueError(f"未在 {img_dir} 找到图片文件")
        
        self.total_images = len(self.images)
        self.current_idx = 0
        
        # 标注状态
        self.keypoints = np.zeros((TOTAL_KPTS, 2), np.float32)
        self.visibility = np.zeros(TOTAL_KPTS, np.int32)
        self.selected_point = 0
        self.dragging = False
        self.drag_point = -1
        
        # 缩放和平移
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_start = None
        
        # 窗口设置
        self.window_name = "YOLOv8 Pose Annotator Pro"
        self.canvas_width = 1600
        self.canvas_height = 900
        
        # 统计信息
        self.saved_count = 0
        self.saved_images = set()
        self.current_modified = False
        
        # 加载中文字体
        self.font_cn = self.load_chinese_font()
        
        # 加载第一张图片
        self.load_image(0)
    
    def load_chinese_font(self):
        """加载中文字体"""
        font_paths = [
            'C:\\Windows\\Fonts\\msyh.ttc',      # 微软雅黑 (Windows)
            'C:\\Windows\\Fonts\\simhei.ttf',    # 黑体 (Windows)
            'C:\\Windows\\Fonts\\simsun.ttc',    # 宋体 (Windows)
            '/System/Library/Fonts/PingFang.ttc', # 苹方 (macOS)
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf', # Linux
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', # 文泉驿 (Linux)
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return {
                        'large': ImageFont.truetype(font_path, 24),
                        'medium': ImageFont.truetype(font_path, 18),
                        'small': ImageFont.truetype(font_path, 14),
                        'tiny': ImageFont.truetype(font_path, 12)
                    }
                except Exception as e:
                    continue
        
        print("警告: 未找到中文字体，使用默认字体")
        return {
            'large': ImageFont.load_default(),
            'medium': ImageFont.load_default(),
            'small': ImageFont.load_default(),
            'tiny': ImageFont.load_default()
        }
    
    def load_image(self, idx):
        """加载指定索引的图片"""
        if idx < 0 or idx >= self.total_images:
            raise IndexError(f"图片索引超出范围: {idx}")
        
        self.current_idx = idx
        img_path = self.images[idx]
        
        # 读取图片
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
        
        self.original_img = img
        self.img_height, self.img_width = img.shape[:2]
        
        # 重置视图参数
        self.scale = min(self.canvas_width / self.img_width, self.canvas_height / self.img_height)
        self.offset_x = (self.canvas_width - self.img_width * self.scale) / 2
        self.offset_y = (self.canvas_height - self.img_height * self.scale) / 2
        self.selected_point = 0
        self.current_modified = False
        
        # 推理关键点
        self.infer_pose()
    
    def reset_view(self):
        """重置视图到初始状态"""
        self.scale = min(self.canvas_width / self.img_width, self.canvas_height / self.img_height)
        self.offset_x = (self.canvas_width - self.img_width * self.scale) / 2
        self.offset_y = (self.canvas_height - self.img_height * self.scale) / 2
        print("✓ 视图已重置")
    
    def infer_pose(self):
        """使用YOLOv8推理姿态"""
        results = self.model.predict(self.original_img, verbose=False)[0]
        
        # 初始化关键点
        self.keypoints = np.zeros((TOTAL_KPTS, 2), np.float32)
        self.visibility = np.zeros(TOTAL_KPTS, np.int32)
        
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            pts = results.keypoints.xy[0].cpu().numpy()
            conf = results.keypoints.conf[0].cpu().numpy()
            
            for i in range(min(TOTAL_KPTS, len(pts))):
                self.keypoints[i] = pts[i]
                # 根据置信度设置可见性
                if conf[i] > 0.6:
                    self.visibility[i] = 2  # 可见
                elif conf[i] > 0.3:
                    self.visibility[i] = 1  # 遮挡
                else:
                    self.visibility[i] = 0  # 不可见
        
        print(f"✓ 推理完成 - 可见: {np.sum(self.visibility == 2)}, 遮挡: {np.sum(self.visibility == 1)}, 不可见: {np.sum(self.visibility == 0)}")
    
    def image_to_canvas(self, x, y):
        """图像坐标转换为画布坐标"""
        cx = x * self.scale + self.offset_x
        cy = y * self.scale + self.offset_y
        return int(cx), int(cy)
    
    def canvas_to_image(self, cx, cy):
        """画布坐标转换为图像坐标"""
        x = (cx - self.offset_x) / self.scale
        y = (cy - self.offset_y) / self.scale
        return x, y
    
    def draw_canvas(self):
        """绘制标注画布"""
        # 创建黑色画布
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), np.uint8)
        
        # 缩放并放置图像
        scaled_w = int(self.img_width * self.scale)
        scaled_h = int(self.img_height * self.scale)
        
        if scaled_w > 0 and scaled_h > 0:
            resized = cv2.resize(self.original_img, (scaled_w, scaled_h))
            
            # 计算放置位置
            x1 = int(self.offset_x)
            y1 = int(self.offset_y)
            x2 = x1 + scaled_w
            y2 = y1 + scaled_h
            
            # 裁剪到画布范围内
            src_x1 = max(0, -x1)
            src_y1 = max(0, -y1)
            src_x2 = scaled_w - max(0, x2 - self.canvas_width)
            src_y2 = scaled_h - max(0, y2 - self.canvas_height)
            
            dst_x1 = max(0, x1)
            dst_y1 = max(0, y1)
            dst_x2 = min(self.canvas_width, x2)
            dst_y2 = min(self.canvas_height, y2)
            
            if src_x2 > src_x1 and src_y2 > src_y1:
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = \
                    resized[src_y1:src_y2, src_x1:src_x2]
        
        # 绘制骨架
        for a, b in SKELETON:
            if self.visibility[a] > 0 and self.visibility[b] > 0:
                pt1 = self.image_to_canvas(*self.keypoints[a])
                pt2 = self.image_to_canvas(*self.keypoints[b])
                cv2.line(canvas, pt1, pt2, COLORS['skeleton'], LINE_THICKNESS)
        
        # 绘制关键点
        for i, (x, y) in enumerate(self.keypoints):
            cx, cy = self.image_to_canvas(x, y)
            
            # 根据可见性选择颜色
            if self.visibility[i] == 2:
                color = COLORS['visible']
            elif self.visibility[i] == 1:
                color = COLORS['occluded']
            else:
                color = COLORS['invisible']
            
            # 绘制关键点
            if i == self.selected_point:
                radius = SELECTED_RADIUS
                cv2.circle(canvas, (cx, cy), radius + 1, COLORS['selected'], -1)
                cv2.circle(canvas, (cx, cy), radius, color, -1)
            else:
                radius = POINT_RADIUS
                cv2.circle(canvas, (cx, cy), radius, color, -1)
            
            # 绘制点编号
            text_pos = (cx + 6, cy - 6)
            cv2.putText(canvas, str(i), text_pos, FONT, 
                       FONT_SCALE, COLORS['text'], FONT_THICKNESS)
        
        # 绘制信息栏
        canvas = self.draw_info_panel(canvas)
        
        return canvas
    
    def draw_info_panel(self, canvas):
        """绘制信息面板（使用PIL支持中文）"""
        # 转换为PIL图像以支持中文
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 左上角信息栏
        info_y = 25
        line_height = 30
        
        # 图片进度
        info = f"图片: {self.current_idx + 1}/{self.total_images}"
        draw.text((15, info_y), info, font=self.font_cn['medium'], fill=(255, 255, 255))
        info_y += line_height
        
        # 文件名
        filename = self.images[self.current_idx].name
        if len(filename) > 35:
            filename = filename[:32] + "..."
        draw.text((15, info_y), f"档案: {filename}", font=self.font_cn['small'], fill=(255, 255, 255))
        info_y += line_height
        
        # 保存状态
        is_saved = self.current_idx in self.saved_images
        if is_saved and not self.current_modified:
            status_text = "[OK] 状态: 已保存"
            status_color = (0, 255, 0)
        elif self.current_modified:
            status_text = "[!] 状态: 已修改(未保存)"
            status_color = (255, 165, 0)
        else:
            status_text = "[X] 状态: 未保存"
            status_color = (0, 0, 255)
        
        draw.text((15, info_y), status_text, font=self.font_cn['small'], fill=status_color)
        info_y += line_height
        
        # 当前选中点
        kpt_name = KEYPOINT_NAMES[self.selected_point]
        vis_text = ['不可见', '遮挡', '可见'][self.visibility[self.selected_point]]
        draw.text((15, info_y), f"关键点 {self.selected_point}: {kpt_name} ({vis_text})", 
                 font=self.font_cn['tiny'], fill=(255, 255, 255))
        info_y += line_height
        
        # 缩放比例
        draw.text((15, info_y), f"缩放: {self.scale:.2f}x", 
                 font=self.font_cn['tiny'], fill=(255, 255, 255))
        info_y += line_height
        
        # 已保存数量
        draw.text((15, info_y), f"已保存: {self.saved_count}/{self.total_images}", 
                 font=self.font_cn['tiny'], fill=(0, 255, 0))
        
        # 左下角操作说明
        help_x = 15
        box_width = 250
        box_height = 490
        help_y = self.canvas_height - box_height - 10
        
        # 绘制半透明背景
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(help_x, help_y), (help_x + box_width, help_y + box_height)], 
                              fill=(0, 0, 0, 160))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # 操作说明文字
        text_x = help_x + 10
        text_y = help_y + 10
        
        help_texts = [
            ("=== YOLOv8 姿态 ===", (100, 200, 255), 'small', True),
            ("可见性:", (180, 180, 180), 'tiny', False),
            ("  0=不可见", (200, 200, 200), 'tiny', False),
            ("  1=遮挡", (200, 200, 200), 'tiny', False),
            ("  2=可见", (200, 200, 200), 'tiny', False),
            ("", None, None, False),
            ("=== 保存 ===", (100, 200, 255), 'small', True),
            ("[S] 保存当前", (200, 200, 200), 'tiny', False),
            ("", None, None, False),
            ("=== 导航 ===", (100, 200, 255), 'small', True),
            ("[A] 上一张", (200, 200, 200), 'tiny', False),
            ("[D] 下一张", (200, 200, 200), 'tiny', False),
            ("", None, None, False),
            ("=== 可见性 ===", (100, 200, 255), 'small', True),
            ("[0] 设为不可见", (200, 200, 200), 'tiny', False),
            ("[1] 设为遮挡", (200, 200, 200), 'tiny', False),
            ("[2] 设为可见", (200, 200, 200), 'tiny', False),
            ("", None, None, False),
            ("=== 视图 ===", (100, 200, 255), 'small', True),
            ("[R] 重置视图", (200, 200, 200), 'tiny', False),
            ("[滚轮] 缩放", (200, 200, 200), 'tiny', False),
            ("[右键] 平移", (200, 200, 200), 'tiny', False),
            ("[左键] 移动点", (200, 200, 200), 'tiny', False),
            ("", None, None, False),
            ("=== 其他 ===", (100, 200, 255), 'small', True),
            ("[I] 重新推理", (200, 200, 200), 'tiny', False),
            ("[Q] 完成导出", (200, 200, 200), 'tiny', False),
        ]
        
        for text, color, font_size, is_title in help_texts:
            if text == "":
                text_y += 6
                continue
            draw.text((text_x, text_y), text, font=self.font_cn[font_size], fill=color)
            text_y += 22 if is_title else 18
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self.canvas_to_image(x, y)
            min_dist = float('inf')
            nearest_idx = -1
            
            for i, (kx, ky) in enumerate(self.keypoints):
                dist = np.sqrt((kx - img_x)**2 + (ky - img_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
            
            threshold = 20 / self.scale
            if min_dist < threshold:
                self.selected_point = nearest_idx
                self.drag_point = nearest_idx
                self.dragging = True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.drag_point >= 0:
                img_x, img_y = self.canvas_to_image(x, y)
                img_x = np.clip(img_x, 0, self.img_width)
                img_y = np.clip(img_y, 0, self.img_height)
                self.keypoints[self.drag_point] = [img_x, img_y]
                self.current_modified = True
            
            elif self.pan_start is not None:
                dx = x - self.pan_start[0]
                dy = y - self.pan_start[1]
                self.offset_x += dx
                self.offset_y += dy
                self.pan_start = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_point = -1
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pan_start = (x, y)
        
        elif event == cv2.EVENT_RBUTTONUP:
            self.pan_start = None
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            img_x, img_y = self.canvas_to_image(x, y)
            
            if flags > 0:
                new_scale = self.scale * 1.1
            else:
                new_scale = self.scale / 1.1
            
            new_scale = np.clip(new_scale, 0.1, 10.0)
            
            self.offset_x = x - img_x * new_scale
            self.offset_y = y - img_y * new_scale
            self.scale = new_scale
    
    def save_annotation(self):
        """保存当前标注（严格按照YOLOv8 Pose格式）"""
        # 只使用可见或遮挡的关键点来计算边界框
        visible_kpts = []
        for i in range(TOTAL_KPTS):
            if self.visibility[i] > 0:
                visible_kpts.append(self.keypoints[i])
        
        if len(visible_kpts) == 0:
            visible_kpts = self.keypoints
        
        visible_kpts = np.array(visible_kpts)
        
        # 计算边界框
        x_min = visible_kpts[:, 0].min()
        y_min = visible_kpts[:, 1].min()
        x_max = visible_kpts[:, 0].max()
        y_max = visible_kpts[:, 1].max()
        
        # 归一化边界框
        bbox_center_x = ((x_min + x_max) / 2) / self.img_width
        bbox_center_y = ((y_min + y_max) / 2) / self.img_height
        bbox_width = (x_max - x_min) / self.img_width
        bbox_height = (y_max - y_min) / self.img_height
        
        # 确保在 [0, 1] 范围内
        bbox_center_x = np.clip(bbox_center_x, 0, 1)
        bbox_center_y = np.clip(bbox_center_y, 0, 1)
        bbox_width = np.clip(bbox_width, 0, 1)
        bbox_height = np.clip(bbox_height, 0, 1)
        
        # 构建标注字符串
        label_parts = [
            str(CLASS_ID),
            f"{bbox_center_x:.16f}",
            f"{bbox_center_y:.16f}",
            f"{bbox_width:.16f}",
            f"{bbox_height:.16f}"
        ]
        
        # 添加关键点
        for i in range(TOTAL_KPTS):
            kpt_x = self.keypoints[i, 0] / self.img_width
            kpt_y = self.keypoints[i, 1] / self.img_height
            
            kpt_x = np.clip(kpt_x, 0, 1)
            kpt_y = np.clip(kpt_y, 0, 1)
            
            label_parts.extend([
                f"{kpt_x:.16f}",
                f"{kpt_y:.16f}",
                str(self.visibility[i])
            ])
        
        # 保存标注文件
        img_name = self.images[self.current_idx].stem
        label_path = self.labels_dir / f"{img_name}.txt"
        
        with open(label_path, 'w') as f:
            f.write(" ".join(label_parts))
        
        # 复制图片
        img_out_path = self.images_dir / self.images[self.current_idx].name
        shutil.copy(self.images[self.current_idx], img_out_path)
        
        # 保存可视化
        vis_canvas = self.draw_canvas()
        vis_path = self.vis_dir / f"{img_name}.jpg"
        cv2.imwrite(str(vis_path), vis_canvas)
        
        # 更新保存状态
        if self.current_idx not in self.saved_images:
            self.saved_count += 1
        self.saved_images.add(self.current_idx)
        self.current_modified = False
        
        print(f"✓ 已保存: {img_name}")
        print(f"  - bbox: center=({bbox_center_x:.3f}, {bbox_center_y:.3f}), size=({bbox_width:.3f}x{bbox_height:.3f})")
        print(f"  - 可见关键点: {np.sum(self.visibility == 2)}, 遮挡: {np.sum(self.visibility == 1)}, 不可见: {np.sum(self.visibility == 0)}")
    
    def create_dataset_yaml(self):
        """创建数据集配置文件（YOLOv8格式）"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images',
            'val': 'images',
            'names': {0: CLASS_NAME},
            'kpt_shape': [TOTAL_KPTS, 3],
            'flip_idx': [
                0,  # nose
                2, 1,  # left_eye<->right_eye
                3, 4, 5,  # neck, body_mid, tail_base
                8, 9, 6, 7,  # left<->right front
                12, 13, 10, 11,  # left<->right hind
                14, 15, 16  # tail
            ]
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        # 创建README
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# YOLOv8 Pose 数据集

## 数据集信息
- 类别数: 1 ({CLASS_NAME})
- 关键点数: {TOTAL_KPTS}
- 总图片数: {self.saved_count}
- 格式: YOLOv8 Pose

## 关键点定义
""")
            for i, name in enumerate(KEYPOINT_NAMES):
                f.write(f"{i}: {name}\n")
            
            f.write(f"""
## 可见性标志
- 0: 不可见
- 1: 遮挡
- 2: 可见

## 训练命令示例
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

## 目录结构
```
{self.output_dir.name}/
├── dataset.yaml          # 数据集配置
├── README.md            # 说明文档
├── images/              # 训练图片
├── labels/              # 标注文件
└── visualizations/      # 可视化图片
```
""")
        
        print(f"\n✓ 数据集配置已保存: {yaml_path}")
        print(f"✓ 说明文档已保存: {readme_path}")
    
    def run(self):
        """运行标注器"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_width, self.canvas_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*70)
        print("YOLOv8 姿态标注工具")
        print("="*70)
        print(f"总图片数: {self.total_images}")
        print(f"输出目录: {self.output_dir}")
        print(f"关键点数量: {TOTAL_KPTS}")
        print("\n操作说明:")
        print("  S - 保存当前标注")
        print("  A - 上一张 | D - 下一张")
        print("  Q - 完成并导出")
        print("  0/1/2 - 设置可见性")
        print("  R - 重置视图 | I - 重新推理")
        print("  鼠标: 滚轮缩放 | 右键平移 | 左键移动点")
        print("="*70 + "\n")
        
        while True:
            canvas = self.draw_canvas()
            cv2.imshow(self.window_name, canvas)
            
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q') or key == 27:
                response = messagebox.askyesno("确认", 
                                              f"已保存 {self.saved_count} 张图片\n确定要完成标注吗?")
                if response:
                    break
            
            elif key == ord('s'):
                self.save_annotation()
            
            elif key == ord('d'):
                if self.current_idx < self.total_images - 1:
                    self.load_image(self.current_idx + 1)
                else:
                    print("已经是最后一张图片了!")
            
            elif key == ord('a'):
                if self.current_idx > 0:
                    self.load_image(self.current_idx - 1)
                else:
                    print("已经是第一张图片了!")
            
            elif key == ord('r'):
                self.reset_view()
            
            elif key in [ord('0'), ord('1'), ord('2')]:
                self.visibility[self.selected_point] = int(chr(key))
                self.current_modified = True
            
            elif key == ord('i'):
                self.infer_pose()
                self.current_modified = True
        
        # 關閉視窗
        cv2.destroyAllWindows()
        
        # 生成數據集配置文件
        self.create_dataset_yaml()
        
        print("\n" + "="*70)
        print("標註完成！")
        print(f"總共保存: {self.saved_count} 張圖片")
        print(f"輸出目錄: {self.output_dir}")
        print("="*70)


# =====================
# GUI 启动界面
# =====================
class AnnotatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLOv8 Pose 标注工具")
        self.root.geometry("750x600")
        self.root.resizable(True, True)
        
        # 样式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        
        # 标题
        title = ttk.Label(self.root, text="YOLOv8 Pose 标注工具", style='Title.TLabel')
        title.pack(pady=20)
        
        subtitle = ttk.Label(self.root, text="严格遵循 YOLOv8 & Roboflow 标注格式", 
                            font=('Arial', 10), foreground='gray')
        subtitle.pack()
        
        # 模型选择
        frame1 = ttk.Frame(self.root, padding=10)
        frame1.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frame1, text="模型文件 (.pt):", width=15).pack(side='left')
        self.model_var = tk.StringVar(value=MODEL_PATH)
        ttk.Entry(frame1, textvariable=self.model_var, width=45).pack(side='left', padx=5)
        ttk.Button(frame1, text="浏览", command=self.choose_model).pack(side='left')
        
        # 图片文件夹
        frame2 = ttk.Frame(self.root, padding=10)
        frame2.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frame2, text="图片文件夹:", width=15).pack(side='left')
        self.img_var = tk.StringVar(value=IMG_DIR)
        ttk.Entry(frame2, textvariable=self.img_var, width=45).pack(side='left', padx=5)
        ttk.Button(frame2, text="浏览", command=self.choose_images).pack(side='left')
        
        # 输出文件夹
        frame3 = ttk.Frame(self.root, padding=10)
        frame3.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(frame3, text="输出文件夹:", width=15).pack(side='left')
        self.output_var = tk.StringVar(value=OUTPUT_DIR)
        ttk.Entry(frame3, textvariable=self.output_var, width=45).pack(side='left', padx=5)
        ttk.Button(frame3, text="浏览", command=self.choose_output).pack(side='left')
        
        # 格式说明
        info_frame = ttk.LabelFrame(self.root, text="YOLOv8 Pose 格式说明", padding=15)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """标注格式: <class> <bbox_x> <bbox_y> <bbox_w> <bbox_h> <kpt1_x> <kpt1_y> <kpt1_vis> ...

✓ 所有坐标归一化到 [0, 1]
✓ bbox 为中心点坐标和宽高
✓ 可见性: 0=不可见, 1=遮挡, 2=可见
✓ 兼容 Roboflow 和 Ultralytics YOLOv8"""
        
        ttk.Label(info_frame, text=info_text, justify='left', 
                 foreground='#333', font=('Courier', 9)).pack()
        
        # 使用说明
        usage_frame = ttk.Frame(self.root, padding=10)
        usage_frame.pack(fill='x', padx=20, pady=5)
        
        usage_text = """使用说明:
• 模型会自动推理姿态，你只需调整和确认
• 鼠标左键拖动关键点，滚轮缩放，右键平移
• 按 S 保存，A/D 切换图片，Q 完成导出"""
        
        ttk.Label(usage_frame, text=usage_text, justify='left', 
                 foreground='gray', font=('Arial', 9)).pack()
        
        # 分隔线
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=20, pady=10)
        
        # 开始按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        start_btn = tk.Button(button_frame, 
                             text="🚀 开始标注", 
                             command=self.start,
                             font=('Arial', 12, 'bold'),
                             bg='#4CAF50',
                             fg='white',
                             activebackground='#45a049',
                             activeforeground='white',
                             padx=40,
                             pady=12,
                             cursor='hand2',
                             relief='raised',
                             borderwidth=2)
        start_btn.pack()
    
    def choose_model(self):
        path = filedialog.askopenfilename(
            title="选择YOLOv8模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if path:
            self.model_var.set(path)
    
    def choose_images(self):
        path = filedialog.askdirectory(title="选择图片文件夹")
        if path:
            self.img_var.set(path)
    
    def choose_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_var.set(path)
    
    def start(self):
        model_path = self.model_var.get()
        img_dir = self.img_var.get()
        output_dir = self.output_var.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件!")
            return
        
        if not os.path.isdir(img_dir):
            messagebox.showerror("错误", "请选择有效的图片文件夹!")
            return
        
        if not output_dir:
            messagebox.showerror("错误", "请选择输出文件夹!")
            return
        
        self.root.destroy()
        
        try:
            annotator = PoseAnnotator(model_path, img_dir, output_dir)
            annotator.run()
        except Exception as e:
            messagebox.showerror("错误", f"标注器启动失败:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = AnnotatorGUI()
    gui.run()