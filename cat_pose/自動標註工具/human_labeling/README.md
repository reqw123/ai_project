# YOLOv8 Pose 数据集

## 数据集信息
- 类别数: 1 (cat)
- 关键点数: 17
- 总图片数: 1
- 格式: YOLOv8 Pose

## 关键点定义
0: nose
1: left_eye
2: right_eye
3: neck
4: body_mid
5: tail_base
6: left_shoulder
7: left_front_paw
8: right_shoulder
9: right_front_paw
10: left_hip
11: left_hind_paw
12: right_hip
13: right_hind_paw
14: tail_mid
15: tail_end1
16: tail_end2

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
human_labeling/
├── dataset.yaml          # 数据集配置
├── README.md            # 说明文档
├── images/              # 训练图片
├── labels/              # 标注文件
└── visualizations/      # 可视化图片
```
