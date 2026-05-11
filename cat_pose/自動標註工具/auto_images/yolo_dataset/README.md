# YOLOv8 Pose 无监督学习数据集

## 数据集信息
- 类别: cat
- 关键点数: 17
- 总样本数: 1
- 聚类数: 1

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

## 聚类分布
- 聚类 0: 1 样本 (100.0%)

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
