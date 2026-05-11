# 🐱 貓咪監測系統 - ST-GCN 版本

## 系統架構設計文檔

## 三層流文件

本專案已補充三層流設計文件，請參考：

- THREE_LAYER_FLOW.md

內容包含：

- 資料流（Data Flow）
- 服務流（Service Flow）
- 模型流（Model Flow）

---

## 📊 整體系統架構

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          貓咪監測系統整體架構                              │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌────────────────────────┐
                         │   Video Source (MP4)   │
                         │   或 IP Camera        │
                         └───────────┬────────────┘
                                     │ cv2.VideoCapture()
                                     ▼
         ┌───────────────────────────────────────────────────────────┐
         │         CatMonitoringSystem (主類)                        │
         │                                                            │
         │  frame = cap.read()                                       │
         └────────┬──────────────────────────────────────────────────┘
                  │
    ┌─────────────┴─────────────┬──────────────────┐
    ▼                           ▼                  ▼
┌─────────────┐         ┌──────────────────┐  ┌──────────────┐
│  YOLO Pose  │         │  Keypoints Buf   │  │  Abnormality │
│   Detection │         │  SEQUENCE_LENGTH │  │  Detection   │
│  (17 pts)   │         │   = 32 frames    │  │  (EMA)       │
└─────────┬───┘         └────────┬─────────┘  └──────┬───────┘
          │                      │                   │
          │ kpts (17,2)          │ kpts buffer      │ activity
          │                      │ (32,17,2)        │ abnormal
          │                      │                   │
          └──────────────────────┼───────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────┐
                   │   ST-GCN 行為分類         │
                   │  (Spatial-Temporal GCN)  │
                   │                          │
                   │  Input:  (1,2,32,17)     │
                   │  Output: (1,4)  logits   │
                   │          probabilities   │
                   └────────────┬─────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
         ┌─────────────────┐      ┌─────────────────┐
         │ Behavior ID     │      │ Confidence      │
         │ (0-3)           │      │ (0.0-1.0)       │
         └────────┬────────┘      └────────┬────────┘
                  │                        │
                  └────────────┬───────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │  ImprovedBehaviorTracker             │
            │  (行為統計和追蹤)                    │
            │                                      │
            │  - 行為轉換檢測                      │
            │  - 時間累積計算                      │
            │  - 次數統計                          │
            │  - 活動力計算                        │
            └────────────┬─────────────────────────┘
                         │
         ┌───────────────┼───────────────┬──────────────┐
         ▼               ▼               ▼              ▼
    ┌────────────┐  ┌──────────┐   ┌─────────┐   ┌──────────────┐
    │ Flask Web  │  │ CSV 日誌 │   │ Node-RED│   │ 螢幕 Overlay │
    │ /stream    │  │ 記錄     │   │ 推送    │   │ (可開關)     │
    │ /status    │  │          │   │ JSON    │   │              │
    └────────────┘  └──────────┘   └─────────┘   └──────────────┘
         │
         └──► MJPEG 串流 @ 30 FPS
              JSON API @ 0.5s 間隔
```

---

## 🔄 數據流向

### 1️⃣ 幀處理流程

```
原始視頻幀 (H×W×3 BGR)
    │
    ├─► YOLO Pose 推論
    │   ├─► bounding box (x1,y1,x2,y2)
    │   └─► keypoints (17,2) + confidence (17,)
    │
    ├─► Keypoints Normalization
    │   ├─► 中心化 (關鍵點 4 為中心)
    │   ├─► 尺度正規化 (以頸部-腰部距離)
    │   └─► 標準化序列 (17,2)
    │
    ├─► Buffer 管理
    │   ├─► append to deque(maxlen=32)
    │   ├─► 當 len(buffer) >= 32 時執行推論
    │   └─► 滑動窗口處理
    │
    ├─► ST-GCN 推論
    │   ├─► tensor reshape (1,2,32,17)
    │   ├─► forward pass through ST-GCN
    │   ├─► softmax 層
    │   └─► argmax + confidence
    │
    ├─► 異常檢測
    │   ├─► 計算關鍵點位移
    │   ├─► EMA 平滑
    │   ├─► 閾值比對
    │   └─► abnormal flag
    │
    ├─► 行為追蹤更新
    │   ├─► 行為映射 (ST-GCN → tracker)
    │   ├─► 行為轉換檢測
    │   ├─► 時間累積
    │   └─► 活動力計算
    │
    └─► 多路輸出
        ├─► Flask 幀 (帶 overlay)
        ├─► CSV 寫入
        ├─► Node-RED 推送 (每 0.5s)
        └─► 即時顯示
```

### 2️⃣ ST-GCN 推論詳解

```
輸入: 32 幀的關鍵點序列 (32,17,2)
  │
  ├─► 標準化
  │   ├─► normalization (如訓練時)
  │   └─► (32,17,2) → (32,17,2)
  │
  ├─► 轉換格式
  │   ├─► (T,V,C) → (C,T,V)
  │   ├─► 添加 batch 維度
  │   └─► (1,2,32,17) tensor on GPU
  │
  ├─► ST-GCN 網絡層
  │   │
  │   ├─► Input BatchNorm
  │   │   └─► (1,2,32,17) → (1,2,32,17)
  │   │
  │   ├─► ST-GCN Block 1 (stride=1)
  │   │   ├─► SpatialGraphConv: 2 → 64
  │   │   ├─► TemporalConv: 64 → 64
  │   │   └─► (1,2,32,17) → (1,64,32,17)
  │   │
  │   ├─► ST-GCN Block 2 (stride=2)
  │   │   ├─► SpatialGraphConv: 64 → 128
  │   │   ├─► TemporalConv: 128 → 128 (stride=2)
  │   │   └─► (1,64,32,17) → (1,128,16,17)
  │   │
  │   ├─► ST-GCN Block 3-N (stride=1)
  │   │   ├─► SpatialGraphConv: 128 → 128
  │   │   ├─► TemporalConv: 128 → 128
  │   │   └─► (1,128,16,17) → (1,128,16,17)
  │   │
  │   ├─► Global Average Pooling
  │   │   └─► (1,128,16,17) → (1,128,1,1) → (1,128)
  │   │
  │   ├─► Dropout (0.5)
  │   │   └─► (1,128) → (1,128)
  │   │
  │   └─► Fully Connected
  │       └─► (1,128) → (1,4) logits
  │
  ├─► Softmax
  │   └─► logits → probabilities [p0,p1,p2,p3]
  │
  └─► 輸出
      ├─► behavior_id = argmax(probs)  # 0-3
      ├─► confidence = max(probs)      # 0-1
      └─► class_probs = probs          # [p0,p1,p2,p3]
```

### 3️⃣ 行為追蹤邏輯

```
ST-GCN behavior_id (0-3)
  │
  ├─► Mapping to Tracker Category
  │   ├─► 0 (walk) → "normal"
  │   ├─► 1 (lying) → "normal"
  │   ├─► 2 (lick) → "groom"
  │   └─► 3 (shake) → "shake"
  │
  ├─► 行為轉換檢測
  │   ├─► if new_behavior != current_behavior
  │   │   ├─► 計算上一行為持續時間
  │   │   ├─► 時間累積: behavior_time[prev] += duration
  │   │   ├─► 次數增加: behavior_count[prev] += 1
  │   │   ├─► 記錄到歷史
  │   │   └─► 更新 current_behavior
  │   │
  │   └─► else
  │       └─► 繼續積累當前行為時間
  │
  ├─► 活動力計算
  │   ├─► 計算幀間關鍵點位移
  │   ├─► activity = sum(displacement[valid_kpts]) * 10
  │   ├─► 加入活動力窗口 (maxlen=60)
  │   ├─► 取最近 1 秒的數據
  │   └─► 時間加權平均 → activity_score (0-100)
  │
  └─► 每日重置
      └─► if today != last_reset_date
          └─► 重置所有統計計數器
```

---

## 🏗 核心類別設計

### CatBehaviorSTGCN 類別

```python
CatBehaviorSTGCN
│
├─ __init__(model_path, device, sequence_length, num_classes)
│   ├─ 構建 STGCN 模型結構
│   ├─ 載入預訓練權重
│   └─ 移至指定設備 (GPU/CPU)
│
├─ normalize_keypoints(keypoints_sequence)
│   ├─ 中心化 (以關鍵點 4)
│   ├─ 尺度正規化 (以頸部-腰部距離)
│   └─ 返回標準化後的序列
│
├─ predict(keypoints_sequence)
│   ├─ 驗證序列長度
│   ├─ 標準化關鍵點
│   ├─ 格式轉換 (T,V,C) → (1,C,T,V)
│   ├─ GPU 推論
│   └─ 返回 (behavior_id, confidence, class_probs)
│
└─ __call__(keypoints_sequence)
    └─ 便利函數，呼叫 predict()
```

### CatMonitoringSystem 類別

```python
CatMonitoringSystem
│
├─ __init__(width, height)
│   ├─ 載入 YOLO 模型
│   ├─ 載入 ST-GCN 模型
│   ├─ 初始化視頻捕捉
│   ├─ 開啟 CSV 日誌檔案
│   └─ 初始化狀態變數
│
├─ process_frame(frame)
│   ├─ FPS 計算
│   ├─ YOLO 推論
│   ├─ 異常檢測 (EMA)
│   ├─ 關鍵點緩衝
│   ├─ ST-GCN 推論 (buffer 滿時)
│   ├─ 行為追蹤更新
│   ├─ 視覺化繪圖
│   ├─ CSV 記錄
│   ├─ Node-RED 推送 (0.5s 間隔)
│   └─ 返回處理後的幀
│
├─ generate_frames()
│   ├─ 無限迴圈讀取視頻
│   ├─ 呼叫 process_frame()
│   ├─ JPEG 編碼
│   └─ 產生 MJPEG 幀供 Flask 串流
│
└─ cleanup()
    ├─ 釋放視頻捕捉資源
    ├─ 關閉 CSV 檔案
    └─ 關閉 OpenCV 窗口
```

### ImprovedBehaviorTracker 類別

```python
ImprovedBehaviorTracker
│
├─ __init__()
│   ├─ behavior_time = {normal, groom, scratch, shake}
│   ├─ behavior_count = {normal, groom, scratch, shake}
│   ├─ behavior_history = deque(maxlen=100)
│   ├─ current_behavior, behavior_start_time
│   ├─ activity_window = deque(maxlen=60)
│   ├─ alerts = deque(maxlen=50)
│   └─ last_reset = today
│
├─ check_daily_reset()
│   └─ if today != last_reset_date: 重置所有統計
│
├─ map_gcn_to_tracker(behavior_id)
│   └─ 將 ST-GCN 輸出 (0-3) 映射到追蹤器類別
│
├─ update(behavior_id, activity_value)
│   ├─ 行為映射
│   ├─ 行為轉換檢測
│   ├─ 時間累積和次數計數
│   ├─ 歷史記錄
│   └─ 活動力窗口更新
│
├─ get_activity_score()
│   ├─ 取最近 1 秒的活動力數據
│   ├─ 時間加權平均
│   └─ 返回 0-100 的分數
│
├─ get_today_stats()
│   └─ 返回當日統計字典
│
├─ add_alert(alert_type, message)
│   └─ 新增警報記錄
│
└─ get_alerts()
    └─ 返回警報列表
```

---

## 📡 Node-RED 數據契約

### Push 間隔
**每 0.5 秒推送一次** (可在 config.py 中調整)

### JSON 結構保證
```json
{
  "current": {
    "text": "string",
    "emoji": "string",
    "timestamp": "HH:MM:SS"
  },
  "activity_score": "number (0-100)",
  "today_stats": {
    "normal": "int",
    "normal_time": "float",
    "groom": "int",
    "groom_time": "float",
    "scratch": "int",
    "scratch_time": "float",
    "shake": "int",
    "shake_time": "float",
    "active_time": "float",
    "rest_time": "float"
  },
  "behavior_log": [
    {
      "behavior": "string",
      "gcn_id": "int",
      "time": "HH:MM:SS",
      "duration": "float"
    }
  ],
  "alerts": [
    {
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "type": "string",
      "message": "string"
    }
  ],
  "system": {
    "ip": "string",
    "model": "string",
    "version": "string",
    "gcn_confidence": "float"
  }
}
```

### 向後相容性
- 所有欄位名稱與 LSTM 版本**完全相同**
- Node-RED 無需任何修改
- 只有 `system.model` 和 `system.version` 略有變化
- 行為映射邏輯於后端完成

---

## 🧠 ST-GCN 神經網絡結構

### 層配置

```
Layer 1: STGCNBlock (stride=1)
├─ in: (1,2,32,17)
├─ SpatialGraphConv: 2→64
├─ TemporalConv: kernel=9, stride=1
└─ out: (1,64,32,17)

Layer 2: STGCNBlock (stride=2)
├─ in: (1,64,32,17)
├─ SpatialGraphConv: 64→128
├─ TemporalConv: kernel=9, stride=2
└─ out: (1,128,16,17)

Layer 3-N: STGCNBlock (stride=1)
├─ in: (1,128,16,17)
├─ SpatialGraphConv: 128→128
├─ TemporalConv: kernel=9, stride=1
└─ out: (1,128,16,17)

Global Average Pool
├─ in: (1,128,16,17)
└─ out: (1,128)

Dropout (p=0.5)
└─ out: (1,128)

FC
├─ in: (1,128)
└─ out: (1,4) logits

Softmax (推論時)
└─ out: (1,4) probabilities
```

### 骨架圖拓撲

```
關鍵點索引 (COCO 17 點):
0: nose
1-2: eyes (left, right)
3: neck
4-5: shoulders (left, right)
6-7: elbows (left, right)
8-9: wrists (left, right)
10-11: hips (left, right)
12-13: knees (left, right)
14-15: ankles (left, right)
16: tail (貓咪特定)

圖邊連接 (17 條邊):
- 頭部三角: (0,1), (0,2), (1,2)
- 身體脊椎: (0,3), (3,4), (4,5)
- 前肢: (3,6)-(6,7), (3,8)-(8,9)
- 後肢: (5,10)-(10,11), (5,12)-(12,13)
- 尾巴: (5,14)-(14,15)-(15,16)

自環: 每個節點連接自己
結果: 17×17 鄰接矩陣，度數正規化
```

---

## ⚡ 性能最佳化

### GPU 加速
- **轉移到 GPU**: 所有 tensor 操作移至 CUDA
- **批次推論**: 可擴展以支持多貓監測
- **模型量化**: 可考慮 INT8 量化減少顯存占用

### CPU 優化
- **OpenCV 多執行緒**: frame 讀取在獨立執行緒
- **高效的 deque**: 使用 `maxlen` 自動管理記憶體
- **條件推論**: 只在 buffer 滿時執行 ST-GCN

### 記憶體管理
- **動態分配**: 關鍵點 buffer 固定大小 (32)
- **行為歷史**: 限制在 100 筆記錄
- **活動力窗口**: 限制在 60 筆記錄

---

## 🔍 調試和監控

### 啟用詳細日誌

在 `cat_monitoring_stgcn_integrated.py` 中：

```python
# 啟用 FPS 計時
print(f"FPS: {self.fps_display:.1f}")

# 啟用 ST-GCN 推論計時
import time
t0 = time.time()
behavior_id, confidence, probs = self.gcn_model.predict(seq)
print(f"GCN推論: {(time.time()-t0)*1000:.2f}ms")

# 檢查緩衝區狀態
print(f"Keypoints Buffer: {len(self.keypoints_buffer)}/{SEQUENCE_LENGTH}")

# 驗證異常檢測
print(f"EMA Motion: {self.ema_motion:.6f}, Abnormal: {abnormal}")
```

### 監控 GPU 使用

```bash
# 實時查看 GPU 使用
watch -n 0.5 nvidia-smi

# 或單次查詢
nvidia-smi
```

### 檢查 Node-RED 通訊

```bash
# 監控推送的 JSON
curl http://localhost:5000/status | jq '.'

# 檢查 Node-RED 活躍度
tail -f logs/node_red.log  # (如有)
```

---

## 📈 可擴展性設計

### 多貓監測
當前: 只追蹤第一隻貓 (results[0])

擴展方案:
```python
# 修改 YOLO 處理邏輯
for i, (kpts, box) in enumerate(zip(results.keypoints.xy, results.boxes)):
    # 為每隻貓建立獨立的 buffer 和追蹤器
    cat_id = f"cat_{i}"
    if cat_id not in trackers:
        trackers[cat_id] = CatMonitoringSystem()
    # 獨立推論和追蹤
```

### 行為類別擴展
當前: 4 類 (walk, lying, lick, shake)

添加新類別:
1. 重新訓練 ST-GCN 模型 (修改 `num_classes`)
2. 更新 `BEHAVIOR_CLASSES` 列表
3. 更新映射邏輯在 `ImprovedBehaviorTracker.map_gcn_to_tracker()`
4. 更新顯示文字和 emoji

### 異常檢測擴展
當前: EMA 動作檢測

可添加:
- 異常行為序列偵測 (unlikely behavior transitions)
- 持久異常行為警報 (lying too long)
- 多模態融合 (聲音 + 視覺)

---

## 🎓 學習資源

### ST-GCN 理解
1. 閱讀原論文: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"
2. 理解 graph convolution: https://tkipf.github.io/graph-convolutional-networks/
3. 時間卷積: https://arxiv.org/abs/1612.08242

### 優化技巧
1. 試驗不同的 `SEQUENCE_LENGTH` (24, 32, 48)
2. 調整 `EMA_ALPHA` 觀察平滑效果
3. 分析 confidence 分佈，調整決策閾值

### 部署最佳實踐
1. 使用 gunicorn/uwsgi 代替 Flask 開發伺服器
2. 實施適當的日誌輪轉機制
3. 監控 GPU 記憶體洩漏
4. 定期備份 CSV 日誌

---

**文檔版本**: v1.0  
**對應系統版本**: v4.0-stgcn  
**最後更新**: 2024 年
