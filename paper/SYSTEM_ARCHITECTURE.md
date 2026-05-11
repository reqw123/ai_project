"""
系統架構與模組設計文檔
"""

# ============================================================================
# 系統架構與模組設計
# ============================================================================

## 整體架構圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                    幾何行為分析系統架構圖                             │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  影片輸入        │
    │  (mp4/avi/mov)  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────────────────────────┐
    │  YOLO-Pose 關鍵點檢測                │
    │  ├─ J=17 個身體關鍵點                │
    │  ├─ 置信度評分                       │
    │  └─ 邊界框 (可選)                    │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │  幾何計算層 (geometry/)                      │
    │  ├─ BodyCoordinateSystem                     │
    │  │  ├─ 身體軸線計算 (chest→hip)             │
    │  │  ├─ 法線计算 (垂直於軸線)                │
    │  │  └─ 相對座標轉換                         │
    │  │                                          │
    │  ├─ RegionDefinition                        │
    │  │  ├─ 身體中心橢圓 (BODY_CENTER)          │
    │  │  ├─ 四肢長條 (LIMB_FL/FR/HL/HR)        │
    │  │  ├─ 鼻子接觸區 (NOSE_CONTACT)          │
    │  │  └─ 自適應區域大小調整                  │
    │  │                                          │
    │  └─ DirectionStabilizer                     │
    │     ├─ 多幀方向穩定化                      │
    │     ├─ 翻轉檢測與修正                      │
    │     └─ 方向信心度計算                      │
    │                                             │
    └────────┬────────────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────────┐
    │  特徵工程層 (feature.py)          │
    │  ├─ 位置特徵                      │
    │  │  ├─ nose_body_u               │
    │  │  ├─ nose_body_v               │
    │  │  └─ 正規化處理                │
    │  │                               │
    │  ├─ 距離特徵                      │
    │  │  ├─ dist_to_body_center       │
    │  │  ├─ dist_to_limb_*            │
    │  │  └─ 正規化至 body_length      │
    │  │                               │
    │  ├─ 接觸特徵 (boolean)            │
    │  │  ├─ in_body_center            │
    │  │  ├─ in_limb_fl/fr/hl/hr       │
    │  │  └─ 區域交集判定              │
    │  │                               │
    │  ├─ 方向特徵                      │
    │  │  ├─ head_dir_u/v              │
    │  │  ├─ head_direction_angle      │
    │  │  └─ 頭部指向判定              │
    │  │                               │
    │  └─ 舔舐區域判定                  │
    │     ├─ primary_lick_zone         │
    │     └─ primary_lick_zone_conf    │
    │                                  │
    └────────┬───────────────────────────┘
             │ ← 特徵向量 (17-D)
             │
             ▼
    ┌──────────────────────────────────────┐
    │  行為分析層 (behavior.py)            │
    │  ├─ Instantaneous Judgment         │
    │  │  └─ 單幀舔舐判定                │
    │  │                                 │
    │  ├─ Temporal Confirmation          │
    │  │  ├─ 多幀確認機制                │
    │  │  ├─ 狀態轉移機                  │
    │  │  ├─ 信心度管理                  │
    │  │  └─ 允許間隔容忍                │
    │  │                                 │
    │  └─ 行為標籤輸出                    │
    │     ├─ BehaviorLabel               │
    │     ├─ duration_frames             │
    │     ├─ confidence                  │
    │     └─ zone_code                   │
    │                                    │
    └────────┬────────────────────────────┘
             │ ← 確認行為
             │
    ┌────────┴──────────┬────────────────────┐
    ▼                   ▼                    ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│  CSV 輸出   │  │  可視化      │  │  ST-GCN 輸入     │
│             │  │  (Display)   │  │  (Model Input)   │
│ 結構化特徵  │  │              │  │                  │
│ + 行為標籤  │  │ Frame 顯示   │  │ 特徵序列         │
│             │  │ ├─ 坐標系    │  │ (T×17)           │
│             │  │ ├─ 區域      │  │ + 骨架序列       │
│             │  │ ├─ 軌跡      │  │ 用於行為分類     │
│             │  │ └─ 行為標籤  │  │                  │
└─────────────┘  └──────────────┘  └──────────────────┘
```

## 模組詳細說明

### 1. geometry/ (幾何計算子包)

```
geometry/
├── __init__.py
├── body_coordinate_system.py          [核心模組 1]
│   ├── BodyFrameData
│   │   ├─ 單幀身體座標系資料
│   │   ├─ keypoints, keypoint_conf
│   │   ├─ chest, hip, nose
│   │   ├─ body_axis_unit, body_normal
│   │   └─ body_relative_coords
│   │
│   └── BodyCoordinateSystem
│       ├─ __init__(chest_conf_threshold, ...)
│       ├─ compute_frame_geometry()
│       ├─ get_relative_position()       [螢幕→身體座標]
│       ├─ get_pixel_position()          [身體→螢幕座標]
│       └─ compute_head_direction_vector()
│
├── region_definition.py                [核心模組 2]
│   ├── RegionType (Enum)
│   │   ├─ BODY_CENTER
│   │   ├─ LIMB_FL/FR/HL/HR
│   │   └─ NOSE_CONTACT
│   │
│   ├── EllipseRegion (dataclass)
│   │   ├─ center, radius_u, radius_v
│   │   └─ axis_u, axis_v
│   │
│   ├── StripRegion (dataclass)
│   │   ├─ p_start, p_end
│   │   ├─ half_width
│   │   ├─ axis_unit, normal
│   │   └─ corners
│   │
│   └── RegionDefinition
│       ├─ compute_all_regions()
│       ├─ compute_body_center_region()
│       ├─ compute_limb_region()
│       ├─ compute_nose_contact_region()
│       ├─ point_in_ellipse()            [判定點在橢圓內]
│       ├─ point_in_strip()              [判定點在長條內]
│       └─ point_distance_to_strip()     [點到長條距離]
│
└── direction_stabilizer.py             [核心模組 3]
    ├── DirectionStabilizer
    │   ├─ __init__(window_size, flip_threshold)
    │   ├─ update()
    │   ├─ _detect_flip()                [翻轉檢測]
    │   ├─ _update_stable_direction()    [穩定方向更新]
    │   ├─ correct_left_right_labels()   [左右判定]
    │   └─ get_statistics()
```

### 2. feature.py (特徵工程)

```
feature.py
├── LickZoneType (Enum)
│   ├─ NO_TARGET (-1)
│   ├─ BODY_CENTER (0)
│   ├─ LIMB_FL/FR/HL/HR (1-4)
│
├── GeometricFeatures (dataclass)
│   ├─ 基本資訊
│   │   ├─ frame_idx, timestamp
│   │   ├─ is_valid
│   │
│   ├─ 位置特徵
│   │   ├─ nose_body_u, nose_body_v
│   │
│   ├─ 距離特徵
│   │   ├─ dist_to_body_center
│   │   ├─ dist_to_limb_fl/fr/hl/hr
│   │
│   ├─ 接觸特徵 (boolean)
│   │   ├─ in_body_center
│   │   ├─ in_limb_fl/fr/hl/hr
│   │
│   ├─ 方向特徵
│   │   ├─ head_dir_u, head_dir_v
│   │   ├─ head_direction_angle
│   │
│   ├─ 舔舐區域
│   │   ├─ primary_lick_zone
│   │   ├─ primary_lick_zone_confidence
│   │
│   ├─ 信心度
│   │   ├─ chest_conf, hip_conf, nose_conf
│   │   ├─ overall_confidence
│   │   ├─ direction_confidence
│   │
│   └─ 方法
│       ├─ from_frame_data()             [從 BodyFrameData 計算]
│       ├─ to_feature_vector()           [轉為數值向量]
│       ├─ to_dict()                     [轉為字典 (CSV)]
│
└── FeatureExtractor
    ├─ __init__(region_def)
    └─ extract(frame_data, regions)   [抽取單幀特徵]
```

### 3. behavior.py (行為分析)

```
behavior.py
├── BehaviorState (Enum)
│   ├─ IDLE
│   ├─ LICKING
│   ├─ SCRATCHING
│   ├─ EXPLORING
│   ├─ UNKNOWN
│
├── BehaviorLabel (dataclass)
│   ├─ primary_zone
│   ├─ state
│   ├─ entry_frame, duration_frames
│   ├─ confidence
│   ├─ first_detection_frame, last_update_frame
│   └─ is_active()
│
├── TemporalContext (dataclass)
│   ├─ 參數
│   │   ├─ min_frames_for_confirmation
│   │   ├─ max_gap_frames
│   │   ├─ decay_rate
│   │
│   └─ 狀態
│       ├─ feature_history
│       ├─ zone_history
│       ├─ state_history
│       ├─ current_frame
│       ├─ confirmed_zone
│
└── BehaviorAnalyzer
    ├─ __init__(min_frames, max_gap, decay_rate)
    ├─ update(frame_idx, features)      [主要方法]
    │   └─ 返回 (confirmed_behavior, instantaneous_conf)
    ├─ _instant_judgment(features)      [單幀判斷]
    ├─ _temporal_confirmation()          [時間確認]
    ├─ get_current_state_summary()
    └─ reset()
```

### 4. visualization_enhanced.py (可視化)

```
visualization_enhanced.py
└── EnhancedVisualizer
    ├─ __init__(enable_drawing)
    ├─ draw_all(frame, frame_data, regions, features, behavior)
    ├─ _draw_body_coordinate_system()   [坐標系]
    ├─ _draw_regions()                  [區域]
    ├─ _draw_ellipse_region()
    ├─ _draw_strip_region()
    ├─ _draw_nose_trajectory()          [軌跡]
    ├─ _draw_feature_info()             [特徵面板]
    └─ _draw_behavior_info()            [行為標籤]
```

### 5. geometric_behavior_pipeline.py (整合管道)

```
geometric_behavior_pipeline.py
└── GeometricBehaviorAnalysisPipeline
    ├─ __init__(yolo_model_path, output_csv_path, ...)
    ├─ process_video(video_path, max_frames, display_window)
    ├─ _init_csv_file()
    ├─ _write_features_to_csv()
    ├─ get_statistics()
    ├─ close()
    │
    └─ 內部元件
        ├─ keypoint_detector (KeypointDetector)
        ├─ body_coord_system (BodyCoordinateSystem)
        ├─ region_def (RegionDefinition)
        ├─ direction_stabilizer (DirectionStabilizer)
        ├─ feature_extractor (FeatureExtractor)
        ├─ behavior_analyzer (BehaviorAnalyzer)
        └─ visualizer (EnhancedVisualizer)
```

## 數據流示例

```
第 t 幀的數據流：

1. 輸入: frame (H×W×3 RGB)

2. YOLO 檢測
   kpts (17×2), kpt_conf (17×), bbox, conf
   ↓

3. 身體座標系計算
   BodyFrameData {
     keypoints, keypoint_conf,
     chest, hip, nose,
     body_axis_unit, body_normal,
     body_length,
     body_relative_coords
   }
   ↓

4. 方向穩定化
   DirectionStabilizer.update() →
   (flip_detected, dir_confidence)
   ↓

5. 區域計算
   regions {
     BODY_CENTER: EllipseRegion,
     LIMB_FL/FR/HL/HR: StripRegion,
     NOSE_CONTACT: EllipseRegion
   }
   ↓

6. 特徵抽取
   GeometricFeatures {
     nose_body_u, nose_body_v,
     dist_to_*,
     in_body_center, in_limb_*,
     head_dir_u/v, head_direction_angle,
     overall_confidence,
     primary_lick_zone
   }
   ↓

7. 行為分析
   BehaviorAnalyzer.update() →
   (confirmed_behavior, instantaneous_conf)
   ↓

8. 輸出
   a) CSV 記錄
   b) 顯示幀 (可視化)
   c) 行為標籤 (用於後續)
```

## 關鍵參數一覽表

| 參數 | 預設值 | 說明 |
|------|--------|------|
| CHEST_CONF_THRESHOLD | 0.3 | 胸部關鍵點信心門檣 |
| HIP_CONF_THRESHOLD | 0.3 | 臀部關鍵點信心門檣 |
| NOSE_CONF_THRESHOLD | 0.25 | 鼻子關鍵點信心門檣 |
| MIN_BODY_LENGTH_PX | 5.0 | 最小身體長度 |
| BODY_CENTER_WIDTH_RATIO | 0.65 | 身體中心寬度比 |
| BODY_CENTER_HEIGHT_RATIO | 0.27 | 身體中心高度比 |
| LIMB_HALF_WIDTH_RATIO | 0.05 | 四肢寬度比 |
| LIMB_HALF_WIDTH_MIN_PX | 5.0 | 四肢最小寬度 |
| NOSE_CONTACT_WIDTH_RATIO | 0.16 | 鼻子接觸寬度比 |
| NOSE_CONTACT_HEIGHT_RATIO | 0.16 | 鼻子接觸高度比 |
| DIR_WINDOW_SIZE | 5 | 方向穩定化窗口 |
| DIR_FLIP_THRESHOLD | 0.7 | 翻轉判定門檣 |
| MIN_FRAMES_CONFIRMATION | 3 | 行為確認最小幀數 |
| MAX_GAP_FRAMES | 1 | 允許的最大間隔 |
| CONFIDENCE_DECAY_RATE | 0.1 | 置信度衰減率 |

## 擴展性設計

系統模組化設計使得易於擴展：

1. **替換關鍵點檢測器**
   - 改變 KeypointDetector，使用其他模型
   - 只需保證輸出格式相同

2. **調整區域參數**
   - 修改 BODY_CENTER_WIDTH_RATIO 等常數
   - 或在 RegionDefinition 中添加新區域類型

3. **添加新行為類型**
   - 在 BehaviorState 中添加新枚舉值
   - 在 BehaviorAnalyzer 中添加判定邏輯

4. **集成到其他框架**
   - 提取 feature.to_feature_vector() 直接餵入模型
   - 或使用 behavior 標籤進行訓練

## 性能優化建議

1. **GPU 加速**
   - YOLO 推論已在 GPU 上
   - 幾何計算可用 CuPy/Numba 加速

2. **批處理**
   - 多個視訊並行處理
   - 減少 I/O 開銷

3. **剪枝**
   - 跳過信心度過低的幀
   - 早期退出檢測

4. **緩存**
   - 預計算常用參數
   - 減少重複計算
"""

# 此為文本格式的架構文檔
