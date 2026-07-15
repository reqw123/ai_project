# ST-GCN 模型設計全技術分析

更新日期：2026-06-03
來源檔案：`cat_monitoring_system/models/stgcn_model.py`、`cat_monitoring_system/tools/0_train_gcn.py`

本文件完整列出本專案 ST-GCN 採用的所有技術設計，依模型架構、特徵工程、前處理管線、訓練策略、資料增強、推論策略六大類整理，並說明每項設計的實作位置與設計動機。

---

## 一、圖拓撲設計（Graph Topology）

### 1.1 靜態鄰接矩陣（Static Adjacency Matrix）

```python
# stgcn_model.py: get_adjacency_matrix()
edges = [
    (0,1),(0,2),(1,2),           # 頭部
    (0,3),(3,4),(4,5),           # 軀幹
    (3,6),(6,7),(3,8),(8,9),     # 前肢
    (5,10),(10,11),(5,12),(12,13), # 後肢
    (5,14),(14,15),(15,16)       # 尾巴
]
```

- **類型**：固定拓撲（不可學習），基於貓體解剖結構手工定義
- **自連結**：對角線全為 1（A[i,i]=1），確保每個關節保留自身資訊
- **對稱性**：雙向邊，A[i,j] = A[j,i] = 1
- **貓體特化**：17 個關節點完全重映射自 COCO 17 點，對應鼻尖、雙耳、前胸、背中、髖部、四肢、尾巴，與人體 ST-GCN 的骨架定義不同

### 1.2 三分組分區鄰接矩陣（K=3 Partition Adjacency）

```python
# stgcn_model.py: get_stgcn_partition_adjacency()
A_root    = np.eye(17)                            # 自連結（I）
A_close   = (A > 0) - A_root                      # 1-hop 直接鄰居
A_further = clip((A² > 0) - (A > 0), 0, 1)        # 2-hop 兩步鄰居
```

| 分區 | 物理意義 | 數學定義 |
|---|---|---|
| A_root（自連結） | 保留關節自身特徵 | 單位矩陣 I |
| A_close（近鄰） | 直接相連的關節 | A - I |
| A_further（遠鄰） | 兩步可達但非直接相連 | clip(A² - A, 0, 1) |

**設計動機**：不同距離的關節對行為辨識貢獻不同（如 walk 的前後肢同步需要遠鄰傳遞），分區讓模型能對不同鄰域賦予不同權重。

### 1.3 對稱正規化（Symmetric Normalization）

```python
# stgcn_model.py: normalize_adjacency_matrix()
# D^{-1/2} A D^{-1/2}
normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
```

- 防止高度數節點（如前胸 chest，連接頭部+前肢+軀幹）主導訊息傳遞
- 訓練穩定性更高，梯度不易爆炸

---

## 二、模型架構設計（Model Architecture）

### 2.1 整體結構概覽

```
輸入 (N, C, T=16, V=17)
    │
    ├── BatchNorm2d(C)            ← 輸入正規化
    ├── JointAttention            ← 關節注意力（可開關）
    ├── Input Dropout (p=0.05)
    │
    ├── STGCNBlock 1: C→64,   stride=1
    ├── STGCNBlock 2: 64→128, stride=2  ← T 降採樣：16→8
    ├── STGCNBlock 3: 128→128, stride=1
    │
    ├── AdaptiveAvgPool2d(1,1)    ← 全局時空池化
    ├── Dropout (p=0.5)
    └── Linear(128 → 5)          ← 5 類行為分類
```

### 2.2 關節注意力機制（JointAttention）

```python
# stgcn_model.py: class JointAttention
class JointAttention(nn.Module):
    def __init__(self, in_channels):
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.conv(x)        # (N,1,T,V) — per-frame per-joint scalar
        return self.sigmoid(w)  # attention weight ∈ (0,1)

# 使用方式
bn_x = self.bn_input(x)
attn = self.joint_attention(bn_x)
x    = bn_x * attn              # 逐元素縮放
```

- **類型**：Soft Attention，可與輸入共同端對端訓練
- **作用粒度**：per-sample × per-frame × per-joint（最細粒度）
- **設計動機**：貓咪不同行為的關鍵關節不同（lick 著重頭部、walk 著重四肢、shake 著重頭頸），Attention 讓模型自動學習哪些關節更重要
- **可開關**：`USE_ATTENTION=0` 時替換為 `nn.Identity()`，用於消融對照

### 2.3 可學習的分區重要性（Learnable Partition Importance）

```python
# stgcn_model.py: class SpatialGraphConv
self.partition_importance = nn.Parameter(torch.ones(self.K, dtype=torch.float32))

# 前向傳播
x = torch.einsum('nkctv,kvw->nkctw', x, self.A)          # 圖卷積
x = x * self.partition_importance.view(1, self.K, 1, 1, 1) # 加權
x = x.sum(dim=1)
```

- 每個 STGCNBlock 都有獨立的 K=3 個可學習純量
- 讓模型訓練時自動調整「自連結 / 近鄰 / 遠鄰」三個分區的相對重要性
- 初始化為全 1，不偏向任一分區

### 2.4 多尺度時間卷積（Multi-Scale Temporal Convolution）

```python
# stgcn_model.py: class MultiScaleTemporalConv
self.branches = nn.ModuleList([
    TemporalConv(in_ch, out_ch, k=3,  stride=stride),
    TemporalConv(in_ch, out_ch, k=5,  stride=stride),
    TemporalConv(in_ch, out_ch, k=9,  stride=stride),
])
self.branch_logits = nn.Parameter(torch.zeros(3))  # 可學習分支權重

def forward(self, x):
    branch_weights = torch.softmax(self.branch_logits, dim=0)
    x = sum(w * branch(x) for w, branch in zip(branch_weights, self.branches))
    return self.out_relu(x)
```

| 卷積核 | 感受野 | 捕捉的動作特性 |
|:---:|:---:|---|
| k=3 | ~0.1 秒 | 細粒度瞬間動作（如甩頭開始） |
| k=5 | ~0.17 秒 | 中粒度動作節奏（如舔舐頻率） |
| k=9 | ~0.3 秒 | 粗粒度動作趨勢（如行走週期） |

- 分支權重透過 `softmax(branch_logits)` 正規化，三分支之和恆為 1
- 初始化 `branch_logits=0` → 初始三分支等權（各 1/3）
- 每個 STGCNBlock 有獨立的 branch_logits，不同深度可學習不同時間尺度偏好

### 2.5 殘差連結（Residual Connection）

```python
# stgcn_model.py: class STGCNBlock
# 三種殘差模式
if not residual:
    self.residual_conv = _ZeroResidual()       # 全零（殘差=關閉）
elif in_channels == out_channels and stride == 1:
    self.residual_conv = nn.Identity()         # 直通（維度不變）
else:
    self.residual_conv = nn.Sequential(        # 1×1 Conv 維度匹配
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride,1)),
        nn.BatchNorm2d(out_ch)
    )

def forward(self, x):
    res = self.residual_conv(x)
    x = self.sgc(x)
    x = self.tcn(x)
    x = x + res              # 殘差相加
    x = self.relu(x)
    x = self.dropout(x)
    return x
```

- Block 2（64→128, stride=2）使用 1×1 Conv 殘差，同時處理通道擴張與時間降採樣
- 防止深層梯度消失，允許特徵繞過部分 Block 直接傳遞

### 2.6 多層 Dropout 正則化

| 位置 | 類型 | 比例 | 作用 |
|---|---|:---:|---|
| 輸入後 | `Dropout2d` | 0.05 | 隨機遮蔽整個通道，增加輸入多樣性 |
| 每個 Block 後 | `Dropout2d` | 0.15 | 空間-時間 dropout，防止過擬合 |
| 全連接前 | `Dropout` | 0.50 | 強正則，在僅有 128 維特徵時尤為重要 |

### 2.7 全局自適應池化（AdaptiveAvgPool2d）

```python
self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
# (N, 128, T', V) → (N, 128, 1, 1) → (N, 128)
```

- 對時間維度 T' 和關節維度 V 同時做全局平均
- 輸入長度改變時不需要修改模型（推論時序列長度彈性）

---

## 三、特徵工程（Feature Engineering）

### 3.1 五種特徵通道模式

| 模式 | 通道數 | 特徵組成 |
|---|:---:|---|
| `xy` | 2 | x, y |
| `xy_conf` | 3 | x, y, conf |
| `xy_conf_v` | 5 | x, y, conf, vx, vy |
| `xy_conf_v_bone` | 7 | x, y, conf, vx, vy, bone_x, bone_y |
| `xy_conf_v_bone_bmotion` | 9 | x, y, conf, vx, vy, bone_x, bone_y, bone_mx, bone_my |

### 3.2 速度特徵（Velocity Feature）

```python
# stgcn_model.py: add_velocity_feature()
velocity = np.zeros_like(sequence)
velocity[1:] = sequence[1:] - sequence[:-1]  # 一階時間差分
```

- 在已正規化的座標上計算，確保速度尺度與位置尺度一致
- 第一幀速度為零（邊界條件）
- 對 walk（持續位移）與 shake（快速振盪）最具判別力

### 3.3 骨段向量特徵（Bone Feature）

```python
# stgcn_model.py: compute_bone_feature()
parents = [0,0,0,0, 3,4, 3,6, 3,8, 5,10, 5,12, 5,14,15]
bone = sequence - sequence[:, parents, :]  # 相對父節點的偏移向量
bone[:, 0, :] = 0.0                        # 根節點（nose）骨段為 0
```

- 每個關節指向其父節點的方向向量，編碼局部肢體方向
- 根節點（索引 0，鼻尖）定義為 0，不做相減
- 在正規化座標上計算，尺度不受體型影響

### 3.4 骨段速度特徵（Bone Motion Feature）

```python
# stgcn_model.py: compute_bone_motion_feature()
bone = compute_bone_feature(sequence)
bone_motion = np.zeros_like(bone)
bone_motion[1:] = bone[1:] - bone[:-1]    # 骨段向量的時間差分
```

- 捕捉肢體方向的變化率，對 shake（方向快速翻轉）特別有效
- 是位置速度之外的第二層時間動態資訊

### 3.5 關鍵點信心值通道（Confidence Channel）

- 直接將 YOLO 輸出的逐關節信心值作為特徵通道輸入模型
- 讓模型學習「低信心的關節座標不可靠」，自動降低其影響
- 與 `interpolate_missing` 的補點策略互補：補點後的座標信心值仍為插值前的值

---

## 四、前處理管線（Pre-processing Pipeline）

> **關鍵設計**：訓練與推論使用完全相同的前處理順序，確保分布一致性。

### 4.1 時序插值補點（interpolate_missing）

```python
# 對每個關節，在時間軸上對低信心幀做線性插值
for v in range(17):
    valid = conf[:, v] > 0.1
    seq[:, v, 0] = np.interp(idx, idx[valid], seq[valid, v, 0])
    seq[:, v, 1] = np.interp(idx, idx[valid], seq[valid, v, 1])
```

- 閾值 0.1（比推論時的 0.5 低，保留更多有效幀）
- 若整段全部低信心，設為 0（中心化後的原點）

### 4.2 左右翻轉對齊（flip_normalize）

```python
# 多數決：mid_back(4) 應在 hip(5) 右側（x 較大）
valid = (mid_back_x != 0) & (hip_x != 0)
should_flip = mean(mid_back_x[valid] < hip_x[valid]) > 0.5
# 若多數幀是反向的，對所有幀做水平翻轉
if should_flip:
    seq[t,:,0] = 2 * mid_x - seq[t,:,0]
```

- **選擇 mid_back/hip 而非 nose/tail**：鼻子常被遮蔽或消失，mid_back/hip 是穩定的軀幹中央關節
- 序列級多數決：只用兩個關節都有效的幀投票，避免遮蔽幀污染決策
- 必須在 `orientation_normalize` 之前執行

### 4.3 方向正規化（orientation_normalize）

```python
# 將 mid_back(4)→hip(5) 軸旋轉至 y 軸正向
axis = hip - mid_back
angles = arctan2(axis[:,1], axis[:,0])
rot_angles = π/2 - angles          # 目標角度為 90°（y 軸正向）
# 逐幀旋轉，以 mid_back 為旋轉中心
```

- 消除貓咪朝向不同（向左走 vs 向右走）帶來的特徵差異
- 旋轉後 mid_back 始終在 hip 上方，骨架姿態方向一致
- **選擇 mid_back→hip 軸而非 nose→hip**：nose 常被遮蔽，軀幹中段關節更穩定

### 4.4 中心化與尺度正規化（normalize_skeleton_coords）

```python
# 以 mid_back(4) 為原點中心化
seq -= seq[:, 4:5, :]

# 以胸-髖距為尺度縮放
body_size = mean(||seq[:,3,:] - seq[:,5,:]||)  # chest(3) 到 hip(5) 的平均距離
seq /= body_size
```

- 消除貓咪在畫面中的位置差異（攝影機遠近、貓的位置）
- 消除個體體型差異（大貓 vs 小貓）
- 以軀幹長度（前胸到髖部）為尺度，不受四肢姿態影響

### 4.5 前處理順序的設計理由

```
interpolate_missing  →  flip_normalize  →  orientation_normalize  →  normalize_skeleton_coords
      ↑                      ↑                      ↑                          ↑
  先補齊缺失         在原始座標做翻轉         在翻轉後旋轉方向            最後縮放位置
  才能計算翻轉       （原始座標翻轉更穩定）    （翻轉保證旋轉方向一致）    （依賴前三步的結果）
```

---

## 五、訓練策略（Training Strategy）

### 5.1 類別加權損失函數（Class-Weighted Loss）

```python
# 0_train_gcn.py
class_weights = tensor([total / (NUM_CLASSES * count_c) for c in range(5)])
criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=...)
```

- 自動計算各類別權重：稀少類別（如 shake）獲得更高損失權重
- 防止模型偏向訓練樣本多的類別

### 5.2 Label Smoothing

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ε)
# ε 由 CONFIG['LABEL_SMOOTHING'] 設定
```

- 將 one-hot 標籤軟化：目標類別機率 = 1-ε，其餘各類別機率 = ε/(K-1)
- 防止模型對訓練標籤過度自信，提升泛化能力
- 對行為邊界區段的噪聲標籤具有容忍性

### 5.3 加權隨機採樣器（WeightedRandomSampler）

```python
sample_weights = [1.0 / label_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train), replacement=True)
```

- 在每個 epoch 的 mini-batch 層級也平衡類別分布
- 與類別加權損失函數雙重解決類別不平衡

### 5.4 模型 EMA（Exponential Moving Average）

```python
# 0_train_gcn.py: class ModelEMA
# EMA 參數更新
ema_v = ema_v * 0.999 + 0.001 * model_v

# BatchNorm running stats 直接複製（不做 EMA）
if k.endswith(('running_mean', 'running_var')):
    ema_v.copy_(model_v)
```

- 維護模型參數的指數移動平均，作為評估與最終推論模型
- **BatchNorm 特殊處理**：running_mean/running_var 直接複製最新值，避免統計量滯後導致推論輸出退化
- `decay=0.999`：平均跨度約 1000 個 update 步驟

### 5.5 學習率調度（ReduceLROnPlateau）

```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
# 監控 val_loss，而非 val_accuracy（loss 更穩定）
```

- 連續 5 個 epoch val_loss 未改善時，LR 乘以 0.5
- 自適應調整，不需手動設定 milestone

### 5.6 梯度裁剪（Gradient Clipping）

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

- 防止梯度爆炸，對 RNN/GCN 類模型尤為重要
- max_norm=1.0 是較保守的設定

### 5.7 早停（Early Stopping）

```python
# patience=10：連續 10 個 epoch val_loss 未改善即停止
```

### 5.8 影片級資料切分（Video-Level Split）

```python
# 0_train_gcn.py
# 以影片為單位做 train/val 切分，而非以序列為單位
train_vids, val_vids = train_test_split(video_ids, ...)
```

- **防止資料洩漏（Data Leakage）**：滑動窗切出的相鄰序列有 50% 重疊，若以序列為單位切分，相似序列會同時出現在 train 和 val
- 強制模型在未見過的影片上泛化
- 自動修補 val 缺類：若某類別只有 1 支影片全在 train，會強制移 1 支到 val

### 5.9 優化器選擇（可配置）

| 優化器 | 特性 |
|---|---|
| Adam（預設） | 自適應學習率，收斂快 |
| AdamW | Adam + 解耦 weight decay，正則化更純粹 |
| SGD + momentum=0.9 | 收斂慢但泛化通常更好 |

---

## 六、資料增強（Data Augmentation）

> 僅在訓練階段套用，驗證/推論不使用。

### 6.1 時間增強（Temporal Augmentation）

```python
# 0_train_gcn.py: temporal_augment()
```

| 方法 | 參數 | 效果 |
|---|---|---|
| 隨機時間偏移 | max_shift=2 | 模擬不同相位的動作起始 |
| 隨機幀丟棄 | max_drop=2 | 模擬幀率不穩或跳幀 |
| 座標抖動 | jitter_std=0.01 | 模擬 YOLO 偵測雜訊 |

### 6.2 空間增強（Spatial Augmentation）

```python
# 0_train_gcn.py: spatial_augment()
```

| 方法 | 參數 | 效果 |
|---|---|---|
| 隨機旋轉 | ±12° | 模擬攝影機角度偏移 |
| 隨機縮放 | 0.90–1.10× | 模擬貓咪距離攝影機遠近 |
| 平移噪聲 | std=0.04 | 模擬正規化殘差偏移 |
| 隨機裁剪 | prob=0.30 | 模擬部分身體超出畫面 |
| 關節遮蔽 | prob=0.25, max=2 | 模擬 YOLO 低信心關節消失 |

---

## 七、推論策略（Inference Strategy）

### 7.1 滑動窗（Sliding Window）

```python
# T=16 幀，stride=16 幀（不重疊推論）
# 16 幀 × (1/30s) ≈ 0.53 秒一次推論
```

### 7.2 信心門檻回退（Confidence Threshold Fallback）

```python
confidence = max(softmax_probs)
if confidence < 0.80:
    behavior_id = LOW_CONF_ID  # 不給行為標籤，顯示「正常」
```

- 防止模型在姿態模糊時給出錯誤標籤
- 0.80 的高門檻確保只有高信心的預測才輸出

### 7.3 自動 Checkpoint 通道數推斷

```python
# 從 bn_input.weight 推斷 in_channels
self.in_channels = int(state_dict['bn_input.weight'].shape[0])

# 從 fc.weight 推斷 num_classes
ckpt_num_classes = int(state_dict['fc.weight'].shape[0])

# 從 joint_attention 權重存在與否推斷 use_attention
use_attention = any(k.startswith('joint_attention.') for k in state_dict.keys())
```

- 載入模型時自動適配訓練時的設定，不需手動指定參數

---

## 八、技術設計總表（論文用）

| 技術類別 | 設計名稱 | 實作位置 | 對應論文術語 |
|---|---|---|---|
| 圖拓撲 | 貓體特化靜態鄰接矩陣 | `get_adjacency_matrix()` | Domain-specific graph topology |
| 圖拓撲 | K=3 分區鄰接（自/近/遠鄰） | `get_stgcn_partition_adjacency()` | Spatial partition strategy |
| 圖拓撲 | 對稱度正規化 D⁻¹/²AD⁻¹/² | `normalize_adjacency_matrix()` | Symmetric normalization |
| 空間建模 | 可學習分區重要性 | `SpatialGraphConv.partition_importance` | Learnable partition importance |
| 注意力 | 關節注意力 JointAttention | `JointAttention` | Per-joint spatial attention |
| 時間建模 | 多尺度時間卷積（k=3,5,9） | `MultiScaleTemporalConv` | Multi-scale temporal convolution |
| 時間建模 | 可學習分支權重（softmax） | `branch_logits` | Adaptive temporal scale weighting |
| 特徵工程 | 位置特徵 | x, y | Spatial coordinates |
| 特徵工程 | 速度特徵 | vx, vy（一階差分） | Joint velocity |
| 特徵工程 | 骨段向量 | bone_x, bone_y | Bone vector |
| 特徵工程 | 骨段速度 | bone_mx, bone_my | Bone motion |
| 特徵工程 | 信心值通道 | conf | Keypoint confidence |
| 前處理 | 時序插值補點 | `interpolate_missing()` | Temporal interpolation |
| 前處理 | 左右翻轉對齊 | `flip_normalize()` | Horizontal flip normalization |
| 前處理 | 方向正規化 | `orientation_normalize()` | Orientation normalization |
| 前處理 | 中心化與尺度正規化 | `normalize_skeleton_coords()` | Skeleton normalization |
| 訓練 | 類別加權損失 | `CrossEntropyLoss(weight=...)` | Class-weighted loss |
| 訓練 | Label Smoothing | `label_smoothing=ε` | Label smoothing regularization |
| 訓練 | 加權隨機採樣 | `WeightedRandomSampler` | Weighted random sampling |
| 訓練 | 模型 EMA | `ModelEMA(decay=0.999)` | Exponential moving average |
| 訓練 | 梯度裁剪 | `clip_grad_norm_(max_norm=1.0)` | Gradient clipping |
| 訓練 | ReduceLROnPlateau | LR ×0.5 on plateau | Adaptive learning rate |
| 訓練 | 早停 | patience=10 | Early stopping |
| 訓練 | 影片級資料切分 | video-level train/val split，重疊視窗只在同一 split 內部產生，不跨 split | Video-level data split (prevents sliding-window leakage) |
| 訓練 | 滑動視窗步長（訓練） | `WINDOW_STRIDE=8`（50% 重疊），定義於 `stgcn_config.yaml`，與推論步長（=16）各自獨立 | Training window stride (50% overlap) |
| 訓練 | bbox 缺失過濾 | `MAX_NO_DETECT_FRAMES=2`；視窗內 YOLO 未偵測到貓的幀數超過此值則丟棄，定義於 `stgcn_config.yaml` | Bbox-based detection quality filter |
| ~~訓練~~ | ~~LABEL_PURITY_THRESHOLD~~ | ~~已移除~~：標注方式以 unannotated 作為行為間緩衝，視窗若通過 unannotated 過濾則純度必為 100%，此門檻永遠不觸發 | ~~Removed (dead code)~~ |
| 增強 | 時間偏移 / 幀丟棄 / 抖動 | `temporal_augment()` | Temporal augmentation |
| 增強 | 旋轉 / 縮放 / 平移 / 遮蔽 | `spatial_augment()` | Spatial augmentation |
| 推論 | 信心門檻回退（0.80） | `LOW_CONF_ID` | Confidence threshold fallback |
| 實驗 | 五特徵模式消融 | `ABLATION_MODES` | Feature ablation study |
| 實驗 | Attention on/off 對照 | `USE_ATTENTION` | Attention ablation |
