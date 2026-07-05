# ST-GCN 技術文獻對照報告
## 貓咪行為辨識系統 — 採用 / 未採用技術 × 文獻溯源

更新日期：2026-06-30  
對應實作：`cat_monitoring_system/models/stgcn_model.py`、`cat_monitoring_system/stgcn_config.yaml`、`cat_monitoring_system/0_train_gcn.py`

---

## 一、基礎架構

### 1.1 原始 ST-GCN

| 項目 | 內容 |
|------|------|
| **文獻** | Yan et al., *"Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"*, AAAI 2018 |
| **核心貢獻** | 將骨架序列建模為時空圖，交替做空間圖卷積（捕捉關節關係）與時間卷積（捕捉動作演化） |
| **本專案採用** | 整體 STGCNBlock 架構（Spatial GCN → Temporal Conv → 殘差相加）完全沿用此框架 |

### 1.2 圖的對稱正規化 D⁻¹/²AD⁻¹/²

| 項目 | 內容 |
|------|------|
| **文獻** | Kipf & Welling, *"Semi-Supervised Classification with Graph Convolutional Networks"*, ICLR 2017 |
| **核心貢獻** | 對稱正規化消除高度數節點（如前胸，連接頭部+前肢+軀幹）對訊息傳遞的主導效應，並提升訓練穩定性 |
| **本專案採用** | `normalize_adjacency_matrix()` 對三個 partition 矩陣各自正規化 |

### 1.3 K=3 空間分區策略（root / close / further）

| 項目 | 內容 |
|------|------|
| **文獻** | Yan et al., AAAI 2018（同上） |
| **三分區定義** | A_root（自連結 I）、A_close（1-hop 鄰居）、A_further（2-hop 且非直接相連） |
| **本專案採用** | `get_stgcn_partition_adjacency()`；三個分區各對應一組 Conv 權重，再加權融合 |

### 1.4 殘差連結（Residual Connection）

| 項目 | 內容 |
|------|------|
| **文獻** | He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016 |
| **本專案採用** | `STGCNBlock` 三模式殘差：恆等映射 / 零殘差 / 1×1 Conv 維度匹配（通道擴張或 stride 降採樣時） |

---

## 二、已採用的現代改良技術

### 2.1 可學習分區重要性（Learnable Partition Importance）

| 項目 | 內容 |
|------|------|
| **文獻** | Shi et al., *"Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition"*, CVPR 2019 (**2s-AGCN**) |
| **原論文做法** | 除固定鄰接矩陣外，加入可學習的殘差鄰接矩陣 B（adaptive graph），以 A+B 做卷積 |
| **本專案採用** | 簡化版：每個 partition 配一個可學習純量 `partition_importance`（`nn.Parameter`），訓練時自動調整三分區的相對權重，不修改鄰接矩陣本身 |
| **實作位置** | `SpatialGraphConv.partition_importance`，`einsum` 後乘 `partition_importance` 再 `sum(dim=1)` |

### 2.2 多尺度時間卷積（Multi-Scale Temporal Convolution）

| 項目 | 內容 |
|------|------|
| **文獻** | Liu et al., *"Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition"*, CVPR 2020 (**MS-G3D**) |
| **原論文做法** | 同時使用多個時間 kernel size 的 dilated/non-dilated 卷積分支，捕捉短中長時間尺度動作特徵 |
| **本專案採用** | 三個平行 `TemporalConv`（k=3, 5, 9），以可學習 `branch_logits`（初始化為 0）經 softmax 加權融合，每個 STGCNBlock 有獨立的分支權重 |
| **感受野** | k=3 ≈ 0.1s（細粒度）/ k=5 ≈ 0.17s（中）/ k=9 ≈ 0.3s（粗粒度，walk 一步週期） |
| **實作位置** | `MultiScaleTemporalConv`；`branch_logits` 初始化為全零 → 三分支初始等權各 1/3 |

### 2.3 關節空間注意力（Joint Spatial Attention）

| 項目 | 內容 |
|------|------|
| **最近文獻（空間注意力）** | Woo et al., *"CBAM: Convolutional Block Attention Module"*, ECCV 2018 |
| **骨架領域文獻** | Zhang et al., *"Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition"*, CVPR 2020 (**AAGCN**) |
| **骨架領域文獻** | Si et al., *"An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition"*, CVPR 2019 |
| **本專案實作** | 1×1 Conv2d(C→1) + Sigmoid，輸出 (N,1,T,V)，對 BN 後的輸入逐元素縮放 |
| **作用機制** | 讓模型自動學習「哪個時間步哪個關節更重要」，抑制低信心或不相關關節的訊號 |
| **可開關** | `USE_ATTENTION` 控制；載入 checkpoint 時從 `joint_attention.*` key 存在與否自動判定 |

#### 2.3.1 JointAttention 詳細設計

```
輸入 x: (N, C, T, V)
   ↓
bn_input  →  BN 正規化後的 bn_x: (N, C, T, V)
   ↓
JointAttention:
   Conv2d(C, 1, kernel=1)  →  (N, 1, T, V)
   Sigmoid                 →  每格值 ∈ (0, 1)
   ↓
bn_x × attn  (broadcast along C dim)
   ↓
input_dropout → 進入 STGCNBlock 堆疊
```

#### 2.3.2 JointAttention 的限制與比較

| 面向 | 本實作 JointAttention | Graph Attention (GAT 風格) | Transformer Self-Attention |
|------|----------------------|--------------------------|---------------------------|
| **作用位置** | 僅輸入層一次 | 每個 GCN 層皆有 | 每個 Block 皆有 |
| **關節間互動** | 無（每關節獨立算 weight） | 有（edge-level query-key） | 有（全域 query-key dot product） |
| **時間維度** | 有（T 維度輸出） | 通常只空間 | 有（temporal self-attention）|
| **計算量** | 極輕（1×1 Conv） | 中（O(V²) per layer） | 重（O(T²V²) full attention） |
| **文獻對應** | CBAM spatial gate | Veličković et al. GAT (ICLR 2018) | Vaswani et al. (NeurIPS 2017) |

**本設計的核心限制**：不能捕捉「關節 A 因為關節 B 的狀態而變重要」這種跨關節的動態依賴關係。

### 2.4 骨架特徵多通道工程

#### 2.4.1 骨段向量特徵（Bone Feature）

| 項目 | 內容 |
|------|------|
| **文獻** | Shi et al., *"Skeleton-Based Action Recognition with Directed Graph Neural Networks"*, CVPR 2019 (**DGNN**) |
| **文獻** | Shi et al., 2s-AGCN, CVPR 2019（雙流架構中的 bone stream） |
| **定義** | `bone[i] = joint[i] − joint[parent[i]]`，即相對父節點的偏移向量，編碼局部肢體方向 |
| **本專案採用** | 合併為單流多通道（而非獨立 bone stream），bone_x / bone_y 作為特徵通道 6-7 |

#### 2.4.2 關節速度特徵（Joint Velocity）

| 項目 | 內容 |
|------|------|
| **文獻** | Li et al., *"Actional-Structural Graph Convolutional Networks"*, CVPR 2019 (**AS-GCN**) |
| **定義** | `v[t] = xy[t] − xy[t-1]`，一階時間差分，第一幀為 0 |
| **本專案採用** | vx / vy 作為特徵通道 4-5 |

#### 2.4.3 骨段速度特徵（Bone Motion）

| 項目 | 內容 |
|------|------|
| **定義** | `bone_motion[t] = bone[t] − bone[t-1]`，骨架向量的時間差分 |
| **作用** | 捕捉肢體方向的變化率，對 shake（頭頸快速翻轉）判別力強 |
| **文獻** | 延伸自 DGNN/2s-AGCN 的 bone stream 概念，結合 velocity 的時間差分思路 |

#### 2.4.4 關鍵點信心值通道（Keypoint Confidence）

| 項目 | 內容 |
|------|------|
| **作用** | 將 YOLO 逐關節信心值直接作為特徵通道，讓模型學習「低信心座標不可靠」 |
| **文獻** | 本領域無直接對應論文，屬 domain-specific 工程設計；概念上呼應 Sun et al., *"Deep High-Resolution Representation Learning for Human Pose Estimation"*, CVPR 2019 中對關鍵點不確定性的處理思路 |

### 2.5 骨架前處理正規化

骨架序列在送入模型前依序經過四道前處理步驟。以下分別說明各步驟的目的、實作函式、以及在文獻中的對應程度。

| 步驟 | 實作函式（`detectors/behavior_classifier.py`） | 目的 |
|:----:|----------------------------------------------|------|
| ① | `interpolate_missing()` | 時間軸線性插值補全低信心關節 |
| ② | `flip_normalize()` | 序列級多數決翻轉，統一貓咪朝向 |
| ③ | `orientation_normalize()` | 旋轉使 mid_back→hip 軸對齊 y 軸正向 |
| ④ | `normalize_skeleton_coords()` | 以 mid_back 為原點 + 胸髖距為尺度縮放 |

#### 2.5.1 `interpolate_missing()` — 缺失關鍵點插值

| 項目 | 內容 |
|------|------|
| **做法** | 對信心值低於閾值（訓練 0.1、推論 0.5）的關節，以前後最近有效幀做線性插值補全 |
| **文獻** | 標準時序資料缺失值處理慣例；無特定論文，廣泛用於 pose estimation 後處理 |
| **本設計特點** | 訓練與推論使用不同閾值：訓練閾值寬鬆以保留更多樣本；推論閾值嚴格以防止噪點傳播 |

#### 2.5.2 `flip_normalize()` — 水平翻轉朝向統一

| 項目 | 內容 |
|------|------|
| **做法** | 計算序列中「mid_back.x > hip.x」成立的幀數比例（多數決），若多數幀中 mid_back 在 hip 左側，則對整段序列做水平翻轉（x → −x），確保所有序列的 mid_back 均在 hip 右側 |
| **實作位置** | `behavior_classifier.py: flip_normalize()` |
| **文獻對應（2D 增強）** | 水平翻轉在文獻中幾乎一律作為**資料增強**手段，而非訓練前的方向統一步驟。例：Shi et al. (*2s-AGCN*, CVPR 2019)、PYSKL (arXiv 2205.09443) 均以 random horizontal flip 做增強 |
| **文獻對應（3D 方向正規化）** | 3D 骨架論文中有明確的朝向正規化做法：以肩膀與髖部向量之叉積計算身體前向，再旋轉對齊。見 Ke et al. (*"A New Representation of Skeleton Sequences for 3D Action Recognition"*, CVPR 2017) |
| **本設計與文獻的差異** | 現有 2D ST-GCN 論文**無直接對應**的 2D 水平翻轉正規化（pre-normalization）做法；本做法是針對 2D 俯視角貓咪影片的 domain-specific 設計，以 mid_back/hip 的 x 座標差值替代 3D 叉積 |
| **已知限制** | 當貓咪直接朝向或遠離鏡頭時，mid_back.x ≈ hip.x，翻轉決策不穩定（雜訊主導） |

#### 2.5.3 `orientation_normalize()` — 骨架軸對齊旋轉

| 項目 | 內容 |
|------|------|
| **做法** | 計算 mid_back→hip 向量的方向角 θ，對序列每幀的所有關節以原點為中心旋轉 (90°−θ)，使該向量與 y 軸正方向對齊 |
| **實作位置** | `behavior_classifier.py: orientation_normalize()` |
| **文獻對應（3D 論文）** | Shahroudy et al., *"NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"*, CVPR 2016：以 "middle of the spine" 為原點，做 3D 旋轉使脊椎向量平行 Y 軸；Shi et al. (*2s-AGCN*, CVPR 2019)：「rotating all skeletons so that the spine of the person in the first frame is **parallel with the z-axis**」 |
| **文獻對應（骨架正規化綜述）** | 「Preprocessing steps such as normalizing coordinates to a body-centric frame via **translation to the middle spine joint, rotation to align shoulder and spine vectors**, and scaling based on torso length are commonly applied.」（3D Skeleton-Based Action Recognition: A Review, arXiv 2506.00915） |
| **本設計與文獻的關係** | 概念與 NTU RGB-D / 2s-AGCN 的 spine 軸對齊**完全一致**，本實作為 2D 降維版（2D 旋轉矩陣替代 3D 旋轉），可直接引用上述論文並說明「adapted to 2D by replacing 3D rotation with a 2D rotation matrix」 |

#### 2.5.4 `normalize_skeleton_coords()` — 中心化 + 體型縮放

| 項目 | 內容 |
|------|------|
| **做法** | ① **中心化**：每幀所有關節 x,y 減去該幀 mid_back 的座標（per-frame centering）；② **縮放**：計算序列 16 幀中 chest–hip 距離的均值作為 body_size，所有座標除以 body_size |
| **實作位置** | `models/stgcn_model.py: normalize_skeleton_coords(center_joint=4, chest_joint=3, lower_body_joint=5)` |
| **文獻對應（中心化）** | Shahroudy et al., NTU RGB-D, CVPR 2016：「translated to the body coordinate system with its origin on the **middle of the spine** joint」；2s-AGCN：「aligning the center point of the person in the 1st frame with the origin」；PYSKL 沿用 CTR-GCN 的相同做法 |
| **文獻對應（縮放）** | Shahroudy et al., NTU RGB-D, CVPR 2016：「all 3D points are **scaled based on the distance between 'spine base' and 'spine' joints**」；Torso-centered normalization 如 ResearchGate 圖示 (fig3, MDPI 2020) 亦展示以軀幹長為縮放基準 |
| **本設計與文獻的關係** | 三種方法中**文獻支撐最強**，中心化選點（mid_back ↔ middle spine）與縮放基準（chest-hip ↔ spine base to spine）在 NTU RGB-D 原論文中有直接對應，可直接引用 |
| **與文獻的微差異** | NTU RGB-D 取**第 1 幀**的中心做中心化，本實作為**逐幀**中心化（per-frame centering on mid_back）；縮放取**16 幀均值**而非單幀，使尺度估計更穩定 |

---

## 三、訓練策略文獻對照

| 技術 | 文獻 | 本專案實作 |
|------|------|----------|
| **Label Smoothing** | Szegedy et al., *"Rethinking the Inception Architecture"*, CVPR 2016；Müller et al., *"When Does Label Smoothing Help?"*, NeurIPS 2019 | `CrossEntropyLoss(label_smoothing=0.01)` |
| **AdamW 優化器** | Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"*, ICLR 2019 | `lr=5e-5, weight_decay=1e-4` |
| **模型 EMA** | Polyak & Juditsky, *"Acceleration of Stochastic Approximation by Averaging"*, SIAM 1992；現代應用見 Cai et al., *"Once-for-All"*, ICLR 2020 | `decay=0.999`；BN running stats 直接複製而非 EMA |
| **加權隨機採樣** | He et al., *"Learning from Imbalanced Data"*, IEEE TKDE 2009；實作參考 Lin et al., *"Focal Loss"*, ICCV 2017（類別不平衡處理的代表性工作） | `WeightedRandomSampler`；搭配類別加權 CrossEntropyLoss 雙重平衡 |
| **梯度裁剪** | Pascanu et al., *"On the difficulty of training recurrent neural networks"*, ICML 2013 | `clip_grad_norm_(max_norm=1.0)` |
| **ReduceLROnPlateau** | 標準自適應 LR 調度，廣泛使用 | `patience=5, factor=0.5`，監控 val_loss |
| **早停（Early Stopping）** | Prechelt, *"Early Stopping — But When?"*, Neural Networks 1998 | `patience=10` |
| **影片級資料切分** | Zoph et al., *"Neural Architecture Search with Reinforcement Learning"*（含 data leakage 討論）；Tran et al., *"Learning Spatiotemporal Features with 3D CNNs"*, ICCV 2015（video-level split 實踐） | 以影片為單位 train/val split，防止滑動窗重疊序列造成資料洩漏 |

### 3.1 資料增強文獻對照

| 增強方式 | 文獻 | 本專案設定 |
|---------|------|----------|
| **旋轉 / 縮放 / 平移** | Chen et al., *"Channel-wise Topology Refinement Graph Convolution"*, ICCV 2021 (**CTR-GCN**) 使用類似 spatial augmentation | ±12°旋轉、0.9-1.1× 縮放、std=0.04 平移 |
| **關節隨機遮蔽（Occlusion）** | 概念呼應 He et al., *"Masked Autoencoders Are Scalable Vision Learners"*, CVPR 2022 的 masking 思路（skeleton 版本） | prob=0.25，最多遮蔽 2 個關節 |
| **幀丟棄 / 時間抖動** | 文獻：Liu et al., *"Skeleton-Based Human Action Recognition with Global Context-Aware Attention LSTM Networks"*, IEEE TIP 2018 | max_drop=2 幀，jitter_std=0.01 |

---

## 四、尚未採用的主要技術（優化方向）

### 4.1 動態鄰接矩陣（Dynamic / Channel-wise Topology）

| 項目 | 內容 |
|------|------|
| **文獻** | Chen et al., *"Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition"*, ICCV 2021 (**CTR-GCN**) |
| **核心思路** | 鄰接矩陣不固定，由輸入特徵動態計算 channel-wise 拓撲，每個 channel 可有不同的關節連結強度 |
| **未採用原因** | 計算量增加（需對每 batch 動態生成鄰接矩陣）；小資料集 + 5 類場景中過擬合風險提高 |
| **若採用預期收益** | 模型能依貓咪姿態動態強調不同關節連結，如 lick 時頭頸連結強、walk 時前後肢連結強 |

### 4.2 圖注意力（Graph Attention Network 風格）

| 項目 | 內容 |
|------|------|
| **基礎文獻** | Veličković et al., *"Graph Attention Networks"*, ICLR 2018 (**GAT**) |
| **骨架領域** | Huang et al., *"Part-level Graph Convolutional Network for Skeleton-Based Action Recognition"*, AAAI 2020 |
| **核心思路** | 每條邊的權重由兩端關節特徵計算（query-key dot product + softmax），關節之間動態互相決定重要性 |
| **vs 本實作 JointAttention** | GAT 的 attention 在邊上（關節間），本實作的 attention 在節點上（關節獨立）；GAT 能捕捉「關節 A 因 B 的狀態而變重要」的依賴 |
| **未採用原因** | 實作複雜度高；O(V²) 邊注意力在 17 節點的小圖上優勢有限 |

### 4.3 時空 Transformer（Spatio-Temporal Transformer）

| 項目 | 內容 |
|------|------|
| **文獻** | Plizzari et al., *"Skeleton-Based Action Recognition via Spatial and Temporal Transformer Networks"*, CVIU 2021 (**ST-TR**) |
| **文獻** | Qiu et al., *"Spatio-Temporal Tuples Transformer for Skeleton-Based Action Recognition"*, arXiv 2022 (**STTFormer**) |
| **核心思路** | 以 Multi-Head Self-Attention 替代 GCN，可捕捉任意長距離關節/時間依賴 |
| **未採用原因** | 需要大量訓練資料才能發揮優勢；序列長度 T=16、關節數 V=17 的設定讓 Self-Attention 的優勢不顯著；推論延遲增加 |

### 4.4 雙流架構（Two-Stream: Joint + Bone）

| 項目 | 內容 |
|------|------|
| **文獻** | Shi et al., *"Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition"*, CVPR 2019 (**2s-AGCN**) |
| **核心思路** | Joint stream 和 Bone stream 各自獨立訓練，最後 softmax 分數加權 ensemble |
| **本專案採用方式** | 將骨架向量合併為單流多通道（7ch 或 9ch），未分流 |
| **未採用原因** | 訓練成本加倍；現有多通道單流已包含相同資訊，在 5 類小場景下效益有限 |
| **若採用預期收益** | ensemble 通常可提升 1-3% 準確率，對論文表格有幫助 |

### 4.5 InfoGCN（資訊幾何 + 可學習圖）

| 項目 | 內容 |
|------|------|
| **文獻** | Chi et al., *"InfoGCN: Representation Learning for Human Skeleton-based Action Recognition"*, CVPR 2022 |
| **核心思路** | 以資訊最大化目標學習骨架表示，結合 context-dependent 鄰接矩陣與注意力嵌入 |
| **未採用原因** | 架構複雜；SOTA 主要在 NTU RGB+D 等大型人體資料集驗證，轉移到貓體小資料集的效益未知 |

### 4.6 PoseC3D（體素化骨架 + 3D CNN）

| 項目 | 內容 |
|------|------|
| **文獻** | Duan et al., *"Revisiting Skeleton-based Action Recognition"*, CVPR 2022 (**PoseC3D**) |
| **核心思路** | 將骨架轉換為 3D heatmap volume（K×T×H×W），用 3D CNN 建模，避免圖表示的歸納偏置 |
| **未採用原因** | 輸入格式完全不同（需重新設計整個 pipeline）；記憶體需求大；即時推論困難 |

---

## 五、技術選型總表（論文用速查）

| 技術 | 採用 | 文獻 | 關鍵字（搜尋用） |
|------|:----:|------|----------------|
| ST-GCN 骨幹架構 | ✅ | Yan et al., AAAI 2018 | Spatial Temporal Graph Convolutional Network |
| 對稱正規化 D⁻¹/²AD⁻¹/² | ✅ | Kipf & Welling, ICLR 2017 | Graph Convolutional Network, symmetric normalization |
| K=3 分區鄰接 | ✅ | Yan et al., AAAI 2018 | Spatial partitioning strategy |
| 可學習分區重要性 | ✅ | Shi et al. (2s-AGCN), CVPR 2019 | Adaptive graph convolution, learnable adjacency |
| 多尺度時間卷積 | ✅ | Liu et al. (MS-G3D), CVPR 2020 | Multi-scale temporal convolution |
| 可學習分支加權（softmax） | ✅ | MS-G3D 概念延伸 | Adaptive temporal scale weighting |
| 關節空間注意力（input gate） | ✅ | Woo et al. (CBAM), ECCV 2018；Zhang et al. (AAGCN), CVPR 2020 | Joint attention, spatial attention gate |
| 殘差連結 | ✅ | He et al., CVPR 2016 | Residual connection |
| 骨段向量特徵 | ✅ | Shi et al. (DGNN), CVPR 2019 | Bone feature, directed graph |
| 速度特徵（一階差分） | ✅ | Li et al. (AS-GCN), CVPR 2019 | Joint velocity, temporal difference |
| 骨段速度特徵 | ✅ | DGNN + velocity 組合延伸 | Bone motion feature |
| 信心值通道 | ✅ | Domain-specific | Keypoint confidence as feature |
| ②`flip_normalize` — 水平翻轉朝向統一 | ✅ | Domain-specific（2D 設計）；概念呼應 Ke et al., CVPR 2017 的 3D 朝向正規化 | Skeleton orientation normalization, horizontal flip pre-normalization |
| ③`orientation_normalize` — 脊椎軸對齊旋轉 | ✅ | Shahroudy et al. (NTU RGB+D), CVPR 2016；Shi et al. (2s-AGCN), CVPR 2019 | Spine alignment, subject-centric rotation, skeleton axis normalization |
| ④`normalize_skeleton_coords` — 中心化＋體型縮放 | ✅ | Shahroudy et al. (NTU RGB+D), CVPR 2016；Shi et al. (2s-AGCN), CVPR 2019；PYSKL (arXiv 2205.09443) | Body-centric coordinate normalization, torso-length scale normalization |
| Label Smoothing | ✅ | Szegedy et al., CVPR 2016；Müller et al., NeurIPS 2019 | Label smoothing regularization |
| AdamW | ✅ | Loshchilov & Hutter, ICLR 2019 | Decoupled weight decay |
| 模型 EMA | ✅ | Polyak & Juditsky, SIAM 1992 | Exponential moving average |
| 加權隨機採樣 | ✅ | He et al., TKDE 2009；Lin et al. (Focal Loss), ICCV 2017 | Imbalanced learning, weighted sampling |
| 梯度裁剪 | ✅ | Pascanu et al., ICML 2013 | Gradient clipping |
| 影片級資料切分 | ✅ | Data leakage 防範最佳實踐 | Video-level data split |
| 動態鄰接矩陣（CTR-GCN） | ❌ | Chen et al., ICCV 2021 | Channel-wise topology refinement |
| 圖注意力（GAT 風格） | ❌ | Veličković et al., ICLR 2018 | Graph Attention Network |
| Transformer 時空建模 | ❌ | Plizzari et al., CVIU 2021；Qiu et al. 2022 | Spatio-temporal transformer |
| 雙流架構（joint + bone 分開） | ❌ | Shi et al. (2s-AGCN), CVPR 2019 | Two-stream skeleton action recognition |
| InfoGCN | ❌ | Chi et al., CVPR 2022 | Information-theoretic graph convolution |
| PoseC3D（體素骨架） | ❌ | Duan et al., CVPR 2022 | 3D heatmap volume, skeleton action |

---

## 六、本設計的定位

本實作約等於 **2020–2021 年的實用改良版 ST-GCN**，在以下維度做了取捨：

- **保留**：架構輕量（3 層，T=16，V=17）、即時推論友善、可在無 GPU 機器部署
- **採用**：多尺度時間建模（MS-G3D 風格）、可學習圖權重（2s-AGCN 概念）、豐富特徵工程（bone/velocity/conf）、完整正則化策略
- **未採用**：動態圖（CTR-GCN）、GAT、Transformer ← 主因是資料量小（5 類、數十支影片）、需即時推論

對於論文撰寫，本設計的主要 novelty 在於：**貓體特化的骨架圖設計**、**前處理正規化流程**（flip → orientation → body-size）、以及**信心值通道**這個 domain-specific 特徵，這三點在現有人體動作辨識文獻中無直接對應。

**三種正規化方法的論文引用策略：**

| 方法 | 引用策略 | 說明 |
|------|----------|------|
| `flip_normalize` | 說明為 domain-specific 2D 設計，概念呼應 Ke et al. CVPR 2017 | 現有 2D 論文無直接對應；3D 文獻以叉積做朝向正規化，本做法以 x 座標差值替代 |
| `orientation_normalize` | 直接引用 Shahroudy et al. CVPR 2016、Shi et al. CVPR 2019，說明為其 2D 適配版 | 概念完全對應「rotate spine/body-axis to align with coordinate axis」，差別只在 3D→2D |
| `normalize_skeleton_coords` | 直接引用 Shahroudy et al. CVPR 2016（NTU RGB+D 原論文） | 中心化選點與縮放基準在 NTU 論文有逐字對應，可直接引用 |

---

*對應詳細實作說明見 `0_STGCN_DESIGN_ANALYSIS.md`*
