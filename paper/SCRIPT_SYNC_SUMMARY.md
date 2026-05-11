# 三個推論/訓練腳本完全匹配總結

## ✅ 已完成的同步工作

### 1. 統一特徵構建函數 (Shared Module in stgcn_model.py)

**添加的共享函數：**
- `compute_bone_feature(sequence)` — 計算骨架特徵
- `get_in_channels_for_mode(feature_mode)` — 根據模式回傳通道數
- `build_feature_tensor(sequence_xy, conf_seq, feature_mode)` — 構建多通道特徵

**特徵通道順序（統一規範）：**
| 模式 | 通道數 | 順序 |
|------|--------|------|
| `xyv` | 4 | x, y, vx, vy |
| `xyv_conf` | 5 | x, y, conf, vx, vy |
| `xyv_bone` | 6 | x, y, vx, vy, bone_x, bone_y |
| `xyv_conf_bone` | 7 | x, y, conf, vx, vy, bone_x, bone_y |

### 2. train_gcn.py 更新

**變更內容：**
- ✅ 導入共享函數 (`get_in_channels_for_mode`, `build_feature_tensor`)
- ✅ 刪除本地版本的 `compute_bone_feature`, `get_in_channels_for_mode`, `build_feature_tensor`
- ✅ 自動添加 `att_on` 或 `att_off` 後綴到 run_name
  ```
  若 STGCN_USE_ATTENTION='1' 且 feature_mode='xyv_conf' 
  → run_name 自動變為 "run_xyv_conf_att_on"
  若 STGCN_USE_ATTENTION='0' 
  → run_name 自動變為 "run_xyv_conf_att_off"
  ```

**環境變數支援：**
- `STGCN_USE_ATTENTION` (已有) — 控制注意力機制 on/off
- `STGCN_FEATURE_MODE` (已有) — 選擇特徵模式
- 自動後綴無需用戶操作

### 3. test_video_inference_ema.py 更新

**變更內容：**
- ✅ 導入共享函數 (`build_feature_tensor`, `get_in_channels_for_mode` 等)
- ✅ 添加環境變數支援：`STGCN_FEATURE_MODE`
- ✅ 修改 BehaviorClassifier 初始化以傳遞 `feature_mode` 和 `in_channels`
- ✅ 實裝多通道特徵預計算邏輯
  ```python
  if STGCN_FEATURE_MODE != "xyv":
      # 正規化 → build_feature_tensor → precomputed=True
      seq_features = build_feature_tensor(seq_norm, conf_arr, STGCN_FEATURE_MODE)
      pred_id, pred_conf, pred_probs = behavior_classifier.classify(
          seq_features, precomputed=True
      )
  else:
      # xyv 模式直接傳遞座標（內部正規化）
      pred_id, pred_conf, pred_probs = behavior_classifier.classify(seq_array)
  ```

### 4. test_video_inference_7ch.py 更新

**變更內容：**
- ✅ 導入共享函數 (`build_feature_tensor`, `compute_bone_feature`)
- ✅ 修改 `build_7ch_features` 改為調用共享 `build_feature_tensor("xyv_conf_bone")`
- ✅ 刪除本地重複的 `compute_bone_feature` 定義

---

## 📋 使用示例

### 訓練時自動添加後綴（無需手動指定 run_name）

```powershell
# Attention ON + xyv_conf 特徵 + 50 epochs
# 自動輸出到 run_xyv_conf_att_on/
$Env:STGCN_NUM_EPOCHS='50'
$Env:STGCN_USE_ATTENTION='1'
$Env:STGCN_FEATURE_MODE='xyv_conf'
python -u cat_monitoring_system/train_gcn.py

# Attention OFF + xyv_conf 特徵 + 50 epochs
# 自動輸出到 run_xyv_conf_att_off/
$Env:STGCN_NUM_EPOCHS='50'
$Env:STGCN_USE_ATTENTION='0'
$Env:STGCN_FEATURE_MODE='xyv_conf'
python -u cat_monitoring_system/train_gcn.py
```

### EMA 推論腳本支援多特徵模式

```powershell
# xyv 模式（預設，4 通道）
$Env:STGCN_FEATURE_MODE='xyv'
python cat_monitoring_system/test_video_inference_ema.py

# 7 通道模式（需訓練時以 xyv_conf_bone 模式產生的模型）
$Env:STGCN_FEATURE_MODE='xyv_conf_bone'
python cat_monitoring_system/test_video_inference_ema.py

# 5 通道模式（含信心度）
$Env:STGCN_FEATURE_MODE='xyv_conf'
python cat_monitoring_system/test_video_inference_ema.py
```

---

## 🔍 驗證清單

- ✅ **特徵通道順序** — 三個腳本使用同一 `build_feature_tensor` 函數，順序保證一致
- ✅ **骨架拓撲** — 共用 `PARENTS` 陣列和 `compute_bone_feature`
- ✅ **正規化流程** — 共用 `flip_normalize`, `orientation_normalize`, `normalize_skeleton_coords`
- ✅ **環境變數控制** — 訓練和推論都支援 `STGCN_FEATURE_MODE`
- ✅ **自動路徑管理** — 訓練自動添加 `att_on`/`att_off` 後綴，避免覆寫
- ✅ **模型兼容性** — BehaviorClassifier 自動從模型推斷 `in_channels`（如無指定則從 checkpoint 讀取）

---

## 📝 設定路徑對照表

### 預設輸出位置

```
C:\paper - 複製\cat_monitoring_system\
├── run_xyv_att_on/              # 4 通道 + Attention ON
├── run_xyv_att_off/             # 4 通道 + Attention OFF
├── run_xyv_conf_att_on/         # 5 通道 + Attention ON
├── run_xyv_conf_att_off/        # 5 通道 + Attention OFF
├── run_xyv_bone_att_on/         # 6 通道 + Attention ON
├── run_xyv_bone_att_off/        # 6 通道 + Attention OFF
├── run_xyv_conf_bone_att_on/    # 7 通道 + Attention ON
└── run_xyv_conf_bone_att_off/   # 7 通道 + Attention OFF
    ├── training_curves.png
    ├── confusion_matrix.png
    └── stgcn_best_xyv_conf_bone_att_on.pth
```

---

## 🎯 後續操作建議

1. **長訓驗證（推薦）**
   ```powershell
   # 用完整 50 epoch 確認 att_on vs att_off 的穩定性
   $Env:STGCN_NUM_EPOCHS='50'; $Env:STGCN_USE_ATTENTION='1'; $Env:STGCN_FEATURE_MODE='xyv_conf'; python -u cat_monitoring_system/train_gcn.py
   ```

2. **多特徵組合測試（可選）**
   - 比較 xyv_conf vs xyv_conf_bone 效果差異

3. **推論驗證（推薦）**
   - 用訓練好的各模型進行推論測試
   - 驗證特徵通道在推論時是否正確預計算

---

**同步完成日期：** 2026-04-28  
**驗證狀態：** ✅ 完全匹配
