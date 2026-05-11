# 訓練與推論執行指南

## 前置檢查

確保以下路徑存在並有數據：
- 骨架數據：`C:\cat_pose\gcn_pose\skeletons\` （JSON 檔案）
- 模型輸出目錄：`C:\cat_pose\gcn_pose\models\` （會自動建立）
- 結果輸出目錄：`C:\paper - 複製\cat_monitoring_system\` （會自動建立）

---

## 1️⃣ 訓練模型

### 基礎訓練（預設設定）
```powershell
cd C:\paper
python -u cat_monitoring_system/train_gcn.py
```

### 自訂訓練（常用環境變數）

#### 控制 Attention（關鍵選項）
```powershell
# ✅ 啟用 JointAttention（推薦，精度更高）
$Env:STGCN_USE_ATTENTION='1'; python -u cat_monitoring_system/train_gcn.py

# ⚪ 禁用 JointAttention（baseline 對照）
$Env:STGCN_USE_ATTENTION='0'; python -u cat_monitoring_system/train_gcn.py
```

#### 控制訓練參數
```powershell
# 短訓練（快速實驗）
$Env:STGCN_NUM_EPOCHS='3'; $Env:STGCN_BATCH_SIZE='8'; python -u cat_monitoring_system/train_gcn.py

# 長訓練（完整評估）
$Env:STGCN_NUM_EPOCHS='50'; $Env:STGCN_BATCH_SIZE='8'; $Env:STGCN_LEARNING_RATE='0.001'; python -u cat_monitoring_system/train_gcn.py
```

#### 控制特徵模式（多模式 ablation）
```powershell
# 單模式（默認 xyv，關閉 ablation study）
$Env:STGCN_FEATURE_MODE='xyv_conf'; $Env:STGCN_RUN_ABLATION='0'; python -u cat_monitoring_system/train_gcn.py

# 多模式 ablation（訓練所有特徵組合）
$Env:STGCN_RUN_ABLATION='1'; python -u cat_monitoring_system/train_gcn.py
```

#### 完整範例：xyv_conf 模式，長訓，Attention on
```powershell
$Env:STGCN_NUM_EPOCHS='50'; $Env:STGCN_BATCH_SIZE='8'; $Env:STGCN_USE_ATTENTION='1'; $Env:STGCN_RUN_ABLATION='0'; $Env:STGCN_FEATURE_MODE='xyv_conf'; python -u cat_monitoring_system/train_gcn.py
```

### 環境變數完整清單
| 變數 | 預設值 | 說明 |
|------|--------|------|
| `STGCN_NUM_EPOCHS` | 40 | 訓練 epoch 數 |
| `STGCN_BATCH_SIZE` | 8 | 批次大小 |
| `STGCN_LR` | 0.001 | 學習率 |
| `STGCN_FEATURE_MODE` | xyv | 特徵模式（xyv / xyv_conf / xyv_bone / xyv_conf_bone） |
| `STGCN_USE_ATTENTION` | 1 | 是否啟用 JointAttention（0=off, 1=on） |
| `STGCN_RUN_ABLATION` | 1 | 是否執行 ablation study（0=off, 1=on） |
| `STGCN_RANDOM_SEED` | 42 | 隨機種子 |
| `STGCN_EARLY_STOP` | 10 | Early stopping patience |

---

## 2️⃣ 推論與測試

### 推論單支影片（EMA 平滑版本）
```powershell
# 需要修改 test_video_inference_ema.py 中的：
# - VIDEO_PATH：影片路徑
# - MODEL_PATH：模型路徑
# - OUTPUT_FOLDER：輸出目錄

python -u cat_monitoring_system/test_video_inference_ema.py
```

### 推論單支影片（7 通道特徵版本）
```powershell
python -u cat_monitoring_system/test_video_inference_7ch.py
```

---

## 3️⃣ 訓練輸出

執行完成後，結果會存放在：
```
C:\paper - 複製\cat_monitoring_system\
├── run_xyv/
│   ├── confusion_matrix.png          （驗證集混淆矩陣）
│   ├── training_curves.png           （訓練/驗證曲線）
│   └── ...
├── run_xyv_conf/
├── run_xyv_bone/
├── run_xyv_conf_bone/
└── ablation_summary.csv              （多模式摘要表）
```

模型檔案存放在：
```
C:\cat_pose\gcn_pose\models\
├── stgcn_best_xyv.pth
├── stgcn_best_xyv_conf.pth
├── stgcn_best_xyv_bone.pth
└── stgcn_best_xyv_conf_bone.pth
```

---

## 4️⃣ 快速實驗範例

### 方案 A：快速對照（3 epoch，Attention on/off）
```powershell
# Baseline（Attention off）
$Env:STGCN_NUM_EPOCHS='3'; $Env:STGCN_USE_ATTENTION='0'; $Env:STGCN_RUN_ABLATION='0'; $Env:STGCN_FEATURE_MODE='xyv_conf'; python -u cat_monitoring_system/train_gcn.py

# Attention enabled
$Env:STGCN_NUM_EPOCHS='3'; $Env:STGCN_USE_ATTENTION='1'; $Env:STGCN_RUN_ABLATION='0'; $Env:STGCN_FEATURE_MODE='xyv_conf'; python -u cat_monitoring_system/train_gcn.py
```

### 方案 B：完整 ablation（所有特徵模式，40 epoch）
```powershell
$Env:STGCN_NUM_EPOCHS='40'; $Env:STGCN_RUN_ABLATION='1'; python -u cat_monitoring_system/train_gcn.py
```

### 方案 C：單一最佳配置（xyv_conf，50 epoch，Attention on）
```powershell
$Env:STGCN_NUM_EPOCHS='50'; $Env:STGCN_BATCH_SIZE='8'; $Env:STGCN_USE_ATTENTION='1'; $Env:STGCN_RUN_ABLATION='0'; $Env:STGCN_FEATURE_MODE='xyv_conf'; python -u cat_monitoring_system/train_gcn.py
```

---

## 5️⃣ 注意事項

✅ **能做的事**
- 自由組合環境變數進行不同配置訓練
- 查看混淆矩陣與訓練曲線圖 PNG 檔
- 切換 Attention on/off 做對照實驗
- 修改超參數（epoch、batch size、學習率等）

⚠️ **注意**
- 確保數據路徑正確（JSON 檔案需含逐幀標籤）
- 長訓練可能耗時（取決於 GPU/CPU 與 epoch 數）
- 每次訓練會覆寫相同 `run_name` 的舊結果，若要保留需改 `run_name`
- 確保磁盤空間足夠（模型 ~10MB，混淆矩陣/曲線圖 ~100-500KB）

---

## 6️⃣ 常見問題

**Q: 訓練一次要多久？**
- 3 epoch（快速測試）：~2-5 分鐘
- 40-50 epoch（完整訓練）：~30-60 分鐘（取決於 GPU）

**Q: 怎麼知道訓練有沒有進度？**
- 每個 epoch 會輸出 `Train Loss`, `Val Acc`, `Macro-F1` 等指標
- 終端會顯示進度條與每個 batch 的 loss
- 最終輸出會顯示 `Best model saved` 與結果路徑

**Q: 能否同時跑多個訓練？**
- 可以，但建議在不同終端視窗執行，避免 GPU/CPU 競爭
- 確保使用不同的 `run_name` 或結果目錄避免覆寫

**Q: 推論時怎麼用新訓練的模型？**
- 在推論腳本中把 `MODEL_PATH` 設為訓練輸出的模型路徑
- 例如：`C:\cat_pose\gcn_pose\models\stgcn_best_xyv_conf.pth`
