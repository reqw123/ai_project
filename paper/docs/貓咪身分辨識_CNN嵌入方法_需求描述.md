# 🐈 貓咪身分辨識（多貓過濾）— 需求描述

> **主題**：多貓同框時判定「畫面裡哪隻是目標貓」，只讓目標貓進入行為統計。
> **實作方法**：MobileNetV3-Small CNN 分類頭（2026-08 起；取代第一版 HSV 色彩直方圖）。細節見下方各節與最上方狀態框。

> [!IMPORTANT]
> **狀態（2026-08）：已實作並完全取代顏色比對，無 histogram fallback。**
>
> - 工具鏈：`tools/cat_identity/1_build_dataset.py` → `tools/cat_identity/2_train.py`
>   （MobileNetV3-Small ImageNet 微調 → `C:\ai_project\identity_models\`）→
>   `tools/cat_identity/3_infer_video.py`（影片端逐幀辨識 + 視覺化）。
> - 產線：`detectors/identity_verifier.py` 整支改寫成 CNN + N 幀多數決平滑；
>   `verify()` 對外簽章不變（`(is_target, matched_key, score)`）。
> - config：`CatIdentityConfig` 移除 `TARGET_CAT_PROFILE_PATH` / `OTHER_CAT_PROFILE_PATH`，
>   新增 `IDENTITY_MODEL_PATH` / `TARGET_CAT_CLASS`（settings GUI「貓咪身份驗證」分頁同步）。
> - 顏色版腳本 `tools/3_cat_identity_verification_test.py` 與 HSV 基準檔已刪除。
> - 決策 D8（histogram/CNN 混合或旗標切換）→ **不做**，完全取代。
>
> 以下保留為當初的需求脈絡與驗收標準，其中「方法旗標 / 向後相容 / 無回歸（切回 histogram）」
> 相關敘述已被上述決策取代。

---

## 🚀 30 秒 TL;DR

> ✅ **已完成（2026-08）**。以下為當初提案時的摘要，措辭保留為「要做什麼」的規劃視角；現況一律以本頁最上方狀態框為準。

- **做了什麼**：用 ImageNet 預訓練的 MobileNetV3-Small 微調成 N 類分類器，**完全取代**原 `IdentityVerifier` 的 HSV 顏色直方圖特徵（無 fallback、無旗標切換）。
- **為什麼**：顏色直方圖只有色相/飽和度資訊，丟掉紋理/斑紋，貓數增加或有兩隻顏色相近時區分度不夠；對光線/姿勢/動態模糊的容錯也薄弱。
- **範圍**：目前 2 隻貓，未來最多 3 隻，**封閉集合**（固定住戶貓 + 訪客貓歸「unknown」）。**GPU 部署**。
- **實際比原規劃多動的**：`identity_verifier.py` 整支改寫（不只換特徵層）——比對層從「gallery 最近鄰 + Bhattacharyya」換成「分類頭 softmax + 信心門檻」；多幀多數決平滑保留；`FrameProcessor` 側 `_select_target_instance` 多貓挑選邏輯配合調整。
- **延遲**：GPU 上每幀多 1–2 ms，實測感覺不到差異。

---

## 1. 背景與問題

### 1.1 改動前的方法（HSV 色彩直方圖，已於 2026-08 汰換）

> 下表與「特徵」段落描述的是**改動前**的狀態，保留供理解動機。現況見本頁最上方狀態框。

| 元件 | 改動前的角色 |
|---|---|
| `tools/3_cat_identity_verification_test.py`（已刪除） | 測試腳本：`enroll`（建基準，可讀 `crops/` 圖片資料夾或影片）/ `verify`（測試影片逐幀判定）/ `diagnose`（leave-one-out 分離度量化） |
| `detectors/identity_verifier.py` → `IdentityVerifier` | 產線模組：載入 enroll 好的基準檔，對一個 bbox 回答「是不是目標貓」 |
| `processors/frame_processor.py` | 呼叫端：非目標貓的幀當成「沒偵測到貓」處理，不計入統計 |
| `config.py` → `CatIdentityConfig` | `TARGET_CAT_PROFILE_PATH` / `OTHER_CAT_PROFILE_PATH`（已移除）、`ENABLE_IDENTITY_VERIFICATION` |
| `tools/cat_identity/1_build_dataset.py` | 從「目標貓 / 他貓」影片抽幀，輸出 `train_data/cat_identity/dataset/crops/<類別>/`（bbox 裁切圖）與 `frames/<類別>/`（整張原圖）。CNN 訓練資料來源（餵給 `cat_identity/2_train.py`） |

**改動前的特徵**：bbox 裁切區的 HSV H-S 2D 直方圖（H_BINS=30, S_BINS=32），跟 enroll gallery 比最近鄰 Bhattacharyya 距離，取最近的類別；最近的也超過 `UNKNOWN_DISTANCE_CEILING` 就判 unknown。IoU 追蹤 + 多幀多數決平滑。

### 1.2 為什麼要換

- **資訊量不足**：H-S 直方圖丟掉紋理與斑紋，顏色在典型住戶大約只有 3–5 bit 的身分資訊。2 隻顏色差很多的貓夠用，**3 隻裡有兩隻相近**（兩隻虎斑、灰 vs 黑白）就會重疊。
- **容錯薄弱**：光線變化、姿勢、動態模糊、部分遮擋都會扭曲直方圖。
- **成本隨貓數線性成長**：最近鄰比對是 O(貓數 × 樣本數)。CNN 前向是 O(1)（與貓數無關）。

> [!IMPORTANT]
> **動手前的前置條件**：先跑 `RUN_MODE="diagnose"` 把現有 2–3 隻貓都 enroll，看 leave-one-out 準確率與組間距離矩陣。
> - LOO ≥ 95% 且沒有任一對組間距離 < 0.15 → 顏色特徵還夠，暫緩本方法。
> - LOO < 90%，或某兩隻組間距離 < 0.15 → 執行本方法。

---

## 2. 目標

1. 用小型 CNN 嵌入取代 HSV 直方圖，提升 2–3 隻貓（含相近毛色）的辨識準確率與環境容錯。
2. 對推論延遲的影響在 GPU 上「感覺不到」（見 [§5 非功能需求](#5-非功能需求)）。
3. **比對層 / 追蹤層 / 呼叫端邏輯維持不變**，只換掉「一個 bbox → 特徵向量」這一步。
4. 沿用現有的 fail-safe 設計：模型載入失敗 → 整個模組停用 → 回退成「偵測到的貓一律當目標貓」。

---

## 3. 範圍

### ✅ 在範圍內

- 訓練資料準備流程（沿用 / 擴充 `cat_identity/1_build_dataset.py` 的輸出）。
- 一支新的訓練腳本（放 `tools/`，config 驅動、可重現）。
- ImageNet 預訓練 backbone 的微調（凍結策略、augmentation、驗證）。
- 推論端嵌入擷取，替換 `IdentityVerifier._extract_histogram` → embedding。
- 模型產物（權重檔）的存放位置與版本管理。
- `identity_verifier.py` / `config.py` 的整合改動與向後相容（可切回舊方法或停用）。
- 對應的 `diagnose` 能力（用 CNN 嵌入重算 leave-one-out）。
- 文件更新（本目錄相關文件 + `模組責任畫分.md` + `獨立運行腳本索引.md`）。

### ❌ 不在範圍內（明確排除）

| 項目 | 理由 |
|---|---|
| 換掉 `SimpleTracker` / 導入 ByteTrack 等真正的 MOT | 另一個獨立議題，見雙貓討論結論 |
| `FrameProcessor` 逐貓維護 N 套狀態（N 個基線 / 行為 log / 消失計時器） | 真「多貓系統」的重構，本次只做「單目標貓 + 過濾其他貓」 |
| 開放集合 / 貓數 > 3 / 線上自動註冊新貓 | 封閉集合假設；新增第 3 隻用「重訓」處理 |
| 行為辨識、STGCN 相關 | 無關 |
| 蒸餾成更小的自訂網 | GPU 部署下非必要（見 §5）；只有 CPU 延遲實測不行才考慮，屬後續 |

---

## 4. 功能需求

### 4.1 訓練資料

- **來源**：`tools/cat_identity/1_build_dataset.py` 產生的 `cat_identity/dataset/crops/<類別>/`，每隻貓一個類別資料夾（訓練用 crop；`frames/` 原圖備用，供之後改裁切策略或做偵測標註）。
- **品質關卡**：沿用 enroll 標準（剛好一隻貓、conf ≥ 0.6、bbox 高度 ≥ 畫面 15%、Laplacian 清晰度 ≥ 60）。
- **數量目標**：每隻貓 **150–300 張有效裁切圖**（微調預訓練 backbone 的 few-shot 區間）。
- **多樣性硬性要求**：每隻貓的樣本要**跨 8 支以上不同時段 / 光線 / 角度的影片**。200 張全來自同一支連續影片 ≈ 只有 5–10 個獨立樣本，不算數。
- **切分**：訓練 / 驗證要**以整支影片為單位切**（held-out clip），不是同片抽幀。驗證集至少涵蓋每隻貓 2 支未進訓練的影片。
- **標註**：資料夾名稱即類別標籤，不需逐幀標註。

### 4.2 模型

- ✅ **架構**：ImageNet 預訓練的小型 backbone。預設候選 **MobileNetV3-Small**（~2.5M 參數）；備案 MobileNetV2 width=0.5、EfficientNet-Lite0。最終選型見 §7。
- ✅ **微調策略**：凍結 backbone 前段，只訓分類頭 + 最後 1 個 block。
- **輸出頭**：3-class（目前 2 類，預留擴充）+ 信心門檻判 unknown。是否改成「嵌入向量 + gallery 最近鄰」見 §7。
- **輸入解析度**：112×112 或 128×128（letterbox 裁切圖），見 §7。
- **精度**：推論用 fp16。

### 4.3 訓練腳本（新）

- 放 `tools/`，命名與現有工具一致（例如 `3_train_cat_identity_cnn.py`）。
- **不吃指令列參數**，改檔案開頭「使用者設定區」（跟其他 tools 一致）。
- config：資料集路徑、backbone、凍結層數、輸入尺寸、batch、epoch、學習率、augmentation 開關、輸出路徑。
- **augmentation 政策**：水平翻轉、亮度/對比、輕微高斯模糊、random resized crop。
  > [!WARNING]
  > **色相（hue）jitter 要嚴格克制或關閉** — 顏色仍是重要身分訊號，過度 hue jitter 會把它洗掉。飽和度/亮度可小幅擾動。
- **輸出**：權重檔 + 訓練曲線 + 混淆矩陣 + held-out clip 準確率報告 + 一份 `_run_meta.json`（backbone、參數、資料集雜湊、每類張數）。輸出目錄沿用 `eval_results/<工具名>/<時間戳>/` 慣例。
- **可重現**：固定 random seed，記錄資料集清單。

### 4.4 推論端

- 新增一個嵌入擷取器（放 `detectors/`），輸入 = 原始 frame + bbox，輸出 = 特徵向量（或 class logits）。
- 每幀對 `KeypointDetector` 選出的**單一 bbox** 跑一次，batch=1。
- 模型在模組建構子載入到 GPU；載入失敗拋例外（由呼叫端 catch 後停用整個模組）。
- 裁切 / letterbox 邏輯與訓練端**共用同一份程式**（避免 train/inference skew）。

### 4.5 比對邏輯（維持現狀，只換距離來源）

- **若走嵌入路線**：gallery 最近鄰，距離改用 **cosine**；`UNKNOWN_DISTANCE_CEILING` 重新校準。
- **若走分類頭路線**：取 softmax 最大類；最大信心低於門檻 → unknown。
- 多幀多數決平滑（`SMOOTH_WINDOW`）、IoU 追蹤（`SimpleTracker`）**不動**。
- `IdentityVerifier.verify()` 對外介面不變：回傳 `(is_target: bool, matched_key, distance/score)`。

### 4.6 整合

- `config.py` `CatIdentityConfig` 新增：CNN 權重檔路徑、方法切換旗標（`histogram` / `cnn`）、CNN 版 unknown 門檻。
- `identity_verifier.py`：依旗標選特徵後端；兩種後端共用比對層。
- **向後相容**：旗標預設維持 `histogram`，或 CNN 權重不存在時自動退回 `histogram`；`ENABLE_IDENTITY_VERIFICATION=False` 時完全不載入。
- `frame_processor.py` **不需改動**（介面不變）。

### 4.7 diagnose 能力

- `3_cat_identity_verification_test.py`（或新腳本）能用 CNN 嵌入重跑 leave-one-out 與組間距離矩陣，對照顏色直方圖版的數字，量化「換了之後好多少」。

---

## 5. 非功能需求

| 面向 | 需求 |
|---|---|
| **延遲（GPU）** | 每幀身分辨識額外開銷 ≤ 2 ms（fp16、單張 crop）。整條 pipeline 幀率相對「停用身分辨識」的下降實測不可感知（目標 < 3%）。 |
| **延遲（CPU 後備）** | 非本次目標，但選型時傾向仍能在 CPU 跑（不強制達標）。 |
| **模型大小** | 權重檔 ≤ 5 MB（fp16）。 |
| **依賴** | 只用 `torch` / `torchvision`（`ultralytics` 已引入，等於免費）。不新增其他重量級依賴。torchvision 若未直接安裝需列入需求。 |
| **記憶體** | GPU 額外佔用 ≤ 30 MB（權重 + 啟用值）。 |
| **fail-safe** | 權重缺失 / 載入失敗 / 前向例外 → 不擋該幀，回退成「偵測到的貓一律當目標貓」，與現行行為一致。 |
| **可維護性** | 訓練可重現；權重檔版本化並記錄來源 `_run_meta.json`；門檻值寫在 config 且有註解說明校準方式。 |
| **無回歸** | 方法旗標切回 `histogram` 時，行為與改動前完全一致。 |

---

## 6. 驗收標準

1. **準確率**：在 held-out clip 驗證集上，2–3 隻貓的整體辨識準確率 **≥ 97%**（且明顯優於同資料集的顏色直方圖 LOO 基準）。
2. **最難的一對**：`diagnose` 報告中「最容易混的兩隻」的兩兩準確率 ≥ 95%。
3. **延遲**：實測 GPU 每幀額外開銷 ≤ 2 ms；整體幀率下降 < 3%。
4. **fail-safe**：刪除 / 損毀權重檔後系統照常啟動、照常運作（回退行為），有 log 提示。
5. **無回歸**：旗標 = `histogram` 時，`verify` 逐幀輸出與改動前逐位元一致（或差異可解釋）。
6. **文件**：本文件、`模組責任畫分.md`、`獨立運行腳本索引.md` 已更新。

---

## 7. 待決策事項

> [!IMPORTANT]
> 這些會影響設計，進實作前要拍板。

| # | 決策點 | 選項 / 傾向 |
|---|---|---|
| D1 | **輸出頭形式** | ✅ 已定：**(a) 2-class 分類頭 + 信心門檻 unknown**（`cat_identity/2_train.py` 已實作，`cat_identity/3_infer_video.py` 沿用）。封閉集合 ≤ 3 隻，加第 3 隻時把頭改 3-class 重訓即可。 |
| D2 | **backbone 選型** | ✅ 已定：**MobileNetV3-Small（ImageNet 預訓練，凍結 backbone + 解凍最後 1 block）**。group-split 驗證 test 100%（2 貓、5 支 held-out 影片），暫不需要比其他 backbone。 |
| D3 | **輸入解析度** | ✅ 已定：**128**（`image_size` 存進權重檔，推論端讀出來用）。 |
| D4 | **權重檔存放** | ✅ 已定：**`C:\ai_project\identity_models\`**（與 `yolo_models\` / `stgcn_models\` 同層，flat 放 `best/last_cat_identity.pt` + `class_names.json`）。訓練過程產物（曲線 / 混淆矩陣 / csv / logs）另存 `tools/train_data/（已併入 identity_models/run_*）`。 |
| D5 | **加入第 3 隻貓的流程** | 重訓整個頭（封閉集合乾淨）vs 增量。傾向重訓，並在文件寫清楚「新增貓 = 重跑訓練腳本」。 |
| D6 | **enroll gallery 是否保留** | 若走 D1(a) 分類頭，gallery 不再需要；但 `diagnose` 的 LOO 仍需要每類樣本。需定義新的「每類代表樣本」來源。 |
| D7 | **訓練資料標註量** | 150–300/貓是否足夠由 D2 選型與初次驗證結果回饋；不足時的補救是「多拍片」而非「多抽幀」。 |
| D8 | **是否做 histogram → CNN 的混合** | 例如只有直方圖判定曖昧時才跑 CNN。對 ≤ 3 隻貓可能過度設計，傾向不做。 |

---

## 8. 交付物

- [x] `tools/cat_identity/1_build_dataset.py` — 裁切圖資料集
- [x] `tools/cat_identity/2_train.py` — 訓練腳本（config 驅動、run 資料夾 + latest.pt）
- [x] `tools/cat_identity/3_infer_video.py` — 影片端逐幀辨識 + 視覺化（取代顏色版 verify）
- [x] `detectors/identity_verifier.py` — 整支改寫成 CNN + N 幀多數決，`verify()` 簽章不變，fail-safe 保留
- [x] `config.py` `CatIdentityConfig` — `IDENTITY_MODEL_PATH` / `TARGET_CAT_CLASS`（移除 profile 路徑）
- [x] `settings_manager.py` / `settings_window` / `runtime_settings*.json` — 分頁欄位同步
- [x] `tools/3_cat_identity_verification_test.py` + HSV 基準檔 — 已刪除（diagnose 由 train 的混淆矩陣 + infer 覆蓋）
- [x] 文件更新：本文件、`設定視窗欄位對照表.md`、`設定分頁模組與核心函式對照表.md`、`獨立運行腳本索引.md`、`0_AI_專案導覽地圖.md`、`文獻回顧補遺與最新進展_2026-08.md`（§四）、`YOLO-Pose應用文獻與專案優化建議.md`、`資料層架構現況與統一管理評估.md`

---

## 9. 相依與順序

```
diagnose 現況（確認確實需要換）
        │
        ▼
cat_identity/1_build_dataset.py 收足 150–300 張/貓（跨 8+ 支影片）
        │
        ▼
D1–D3 拍板 → 訓練腳本 → 微調 → held-out clip 驗證（達 §6 標準？）
        │
        ▼
推論器 + identity_verifier 整合（旗標預設仍 histogram）
        │
        ▼
實測延遲（§6.3）→ 切換旗標為 cnn → 回歸測試
```

> [!NOTE]
> 實際落地跳過了「旗標 / 向後相容 / 回歸切回 histogram」這幾步（決策 D8 = 完全取代）：`identity_verifier.py` 整支改寫成 CNN，HSV 程式與基準檔直接刪除，沒有 `histogram` / `cnn` 切換旗標。上圖最後兩步的「旗標」字樣只反映當初規劃。

---

## 📎 相關檔案

- `paper/cat_monitoring_system/tools/cat_identity/1_build_dataset.py` — 訓練資料集（bbox 裁切圖）
- `paper/cat_monitoring_system/tools/cat_identity/2_train.py` — MobileNetV3-Small 訓練（run 資料夾 + `latest.pt`）
- `paper/cat_monitoring_system/tools/cat_identity/3_infer_video.py` — 影片端逐幀辨識 + 視覺化（取代舊 `3_cat_identity_verification_test.py` 的 verify/diagnose）
- `paper/cat_monitoring_system/detectors/identity_verifier.py` — 產線模組（已整支改寫成 CNN）
- `paper/cat_monitoring_system/processors/frame_processor.py` — 呼叫端（`_select_target_instance` 多貓挑選）
- `paper/config.py` — `CatIdentityConfig`（`IDENTITY_MODEL_PATH` / `TARGET_CAT_CLASS` / `IDENTITY_CONF_THRESHOLD` / `IDENTITY_FILTER_HYSTERESIS_FRAMES`）
