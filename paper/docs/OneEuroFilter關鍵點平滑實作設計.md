# One Euro Filter 關鍵點平滑——實作設計文件（尚未實作）

**狀態**：規劃階段，尚未寫程式。這份文件是給之後動手實作時的規格與流程參考，目的是讓實作跟既有 `models/keypoint_kalman.py` 的架構、config 慣例、驗證流程完全一致，不用重新設計一遍。
**背景**：見 `docs/姿態估計推論前後處理技術文獻與應用建議.md` 第三節、memory `project_pose_pre_post_processing`——One Euro Filter 是文獻掃描找到、目前專案沒有的技術，理論上比 Kalman 更適合處理「shake 需要低延遲、stop 需要強平滑」這種同一組全域參數顧不到兩頭的問題。
**日期**：2026-08-11

---

## 一、演算法本身

One Euro Filter（Casiez, Roussel & Vogel, 2012, CHI）是**自適應指數移動平均**：平滑係數 `α` 不是固定值，而是每一幀根據「訊號目前移動多快」重新算。核心是兩層 EMA：

1. 先對訊號的**變化率（速度）**做一次 EMA 平滑，得到 `dx_hat`
2. 用 `dx_hat` 算出這一幀該用多強的平滑（截止頻率），再對**訊號本身**做一次 EMA 平滑

參考虛擬碼（標準實作，`t_e` = 幀間時間間隔，本專案固定 `dt=1.0`，跟 Kalman 的 `F` 矩陣假設一致）：

```python
def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exp_smooth(a, x, x_prev):
    return a * x + (1 - a) * x_prev

# 每一幀：
a_d = smoothing_factor(t_e, d_cutoff)              # d_cutoff 通常固定 1.0，很少調
dx = (x - x_prev) / t_e                             # 估計速度
dx_hat = exp_smooth(a_d, dx, dx_prev)               # 平滑後的速度估計

cutoff = min_cutoff + beta * abs(dx_hat)            # 速度快 → cutoff 變大 → 平滑變弱
a = smoothing_factor(t_e, cutoff)
x_hat = exp_smooth(a, x, x_prev)                    # 平滑後的訊號

x_prev, dx_prev = x_hat, dx_hat                     # 存起來供下一幀用
```

**兩個要調的參數**：
- `min_cutoff`：訊號靜止時的平滑基準——調低 = 靜止時更平滑（壓 stop 的抖動）
- `beta`：速度係數——調高 = 速度快時平滑放鬆得更快（降 shake 的延遲）
- `d_cutoff`：速度估計本身的平滑，通常固定 1.0 不調（官方建議值）

---

## 二、跟 Kalman 的關鍵設計差異（動手前務必想清楚）

1. **One Euro 不需要「有沒有量測」這個分支**。Kalman 的 `step(z, has_measurement)` 在缺測時只 predict、不 update，靠等速度模型外推；One Euro 是純反應式濾波器，沒有這種機制。但**這個差異在本專案不影響替換**——實際插入點永遠是 `interpolate_missing()` 先補完缺測，Kalman 的外推能力在現有用法裡本來就沒被用到（詳見 memory `project_pose_pre_post_processing`）。所以 `oneeuro_smooth_sequence()` 的函式簽名可以維持跟 `kalman_smooth_sequence()` 同樣的 `(sequence, conf, threshold, ...)` 介面（方便兩邊 drop-in 替換、共用 dispatch 邏輯），但內部**不需要**也不應該假裝自己有 predict-only 分支去外推缺測——`conf`/`threshold` 這兩個參數在 One Euro 版本裡實際上用不太到（因為序列進來時已經被 `interpolate_missing()` 處理過），保留只是為了介面一致，函式內部要用註解講清楚這件事，不要照抄 Kalman 的邏輯造成誤導。

2. **One Euro 是逐座標軸獨立運作**（x、y 分開各自一個濾波器狀態），不像 Kalman 用 4 維狀態向量把 x/y/vx/vy 綁在一起估。這是文獻上的標準做法（MediaPipe/MMPose 都這樣做），實作上用 numpy 對 `[x, y]` 向量直接做 elementwise 運算即可，不需要真的拆成兩個物件。

3. **沒有協方差/不確定性輸出**。Kalman 的 `P` 矩陣理論上可以額外拿來當 `skeleton_quality_assessment.py` 的輔助訊號（`docs/YOLO-Pose應用文獻與專案優化建議.md` 提過這個構想，但從未真的實作），One Euro 沒有對應的量。目前這個構想本來就沒實作，所以不算實質損失，這裡只是誠實記一筆。

---

## 三、預計新增/修改的檔案（照抄 Kalman 當時的整合模式）

| 檔案 | 動作 |
|---|---|
| `models/keypoint_oneeuro.py`（新檔） | 仿照 `keypoint_kalman.py` 結構：`OneEuroFilterKeypoint` 類別（單一 2D 關鍵點的濾波器狀態）+ `oneeuro_smooth_sequence(sequence, conf, threshold, min_cutoff, beta, d_cutoff)` 函式（逐關節套用，介面對齊 `kalman_smooth_sequence`） |
| `stgcn_config.yaml` | `SMOOTHING_KIND` 的合法值增加 `"oneeuro"`；新增 `ONEEURO_MIN_CUTOFF`、`ONEEURO_BETA`、`ONEEURO_D_CUTOFF` 三個設定值 |
| `tools/0_train_gcn.py` | `CatSkeletonDataset._load_sequences()` 的 `smoothing_kind` dispatch 增加 `elif self.smoothing_kind == "oneeuro":` 分支（呼叫 `oneeuro_smooth_sequence`），`train_model()`/`run_kp_ema_ablation()` 等讀取 config 的地方比照 `KALMAN_PROCESS_NOISE`/`KALMAN_MEASUREMENT_NOISE` 的模式加對應三個變數 |
| `tools/_smoothing_eval_common.py` | 不用改——`run_comparison()` 吃的是外部傳入的 `preprocess_fn`，跟平滑方式無關 |
| `tools/eval_accuracy_smoothing_compare.py`（**2026-08-11 使用者已手動刪除此檔，需要重新建立，不是修改**） | 仿照原本的設計重寫：`_build_preprocess_fn()` 依 `cfg["kind"]` 分派（`none`/`kalman`/`oneeuro`）、`SMOOTHING_CONFIGS`/`EVAL_CLASSES` 寫死在程式碼裡管理，呼叫 `_smoothing_eval_common.run_comparison()`（這支共用模組還在，沒被刪）。新版可以只保留 `none`/`oneeuro` 兩組起手（Kalman 那條路線已收尾，不必再預設帶著） |
| `tools/eval_gcn_compare.py` | `evaluate_video()` 的 `smoothing_kind` dispatch 增加 `oneeuro` 分支（`kalman_smooth_sequence` 旁邊加 `oneeuro_smooth_sequence` 呼叫）；`HARD_MODELS` 之後可以比照現有 122/126/127/128 的模式，加一筆 One Euro 訓練出來的模型進去比較 |

---

## 四、參數起始值——目前沒有實測依據，不要憑感覺填

Kalman 的 Q=90/R=70 是拿真實 stop/shake 影片實測校準出來的（見 `models/keypoint_kalman.py` 開頭註解），**One Euro 的 `min_cutoff`/`beta` 目前完全沒有對應的實測**。官方論文/常見實作的起始建議值通常是 `min_cutoff=1.0, beta=0.0, d_cutoff=1.0`（`beta=0.0` 等於退化成固定截止頻率的普通低通濾波器，之後再逐步調高 beta 觀察高速時延遲有沒有改善）——這只是業界通用的保守起點，不是針對貓咪關鍵點座標尺度校準過的數字，動手實作時第一步應該是比照 Kalman 當時的做法，用真實 YOLO 偵測結果實測校準（例如 stop 類別量高信心幀間的座標抖動、shake 類別量速度變化幅度），而不是直接套用預設值就拿去跑準確度比較。

---

## 五、驗證流程——照 Kalman 走過的路線，不要跳步驟

嚴格複用 `project_kalman_smoothing_eval` 記憶裡記錄的分階段流程，這是三輪實驗換來的教訓，不要因為理論上 One Euro 更合理就跳過驗證：

1. **推論期套用測試**（不需要重訓，最便宜）：`eval_accuracy_smoothing_compare.py` 已被刪除，需要先重建（見上方第三節），`SMOOTHING_CONFIGS` 加一組 One Euro 設定，先看看跟現行無平滑 baseline（122）比起來準確度如何。如果推論期套用就有明顯正面訊號，才有理由往下一步投入。
2. **參數敏感度掃描**：比照 `KALMAN_PARAM_SWEEP` 的做法，圍繞第一步的起點掃幾組不同 `min_cutoff`/`beta`。
3. **重新訓練＋匹配評估**：只有前兩步顯示有希望時才做——`SMOOTHING_KIND: "oneeuro"` 重訓一個模型，用 `eval_gcn_compare.py` 跟 122 baseline（以及既有的 126/127/128 Kalman 模型，如果還想留著當對照組）做匹配評估比較。
4. **McNemar 顯著性檢定**：`eval_gcn_compare.py` 已經內建，直接看有沒有統計顯著差異，不要只看數字大小就下結論。

**如果第 1 步就沒有訊號**（One Euro 推論期套用也沒有贏過無平滑），可以直接參考 Kalman 的最終結論——這可能代表瓶頸根本不在「關鍵點平滑演算法選哪個」，而是 ST-GCN 對這一層雜訊本來就不敏感，不用再重複投入到第 2、3 步。

---

## 誠實限制

- 本文件的演算法虛擬碼是標準參考實作，但沒有真的寫成 Python 跑過，實作時務必對照官方 reference（`Casiez et al. 2012` 或 MMPose 的 `one_euro_filter.py`，搜尋結果裡有找到一個相關 issue：`open-mmlab/mmpose#1176`，指出過官方實作曾有一個小 bug，實作前建議看一下這個 issue 討論內容）驗證公式細節，不要照抄本文件的虛擬碼就假設一定正確。
- 參數起始值（第四節）純粹是文獻常見慣例，不是校準過的數字，正式使用前一定要照 Kalman 的模式重新校準。
