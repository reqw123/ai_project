# 貓咪行為辨識系統 — 個體化基線健康監測

以 YOLO-Pose + ST-GCN 辨識貓咪行為，為單一貓咪建立個體化基線，偵測偏離日常行為模式的健康異常訊號。

## Language

**個體化基線（Individualized Baseline）**：
以單一貓咪過去一段時間（`baseline.py`：預設近 30 天，排除監控時數不足或手動排除的日子）的每日行為指標，計算出的 mean / median / std / MAD / EWMA 統計量，作為判斷「今天算不算異常」的比較基準。
_Avoid_: 常模、normal range（這兩個詞在文獻脈絡中常指群體常模，本專案的基線是個體專屬）

**偏差（Deviation）**：
今日某指標相對於個體化基線的偏離程度，統一以 `sigma_equivalent`（等效常態 z 值）表示。連續型指標（時長，如 `lick_time`）用穩健 z-score（MAD-based）；稀疏事件計數（如 `scratch_count`）用 Poisson／Negative-Binomial 尾機率換算。
_Avoid_: 舊版單一 z-score（`z=(今日值-mean)/std`）已知在稀疏計數指標上有假警報問題，見 `analytics/README.md`

**Class A / Class B / Class C**：
`fusion.py` 的三類證據分組——Class A 是「關鍵自我照護行為」（lick/scratch 的時長與次數，權重合計 0.85 內部再分配，具 Single Behavior Critical Rule 可單獨觸發等級覆蓋）；Class B 是「輔助活動行為」（shake/walk/stop）；Class C 是節律／行為轉移分析（目前仍由 Node-RED 純聚合計算，未 Python 化）。三者以 45% / 25% / 30% 融合為最終分數。
_Avoid_: dScore / fScore / rScore / tScore（`貓咪個體化基線.md`、`NODE_RED_FUNCTIONS.md` 描述的舊版四維加權設計，已由 Class A/B/C 取代，見 `docs/adr/0001-統一健康風險評分引擎.md`）

**風險等級（Risk Level）**：
融合分數對應的四級分類：Normal（<20）／Mild Behavioral Deviation（<45）／Moderate Behavioral Deviation（<70）／Severe Behavioral Deviation（≥70）。Class A 觸發 override 時，等級不會低於 override 指定的等級。
_Avoid_: dScore 系列文件中的 Attention / Warning / High Risk 命名（舊版用語）

**事件標記（Event Tag）／排除日（Excluded Date）**：
飼主可對某一天標記已知非健康原因（看醫生／換飼料／訪客／緊張／服藥），標記後的日子可被排除於基線計算之外，避免已知干擾污染基線。目前僅影響「基線計算」，尚未影響「當日風險分數判定」（見待辦事項）。

## 文獻依據

完整文獻清單見 `docs/個體化基線與異常偵測文獻來源.md`（個體化基線建構方法論、複合風險分數加權方法論、動物個體化異常偵測前例、混淆因子處理、貓咪自我照護行為臨床基礎，共約 30 筆，2026-08 彙整）。`baseline.py`／`deviation.py`／`fusion.py` 模組說明已內嵌對應引用。

## 已知待處理的技術缺口

- **權重（已梳理完成，維持現狀＋誠實揭露限制）**：Class A/B/C 內部子權重與三者融合權重皆為專家經驗設定，無資料驅動校準。
  - Class A 時長權重高於次數（`fusion.py`）：有文獻依據，論述為「量測穩定性」（次數受 bout 切分規則雜訊影響較大），非「時長較嚴重」——VAScat（Colombo et al. 2022）顯示 lick/scratch 對應不同臨床徵象、弱相關 r=0.26，支持兩者分開計分但不支持時長更嚴重的論述。
  - Class B `shake_count` 權重最高（0.40）：**無文獻支持**，獸醫共識為「甩頭對耳疾敏感但不具鑑別力」，已在 `fusion.py` 註解中明確標註為未驗證設計選擇。
  - Class A(45%) > Class C(30%) > Class B(25%) 融合權重：**「自我照護比活動量更具特異性」的隱含假設與疾病行為文獻矛盾**（Lopes et al. 2021；Merck Vet Manual 皆視兩者為同一組非特異性疾病行為症候群）；Class C 權重有間接支持（Wagner et al. 2021 節律異常偵測牛隻疾病 95% recall）但無法佐證 30% 這個數字。已決定：論文誠實寫成限制，不調整現有數字（見 `fusion.py` 註解）。
  - sigma 門檻（2.5/3.0/4.0σ）：經驗設定，已記錄未來以 VAScat 的 ROC 校準法重新校準的方法論路徑。
- **基線統計方法論（已梳理完成）**：MAD 穩健 z-score（連續指標）+ Poisson/NB 尾機率（稀疏計數）是對兩支獨立文獻傳統（穩健統計；流行病學計數監測）的工程綜合，非單一文獻的直接沿用，已寫入 `baseline.py`/`deviation.py` 模組說明。最直接的同領域前例是 Silva, Ribeiro & Gama (2025)（5貓+5狗個體行為異常偵測，非監督 ML 而非統計模型）。GP／狀態空間／貝氏階層模型三個替代方案文獻對應更精準但落地成本高（尤其貝氏階層需要跨貓資料），列未來工作。
- 冷啟動（基線資料 <7 天）目前直接關閉判斷（`deviation_available=false`），未採用族群先驗＋個體貝氏壓縮（James-Stein/經驗貝氏）等漸進式方案（已決定暫不展開，列未來工作）
- 尚無真實健康異常事件的 ground truth 可驗證風險分數的敏感度/特異度——目前也**沒有可信的真實連續即時監測多日資料**（`C:\a\global.json`，2026-08-10 前為 `.node-red/context/global/global.json`，裡面的歷史記錄來自離線影片批次測試，不適用於基線驗證），這比「資料量不夠」更根本，直接影響自我一致性切半驗證、事件標記弱 GT 等驗證方法能否執行
- `貓咪主控.json` 與 `cat_health_v3_flow.json` 兩套健康引擎目前同時匯入運作（見 `docs/adr/0001-統一健康風險評分引擎.md`），已決定統一以 Class A/B/C 為準、停用前者的重複引擎與 Discord 警示，實際 Node-RED 節點停用尚未執行
- **事件標記／排除日的作用範圍尚未擴及當日判定**（上面「事件標記」條目提到的缺口，之前只寫「見待辦事項」卻沒有對應項目，這裡補上）：確認過 `deviation.py`/`fusion.py` 目前完全不讀 `excluded_dates`，排除某天只會讓那天不進入未來的基線計算，不會讓「今天」被標記排除後跳過或調整當日風險分數判定——例如飼主當天標記「看醫生」，系統仍會照常對這天算偏差分數並可能觸發警示。是否要讓排除也影響當日判定（以及怎麼影響，例如當天分數要不要顯示但加註免責、還是乾脆不計）尚未決定，列未來工作。
