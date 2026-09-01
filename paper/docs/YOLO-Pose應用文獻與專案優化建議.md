# YOLO-Pose 應用文獻回顧與本專案優化建議

**用途**：透過 `/deep-research`（three-way-scan 模式，WHY/HOW/WHAT 架構）搜尋 10 篇 YOLO-Pose 應用相關文獻（優先納入有開放原始碼者），並逐篇對應本專案（`cat_monitoring_system`）可以參考或優化的具體模組。
**方法**：多輪 WebSearch + WebFetch 直接搜尋與核對（非透過 deep-research 完整 13-agent pipeline——本次任務範圍明確（10篇+專案應用性分析），未執行 devil's advocate/ethics review/editor-in-chief 等審稿層級步驟，屬於輕量文獻掃描，非完整學術報告）。
**日期**：2026-08-10
**誠實揭露**：#5、#8 兩篇因來源網站（PMC、ScienceDirect）阻擋自動化存取（reCAPTCHA / 403），僅能依搜尋引擎摘要而非全文核對，已在各自條目標註；其餘 8 篇已直接讀取原文/摘要頁核對。開源狀態逐篇查證，查無公開原始碼者誠實標註「未查到」，不做臆測。

---

## 十篇文獻總覽

| # | 標題 | 出處/年份 | 物種/對象 | 開源狀態 |
|---|---|---|---|---|
| 1 | YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss | arXiv:2204.06806, CVPRW 2022 | 人體 | ✅ 有 |
| 2 | T-LEAP: Occlusion-robust pose estimation of walking cows using temporal information | arXiv:2104.08029, 2021 | 乳牛 | ✅ 有（2026-08-10 深入研究時補查到，見下方「深入」章節） |
| 3 | Multi-animal pose estimation, identification and tracking with DeepLabCut | Nature Methods, 2022 (Lauer et al.) | 多物種（鼠/獼猴/魚群等） | ✅ 有 |
| 4 | YOLO-BCD: A Lightweight Multi-Module Fusion Network for Real-Time Sheep Pose Estimation | Sensors 25(9):2687, 2025 | 綿羊 | ❌ 未查到 |
| 5 | Research on the Behavior Recognition of Beef Cattle Based on the Improved Lightweight CBR-YOLO Model in Multi-Scene Weather | PMC11475345, 2024（僅摘要層級核對） | 肉牛 | ❌ 未查到 |
| 6 | Monitoring Cattle Ruminating Behavior Based on an Improved Keypoint Detection Model | PMC11200719, 2024 | 乳牛 | ❌ 明確標示「資料備索、無公開程式碼」 |
| 7 | Lightweight cattle pose estimation with fusion of reparameterization and an attention mechanism | PLOS ONE, 2024 | 牛 | ⚠️ 僅資料集開源（figshare），程式碼未提及 |
| 8 | Deep learning for visual animal monitoring (detection, tracking, pose estimation, and behavior classification): A comprehensive review | ScienceDirect, 2025（僅摘要層級核對） | 綜述（跨物種） | N/A（綜述論文） |
| 9 | Appearance-based computer vision pipeline for multi-animal monitoring of canine activity, behavior and clinical observations | Frontiers in Toxicology, 2026 | 犬（實驗室群養） | ❌ 未查到 |
| 10 | Custom-training Ultralytics YOLO11 for dog pose estimation（含 Dog-Pose 資料集） | Ultralytics 官方文件/部落格 | 犬 | ✅ 有（YOLO11 本身開源，資料集公開） |

> 註：#10 是廠商技術文件/資料集發布，不是同儕審查論文——刻意保留是因為它跟本專案用的是**同一個模型家族（YOLO11）**，實務參考價值高，但引用時需區分於學術文獻，不可視為同等級證據。

---

## 逐篇 WHY / HOW / WHAT

<a id="sec-1"></a>
### 1. YOLO-Pose（Maji, Nagori, Mathew & Poddar, 2022）
- **WHY**：傳統多人姿態估計要嘛是 top-down（先偵測人再估姿態，慢）要嘛是 bottom-up（先估關鍵點再分組，需要複雜後處理），兩者都不是端到端優化。
- **HOW**：把姿態估計當成物件偵測的延伸，單次前向傳播同時輸出 bounding box 與關鍵點，直接用 Object Keypoint Similarity（OKS）當 loss 訓練，不用 surrogate loss、不用 test-time augmentation。
- **WHAT**：COCO 上達到 SOTA，且完全不需要關鍵點分組後處理。
- 開源：https://github.com/TexasInstruments/edgeai-yolov5 、https://github.com/TexasInstruments/edgeai-yolox

<a id="sec-2"></a>
### 2. T-LEAP（cow, 2021）
- **WHY**：靜態（單幀）姿態估計模型在動物被遮蔽（例如柵欄、其他牛隻擋住）時準確度大幅下降，但酪農場的步態分析（跛行偵測）恰好最需要在這種真實雜訊環境下維持準確度。
- **HOW**：把基礎 LEAP 姿態模型擴充成吃「前面連續幾幀」的時序資訊（T-LEAP），而非逐幀獨立推論。架構上用 3D 卷積把時間維度也納入卷積運算，`seq_length` 參數可設 1（=靜態 LEAP）到 4（=T-LEAP），PyTorch 實作。
- **WHAT**：無遮蔽時兩版準確度都到 99%；人工加入遮蔽後，時序版比靜態版最多提升 32.9%；換成沒看過的牛時仍有 87.6% 準確度（已知牛 93.8%）。
- 開源：https://github.com/hrussel/t-leap （README 未明確說明 `seq_length` 視窗是「只看過去幀」還是「涵蓋未來幀」，這點會直接影響能不能無延遲用在即時串流上，詳見下方「深入」章節）

<a id="sec-3"></a>
### 3. DeepLabCut multi-animal（Lauer et al., 2022, Nature Methods）
- **WHY**：多隻動物互動時（尤其外觀相似的同種動物）會互相遮蔽，關鍵點很難正確對應到「是哪一隻動物的」，一般多人姿態估計的方法在近距離互動的同種動物身上表現不好。
- **HOW**：(1) 用 part affinity fields 資料驅動地決定骨架連線（不用手動設計骨架拓樸）；(2) 局部先用橢圓/框追蹤器產生短 tracklet，再用網路流優化（結合運動/距離/形狀/動態多種成本函數）把短 tracklet 縫合成長軌跡；(3) 身分預測用監督式學習（有標記動物）或無監督 transformer-based metric learning（無標記動物）。
- **WHAT**：關鍵點誤差 2.65–5.25 像素，優於 SOTA COCO 模型（HRNet-AE、ResNet-AE）；獼猴頭部關鍵點身分預測準確率達 99.2%；在鼠/獼猴/14 條魚群等多種場景驗證過。
- 開源：https://github.com/DeepLabCut/DeepLabCut ，基準資料集 https://benchmark.deeplabcut.org/

<a id="sec-4"></a>
### 4. YOLO-BCD（sheep, Sun et al., 2025, Sensors）
- **WHY**：農場環境光照多變、常有遮蔽，一般姿態模型參數量大、算力需求高，不適合即時農場部署。
- **HOW**：多層級輕量化設計，強化特徵融合機制 + 空間-通道注意力模組（multi-module fusion）。
- **WHAT**：389.12 FPS、5.5 GFLOPs、2.433M 參數，91.7% 辨識準確度（三種姿勢：站立/趴臥/側臥）。

<a id="sec-5"></a>
### 5. CBR-YOLO（beef cattle, 2024）——⚠️僅摘要層級核對
- **WHY**：一般行為辨識模型在多變天氣場景（不同光照、能見度）下辨識能力不穩定。
- **HOW**：以 YOLOv8 為基礎的輕量化改進模型（CBR-YOLO），文字說明強調多場景天氣穩健性，但因全文存取受阻，改進細節（哪些模組被替換）無法進一步核實。
- **WHAT**：平均準確率 90.2%，優於 12 種 SOTA 物件偵測模型（此數字來自搜尋引擎摘要，未經全文核對，引用時應標註為二手資訊）。

<a id="sec-6"></a>
### 6. Cattle rumination keypoint detection（2024）
- **WHY**：反芻行為（rumination）是牛隻健康指標，但人工觀察或穿戴式裝置都不理想（費工或可能傷害動物），需要非接觸式自動化方案。
- **HOW**：改良版 YOLOv8-pose（加 SimSPPF 降低運算複雜度、ECA 注意力機制、RepGFPN 重構 neck），只偵測**鼻子與嘴巴兩個關鍵點**，計算兩點歐氏距離產生「咀嚼運動曲線」，低通濾波去噪後用多條件門檻峰值偵測數咀嚼次數。
- **WHAT**：96% mAP（比基礎版提升 2.8%）；10 支測試影片咀嚼次數平均誤差僅 5.6%（標準誤差 2.23%）；同時能估算反芻時長、咀嚼頻率。

<a id="sec-7"></a>
### 7. Lightweight cattle pose（reparameterization + attention, PLOS ONE 2024）
- **WHY**：傳統 heatmap-based 姿態估計方法計算複雜度高、偵測速度慢，不利複雜農場環境即時部署。
- **HOW**：EfficientRepBiPAN——訓練時多分支結構、推論時（依 RepVGG 方法論）轉換成單分支結構做重參數化；搭配 SimAM（3D 無參數注意力機制），統一通道與空間尺度、不額外增加參數量就能強化特徵判別力。
- **WHAT**：AP₀.₅ 比基礎 YOLOv8n-pose 提升 4.3%（達 92.3%），參數量減少 0.16M、運算量減少 1.0 GFLOPs，收斂速度也更快。
- 開源：僅資料集（figshare, DOI 10.6084/m9.figshare.25989082），程式碼未提及。

<a id="sec-8"></a>
### 8. Deep learning for visual animal monitoring：comprehensive review（2025）——⚠️僅摘要層級核對
- **WHY/HOW/WHAT**：全文存取受阻（403），僅能引用搜尋引擎摘要層級資訊——涵蓋偵測、追蹤、姿態估計、行為分類四個階段的動物監測文獻系統性回顧。因無法核對全文的具體 gap 分析內容，這篇在本文件中主要當作「這個領域現在怎麼分類自己的研究範疇」的框架參考，不引用其具體結論數字。

<a id="sec-9"></a>
### 9. Appearance-based canine multi-animal monitoring pipeline（Frontiers, 2026）
- **WHY**：藥物安全性試驗中，實驗室犬隻的行為評估仰賴技術員在房內觀察，費時費力又容易有人為偏誤，需要不影響動物福祉的連續、客觀監測方式。
- **HOW**：五段式管線——(1) YOLOv2 偵測所有犬隻（F1 94.8%）；(2) ResNet-18 辨識三種顏色反光背心做身分分類（80.3%）+ 修改版 Jonker-Volgenant 演算法搭配 Kalman filter 做跨幀跨鏡頭關聯；(3) ResNet-18 姿態分類器（躺/坐/站/起身/趴下五類，95.2%）；(4) ViT-S/16 行為分類器（進食/飲水，含時序上下文，94%）；(5) 擴充版 ViT-S/16 臨床觀察分類器（11 種臨床徵象如共濟失調、抽搐、震顫，79% top-1）。
- **WHAT**：群養動物重識別準確率 95.3–96.2%；AI 活動追蹤跟加速度計量測相關係數 r=0.965；180 萬+ 標註幀訓練，1800 萬+ 幀現場驗證；成功偵測到藥物誘發的共濟失調、不自主運動等徵象，與獸醫觀察吻合。

<a id="sec-10"></a>
### 10. Ultralytics YOLO11 dog-pose custom training（廠商技術文件，非同儕審查論文）
- **WHY**：YOLO11 本身沒有現成的犬類關鍵點模型，需要客製化訓練才能用於寵物姿態分析。
- **HOW**：提供 Dog-Pose 資料集（6,773 訓練 + 1,703 驗證圖片，**24 個關鍵點**）供客製化訓練 YOLO11-pose，並提出可結合穿戴式裝置（智慧項圈）做健康指標監測、即時分析姿態抓跛行/僵硬等異常動作的應用構想。
- **WHAT**：屬於技術文件/資料集發布，非正式實驗結果論文。

---

## 對本專案（cat_monitoring_system）的具體應用建議

依相關程度排序，每項都指出對應到專案裡的哪個檔案/模組：

### 高相關：可能直接帶來準確度或穩健度提升

**1. 用時序模型取代目前的 EMA 平滑來處理遮蔽（對應 T-LEAP，#2）**

目前 `STGCNConfig.KP_EMA_ALPHA`（`config.py`）+ `BehaviorTrackingConfig.CAT_MISSING_TOLERANCE_FRAMES` 的做法，本質上是「遮蔽時沿用最後一次偵測到的關鍵點」+「簡單指數平滑」。T-LEAP 的做法是**訓練一個明確吃時序資訊的姿態模型**，用鄰近幀的資訊主動推論被遮蔽的關鍵點，而不是被動沿用舊值——論文顯示這在人工遮蔽情境下比靜態模型提升最多 32.9% 準確度。

具體可以做的實驗：把 `models/stgcn_model.py`/YOLO 推論管線的關鍵點輸入，改成同時吃「前 N 幀」而非只吃當幀，用類似 T-LEAP 的方式重新設計遮蔽情境下的關鍵點插值（而不是目前「沿用最後一次偵測值」的簡化處理）。這跟 `processors/skeleton_quality_assessment.py` 現有的幾何合理性檢查是互補而非取代關係——SQA 抓的是「這幀關鍵點看起來不合理」，T-LEAP 式的方法解決的是「遮蔽時關鍵點該怎麼估」。

**2. 多貓身分辨識可以參考 DeepLabCut 的做法（對應 #3）**

`CatIdentityConfig`/`detectors/identity_verifier.py` 目前的**身分機制只做「這是不是目標貓」的封閉集合判定**（多貓時過濾非目標貓），**沒有真正的多動物身分追蹤**——沒有跨幀身分保持、沒有 tracklet 縫合，多貓同框時是靠 bbox IoU 空間延續 + 逐框評分挑出最像目標貓的那隻。（實作上 2026-08 起是 MobileNetV3-Small CNN 分類頭 + 信心門檻判 unknown + N 幀多數決，先前的色彩直方圖 H-S bin 比對已完全汰換。）DeepLabCut 的「局部 tracklet + 網路流全域縫合 + 無監督 metric learning 身分預測」是專門為「外觀相似的同物種動物互動」設計的，如果日後這個系統要支援多貓家庭（`docs/口試準備`裡提過的「個體辨識」是口試三大硬核問題之一），這篇的方法論比目前的封閉集合分類更適合處理「兩隻花色相近的貓在畫面裡持續靠近、互換位置」這種需要穩定分辨個體軌跡的情境。

**3. 用獨立關鍵點距離訊號交叉驗證 ST-GCN 的 lick 判斷（對應 #6）**

牛隻反芻論文的做法——只用兩個關鍵點（鼻/嘴）的距離變化曲線 + 峰值偵測就能算出咀嚼次數——結構上跟 `plugins/lick_stage/` 現在做的「鼻尖接觸梯形區域判定」概念相近，但反芻論文完全獨立於行為分類模型（ST-GCN）之外，是純幾何訊號。

這給了一個有意思的優化方向：目前 `lick_stage` plugin 判定的是「鼻子有沒有碰到身體特定區域」（空間判定），而 ST-GCN 判定的是「這個時間窗的整體動作像不像 lick」（時序模式判定）——兩者目前各自獨立運作、沒有互相驗證。可以參考反芻論文的峰值偵測手法，額外算一條「鼻子-目標區域距離」的時間序列訊號，跟 ST-GCN 的 lick 分類結果做交叉比對，用途上類似這個專案已經在用的「兩個獨立引擎互相驗證」設計哲學（第九～十三節文件裡個體化基線 Python/Node-RED 雙引擎的精神），只是這次是套用在行為偵測層而不是統計分析層。

### 中相關：架構層面的優化參考，非急迫

**4. 輕量化骨幹網路技術（對應 #4、#7）**

YOLO-BCD（重參數化 + 注意力模組，389 FPS）跟 PLOS ONE 那篇（EfficientRepBiPAN 重參數化 + SimAM 無參數注意力）都是「不犧牲太多準確度、明顯降低運算量」的具體技術路線。如果這個專案未來要往邊緣裝置部署（`config.py` 裡已經有 `_resolve_runtime_device()` 的 CPU/CUDA 自動偵測邏輯，暗示已經在考慮非 GPU 環境），這兩篇提供了具體、可實作的輕量化技術名稱（RepVGG 式重參數化、SimAM），比空泛地說「用更小的模型」更有操作性。

**5. 天氣/光照穩健性的資料增強策略（對應 #5，注意此篇僅摘要層級核對）**

CBR-YOLO 強調的「多場景天氣穩健性」提示了一個目前訓練管線（`0_train_gcn.py`、`stgcn_config.yaml`）可能沒有系統性處理的面向：訓練資料的光照/場景多樣性夠不夠。這篇的細節沒辦法核實，但方向本身值得對照——可以檢視現有訓練影片的光照條件分布，評估是否需要針對性擴增。

**2026-08-11 實測驗證（`tools/eval_lighting_distribution.py`）**：這個猜測被證實了。抽樣全部 501 支訓練影片（每支 5 幀，算 HSV V channel 均值當亮度代理指標）：全體平均亮度 137.8（0-255 尺度）、標準差 34.9，但「很暗（V<60）」的影片只有 6 支（1.2%），且這 6 支實際亮度也只是 27~59，沒有一支是真正的夜間/低光影片——資料集幾乎不含低光照條件。貓晨昏活動較多，居家監控情境下低光很常見，這是一個有數據佐證的真實缺口，可以直接寫進論文 Limitations/Future Work。腳本已建好可重複執行（`eval_results/lighting_distribution/` 底下累積時間戳快照），之後補拍夜間/昏暗場景影片可以重跑比對改善情況。

### 低相關：框架/敘事參考價值大於技術可直接套用

**6. 完整多階段管線的敘事框架（對應 #9）**

Frontiers 那篇犬隻臨床監測管線（偵測→身分/追蹤→姿態分類→行為分類→臨床徵象分類）是目前找到跟本專案架構最接近的完整類比——本專案是 YOLO 偵測+姿態 → ST-GCN 行為分類 → 個體化基線統計（`analytics/`），對應到它的前三段；它多出來的「臨床徵象分類器」（用監督式學習直接辨識共濟失調、震顫等 11 種臨床徵象）是本專案目前沒有、但概念上可以作為未來方向的一層——現在的健康風險評分（`fusion.py`）是統計偏差驅動（跟基線比對），不是學習型的異常徵象辨識；等累積足夠標記資料，可以參考這篇的做法加一層監督式分類器。這對論文寫作也有幫助：可以用這篇當作「本專案架構在文獻中的定位」的直接對照文獻。

**7. 同模型家族的關鍵點數量參考（對應 #10）**

本專案 YOLO11s-pose 用 17 個關鍵點（含 `ext_body_zones` 用到的尾根/尾中/尾尖）；Ultralytics 官方犬類資料集用 24 個關鍵點。差異可以當作評估「目前 17 點夠不夠支撐 7 區身體分區判定精度」的參考基準，但不是急迫問題（因為貓的關鍵點需求跟狗不完全一樣，不能直接套用）。

---

## 深入：即時串流下的遮蔽穩健關鍵點處理技術（2026-08-10 追加）

延續「高相關建議 1」（T-LEAP 式時序遮蔽處理），針對本專案**即時串流**這個限制條件（跟 T-LEAP 論文的離線步態分析情境不同，本專案 `frame_processor.py` 是逐幀處理直播影像，不能無限制地往後看未來幀）做了更深入的技術路線比較。

### 關鍵限制：本專案能接受多少「延遲」？

`STGCNConfig.SEQUENCE_LENGTH=16` 代表 ST-GCN 本來就要等 16 幀累積夠了才分類，`WINDOW_STRIDE=2` 也已經隱含一點延遲——這代表本專案的架構**不是嚴格的單幀零延遲系統**，已經有現成的緩衝視窗可以利用。真正的限制是：新技術不能引入「需要看到遠遠超過現有緩衝視窗的未來幀」這種離線批次處理的假設。

### 三條技術路線比較

| 路線 | 代表方法 | 是否需要重新訓練 | 即時可行性 | 開源狀態 |
|---|---|---|---|---|
| **A. 訊號處理層 Kalman Filter** | KeySORT（cattle, 2025）、`rat_tracking`、`kalman-tracker` | 不需要——套在現有 YOLO 輸出後面做後處理 | ✅ 設計上就是線上即時（KeySORT 全名 Keypoint Simple and Online Realtime Tracking） | KeySORT 論文本身未查到公開碼；`rat_tracking`（PyTorch，鼠類關鍵點 Kalman/Particle filter）✅ 開源可參考 |
| **B. 端到端時序模型** | T-LEAP（cow, 2021） | 需要——3D 卷積架構要重新訓練 | ⚠️ 不確定——`seq_length` 視窗是否只看過去幀，原始碼 README 沒寫清楚，需要直接讀程式碼才能確認 | ✅ https://github.com/hrussel/t-leap（PyTorch） |
| **C. 多模型 Ensemble 平滑** | EKS Ensemble Kalman Smoother | 不需要重新訓練，但需要**多組**已訓練模型的預測結果做輸入 | ❌ 明確是離線批次系統（官方文件寫明：不是即時因果濾波器，要先產生完整序列的多組預測才能跑） | ✅ https://github.com/paninski-lab/eks |

### 補充文獻：Towards Multi-Modal Animal Pose Estimation（2024 survey, 176 篇）

搜尋過程中找到一篇更新、更完整的動物姿態估計綜述，可以取代原本因 403 被擋的 ScienceDirect 那篇（#8）當作「這個領域怎麼分類自己的時序方法」的框架參考：

- **標題**：Towards Multi-Modal Animal Pose Estimation: A Survey and In-Depth Analysis
- **作者**：Deng, Q., Deb, O., Patel, A., Rupprecht, C., Torr, P., Trigoni, N., & Markham, A.（2024）
- **連結**：https://arxiv.org/abs/2410.09312 ｜ 開源專案頁：https://github.com/ChennyDeng/MM-APE
- 這篇把「用時序資訊做動物姿態估計」分成兩大類：**後處理式**（先逐幀預測，再做離線批次時序平滑，EKS 屬於這類）跟**端到端時序方法**（LEAP 系列擴展、光流法、半監督/無監督影片自適應，T-LEAP 屬於這類）。這個分類方式正好對應上面表格的路線 A/C（後處理）vs. 路線 B（端到端）。

### 具體建議：分階段導入，不要一次跳到最複雜的方案

1. **第一步（低成本、可以馬上做）**：把 `STGCNConfig.KP_EMA_ALPHA` 目前的簡單指數平滑，換成逐關鍵點的 Kalman filter（等速度或等加速度模型）。這不需要重新訓練任何模型，純粹是訊號處理層的替換，`rat_tracking` repo 的做法可以當實作參考。相較於現在「遮蔽時沿用最後一次偵測值」，Kalman filter 能給出「預測位置 + 不確定性」，且不確定性本身可以額外拿來當 `skeleton_quality_assessment.py` 幾何合理性檢查的輔助訊號（目前 SQA 判斷的是「這幀看起來合不合理」，Kalman 的協方差判斷的是「這個預測有多不可信」，兩者互補）。
2. **第二步（中成本，先驗證第一步不夠用才做）**：如果 Kalman filter 的效果仍不足以應付長時間遮蔽（例如貓完全被家具擋住超過 `CAT_MISSING_TOLERANCE_FRAMES`），再考慮 T-LEAP 式的端到端時序模型——但**動手前必須先讀 `t-leap` 原始碼確認 `seq_length` 視窗是不是純過去幀**，否則移植到即時串流架構會遇到「需要未來幀」的根本性障礙，不是調參數就能解決的問題。
3. **第三步（追蹤觀察，非現在要做）**：KeySORT（2025）從論文名稱跟摘要看是專為這個情境設計的方法（線上即時、免 bounding box、直接適配關鍵點），但目前查無公開程式碼，值得列入之後的文獻追蹤名單（可用這個技能的 `monitoring_agent` 概念手動定期查一次），一旦釋出程式碼會是比自己刻 Kalman filter 更省工的選項。

---

## 誠實限制

- 本次搜尋非窮盡式系統性回顧，是聚焦「YOLO-pose + 應用/開源」的定向掃描（three-way-scan 模式），不保證涵蓋該領域全部重要文獻。
- #5、#8 兩篇因網站阻擋自動化存取，僅核對到摘要層級，未讀全文，文件中已個別標註。
- 「深入」章節的 T-LEAP（PDF 過大，工具無法解析二進位內容）與 KeySORT（同樣是 PDF 二進位問題）也只核對到 arXiv 摘要頁層級，沒有讀到完整方法論細節——尤其 T-LEAP 的 `seq_length` 視窗方向（純過去幀 vs. 含未來幀）**這個對本專案最關鍵的問題，最終沒有查證到明確答案**，文件裡已標記為「動手前必須先讀原始碼確認」，不是確定的結論。
- 「開源狀態」是搜尋當下（2026-08-10）查證結果，部分論文可能後續才釋出程式碼，屬於時效性資訊。
- 本文件由 AI 研究輔助工具（WebSearch/WebFetch）產生，所有連結均為搜尋當下實際核對過的真實來源，未經人工二次查證前不建議直接寫入論文正式引用，需依各期刊格式規範重新核對完整書目資訊（作者全名、卷期頁碼等）。

---

## 後續實作進度（2026-08-11 更新）

延續「深入」章節路線 A（Kalman filter 後處理，不需要重新訓練），實際落地情況：

- **`models/keypoint_kalman.py`**：逐關鍵點等速度 Kalman filter 實作完成，`models/tests/test_keypoint_kalman.py`（7 則單元測試，純合成軌跡驗證）全數通過，含關鍵情境驗證（視窗尾端遮蔽時明顯優於線性插值的 flatline 行為）。
- **`tools/eval_accuracy_smoothing_compare.py`**（+ 共用核心 `tools/_smoothing_eval_common.py`）：一支腳本一次跑完「無平滑 vs. Kalman 平滑」在標記測試影片上的準確度比較（同一 checkpoint、同一批影片，YOLO 偵測只跑一次），比較設定寫死在 `SMOOTHING_CONFIGS` 清單管理，不走命令列參數。已用真實 GPU 模型 + 真實影片跑過端到端煙霧測試確認正確（YOLO 呼叫次數、預測結果、window 數皆核對過）。**尚未執行過完整 24 支影片的正式比較跑**，還沒有「Kalman 有沒有幫助」的實際數據。
- **`stgcn_config.yaml` + `tools/0_train_gcn.py`**（路線 B 的準備工作）：新增 `SMOOTHING_KIND`（`none`/`ema`/`kalman`）、`KALMAN_PROCESS_NOISE`、`KALMAN_MEASUREMENT_NOISE` 設定，`CatSkeletonDataset`／`train_model()` 已能依 `smoothing_kind` 分派到 Kalman 平滑（而非只有 EMA），預設值（`SMOOTHING_KIND: "ema"` + `KP_EMA_ALPHA: 1.0`）跟這次改動前的行為完全一致，已用真實 505 支骨架 JSON 資料跑過端到端驗證（ema 與 kalman 兩種設定都能正常載入、序列數一致、非法 `smoothing_kind` 會明確拋錯而非靜默忽略）。

**刻意還沒做的部分**：`run_kp_ema_ablation()` 這個消融實驗執行器（會實際觸發多次完整訓練、耗費 GPU 時數）目前仍然只支援掃 `ABLATION_KP_EMA_ALPHAS` 這個 EMA alpha 清單，沒有對應的 `run_smoothing_ablation()` 可以把 Kalman 設定也納入消融比較；`train_model()` 的檔名/console 標記邏輯（`alpha_tag`）也還沒擴充成能顯示 Kalman 參數（目前手動傳 `smoothing_kind="kalman"` 訓練時，這些資訊只會出現在 `params_snapshot.json` 裡，不會反映在檔名上）。這是刻意的分階段順序：`eval_accuracy_smoothing_compare.py` 的推論時驗證還沒跑出實際數據支持「Kalman 有幫助」之前，先不投入會實際耗費大量 GPU 時數的訓練消融執行器擴充。

### Kalman 參數（Q/R）校準（2026-08-11）

原本 `process_noise=1.0`／`measurement_noise=5.0` 是沒有依據的保守猜測。用真實 YOLO 偵測結果實測校準：

- **measurement_noise（R）**：拿 `stop` 類別影片（幾乎靜止）算高信心幀之間的座標抖動變異數 → 實測 ≈70（像素²，套用點在 YOLO 原始像素座標，`flip_normalize`/`normalize_skeleton_coords` 之前）。
- **process_noise（Q）**：拿 `shake` 類別影片（快速動作）算「真實軌跡 vs. 等速度模型預測」的殘差變異數 → 實測 ≈94（像素²）。

原本的 1.0/5.0 在絕對尺度上小了快兩個數量級，相對於實際關鍵點座標（數百像素量級）的變化幅度，會讓 Kalman 的協方差收斂過快、對新量測反應過度遲鈍——這不是「平滑強度剛好而已比較保守」的差異，是尺度沒對齊的問題。已把 `models/keypoint_kalman.py`／`stgcn_config.yaml`／`tools/eval_accuracy_smoothing_compare.py` 的預設值/範例統一改成 Q=90、R=70 這組實測起點。

**已知限制**：這只是單支 stop 影片＋單支 shake 影片的一次性測量，沒有每個關節分開估、沒有跨多支影片取統計穩健值。而且 Q/R 的比值（實測 ≈1.34，遠高於原本猜的 0.2）證實了先前的定性擔心：shake 需要的平滑強度跟 stop 差很多，用同一組全域參數是妥協。`eval_accuracy_smoothing_compare.py` 的 `KALMAN_PARAM_SWEEP` 已經改成圍繞這個實測起點往「較強」「較弱」兩個方向各掃一組，之後真正決定要用哪組數值，應該看這個 sweep 在標記影片上的每類別準確度，而不是只看這裡的統計量。

### 路線 A 結論：Kalman 平滑沒有找到顯著幫助（2026-08-11 定案）

三階段實驗，由淺入深：

1. **推論期套用（不重訓）**：`eval_accuracy_smoothing_compare.py` comparison_002/003，baseline 模型（122，無平滑訓練）套用 Kalman 平滑，overall accuracy/macro F1 隨平滑強度**單調變差**（越平滑越傷），主因 shake 這種快速動作訊號被抹平。
2. **重新訓練＋匹配評估（單一參數）**：`stgcn_config.yaml` 切到 `SMOOTHING_KIND: "kalman"`（Q90/R70）重訓出 126，跟 122 各自用匹配前處理比較（`eval_gcn_compare.py` comparison_024）——overall accuracy 統計上打平（McNemar p=1.0，n_discordant=7），shake 的退步消失了（證明 1. 的退步是分布不匹配造成的副作用），但也沒有換來實質提升。
3. **三組候選參數各自重訓＋匹配評估**：再訓 127（Q45/R140 較強）、128（Q180/R35 較弱），四模型（122/126/127/128）一起匹配比較（comparison_025）——**Q90/R70（126）是三組 Kalman 裡表現最好的**（最初用真實 YOLO 偵測雜訊校準出來的參數，不是亂猜的），但跟 122 比較 McNemar 檢定仍不顯著。

**結論**：不管是推論期套用、單一參數匹配重訓、還是三組參數各自匹配重訓，Kalman 平滑都沒有找到顯著贏過不平滑的證據。`stgcn_config.yaml` 已改回 `SMOOTHING_KIND: "ema"`（等同不平滑，恢復現行部署行為），`KALMAN_PROCESS_NOISE`/`KALMAN_MEASUREMENT_NOISE` 保留 90.0/70.0（126 那組，三組裡最好的）當紀錄。路線 A 到此收尾，不再投入。詳細數字見 memory: `project_kalman_smoothing_eval`。

**重要澄清**：以上實驗測的都是「Kalman 平滑對整體辨識準確率的影響」，不是路線 A 原本設想的「遮蔽情境下的穩健度」——沒有另外切出一組刻意遮蔽/長時間偵測失敗的測試子集來驗證「Kalman 是否至少在真的發生遮蔽時比線性插值更好」。如果之後還想從「抗遮蔽」這個原始動機切入，需要另外設計含遮蔽片段的測試集，不能直接套用這裡的結論。

### 路線 B：使用者決定不採納（2026-08-11）

路線 A 收尾後，路線 B（T-LEAP 式端到端時序模型）是文件原訂的下一階段，但使用者決定到此為止、不投入——**尚未進行到「讀 `t-leap` 原始碼確認 `seq_length` 視窗方向」這一步就決定不採納**，所以 T-LEAP 對本專案是否技術可行（純過去幀 vs. 含未來幀）目前仍是未查證狀態，不是「查證後發現不可行」，純粹是優先順序/資源考量下的決定。

**三條路線最終狀態**：A（Kalman）已測試但未找到顯著幫助；B（T-LEAP）使用者決定不採納；C（EKS）文件查證階段即已排除（離線批次系統，不適用即時串流）。「即時串流下的遮蔽穩健關鍵點處理」這個優化方向到此收尾，現行做法（`interpolate_missing` + EMA/`KP_EMA_ALPHA=1.0`）維持不變。
