# 資料集切分示意圖（Dataset Split Diagram）

更新：2026-09-04（依實際程式碼核對修正：切分方式改為「僅 TRAIN/VAL 兩路、以影片為單位」，移除與程式碼不符的 TRAIN/VAL/TEST 三路影像切分；行為類別占比數字仍為待補資料，見下方「⚠ 尚未驗證」說明）

> [!IMPORTANT]
> **⚠ 尚未驗證 / 待補**：下方「行為分布」區塊各類別的張數與百分比（`1500/300/200/…`）目前是**未經查證的既有佔位數字**，本次核對範圍內沒有取得實際資料集（`skeletons/` 資料夾在本次核對環境中不存在），因此**沒有修改或杜撰這些數字**，僅原樣保留並在此註記其真實性尚待用實際資料集統計值確認，撰寫論文前務必重新核實。

> [!NOTE]
> **✅ 已核對並修正的部分：切分方式本身**
>
> 原圖上半部畫的是 **TRAIN／VAL／TEST 三路、以「張數（Images）」為單位、75%／15%／10%** 的切分，但這與 `cat_monitoring_system/tools/0_train_gcn.py`（`split_train_val_indices()`）與 `stgcn_config.yaml`（`TRAIN_TEST_SPLIT: 0.15`）的實際作法不符：
>
> - ST-GCN 訓練管線**只切成 TRAIN／VAL 兩路，沒有獨立的 TEST 集**。
> - 切分單位是**影片（video）**，不是張數／影格：每支骨架 JSON（一支影片）先整支歸入 train 或 val，之後才在各自集合內部切滑動窗序列，避免同一支影片的重疊視窗同時落在 train 與 val 造成資料洩漏（防 leakage）。
> - 切分比例由 `TRAIN_TEST_SPLIT: 0.15` 決定：**保留約 15% 影片作為 VAL，其餘約 85% 為 TRAIN**（非原圖的 75/15/10）。
> - 採用 `sklearn.train_test_split` 依「影片主要行為標籤」做 stratify（類別分層抽樣），`RANDOM_SEED=42`；若某類別影片數不足以分層則自動退回 random split；若某類別有 ≥2 支影片卻沒被分到 val，會自動從 train 移 1 支到 val，盡量讓每個可分類別在 val 都有代表。
>
> 下方 mermaid 圖已依此修正為兩路切分，僅標示核對過的切分**比例**，不附加未經驗證的影像張數。

```mermaid
flowchart TB

%% =========================
%% Dataset Split
%% =========================

subgraph SPLIT["Dataset Split（以影片為單位，見上方核對說明）"]
direction TB

    subgraph TOP[" "]
    direction LR

        A["TRAIN"]:::train
        B["VAL"]:::val

    end

    subgraph BOTTOM[" "]
    direction LR

        D["≈85% 影片<br/>(TRAIN_TEST_SPLIT=0.15)"]:::info
        E["≈15% 影片<br/>(TRAIN_TEST_SPLIT=0.15)"]:::info

    end

    A -.- D
    B -.- E

end

%% =========================
%% Behavior Distribution
%% =========================

subgraph DIST["資料分布（Behavior Distribution，⚠ 數字尚待以實際資料集核實，見上方說明）"]
direction LR

    G["行走 walk<br/><br/>1500 張<br/>75%"]:::walk
    H["舔舐 lick<br/><br/>300 張<br/>15%"]:::lick
    I["搔抓 scratch<br/><br/>200 張<br/>10%"]:::scratch
    J["甩頭 shake<br/><br/>. 張<br/>.%"]:::shake
    K["靜止 stop<br/><br/>0 張<br/>0%"]:::stop

end

%% =========================
%% Notes
%% =========================

NOTE["甩頭行為具有高度一致且明顯的動態特徵，<br/>因此不額外設計特殊特徵工程。<br/>（2026-09-04 核對：stgcn_model.py 目前確實沒有 shake 專屬的特徵/分類頭，<br/>此說法與現況相符）"]:::note

%% =========================
%% Data Collection
%% =========================

subgraph SOURCE["資料蒐集方式（Data Collection Method）"]
direction LR

    L["網路蒐集"]:::source
    M["自行拍攝"]:::source
    N["委託親朋好友"]:::source
    O["AI 生成"]:::source

end

%% =========================
%% Style
%% =========================

classDef train fill:#008CFF,color:#FFFFFF,stroke:#66C2FF,stroke-width:4px,font-size:30px,font-weight:bold
classDef val fill:#00D26A,color:#FFFFFF,stroke:#7DFFB2,stroke-width:4px,font-size:30px,font-weight:bold

classDef info fill:none,color:#FFFFFF,stroke:none,font-size:22px,font-weight:bold

classDef walk fill:#1E88E5,color:#FFFFFF,stroke:#90CAF9,stroke-width:3px,font-size:22px,font-weight:bold
classDef lick fill:#8E24AA,color:#FFFFFF,stroke:#CE93D8,stroke-width:3px,font-size:22px,font-weight:bold
classDef scratch fill:#E53935,color:#FFFFFF,stroke:#FFCDD2,stroke-width:3px,font-size:22px,font-weight:bold
classDef shake fill:#FB8C00,color:#FFFFFF,stroke:#FFE0B2,stroke-width:3px,font-size:22px,font-weight:bold
classDef stop fill:#546E7A,color:#FFFFFF,stroke:#B0BEC5,stroke-width:3px,font-size:22px,font-weight:bold

classDef source fill:#263238,color:#FFFFFF,stroke:#90A4AE,stroke-width:3px,font-size:20px,font-weight:bold

classDef note fill:#111111,color:#FFFFFF,stroke:#AAAAAA,stroke-width:2px,font-size:18px

style A width:140px,height:140px
style B width:140px,height:140px

style SPLIT fill:none,stroke:none
style DIST fill:none,stroke:none
style SOURCE fill:none,stroke:none
style TOP fill:none,stroke:none
style BOTTOM fill:none,stroke:none

linkStyle 0 stroke:#66C2FF,stroke-width:2px,stroke-dasharray: 3 3
linkStyle 1 stroke:#7DFFB2,stroke-width:2px,stroke-dasharray: 3 3
```

---

## 補充：與程式碼對照的關鍵事實（2026-09-04 核對）

| 項目 | 圖中原內容 | 程式碼實際行為 | 依據 |
|---|---|---|---|
| 切分路數 | TRAIN / VAL / TEST 三路 | 僅 TRAIN / VAL 兩路，**無獨立 TEST 集** | `0_train_gcn.py: split_train_val_indices()` |
| 切分單位 | 「張（Images）」 | **影片（video）**，同一支影片的所有滑動窗序列必定落在同一路，防止 data leakage | 同上 |
| 切分比例 | 75% / 15% / 10% | **≈85% / ≈15%**（`TRAIN_TEST_SPLIT: 0.15`） | `stgcn_config.yaml` |
| 抽樣方式 | 未說明 | 依影片主要行為標籤做 stratify（類別分層），`RANDOM_SEED=42`；影片數不足時自動退回 random split；缺類別時自動從 train 移 1 支影片補進 val | `0_train_gcn.py: split_train_val_indices()` |
| 行為類別 | 行走／舔舐／搔抓／甩頭／暫定（5 類） | 與程式碼 `BEHAVIOR_PREFIXES`（`walk/lick/scratch/shake/stop`，共 5 類）一致；「暫定」已改標為「靜止 stop」，因為 `stop` 在目前程式碼中是正式定義、已有訓練模型使用的類別，不是尚未定案的暫定項目 | `stgcn_config.yaml: BEHAVIOR_PREFIXES` |
| 各類別張數/占比 | 具體數字（1500/300/200/…） | **無法從程式碼驗證**——這是資料集內容統計，不是程式邏輯，需要直接統計 `SKELETON_DATA_FOLDER` 底下的實際資料 | 本次核對環境無法存取 `skeletons/` 資料夾 |
