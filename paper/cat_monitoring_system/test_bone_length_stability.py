"""
Bone Length Stability Analysis — 骨段長度一致性診斷工具
多影片版（1~10 支）：依序處理每支影片，結果分開存，最後再疊圖比較。
支援兩種執行模式（見下方「執行模式」）。

════════════════════════════════════════════════════════════════════
目的
════════════════════════════════════════════════════════════════════
為「GCN 分類為主、幾何判斷為輔」的雙重判定機制找適當門檻值。

目前有五個逐窗口指標，組成一個「Skeleton Quality Assessment（SQA）」架構，
分兩個層級：

Spatial Consistency（看某一幀本身合不合理）：
  - torso_ratio：Chest-Hip 距離 ÷ YOLO bbox 對角線。bbox 是獨立於關鍵點
    的另一個輸出頭，不受關鍵點集體塌陷影響，可以當外部參照。
  - midback_offset_ratio：MidBack 偏離 Chest-Hip 虛擬中點的距離 ÷
    Chest-Hip 距離。MidBack 標在拱背最高點，跟 Chest/Hip 天生形成三角形，
    偏移量超過解剖合理性上限視為可疑。

Temporal Consistency（看連續幀之間變化合不合理）：
  - midback_angle_jitter：Chest-MidBack-Hip 夾角在連續幀之間的變化量
    （度/幀），不看絕對角度大小（會隨拱背弧度/姿勢變化），只看是否
    忽大忽小亂跳。
  - torso_ratio_jitter（Skeleton Scale Jitter）：torso_ratio 逐幀值在
    相鄰幀之間的變化量，抓「整體骨架尺度忽大忽小」。
  - bone_length_oscillation（Bone Length Oscillation）：各骨段長度
    （正規化座標下）在相鄰幀之間的變化量，取所有納入分析骨段裡的最大值，
    抓「反覆縮放/跳動」而非平均長度，跟骨段長度的離散度（CV）是不同
    統計量。

本腳本重用 models/stgcn_model.py 既有的缺值補值函式（interpolate_missing），
確保這裡看到的數值跟未來真正接進 behavior_classifier.py 的判斷邏輯一致。

════════════════════════════════════════════════════════════════════
執行模式
════════════════════════════════════════════════════════════════════
用 RUN_MODE 切換（留 None 則執行時用選單詢問，輸入 1 或 2）：
  [1] "batch" 背景統計分析——不開視窗、不畫骨架/面板、不等待按鍵，單純
      依序讀取每一幀跑偵測＋節流算 overlay，跑完整支/整批影片，輸出
      CSV/圖表（單支影片圖表＋多影片比較圖＋門檻分析圖），是唯一會產生
      統計結果的模式，速度最快，適合放著跑很多支或很長的影片。
  [2] "gui"   GUI 視覺偵測——純觀看/測試，開視窗即時顯示骨架、穩定度
      面板（五個 SQA 指標用綠色=正常、紅色=異常二元顯示）、MidBack 夾角，
      可 [space] 暫停、暫停後 [a]/[d] 逐幀回看，
      適合肉眼核對單一片段哪裡出問題——不收集資料、不輸出 CSV、不產生
      任何圖表。單支影片播完會自動從頭循環播放，直到按 [1]/[2] 才切換
      到上一支/下一支影片（可循環繞回），跟 1_run_video_inference.py 的
      LOOP_PLAYBACK + 1/2 切換邏輯一致。
兩種模式共用同一套 INPUT_MODE/VIDEO_PATHS/VIDEO_FOLDER 設定跟候選門檻，
只是「有沒有開視窗」「有沒有輸出統計結果」不同。

════════════════════════════════════════════════════════════════════
使用方式
════════════════════════════════════════════════════════════════════
  1. 設定 INPUT_MODE 選擇輸入方式（二選一）：
       "paths"  → 手動列出下方 VIDEO_PATHS 裡的 1~10 支個別影片路徑
       "folder" → 改指定 VIDEO_FOLDER 一個資料夾，自動抓裡面的影片
                  （依檔名排序，超過 10 支只取前 10 支）
     建議至少一支是追蹤穩定的「正常」影片、一支是已知會被誤判成
     walk/shake 的「抖動」影片，這樣最後的疊圖比較才看得出差異
  2. 執行腳本：依序處理每支影片；GUI 模式視窗右上角面板即時顯示各項
     指標數值（見「執行模式」的顏色說明）；背景模式沒有畫面，只在
     終端機印進度
  3. GUI 模式單支影片會循環播放，播放時可按 [space] 暫停、[1]/[2] 切換
     到上一支/下一支影片、[q] 結束整個檢視；暫停後可用 [a]/[d] 逐幀
     後退/前進檢視（不重跑 YOLO，只讀取快取）
  4.（僅背景模式）每支影片各自的 CSV + 五指標圖表（torso_ratio、
     midback_offset_ratio、midback_angle_jitter、torso_ratio_jitter、
     bone_length_oscillation 各自的時序線圖 + 分佈直方圖）分開存到
     OUTPUT_DIR / run_YYYYMMDD_HHMMSS / <影片檔名>/
  5.（僅背景模式）全部影片跑完後，額外產生一張「多影片疊圖比較」：把
     每支影片的時序線與分佈直方圖疊在同一張圖上（不同顏色），方便直接
     比較「正常」跟「抖動」影片的落點差在哪裡，存到 run 根目錄的
     comparison_chart.png，並在最後互動顯示
"""
import sys
import csv
from collections import deque
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # 讓 config.py（repo 根目錄）可被 import

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    compute_bone_feature,
)
from utils.constants import BEHAVIOR_COLORS, LOW_CONF_ID
from utils.helpers import get_behavior_name
from config import BehaviorTrackingConfig

# ╔══════════════════════════════════════════════════════════════════╗
# ║          使 用 者 設 定 區（每次執行前只需修改此區）             ║
# ╚══════════════════════════════════════════════════════════════════╝

# 執行模式二選一，用 RUN_MODE 切換（留 None 則每次執行時用選單詢問）：
#   "batch" → 背景統計分析：不開視窗、不畫骨架/面板、不等待按鍵，單純跑完
#             整支/整批影片，直接輸出 CSV/圖表，速度最快，適合放著跑很多
#             支或很長的影片。
#   "gui"   → GUI 視覺偵測：開視窗即時顯示骨架、穩定度面板、MidBack 夾角，
#             可 [space] 暫停、暫停後 [a]/[d] 逐幀回看，適合肉眼核對單一
#             片段哪裡出問題。
# 兩種模式共用下面同一套 INPUT_MODE/VIDEO_PATHS/VIDEO_FOLDER 設定跟門檻，
# 差別只在於「有沒有開視窗」，輸出的 CSV/圖表格式完全一樣。
RUN_MODE = None  # None（啟動時詢問）、"batch" 或 "gui"

# 兩種輸入模式二選一，用 INPUT_MODE 切換：
#   "paths"  → 用下面 VIDEO_PATHS 手動列出的 1~10 支個別影片檔案路徑（原本的做法）
#   "folder" → 改成指定 VIDEO_FOLDER 一個資料夾，自動抓出裡面的影片檔（依檔名排序，
#              最多取前 10 支，超過會印警告並截斷）——不用手動一支一支列路徑
INPUT_MODE = "paths"  # "paths" 或 "folder"

VIDEO_PATHS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\shake5772.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\暫存\frontal_walk\frontal_walk5566.mp4",
    r"C:\Users\homec\OneDrive\圖片\貓咪\自行拍攝\家貓\video_rating\B\2026-07-06 04_59_55.mp4",
    r"C:\Users\homec\Downloads\新增資料夾\(858) 公園的小貓慢慢靠近…原來只是想被摸摸，瞬間被療癒了🐾 - YouTube - Google Chrome 2026-07-04 10-36-05.mp4",
    # r"C:\Users\homec\OneDrive\圖片\貓咪\自行拍攝\影片\7月7日 (6).mp4",  # 檔案不存在，暫時跳過
]
# 1~10 支「單一影片檔案」路徑（不是資料夾），依序處理。建議至少放一支
# 追蹤穩定的正常影片、一支已知會誤判成 walk/shake 的抖動影片做對照。
# 只有 INPUT_MODE = "paths" 時才會用到這份清單。

VIDEO_FOLDER = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\暫存\frontal_walk"
VIDEO_FOLDER_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
# 只有 INPUT_MODE = "folder" 時才會用到：資料夾底下（不含子資料夾）
# 副檔名符合上面清單的影片檔，依檔名排序後最多取前 10 支。

YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_124.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640

# ===== ST-GCN 行為分類（只取分類結果 + GUI 顯示標籤，用來跟骨架穩定度面板
# 對照觀察——這裡不做 1_run_video_inference.py 那套 hysteresis 顯示延遲／
# 累積時長／發生次數統計／CSV 報表，那些是正式推論才需要的邏輯，這支診斷
# 腳本只要能看到「GCN 這一刻覺得是什麼」就夠了。feature_mode/in_channels
# 留給 BehaviorClassifier 依 config.py 的 STGCNConfig.FEATURE_MODE 及
# checkpoint 的 bn_input 通道數自動判斷/校正，不在這裡寫死。=====
STGCN_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_results\run_122_xy_conf_v_bone_att_on\122_best_model.pth"
STGCN_NORMALIZE = True
BEHAVIOR_MIN_CONFIDENCE = BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD

OUTPUT_DIR = Path(r"C:\ai_project\paper\output\bone_length_stability")

SEQUENCE_LENGTH = 16          # 跟 ST-GCN 實際推論窗口一致（T=16）
BONE_CONF_THRESHOLD = 0.3     # 骨段兩端關鍵點信心低於此值，該幀不納入該項計算

# midback_offset_ratio／midback_angle_jitter 這兩個指標用固定幀數門檻
# （不隨窗口長度 SEQUENCE_LENGTH 縮放），只要窗口內有效樣本數低於這裡設的
# 幀數，就標記為 NaN，不採信樣本太少、雜訊沒被平均掉的統計量。分開兩個變數
# 管理是因為兩者的「樣本」單位不同：midback_offset_ratio 數的是「幀數」，
# midback_angle_jitter 數的是「相鄰幀對」（角度差是兩幀之間算出來的）。
MIN_VALID_FRAMES_MIDBACK_OFFSET =1              # midback_offset_ratio 至少要有幾幀有效才採信
MIN_VALID_FRAME_PAIRS_MIDBACK_ANGLE_JITTER = 1   # midback_angle_jitter 至少要有幾組相鄰有效幀對才採信

# torso_ratio_jitter／bone_length_oscillation 這兩個指標一樣是「相鄰幀對」
# 的統計量，用同一套固定幀對數門檻管理邏輯。
MIN_VALID_FRAME_PAIRS_TORSO_RATIO_JITTER = 3    # torso_ratio_jitter 至少要有幾組相鄰有效幀對才採信
MIN_VALID_FRAME_PAIRS_BONE_OSCILLATION = 3      # bone_length_oscillation 每根骨段至少要有幾組相鄰有效幀對才採信

# 面板/圖表用的候選門檻線（2026-07-09 已用 shake5772/walk_13535868/
# 7月3日scratch(2)(1) 三支影片的 batch 結果校準過一輪，見下方各項調整依據）

# ============================================================================
# ===== torso_ratio（下界）── 軀幹相對 bbox 塌陷 =====
# ============================================================================
# torso_ratio（胸髖距離 ÷ bbox 對角線）專屬的候選門檻——數值「低於」這個
# 值才視為可疑（軀幹相對 bbox 塌陷）。原本猜 0.15，但實測 pooled 資料最小值
# 是 0.159，這個門檻從未被觸發過（4 支影片、468 幀，flag rate 恆為 0%）。
# 用 ECDF 拆開來看，shake5772 有約 35% 的幀落在 0.28 以下，其餘乾淨影片
# （walk/scratch）在 0.28 以下幾乎沒有分布，於是調高到 0.28：shake5772
# flag rate 34.8%、其餘影片仍 0%，乾淨分離。用 torso_ratio_threshold_chart
# 持續驗證/調整。
CANDIDATE_TORSO_RATIO_THRESHOLD = 0.28

# ============================================================================
# ===== torso_ratio_inflated（上界）── 尺度正規化失效、chest-hip 被撐大 =====
# ============================================================================
# torso_ratio 上界候選門檻：跟上面的下界門檻是同一個原始比例（chest-hip /
# bbox 對角線），只是換一個方向。下界抓的是「軀幹相對 bbox 塌陷」；這個
# 上界抓的是另一種尺度正規化失效模式——chest 或 hip 其中一個關鍵點整個
# 窗口都偵測不到時，interpolate_missing() 會把它填成畫面原點 (0,0)，中心化
# 後這個假座標離 mid_back 通常很遠，量出來的 chest-hip 距離反而會被撐得比
# bbox 本身還誇張，不會被下界門檻抓到。先猜一個起點，之後用
# torso_ratio_inflated_threshold_chart 驗證/調整（沿用同一份 torso_ratio
# 資料，只是換個門檻方向分析，不需要另外收集資料）。
CANDIDATE_TORSO_RATIO_UPPER_THRESHOLD = 0.55

# ============================================================================
# ===== midback_angle_jitter ── Chest-MidBack-Hip 夾角的逐幀抖動程度 =====
# ============================================================================
# midback_angle_jitter（Chest-MidBack-Hip 夾角，連續幀之間的抖動程度，單位
# 度/幀）專屬的候選門檻。原本考慮用「窗口平均角度」設絕對門檻，但 MidBack
# 標在背部拱起最高點、跟 Chest/Hip 天生是三角形（不共線），角度本身會隨
# 貓的拱背弧度/姿勢大幅變化——絕對值沒有一個放諸四海皆準的「正常範圍」，
# 用絕對門檻很容易把姿勢不同的正常貓誤判成異常。改成量「連續幀之間角度
# 變化了多少」：不管姿勢本身的絕對角度是多少，只要偵測穩定，這個角度在
# 短短 16 幀窗口內應該平滑變化，不會忽大忽小亂跳；跳動大代表偵測不穩定，
# 跟絕對角度無關，方向是數值「越大」越可疑。先猜一個起點，之後用真實
# 影片跟 midback_angle_jitter_threshold_chart 驗證/調整。
CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD = 5.0

# ============================================================================
# ===== midback_offset_ratio ── MidBack 偏離 Chest-Hip 虛擬中點的比例 =====
# ============================================================================
# midback_offset_ratio（MidBack 偏離 Chest-Hip 虛擬中點的距離 ÷ Chest-Hip
# 距離本身）專屬的候選門檻。同樣沒有標註分布統計出來的「正常基準值」，這裡
# 抓的是比較粗略的解剖合理性上限：MidBack 是拱背的頂點，偏移量正常情況下
# 不應該超過 Chest-Hip 這條底邊自身的長度（等於 1.0 倍），超過這個量級比較
# 可能是關鍵點被誤判/飄移到其他位置，而不是單純拱背弧度大。方向是數值
# 「越大」越可疑。原本猜 1.0，但實測 ECDF 顯示乾淨影片（walk/scratch/對照
# 組）的最大值都不超過 0.75，跟 shake5772 的長尾（一路延伸到 1.47）之間有
# 明顯空隙，於是調低到 0.8：shake5772 flag rate 從 14.0% 提升到 25.0%，其餘
# 影片仍是 0%，沒有誤傷。用 midback_offset_threshold_chart 持續驗證/調整。
CANDIDATE_MIDBACK_OFFSET_THRESHOLD = 0.8

# ============================================================================
# ===== torso_ratio_jitter（Skeleton Scale Jitter）── 整體骨架尺度的逐幀抖動 =====
# ============================================================================
# torso_ratio_jitter（Skeleton Scale Jitter）：torso_ratio 逐幀值在相鄰幀
# 之間的變化量。跟 midback_angle_jitter 同樣道理——不看 torso_ratio 絕對值
# 本身（那個已經有 CANDIDATE_TORSO_RATIO_THRESHOLD 在管），只看窗口內是否
# 忽大忽小亂跳，抓的是「整體骨架尺度不穩定內縮/外放」而不是「尺度太小」。
# 方向是數值越大越可疑。先猜一個起點，之後用真實影片跟
# torso_ratio_jitter_threshold_chart 驗證/調整。
CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD = 0.03

# ============================================================================
# ===== bone_length_oscillation ── 各骨段長度的逐幀振盪（取最大值）=====
# ============================================================================
# bone_length_oscillation：各骨段長度（正規化座標下，chest-hip 距離縮放為
# 1.0 單位）在相鄰幀之間的變化量，取所有納入分析的骨段中最大值——用 MAX
# 而非平均，避免單一骨段的劇烈振盪被其他穩定骨段平均掉（跟過去 max_cv 想
# 抓「單一骨段異常」的道理一樣）。跟 midback_angle_jitter/torso_ratio_jitter
# 是同一類「看變化量、不看絕對值」的指標，方向是數值越大越可疑。先猜一個
# 起點，之後用真實影片跟 bone_length_oscillation_threshold_chart 驗證/調整。
CANDIDATE_BONE_OSCILLATION_THRESHOLD = 0.05

# ============================================================================
# ===== BODY_AXIS_* ── Body Axis Proportion Analysis 用的門檻/參考值 =====
# ============================================================================
# 完整設計動機/計算流程見 compute_body_axis_geometry()／
# compute_body_axis_score_jitter() 的 docstring（在檔案後段）；這裡只集中
# 放常數本身，跟其他 CANDIDATE_*_THRESHOLD 放在同一區，方便一次看到全部
# 會影響異常判斷的門檻。
#
# 正常貓咪身體主軸的參考比例（以 Chest-Hip 距離為 1.0 正規化）。2026-07-09
# 用 shake5772.mp4 兩幀真實畫面校準過一輪：frame 12（坐姿、抬頭看、bbox
# conf 0.96，姿勢乾淨判定為正常）量出來是 [0.671, 0.749, 0.695]，但原本
# 瞎猜的 [0.35, 0.51, 0.51] 讓這個明顯正常的畫面也只拿到 28.4 分——顯示
# 原本的參考值跟真實貓的比例差太多。改用這幀的實測值當參考點，同一支
# 影片 frame 86（蹲低拱背，torso_ratio/midback_offset 也同時被判定異常）
# 量出來是 [0.829, 1.159, 1.549]，跟新參考值的 geometry_error ≈ 0.96，
# 用來反推 BODY_AXIS_ERROR_SIGMA（見下）。注意：只有 2 個樣本點，統計力
# 很弱，之後應該用更多影片重新收斂。
#
# nose_chest 依使用者要求已從 Score 計算中移除（見 compute_body_axis_
# geometry() 註解），這裡保留這個 key 純粹是留紀錄／未來想恢復時方便，
# 目前不會被讀取用於評分。
BODY_AXIS_REFERENCE_RATIOS = {
    "nose_chest": 0.67,      # Nose-Chest / Chest-Hip（保留紀錄，Score 不使用）
    "chest_midback": 0.75,   # Chest-MidBack / Chest-Hip
    "midback_hip": 0.70,     # MidBack-Hip / Chest-Hip
}

# 校準依據：frame 12（正常）geometry_error ≈ 0.02 要落在高分區、frame 86
# （蹲低拱背，判定異常）geometry_error ≈ 0.96 要落在 BODY_AXIS_RELIABLE_
# THRESHOLD 以下——0.7 這個 sigma 讓正常幀仍接近 100 分、異常幀落在
# 25 分附近（明確不可信但不會像原本 sigma=0.35 那樣直接砍到 2~3 分、
# 看不出跟「完全崩壞」的差別）。
BODY_AXIS_ERROR_SIGMA = 0.70         # geometry_error → body_axis_score 指數衰減的尺度常數
BODY_AXIS_RELIABLE_THRESHOLD = 50.0  # body_axis_score 低於此值視為「身體主軸比例不可信」

# compute_body_axis_score_jitter() 用的門檻：量測 body_axis_score 在一個
# 窗口內的變化振幅（max - min），用來區分「持續穩定的低分（例如正面視角
# 造成的系統性失真，振幅約 0~1）」跟「大範圍反覆跳動的低分（骨架真的不
# 穩定/誤判，實測觀察 score 在 10~30 之間來回橫跳，振幅約 20）」——這兩種
# 情況單幀分數可能一樣低，但背後成因不同。門檻是憑使用者實測觀察粗估的
# 起點，不是從大量真實資料統計出來的基準值，之後應該用更多影片重新校準。
BODY_AXIS_MIN_VALID_SAMPLES = 3          # 窗口內至少要有幾幀有效才採信振幅值
BODY_AXIS_SCORE_JITTER_THRESHOLD = 12.0  # body_axis_score 窗口內振幅的候選門檻（方向：越大越可疑）

# ============================================================================
# ===== SQA_ENABLED_THRESHOLDS ── GUI 模式（模式2）面板「是否採用」門檻開關 =====
# ============================================================================
# 上面所有 CANDIDATE_*_THRESHOLD／BODY_AXIS_* 門檻一律都會畫在圖表上或
# 供對應計算函式使用；但 GUI 模式左上角面板（draw_bone_stability_panel/
# draw_body_axis_panel）的綠/紅二元判定、以及 apply_sqa_dual_judgment()
# 的覆蓋決策，只有「這裡列出的指標」才會套用門檻判斷。把某一行整行
# 註解掉（或刪除該 key），該指標就只顯示數值、改用中性色，不再判定
# 異常/正常——不影響其餘指標運行，也不會出錯（sqa_check_reliable()/
# _status_color_for() 都用 .get() 查表，查不到就跳過門檻判斷）。
# value = (門檻值, 方向)："above"=數值越大越可疑，"below"=數值越小越可疑。
#
# 這是全檔案唯一一個「是否參與異常判斷」的登記表，8 項指標（6 項 Bone
# Stability + 2 項 Body Axis）全部集中在這一個字典裡，不會散落在其他地方。
SQA_ENABLED_THRESHOLDS: dict = {
    "torso_ratio": (CANDIDATE_TORSO_RATIO_THRESHOLD, "below"),
    # torso_ratio_inflated 跟上面的 torso_ratio 是同一個原始數值（compute_
    # bone_stability_overlay 直接把 torso_ratio 存成這個別名 key），只是換
    # 一個方向獨立判定，讓兩個方向可以各自單獨開關，不會互相牽動。
    "torso_ratio_inflated": (CANDIDATE_TORSO_RATIO_UPPER_THRESHOLD, "above"),
    "midback_offset_ratio": (CANDIDATE_MIDBACK_OFFSET_THRESHOLD, "above"),
    "midback_angle_jitter": (CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, "above"),
    "torso_ratio_jitter": (CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, "above"),
    "bone_length_oscillation": (CANDIDATE_BONE_OSCILLATION_THRESHOLD, "above"),
    # 以下兩項是 Body Axis Proportion Analysis 的分數（見檔案後段
    # compute_body_axis_geometry()/compute_body_axis_score_jitter()），
    # 呼叫端（analyze_single_video）會把這兩個函式的回傳值併入同一個 ovl
    # dict，這兩項門檻才會真的生效。
    "body_axis_score": (BODY_AXIS_RELIABLE_THRESHOLD, "below"),
    "body_axis_score_jitter": (BODY_AXIS_SCORE_JITTER_THRESHOLD, "above"),
}

# ============================================================================
# ===== ENABLE_SQA_DUAL_JUDGMENT ── GCN 分類為主、幾何判斷為輔的雙重判定總開關 =====
# ============================================================================
# True：GUI 模式（模式2）套用「GCN 分類為主、幾何判斷為輔」雙重判定——只要
# SQA_ENABLED_THRESHOLDS 裡任一啟用的指標超標，右上角顯示的 GCN 分類結果就
# 會被覆蓋成 LOW_CONF（信心值歸零），跟正式推論腳本 1_run_video_inference.py
# 的覆蓋規則一致，可以直接在這支診斷腳本上肉眼確認覆蓋規則對不對。
# False：兩個訊號各自獨立顯示（GCN 分類 vs 幾何面板），互不覆蓋，單純比對
# 兩者是否吻合，適合還在校準門檻、還不確定要不要正式套用覆蓋時使用。
ENABLE_SQA_DUAL_JUDGMENT = True

# bone_length_oscillation 排除的關鍵點（不納入分析）：
#   14,15,16（尾巴）——天生會彎曲，長度變化是真實生理現象，不是偵測雜訊，
#     混進來會污染訊號（跟 test_pose_jitter_analysis.py 的預設一致）。
#   1,2,3（左耳、右耳、鼻子→胸的骨段）——甩頭等行為時頭部轉動快，YOLO
#     對快速移動的頭部關鍵點（鼻子/耳朵）偵測誤差本來就會變大（動作模糊、
#     角速度快），這是「動作快慢」造成的雜訊，不是骨架真的不可信，混進來
#     會讓 bone_length_oscillation 把正常的甩頭動作誤判成異常。
EXCLUDED_KEYPOINTS: set = {1, 2, 3, 14, 15, 16}

# 只在每 N 幀重算一次 overlay 指標（其餘幀沿用上次結果顯示，不重新跑前
# 處理），對齊正式推論的 STGCNConfig.WINDOW_STRIDE 預設值——減少重複運算
# 量，同時讓這裡取樣的頻率更貼近正式部署時這個門檻機制實際會跑的頻率。
OVERLAY_STRIDE = 2

WINDOW_NAME = "Bone Length Stability"
DISPLAY_SIZE = (1080, 720)

# ===== 信心值門檻設定（bbox conf / keypoint conf，集中管理）=====
YOLO_CONF_THRESHOLD = 0.5      # YOLO bbox 偵測信心門檻
DRAW_KP_CONF_THRESHOLD = 0.5  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）

# ╔══════════════════════════════════════════════════════════════════╗
# ║                      以 下 無 需 修 改                           ║
# ╚══════════════════════════════════════════════════════════════════╝

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2), (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]

# 跟 models/stgcn_model.py::compute_bone_feature 內部用的同一份骨架拓撲，
# bone_length_oscillation 逐骨段計算要用。
_PARENTS_17 = np.array([0, 0, 0, 0, 3, 4, 3, 6, 3, 8, 5, 10, 5, 12, 5, 14, 15])
_ACTIVE_BONE_IDS = [i for i in range(1, 17) if i not in EXCLUDED_KEYPOINTS]

WHITE = (255, 255, 255)

# 從影片路徑（含資料夾名，因為你的檔案習慣常把行為標在上層資料夾，
# 例如 ...\walk\1 (43).mp4，檔名本身反而沒有）比對這五類關鍵字，
# 大小寫不敏感，比對不到回傳 "N/A"。
BEHAVIOR_KEYWORDS = ["walk", "lick", "scratch", "shake", "stop"]

# 每支影片依「索引」（VIDEO_PATHS 裡的順序，1-based）給一個獨立顏色，
# 不再依行為分組——同一類行為的影片也會各自用不同顏色，方便逐一辨認曲線。
# 行為標籤仍會顯示在圖例文字跟標題裡，只是不再拿來決定顏色。
VIDEO_INDEX_COLORS = [
    "#3498db",  # 1 藍
    "#f39c12",  # 2 金黃色（原本的橘色 #e67e22 跟 4 號紅色色相太接近，改用色相差更大的金黃）
    "#2ecc71",  # 3 綠
    "#c0392b",  # 4 磚紅色（原本的紅色 #e74c3c 偏橘調，改成更深、更偏紫的紅，跟 2 號拉開差距）
    "#9b59b6",  # 5 紫
    "#1abc9c",  # 6 青綠色（跟 3 號綠拉開色相差距）
    "#e67e22",  # 7 橘（跟 2 號金黃不相鄰，仍可辨識）
    "#34495e",  # 8 深藍灰
    "#e91e63",  # 9 桃紅
    "#7f8c8d",  # 10 灰
]


def color_for_index(index: int) -> str:
    return VIDEO_INDEX_COLORS[(index - 1) % len(VIDEO_INDEX_COLORS)]


def show_fig_fit_to_content(fig):
    """在部分 Windows/Tk 環境下，plt.show() 開出來的視窗預設尺寸跟圖表本身
    無關（可能沿用畫面預設大小或被自動放大/最大化），導致視窗四周留下大片
    空白、圖表本體反而被擠在一角。這裡在顯示前，明確把 Tk 視窗尺寸重設成
    跟 figure 實際畫布尺寸（依 dpi 換算成像素）一致，視窗會貼齊圖表內容，
    不留多餘空白。任何一步失敗（例如非 TkAgg 後端）都直接忽略，退回預設
    行為，不影響圖表照常顯示。"""
    try:
        w_px, h_px = fig.canvas.get_width_height()
        mng = plt.get_current_fig_manager()
        mng.window.state("normal")
        mng.window.wm_geometry(f"{w_px}x{h_px}+50+50")
    except Exception:
        pass
    plt.show()


def ascii_safe_label(text: str, fallback: str) -> str:
    """matplotlib 預設字型（DejaVu Sans）沒有中文字符集，圖表裡混進中文
    （例如影片檔名剛好是中文，如「7月2日 (4).mp4」）會整段顯示成方塊。
    這裡把非 ASCII 字元換成空格（不是直接刪除）再收斂多餘空白，避免像
    「7月2日」這種中間插著中文字的字串被砍字砍成看起來像另一個數字的
    「72」——保留空格能讓「7」「2」維持是分開的兩截，不會誤黏在一起。
    濾完是空字串就用 fallback（例如 video_4）。"""
    cleaned = "".join(ch if ord(ch) < 128 else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else fallback


def infer_behavior_label(video_path) -> str:
    """從完整路徑（資料夾名或檔名任一處）比對 walk/lick/scratch/shake/stop，
    大小寫不敏感，找不到回傳 "N/A"。五個關鍵字彼此不互為子字串，不會誤配。"""
    lowered = str(video_path).lower()
    for kw in BEHAVIOR_KEYWORDS:
        if kw in lowered:
            return kw
    return "N/A"


def compute_bone_stability_overlay(seq_window, conf_window, bbox_window=None):
    """seq_window: (T, 17, 2) 原始座標；conf_window: (T, 17)。
    回傳這個窗口內的五個骨架可信度指標（Skeleton Quality Assessment）：
      Spatial Consistency：torso_ratio、midback_offset_ratio
      Temporal Consistency：midback_angle_jitter、torso_ratio_jitter、
                            bone_length_oscillation
    各自的計算方式見下方對應區塊的註解。

    前處理管線與 skeleton_visualizer.py::compute_velocity_overlay 一致，
    確保這裡看到的數值跟真正的推論路徑同源。

    bbox_window: (T, 4) 原始像素座標的 (x1,y1,x2,y2)，缺值用 NaN 填——用來算
    torso_ratio／torso_ratio_jitter（胸髖距離 ÷ bbox 對角線，及其逐幀變化
    量）。這是唯一不靠關鍵點自己互相參照的兩個指標：其餘三個
    （midback_offset_ratio／midback_angle_jitter／bone_length_oscillation）
    都只用關鍵點彼此的相對關係，如果軀幹（胸/mid_back/髖）整組一起塌陷，
    這些指標會失效（分子分母同步縮小，比例不變）；bbox 是 YOLO 另一個獨立
    輸出頭（框選回歸），相對不受關鍵點塌陷影響，可以當外部參照，抓「軀幹
    持續性地錯但很穩定」這種其他指標的共同盲點。
    """
    seq = interpolate_missing(seq_window, conf_window, threshold=0.1)
    T = seq_window.shape[0]

    chest_hip_valid = (conf_window[:, 3] >= BONE_CONF_THRESHOLD) & (conf_window[:, 5] >= BONE_CONF_THRESHOLD)

    torso_ratio = float("nan")
    # torso_ratio_jitter（Skeleton Scale Jitter）：torso_ratio 逐幀值在相鄰
    # 幀之間的變化量，抓的是「整體骨架尺度忽大忽小」，跟 torso_ratio 本身
    # 的絕對值高低（軀幹相對 bbox 是否塌陷）是兩件事、互補判斷。
    torso_ratio_jitter = float("nan")
    if bbox_window is not None:
        bbox_valid = ~np.isnan(bbox_window).any(axis=1)
        combined_valid = chest_hip_valid & bbox_valid
        if np.any(combined_valid):
            bw_all = bbox_window[:, 2] - bbox_window[:, 0]
            bh_all = bbox_window[:, 3] - bbox_window[:, 1]
            bbox_diag_all = np.sqrt(bw_all ** 2 + bh_all ** 2)
            body_size_all = np.linalg.norm(seq[:, 3, :2] - seq[:, 5, :2], axis=1)
            diag_ok_all = combined_valid & (bbox_diag_all > 1e-6)

            torso_ratio_per_frame = np.full(T, np.nan, dtype=np.float64)
            torso_ratio_per_frame[diag_ok_all] = body_size_all[diag_ok_all] / bbox_diag_all[diag_ok_all]

            valid_ratios = torso_ratio_per_frame[diag_ok_all]
            if valid_ratios.size > 0:
                torso_ratio = float(np.mean(valid_ratios))

            pair_ok = diag_ok_all[1:] & diag_ok_all[:-1]
            if int(np.sum(pair_ok)) >= MIN_VALID_FRAME_PAIRS_TORSO_RATIO_JITTER:
                ratio_diffs = np.abs(torso_ratio_per_frame[1:][pair_ok] - torso_ratio_per_frame[:-1][pair_ok])
                torso_ratio_jitter = float(np.mean(ratio_diffs))

    # 虛擬點：胸(3)與髖(5)的中點。注意：這裡的標註慣例是把 mid_back 點在
    # 貓背部最頂端的毛（拱背最高點），所以 Chest-MidBack-Hip 三點正常情況
    # 下就是一個「三角形」，不是一直線——mid_back 偏離這個虛擬中點的距離
    # 恆為正、且大小跟貓拱背弧度/姿勢有關，並非 0 才代表正確。目前沒有
    # 已知的「正常基準值」可用（沒做過標註分布統計），所以這個比例只當
    # 參考數值顯示，不套用越靠近 0 越好的門檻判斷（面板顯示走中性色，
    # 不用 heat color）。真正可能有鑑別力的是「這個比例在時間窗口內夠不
    # 夠穩定」（同一姿勢下應該穩定），但目前沒有實作對應的抖動指標。
    midback_valid = chest_hip_valid & (conf_window[:, 4] >= BONE_CONF_THRESHOLD)
    midback_offset_ratio = float("nan")
    if np.any(midback_valid):
        virtual_pt = (seq[:, 3, :2] + seq[:, 5, :2]) / 2.0
        raw_offset = np.linalg.norm(seq[:, 4, :2] - virtual_pt, axis=1)
        body_size_per_frame = np.linalg.norm(seq[:, 3, :2] - seq[:, 5, :2], axis=1)
        frame_ok = midback_valid & (body_size_per_frame > 1e-6)
        if int(np.sum(frame_ok)) >= MIN_VALID_FRAMES_MIDBACK_OFFSET:
            ratio_vals = raw_offset[frame_ok] / body_size_per_frame[frame_ok]
            midback_offset_ratio = float(np.mean(ratio_vals))

    # Chest-MidBack-Hip 夾角的「連續幀抖動程度」（單位：度/幀），不是窗口
    # 平均角度本身。原本想直接對窗口平均角度設絕對門檻，但 MidBack 標在
    # 拱背最高點、跟 Chest/Hip 天生形成三角形，角度會隨貓的拱背弧度/姿勢
    # 大幅變化，絕對值沒有放諸四海皆準的「正常範圍」，用絕對門檻很容易
    # 把姿勢不同的正常貓誤判成異常。改成量「這一幀跟前一幀比，角度變了
    # 幾度」，只在相鄰兩幀都有效時才計入——不管姿勢本身的絕對角度是多少，
    # 只要偵測穩定，這個角度在短短 16 幀窗口內應該平滑變化；跳動大代表
    # 偵測不穩定，跟絕對角度無關。跟 compute_midback_angle 用同一套幾何
    # 公式，這裡向量化算整個窗口。有效幀對數不足
    # MIN_VALID_FRAME_PAIRS_MIDBACK_ANGLE_JITTER 時標記為 NaN，避免只憑
    # 一兩組差值就代表整個窗口，樣本太少雜訊沒被平均掉。
    midback_angle_jitter = float("nan")
    if np.any(midback_valid):
        chest_pt = seq[:, 3, :2]
        midback_pt = seq[:, 4, :2]
        hip_pt = seq[:, 5, :2]
        v1 = chest_pt - midback_pt
        v2 = hip_pt - midback_pt
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        angle_ok = midback_valid & (n1 > 1e-6) & (n2 > 1e-6)
        angle_deg = np.full(T, np.nan, dtype=np.float64)
        if np.any(angle_ok):
            cos_angle = np.clip(
                np.sum(v1[angle_ok] * v2[angle_ok], axis=1) / (n1[angle_ok] * n2[angle_ok]), -1.0, 1.0
            )
            angle_deg[angle_ok] = np.degrees(np.arccos(cos_angle))
        pair_ok = angle_ok[1:] & angle_ok[:-1]
        if int(np.sum(pair_ok)) >= MIN_VALID_FRAME_PAIRS_MIDBACK_ANGLE_JITTER:
            angle_diffs = np.abs(angle_deg[1:][pair_ok] - angle_deg[:-1][pair_ok])
            midback_angle_jitter = float(np.mean(angle_diffs))

    # bone_length_oscillation：各骨段長度（正規化座標下，chest-hip 距離縮放
    # 為 1.0 單位——用同一套 flip/orientation/scale 正規化管線，確保這裡看
    # 到的尺度跟真正的推論路徑同源）在相鄰幀之間的變化量。這是專門抓「反覆
    # 縮放/跳動」的指標：一個骨段如果在 40/28/41/26/39/27 之間跳動，平均長度
    # 可能跟穩定在 40/39/40/39/40 差不多，但逐幀變化量差非常多——跟骨段長度
    # 本身的 CV（離散度）是不同統計量。取所有納入分析骨段裡的最大值，避免
    # 單一骨段的劇烈振盪被其他穩定骨段平均掉。
    bone_length_oscillation = float("nan")
    norm_seq = flip_normalize(seq)
    norm_seq = orientation_normalize(norm_seq)
    norm_seq = normalize_skeleton_coords(norm_seq)
    bone_xy = compute_bone_feature(norm_seq)         # (T, 17, 2)
    bone_len = np.linalg.norm(bone_xy, axis=-1)      # (T, 17)

    seg_oscillation = np.full(17, np.nan, dtype=np.float64)
    for j in _ACTIVE_BONE_IDS:
        parent = int(_PARENTS_17[j])
        bone_valid = (conf_window[:, j] >= BONE_CONF_THRESHOLD) & (conf_window[:, parent] >= BONE_CONF_THRESHOLD)
        bone_pair_ok = bone_valid[1:] & bone_valid[:-1]
        if int(np.sum(bone_pair_ok)) < MIN_VALID_FRAME_PAIRS_BONE_OSCILLATION:
            continue
        length_diffs = np.abs(bone_len[1:, j][bone_pair_ok] - bone_len[:-1, j][bone_pair_ok])
        seg_oscillation[j] = float(np.mean(length_diffs))

    valid_oscillation = seg_oscillation[~np.isnan(seg_oscillation)]
    if valid_oscillation.size > 0:
        bone_length_oscillation = float(np.max(valid_oscillation))

    return {
        "midback_offset_ratio": midback_offset_ratio,
        "torso_ratio": torso_ratio,
        # torso_ratio 的別名，給 SQA_ENABLED_THRESHOLDS 的「上界」判定用（同一
        # 個數值，兩個方向各自獨立開關），不是另外算出來的新指標。
        "torso_ratio_inflated": torso_ratio,
        "midback_angle_jitter": midback_angle_jitter,
        "torso_ratio_jitter": torso_ratio_jitter,
        "bone_length_oscillation": bone_length_oscillation,
    }


# ============================================================================
# ===== Body Axis Proportion Analysis（貓咪身體主軸幾何比例分析）——
#       獨立的 SQA 新模組 =====
# ============================================================================
# 跟上面 compute_bone_stability_overlay()（Spatial/Temporal Consistency）
# 完全獨立、互不依賴：
#   - Spatial/Temporal Consistency 看的是「這個窗口的偵測穩不穩定/可信度」，
#     需要一整個時間窗口（T 幀）才能算。
#   - 這個模組看的是「這一幀骨架的身體主軸比例，符不符合真實貓咪的幾何
#     分布」，只需要單一幀的 17 個關鍵點座標。
#
# 動機：目的是過濾 YOLO Pose 因低信心或背景誤檢（棉被、枕頭、家具紋理）
# 產生的不合理骨架——即使骨架看起來完整、時間上也很穩定，只要身體主軸
# 比例離譜，就能判斷這不是一隻真的貓。
#
# 身體主軸原本有三段，但 Score 只用其中兩段：
#   1. Nose  → Chest    （只算出來給面板參考顯示，不計入 Score）
#   2. Chest → MidBack  （計入 Score）
#   3. MidBack → Hip    （計入 Score）
# 2026-07-09 依使用者要求，把 Nose-Chest 從 Score 計算中移除，只保留
# Chest-MidBack/MidBack-Hip 這兩段參與評分；nose_chest_ratio 仍會算出來
# 並保留在回傳結果與面板顯示裡，純粹當參考數值，不影響 geometry_error/
# body_axis_score。
# 刻意不分析四肢：四肢容易受遮擋/舔舐/行走姿態影響，身體主軸在大部分
# 姿勢下相對穩定，比較適合當幾何品質判斷依據。
#
# 使用約定（呼叫端要遵守）：
#   - 只傳入單一幀的 17 個關鍵點座標（kpts: (17, 2)）。
#   - 不使用信心值、不使用時間序列、不使用 ST-GCN 輸出——如果需要先用
#     信心值篩選/插補座標，請在呼叫前處理好，這裡只管幾何比例本身。
#   - 目前完全獨立運作，之後要整合進 Skeleton Quality Assessment 當作
#     第三類評估指標（Geometric Plausibility），只要把這個函式的回傳值
#     併入同一個 ovl dict 即可，不需要更動這裡的程式碼。
#
# 已知限制：Chest→Hip 這條正規化尺度基準，在貓正面朝鏡頭時會因透視縮短
# 而失真（跟本檔案 torso_ratio 面對的問題一樣），這裡目前沒有處理視角
# 問題，是身體主軸比例分析的最小可行版本，之後校準門檻時建議連同視角
# 一起考慮。
#
# 所有門檻/參考值常數（BODY_AXIS_REFERENCE_RATIOS、BODY_AXIS_ERROR_SIGMA、
# BODY_AXIS_RELIABLE_THRESHOLD、BODY_AXIS_MIN_VALID_SAMPLES、
# BODY_AXIS_SCORE_JITTER_THRESHOLD）統一集中管理在檔案前段跟其他
# CANDIDATE_*_THRESHOLD 同一區（SQA_ENABLED_THRESHOLDS 定義之前），這裡
# 不重複定義——這樣全部會影響異常判斷的門檻只需要在一個地方找。
# ============================================================================


def compute_body_axis_geometry(kpts, nose_joint=0, chest_joint=3, midback_joint=4, hip_joint=5):
    """單幀「身體主軸幾何比例分析」（Body Axis Proportion Analysis）。跟本
    檔案其餘 SQA 指標（compute_bone_stability_overlay 的 Spatial/Temporal
    Consistency）完全獨立：那邊看「這個窗口的偵測穩不穩定」，這裡看「這
    一幀骨架的身體主軸比例，符不符合真實貓咪的幾何分布」——目的是過濾
    YOLO Pose 因低信心或背景誤檢（棉被、枕頭、家具紋理）產生的不合理骨架，
    即使骨架看似完整、時間上也穩定，也能判斷是否符合貓咪的身體比例。

    只吃單一幀的 17 個關鍵點座標（kpts: (17, 2)），不使用信心值、不使用
    時間序列、不使用 ST-GCN 輸出——呼叫端如果需要先用信心值篩選/插補
    座標，請在呼叫前處理好，這裡只管幾何比例本身。

    只分析身體主軸三段（Nose→Chest、Chest→MidBack、MidBack→Hip），刻意
    排除四肢——四肢容易受遮擋/舔舐/行走姿態影響，身體主軸在大部分姿勢下
    相對穩定，比較適合當幾何品質判斷依據。2026-07-09 依使用者要求，
    Nose→Chest 這段只算出來給面板參考顯示，不計入 Score（見下方計算
    流程第 3 步）。

    計算流程：
      1. 算 Nose-Chest、Chest-MidBack、MidBack-Hip 三段的歐氏距離。
      2. 用 Chest-Hip 距離正規化（而非原始像素長度），避免不同攝影距離
         造成的尺度差異：
           R1 = Nose-Chest    / Chest-Hip   （僅供參考，不計入 Score）
           R2 = Chest-MidBack / Chest-Hip
           R3 = MidBack-Hip   / Chest-Hip
      3. Geometry Vector = [R2, R3]（不含 R1），跟 BODY_AXIS_REFERENCE_RATIOS
         代表的正常貓咪參考比例算歐氏距離，得到 geometry_error——偏差
         越小代表越符合正常貓咪的身體比例，偏差越大代表可能發生幾何
         崩壞或誤檢。
      4. geometry_error 用指數衰減轉成 body_axis_score（0~100）。

    Returns: dict，包含 nose_chest_ratio（僅供參考，不影響評分）/
    chest_midback_ratio / midback_hip_ratio / geometry_vector（[R2, R3]，
    只含實際計入評分的兩項）/ geometry_error / body_axis_score（0~100，
    資料不足時為 NaN）/ body_axis_reliable（body_axis_score 是否
    >= BODY_AXIS_RELIABLE_THRESHOLD；證據不足導致無法計算時，保守回傳
    True，不否決）。
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    chest_hip_dist = float(np.linalg.norm(kpts[chest_joint] - kpts[hip_joint]))

    if chest_hip_dist < 1e-6:
        return {
            "nose_chest_ratio": float("nan"),
            "chest_midback_ratio": float("nan"),
            "midback_hip_ratio": float("nan"),
            "geometry_vector": [float("nan"), float("nan")],
            "geometry_error": float("nan"),
            "body_axis_score": float("nan"),
            "body_axis_reliable": True,
        }

    nose_chest = float(np.linalg.norm(kpts[nose_joint] - kpts[chest_joint]))
    chest_midback = float(np.linalg.norm(kpts[chest_joint] - kpts[midback_joint]))
    midback_hip = float(np.linalg.norm(kpts[midback_joint] - kpts[hip_joint]))

    r1 = nose_chest / chest_hip_dist       # 僅供參考顯示，不計入 Score
    r2 = chest_midback / chest_hip_dist
    r3 = midback_hip / chest_hip_dist
    geometry_vector = [r2, r3]             # 只含實際計入評分的兩項

    diffs = np.array([
        r2 - BODY_AXIS_REFERENCE_RATIOS["chest_midback"],
        r3 - BODY_AXIS_REFERENCE_RATIOS["midback_hip"],
    ])
    geometry_error = float(np.linalg.norm(diffs))
    body_axis_score = float(100.0 * np.exp(-geometry_error / BODY_AXIS_ERROR_SIGMA))
    body_axis_reliable = bool(body_axis_score >= BODY_AXIS_RELIABLE_THRESHOLD)

    return {
        "nose_chest_ratio": r1,
        "chest_midback_ratio": r2,
        "midback_hip_ratio": r3,
        "geometry_vector": geometry_vector,
        "geometry_error": geometry_error,
        "body_axis_score": body_axis_score,
        "body_axis_reliable": body_axis_reliable,
    }


# ============================================================================
# ===== body_axis_score_jitter ── Body Axis Score 的窗口內振幅（時間維度搭檔指標）=====
# ============================================================================
# 跟 torso_ratio 有 torso_ratio_jitter、midback_offset_ratio 有
# midback_angle_jitter 是同一套設計邏輯。門檻常數（BODY_AXIS_MIN_VALID_
# SAMPLES/BODY_AXIS_SCORE_JITTER_THRESHOLD）統一管理在 compute_body_axis_
# geometry() 上方那個 BODY_AXIS_* 設定區塊，這裡不重複定義。
def compute_body_axis_score_jitter(kpts_window):
    """量測 body_axis_score 在一個時間窗口內的變化振幅——跟
    compute_body_axis_geometry() 的單幀版本分開，這裡需要一整個窗口
    （T 幀）的關鍵點座標才能算，屬於 Temporal Consistency 類的搭檔指標。

    kpts_window: (T, 17, 2)，每一幀的關鍵點座標，跟 compute_body_axis_geometry()
    要求一致：不使用信心值、不使用 ST-GCN 輸出。

    Returns: (amplitude, valid_sample_count)。amplitude = 窗口內所有有效
    body_axis_score 的 max - min（振幅，不是逐幀差值的平均），能區分「持續
    穩定的低分」跟「大範圍反覆跳動的低分」；有效幀數低於
    BODY_AXIS_MIN_VALID_SAMPLES 時，amplitude 回傳 NaN（樣本太少不採信，
    不強行給一個雜訊值）。
    """
    kpts_window = np.asarray(kpts_window, dtype=np.float64)
    t_len = kpts_window.shape[0]
    scores = np.array([
        compute_body_axis_geometry(kpts_window[t])["body_axis_score"]
        for t in range(t_len)
    ])
    valid_scores = scores[np.isfinite(scores)]
    valid_sample_count = int(valid_scores.size)
    if valid_sample_count < BODY_AXIS_MIN_VALID_SAMPLES:
        return float("nan"), valid_sample_count
    amplitude = float(np.max(valid_scores) - np.min(valid_scores))
    return amplitude, valid_sample_count


# body_axis_score/body_axis_score_jitter 兩項門檻已經直接寫進檔案前段的
# SQA_ENABLED_THRESHOLDS 字典 literal 裡（不再用 .update() 補），這裡不
# 重複註冊。呼叫端（analyze_single_video）記得把 compute_body_axis_
# geometry()/compute_body_axis_score_jitter() 的結果併入同一個 ovl dict，
# 這兩項門檻才會真的生效。


def classify_bone_window(kpts_arr, conf_arr, classifier):
    """對 compute_bone_stability_overlay 用的同一個滑動窗口跑一次 ST-GCN
    分類，回傳 (behavior_id, confidence, probs)，方便在畫面上跟骨架穩定度
    面板並排對照：「GCN 這一刻覺得是什麼」跟「幾何指標覺得這幀骨架可不可信」
    是兩個獨立訊號，這裡只負責取得前者、不做任何覆蓋/合併判斷。

    kpts_arr/conf_arr: (T, 17, 2) / (T, 17)，跟傳給 compute_bone_stability_overlay
    的 seq_window/conf_window 是同一份、未正規化的原始像素座標——
    BehaviorClassifier.classify() 內部會自己完成 flip/orientation/scale
    正規化與特徵組裝，呼叫端不用重複處理。

    classifier 為 None（模型尚未成功載入）時直接回傳 LOW_CONF_ID，不丟例外。
    """
    if classifier is None:
        return LOW_CONF_ID, 0.0, None
    seq_xy = interpolate_missing(kpts_arr, conf_arr, threshold=0.0)
    behavior_id, confidence, probs = classifier.classify(seq_xy, conf_arr)
    if behavior_id is None:
        return LOW_CONF_ID, 0.0, probs
    return int(behavior_id), float(confidence), probs


def sqa_check_reliable(ovl: dict) -> tuple:
    """依 SQA_ENABLED_THRESHOLDS 檢查這個窗口的五項指標，回傳
    (reliable, failed_checks)。只有存在於 SQA_ENABLED_THRESHOLDS 裡的指標
    才會參與判定——跟面板顏色用的是同一份設定，把某一項整行註解掉，就等於
    它完全不參與「是否不可信」的判定。數值是 NaN（該指標資料不足）時不
    計入不可信判斷，跟 1_run_video_inference.py 的 compute_skeleton_quality
    邏輯一致。"""
    failed = []
    for key, (threshold, direction) in SQA_ENABLED_THRESHOLDS.items():
        value = ovl.get(key, float("nan"))
        if not np.isfinite(value):
            continue
        is_bad = (value > threshold) if direction == "above" else (value < threshold)
        if is_bad:
            failed.append(key)
    return len(failed) == 0, failed


def apply_sqa_dual_judgment(cls, ovl):
    """「GCN 分類為主、幾何判斷為輔」雙重判定：ENABLE_SQA_DUAL_JUDGMENT 為
    True 時，只要這個窗口被 sqa_check_reliable 判定不可信，就把 GCN 分類
    結果 cls 覆蓋成 (LOW_CONF_ID, 0.0, 原本的 probs)，跟正式推論腳本
    1_run_video_inference.py 的覆蓋規則一致。為 False 時原樣回傳 cls，
    兩個訊號各自獨立顯示，不覆蓋——校準門檻階段用這個模式肉眼比對。
    cls 或 ovl 為 None（該幀還沒有有效資料）時原樣回傳，不處理。"""
    if not ENABLE_SQA_DUAL_JUDGMENT or cls is None or ovl is None:
        return cls
    reliable, _failed = sqa_check_reliable(ovl)
    if reliable:
        return cls
    _behavior_id, _confidence, probs = cls
    return LOW_CONF_ID, 0.0, probs


def draw_behavior_label(frame, behavior_id, confidence):
    """在畫面右上角顯示 ST-GCN 分類結果（跟左上角的骨架穩定度面板分開放，
    左右對稱排版），方便同時看「GCN 分類」跟「幾何判斷」兩個訊號有沒有
    互相印證。信心低於 BEHAVIOR_MIN_CONFIDENCE 或 behavior_id 為
    LOW_CONF_ID 時顯示 LOW_CONF（灰色）。ENABLE_SQA_DUAL_JUDGMENT 開啟時，
    呼叫端傳進來的 behavior_id/confidence 已經是套用 apply_sqa_dual_judgment
    覆蓋後的最終結果，跟正式推論腳本 1_run_video_inference.py 的顯示規則
    一致。"""
    h, w = frame.shape[:2]
    ui_scale = max(0.9, min(1.6, np.hypot(w, h) / 1500.0))
    is_low = (behavior_id == LOW_CONF_ID) or (confidence < BEHAVIOR_MIN_CONFIDENCE)
    label = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
    color = (150, 150, 150) if is_low else BEHAVIOR_COLORS.get(behavior_id, (255, 255, 255))
    text = f"GCN: {label.upper()}" if is_low else f"GCN: {label.upper()} {confidence * 100:.1f}%"
    font_scale = 0.65 * ui_scale
    (text_w, _text_h), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    margin = int(14 * ui_scale)
    tx = w - text_w - margin
    ty = margin + int(26 * ui_scale)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

    # 目前雙重判定開關狀態（[j] 鍵即時切換，j = Judgment），跟 GCN 標籤放在
    # 同一欄右對齊、疊在下一行，讓使用者不用看終端機就知道現在覆蓋規則有
    # 沒有生效。
    dual_text = f"[DUAL: {'ON' if ENABLE_SQA_DUAL_JUDGMENT else 'OFF'}]  press [j] to toggle"
    dual_fs = 0.45 * ui_scale
    (dual_w, _dual_h), _ = cv2.getTextSize(dual_text, cv2.FONT_HERSHEY_SIMPLEX, dual_fs, 1)
    dual_color = (100, 220, 255) if ENABLE_SQA_DUAL_JUDGMENT else (150, 150, 150)
    dual_tx = w - dual_w - margin
    dual_ty = ty + int(20 * ui_scale)
    cv2.putText(frame, dual_text, (dual_tx, dual_ty), cv2.FONT_HERSHEY_SIMPLEX, dual_fs, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, dual_text, (dual_tx, dual_ty), cv2.FONT_HERSHEY_SIMPLEX, dual_fs, dual_color, 1, cv2.LINE_AA)
    return frame


# 目前五個都有候選門檻、可以判定「正常/異常」的指標，分兩個層級（詳見模組
# 開頭「目的」段落）：
#   Spatial Consistency（看某一幀本身合不合理）：
#     torso_ratio（< CANDIDATE_TORSO_RATIO_THRESHOLD 視為異常）、
#     midback_offset_ratio（> 門檻視為異常）。
#   Temporal Consistency（看連續幀之間變化合不合理）：
#     midback_angle_jitter、torso_ratio_jitter、bone_length_oscillation
#     （> 各自門檻視為異常）。
# GUI 面板上這五項都用綠/紅二元顯示，不是連續熱力色階：正常＝綠色、
# 異常＝紅色，一眼就能看出這個窗口有沒有被判定為不可信。
STATUS_GOOD_COLOR = (0, 210, 0)   # 綠：沒超過門檻，正常
STATUS_BAD_COLOR = (0, 0, 230)    # 紅：超過門檻，判定異常


def _status_color(value, threshold, direction="above"):
    """依門檻判斷 value 正常還是異常，回傳綠色或紅色（二元，不是漸層）。
    direction="above"：value > threshold 才算異常（jitter/oscillation 系列
    指標用）。direction="below"：value < threshold 才算異常（torso_ratio
    這類「越小越可疑」的指標用）。value 是 NaN（該幀沒有有效資料）時回傳
    灰色。"""
    if not np.isfinite(value):
        return (150, 150, 150)
    is_bad = (value > threshold) if direction == "above" else (value < threshold)
    return STATUS_BAD_COLOR if is_bad else STATUS_GOOD_COLOR


def _status_color_for(metric_key, value):
    """依 SQA_ENABLED_THRESHOLDS 查表決定面板顏色：查得到就照該指標的
    (門檻, 方向) 用 _status_color 判斀綠/紅；查不到（該項被註解掉/移除）
    就一律回傳中性色，只顯示數值、不做異常判定。"""
    entry = SQA_ENABLED_THRESHOLDS.get(metric_key)
    if entry is None:
        return NEUTRAL_TEXT_COLOR if np.isfinite(value) else (150, 150, 150)
    threshold, direction = entry
    return _status_color(value, threshold, direction)


def _status_color_for_multi(metric_keys, value):
    """跟 _status_color_for 同一套查表邏輯，但用在「同一個數值同時掛了
    好幾個獨立門檻」的欄位（例如 torso_ratio 本身、torso_ratio_inflated
    兩個 key 其實是同一個 torso_ratio 數值，只是方向相反）。metric_keys
    裡任一個有啟用的門檻判定為異常，就回傳紅色；有啟用但全部正常回傳
    綠色；metric_keys 全部都被註解掉/移除時才退回中性色。之後再加其他
    「一個數值、多個方向」的指標，直接重用這個函式即可，不用重複寫。"""
    if not np.isfinite(value):
        return (150, 150, 150)
    any_enabled = False
    for key in metric_keys:
        entry = SQA_ENABLED_THRESHOLDS.get(key)
        if entry is None:
            continue
        any_enabled = True
        threshold, direction = entry
        is_bad = (value > threshold) if direction == "above" else (value < threshold)
        if is_bad:
            return STATUS_BAD_COLOR
    return STATUS_GOOD_COLOR if any_enabled else NEUTRAL_TEXT_COLOR


FONT_SCALE_MULT = 1.5  # 面板整體字體放大倍率
NEUTRAL_TEXT_COLOR = (0, 200, 255)  # 中等亮度、高飽和的琥珀色——比灰白更顯眼，但不刺眼
# 給「沒有已知好壞門檻、純參考數值」的欄位用的固定中性綠色——避免暗示
# 某個方向比較好/壞。目前只剩疊在關鍵點上的即時 MidBack 夾角
# （draw_midback_angle，單幀幾何量、不是窗口統計指標）用這個顏色——
# midback_offset_ratio 已經改用 _status_color 二元判斷，不再套用。
REFERENCE_VALUE_COLOR = (100, 210, 100)


def draw_bone_stability_panel(frame, ovl):
    if ovl is None:
        return frame
    h, w = frame.shape[:2]
    ui_scale = max(0.9, min(1.6, np.hypot(w, h) / 1500.0))
    panel_w = int(430 * ui_scale)
    panel_h = int(210 * ui_scale)
    # 貼齊畫面左上角邊界，不留額外偏移（原本 y0 多加了 44px 的空白間距，
    # 使用者要求整個面板要貼齊邊界，這裡拿掉那段偏移）。
    pad = int(6 * ui_scale)
    x0 = pad
    y0 = pad
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (120, 120, 120), 1, cv2.LINE_AA)

    tx = x0 + int(8 * ui_scale)
    ty = y0 + int(22 * ui_scale)
    f = FONT_SCALE_MULT
    cv2.putText(frame, "Bone Stability (window)", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f * ui_scale, NEUTRAL_TEXT_COLOR, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # mid_back 偏離「胸髖虛擬中點」的比例。MidBack 標註在背部拱起最高點，
    # 跟胸/髖正常就會形成三角形（不共線），這個比例恆為正、大小跟拱背
    # 弧度/姿勢有關——沒有標註分布統計出來的精確基準值，門檻是粗略的解剖
    # 合理性上限（見 CANDIDATE_MIDBACK_OFFSET_THRESHOLD 註解），數值越大
    # 越可疑，一樣用 _status_color 二元綠/紅顯示。
    mb_ratio = ovl.get("midback_offset_ratio", float("nan"))
    mb_ratio_color = _status_color_for("midback_offset_ratio", mb_ratio)
    cv2.putText(frame, f"midback offset: {mb_ratio:.4f}" if np.isfinite(mb_ratio) else "midback offset: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, mb_ratio_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # 胸髖距離 ÷ bbox 對角線——唯一不靠關鍵點自己互相參照的指標，用 YOLO
    # 獨立的框選輸出當外部參照，抓「軀幹整組一起塌陷」這種其他指標的共同
    # 盲點。同時檢查兩個方向：太小＝軀幹相對 bbox 塌陷，太大＝chest/hip
    # 其中一點整個消失、fallback 到畫面角落座標導致距離被撐大（尺度正規化
    # 失效的另一種樣態）；兩個方向任一異常這一格就顯示紅色。
    torso_ratio = ovl.get("torso_ratio", float("nan"))
    torso_color = _status_color_for_multi(("torso_ratio", "torso_ratio_inflated"), torso_ratio)
    cv2.putText(frame, f"torso ratio: {torso_ratio:.4f}" if np.isfinite(torso_ratio) else "torso ratio: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, torso_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # Chest-MidBack-Hip 夾角的連續幀抖動程度（度/幀）——不看絕對角度大小
    # （那個沒有放諸四海皆準的正常範圍，容易因為姿勢不同誤判），只看這一
    # 幀跟前一幀比角度變了幾度。方向是數值越大越可疑。
    mb_angle_jitter = ovl.get("midback_angle_jitter", float("nan"))
    mb_angle_jitter_color = _status_color_for("midback_angle_jitter", mb_angle_jitter)
    cv2.putText(frame, f"midback angle jitter: {mb_angle_jitter:.1f}deg/f" if np.isfinite(mb_angle_jitter) else "midback angle jitter: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, mb_angle_jitter_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # Skeleton Scale Jitter：torso_ratio 逐幀值在相鄰幀之間的變化量——抓
    # 「整體骨架尺度忽大忽小」，跟 torso_ratio 本身的絕對值高低是兩件事。
    # 方向是數值越大越可疑。
    torso_jitter = ovl.get("torso_ratio_jitter", float("nan"))
    torso_jitter_color = _status_color_for("torso_ratio_jitter", torso_jitter)
    cv2.putText(frame, f"scale jitter: {torso_jitter:.4f}/f" if np.isfinite(torso_jitter) else "scale jitter: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, torso_jitter_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)
    # Bone Length Oscillation：所有納入分析骨段裡，相鄰幀長度變化量的最大值
    # （正規化座標下）。抓的是「反覆縮放/跳動」而非平均長度，跟骨段長度的
    # 離散度（CV）是不同統計量。方向是數值越大越可疑。
    bone_osc = ovl.get("bone_length_oscillation", float("nan"))
    bone_osc_color = _status_color_for("bone_length_oscillation", bone_osc)
    cv2.putText(frame, f"bone oscillation: {bone_osc:.4f}/f" if np.isfinite(bone_osc) else "bone oscillation: --",
                (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, bone_osc_color, 2, cv2.LINE_AA)
    ty += int(34 * ui_scale)

    # 五個指標只要有一個因為窗口內有效幀（對）數不足門檻而變成 NaN，就在
    # 畫面右上角顯眼標示，跟左上角的面板數值分開，避免使用者誤把 "--" 當成
    # 正常值看過去。
    if not np.isfinite(mb_ratio) or not np.isfinite(mb_angle_jitter) or not np.isfinite(torso_jitter) or not np.isfinite(bone_osc):
        not_eval_text = "NOT EVALUABLE"
        text_scale = 0.7 * f * ui_scale
        (text_w, text_h), _ = cv2.getTextSize(not_eval_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2)
        ex = w - text_w - int(14 * ui_scale)
        ey = int(14 * ui_scale) + text_h
        cv2.putText(frame, not_eval_text, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, text_scale, STATUS_BAD_COLOR, 2, cv2.LINE_AA)

    return frame


def draw_body_axis_panel(frame, axis, jitter_result=None):
    """畫 Body Axis Proportion Analysis 的獨立面板，貼在 Bone Stability
    面板正下方。刻意分開放（不是合併成同一塊）：這個模組看的是「這一幀
    身體主軸比例像不像一隻真的貓」，跟上面 Bone Stability 面板看的「這個
    窗口穩不穩定」是完全不同的問題，混在一起顯示容易讓人誤以為是同一組
    指標在講同一件事。

    jitter_result 是 compute_body_axis_score_jitter() 的回傳值
    (jitter, valid_pair_count)，為 None（該窗口還沒湊滿、節流幀沿用上次
    結果之前）時不畫這一行，其餘欄位照常顯示——單幀分數不依賴窗口，兩者
    節流節奏本來就不一樣。"""
    if axis is None:
        return frame
    h, w = frame.shape[:2]
    ui_scale = max(0.9, min(1.6, np.hypot(w, h) / 1500.0))
    panel_w = int(430 * ui_scale)
    panel_h = int(210 * ui_scale)
    pad = int(6 * ui_scale)
    bone_panel_h = int(210 * ui_scale)
    x0 = pad
    y0 = pad + bone_panel_h + int(6 * ui_scale)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (120, 120, 120), 1, cv2.LINE_AA)

    tx = x0 + int(8 * ui_scale)
    ty = y0 + int(22 * ui_scale)
    f = FONT_SCALE_MULT

    score = axis.get("body_axis_score", float("nan"))
    score_color = _status_color_for("body_axis_score", score)
    title = f"Body Axis Score: {score:.1f}" if np.isfinite(score) else "Body Axis Score: --"
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * f * ui_scale, score_color, 2, cv2.LINE_AA)
    ty += int(30 * ui_scale)

    err = axis.get("geometry_error", float("nan"))
    err_text = f"geometry error: {err:.4f}" if np.isfinite(err) else "geometry error: --"
    cv2.putText(frame, err_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, NEUTRAL_TEXT_COLOR, 2, cv2.LINE_AA)
    ty += int(28 * ui_scale)

    for label, key in (
        ("nose-chest (ref only)", "nose_chest_ratio"),
        ("chest-midback", "chest_midback_ratio"),
        ("midback-hip", "midback_hip_ratio"),
    ):
        value = axis.get(key, float("nan"))
        text = f"{label}: {value:.3f}" if np.isfinite(value) else f"{label}: --"
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, NEUTRAL_TEXT_COLOR, 2, cv2.LINE_AA)
        ty += int(28 * ui_scale)

    if jitter_result is not None:
        jitter, _valid_pair_count = jitter_result
        jitter_color = _status_color_for("body_axis_score_jitter", jitter)
        jitter_text = f"score jitter: {jitter:.1f}/f" if np.isfinite(jitter) else "score jitter: --"
        cv2.putText(frame, jitter_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * f * ui_scale, jitter_color, 2, cv2.LINE_AA)

    return frame


def draw_skeleton(frame, kpts, kpt_conf, sx, sy, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    for (a, b) in SKELETON_EDGES:
        if kpt_conf[a] < conf_thresh or kpt_conf[b] < conf_thresh:
            continue
        pa = (int(kpts[a, 0] * sx), int(kpts[a, 1] * sy))
        pb = (int(kpts[b, 0] * sx), int(kpts[b, 1] * sy))
        cv2.line(frame, pa, pb, (0, 200, 0), 2, cv2.LINE_AA)
    for i in range(kpts.shape[0]):
        if kpt_conf[i] < conf_thresh:
            continue
        p = (int(kpts[i, 0] * sx), int(kpts[i, 1] * sy))
        cv2.circle(frame, p, 4, (0, 0, 220), -1, cv2.LINE_AA)
    return frame


def draw_bbox(frame, bbox, sx, sy, bbox_conf=None):
    """畫 YOLO 偵測到的外框（x1,y1,x2,y2，原始像素座標）。bbox 為 None 或含
    NaN（該幀沒有 bbox）時不畫，直接回傳原圖。這個框跟骨架關鍵點是 YOLO
    兩個獨立輸出頭，畫出來方便直接目視比對 torso_ratio 用的參照框跟骨架
    是否吻合。bbox_conf 不為 None 時，在框的左上角疊一個信心值標籤（仿
    1_run_video_inference.py::draw_test2_style_overlay 的做法），方便對照
    torso_ratio 這類依賴 bbox 的指標是否受到低信心偵測影響。"""
    if bbox is None or np.isnan(bbox).any():
        return frame
    x1 = int(bbox[0] * sx)
    y1 = int(bbox[1] * sy)
    x2 = int(bbox[2] * sx)
    y2 = int(bbox[3] * sy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2, cv2.LINE_AA)

    if bbox_conf is not None:
        label = f"{float(bbox_conf):.2f}"
        label_fs = 0.5
        label_th = 1
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_fs, label_th)
        pad = 3
        label_h = th + baseline + pad * 2
        label_w = tw + pad * 2
        # 框頂部太靠近畫面上緣（放不下標籤）時翻到框內側，避免被裁掉。
        fits_above = (y1 - label_h) >= 0
        rect_y1 = (y1 - label_h) if fits_above else y1
        rect_y2 = y1 if fits_above else (y1 + label_h)
        text_x = x1 + pad
        text_y = rect_y2 - pad - baseline
        cv2.rectangle(frame, (x1, rect_y1), (x1 + label_w, rect_y2), (255, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, label_fs, (0, 0, 0), label_th, cv2.LINE_AA)

    return frame


def compute_midback_angle(kpts, kpt_conf, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """算 Chest(3)-MidBack(4)-Hip(5) 這個夾角（以 MidBack 為頂點），單位度。
    注意：這裡的標註慣例是把 MidBack 點在貓背部拱起的最高點，所以這三點
    正常情況下就是一個三角形，不是一直線——夾角本來就會小於 180 度，實際
    大小跟貓拱背弧度/姿勢有關，並非越接近 180 度就代表偵測越正確。目前
    沒有已知的「正常基準值」，這個角度只當參考數值顯示，不代表風險程度。
    任一點信心不足或兩個向量長度太短（幾乎重疊）時回傳 None。"""
    if kpt_conf[3] < conf_thresh or kpt_conf[4] < conf_thresh or kpt_conf[5] < conf_thresh:
        return None
    chest = kpts[3, :2]
    midback = kpts[4, :2]
    hip = kpts[5, :2]
    v1 = chest - midback
    v2 = hip - midback
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_angle = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def draw_midback_angle(frame, kpts, kpt_conf, sx, sy, conf_thresh=DRAW_KP_CONF_THRESHOLD):
    """把 Chest-MidBack-Hip 夾角畫在 MidBack 關鍵點的螢幕位置旁邊——跟骨架
    穩定度面板（左上角，窗口統計量）不同，這是直接疊在關鍵點上的單幀
    幾何量，方便對照畫面當下貓的姿勢跟角度數字是否合理。"""
    angle = compute_midback_angle(kpts, kpt_conf, conf_thresh)
    if angle is None:
        return frame
    px = int(kpts[4, 0] * sx)
    py = int(kpts[4, 1] * sy)
    color = REFERENCE_VALUE_COLOR
    text = f"{angle:.0f}deg"
    org = (px + 8, py - 8)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def _save_video_results(records, video_path_obj: Path, behavior: str, video_index: int, color: str, out_dir: Path):
    """把一支影片收集到的逐窗口 records 存成 CSV + 10 面板圖表（五個指標各自
    的時序＋分布圖），並印出摘要統計。GUI 模式（analyze_single_video）跟
    背景批次模式
    （analyze_single_video_batch）收集 records 的過程不同（前者即時顯示、
    可暫停/逐幀，後者純背景跑），但收集到的 records 格式完全一樣，輸出
    邏輯抽成這個共用函式，不重複寫兩份。沒有 records 時回傳 None，否則
    回傳 (records, behavior)。"""
    if not records:
        print(f"⚠ {video_path_obj.name}: 沒有收集到任何有效窗口資料（影片太短或骨架偵測失敗），略過輸出")
        return None

    video_out_dir = out_dir / video_path_obj.stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = video_out_dir / "bone_stability_timeseries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_index", "video_name", "behavior", "frame", "time_sec",
            "midback_offset_ratio", "torso_ratio", "midback_angle_jitter",
            "torso_ratio_jitter", "bone_length_oscillation",
        ])
        writer.writeheader()
        writer.writerows(records)

    frames = [r["frame"] for r in records]
    mb_ratios = [r["midback_offset_ratio"] for r in records]
    torso_ratios = [r["torso_ratio"] for r in records]
    mb_angle_jitters = [r["midback_angle_jitter"] for r in records]
    torso_jitters = [r["torso_ratio_jitter"] for r in records]
    bone_oscillations = [r["bone_length_oscillation"] for r in records]

    mb_arr_all = np.array(mb_ratios, dtype=np.float64)
    mb_valid_all = mb_arr_all[~np.isnan(mb_arr_all)]

    torso_arr_all = np.array(torso_ratios, dtype=np.float64)
    torso_valid_all = torso_arr_all[~np.isnan(torso_arr_all)]

    mb_angle_jitter_arr_all = np.array(mb_angle_jitters, dtype=np.float64)
    mb_angle_jitter_valid_all = mb_angle_jitter_arr_all[~np.isnan(mb_angle_jitter_arr_all)]

    torso_jitter_arr_all = np.array(torso_jitters, dtype=np.float64)
    torso_jitter_valid_all = torso_jitter_arr_all[~np.isnan(torso_jitter_arr_all)]

    bone_osc_arr_all = np.array(bone_oscillations, dtype=np.float64)
    bone_osc_valid_all = bone_osc_arr_all[~np.isnan(bone_osc_arr_all)]

    safe_name = ascii_safe_label(video_path_obj.name, f"video_{video_index}")

    # 五個逐窗口指標，各佔一組時序＋分布圖：
    #   Spatial Consistency：midback_offset_ratio、torso_ratio
    #   Temporal Consistency：midback_angle_jitter、torso_ratio_jitter、
    #                         bone_length_oscillation
    # 5 個指標各佔一列，左欄時序線圖（較寬，橫軸是整支影片的幀數，需要較多
    # 橫向空間才看得出趨勢）、右欄分布直方圖（較窄，本身接近方形即可），用
    # width_ratios=[2, 1] 依內容合理分配版面，取代原本 10x1 等高堆疊（時序圖
    # 跟直方圖用同一種尺寸並不合理，而且整張圖高達 40 吋、閱讀時要一直往下捲）。
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1]})

    axes[0, 0].plot(frames, mb_ratios, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[0, 0].axhline(CANDIDATE_MIDBACK_OFFSET_THRESHOLD, color="red", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_OFFSET_THRESHOLD}")
    axes[0, 0].set_xlabel("frame")
    axes[0, 0].set_ylabel("midback offset ratio")
    axes[0, 0].set_title(f"mid_back offset ratio over time - #{video_index} {safe_name} [{behavior}] [KEY METRIC]", fontsize=10, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    if mb_valid_all.size > 0:
        axes[0, 1].hist(mb_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[0, 1].axvline(CANDIDATE_MIDBACK_OFFSET_THRESHOLD, color="red", linestyle="--", linewidth=1,
                            label=f"candidate threshold = {CANDIDATE_MIDBACK_OFFSET_THRESHOLD}")
        axes[0, 1].set_xlabel("midback offset ratio")
        axes[0, 1].set_ylabel("frame count")
        axes[0, 1].set_title("midback offset ratio distribution [KEY METRIC]", fontsize=10, fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(frames, torso_ratios, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[1, 0].axhline(CANDIDATE_TORSO_RATIO_THRESHOLD, color="red", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_THRESHOLD}")
    axes[1, 0].set_xlabel("frame")
    axes[1, 0].set_ylabel("torso ratio (chest-hip / bbox diagonal)")
    axes[1, 0].set_title("torso ratio over time [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    if torso_valid_all.size > 0:
        axes[1, 1].hist(torso_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[1, 1].axvline(CANDIDATE_TORSO_RATIO_THRESHOLD, color="red", linestyle="--", linewidth=1,
                            label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_THRESHOLD}")
        axes[1, 1].set_xlabel("torso ratio (chest-hip / bbox diagonal)")
        axes[1, 1].set_ylabel("frame count")
        axes[1, 1].set_title("torso ratio distribution [KEY METRIC]", fontsize=11, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    axes[2, 0].plot(frames, mb_angle_jitters, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[2, 0].axhline(CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD}")
    axes[2, 0].set_xlabel("frame")
    axes[2, 0].set_ylabel("midback angle jitter (deg/frame)")
    axes[2, 0].set_title("midback angle jitter over time [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)

    if mb_angle_jitter_valid_all.size > 0:
        axes[2, 1].hist(mb_angle_jitter_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[2, 1].axvline(CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                            label=f"candidate threshold = {CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD}")
        axes[2, 1].set_xlabel("midback angle jitter (deg/frame)")
        axes[2, 1].set_ylabel("frame count")
        axes[2, 1].set_title("midback angle jitter distribution [KEY METRIC]", fontsize=11, fontweight="bold")
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)

    axes[3, 0].plot(frames, torso_jitters, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[3, 0].axhline(CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD}")
    axes[3, 0].set_xlabel("frame")
    axes[3, 0].set_ylabel("torso ratio jitter (/frame)")
    axes[3, 0].set_title("skeleton scale jitter over time [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[3, 0].legend()
    axes[3, 0].grid(alpha=0.3)

    if torso_jitter_valid_all.size > 0:
        axes[3, 1].hist(torso_jitter_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[3, 1].axvline(CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, color="red", linestyle="--", linewidth=1,
                            label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD}")
        axes[3, 1].set_xlabel("torso ratio jitter (/frame)")
        axes[3, 1].set_ylabel("frame count")
        axes[3, 1].set_title("skeleton scale jitter distribution [KEY METRIC]", fontsize=11, fontweight="bold")
        axes[3, 1].legend()
        axes[3, 1].grid(alpha=0.3)

    axes[4, 0].plot(frames, bone_oscillations, linewidth=1, color=color, label=f"#{video_index} {behavior}")
    axes[4, 0].axhline(CANDIDATE_BONE_OSCILLATION_THRESHOLD, color="red", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_BONE_OSCILLATION_THRESHOLD}")
    axes[4, 0].set_xlabel("frame")
    axes[4, 0].set_ylabel("bone length oscillation (/frame)")
    axes[4, 0].set_title("bone length oscillation over time [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[4, 0].legend()
    axes[4, 0].grid(alpha=0.3)

    if bone_osc_valid_all.size > 0:
        axes[4, 1].hist(bone_osc_valid_all, bins=40, color=color, edgecolor="white", label=f"#{video_index} {behavior}")
        axes[4, 1].axvline(CANDIDATE_BONE_OSCILLATION_THRESHOLD, color="red", linestyle="--", linewidth=1,
                            label=f"candidate threshold = {CANDIDATE_BONE_OSCILLATION_THRESHOLD}")
        axes[4, 1].set_xlabel("bone length oscillation (/frame)")
        axes[4, 1].set_ylabel("frame count")
        axes[4, 1].set_title("bone length oscillation distribution [KEY METRIC]", fontsize=11, fontweight="bold")
        axes[4, 1].legend()
        axes[4, 1].grid(alpha=0.3)

    png_path = video_out_dir / "bone_stability_chart.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"✅ {video_path_obj.name} 完成 — CSV: {csv_path}")
    print(f"   有效窗口數: {len(records)}")
    if mb_valid_all.size > 0:
        print(f"   midback offset: min={mb_valid_all.min():.4f} median={np.median(mb_valid_all):.4f} "
              f"mean={mb_valid_all.mean():.4f} max={mb_valid_all.max():.4f}")
    if torso_valid_all.size > 0:
        print(f"   torso ratio: min={torso_valid_all.min():.4f} median={np.median(torso_valid_all):.4f} "
              f"mean={torso_valid_all.mean():.4f} max={torso_valid_all.max():.4f}")
    if mb_angle_jitter_valid_all.size > 0:
        print(f"   midback angle jitter: min={mb_angle_jitter_valid_all.min():.1f} median={np.median(mb_angle_jitter_valid_all):.1f} "
              f"mean={mb_angle_jitter_valid_all.mean():.1f} max={mb_angle_jitter_valid_all.max():.1f}")
    if torso_jitter_valid_all.size > 0:
        print(f"   skeleton scale jitter: min={torso_jitter_valid_all.min():.4f} median={np.median(torso_jitter_valid_all):.4f} "
              f"mean={torso_jitter_valid_all.mean():.4f} max={torso_jitter_valid_all.max():.4f}")
    if bone_osc_valid_all.size > 0:
        print(f"   bone length oscillation: min={bone_osc_valid_all.min():.4f} median={np.median(bone_osc_valid_all):.4f} "
              f"mean={bone_osc_valid_all.mean():.4f} max={bone_osc_valid_all.max():.4f}")

    return records, behavior


def analyze_single_video(video_path: str, detector: KeypointDetector, video_index: int, classifier: BehaviorClassifier = None):
    """GUI 視覺偵測模式：純觀看/測試，開視窗即時顯示骨架、穩定度面板、
    MidBack 夾角、ST-GCN 分類標籤，不收集 records、不輸出 CSV、不產生任何
    圖表——統計分析是背景模式（analyze_single_video_batch）的工作，兩者
    職責分開。classifier 為 None 時（模型未載入）只是不畫分類標籤，其餘
    功能不受影響。

    單支影片播完會自動從頭循環播放（不會自己跳下一支），直到使用者按
    [1]/[2] 才切換到上一支/下一支影片，或按 [q] 結束整個檢視——跟
    1_run_video_inference.py 的 LOOP_PLAYBACK + 1/2 切換邏輯一致。

    只在 frame_idx % OVERLAY_STRIDE == 0 時才重算 overlay（其餘幀沿用
    上一次結果顯示，不重跑前處理），對齊正式推論的 WINDOW_STRIDE 節奏。

    暫停時可按 [a]/[d] 逐幀後退/前進檢視：關鍵點偵測結果全部快取在
    kp_cache（frame_idx -> (kpts, kpt_conf, bbox, bbox_conf) 或 None），[a] 往回只在已經
    處理過的範圍內用快取重新計算 overlay 顯示（不必重跑 YOLO），[d] 往前
    若已追到播放前緣則等同正常推進播放一格。

    回傳字串訊號給呼叫端決定下一步："quit"（按 q 或影片打不開/不存在）、
    "prev"（按 1，切上一支）、"next"（按 2，切下一支）。
    """
    video_path_obj = Path(video_path)
    if video_path_obj.is_dir():
        print(f"❌ 略過（指向資料夾，不是影片檔案）: {video_path}")
        return "next"
    if not video_path_obj.exists():
        print(f"❌ 略過（找不到檔案）: {video_path}")
        return "next"

    behavior = infer_behavior_label(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 略過（無法開啟）: {video_path}")
        return "next"

    # frame_idx(1-based) -> (kpts, kpt_conf, bbox) 或 None（偵測失敗）——完整
    # 保留每一幀的偵測結果（只存關鍵點座標/信心值/bbox，體積很小），讓暫停
    # 時可以用 [a]/[d] 在已播放過的範圍內來回檢視，不必重跑 YOLO。bbox 是
    # (x1,y1,x2,y2) 或 None，用來算 torso_ratio。原始畫面本身不快取（太佔
    # 記憶體），要重畫哪一幀就用 cv2.CAP_PROP_POS_FRAMES 現場 seek 回去讀。
    # 循環播放重新開始時會整個清空，不跨圈保留。
    kp_cache: dict = {}

    frame_idx = 0   # 播放前緣：目前這一圈最新已讀取／偵測過的幀號
    cur_idx = 0     # 目前畫面上顯示的幀號（<= frame_idx）
    ovl = None      # 一般播放節奏下持續保留到下一次重算，避免節流幀顯示空白
    cls = None      # ST-GCN 分類結果 (behavior_id, confidence, probs)，節流邏輯跟 ovl 一致
    axis = None     # Body Axis Proportion Analysis 結果，單幀計算，不受 OVERLAY_STRIDE 節流影響
    axis_jitter = None  # Body Axis Score Jitter 結果 (jitter, valid_pair_count)，需要窗口，節流邏輯跟 ovl 一致
    loop_count = 0  # 這支影片目前播放到第幾圈（0 = 第一遍還沒播完）

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *DISPLAY_SIZE)

    print(f"▶ 開始檢視 #{video_index}: {video_path}  [{behavior}]")
    print(f"  [space]=暫停/繼續  [a]/[d]=暫停時逐幀後退/前進  [1]/[2]=上一支/下一支影片  "
          f"[j]=切換雙重判定 ON/OFF  [q]=結束"
          f"  （每 {OVERLAY_STRIDE} 幀重算一次；單支影片會循環播放；純觀看，不輸出 CSV/圖表）")

    def _window_ending_at(end_idx):
        """從 kp_cache 取出結束於 end_idx（含）的連續 SEQUENCE_LENGTH 幀窗口，
        只要中間任何一幀偵測失敗或還沒處理過就回傳 None——要求連續成功
        幀數。bbox 缺值（沒有 bbox 的幀）不影響整個窗口，用 NaN 填那一列，
        torso_ratio 內部自己用 NaN mask 排除。"""
        start_idx = end_idx - SEQUENCE_LENGTH + 1
        if start_idx < 1:
            return None
        kpts_list, conf_list, bbox_list = [], [], []
        for i in range(start_idx, end_idx + 1):
            entry = kp_cache.get(i)
            if entry is None:
                return None
            kpts_list.append(entry[0])
            conf_list.append(entry[1])
            bbox_list.append(entry[2] if entry[2] is not None else np.full(4, np.nan))
        return np.stack(kpts_list, axis=0), np.stack(conf_list, axis=0), np.stack(bbox_list, axis=0)

    def _draw_display(target_idx, frame_bgr, kpts, kpt_conf, bbox, bbox_conf, axis_for_display, axis_jitter_for_display, ovl_for_display, cls_for_display, manual):
        h, w = frame_bgr.shape[:2]
        sx = DISPLAY_SIZE[0] / w
        sy = DISPLAY_SIZE[1] / h
        disp = cv2.resize(frame_bgr, DISPLAY_SIZE)
        disp = draw_bbox(disp, bbox, sx, sy, bbox_conf)
        if kpts is not None:
            disp = draw_skeleton(disp, kpts, kpt_conf, sx, sy)
            disp = draw_midback_angle(disp, kpts, kpt_conf, sx, sy)
        if ovl_for_display is not None:
            disp = draw_bone_stability_panel(disp, ovl_for_display)
        if axis_for_display is not None:
            disp = draw_body_axis_panel(disp, axis_for_display, axis_jitter_for_display)
        if cls_for_display is not None:
            cls_behavior_id, cls_confidence, _ = cls_for_display
            disp = draw_behavior_label(disp, cls_behavior_id, cls_confidence)
        tag = "  [MANUAL a/d]" if manual else ""
        loop_tag = f"  loop {loop_count + 1}" if loop_count > 0 else ""
        cv2.putText(disp, f"#{video_index} {video_path_obj.name} [{behavior}]  frame {target_idx}{loop_tag}{tag}",
                    (10, DISPLAY_SIZE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        return disp

    def _restart_loop():
        """單支影片播完自動從頭重播：seek 回第 0 幀，清空這一圈的快取跟
        播放前緣（不跨圈保留，行為等同重新開始播放同一支影片）。"""
        nonlocal frame_idx, cur_idx, ovl, cls, axis, axis_jitter, loop_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        kp_cache.clear()
        frame_idx = 0
        cur_idx = 0
        ovl = None
        cls = None
        axis = None
        axis_jitter = None
        loop_count += 1

    def _advance_playback():
        """往前推進播放正好一幀（正常播放的每一幀、或暫停時 [d] 追到播放
        前緣後再往前，都走這一條路徑，避免兩處重複同一段邏輯）。回傳
        (ok, new_display)，ok=False 代表這一圈已讀完（呼叫端會觸發重播）。"""
        nonlocal frame_idx, cur_idx, ovl, cls, axis, axis_jitter
        ret, frame_bgr = cap.read()
        if not ret:
            return False, None
        frame_idx += 1
        cur_idx = frame_idx

        kpts, kpt_conf, bbox, det_conf = detector.detect(frame_bgr)
        kp_cache[frame_idx] = (kpts.copy(), kpt_conf.copy(), bbox.copy() if bbox is not None else None, det_conf) if kpts is not None else None

        if kpts is None:
            ovl = None
            cls = None
            axis = None
            axis_jitter = None
        else:
            # Body Axis Proportion Analysis 只看這一幀本身，跟窗口/節流無關，每幀都重算。
            axis = compute_body_axis_geometry(kpts)
            window = _window_ending_at(frame_idx)
            if window is not None:
                if frame_idx % OVERLAY_STRIDE == 0:
                    seq_window, conf_window, bbox_window = window
                    ovl = compute_bone_stability_overlay(seq_window, conf_window, bbox_window)
                    # Body Axis Score Jitter 是窗口層級的時間抖動量，跟 ovl 用
                    # 同一個窗口、同一個節流節奏重算。把 Body Axis 的結果併入
                    # 同一個 ovl dict，SQA_ENABLED_THRESHOLDS/apply_sqa_dual_
                    # judgment 才吃得到這兩項，覆蓋規則才會真的生效。
                    axis_jitter = compute_body_axis_score_jitter(seq_window)
                    ovl["body_axis_score"] = axis["body_axis_score"]
                    ovl["body_axis_score_jitter"] = axis_jitter[0]
                    cls = classify_bone_window(seq_window, conf_window, classifier)
                    cls = apply_sqa_dual_judgment(cls, ovl)
                # else: 節流幀，沿用上一次算好的 ovl/cls/axis_jitter 顯示，不重算
            else:
                ovl = None
                cls = None
                axis_jitter = None

        return True, _draw_display(frame_idx, frame_bgr, kpts, kpt_conf, bbox, det_conf, axis, axis_jitter, ovl, cls, manual=False)

    def _seek_to_cached(target_idx):
        """[a]/[d] 手動逐幀檢視：target_idx 必須在 1..frame_idx（已處理過的
        範圍）內，用 kp_cache 重新算 overlay/分類顯示，不寫進 records、不
        重跑 YOLO，只重新 seek 讀取那一幀的原始畫面來畫。"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx - 1)
        ret, frame_bgr = cap.read()
        if not ret:
            return None
        entry = kp_cache.get(target_idx)
        kpts_v, kconf_v, bbox_v, det_conf_v = entry if entry is not None else (None, None, None, None)
        axis_v = compute_body_axis_geometry(kpts_v) if kpts_v is not None else None
        window = _window_ending_at(target_idx)
        axis_jitter_v = compute_body_axis_score_jitter(window[0]) if window is not None else None
        ovl_v = compute_bone_stability_overlay(*window) if window is not None else None
        if ovl_v is not None:
            # 跟 _advance_playback 一致：把 Body Axis 的結果併入同一個 ovl
            # dict，SQA_ENABLED_THRESHOLDS/apply_sqa_dual_judgment 才吃得到。
            ovl_v["body_axis_score"] = axis_v["body_axis_score"] if axis_v is not None else float("nan")
            ovl_v["body_axis_score_jitter"] = axis_jitter_v[0] if axis_jitter_v is not None else float("nan")
        cls_v = classify_bone_window(window[0], window[1], classifier) if window is not None else None
        cls_v = apply_sqa_dual_judgment(cls_v, ovl_v)
        return _draw_display(target_idx, frame_bgr, kpts_v, kconf_v, bbox_v, det_conf_v, axis_v, axis_jitter_v, ovl_v, cls_v, manual=True)

    paused = False
    display = None
    signal = "next"
    while True:
        if not paused:
            ok, new_display = _advance_playback()
            if not ok:
                _restart_loop()
                continue
            display = new_display

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key == ord('q'):
            signal = "quit"
            break
        if key == ord('2'):
            signal = "next"
            break
        if key == ord('1'):
            signal = "prev"
            break
        if key == ord(' '):
            paused = not paused
        elif key == ord('j'):
            # 即時切換「GCN 分類為主、幾何判斷為輔」雙重判定（j = Judgment），
            # 不用改程式碼重跑：修改的是模組層級全域變數，
            # apply_sqa_dual_judgment() 每次呼叫都會讀到最新值。切換後立刻
            # 用 _seek_to_cached 重繪目前這一幀，不必等下一次 OVERLAY_STRIDE
            # 節流重算才看得到效果。
            global ENABLE_SQA_DUAL_JUDGMENT
            ENABLE_SQA_DUAL_JUDGMENT = not ENABLE_SQA_DUAL_JUDGMENT
            print(f"  [j] 雙重判定切換為: {'ON' if ENABLE_SQA_DUAL_JUDGMENT else 'OFF'}")
            if cur_idx >= 1:
                new_display = _seek_to_cached(cur_idx)
                if new_display is not None:
                    display = new_display
        elif paused and key == ord('a'):
            if cur_idx > 1:
                new_display = _seek_to_cached(cur_idx - 1)
                if new_display is not None:
                    cur_idx -= 1
                    display = new_display
        elif paused and key == ord('d'):
            if cur_idx < frame_idx:
                new_display = _seek_to_cached(cur_idx + 1)
                if new_display is not None:
                    cur_idx += 1
                    display = new_display
            else:
                # 已經追到播放前緣，[d] 再往前一格＝正常推進播放一幀
                ok, new_display = _advance_playback()
                if ok:
                    display = new_display

    cap.release()
    return signal


def analyze_single_video_batch(video_path: str, detector: KeypointDetector, out_dir: Path, video_index: int):
    """背景統計分析模式：不開視窗、不畫骨架/面板、不等待按鍵，單純依序讀取
    每一幀跑偵測，節流幀（OVERLAY_STRIDE）算 overlay，跑完整支影片才輸出
    CSV/圖表。跟 GUI 模式（analyze_single_video）比較：
      - 拿掉 cv2.namedWindow/imshow/waitKey 與所有畫面繪製（resize、
        draw_skeleton、draw_bbox、draw_midback_angle、draw_bone_stability_panel）—
        沒人在看的時候這些純粹是浪費時間。
      - 拿掉支援 [a]/[d] 逐幀回看用的 kp_cache（整支影片全部保留），改回
        固定長度 SEQUENCE_LENGTH 的 deque 滾動窗口——背景模式不需要回看，
        用有界緩衝區記憶體用量不會隨影片長度增加。
    收集到的 records 格式、CSV/圖表輸出邏輯跟 GUI 模式完全一致（見
    _save_video_results），行為判斷、門檻設定也共用同一套設定值，兩種
    模式產出的分析結果可以直接放在一起比較。

    回傳 (records, behavior)，behavior 是從路徑推斷出的行為標籤。
    """
    video_path_obj = Path(video_path)
    if video_path_obj.is_dir():
        print(f"❌ 略過（指向資料夾，不是影片檔案）: {video_path}")
        return None
    if not video_path_obj.exists():
        print(f"❌ 略過（找不到檔案）: {video_path}")
        return None

    behavior = infer_behavior_label(video_path)
    color = color_for_index(video_index)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 略過（無法開啟）: {video_path}")
        return None

    kp_buffer = deque(maxlen=SEQUENCE_LENGTH)
    conf_buffer = deque(maxlen=SEQUENCE_LENGTH)
    bbox_buffer = deque(maxlen=SEQUENCE_LENGTH)

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    records = []

    print(f"▶ [背景] 開始分析 #{video_index}: {video_path}  [{behavior}]")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        kpts, kpt_conf, bbox, _det_conf = detector.detect(frame)
        if kpts is None:
            kp_buffer.clear()
            conf_buffer.clear()
            bbox_buffer.clear()
            continue

        kp_buffer.append(kpts)
        conf_buffer.append(kpt_conf)
        bbox_buffer.append(bbox if bbox is not None else np.full(4, np.nan))

        if len(kp_buffer) == SEQUENCE_LENGTH and frame_idx % OVERLAY_STRIDE == 0:
            seq_window = np.stack(kp_buffer, axis=0)
            conf_window = np.stack(conf_buffer, axis=0)
            bbox_window = np.stack(bbox_buffer, axis=0)
            ovl = compute_bone_stability_overlay(seq_window, conf_window, bbox_window)
            records.append({
                "video_index": video_index,
                "video_name": video_path_obj.stem,
                "behavior": behavior,
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 3),
                "midback_offset_ratio": ovl["midback_offset_ratio"],
                "torso_ratio": ovl["torso_ratio"],
                "midback_angle_jitter": ovl["midback_angle_jitter"],
                "torso_ratio_jitter": ovl["torso_ratio_jitter"],
                "bone_length_oscillation": ovl["bone_length_oscillation"],
            })

    cap.release()
    print(f"   共讀取 {frame_idx} 幀")

    return _save_video_results(records, video_path_obj, behavior, video_index, color, out_dir)


def build_comparison_chart(video_results: dict, out_dir: Path):
    """video_results: {video_name: (records, behavior, video_index)}。把每支
    影片疊在同一張時序圖跟同一張分佈直方圖上，顏色依「索引」區分（每支影片
    獨立顏色，不再依行為分組），並在時序線的右端標上索引數字，不用回頭查
    圖例就能認出哪條線是哪支影片；行為標籤仍保留在圖例文字裡。
    density 正規化避免因影片長度不同而失真。存到 run 根目錄的
    comparison_chart.png。
    """
    # 跟 _save_video_results 同一套版面邏輯：5 個指標各佔一列，左欄時序線圖
    # （較寬）、右欄分布直方圖（較窄），依內容合理分配版面，取代原本 10x1
    # 等高堆疊。
    fig, axes = plt.subplots(5, 2, figsize=(17, 20), constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1]})

    for name, (records, behavior, video_index) in video_results.items():
        color = color_for_index(video_index)
        safe_name = ascii_safe_label(name, f"video_{video_index}")
        legend_label = f"#{video_index} {behavior}: {safe_name}"
        times = [r["time_sec"] for r in records]

        mb_ratios = np.array([r["midback_offset_ratio"] for r in records], dtype=np.float64)
        axes[0, 0].plot(times, mb_ratios, linewidth=1, color=color, alpha=0.85, label=legend_label)
        mb_valid_mask = ~np.isnan(mb_ratios)
        if np.any(mb_valid_mask):
            last_mi = np.where(mb_valid_mask)[0][-1]
            axes[0, 0].annotate(
                str(video_index), xy=(times[last_mi], mb_ratios[last_mi]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        mb_valid = mb_ratios[mb_valid_mask]
        if mb_valid.size > 0:
            axes[0, 1].hist(mb_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        torso_ratios = np.array([r["torso_ratio"] for r in records], dtype=np.float64)
        axes[1, 0].plot(times, torso_ratios, linewidth=1.3, color=color, alpha=0.9, label=legend_label)
        torso_valid_mask = ~np.isnan(torso_ratios)
        if np.any(torso_valid_mask):
            last_ti = np.where(torso_valid_mask)[0][-1]
            axes[1, 0].annotate(
                str(video_index), xy=(times[last_ti], torso_ratios[last_ti]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        torso_valid = torso_ratios[torso_valid_mask]
        if torso_valid.size > 0:
            axes[1, 1].hist(torso_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        mb_angle_jitters = np.array([r["midback_angle_jitter"] for r in records], dtype=np.float64)
        axes[2, 0].plot(times, mb_angle_jitters, linewidth=1.3, color=color, alpha=0.9, label=legend_label)
        mb_angle_jitter_valid_mask = ~np.isnan(mb_angle_jitters)
        if np.any(mb_angle_jitter_valid_mask):
            last_ai = np.where(mb_angle_jitter_valid_mask)[0][-1]
            axes[2, 0].annotate(
                str(video_index), xy=(times[last_ai], mb_angle_jitters[last_ai]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        mb_angle_jitter_valid = mb_angle_jitters[mb_angle_jitter_valid_mask]
        if mb_angle_jitter_valid.size > 0:
            axes[2, 1].hist(mb_angle_jitter_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        torso_jitters = np.array([r["torso_ratio_jitter"] for r in records], dtype=np.float64)
        axes[3, 0].plot(times, torso_jitters, linewidth=1.3, color=color, alpha=0.9, label=legend_label)
        torso_jitter_valid_mask = ~np.isnan(torso_jitters)
        if np.any(torso_jitter_valid_mask):
            last_ji = np.where(torso_jitter_valid_mask)[0][-1]
            axes[3, 0].annotate(
                str(video_index), xy=(times[last_ji], torso_jitters[last_ji]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        torso_jitter_valid = torso_jitters[torso_jitter_valid_mask]
        if torso_jitter_valid.size > 0:
            axes[3, 1].hist(torso_jitter_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

        bone_oscillations = np.array([r["bone_length_oscillation"] for r in records], dtype=np.float64)
        axes[4, 0].plot(times, bone_oscillations, linewidth=1.3, color=color, alpha=0.9, label=legend_label)
        bone_osc_valid_mask = ~np.isnan(bone_oscillations)
        if np.any(bone_osc_valid_mask):
            last_oi = np.where(bone_osc_valid_mask)[0][-1]
            axes[4, 0].annotate(
                str(video_index), xy=(times[last_oi], bone_oscillations[last_oi]),
                xytext=(5, 0), textcoords="offset points",
                color=color, fontsize=11, fontweight="bold", va="center",
            )

        bone_osc_valid = bone_oscillations[bone_osc_valid_mask]
        if bone_osc_valid.size > 0:
            axes[4, 1].hist(bone_osc_valid, bins=40, color=color, alpha=0.45, density=True, label=legend_label, edgecolor="none")

    axes[0, 0].axhline(CANDIDATE_MIDBACK_OFFSET_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_OFFSET_THRESHOLD}")
    axes[0, 0].set_xlabel("time (sec)")
    axes[0, 0].set_ylabel("midback offset ratio")
    axes[0, 0].set_title("mid_back offset ratio over time (line-end number = video index) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].axvline(CANDIDATE_MIDBACK_OFFSET_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_OFFSET_THRESHOLD}")
    axes[0, 1].set_xlabel("midback offset ratio")
    axes[0, 1].set_ylabel("density")
    axes[0, 1].set_title("midback offset ratio distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].axhline(CANDIDATE_TORSO_RATIO_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_THRESHOLD}")
    axes[1, 0].set_xlabel("time (sec)")
    axes[1, 0].set_ylabel("torso ratio (chest-hip / bbox diagonal)")
    axes[1, 0].set_title("torso ratio over time (line-end number = video index) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].axvline(CANDIDATE_TORSO_RATIO_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_THRESHOLD}")
    axes[1, 1].set_xlabel("torso ratio (chest-hip / bbox diagonal)")
    axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title("torso ratio distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)

    axes[2, 0].axhline(CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD}")
    axes[2, 0].set_xlabel("time (sec)")
    axes[2, 0].set_ylabel("midback angle jitter (deg/frame)")
    axes[2, 0].set_title("midback angle jitter over time (line-end number = video index) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(alpha=0.3)

    axes[2, 1].axvline(CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD}")
    axes[2, 1].set_xlabel("midback angle jitter (deg/frame)")
    axes[2, 1].set_ylabel("density")
    axes[2, 1].set_title("midback angle jitter distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(alpha=0.3)

    axes[3, 0].axhline(CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD}")
    axes[3, 0].set_xlabel("time (sec)")
    axes[3, 0].set_ylabel("torso ratio jitter (/frame)")
    axes[3, 0].set_title("skeleton scale jitter over time (line-end number = video index) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[3, 0].legend(fontsize=8)
    axes[3, 0].grid(alpha=0.3)

    axes[3, 1].axvline(CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD}")
    axes[3, 1].set_xlabel("torso ratio jitter (/frame)")
    axes[3, 1].set_ylabel("density")
    axes[3, 1].set_title("skeleton scale jitter distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[3, 1].legend(fontsize=8)
    axes[3, 1].grid(alpha=0.3)

    axes[4, 0].axhline(CANDIDATE_BONE_OSCILLATION_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_BONE_OSCILLATION_THRESHOLD}")
    axes[4, 0].set_xlabel("time (sec)")
    axes[4, 0].set_ylabel("bone length oscillation (/frame)")
    axes[4, 0].set_title("bone length oscillation over time (line-end number = video index) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[4, 0].legend(fontsize=8)
    axes[4, 0].grid(alpha=0.3)

    axes[4, 1].axvline(CANDIDATE_BONE_OSCILLATION_THRESHOLD, color="black", linestyle="--", linewidth=1,
                        label=f"candidate threshold = {CANDIDATE_BONE_OSCILLATION_THRESHOLD}")
    axes[4, 1].set_xlabel("bone length oscillation (/frame)")
    axes[4, 1].set_ylabel("density")
    axes[4, 1].set_title("bone length oscillation distribution (density-normalized) [KEY METRIC]", fontsize=11, fontweight="bold")
    axes[4, 1].legend(fontsize=8)
    axes[4, 1].grid(alpha=0.3)

    png_path = out_dir / "comparison_chart.png"
    fig.savefig(png_path, dpi=150)
    print(f"\n✅ 多影片比較圖已存: {png_path}")
    show_fig_fit_to_content(fig)


def build_metric_threshold_chart(video_results: dict, out_dir: Path, metric_key: str,
                                  metric_label: str, candidate_threshold: float,
                                  out_filename: str, show: bool = False, direction: str = "above"):
    """通用版門檻分析圖，metric_key 是參數，同一套方法可以套在任何一個逐窗口
    指標上——目前拿來看 torso_ratio／midback_offset_ratio／
    midback_angle_jitter 這幾個候選「骨架不可信」訊號，看哪一個對「正常 vs
    抖動」影片的分離效果最乾淨，而不是各寫一份重複邏輯。

    direction 決定「異常」的方向：
      "above"（預設，midback_offset_ratio／midback_angle_jitter 用）——
      數值越大越可疑，flag = 數值 > 門檻。
      "below"（torso_ratio 這類「軀幹相對 bbox 塌陷」的指標用）——數值越小
      越可疑，flag = 數值 < 門檻。

    畫三張圖：
      1. 逐支影片的經驗累積分布（ECDF）：x 軸是指標值，y 軸是「小於等於
         這個值的幀數佔全部幀數的比例」。理想情況下，穩定影片的曲線應該在
         候選門檻線之前就已經接近頂端，問題影片的曲線則會在門檻線之後還有
         明顯一段沒爬完——兩者在門檻附近的垂直距離，就是這個門檻分不分得開
         的直接證據。
      2. 逐支影片「被判為異常的幀數比例」長條圖：把①的判讀量化成一個
         數字，直接排序比較，一眼看出哪支影片被判定為「常常異常」。
      3. 全部影片合併的 pooled 直方圖（y 軸用 log scale），搭配候選門檻線，
         找整體分布有沒有自然斷層（gap）。

    回傳 flag_rates（[(video_index, behavior, safe_name, flag_rate_pct, color), ...]，
    依 flag_rate 由高到低排序），供呼叫端把多個指標的結果放在一起做最終比較。
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), constrained_layout=True)

    pooled = []
    flag_rates = []  # [(video_index, behavior, safe_name, flag_rate_pct, color), ...]

    for name, (records, behavior, video_index) in video_results.items():
        color = color_for_index(video_index)
        safe_name = ascii_safe_label(name, f"video_{video_index}")
        legend_label = f"#{video_index} {behavior}: {safe_name}"

        vals = np.array([r[metric_key] for r in records], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            continue
        pooled.append(valid)

        sorted_vals = np.sort(valid)
        cum_frac = np.arange(1, sorted_vals.size + 1) / sorted_vals.size
        axes[0].plot(sorted_vals, cum_frac, linewidth=1.4, color=color, label=legend_label)

        if direction == "above":
            flag_rate = float(np.mean(valid > candidate_threshold)) * 100.0
        else:
            flag_rate = float(np.mean(valid < candidate_threshold)) * 100.0
        flag_rates.append((video_index, behavior, safe_name, flag_rate, color))

    axes[0].axvline(candidate_threshold, color="black", linestyle="--", linewidth=1.5,
                     label=f"candidate threshold = {candidate_threshold}")
    for ref in (0.90, 0.95, 0.99):
        axes[0].axhline(ref, color="gray", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel(metric_label)
    axes[0].set_ylabel("cumulative fraction of frames")
    axes[0].set_title(f"ECDF per video - {metric_label} (dashed = candidate threshold, dotted = 90/95/99%)", fontsize=10)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    flag_rates.sort(key=lambda t: t[3], reverse=True)
    # 影片名稱後面加上排名（No.1 = flag rate 最高），排序邏輯本身沒變，
    # 只是把「第幾名」明確標在標籤上，不用自己數長條圖位置。
    labels = [f"No.{rank} #{idx} {beh}\n{nm}" for rank, (idx, beh, nm, _rate, _c) in enumerate(flag_rates, start=1)]
    rates = [rate for *_, rate, _c in flag_rates]
    bar_colors = [c for *_, c in flag_rates]
    bars = axes[1].bar(labels, rates, color=bar_colors)
    for bar, rate in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{rate:.1f}%",
                      ha="center", va="bottom", fontsize=9)
    op_symbol = ">" if direction == "above" else "<"
    axes[1].set_ylabel(f"% of frames with {metric_label} {op_symbol} threshold")
    axes[1].set_title(f"Flag rate per video at threshold = {candidate_threshold} (sorted, high to low)", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    if pooled:
        pooled_all = np.concatenate(pooled)
        counts, bin_edges, _patches = axes[2].hist(pooled_all, bins=60, color="#7f8c8d", edgecolor="white")
        axes[2].axvline(candidate_threshold, color="red", linestyle="--", linewidth=1.5,
                         label=f"candidate threshold = {candidate_threshold}")
        # 門檻線下方標出確切數值（紅字，跟門檻線同色）——用 get_xaxis_transform()
        # 讓 x 還是照資料座標對齊門檻線，但 y 改用「軸的比例」（0=底部），
        # 這樣不管 log 座標的 y 範圍怎麼變，這個數字永遠穩穩貼在 x 軸正下方。
        axes[2].text(candidate_threshold, -0.06, f"{candidate_threshold:.3f}",
                      transform=axes[2].get_xaxis_transform(), color="red", fontsize=9,
                      fontweight="bold", ha="center", va="top", clip_on=False)
        axes[2].set_yscale("log")
        axes[2].set_xlabel(metric_label)
        axes[2].set_ylabel("frame count (log scale)")
        axes[2].set_title(f"Pooled distribution across all videos - {metric_label} (log y-axis, look for a natural gap)", fontsize=10)
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

        # 全部影片加總的總幀數、被判為異常的幀數、佔比——三項數字直接寫在
        # 圖上（右上角文字框），不用再自己心算長條圖裡有幾根、加起來多少。
        total_frames = int(pooled_all.size)
        if direction == "above":
            flagged_count = int(np.sum(pooled_all > candidate_threshold))
        else:
            flagged_count = int(np.sum(pooled_all < candidate_threshold))
        flagged_pct = flagged_count / total_frames * 100.0 if total_frames > 0 else float("nan")
        summary_text = (
            f"Total frames (all videos): {total_frames}\n"
            f"Frames flagged ({metric_label} {op_symbol} {candidate_threshold:.3f}): {flagged_count}\n"
            f"% flagged: {flagged_pct:.1f}%"
        )
        axes[2].text(0.98, 0.95, summary_text, transform=axes[2].transAxes,
                      ha="right", va="top", fontsize=9,
                      bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

        print(f"\n📊 [{metric_label}] Pooled 總覽（{len(pooled)} 支影片加總）：總幀數 {total_frames}，"
              f"被判為異常（{op_symbol} {candidate_threshold:.3f}）{flagged_count} 幀，"
              f"佔比 {flagged_pct:.1f}%")

        # 每一根非空的長條上方標出確切幀數——log 座標只能目測大概高度，
        # 直接把數字寫在長條頂端最直觀。用直式文字（rotation=90）避免
        # 低數值區間相鄰長條的標籤互相重疊。
        bin_width = bin_edges[1] - bin_edges[0]
        for lo, cnt in zip(bin_edges[:-1], counts):
            if cnt > 0:
                axes[2].text(lo + bin_width / 2, cnt, str(int(cnt)),
                             rotation=90, ha="center", va="bottom", fontsize=6.5)

        # log 座標的長條圖只能目測大概高度，這裡額外把每一格 bin 的精確
        # 幀數印到終端機（只列有資料的 bin，大部分空 bin 沒必要洗版），
        # 跟長條上方的標籤互相對照，方便精確核對「斷層」從哪個數值開始。
        print(f"\n📊 [{metric_label}] Pooled 直方圖逐格明細（bin 寬度 = {bin_edges[1] - bin_edges[0]:.4f}，"
              f"共 {int(pooled_all.size)} 幀，{len(pooled)} 支影片加總）：")
        for lo, hi, cnt in zip(bin_edges[:-1], bin_edges[1:], counts):
            if cnt > 0:
                marker = "  <-- threshold" if lo <= candidate_threshold < hi else ""
                print(f"   [{lo:.4f}, {hi:.4f}): {int(cnt)} 幀{marker}")

    png_path = out_dir / out_filename
    fig.savefig(png_path, dpi=150)
    print(f"\n✅ [{metric_label}] 門檻分析圖已存: {png_path}")

    print(f"\n📊 [{metric_label}] 門檻分析摘要（候選門檻 = {candidate_threshold}）：")
    for idx, beh, nm, rate, _c in flag_rates:
        print(f"   #{idx} {beh} ({nm}): {rate:.2f}% 的幀被判為異常")

    if show:
        show_fig_fit_to_content(fig)
    else:
        plt.close(fig)

    return flag_rates


def resolve_video_paths() -> list:
    """依 INPUT_MODE 決定實際要處理的影片路徑清單："paths" 直接回傳
    VIDEO_PATHS；"folder" 則掃描 VIDEO_FOLDER（不含子資料夾）裡符合
    VIDEO_FOLDER_EXTENSIONS 的影片檔，依檔名排序，最多取前 10 支
    （超過 10 支會印警告並截斷，維持跟 "paths" 模式一樣 1~10 支的限制）。"""
    if INPUT_MODE == "paths":
        return list(VIDEO_PATHS)

    if INPUT_MODE == "folder":
        folder = Path(VIDEO_FOLDER)
        if not folder.is_dir():
            print(f"❌ VIDEO_FOLDER 不是有效的資料夾路徑: {folder}")
            return []
        found = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_FOLDER_EXTENSIONS
        )
        if not found:
            print(f"❌ 在資料夾裡找不到任何符合 {VIDEO_FOLDER_EXTENSIONS} 的影片: {folder}")
            return []
        if len(found) > 10:
            print(f"⚠ 資料夾裡有 {len(found)} 支影片，本工具最多同時分析 10 支，"
                  f"只取檔名排序後的前 10 支：{[p.name for p in found[:10]]}")
        return [str(p) for p in found[:10]]

    print(f"❌ INPUT_MODE 只能是 \"paths\" 或 \"folder\"，目前設定了 {INPUT_MODE!r}")
    return []


def _prompt_run_mode() -> str:
    """RUN_MODE 沒有預先寫死時，啟動時用選單詢問要跑哪個模式。"""
    print("請選擇執行模式：")
    print("  [1] 背景統計分析（不開視窗，跑最快，直接輸出 CSV/圖表）")
    print("  [2] GUI 視覺偵測（開視窗即時查看骨架與面板，可暫停/逐幀回看）")
    while True:
        choice = input("請輸入 1 或 2：").strip()
        if choice == "1":
            return "batch"
        if choice == "2":
            return "gui"
        print("輸入無效，請重新輸入 1 或 2")


def main():
    run_mode = RUN_MODE if RUN_MODE in ("batch", "gui") else _prompt_run_mode()
    print(f"執行模式: {'背景統計分析' if run_mode == 'batch' else 'GUI 視覺偵測'}\n")

    video_paths = resolve_video_paths()
    n = len(video_paths)
    if not (1 <= n <= 10):
        print(f"❌ 需要 1~10 支影片，目前解析出 {n} 支（INPUT_MODE = {INPUT_MODE!r}）")
        return

    detector = KeypointDetector(YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD)

    if run_mode == "gui":
        # ST-GCN 分類只有 GUI 模式的畫面標籤用得到，背景批次模式不需要載入
        # 這個模型（省下載入時間）。載入失敗時印出警告、classifier 留 None，
        # GUI 照常運作，只是不畫分類標籤（骨架穩定度面板等其餘功能不受影響）。
        try:
            classifier = BehaviorClassifier(STGCN_MODEL_PATH, device=INFERENCE_DEVICE,
                                             sequence_length=SEQUENCE_LENGTH, normalize=STGCN_NORMALIZE)
        except Exception as e:
            print(f"⚠ 無法載入 ST-GCN 模型（{STGCN_MODEL_PATH}）：{e}，將不顯示分類標籤")
            classifier = None

        # 純觀看/測試：不輸出 CSV、不產生任何圖表。單支影片會循環播放，
        # 按 [1]/[2] 才切換到上一支/下一支（可循環繞回），按 [q] 結束整個
        # 檢視——用 while + 可前後移動的索引，不是單純從頭到尾的 for 迴圈。
        print(f"共 {n} 支影片待檢視\n")
        current_video_idx = 0
        while True:
            video_path = video_paths[current_video_idx]
            signal = analyze_single_video(video_path, detector, current_video_idx + 1, classifier)
            print()
            if signal == "quit":
                break
            elif signal == "prev":
                current_video_idx = (current_video_idx - 1) % n
            else:  # "next"
                current_video_idx = (current_video_idx + 1) % n
        cv2.destroyAllWindows()
        return

    # 背景統計分析：跑完整批影片，輸出 CSV + 全部圖表分析。
    out_dir = OUTPUT_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"輸出目錄: {out_dir}")
    print(f"共 {n} 支影片待分析\n")

    video_results = {}
    for video_index, video_path in enumerate(video_paths, start=1):
        result = analyze_single_video_batch(video_path, detector, out_dir, video_index)
        if result is not None:
            records, behavior = result
            video_results[Path(video_path).stem] = (records, behavior, video_index)
        print()

    if not video_results:
        print("⚠ 所有影片都沒有產生有效資料，無法產生比較圖")
        return

    if len(video_results) >= 2:
        build_comparison_chart(video_results, out_dir)
    else:
        print("只有 1 支影片有有效資料，略過多影片比較圖（該影片自己的圖表已存在對應子資料夾）")
        # 單支影片時也把它的圖秀出來，維持互動檢視體驗
        only_name = next(iter(video_results))
        img = plt.imread(str(out_dir / only_name / "bone_stability_chart.png"))
        fig = plt.figure(figsize=(12, 15))
        plt.imshow(img)
        plt.axis("off")
        show_fig_fit_to_content(fig)

    # 門檻分析圖不需要多支影片才有意義（單支影片也能看自己的 ECDF／flag rate），
    # 所以獨立於上面的比較圖分支之外，只要有資料就跑。

    # torso_ratio 不靠關鍵點自己互相參照（用 bbox 當外部參照），方向是
    # 數值越小越可疑，單獨存一張圖。
    build_metric_threshold_chart(
        video_results, out_dir, "torso_ratio", "torso ratio (chest-hip / bbox diagonal)",
        CANDIDATE_TORSO_RATIO_THRESHOLD, "torso_ratio_threshold_chart.png", show=False, direction="below",
    )
    # torso_ratio 的「上界」校準圖：跟上面用同一份 torso_ratio 資料，只是換
    # 個門檻方向分析（不需要另外收集資料）。抓的是 chest 或 hip 其中一點
    # 整個窗口偵測不到、fallback 到畫面原點座標，導致量出來的 chest-hip
    # 距離被撐得比 bbox 本身還誇張——這是尺度正規化失效的另一種樣態，跟
    # 上面「軀幹相對 bbox 塌陷」的失效方向相反，下界門檻抓不到。
    build_metric_threshold_chart(
        video_results, out_dir, "torso_ratio", "torso ratio (chest-hip / bbox diagonal) - inflated check",
        CANDIDATE_TORSO_RATIO_UPPER_THRESHOLD, "torso_ratio_inflated_threshold_chart.png", show=False, direction="above",
    )
    # midback_angle_jitter：看 Chest-MidBack-Hip 夾角「連續幀之間」變化了幾
    # 度，不看絕對角度大小（貓咪拱背幅度本來就沒有放諸四海皆準的正常值，用
    # 絕對門檻容易誤判）。方向是數值越大越可疑，單獨存一張圖。
    build_metric_threshold_chart(
        video_results, out_dir, "midback_angle_jitter", "midback angle jitter (deg/frame)",
        CANDIDATE_MIDBACK_ANGLE_JITTER_THRESHOLD, "midback_angle_jitter_threshold_chart.png", show=False, direction="above",
    )
    # midback_offset_ratio：MidBack 偏離 Chest-Hip 虛擬中點的距離 ÷ Chest-Hip
    # 距離本身，門檻是粗略的解剖合理性上限（偏移量不該超過底邊自身長度），
    # 不是精確統計出來的基準值。方向是數值越大越可疑，單獨存一張圖。
    build_metric_threshold_chart(
        video_results, out_dir, "midback_offset_ratio", "midback offset ratio",
        CANDIDATE_MIDBACK_OFFSET_THRESHOLD, "midback_offset_threshold_chart.png", show=False, direction="above",
    )
    # torso_ratio_jitter（Skeleton Scale Jitter）：torso_ratio 逐幀值在相鄰
    # 幀之間的變化量，抓「整體骨架尺度忽大忽小」，跟 torso_ratio 絕對值本身
    # 是互補的兩件事。方向是數值越大越可疑，單獨存一張圖。
    build_metric_threshold_chart(
        video_results, out_dir, "torso_ratio_jitter", "torso ratio jitter (/frame)",
        CANDIDATE_TORSO_RATIO_JITTER_THRESHOLD, "torso_ratio_jitter_threshold_chart.png", show=False, direction="above",
    )
    # bone_length_oscillation：各骨段長度（正規化座標下）相鄰幀變化量的最大
    # 值，抓「反覆縮放/跳動」而非平均長度，跟骨段長度離散度（CV）是不同
    # 統計量。方向是數值越大越可疑，單獨存一張圖。
    build_metric_threshold_chart(
        video_results, out_dir, "bone_length_oscillation", "bone length oscillation (/frame)",
        CANDIDATE_BONE_OSCILLATION_THRESHOLD, "bone_length_oscillation_threshold_chart.png", show=False, direction="above",
    )


if __name__ == "__main__":
    main()
