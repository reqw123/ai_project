# ============================================================
#  骨架資料集收集與手動標注工具
# ============================================================
#  七種執行模式（對應檔尾 __main__ 選單，數字為輸入代號）：
#
#  模式  函式                              作用
#  ────  ────────────────────────────────  ──────────────────────────────
#  1     process_all_videos()              批次推論 VIDEO_FOLDERS（模型專用/<split>/<類別>/）
#                                           影片 (YOLO-Pose)，增量、跳過已有
#                                           JSON，frame label 先填資料夾名稱；
#                                           新檔尚未標記，需到模式 2 勾選整段
#                                           或逐段標記後才能訓練
#
#  2     manual_action_labeling()          連續手動標記 OUTPUT_FOLDER 內多個
#                                           skeleton JSON；預設只列尚未確認，
#                                           支援類別/檔名篩選、分頁與批次勾選
#                                           整段有效，也可逐段標記事件區間
#
#  3     reextract_preserve_labels()       重新推論骨架（換新模型後用），依
#                                           timestamp 對應還原既有
#                                           action_intervals 與 frame label，
#                                           不受幀數變動影響
#
#  4     check_discarded_files()           純讀取報告：檢查有哪些影片／幀／
#                                           訓練視窗被捨棄或過濾、未進入訓練
#                                           資料，不修改任何檔案
#
#  5     reextract_preserve_labels_        同模式 3，但只針對指定的部分檔
#        selected()                        案，不用重跑整個資料集
#
#  6     label_test_set()                  【測試集專用】進來後再選一次要
#                                           做哪個步驟：1.只抽骨架（增量提
#                                           取 TEST_VIDEO_FOLDERS 到獨立的
#                                           TEST_OUTPUT_FOLDER）2.只標註
#                                           （骨架需已存在，直接進入標註
#                                           迴圈，沿用模式 2 介面）3.兩者都
#                                           做。用途：scratch/shake 這類事
#                                           件占比低的類別，測試集也需要逐
#                                           幀標註才能算出跟訓練同一套「乾
#                                           淨事件視窗」準確率，不能只用資
#                                           料夾名稱當整支影片的標籤。不論
#                                           選哪個步驟，結束後都自動印出
#                                           review_test_labels() 報告
#
#  7     review_test_labels()              純讀取 TEST_OUTPUT_FOLDER 目前的
#                                           標註結果，不開影片視窗，隨時可
#                                           執行。終端印出依行為分組的表格
#                                           （區間數/事件幀數/事件佔比/每個
#                                           已標區間的起訖影片時間，精確到
#                                           小數1位），同步輸出 CSV
#                                           (_test_label_report.csv)——這是
#                                           「怎麼知道被打上什麼標籤」的主
#                                           要管道
#
#  模式 6/7 用的 TEST_VIDEO_FOLDERS/TEST_OUTPUT_FOLDER 跟模式 1-5 用的
#  VIDEO_FOLDERS/OUTPUT_FOLDER 是完全獨立的路徑（見下方 Configuration），
#  標測試集不會誤觸或混進訓練資料。
# ============================================================
#  frame label 三種狀態：
#
#  狀態  說明                                    frame label        需處理
#  ────  ──────────────────────────────────────  ─────────────────  ──────
#  1     批次提取（process_all_videos），          全段 = 資料夾名稱   是（尚未標記，
#        從未開啟標注模式                          (walk/lick/…)       訓練會被擋下）
#
#  2     開啟標注模式，有標記至少一個區間           區間內 = 行為標籤   否
#                                                其餘幀 = unannotated
#                                                （訓練時自動過濾）
#
#  3     開啟標注模式，未標任何區間直接儲存         保留原有 label      否
#        （保護機制：action_intervals 為空時       不覆寫              ）
#        不覆寫 frame label）
# ============================================================

# 2026-09-24 起每筆骨架都必須有標記區段（片段或整段）；判斷規則在
# utils/skeleton_splits.py 的 annotation_state()，0_train_gcn.py 遇到尚未標記的檔案會拒絕訓練。
# 人工確認紀錄：annotation_review；舊檔有 action_intervals 即視為已標記。
# 批次整段確認寫入 [0, len(frames)-1] 區間及每幀 label；手動清空記為 manual_empty。
# 模式 2：u 尚未確認 / a 清除篩選並展開全部 / c 類別 / /關鍵字 / b 批次勾選 / n,p 翻頁。

import os
import re
import json
import time
import cv2
import numpy as np
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

_BEHAVIOR_RE    = re.compile(r"(walk|lick|scratch|shake|stop)", re.IGNORECASE)
_BEHAVIOR_ORDER = ['walk', 'lick', 'scratch', 'shake', 'stop']

def _parse_behavior(name: str):
    """從檔名抽取行為關鍵字，找不到回傳 None。"""
    m = _BEHAVIOR_RE.search(name)
    return m.group(1).lower() if m else None


def _natural_sort_key(path):
    """依檔名做自然排序（數字部分視為整數比較，例如 walk_2 排在 walk_10 之前）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path.name.lower())]

# ==================== Configuration ====================
# ==================== Configuration ====================
# VIDEO_FOLDERS（影片來源）：模型專用/<split>/<類別>/，跟 skeletons/ 同一套 train/val/test
# 切分、影片的 split 永遠跟它的骨架一致（gcn_dataset_manager 搬骨架時影片跟著搬）。
# 清單由 skeleton_splits.video_class_folders() 列出所有存在的類別資料夾，在下方 import
# 之後設定；類別＝影片所在資料夾名稱。新影片放進哪個 模型專用/<split>/<類別>/，骨架就放同一個 split。
# 骨架資料集根目錄；底下分 train/ val/ test/，每個再分類別資料夾（見 cat_monitoring_system/utils/
# skeleton_splits.py）。新抽的骨架寫進影片所在的 split，已存在的檔案重抽時寫回原本所在的子資料夾，
# 切分調整用 tools/gcn_dataset_manager.py（模式 2）。
OUTPUT_FOLDER = r"C:\ai_project\paper\skeletons/"
MODEL_PATH = str(Path(__file__).resolve().parents[3] / "yolo_models" / "v11s_152.pt")  # You can use yolov8s-pose.pt, yolov8m-pose.pt for better accuracy

# 若設定 YOLO_MODEL_PATH 環境變數，優先使用該模型路徑（覆蓋上面寫死的 MODEL_PATH，對應
# settings_window.py 的「🧠 模型路徑」欄位）；模式 3「換新模型後重新推論骨架」正好用得上。
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()
if _env_yolo_model:
    MODEL_PATH = _env_yolo_model

TARGET_FPS = 30
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # config.py 在 paper/ 根目錄
from config import YOLOConfig as _YOLOConfig
from cat_monitoring_system.utils.skeleton_splits import (
    SPLITS, iter_skeleton_files, skeleton_path_for, find_skeleton, split_of,
    annotation_state as _annotation_state, find_unmarked_skeletons, format_unmarked,
    video_class_folders, split_of_video,
)
from cat_monitoring_system.utils.console_alert import alert_box as _alert_box
VIDEO_FOLDERS = video_class_folders()   # 見上方 Configuration 的說明
IMGSZ = _YOLOConfig.IMAGE_SIZE  # 跟主系統同步（設定視窗 yolo.image_size／環境變數 CAT_MONITORING_YOLO_IMAGE_SIZE，預設 640）
CONF_THRESHOLD = 0.5
KP_CONF_THRESHOLD = 0.5

# 永久排除清單：列出不想再被模式 1 重新提取的影片檔名（不含副檔名）
# 範例：EXCLUDED_STEMS = {"lick_bad_001", "walk_noise_003"}
EXCLUDED_STEMS: set = set()

# ==================== 獨立測試集標註（模式 6/7）====================
# 2026-08-11：eval_gcn_compare.py 等腳本算「獨立測試集準確率」時，是用
# 資料夾名稱當整支影片的 ground truth（沒有逐幀標籤）——但 scratch/shake
# 這種事件式行為在影片裡只占一小段（訓練資料實測：scratch 平均 20.0%、
# shake 平均 10.9%，其餘時間貓在走動/靜止），導致準確率被結構性稀釋，
# 不是模型真的分不清。這裡幫「主要測試」資料夾也補上跟訓練資料同一套
# 逐幀標註（action_intervals），才能算出跟訓練同一套「乾淨事件視窗」
# 準確率。故意跟上面 VIDEO_FOLDERS/OUTPUT_FOLDER 完全分開路徑，避免標
# 測試集時不小心覆寫或混進訓練用的 skeletons/ 資料夾。
# walk/stop 整段都是目標行為（不需要標區間）、lick 事件占比 41.6% 已經
# 100% 準確率，這三類可以不標；這裡預設仍列出全部 5 類資料夾，方便之後
# 有需要時也能補標，不標的類別直接跳過即可（模式 6 不會強迫每類都要標）。
TEST_VIDEO_FOLDERS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\stop",
]
TEST_OUTPUT_FOLDER = r"C:\ai_project\paper\skeletons_test_labeled/"

# ==================== Main Processing Function ====================

# 恢復批次推論多資料夾影片功能
def process_all_videos():
    """
    批次提取骨架（增量模式）。
    每次執行前先比對 OUTPUT_FOLDER 內現有 JSON，已存在的影片直接跳過，
    不清空資料夾。顯示各類別現有資料量後等待確認才載入模型。
    """
    print("="*60)
    print("Skeleton Extraction Pipeline (Batch)")
    print("="*60)
    setup_directories()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    # ── 步驟 1：收集所有影片，與 OUTPUT_FOLDER 現有 JSON 比對檔名 ─────────────
    existing_stems = {p.stem for p in iter_skeleton_files(OUTPUT_FOLDER)}

    video_files = []
    for folder in VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            print(f"[Warning] Video folder not found: {folder}")
            continue
        found = sorted(
            (f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions),
            key=_natural_sort_key,
        )
        video_files.extend(found)

    if not video_files:
        print(f"✗ No video files found in any of the specified folders.")
        print(f"  Supported formats: {', '.join(video_extensions)}")
        return

    skip_list     = [v for v in video_files if v.stem in existing_stems]
    excluded_list = [v for v in video_files if v.stem in EXCLUDED_STEMS]
    todo_list     = [v for v in video_files
                     if v.stem not in existing_stems and v.stem not in EXCLUDED_STEMS]

    print(f"\n影片總數：{len(video_files)}   已存在（跳過）：{len(skip_list)}"
          f"   永久排除：{len(excluded_list)}   待提取：{len(todo_list)}")
    if excluded_list:
        print("  [Excluded]", "  ".join(v.name for v in excluded_list))
    if skip_list:
        print("  [Skip]", "  ".join(v.name for v in skip_list))
    if todo_list:
        print("  [Todo]（骨架放到影片所在的 split）")
        for v in todo_list:
            print(f"    {v.name}  -> skeletons/{split_of_video(v) or 'train'}/")

    # ── 步驟 2：統計現有 skeleton 資料夾各類別數量 ────────────────────────────
    from collections import Counter
    existing_jsons = iter_skeleton_files(OUTPUT_FOLDER)
    class_counts: Counter = Counter()
    for jp in existing_jsons:
        behavior = _parse_behavior(jp.stem) or 'unknown'
        class_counts[behavior] += 1

    print(f"\n現有 skeleton 各類別（共 {len(existing_jsons)} 筆）：")
    sep = "─" * 40
    print(sep)
    for cls in _BEHAVIOR_ORDER + ['unknown']:
        if cls in class_counts:
            bar = '█' * class_counts[cls]
            print(f"  {cls:<10} {class_counts[cls]:>4} 筆  {bar}")
    print(sep)

    if not todo_list:
        print("\n✓ 所有影片皆已提取，無需重新處理。")
        return

    # ── 步驟 3：確認後才載入模型 ─────────────────────────────────────────────
    confirm = input('\n確認後輸入 "ok" 開始提取（其他任意鍵取消）：').strip().lower()
    if confirm != "ok":
        print("✗ 已取消。")
        return

    print()
    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    results_summary = []
    for idx, video_path in enumerate(todo_list, 1):
        print(f"[{idx}/{len(todo_list)}] Processing: {video_path.name}")
        video_id    = video_path.stem
        output_path = skeleton_path_for(OUTPUT_FOLDER, video_id, split=split_of_video(video_path))

        # 從資料夾名稱取得行為標籤（walk/lick/scratch/shake/stop）
        label = video_path.parent.name.lower()
        result = extract_skeleton_from_video(
            video_path,
            pose_extractor,
            target_fps=TARGET_FPS,
            label=label
        )
        if result is None:
            print(f"  ✗ Failed to process video")
            continue
        skeleton_data, actual_fps = result
        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "actual_fps": actual_fps,   # 記錄實際來源 FPS（補償前）
            "fps_compensated": True,    # frames 已經 resample_to_target_fps 補償到等距 target_fps 網格
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD
        }
        save_skeleton_data(skeleton_data, output_path, video_metadata)
        detected_frames = sum(1 for f in skeleton_data if f['detected'])
        detection_rate = (detected_frames / len(skeleton_data) * 100) if skeleton_data else 0
        results_summary.append({
            "video_id": video_id,
            "total_frames": len(skeleton_data),
            "detected_frames": detected_frames,
            "detection_rate": detection_rate
        })
        print()
    print("="*60)
    print("Processing Summary")
    print("="*60)
    for result in results_summary:
        print(f"Video: {result['video_id']}")
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Detected frames: {result['detected_frames']}")
        print(f"  Detection rate: {result['detection_rate']:.1f}%")
        print()
    print(f"✓ All done! Skeleton data saved to: {OUTPUT_FOLDER}")


def extract_test_set_skeletons():
    """
    批次提取「主要測試」資料夾（TEST_VIDEO_FOLDERS）的骨架，輸出到獨立的
    TEST_OUTPUT_FOLDER——跟 process_all_videos() 邏輯相同（增量、跳過已有
    JSON、依資料夾名稱給初始整段標籤），只是刻意不共用 VIDEO_FOLDERS/
    OUTPUT_FOLDER 這兩個全域變數，避免誤觸訓練資料。
    """
    print("="*60)
    print("Skeleton Extraction Pipeline — 獨立測試集（TEST_OUTPUT_FOLDER）")
    print("="*60)
    Path(TEST_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {TEST_OUTPUT_FOLDER}")

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    existing_stems = {p.stem for p in Path(TEST_OUTPUT_FOLDER).glob("*.json")}

    video_files = []
    for folder in TEST_VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            print(f"[Warning] Video folder not found: {folder}")
            continue
        video_files.extend(sorted(
            (f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions),
            key=_natural_sort_key,
        ))

    if not video_files:
        print("✗ No video files found in any of TEST_VIDEO_FOLDERS.")
        return

    todo_list = [v for v in video_files if v.stem not in existing_stems]
    print(f"\n影片總數：{len(video_files)}   已存在（跳過）：{len(video_files) - len(todo_list)}"
          f"   待提取：{len(todo_list)}")
    if not todo_list:
        print("\n✓ 所有測試影片皆已提取，無需重新處理。")
        return
    for v in todo_list:
        print(f"    {v.name}")

    confirm = input('\n確認後輸入 "ok" 開始提取（其他任意鍵取消）：').strip().lower()
    if confirm != "ok":
        print("✗ 已取消。")
        return

    print()
    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    for idx, video_path in enumerate(todo_list, 1):
        print(f"[{idx}/{len(todo_list)}] Processing: {video_path.name}")
        video_id    = video_path.stem
        output_path = Path(TEST_OUTPUT_FOLDER) / f"{video_id}.json"
        label = video_path.parent.name.lower()   # 初始整段標籤，之後模式 6 的標註步驟會覆蓋成事件區間
        result = extract_skeleton_from_video(
            video_path, pose_extractor, target_fps=TARGET_FPS, label=label
        )
        if result is None:
            print(f"  ✗ Failed to process video")
            continue
        skeleton_data, actual_fps = result
        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "actual_fps": actual_fps,
            "fps_compensated": True,
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD,
        }
        save_skeleton_data(skeleton_data, output_path, video_metadata)
        print()
    print(f"✓ 測試集骨架提取完成，共 {len(todo_list)} 支 → {TEST_OUTPUT_FOLDER}")


# ==================== Setup ====================
def setup_directories():
    """Create necessary directories if they do not exist"""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        (Path(OUTPUT_FOLDER) / sp).mkdir(exist_ok=True)
    print(f"✓ Output directory created: {OUTPUT_FOLDER}")


def clear_output_folder():
    """Clear all existing files/folders under OUTPUT_FOLDER."""
    setup_directories()
    for f in Path(OUTPUT_FOLDER).glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print(f"[Warning] Failed to delete {f}: {e}")
    print(f"✓ Cleared skeletons folder: {OUTPUT_FOLDER}")


# ==================== Video Processing ====================
def get_video_fps(cap):
    """Get the FPS of the video"""
    return cap.get(cv2.CAP_PROP_FPS)


def should_process_frame(frame_count, video_fps, target_fps):
    """
    Determine if a frame should be processed based on target FPS

    Args:
        frame_count: Current frame number
        video_fps: Original video FPS
        target_fps: Target FPS for extraction

    Returns:
        bool: True if frame should be processed
    """
    if video_fps <= target_fps:
        return True

    # Calculate frame interval (use max to prevent division by zero)
    interval = video_fps / target_fps
    interval_int = max(1, round(interval))  # Use round instead of int, ensure >= 1
    return frame_count % interval_int == 0


def resample_to_target_fps(skeleton_data, source_fps, target_fps, max_gap_frames=2):
    """
    以每幀真實 timestamp 為基準，把骨架序列重新取樣到嚴格等距的 target_fps 時間網格。

    背景問題：should_process_frame 只在來源 fps > target_fps 時用跳幀降採樣；
    來源 fps < target_fps 時完全不處理（保留全部原始幀）。這會導致同一個模型
    在不同來源幀率的影片上，「差一幀」代表的真實時間長短不一致，使 velocity/
    bone_motion 等逐幀差分特徵的物理尺度隨來源 fps 系統性偏移，汙染訓練資料。

    這裡統一用線性內插把整段影片重新取樣到 target_fps 網格（不論來源比目標
    快或慢皆適用），確保輸出序列裡每一幀間隔都精確等於 1/target_fps 秒。

    連續偵測不到貓的時間超過 max_gap_frames 個「來源取樣間隔」時，視為真實
    空窗，輸出幀標記為 detected=False、keypoints=[]，不會跨越空窗憑空內插出
    假的關鍵點。

    Args:
        skeleton_data: extract_skeleton_from_video 產生的原始逐幀 list[dict]
        source_fps: 來源影片實際 FPS
        target_fps: 目標取樣頻率（訓練時基）
        max_gap_frames: 允許跨越內插的最大空窗（以「來源取樣間隔」為單位）

    Returns:
        list[dict]: 重新取樣後、frame_id 從 0 開始且時間間隔均勻的骨架序列
    """
    if not skeleton_data:
        return skeleton_data

    n_joints = 17
    timestamps = np.array([f['timestamp'] for f in skeleton_data], dtype=np.float64)
    detected = np.array([bool(f.get('detected')) for f in skeleton_data])
    duration = float(timestamps[-1])
    n_out = max(1, int(round(duration * target_fps)) + 1)
    out_t = np.arange(n_out) / float(target_fps)

    def _carry_over(nearest_i):
        extra = {}
        if 'label' in skeleton_data[nearest_i]:
            extra['label'] = skeleton_data[nearest_i]['label']
        return extra

    det_idx = np.where(detected)[0]
    if len(det_idx) == 0:
        # 全片未偵測到任何貓，輸出等距但全空的骨架序列
        return [
            {
                "frame_id": k,
                "original_frame_id": skeleton_data[int(np.argmin(np.abs(timestamps - t)))].get('original_frame_id', 0),
                "timestamp": float(t),
                "detected": False,
                "keypoints": [],
                "bbox": None,
                "num_keypoints": 0,
                **_carry_over(int(np.argmin(np.abs(timestamps - t)))),
            }
            for k, t in enumerate(out_t)
        ]

    det_t = timestamps[det_idx]
    xs = np.zeros((len(det_idx), n_joints))
    ys = np.zeros((len(det_idx), n_joints))
    cs = np.zeros((len(det_idx), n_joints))
    for row, i in enumerate(det_idx):
        for kpt in skeleton_data[i].get('keypoints', []):
            j = kpt['joint_id']
            xs[row, j] = kpt['x']
            ys[row, j] = kpt['y']
            cs[row, j] = kpt['conf']

    native_dt = 1.0 / max(source_fps, 1e-6)
    max_gap = max_gap_frames * native_dt

    out = []
    for k, t in enumerate(out_t):
        nearest_i = int(np.argmin(np.abs(timestamps - t)))
        frame_out = {
            "frame_id": k,
            "original_frame_id": skeleton_data[nearest_i].get('original_frame_id', nearest_i),
            "timestamp": float(t),
            **_carry_over(nearest_i),
        }

        pos = int(np.searchsorted(det_t, t))
        lo = max(0, min(pos - 1, len(det_t) - 1))
        hi = max(0, min(pos, len(det_t) - 1))
        gap = det_t[hi] - det_t[lo]
        out_of_range = (t < det_t[0] - max_gap) or (t > det_t[-1] + max_gap)

        if out_of_range or (lo != hi and gap > max_gap):
            frame_out.update({"detected": False, "keypoints": [], "bbox": None, "num_keypoints": 0})
            out.append(frame_out)
            continue

        keypoints = [
            {
                "joint_id": j,
                "x": float(np.interp(t, det_t, xs[:, j])),
                "y": float(np.interp(t, det_t, ys[:, j])),
                "conf": float(np.interp(t, det_t, cs[:, j])),
            }
            for j in range(n_joints)
        ]
        nearer = lo if abs(t - det_t[lo]) <= abs(t - det_t[hi]) else hi
        frame_out.update({
            "detected": True,
            "keypoints": keypoints,
            "bbox": skeleton_data[det_idx[nearer]].get('bbox'),
            "num_keypoints": len(keypoints),
        })
        out.append(frame_out)

    return out


# ==================== YOLO-Pose Inference ====================
class PoseExtractor:
    """Wrapper class for YOLO-Pose inference"""
    
    def __init__(self, model_path, imgsz=640, conf_threshold=0.5):
        """
        Initialize the pose extractor
        
        Args:
            model_path: Path to YOLO-Pose model
            imgsz: Input image size
            conf_threshold: Confidence threshold for detection
        """
        print(f"Loading YOLO-Pose model from {model_path}...")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        
        # Try to use GPU if available
        self.use_half = False
        try:
            self.model.to("cuda")
            self.use_half = True
            print("✓ Model loaded on GPU")
        except:
            print("✓ Model loaded on CPU")

    def extract_keypoints(self, frame):
        """
        Extract keypoints from a single frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            dict: Dictionary containing keypoints and metadata
                  Returns None if no person/cat detected
        """
        # Run inference
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            quantize=16 if self.use_half else None,
            verbose=False
        )[0]
        
        # Check if keypoints are detected
        if results.keypoints is None or len(results.keypoints.xy) == 0:
            return None
        
        # Warn if multiple cats detected
        num_detected = len(results.keypoints.xy)
        if num_detected > 1:
            print(f"  ⚠ Warning: {num_detected} objects detected, using only the first one")
        
        # Get keypoints for the first detected object (assuming single cat in frame)
        keypoints_xy = results.keypoints.xy[0].cpu().numpy()  # Shape: (num_keypoints, 2)
        keypoints_conf = results.keypoints.conf[0].cpu().numpy()  # Shape: (num_keypoints,)
        
        # Get bounding box if available
        bbox = None
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0]
            bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
        
        # Format keypoints as list of dictionaries
        keypoints_list = []
        for i, (xy, conf) in enumerate(zip(keypoints_xy, keypoints_conf)):
            keypoints_list.append({
                "joint_id": i,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "conf": float(conf)
            })
        
        return {
            "keypoints": keypoints_list,
            "bbox": bbox,
            "num_keypoints": len(keypoints_list)
        }


# ==================== Video Processing Pipeline ====================
def extract_skeleton_from_video(video_path, pose_extractor, target_fps=30, label=None):
    """
    Extract skeleton sequence from a single video
    
    Args:
        video_path: Path to input video
        pose_extractor: PoseExtractor instance
        target_fps: Target FPS for skeleton extraction
    
    Returns:
        list: List of frame data dictionaries
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        return None
    
    # Get video properties
    video_fps = get_video_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Guard: 某些格式無法讀取 FPS（回傳 0），fallback 到 target_fps 避免 ZeroDivisionError
    if video_fps <= 0:
        print(f"  ⚠ Cannot read FPS from video, assuming {target_fps:.0f}fps")
        video_fps = float(target_fps)

    # 低 FPS 提示：實際的時基補償由下方 resample_to_target_fps() 處理，
    # 這裡僅提示來源幀率偏低，內插填補的比例會較高（非真實偵測，動作細節較粗略）
    if video_fps < 24:
        print(f"  ⚠ Source FPS={video_fps:.1f} < 24fps — 將以內插方式補償到 {target_fps:.0f}fps，"
              f"內插比例較高，動作細節解析度低於原生 {target_fps:.0f}fps 影片")

    print(f"  Source FPS: {video_fps:.2f} → target {target_fps}fps  |  Total frames: {total_frames}")
    print(f"  Extracting at {target_fps} FPS...")
    
    skeleton_data = []
    frame_count = 0
    processed_count = 0
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="  Processing", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we should process this frame based on target FPS
        if should_process_frame(frame_count, video_fps, target_fps):
            # Extract keypoints
            pose_data = pose_extractor.extract_keypoints(frame)
            
            # Store frame data
            frame_data = {
                "frame_id": processed_count,
                "original_frame_id": frame_count,
                "timestamp": frame_count / video_fps,
                "detected": pose_data is not None
            }
            if label is not None:
                frame_data["label"] = label

            if pose_data is not None:
                frame_data.update(pose_data)
            else:
                # No detection - store empty keypoints
                frame_data["keypoints"] = []
                frame_data["bbox"] = None
                frame_data["num_keypoints"] = 0
            
            skeleton_data.append(frame_data)
            processed_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()

    print(f"  ✓ Extracted {processed_count} raw frames with skeleton data")

    # 時基補償：不論來源 fps 比 target_fps 快或慢，統一重新取樣到等距的
    # target_fps 網格，確保 velocity/bone_motion 等逐幀差分特徵的物理時間
    # 尺度在整個資料集中一致（詳見 resample_to_target_fps 說明）
    skeleton_data = resample_to_target_fps(skeleton_data, video_fps, target_fps)
    print(f"  ✓ 時基補償後: {len(skeleton_data)} 幀 @ {target_fps}fps（均勻時間網格）")

    return skeleton_data, float(video_fps)


# ==================== Data Export ====================
def save_skeleton_data(skeleton_data, output_path, video_metadata):
    """
    Save skeleton data to JSON file
    
    Args:
        skeleton_data: List of frame dictionaries
        output_path: Path to output JSON file
        video_metadata: Metadata about the video
    """
    output_data = {
        "video_metadata": video_metadata,
        "frames": skeleton_data,
        "total_frames": len(skeleton_data)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to: {output_path}")


# ==================== Main Processing Function ====================

def process_single_video():
    """
    只處理單一影片，推論骨架並存成 skeleton json，供標註用。
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="選擇影片檔案", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")])
    if not video_path:
        print("✗ 未選擇影片")
        return

    setup_directories()
    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )
    video_id = Path(video_path).stem
    # 從資料夾名稱自動推斷標籤；若資料夾名非行為類別可事後手動修改 JSON
    label = Path(video_path).parent.name.lower()
    result = extract_skeleton_from_video(
        video_path,
        pose_extractor,
        target_fps=TARGET_FPS,
        label=label
    )
    if result is None:
        print(f"  ✗ Failed to process video")
        return
    skeleton_data, actual_fps = result
    video_metadata = {
        "video_id": video_id,
        "video_filename": Path(video_path).name,
        "video_path": str(video_path),
        "target_fps": TARGET_FPS,
        "actual_fps": actual_fps,
        "fps_compensated": True,
        "model_used": MODEL_PATH,
        "imgsz": IMGSZ,
        "conf_threshold": CONF_THRESHOLD,
        "kp_conf_threshold": KP_CONF_THRESHOLD
    }
    output_path = skeleton_path_for(OUTPUT_FOLDER, video_id, split=split_of_video(video_path))
    save_skeleton_data(skeleton_data, output_path, video_metadata)
    print(f"\n✓ Skeleton JSON 已儲存: {output_path}\n可直接用於手動標註模式。\n")


# ==================== CSV Export (Alternative Format) ====================
def save_skeleton_data_csv(skeleton_data, output_path):
    """
    Alternative function to save skeleton data in CSV format
    
    Args:
        skeleton_data: List of frame dictionaries
        output_path: Path to output CSV file
    """
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Determine number of keypoints (assuming consistent across frames)
        num_kpts = 0
        for frame in skeleton_data:
            if frame['detected'] and len(frame['keypoints']) > 0:
                num_kpts = len(frame['keypoints'])
                break
        
        # Create header
        header = ['frame_id', 'original_frame_id', 'timestamp', 'detected']
        for i in range(num_kpts):
            header.extend([f'joint{i}_x', f'joint{i}_y', f'joint{i}_conf'])
        
        writer = csv.writer(f)
        writer.writerow(header)
        
        # Write data
        for frame in skeleton_data:
            row = [
                frame['frame_id'],
                frame['original_frame_id'],
                frame['timestamp'],
                int(frame['detected'])
            ]
            
            # Add keypoint data
            if frame['detected']:
                for kpt in frame['keypoints']:
                    row.extend([kpt['x'], kpt['y'], kpt['conf']])
            else:
                # Fill with zeros if not detected
                row.extend([0.0] * (num_kpts * 3))
            
            writer.writerow(row)
    
    print(f"  ✓ CSV saved to: {output_path}")


# ==================== Main Entry Point ====================

# ==================== Manual Action Labeling ====================

# 骨架連線（與 cat_pose 腳本保持一致）
_ANNOT_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]
_ANNOT_EDGE_COLORS = [
    (255,120,60),(255,120,60),(255,120,60),
    (220,220,60),(200,220,60),(160,220,60),
    (102,85,255),(102,85,255),(255,68,204),(255,68,204),
    (255,170,34),(255,170,34),(0,153,255),(0,153,255),
    (80,200,160),(60,170,130),(40,140,100),
]


# 狀態判斷共用 skeleton_splits.annotation_state()，訓練前檢查用的是同一套規則。
_ANNOTATION_STATUS_TEXT = {
    'pending': '尚未人工標記／確認',
    'full_clip': '已確認整段有效',
    'manual_intervals': '已手動標記區間',
    'manual_empty': '已人工清空（無有效區間）',
    'legacy_intervals': '已有區間標記（舊資料）',
}


def _review_record(method, action=None):
    result = {'status': 'reviewed', 'method': method,
              'reviewed_at': datetime.now().astimezone().isoformat(timespec='seconds')}
    if action is not None:
        result['action'] = action
    return result


def _atomic_write_json(path, data):
    """在同目錄寫完暫存檔才替換，避免中斷留下半份 JSON。"""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.stem + '.', suffix='.tmp',
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _whole_clip_action(data, path):
    """只有能確定單一行為時才允許批次整段確認；衝突資料改由人工檢查。"""
    frames = data.get('frames', [])
    if not isinstance(frames, list) or not frames or not all(isinstance(f, dict) for f in frames):
        return None, '沒有可標記的 frames'
    meta = data.get('video_metadata') or {}
    hints = [Path(path).parent.name.lower(), str(meta.get('label', '')).lower(),
             _parse_behavior(Path(path).stem)]
    actions = {a for a in hints if a in _BEHAVIOR_ORDER}
    frame_labels = {f.get('label') for f in frames} - {None, '', 'unannotated'}
    if not frame_labels and any(f.get('label') == 'unannotated' for f in frames):
        # 舊版清空標記後會留下全 unannotated 的檔案，可能是刻意清掉，不能批次救回。
        return None, '全部幀都是 unannotated（可能曾人工清空），請逐檔確認'
    if frame_labels - set(_BEHAVIOR_ORDER):
        return None, '含無法辨識的幀標籤'
    actions.update(frame_labels)
    if len(actions) != 1:
        return None, '類別不明或標籤衝突，請先逐檔檢查'
    return next(iter(actions)), ''


def _scan_annotation_files(folder):
    infos = []
    for path in sorted(iter_skeleton_files(folder), key=_natural_sort_key):
        info = {'path': path, 'behavior': _parse_behavior(path.stem) or 'unknown',
                'state': 'pending', 'n_intervals': 0, 'total_frames': 0,
                'action': None, 'error': '', 'batch_reason': ''}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            info['state'] = _annotation_state(data)
            info['n_intervals'] = len(data.get('action_intervals') or [])
            info['total_frames'] = len(data.get('frames', []))
            info['action'], info['batch_reason'] = _whole_clip_action(data, path)
            info['behavior'] = info['action'] or info['behavior']
        except Exception as exc:
            info['error'] = str(exc)
        infos.append(info)
    order = {b: i for i, b in enumerate(_BEHAVIOR_ORDER + ['unknown'])}
    return sorted(infos, key=lambda fi: (order[fi['behavior']], _natural_sort_key(fi['path'])))


def _filter_annotation_files(infos, state):
    keyword = state.get('keyword', '').casefold()
    # 「尚未標記」清單要把這次工作階段剛標好的檔案留在原位（前面顯示 ✓），
    # 否則每標一筆就從清單消失、後面的流水號全部往前移，很難對照。按 u / r 才會真的拿掉。
    sticky = state.get('sticky', set())
    return [fi for fi in infos
            if (not state.get('pending_only', True) or fi['state'] == 'pending'
                or str(fi['path']) in sticky)
            and (state.get('behavior', 'all') == 'all' or fi['behavior'] == state['behavior'])
            and keyword in fi['path'].name.casefold()]


def _mark_whole_clip(json_path, expected_action):
    """只處理仍未人工確認的檔案；每次寫入前重新讀取、檢查。"""
    path = Path(json_path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if _annotation_state(data) != 'pending':
        raise ValueError('已有人工標記／確認，已保留原檔')
    action, reason = _whole_clip_action(data, path)
    if not action or action != expected_action:
        raise ValueError(reason or '類別已改變，請重新載入清單')
    frames = data['frames']
    for frame in frames:
        frame['label'] = action
    data['total_frames'] = len(frames)
    data['action_intervals'] = [{'action': action, 'start': 0, 'end': len(frames) - 1}]
    data['annotation_review'] = _review_record('full_clip', action)
    _atomic_write_json(path, data)


def _parse_batch_numbers(raw, count):
    """終端備援選取；任何錯誤都拒絕整次輸入，避免部分選取造成誤會。"""
    if raw.lower() == 'all':
        return set(range(count))
    chosen = set()
    for token in raw.replace('，', ',').split(','):
        token = token.strip()
        if not re.fullmatch(r'\d+(?:-\d+)?', token):
            raise ValueError('請輸入編號、範圍（例如 1,3-5）或 all')
        ends = [int(x) for x in token.split('-')]
        start, end = min(ends), max(ends)
        if start < 1 or end > count:
            raise ValueError(f'編號必須介於 1～{count}')
        chosen.update(range(start - 1, end))
    return chosen


def _choose_whole_clips(candidates):
    """可捲動的批次勾選視窗；無 Tk 顯示環境時使用終端多選。"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        root = tk.Tk()
    except (ImportError, RuntimeError) as exc:
        print(f'  無法開啟勾選視窗，改用終端多選：{exc}')
        root = None
    except Exception as exc:
        # 包含無 DISPLAY / Windows Tcl 安裝不完整的 TclError。
        print(f'  無法開啟勾選視窗，改用終端多選：{exc}')
        root = None
    if root is None:
        for i, fi in enumerate(candidates, 1):
            print(f"  [ ] {i:3d}. {fi['path'].name} → {fi['action']}  [{split_of(fi['path'])}]")
        while True:
            raw = input('  勾選編號 1,3-5 / all 全選 / q 取消：').strip()
            if not raw or raw.lower() == 'q':
                return []
            try:
                picked = [candidates[i] for i in sorted(_parse_batch_numbers(raw, len(candidates)))]
                break
            except ValueError as exc:
                print(f'  {exc}')
        print('  本次將整段標記：')
        for fi in picked:
            print(f"    [x] {fi['path']} → {fi['action']}")
        ok = input(f'  共 {len(picked)} 筆；輸入 ok 寫入，其他輸入取消：').strip().lower()
        return picked if ok == 'ok' else []

    root.title('批次勾選：確認整段都是指定行為')
    root.geometry('940x600')
    root.minsize(660, 360)
    result = []
    variables = [tk.BooleanVar(value=False) for _ in candidates]
    selected_count = tk.StringVar()

    def update_count():
        selected_count.set(f'已勾選 {sum(v.get() for v in variables)} / {len(candidates)} 筆')

    def select_all(value):
        for var in variables:
            var.set(value)
        update_count()

    def confirm():
        picked = [fi for fi, var in zip(candidates, variables) if var.get()]
        if not picked:
            messagebox.showinfo('尚未選取', '請先勾選要確認整段有效的資料。', parent=root)
            return
        from collections import Counter
        counts = Counter(fi['action'] for fi in picked)
        summary = '、'.join(f'{a}: {n} 筆' for a, n in sorted(counts.items()))
        if messagebox.askyesno('確認批次標記',
                              f'將 {len(picked)} 筆資料的全部幀標為各自顯示的行為。\n'
                              f'{summary}\n\n寫入整段區間與人工確認紀錄，確定繼續？', parent=root):
            result.extend(picked)
            root.destroy()

    ttk.Label(root, text='只顯示目前篩選範圍內、尚未人工確認且類別明確的資料（含所有頁）。',
              padding=10).pack(anchor='w')
    toolbar = ttk.Frame(root, padding=(10, 0, 10, 8))
    toolbar.pack(fill='x')
    ttk.Button(toolbar, text='全選此清單', command=lambda: select_all(True)).pack(side='left')
    ttk.Button(toolbar, text='全部取消勾選', command=lambda: select_all(False)).pack(side='left', padx=8)
    ttk.Label(toolbar, textvariable=selected_count).pack(side='right')
    bottom = ttk.Frame(root, padding=10)
    bottom.pack(side='bottom', fill='x')
    ttk.Button(bottom, text='確認所選資料整段有效', command=confirm).pack(side='right')
    ttk.Button(bottom, text='取消', command=root.destroy).pack(side='right', padx=8)
    area = ttk.Frame(root)
    area.pack(fill='both', expand=True, padx=10)
    canvas = tk.Canvas(area, highlightthickness=0)
    scrollbar = ttk.Scrollbar(area, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)
    content = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=content, anchor='nw')
    content.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window_id, width=e.width))
    root.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units'))
    root.bind('<Button-4>', lambda e: canvas.yview_scroll(-1, 'units'))
    root.bind('<Button-5>', lambda e: canvas.yview_scroll(1, 'units'))
    for fi, var in zip(candidates, variables):
        text = f"{fi['path'].name}    → {fi['action']}    {fi['total_frames']} 幀    [{split_of(fi['path'])}]"
        ttk.Checkbutton(content, text=text, variable=var, command=update_count,
                        padding=(5, 6)).pack(fill='x', anchor='w')
    root.protocol('WM_DELETE_WINDOW', root.destroy)
    update_count()
    root.mainloop()
    return result


def _batch_confirm_whole_clips(visible):
    candidates = [fi for fi in visible if fi['state'] == 'pending' and not fi['error'] and fi['action']]
    if not candidates:
        print('  目前範圍沒有可批次確認的資料；類別不明或衝突的資料請逐檔標記。')
        return
    picked = _choose_whole_clips(candidates)
    if not picked:
        print('  已取消，未修改任何資料。')
        return
    succeeded = 0
    for fi in picked:
        try:
            _mark_whole_clip(fi['path'], fi['action'])
            succeeded += 1
            print(f"  ✓ {fi['path'].name}：整段 {fi['action']}")
        except Exception as exc:
            print(f"  ✗ {fi['path']}：未寫入（{exc}）")
    print(f'  批次完成：成功 {succeeded}，未寫入 {len(picked) - succeeded}。')


def _remember_queue(state, visible):
    """記下開檔當下的清單順序（已套用類別／搜尋／尚未標記篩選），標記視窗按 n 時照這個順序往下走。"""
    state['queue'] = [str(fi['path']) for fi in visible if not fi['error'] and fi['total_frames']]


def _next_in_queue(state, current):
    """目前清單裡 current 的下一筆；「尚未標記」清單會跳過這段期間已標好的檔案。沒有下一筆回傳 None。"""
    queue = state.get('queue') or []
    if current not in queue:
        return None
    for path in queue[queue.index(current) + 1:]:
        if not Path(path).exists():
            continue
        if state.get('pending_only', True):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    if _annotation_state(json.load(f)) != 'pending':
                        continue
            except Exception:
                continue
        return path
    return None


def _queue_position(state, json_path):
    """(第幾筆, 共幾筆)，顯示在標記視窗標題；不在清單裡回傳 (None, None)。"""
    queue = state.get('queue') or []
    if json_path in queue:
        return queue.index(json_path) + 1, len(queue)
    return None, None


def _list_json_files_menu(folder: str, last_annotated: str = None, menu_state=None):
    """預設只列未確認資料；狀態由呼叫端保留，標完一筆不會重設篩選。"""
    if not Path(folder).exists():
        print(f'[Error] 資料夾不存在: {folder}')
        return None
    state = menu_state if menu_state is not None else {}
    for key, value in [('pending_only', True), ('behavior', 'all'), ('keyword', ''),
                       ('page', 0), ('show_all', False), ('sticky', set())]:
        state.setdefault(key, value)
    page_size = 25
    infos = _scan_annotation_files(folder)
    while True:
        if not infos:
            print(f'[Error] 找不到任何 JSON 檔案: {folder}')
            return None
        # 記住出現過的尚未標記檔案；之後標好了也留在清單原位，流水號不變
        state['sticky'] |= {str(fi['path']) for fi in infos if fi['state'] == 'pending'}
        visible = _filter_annotation_files(infos, state)
        kept_done = sum(fi['state'] != 'pending' for fi in visible) if state['pending_only'] else 0
        done = sum(fi['state'] != 'pending' and not fi['error'] for fi in infos)
        errors = sum(bool(fi['error']) for fi in infos)
        pages = max(1, (len(visible) + page_size - 1) // page_size)
        state['page'] = min(max(0, state['page']), pages - 1)
        print(f"\n{'=' * 80}\n  標記清單：{folder}")
        print(f'  全部 {len(infos)} 筆｜已人工標記／確認 {done}｜尚未確認 {len(infos) - done - errors}｜讀取失敗 {errors}')
        summary = []
        for behavior in _BEHAVIOR_ORDER + ['unknown']:
            group = [fi for fi in infos if fi['behavior'] == behavior and not fi['error']]
            if group:
                summary.append(f"{behavior}: {sum(fi['state'] == 'pending' for fi in group)}")
        print('  各類尚未確認：' + '  '.join(summary))
        scope = '尚未標記／確認' if state['pending_only'] else '全部資料'
        display_mode = ('完整清單（不分頁）' if state['show_all']
                        else f"第 {state['page'] + 1}/{pages} 頁")
        print(f"  顯示：{scope}｜類別 {state['behavior']}｜搜尋 {state['keyword'] or '無'}"
              f"｜符合 {len(visible)} 筆｜{display_mode}")
        if kept_done:
            print(f"  （其中 {kept_done} 筆是這次剛標好的，先留在原位標 ✓；按 u 或 r 重新整理後移除）")
        start = 0 if state['show_all'] else state['page'] * page_size
        rows = visible if state['show_all'] else visible[start:start + page_size]
        for number, fi in enumerate(rows, start + 1):
            detail = ('讀取失敗：' + fi['error']) if fi['error'] else _ANNOTATION_STATUS_TEXT[fi['state']]
            if fi['state'] == 'pending' and fi['batch_reason'] and not fi['error']:
                detail += '；不可批次：' + fi['batch_reason']
            marker = ' ← 上次開啟' if last_annotated and str(fi['path']) == last_annotated else ''
            mark = '✗' if fi['error'] else ('·' if fi['state'] == 'pending' else '✓')
            print(f"  [{number:3d}] {mark} {fi['path'].name}  [{split_of(fi['path'])}]"
                  f"  {fi['total_frames']} 幀 / {fi['n_intervals']} 區間  {detail}{marker}")
        if not visible:
            print('  目前篩選沒有資料，可切換 a 全部資料、c 類別，或 / 清除搜尋。')
        print('  u 尚未標記清單（分頁，移除已標好的）｜a 完整清單（清除所有篩選、不分頁）｜c 類別篩選')
        print('  /關鍵字 搜尋（單獨 / 清除）｜b 批次勾選整段有效（目前篩選的所有頁）')
        print('  n 下一頁｜p 上一頁（恢復分頁）｜r 重新整理｜q 離開')
        print('  編號 開啟逐段標記｜Enter 下一筆尚未確認；尚未標記的資料會擋下訓練。')
        choice = input('  > ').strip()
        command = choice.lower()
        if command == 'q':
            return None
        if command == 'a':
            state.update(pending_only=False, behavior='all', keyword='', page=0, show_all=True)
            infos = _scan_annotation_files(folder)
        elif command == 'u':
            state.update(pending_only=True, page=0, show_all=False, sticky=set())
            infos = _scan_annotation_files(folder)
        elif command == 'c':
            behavior = input('  類別 walk/lick/scratch/shake/stop/unknown；Enter 或 all 全部：').strip().lower() or 'all'
            if behavior in _BEHAVIOR_ORDER + ['unknown', 'all']:
                state.update(behavior=behavior, page=0, show_all=False)
            else:
                print('  無此類別，保留原篩選。')
        elif choice.startswith('/'):
            state.update(keyword=choice[1:].strip(), page=0, show_all=False)
        elif command in ('n', 'p'):
            state['show_all'] = False
            state['page'] += 1 if command == 'n' else -1
        elif command == 'b':
            _batch_confirm_whole_clips(visible)
            infos = _scan_annotation_files(folder)
        elif command == 'r':
            state['sticky'] = set()
            infos = _scan_annotation_files(folder)
        elif choice == '':
            last_idx = next((i for i, fi in enumerate(visible) if str(fi['path']) == last_annotated), -1)
            for fi in visible[last_idx + 1:] + visible[:last_idx + 1]:
                if fi['state'] == 'pending' and not fi['error'] and fi['total_frames']:
                    _remember_queue(state, visible)
                    return str(fi['path'])
            print('  此範圍沒有可開啟的尚未確認資料；可切換清單或輸入 q 離開。')
        elif choice.isdigit() and 1 <= int(choice) <= len(visible):
            fi = visible[int(choice) - 1]
            if fi['error'] or not fi['total_frames']:
                print('  此檔無法讀取或沒有幀，請先檢查資料。')
            else:
                _remember_queue(state, visible)
                return str(fi['path'])
        else:
            print('  請輸入清單編號或上方指令。')


def _annotate_single_skeleton(json_path, file_index=None, total_files=None):
    """
    對單一 skeleton JSON 檔案進行手動標註並儲存。
    改善版：終端選檔、骨架連線繪製、標記狀態視覺、底部時間軸、HUD 面板。
    """
    import tkinter as tk
    from tkinter import filedialog

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames       = data['frames']
    total_frames = len(frames)

    # 載入既有標注區間（支援重新標注同一檔案）
    intervals = []
    for iv in data.get('action_intervals', []):
        intervals.append((iv['start'], iv['end'], iv['action']))

    # 影片路徑
    video_path = None
    if 'video_metadata' in data and 'video_path' in data['video_metadata']:
        video_path = data['video_metadata']['video_path']
    elif 'video_path' in data:
        video_path = data['video_path']
    if not video_path or not os.path.exists(video_path):
        print("✗ 找不到影片路徑，請手動選擇影片檔案（可直接關閉視窗僅標骨架）")
        root2 = tk.Tk()
        root2.withdraw()
        root2.attributes('-topmost', True)
        video_path = filedialog.askopenfilename(
            title="選擇對應影片檔案（可略過）",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        root2.destroy()
        if not video_path or not os.path.exists(video_path):
            video_path = None

    VALID_ACTIONS = ['walk', 'lick', 'scratch', 'shake', 'stop']
    ACTION_COLORS = {
        'walk':        (0,  200,  0),
        'scratch':     (0,  140, 255),
        'lick':        (0,  220, 220),
        'shake':       (60,  60, 220),
        'stop':        (0,  165, 255),
        'unannotated': (40,  40,  40),
    }
    # 從檔名關鍵字預設行為；比對失敗則 fallback 為 walk
    _default_action = _parse_behavior(Path(json_path).stem)
    current_action = _default_action if _default_action in VALID_ACTIONS else 'walk'
    if _default_action:
        print(f"  [檔名預設行為]  {current_action.upper()}  （可按 1-5 手動切換）")

    print("\n操作說明：")
    print("  1/2/3/4/5 切換行為  |  s 標記起點/終點  |  u 撤銷上一個區間")
    print("  a/d 前/後幀  |  [/] 調整步長  |  SPACE 播放/暫停  |  t 跳轉秒數")
    print("  畫面最下方時間軸可點擊/拖曳跳轉、滾輪逐幀微調（拖曳/懸停時正上方會顯示預覽幀號/時間）")
    print("  ESC 儲存並回到清單  |  n 儲存並跳到清單下一筆    [未標記片段訓練時自動捨棄]\n")

    marking           = False
    intervals_touched = False   # 本次工作階段是否曾新增或撤銷過區間（區分「真的沒動」vs「主動清空」）
    go_next           = False   # 按 n 離開：存檔後直接開清單下一筆，不回清單
    start_idx         = None
    cur_idx           = 0
    cap               = None
    playing           = False
    skip_n            = 1
    needs_seek        = True
    cap_next_orig_fid = -1
    last_advance_t    = 0.0
    render_needed     = True
    cached_frame_img  = None
    cached_frame_idx  = -1
    flash_msg         = ''
    flash_until       = 0.0   # time.time() + duration

    # ── 自繪時間軸的滑鼠互動狀態（見 draw_timeline() / _on_annotate_mouse()） ──
    # 跟 video_infer_save.py 最上方那條時間拉桿同一套設計：點擊/拖曳跳轉、拖曳中
    # 即時顯示預覽幀號/時間、滾輪逐幀微調，取代原本純唯讀的時間軸視覺。
    _seek_bar_rect   = None   # (x0, y0, w, h)：時間軸在 show_img 座標系裡的可點擊範圍
    _seek_dragging   = False  # 滑鼠左鍵正在時間軸上按住拖曳
    _seek_hover_frame = None  # 滑鼠懸停/拖曳對應到的幀號（None＝沒有懸停）

    MAX_DISP_W, MAX_DISP_H = 1280, 720
    frame_shape = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        ret, sample = cap.read()
        if ret:
            h, w = sample.shape[:2]
            if w > MAX_DISP_W or h > MAX_DISP_H:
                r = min(MAX_DISP_W / w, MAX_DISP_H / h)
                sample = cv2.resize(sample, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
            frame_shape = sample.shape
    if frame_shape is None:
        frame_shape = (540, 960, 3)

    play_delay_ms = 33
    if cap is not None and cap.isOpened():
        _fps = cap.get(cv2.CAP_PROP_FPS)
        if _fps > 0:
            play_delay_ms = max(16, min(int(1000 / _fps), 66))

    frame_interval = 1
    if len(frames) >= 2:
        _gaps = [frames[i+1].get('original_frame_id', i+1) - frames[i].get('original_frame_id', i)
                 for i in range(min(10, len(frames) - 1))]
        if _gaps:
            frame_interval = max(1, int(round(sum(_gaps) / len(_gaps))))

    # ── 輔助繪圖函式 ──────────────────────────────────────────────────────────

    def draw_skeleton_with_edges(img, keypoints, kp_scale=1.0):
        """彩色骨架連線 + 關節點，取代純點繪製。"""
        kd  = {kpt['joint_id']: kpt for kpt in keypoints}
        w   = img.shape[1]
        lw  = max(1, int(w / 480))
        ro   = max(3, int(w / 320))
        ri   = max(2, int(w / 426))
        for ei, (a, b) in enumerate(_ANNOT_EDGES):
            ka, kb = kd.get(a), kd.get(b)
            if ka and kb and ka['conf'] > 0.2 and kb['conf'] > 0.2:
                pa = (int(ka['x'] * kp_scale), int(ka['y'] * kp_scale))
                pb = (int(kb['x'] * kp_scale), int(kb['y'] * kp_scale))
                col = _ANNOT_EDGE_COLORS[ei] if ei < len(_ANNOT_EDGE_COLORS) else (160, 160, 160)
                cv2.line(img, pa, pb, col, lw, cv2.LINE_AA)
        for kpt in keypoints:
            if kpt['conf'] > 0.2:
                x, y = int(kpt['x'] * kp_scale), int(kpt['y'] * kp_scale)
                cv2.circle(img, (x, y), ro, (0, 0, 0),   -1, cv2.LINE_AA)
                cv2.circle(img, (x, y), ri, (0, 220, 60), -1, cv2.LINE_AA)

    SEEK_ACCENT = (60, 200, 255)   # 時間軸游標／把手顏色（BGR，暖黃橘，跟各行為色塊區分開）

    def _seek_frame_from_x(x, bar_x0, bar_w):
        if total_frames <= 1 or bar_w <= 0:
            return 0
        ratio = max(0.0, min(1.0, (x - bar_x0) / float(bar_w)))
        return int(round(ratio * (total_frames - 1)))

    def _seek_wheel_delta(flags):
        """從 cv2 滑鼠回呼的 flags 解出滾輪方向：flags 高 16 位是有號的滾動量。"""
        d = (flags >> 16) & 0xFFFF
        if d >= 0x8000:
            d -= 0x10000
        return d

    def _on_annotate_mouse(event, x, y, flags, param):
        """時間軸的點擊/拖曳/滾輪處理：跟 video_infer_save.py 的自繪拉桿同一套邏輯。
        點擊/拖曳/滾輪都會暫停播放（跟 a/d/t 等手動導覽鍵一致），並強制下一輪用
        needs_seek 真正 seek（不要用鄰近幀的 grab 捷徑）。"""
        nonlocal cur_idx, needs_seek, playing, render_needed
        nonlocal _seek_dragging, _seek_hover_frame
        if _seek_bar_rect is None:
            return
        bx, by, bw, bh = _seek_bar_rect
        inside = bx <= x <= bx + bw and by <= y <= by + bh

        if event == cv2.EVENT_LBUTTONDOWN:
            if inside:
                _seek_dragging = True
                playing = False
                cur_idx = _seek_frame_from_x(x, bx, bw)
                needs_seek = True
                render_needed = True
        elif event == cv2.EVENT_MOUSEMOVE:
            new_hover = _seek_frame_from_x(x, bx, bw) if inside else None
            if new_hover != _seek_hover_frame:
                _seek_hover_frame = new_hover
                render_needed = True
            if _seek_dragging:
                clamped_x = max(bx, min(x, bx + bw))
                new_idx = _seek_frame_from_x(clamped_x, bx, bw)
                if new_idx != cur_idx:
                    cur_idx = new_idx
                    needs_seek = True
                    render_needed = True
        elif event == cv2.EVENT_LBUTTONUP:
            _seek_dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            if inside:
                step = 1 if _seek_wheel_delta(flags) > 0 else -1
                new_idx = max(0, min(cur_idx + step, total_frames - 1))
                if new_idx != cur_idx:
                    cur_idx = new_idx
                    needs_seek = True
                    playing = False
                    render_needed = True

    def _timeline_geom(h):
        """時間軸高度／與視窗底邊的距離，draw_timeline() 與 draw_hud() 共用。
        軌道加高、底下留較大空白，避免拖曳時一不小心滑出視窗底邊。"""
        bh = max(26, int(h * 0.05))
        mb = max(14, int(h * 0.025))
        return bh, mb

    def draw_timeline(img, total_f, cur_i, ivs, act_cols, pending=None):
        """底部橫向時間軸：顯示各標注區間與當前位置；可點擊/拖曳/滾輪直接跳轉
        （見 _on_annotate_mouse()）。拖曳中或滑鼠懸停在時間軸上時，正上方會顯示
        「第幾幀／時間」預覽文字——這段畫在 draw_hud() 的半透明黑底之上，所以主
        迴圈裡呼叫順序要先 draw_hud() 再 draw_timeline()，順序反過來預覽字會被蓋掉。
        pending=(start_idx, act_col)：標記進行中（已按 s 開始、尚未結束），在軌道上
        以半透明色塊畫出「開始點 → 目前位置」這段尚未儲存的區間。"""
        nonlocal _seek_bar_rect
        h, w  = img.shape[:2]
        bh, mb = _timeline_geom(h)
        mx    = 6
        by    = h - bh - mb
        bx1, bx2 = mx, w - mx
        bw    = bx2 - bx1
        cv2.rectangle(img, (bx1, by), (bx2, by + bh), (55, 55, 55), -1)
        for s, e, act in ivs:
            x1 = bx1 + int(s / max(total_f, 1) * bw)
            x2 = bx1 + int((e + 1) / max(total_f, 1) * bw)
            col = act_cols.get(act, (80, 80, 80))
            cv2.rectangle(img, (x1, by + 1), (max(x1 + 1, x2), by + bh - 1), col, -1)
        if pending is not None:
            p_s, p_col = pending
            lo, hi = min(p_s, cur_i), max(p_s, cur_i)
            x1 = bx1 + int(lo / max(total_f, 1) * bw)
            x2 = bx1 + int((hi + 1) / max(total_f, 1) * bw)
            ov = img.copy()
            cv2.rectangle(ov, (x1, by + 1), (max(x1 + 1, x2), by + bh - 1), p_col, -1)
            cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
            sx = bx1 + int(p_s / max(total_f - 1, 1) * bw)
            cv2.line(img, (sx, by - 4), (sx, by + bh + 4), p_col, 3, cv2.LINE_AA)
        cx = bx1 + int(cur_i / max(total_f - 1, 1) * bw)
        knob_r = max(5, bh // 3)
        cv2.line(img, (cx, by - 3), (cx, by + bh + 3), SEEK_ACCENT, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, by + bh // 2), knob_r + 2, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, by + bh // 2), knob_r, SEEK_ACCENT, -1, cv2.LINE_AA)
        cv2.rectangle(img, (bx1, by), (bx2, by + bh), (120, 120, 120), 1, cv2.LINE_AA)

        # 可點擊範圍：上方多留一點；下方一路延伸到視窗底邊，往下拖不會失去拖曳
        click_pad = 8
        _seek_bar_rect = (bx1, by - click_pad, bw, h - (by - click_pad))

        preview_frame = _seek_hover_frame
        if preview_frame is None and _seek_dragging:
            preview_frame = cur_i
        if preview_frame is not None:
            # 畫成不透明的小色塊 tooltip（跟 draw_marking_indicator() 的 REC 徽章同一種手法），
            # 不管疊在 draw_hud() 的半透明黑底或 line2 說明文字上面，都能完全蓋過去、乾淨可讀，
            # 不會變成文字疊文字的花畫面。
            sc = max(0.5, w / 960.0)
            p_ts = frames[preview_frame].get('timestamp', 0) or 0
            label = f"{preview_frame + 1}/{total_f}  {p_ts:.2f}s"
            fs = 0.42 * sc
            th = max(1, int(round(sc)))
            (tw, tth), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            pad = max(3, int(4 * sc))
            box_w, box_h = tw + pad * 2, tth + baseline + pad * 2
            px = bx1 + int(round(bw * (0.0 if total_f <= 1 else preview_frame / float(total_f - 1))))
            box_x = max(0, min(w - box_w, px - box_w // 2))
            box_y = max(0, by - click_pad - box_h - 2)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1, cv2.LINE_AA)
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), SEEK_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(img, label, (box_x + pad, box_y + pad + tth),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)

    def draw_marking_frame(img, act_col, sc):
        """標記進行中：整個畫面的粗彩色外框（先畫，其他 HUD 疊在上面）。"""
        h, w = img.shape[:2]
        bw   = max(4, int(9 * sc))
        cv2.rectangle(img, (bw // 2, bw // 2), (w - bw // 2, h - bw // 2), act_col, bw)

    def draw_marking_indicator(img, act, act_col, dur_s, sc):
        """標記進行中的 REC 徽章＋大字持續時間。放在頂部資訊欄正下方、右上角，
        必須在 draw_top_bar() 之後呼叫，否則會被頂部欄的半透明黑底蓋暗。"""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad  = int(8 * sc)
        top  = max(22, int(27 * sc)) + int(8 * sc)   # 頂部資訊欄高度（同 draw_top_bar）+ 間距

        rec_txt = f"REC [{act.upper()}]"
        rec_fs, rec_th = 0.6 * sc, max(1, int(2 * sc))
        dur_txt = f"{dur_s:.2f}s"
        dur_fs, dur_th = 1.35 * sc, max(2, int(3 * sc))
        (rw, rh), _ = cv2.getTextSize(rec_txt, font, rec_fs, rec_th)
        (dw, dh), dbl = cv2.getTextSize(dur_txt, font, dur_fs, dur_th)
        dot_r = max(4, int(6 * sc))
        box_w = max(rw + dot_r * 2 + pad, dw) + pad * 2
        box_h = rh + dh + dbl + pad * 3
        x0 = w - box_w - int(12 * sc)
        y0 = top

        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
        cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), act_col, max(2, int(3 * sc)))
        # 第一行：紅點 + REC [ACT]
        cy = y0 + pad + rh // 2
        cv2.circle(img, (x0 + pad + dot_r, cy), dot_r, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.putText(img, rec_txt, (x0 + pad + dot_r * 2 + pad // 2, y0 + pad + rh),
                    font, rec_fs, act_col, rec_th, cv2.LINE_AA)
        # 第二行：大字持續時間（右對齊）
        cv2.putText(img, dur_txt, (x0 + box_w - pad - dw, y0 + pad * 2 + rh + dh),
                    font, dur_fs, (255, 255, 255), dur_th, cv2.LINE_AA)

    def draw_hud(img, line1, line2, sc):
        """底部半透明 HUD（在時間軸上方）。"""
        h, w  = img.shape[:2]
        _bh, _mb = _timeline_geom(h)
        tl_h  = _bh + _mb + 8   # 多留 8px 給時間軸上方的可點擊邊界
        hh    = max(46, int(58 * sc))
        hy    = h - tl_h - hh
        ov    = img.copy()
        cv2.rectangle(ov, (0, hy), (w, h - tl_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
        lh = int(hh * 0.46)
        cv2.putText(img, line1, (int(8 * sc), hy + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * sc, (210, 210, 210),
                    max(1, int(sc)), cv2.LINE_AA)
        cv2.putText(img, line2, (int(8 * sc), hy + lh * 2 - int(3 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47 * sc, (150, 150, 150),
                    max(1, int(sc)), cv2.LINE_AA)

    def draw_top_bar(img, text, sc):
        """頂部細資訊欄。"""
        bh = max(22, int(27 * sc))
        ov = img.copy()
        cv2.rectangle(ov, (0, 0), (img.shape[1], bh), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.62, img, 0.38, 0, img)
        cv2.putText(img, text, (int(8 * sc), bh - int(5 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47 * sc, (170, 210, 255),
                    max(1, int(sc)), cv2.LINE_AA)

    def draw_flash(img, msg, sc):
        """畫面中央短暫提示訊息。"""
        h, w = img.shape[:2]
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.85 * sc, max(1, int(2 * sc)))
        x, y = (w - tw) // 2, h // 2
        ov = img.copy()
        cv2.rectangle(ov, (x - 14, y - th - 14), (x + tw + 14, y + 14), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
        cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85 * sc, (80, 240, 80), max(1, int(2 * sc)), cv2.LINE_AA)

    # ── 視窗 ──────────────────────────────────────────────────────────────────
    file_name  = Path(json_path).name
    # 從 video_path 父資料夾推斷行為類別（比 frames[0]['label'] 穩定，不受標注狀態影響）
    _vp = data.get('video_metadata', {}).get('video_path', '')
    _folder = Path(_vp).parent.name.lower() if _vp else ''
    file_label = _folder if _folder in ('walk', 'lick', 'scratch', 'shake', 'stop') \
                 else (frames[0].get('label', '') if frames else '')
    ctx_prefix = f"[{file_index}/{total_files}] " if file_index and total_files else ""
    win_name   = "Annotation"   # 固定名稱，確保每次重用同一個視窗
    # AUTOSIZE（原本是 KEEPRATIO，可手動拖曳縮放視窗）：時間軸改成可點擊/拖曳跳轉後，
    # 滑鼠座標需要準確對應到 show_img 的實際像素——AUTOSIZE 下視窗永遠貼合 imshow 畫面，
    # 座標保證 1:1，不用另外處理「使用者手動縮放視窗後座標怎麼換算」這個不確定性
    # （KEEPRATIO 下 cv2 的滑鼠座標換算在不同版本/後端行為不一致，換成 AUTOSIZE 才穩）。
    # 代價是不能再手動拖曳縮放這個視窗，畫面大小固定為 MAX_DISP_W x MAX_DISP_H 等比縮小後的尺寸。
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, _on_annotate_mouse)

    # ── 主迴圈 ────────────────────────────────────────────────────────────────
    while True:
        frame_data = frames[cur_idx]

        if cur_idx != cached_frame_idx:
            show_base = None
            kp_scale  = 1.0
            if cap is not None and cap.isOpened():
                target_orig_fid = frame_data.get('original_frame_id', cur_idx)
                gap = target_orig_fid - cap_next_orig_fid
                if needs_seek or gap < 0 or gap > skip_n * frame_interval + 30:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_orig_fid)
                    cap_next_orig_fid = target_orig_fid
                    needs_seek = False
                elif gap > 0:
                    for _ in range(gap):
                        cap.grab()
                    cap_next_orig_fid = target_orig_fid
                ret, img = cap.read()
                cap_next_orig_fid += 1
                if ret:
                    h, w = img.shape[:2]
                    if w > MAX_DISP_W or h > MAX_DISP_H:
                        kp_scale = min(MAX_DISP_W / w, MAX_DISP_H / h)
                        img = cv2.resize(img, (int(w * kp_scale), int(h * kp_scale)),
                                         interpolation=cv2.INTER_LINEAR)
                    show_base = img
            if show_base is None:
                h, w = frame_shape[:2]
                show_base = np.ones((h, w, 3), dtype=np.uint8) * 30
            if frame_data['detected'] and frame_data.get('keypoints'):
                draw_skeleton_with_edges(show_base, frame_data['keypoints'], kp_scale)
            cached_frame_img = show_base
            cached_frame_idx = cur_idx
            render_needed    = True

        if render_needed or time.time() < flash_until:
            show_img  = cached_frame_img.copy()
            sc        = max(0.5, show_img.shape[1] / 960.0)
            timestamp = frame_data.get('timestamp')
            orig_fid  = frame_data.get('original_frame_id')
            n_kpts    = len(frame_data.get('keypoints', []))
            t_str     = f"{timestamp:.2f}s" if timestamp is not None else '--'
            play_str  = '[PLAY]' if playing else '[PAUSE]'

            # 行為在某區間內時高亮現有標注
            for s, e, act in intervals:
                if s <= cur_idx <= e:
                    col = ACTION_COLORS.get(act, (80, 80, 80))
                    cv2.rectangle(show_img, (2, 2),
                                  (show_img.shape[1] - 2, show_img.shape[0] - 2), col, 2)

            # 標記中：粗外框（REC 徽章＋大字 dur 在 draw_top_bar() 之後才畫，避免被蓋暗）
            if marking:
                s_ts  = frames[start_idx].get('timestamp', 0) or 0
                c_ts  = frame_data.get('timestamp', 0) or 0
                dur_s = abs(c_ts - s_ts)
                act_col = ACTION_COLORS.get(current_action, (200, 200, 200))
                draw_marking_frame(show_img, act_col, sc)
                line2 = (f"  from frame {start_idx+1} ({s_ts:.1f}s)  ->  now ({c_ts:.1f}s)"
                         f"  dur={dur_s:.2f}s  |  s=END MARK  u=cancel  ESC=SAVE  n=SAVE+NEXT")
            else:
                act_col = ACTION_COLORS.get(current_action, (200, 200, 200))
                line2 = (f"  1=walk 2=lick 3=scratch 4=shake 5=stop  |  "
                         f"s=START MARK  u=UNDO  a/d=nav  [/]=skip  t=jump  SPACE  ESC=SAVE  n=NEXT")

            line1 = (f"{play_str}  Frame {cur_idx+1}/{total_frames}  ({t_str})"
                     f"  skip:{skip_n}  |  Intervals:{len(intervals)}"
                     + (f"  VidFr:{orig_fid}" if orig_fid is not None else "")
                     + ("" if frame_data.get('detected') else "  [NO DETECT]"))

            # 順序刻意固定：draw_hud() 先畫半透明黑底，draw_timeline() 後畫（含拖曳/懸停
            # 預覽文字），這樣預覽文字才會疊在 HUD 黑底之上，不會被蓋掉（見 draw_timeline() 說明）。
            draw_hud(show_img, line1, line2, sc)
            draw_timeline(show_img, total_frames, cur_idx, intervals, ACTION_COLORS,
                          pending=(start_idx, act_col) if marking else None)

            # 頂部欄
            lbl_tag = f" [{file_label.upper()}]" if file_label else ''
            top_text = (f"{ctx_prefix}{file_name}{lbl_tag}"
                        f"  |  {len(intervals)} interval(s)"
                        + (f"  kpts:{n_kpts}" if frame_data.get('detected') else "  [NO DETECT]"))
            draw_top_bar(show_img, top_text, sc)
            if marking:
                draw_marking_indicator(show_img, current_action, act_col, dur_s, sc)

            # 當前行為色塊（左下角小標籤）
            h_img = show_img.shape[0]
            _bh, _mb = _timeline_geom(h_img)
            tl_h  = _bh + _mb + 8   # 同 draw_hud()
            hh    = max(46, int(58 * sc))
            lbl_y = h_img - tl_h - hh - int(6 * sc)
            cv2.putText(show_img, f'[{current_action.upper()}]',
                        (int(8 * sc), lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72 * sc, act_col,
                        max(1, int(2 * sc)), cv2.LINE_AA)

            # 閃爍訊息
            if time.time() < flash_until:
                draw_flash(show_img, flash_msg, sc)
                render_needed = True   # 保持更新直到 flash 結束
            else:
                render_needed = False

            cv2.imshow(win_name, show_img)

        # 按鍵等待
        if playing or time.time() < flash_until:
            elapsed      = time.time() - last_advance_t
            remaining_ms = max(30, int((play_delay_ms / 1000.0 - elapsed) * 1000)) if playing else 30
            key = cv2.waitKey(remaining_ms) & 0xFF
        else:
            # 原本這裡是單次阻塞的 cv2.waitKey(0)：等一個「真的按鍵」才會返回，中途滑鼠在
            # 時間軸上點擊/拖曳/懸停雖然照樣會觸發 _on_annotate_mouse()（cv2 的訊息幫浦在
            # waitKey 阻塞期間仍會派送滑鼠事件），但畫面不會重繪，使用者會覺得「點了沒反應」，
            # 直到又按了一個鍵才會突然跳過去。改成 50ms 一次的輪詢迴圈，滑鼠回呼裡把
            # render_needed 設成 True 就能提前跳出、立即重繪（拖曳/懸停預覽才會即時跟著動）。
            key = 255
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key != 255 or render_needed:
                    break

        if key == 27:
            break
        elif key == ord('n'):
            go_next = True
            break
        elif key == ord(' '):
            playing = not playing
            if playing:
                last_advance_t = time.time()
            render_needed = True
        elif key == ord('a'):
            playing    = False
            cur_idx    = max(0, cur_idx - skip_n)
            needs_seek = True
            render_needed = True
        elif key == ord('d'):
            playing   = False
            cur_idx   = min(total_frames - 1, cur_idx + skip_n)
            render_needed = True
        elif key == ord('['):  # 2026-09 前是 z，改掉避免跟其他腳本「z=切到 WALK 資料夾」混淆
            skip_n = max(1, skip_n - 1)
            render_needed = True
        elif key == ord(']'):  # 2026-09 前是 x，改掉避免跟其他腳本「x=切到 LICK 資料夾」混淆
            skip_n += 1
            render_needed = True
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            current_action = VALID_ACTIONS[key - ord('1')]
            render_needed  = True
        elif key == ord('s'):
            if not marking:
                start_idx  = cur_idx
                marking    = True
                s_ts = frames[start_idx].get('timestamp', 0) or 0
                flash_msg   = f"MARK START  [{current_action.upper()}]  @ {s_ts:.2f}s"
                flash_until = time.time() + 2.5
                print(f"  [{current_action}] 標記起點: {start_idx+1} ({s_ts:.2f}s)")
            else:
                end_idx = cur_idx
                if end_idx < start_idx:
                    start_idx, end_idx = end_idx, start_idx
                intervals.append((start_idx, end_idx, current_action))
                intervals_touched = True
                s_ts = frames[start_idx].get('timestamp', 0) or 0
                e_ts = frames[end_idx].get('timestamp', 0) or 0
                dur  = abs(e_ts - s_ts)
                flash_msg   = f"SAVED  [{current_action.upper()}]  {s_ts:.2f}s - {e_ts:.2f}s  (dur {dur:.2f}s)"
                flash_until = time.time() + 5.0
                print(f"  [{current_action}]  {s_ts:.2f}s - {e_ts:.2f}s  ({dur:.2f}s)  [total {len(intervals)} interval(s)]")
                marking = False
            render_needed = True
        elif key == ord('u'):
            if marking:
                marking   = False
                flash_msg = 'MARK CANCELLED'
            elif intervals:
                removed   = intervals.pop()
                intervals_touched = True
                flash_msg = f"UNDO  [{removed[2].upper()}]  frames {removed[0]+1}~{removed[1]+1}"
                print(f"  ↩ 撤銷: {removed}")
            else:
                flash_msg = '(nothing to undo)'
            flash_until   = time.time() + 1.2
            render_needed = True
        elif key == ord('t'):
            try:
                raw = input("跳轉（幀號整數 或 秒數如 3.5 / 3.5s）: ").strip()
                if '.' in raw or raw.endswith('s'):
                    sec = np.float64(raw.rstrip('s'))
                    cur_idx = min(range(total_frames),
                                  key=lambda i: abs(frames[i].get('timestamp', 0) - sec))
                else:
                    # 1-based 幀號輸入
                    cur_idx = max(0, min(total_frames - 1, int(raw) - 1))
                needs_seek = True
                playing    = False
                render_needed = True
                print(f"  跳轉到第 {cur_idx+1} 幀 ({frames[cur_idx].get('timestamp', 0):.2f}s)")
            except Exception as e:
                print(f"  ✗ 跳轉失敗: {e}")
        elif playing and key == 0xFF:
            if cur_idx < total_frames - 1:
                cur_idx        = min(cur_idx + 1, total_frames - 1)
                last_advance_t = time.time()
                render_needed  = True
            else:
                playing    = False
                needs_seek = True
                render_needed  = True

    cv2.destroyAllWindows()
    for _ in range(5):   # Windows 需要多次 pump 才能真正關閉視窗
        cv2.waitKey(1)
    if cap:
        cap.release()

    _save_manual_annotation(json_path, data, intervals, intervals_touched)
    return 'next' if go_next else 'menu'


def _save_manual_annotation(json_path, data, intervals, intervals_touched):
    """只在確實編輯區間後更新人工確認；單純開啟再關閉不算確認。"""
    if not intervals_touched:
        print("  [保護] 本次未變更標記，保留原檔與人工確認狀態。")
        return False
    frames = data['frames']
    total_frames = len(frames)
    # 依行為類別分組並合併各自重疊區段
    from collections import defaultdict
    def merge_intervals_for_action(raw):
        raw = sorted([(min(s, e), max(s, e)) for s, e in raw])
        merged = []
        for s, e in raw:
            if not merged or merged[-1][1] < s - 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        return merged

    by_action = defaultdict(list)
    for s, e, act in intervals:
        by_action[act].append((s, e))

    action_intervals = []
    for act, segs in by_action.items():
        for s, e in merge_intervals_for_action(segs):
            action_intervals.append({"action": act, "start": int(s), "end": int(e)})
    action_intervals.sort(key=lambda x: x['start'])

    # 產生 frame-level label
    # 無變更已在函式入口直接返回；此處依使用者本次實際編輯更新逐幀標籤。
    all_intervals = sorted(action_intervals, key=lambda x: x['start'])

    if not action_intervals:
        print("  [清空] 已撤銷所有標記區間，frame label 全部設為 unannotated")
        for frame in frames:
            frame['label'] = 'unannotated'
    else:
        frame_labels = ['unannotated'] * total_frames
        for iv in action_intervals:
            for i in range(iv['start'], iv['end'] + 1):
                if 0 <= i < total_frames:
                    frame_labels[i] = iv['action']
        for i, frame in enumerate(frames):
            frame['label'] = frame_labels[i]

    out_json = data.copy()
    out_json['action_intervals'] = all_intervals
    out_json['frames'] = frames
    out_json['total_frames'] = total_frames
    out_json['annotation_review'] = _review_record(
        'manual_intervals' if all_intervals else 'manual_empty')
    _atomic_write_json(json_path, out_json)
    print(f"\n✓ 已直接覆蓋原標註檔案: {json_path}\n✓ 已自動合併重疊區段，frames 內每一幀都含 label 欄位")
    return True


def _videos_match_skeletons():
    """模式 2 進場檢查（只警告）：VIDEO_FOLDERS 的影片與 OUTPUT_FOLDER 的骨架應依檔名一一對應
    （EXCLUDED_STEMS 除外）。用檔名比對而非只比數量，才抓得到「少一支又多一支」互相抵銷。"""
    from collections import Counter
    missing_folders = [f for f in VIDEO_FOLDERS if not Path(f).exists()]
    if missing_folders:
        _alert_box('警告：找不到影片資料夾，無法核對', [
            ('以下資料夾不存在（OneDrive 還沒同步？）：', missing_folders),
        ])
        return False
    video_stems = Counter(v.stem for v in _scan_video_files_with_index()
                          if v.stem not in EXCLUDED_STEMS)
    skeleton_files = iter_skeleton_files(OUTPUT_FOLDER)
    skeleton_stems = {p.stem for p in skeleton_files}
    skeleton_count = Counter(p.stem for p in skeleton_files)
    skel_dup = sorted(s for s, n in skeleton_count.items() if n > 1)
    duplicated = sorted(s for s, n in video_stems.items() if n > 1)
    video_split = {v.stem: split_of_video(v) for v in _scan_video_files_with_index()}
    split_diff = []
    for js in iter_skeleton_files(OUTPUT_FOLDER):
        v_split, s_split = video_split.get(js.stem), split_of(js)
        if v_split and s_split in SPLITS and v_split != s_split:
            split_diff.append(f"{js.stem}（骨架 {s_split} / 影片 {v_split}）")
    no_skeleton = sorted(set(video_stems) - skeleton_stems, key=lambda s: _natural_sort_key(Path(s)))
    no_video = sorted(skeleton_stems - set(video_stems), key=lambda s: _natural_sort_key(Path(s)))
    print(f"  來源影片 {sum(video_stems.values())} 支（排除 {len(EXCLUDED_STEMS)} 支）｜骨架 {len(skeleton_stems)} 筆")
    if not (duplicated or no_skeleton or no_video or split_diff or skel_dup):
        return True
    sections = []
    if skel_dup:
        sections.append((f'同一個骨架檔名有多份：{len(skel_dup)} 個',
                         [', '.join(f'{s}（{skeleton_count[s]} 份）' for s in skel_dup),
                          '-> 多半是複製而不是搬移；gcn_dataset_manager.py 會列出每一份的位置']))
    if duplicated:
        sections.append((f'同一檔名出現在多個類別資料夾：{len(duplicated)} 支',
                         [', '.join(duplicated), '-> 骨架只有一份，會互相覆蓋；請改檔名']))
    if no_skeleton:
        sections.append((f'有影片、沒有骨架：{len(no_skeleton)} 支',
                         [', '.join(no_skeleton), '-> 先跑模式 1 抽骨架；不要的影片加進 EXCLUDED_STEMS']))
    if split_diff:
        sections.append((f'骨架和影片放在不同的 split：{len(split_diff)} 支',
                         [', '.join(split_diff),
                          '-> 以骨架為準：gcn_dataset_manager.py 模式 3 把影片搬過去',
                          '-> 以影片為準：gcn_dataset_manager.py 模式 2 把骨架搬過去']))
    if no_video:
        sections.append((f'有骨架、找不到來源影片：{len(no_video)} 筆',
                         [', '.join(no_video), '-> 把影片放回 VIDEO_FOLDERS，或移走這些骨架 JSON']))
    _alert_box(f'警告：影片 {sum(video_stems.values())} 支 / 骨架 {len(skeleton_stems)} 筆，對不上'
               if (duplicated or no_skeleton or no_video) else '警告：骨架和影片的 split 不一致', sections,
               footer=('仍可繼續標記，但沒有骨架的影片不會出現在標記清單裡。' if no_skeleton
                       else '仍可繼續標記。'))
    return False


def _advance(menu_state, json_path, exit_action):
    """標記視窗關閉後要開哪一筆：按 n 就回傳清單下一筆，否則 None（回到清單）。"""
    if exit_action != 'next':
        return None
    nxt = _next_in_queue(menu_state, json_path)
    if nxt is None:
        print("  已經是目前清單的最後一筆，回到清單。")
    else:
        print(f"  -> 下一筆：{Path(nxt).name}")
    return nxt


def manual_action_labeling():
    """
    連續標記多個 skeleton JSON 檔案。
    標記完成後自動回到列表，輸入 q 退出。
    """
    print("\n=== Annotation Mode ===")
    print(f"Folder: {OUTPUT_FOLDER}\n")
    _videos_match_skeletons()   # 對不上只警告，不阻止標記

    last_annotated = None
    menu_state = {}  # 在本次工作階段保留類別、搜尋與清單篩選。
    json_path = None
    while True:
        if json_path is None:
            json_path = _list_json_files_menu(OUTPUT_FOLDER, last_annotated=last_annotated,
                                              menu_state=menu_state)
            if not json_path:
                print("\n[Done] Annotation session ended.")
                break

        # 視窗標題顯示在「目前清單」裡的位置
        file_index, total_files = _queue_position(menu_state, json_path)
        exit_action = _annotate_single_skeleton(json_path,
                                                file_index=file_index,
                                                total_files=total_files)

        # 標記完成後在終端列印明確摘要
        sep = '=' * 62
        print(f"\n{sep}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            ivs      = d.get('action_intervals', [])
            n_frames = d.get('total_frames', 0)
            labeled  = sum(1 for fr in d.get('frames', [])
                           if fr.get('label', 'unannotated') != 'unannotated')
            pct      = labeled / n_frames * 100 if n_frames > 0 else 0.0

            # 統計各行為的區間數
            from collections import Counter
            act_counts = Counter(iv['action'] for iv in ivs)
            act_str    = '  '.join(f"{act}x{cnt}" for act, cnt in sorted(act_counts.items()))

            print(f"  FILE      : {Path(json_path).name}")
            print(f"  Status    : {_ANNOTATION_STATUS_TEXT[_annotation_state(d)]}")
            print(f"  Intervals : {len(ivs)}   ({act_str if act_str else 'none'})")
            print(f"  Labeled   : {labeled}/{n_frames} frames  ({pct:.1f}%)")
        except Exception as e:
            print(f"  ANNOTATED : {Path(json_path).name}")
            print(f"  (Could not read summary: {e})")
        print(f"{sep}\n")
        last_annotated = json_path   # 供下次列表顯示「上次標記」
        json_path = _advance(menu_state, json_path, exit_action)

    print("\n[Done] All annotation tasks completed.")


def review_test_labels(folder: str = None):
    """
    純檢視獨立測試集（TEST_OUTPUT_FOLDER）目前的標註結果，不開啟任何影片
    視窗，可以隨時執行、不會誤觸標註流程。

    這是「怎麼知道被打上的資料標籤」的主要管道：終端印出一張表（依行為分
    組，每支影片列出區間數、事件幀數/總幀數、事件佔比），同時把同一份內
    容寫成 CSV（TEST_OUTPUT_FOLDER 底下 _test_label_report.csv），方便之
    後不開終端也能核對，或是貼進論文/報告當附錄。
    """
    folder = folder or TEST_OUTPUT_FOLDER
    p = Path(folder)
    if not p.exists():
        print(f"[Error] 資料夾不存在: {folder}（先跑模式 6 提取骨架）")
        return

    json_files = sorted(p.glob("*.json"), key=_natural_sort_key)
    if not json_files:
        print(f"[Error] {folder} 底下沒有任何 JSON（先跑模式 6 提取骨架）")
        return

    rows = []
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            print(f"  ⚠ 讀取失敗 {jf.name}: {e}")
            continue
        meta      = d.get('video_metadata', {})
        total     = d.get('total_frames', len(d.get('frames', [])))
        ivs       = d.get('action_intervals', [])
        # frames 陣列已經被 resample_to_target_fps() 重取樣成嚴格等距
        # 1/target_fps 秒一幀（見該函式說明），所以幀號換算成影片時間要用
        # target_fps，不能用 actual_fps（來源影片原始幀率，跟重取樣後的
        # frame index 對不上，換算出來的秒數會是錯的）。
        fps       = meta.get('target_fps') or TARGET_FPS
        ev_frames = sum(max(0, iv.get('end', 0) - iv.get('start', 0)) for iv in ivs)
        frac      = (ev_frames / total * 100) if total else 0.0
        behavior  = _parse_behavior(jf.stem) or 'unknown'
        iv_desc   = '; '.join(
            f"{iv.get('action','?')}: {iv.get('start', 0) / fps:.1f}s ~ {iv.get('end', 0) / fps:.1f}s"
            f"  (frame {iv.get('start','?')}-{iv.get('end','?')})"
            for iv in ivs
        ) or '（未標註任何區間）'
        rows.append({
            'behavior': behavior, 'video': jf.name,
            'video_filename': meta.get('video_filename', ''),
            'total_frames': total, 'n_intervals': len(ivs),
            'event_frames': ev_frames, 'event_fraction_pct': round(frac, 1),
            'fps': fps, 'intervals': iv_desc,
        })

    groups = {b: [] for b in _BEHAVIOR_ORDER}
    groups['unknown'] = []
    for r in rows:
        groups[r['behavior']].append(r)

    sep = '─' * 100
    print(f"\n{'='*100}")
    print(f"  獨立測試集標註結果   {folder}")
    print(f"{'='*100}")
    for behavior in list(_BEHAVIOR_ORDER) + ['unknown']:
        grp = groups[behavior]
        if not grp:
            continue
        n_labeled = sum(1 for r in grp if r['n_intervals'] > 0)
        print(f"\n[{behavior.upper()}]  {n_labeled}/{len(grp)} 已標註區間")
        print(sep)
        for r in grp:
            status = '✓' if r['n_intervals'] > 0 else '·未標'
            print(f"  {status}  {r['video']:<24} 總幀數={r['total_frames']:>4}"
                  f"  事件幀數={r['event_frames']:>4}  事件佔比={r['event_fraction_pct']:>5.1f}%"
                  f"  區間=[{r['intervals']}]")

    # 同步輸出 CSV，留一份不依賴終端輸出的紀錄
    import csv
    csv_path = p / "_test_label_report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'behavior', 'video', 'video_filename', 'total_frames',
            'n_intervals', 'event_frames', 'event_fraction_pct', 'fps', 'intervals',
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n✓ 已同步輸出 CSV 報告：{csv_path}")


def label_test_set():
    """
    模式 6 入口，進來後再選一次要做哪個步驟（抽骨架跟標註未必想綁在一
    起做——例如骨架已經抽過、這次只想標註；或只想先把骨架抽完，晚點再
    找時間標）：

      1. 只抽骨架：呼叫 extract_test_set_skeletons()（增量，跳過已有 JSON）
      2. 只標註：假設 TEST_OUTPUT_FOLDER 底下骨架已存在，直接進入標註
         迴圈（沿用模式 2 的 _list_json_files_menu / _annotate_single_skeleton）
      3. 兩者都做：先抽骨架，再進入標註迴圈（原本的預設行為）

    不論選哪個步驟，結束後都會自動印出 review_test_labels() 的完整報
    告，回答「被打上什麼標籤」這件事，不用另外再跑一次模式 7。
    """
    print("\n=== 獨立測試集標註模式 ===")
    print("要執行哪個步驟？")
    print("  1. 只抽骨架（增量提取，不進入標註）")
    print("  2. 只標註（骨架需已存在，直接進入標註迴圈）")
    print("  3. 兩者都做（先抽骨架，再進入標註）")
    sub = input("請選擇 (1/2/3，直接 Enter = 3): ").strip() or '3'
    if sub not in ('1', '2', '3'):
        print("✗ 未選擇正確步驟，程式結束。")
        return

    if sub in ('1', '3'):
        extract_test_set_skeletons()

    if sub in ('2', '3'):
        if sub == '2' and not any(Path(TEST_OUTPUT_FOLDER).glob("*.json")):
            print(f"\n[Error] {TEST_OUTPUT_FOLDER} 底下沒有任何骨架 JSON，"
                  "請先選 1 或 3 抽骨架。")
            return

        print(f"\nFolder: {TEST_OUTPUT_FOLDER}\n")
        last_annotated = None
        menu_state = {}
        json_path = None
        while True:
            if json_path is None:
                json_path = _list_json_files_menu(TEST_OUTPUT_FOLDER, last_annotated=last_annotated,
                                                  menu_state=menu_state)
                if not json_path:
                    print("\n[Done] 測試集標註工作階段結束。")
                    break

            file_index, total_files = _queue_position(menu_state, json_path)
            exit_action = _annotate_single_skeleton(json_path, file_index=file_index,
                                                    total_files=total_files)

            sep = '=' * 62
            print(f"\n{sep}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                ivs      = d.get('action_intervals', [])
                n_frames = d.get('total_frames', 0)
                labeled  = sum(1 for fr in d.get('frames', [])
                               if fr.get('label', 'unannotated') != 'unannotated')
                pct      = labeled / n_frames * 100 if n_frames > 0 else 0.0
                from collections import Counter
                act_counts = Counter(iv['action'] for iv in ivs)
                act_str    = '  '.join(f"{act}x{cnt}" for act, cnt in sorted(act_counts.items()))
                print(f"  ANNOTATED : {Path(json_path).name}")
                print(f"  Intervals : {len(ivs)}   ({act_str if act_str else 'none'})")
                print(f"  Labeled   : {labeled}/{n_frames} frames  ({pct:.1f}%)")
            except Exception as e:
                print(f"  ANNOTATED : {Path(json_path).name}")
                print(f"  (Could not read summary: {e})")
            print(f"{sep}\n")
            last_annotated = json_path
            json_path = _advance(menu_state, json_path, exit_action)

    # 不論剛剛做了哪個步驟，結束後都自動印出完整報告
    review_test_labels(TEST_OUTPUT_FOLDER)


def reextract_preserve_labels(target_stems: set = None):
    """
    以新 YOLO 模型重新推論骨架，並依「時間」（而非幀 index）還原既有 JSON 的
    frame label／action_intervals，因此不論新舊抽取的 TARGET_FPS、補償邏輯或
    總幀數是否一致，標注都能正確對應到新的時間網格。

    target_stems: 若提供，只重新推論檔名（不含副檔名）在此集合內的影片，其餘
    影片完全跳過（連掃描/讀取都不做）；標籤覆蓋邏輯與 target_stems=None（模式 3，
    處理全部影片）完全相同，差別只在於「要重新推論哪些檔案」。
    """
    print("="*60)
    print("Skeleton Re-extraction (preserve annotations)")
    if target_stems:
        print(f"（僅限指定的 {len(target_stems)} 支影片）")
    print("="*60)
    setup_directories()

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    video_files = []
    for folder in VIDEO_FOLDERS:
        vp = Path(folder)
        if not vp.exists():
            print(f"[Warning] 資料夾不存在: {folder}")
            continue
        video_files.extend(sorted(
            (f for f in vp.iterdir() if f.suffix.lower() in video_extensions),
            key=_natural_sort_key,
        ))

    if target_stems is not None:
        found_stems = {v.stem for v in video_files if v.stem in target_stems}
        missing_stems = target_stems - found_stems
        if missing_stems:
            print(f"  ⚠ 在 VIDEO_FOLDERS 裡找不到以下指定檔案，已略過："
                  f"{', '.join(sorted(missing_stems))}")
        video_files = [v for v in video_files if v.stem in target_stems]

    if not video_files:
        print("✗ 找不到任何影片。")
        return

    # 讀取所有既有 JSON 的 action_intervals，連同每一幀的真實 timestamp。
    # 用 timestamp（而非 frame index）還原標注，這樣即使新舊抽取的總幀數不同
    # （例如舊資料在補償邏輯上線前抽取、非 30fps 來源），標注依然能正確對應。
    saved_reviews = {}
    unreadable_stems = set()
    saved_intervals: dict[str, list] = {}
    saved_frame_labels: dict[str, list] = {}   # video_id -> 舊逐幀 label（依 old timestamp 順序）
    saved_timestamps: dict[str, list] = {}     # video_id -> 舊逐幀 timestamp
    for vf in video_files:
        jp = find_skeleton(OUTPUT_FOLDER, vf.stem)
        if jp is not None:
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                ivs = d.get('action_intervals', [])
                saved_intervals[vf.stem] = ivs
                old_frames = d.get('frames', [])
                if 'annotation_review' in d:
                    saved_reviews[vf.stem] = d['annotation_review']
                # 無區間、無人工確認的舊資料仍按原本整段有效處理，不能全設成 unannotated。
                # 舊檔若已整段清空，保留其無有效幀的結果，但不猜測是否曾人工確認。
                legacy_empty = bool(old_frames) and all(
                    fr.get('label') == 'unannotated' for fr in old_frames)
                if not ivs and _annotation_state(d) == 'pending' and not legacy_empty:
                    continue
                old_total_f = len(old_frames) or d.get('total_frames', 0)
                old_meta = d.get('video_metadata') or {}
                old_fps = (old_meta.get('target_fps') if old_meta.get('fps_compensated')
                           else old_meta.get('actual_fps')) or TARGET_FPS
                labels = ['unannotated'] * old_total_f
                for iv in ivs:
                    for i in range(iv['start'], iv['end'] + 1):
                        if 0 <= i < old_total_f:
                            labels[i] = iv['action']
                timestamps = [
                    fr.get('timestamp', i / old_fps) for i, fr in enumerate(old_frames)
                ] or [i / old_fps for i in range(old_total_f)]
                saved_frame_labels[vf.stem] = labels
                saved_timestamps[vf.stem] = timestamps
            except Exception as e:
                print(f"  [Warning] 無法讀取 {jp.name}，本次不重抽以保留原檔: {e}")
                unreadable_stems.add(vf.stem)

    video_files = [v for v in video_files if v.stem not in unreadable_stems]
    if not video_files:
        print('✗ 沒有可安全重抽的影片。')
        return
    annotated = [v for v in video_files if v.stem in saved_frame_labels]
    unannotated = [v for v in video_files if v.stem not in saved_frame_labels]

    print(f"\n影片總數：{len(video_files)}")
    print(f"  有標記／清空紀錄（保留）：{len(annotated)}")
    print(f"  無標記（重抽後仍尚未標記，訓練前需補標）：{len(unannotated)}")
    if annotated:
        print("\n  [含標注]")
        for v in annotated:
            n = len(saved_intervals[v.stem])
            print(f"    {v.name}  →  {n} 個區間將保留")

    confirm = input('\n確認後輸入 "ok" 開始重新推論（其他任意鍵取消）：').strip().lower()
    if confirm != "ok":
        print("✗ 已取消。")
        return

    pose_extractor = PoseExtractor(
        model_path=MODEL_PATH,
        imgsz=IMGSZ,
        conf_threshold=CONF_THRESHOLD
    )

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] {video_path.name}")
        video_id    = video_path.stem
        output_path = skeleton_path_for(OUTPUT_FOLDER, video_id, split=split_of_video(video_path))
        label       = video_path.parent.name.lower()

        result = extract_skeleton_from_video(
            video_path, pose_extractor, target_fps=TARGET_FPS, label=label
        )
        if result is None:
            print("  ✗ 推論失敗，跳過")
            continue
        skeleton_data, actual_fps = result
        if not skeleton_data:
            print('  ✗ 沒有抽取到任何幀，保留原檔')
            continue

        # 還原既有標注：用「時間」而非「幀 index」對應。
        # resample_to_target_fps 補償後的幀數只取決於(影片時長, target_fps)，
        # 與舊版「來源 fps<=target 時 1:1 保留全部原始幀」的抽幀結果不一定相同，
        # 直接沿用舊 index 對應會錯位；改用每幀真實 timestamp 找最近的舊幀取其
        # label，不論新舊總幀數是否相同都能正確對應，資料不會因此報廢。
        old_labels = saved_frame_labels.get(video_id)
        old_ts = saved_timestamps.get(video_id)
        review = saved_reviews.get(video_id) or {}
        if review.get('status') == 'reviewed' and review.get('method') == 'full_clip':
            # 整段確認涵蓋整支影片；重取樣後仍延伸至新的最後一幀。
            full_action = review.get('action')
            if full_action not in _BEHAVIOR_ORDER:
                print('  ✗ 整段確認紀錄缺少有效行為，保留原檔')
                continue
            for fd in skeleton_data:
                fd['label'] = full_action
            new_intervals = [{'action': full_action, 'start': 0, 'end': len(skeleton_data) - 1}]
            print(f'  ✓ 保留整段人工確認：{full_action}，共 {len(skeleton_data)} 幀')
        elif review.get('status') == 'reviewed' and review.get('method') == 'manual_empty':
            for fd in skeleton_data:
                fd['label'] = 'unannotated'
            new_intervals = []
            print('  ✓ 保留人工清空結果：全部幀維持 unannotated')
        elif old_labels and old_ts:
            old_ts_arr = np.asarray(old_ts, dtype=np.float64)
            # 舊資料的取樣間隔，超過這個間隔找不到對應舊幀就視為原本就沒標注的空窗
            old_gap = float(np.median(np.diff(old_ts_arr))) if len(old_ts_arr) > 1 else (1.0 / TARGET_FPS)
            max_gap = max(old_gap, 1.0 / TARGET_FPS) * 2
            frame_labels = []
            for fd in skeleton_data:
                t = fd.get('timestamp', 0.0)
                nearest = int(np.argmin(np.abs(old_ts_arr - t)))
                if abs(old_ts_arr[nearest] - t) <= max_gap:
                    frame_labels.append(old_labels[nearest])
                else:
                    frame_labels.append('unannotated')
            for fd, lbl in zip(skeleton_data, frame_labels):
                fd['label'] = lbl

            # 從還原後的逐幀 label 重新產生 action_intervals（合併連續同標籤區段）
            new_intervals = []
            run_start, run_label = None, None
            for i, lbl in enumerate(frame_labels):
                if lbl != run_label:
                    if run_label not in (None, 'unannotated'):
                        new_intervals.append({"action": run_label, "start": run_start, "end": i - 1})
                    run_start, run_label = i, lbl
            if run_label not in (None, 'unannotated'):
                new_intervals.append({"action": run_label, "start": run_start, "end": len(frame_labels) - 1})

            if new_intervals:
                print(f"  ✓ 依 timestamp 還原標注：{len(new_intervals)} 個 action_intervals"
                      f"（舊 {len(old_labels)} 幀 → 新 {len(skeleton_data)} 幀）")
            else:
                print("  → 還原後無有效標記區間，全部幀維持 unannotated")
        else:
            new_intervals = []
            print("  → 無既有標注，frame label 保持資料夾名稱")

        video_metadata = {
            "video_id": video_id,
            "video_filename": video_path.name,
            "video_path": str(video_path),
            "target_fps": TARGET_FPS,
            "actual_fps": actual_fps,
            "fps_compensated": True,
            "model_used": MODEL_PATH,
            "imgsz": IMGSZ,
            "conf_threshold": CONF_THRESHOLD,
            "kp_conf_threshold": KP_CONF_THRESHOLD
        }
        out_data = {
            "video_metadata": video_metadata,
            "frames": skeleton_data,
            "total_frames": len(skeleton_data),
            "action_intervals": new_intervals,
        }
        if video_id in saved_reviews:
            out_data['annotation_review'] = saved_reviews[video_id]
        elif saved_intervals.get(video_id):
            # 舊的人工區間即使此次沒有對應新幀，仍保留「曾標記」的狀態。
            out_data['annotation_review'] = {
                'status': 'reviewed', 'method': 'manual_intervals',
                'source': 'legacy_action_intervals',
            }
        _atomic_write_json(output_path, out_data)
        print(f"  ✓ 已儲存: {output_path}")

    print("\n✓ 全部重新推論完成。")


def _scan_video_files_with_index():
    """掃描 VIDEO_FOLDERS，回傳依編號排序好的影片清單（供選檔用的編號跟這裡的順序一致）。"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    for folder in VIDEO_FOLDERS:
        vp = Path(folder)
        if not vp.exists():
            continue
        video_files.extend(sorted(
            (f for f in vp.iterdir() if f.suffix.lower() in video_extensions),
            key=_natural_sort_key,
        ))
    return video_files


def _parse_video_selection(raw: str, video_files: list) -> set:
    """
    解析選檔輸入，逗號分隔，可混用以下寫法：
      - 編號：3
      - 編號區段（含頭尾）：3-9
      - 行為名稱（walk/lick/scratch/shake/stop）：選中該行為的全部影片
      - all：全選
      - 其餘一律當作檔名（不含副檔名）直接比對，方便手動指定不在清單編號範圍內的檔案
    回傳選中的檔名（stem）集合；無法解析的編號/區段/行為名稱會印警告但不中斷。
    """
    raw = raw.strip()
    if not raw:
        return set()
    if raw.lower() == 'all':
        return {v.stem for v in video_files}

    stem_by_index = {i: v.stem for i, v in enumerate(video_files, 1)}
    behavior_stems = {
        behavior: {v.stem for v in video_files if _parse_behavior(v.stem) == behavior}
        for behavior in _BEHAVIOR_ORDER
    }

    selected = set()
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r'\d+', token):
            idx = int(token)
            if idx in stem_by_index:
                selected.add(stem_by_index[idx])
            else:
                print(f"  ⚠ 編號 {idx} 超出範圍（共 {len(video_files)} 支），已略過")
        elif re.fullmatch(r'\d+-\d+', token):
            start, end = (int(x) for x in token.split('-'))
            if start > end:
                start, end = end, start
            matched = [stem_by_index[i] for i in range(start, end + 1) if i in stem_by_index]
            if matched:
                selected.update(matched)
            else:
                print(f"  ⚠ 區段 {token} 沒有對應到任何影片，已略過")
        elif token.lower() in behavior_stems:
            matched = behavior_stems[token.lower()]
            if matched:
                selected.update(matched)
                print(f"  ✓ [{token}] 選中 {len(matched)} 支影片")
            else:
                print(f"  ⚠ 找不到任何 [{token}] 的影片，已略過")
        else:
            selected.add(token)  # 當作檔名直接比對，交由下游驗證是否存在
    return selected


def reextract_preserve_labels_selected():
    """
    模式 3 的變體：標籤覆蓋邏輯完全相同（依 timestamp 還原既有 action_intervals／
    frame label），但骨架重新推論只針對使用者手動指定的部分檔案，不會掃描/重推
    VIDEO_FOLDERS 裡的其他影片。適合只想修正少數幾支骨架抽取結果不佳的影片，
    不想為此重跑整個資料集（耗時）的情境。
    """
    print("="*60)
    print("Skeleton Re-extraction (preserve annotations, 指定檔案)")
    print("="*60)

    video_files = _scan_video_files_with_index()
    if not video_files:
        print("✗ VIDEO_FOLDERS 裡找不到任何影片。")
        return

    print(f"\n可選影片清單（共 {len(video_files)} 支）：")
    for i, v in enumerate(video_files, 1):
        print(f"  [{i:>3}] {v.stem}")

    print("\n輸入方式（逗號分隔，可混用）：")
    print("  編號：3        編號區段：3-9        行為全選：scratch        全選：all")
    print("  也可以直接輸入檔名（不含副檔名），例如 scratch_22")
    raw = input("請輸入要重新推論的影片：").strip()
    target_stems = _parse_video_selection(raw, video_files)
    if not target_stems:
        print("✗ 未選擇任何影片，已取消。")
        return
    print(f"\n共選中 {len(target_stems)} 支影片：{', '.join(sorted(target_stems))}")
    reextract_preserve_labels(target_stems=target_stems)


# ==================== Window-level Discard Report ====================
# 以下三個輔助函式複製 0_train_gcn.py CatSkeletonDataset._load_sequences() 的
# 逐視窗捨棄判斷邏輯，讀取同一份 stgcn_config.yaml，確保報告跟實際訓練切窗
# 行為不脫鉤。捨棄決策只取決於 frame label 與 bbox 有無，跟座標數值/插值/EMA
# 平滑無關，所以這裡不需要重做 interpolate_missing 等前處理。

def _find_stgcn_config_path():
    """依 0_train_gcn.py 相同規則尋找 stgcn_config.yaml，可用 STGCN_CONFIG_PATH 環境變數覆寫。"""
    env_path = os.getenv('STGCN_CONFIG_PATH')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    default_path = Path(r"C:\ai_project\paper\cat_monitoring_system\stgcn_config.yaml")
    if default_path.exists():
        return default_path
    local_path = Path(__file__).resolve().parent.parent / 'stgcn_config.yaml'
    if local_path.exists():
        return local_path
    return None


def _load_window_filter_config():
    """讀取切窗相關參數：SEQUENCE_LENGTH / WINDOW_STRIDE / STRICT_WINDOW_FILTER /
    MAX_NO_DETECT_FRAMES / BEHAVIOR_PREFIXES。找不到設定檔或缺欄位時回傳 None
    （呼叫端應改印警告並略過視窗層級報告，不影響既有幀層級報告）。"""
    config_path = _find_stgcn_config_path()
    if config_path is None:
        print("  ⚠ 找不到 stgcn_config.yaml，略過視窗層級報告。")
        return None
    try:
        import yaml
    except ImportError:
        print("  ⚠ 未安裝 PyYAML（pip install pyyaml），略過視窗層級報告。")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  ⚠ 無法讀取 {config_path}：{e}")
        return None

    required = ['SEQUENCE_LENGTH', 'WINDOW_STRIDE', 'STRICT_WINDOW_FILTER', 'BEHAVIOR_PREFIXES']
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"  ⚠ {config_path.name} 缺少必要欄位 {missing}，略過視窗層級報告。")
        return None

    return {
        'sequence_length':   int(cfg['SEQUENCE_LENGTH']),
        'window_stride':     int(cfg['WINDOW_STRIDE']),
        'strict_filter':     bool(cfg['STRICT_WINDOW_FILTER']),
        'max_no_detect':     int(cfg.get('MAX_NO_DETECT_FRAMES', 2)),
        'behavior_prefixes': cfg['BEHAVIOR_PREFIXES'],
        'path':              config_path,
    }


def _classify_windows(frames, wcfg):
    """
    對單支影片的 frames 重演訓練時的滑動切窗，回傳每個 start_idx 的
    (start_idx, kept, reason)。reason 為 None 表示保留；否則為
    'unannotated' / 'no_detect' / 'unknown_label:<label>'。
    幀數 < sequence_length 時回傳 None（訓練時整支影片會被跳過，不計入視窗統計）。
    """
    from collections import Counter

    seq_len = wcfg['sequence_length']
    stride  = wcfg['window_stride']
    strict  = wcfg['strict_filter']
    max_no_detect = wcfg['max_no_detect']
    name_to_idx   = wcfg['behavior_prefixes']

    if len(frames) < seq_len:
        return None

    labels   = [fr.get('label', 'unannotated') for fr in frames]
    detected = [fr.get('bbox') is not None for fr in frames]

    windows = []
    for start_idx in range(0, len(frames) - seq_len + 1, stride):
        window_labels   = labels[start_idx:start_idx + seq_len]
        window_detected = detected[start_idx:start_idx + seq_len]

        if strict:
            if 'unannotated' in window_labels:
                windows.append((start_idx, False, 'unannotated'))
                continue
            best_label = Counter(window_labels).most_common(1)[0][0]
        else:
            annotated = [lbl for lbl in window_labels if lbl != 'unannotated']
            if not annotated:
                windows.append((start_idx, False, 'unannotated'))
                continue
            best_label = Counter(annotated).most_common(1)[0][0]

        if window_detected.count(False) > max_no_detect:
            windows.append((start_idx, False, 'no_detect'))
            continue

        if best_label not in name_to_idx:
            windows.append((start_idx, False, f'unknown_label:{best_label}'))
            continue

        windows.append((start_idx, True, None))

    return windows


def _merge_discarded_runs(windows, frames, seq_len):
    """把連續（在 window 序列中相鄰）且捨棄原因相同的視窗合併成一段，
    標記出該段對應原始影片的 frame 範圍與秒數，方便回頭在影片裡定位。"""
    runs = []
    i, n = 0, len(windows)
    while i < n:
        start_idx, kept, reason = windows[i]
        if kept:
            i += 1
            continue
        j = i
        while j + 1 < n and (not windows[j + 1][1]) and windows[j + 1][2] == reason:
            j += 1
        first_start = windows[i][0]
        last_start  = windows[j][0]
        last_end    = min(last_start + seq_len - 1, len(frames) - 1)
        runs.append({
            'reason':          reason,
            'n_windows':       j - i + 1,
            'start_frame_idx': first_start,
            'end_frame_idx':   last_end,
            'start_orig_fid':  frames[first_start].get('original_frame_id', first_start),
            'end_orig_fid':    frames[last_end].get('original_frame_id', last_end),
            'start_ts':        frames[first_start].get('timestamp'),
            'end_ts':          frames[last_end].get('timestamp'),
        })
        i = j + 1
    return runs


def check_discarded_files():
    """
    檢查有哪些影片／幀是被捨棄或過濾掉、不會進入訓練的資料。
    純粹讀取並印出報告，不修改任何檔案，可安全重複執行。

    涵蓋兩個層級：
      1. 影片層級：EXCLUDED_STEMS 永久排除清單，以及來源資料夾裡
         尚未提取成 JSON（不在 skip 也不在 excluded）的影片。
      2. 幀層級：每個 skeleton JSON 內 label == 'unannotated' 的幀
         （手動標注模式下未標記的片段，訓練時自動過濾，見本檔頂部
         「frame label 三種狀態」說明）以及 detected == False（YOLO
         未偵測到貓、keypoints 為空）的幀數與比例；全部幀都被過濾掉
         的檔案，代表整份 JSON 對訓練沒有任何貢獻。
      3. 視窗層級：讀取 stgcn_config.yaml 的 SEQUENCE_LENGTH/WINDOW_STRIDE/
         STRICT_WINDOW_FILTER/MAX_NO_DETECT_FRAMES，重演 0_train_gcn.py
         CatSkeletonDataset._load_sequences() 的滑動切窗判斷，標出每支
         影片裡「哪些訓練視窗被丟棄、原因為何、對應原始影片的第幾幀/
         第幾秒」（連續同原因的視窗會合併成一段顯示）。
    """
    print("="*60)
    print("Discarded / Filtered Data Report")
    print("="*60)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    # ── 影片層級 ──────────────────────────────────────────────
    existing_stems = {p.stem for p in iter_skeleton_files(OUTPUT_FOLDER)}
    video_files = []
    for folder in VIDEO_FOLDERS:
        video_folder = Path(folder)
        if not video_folder.exists():
            print(f"[Warning] Video folder not found: {folder}")
            continue
        video_files.extend(sorted(
            (f for f in video_folder.iterdir() if f.suffix.lower() in video_extensions),
            key=_natural_sort_key,
        ))

    excluded_list = [v for v in video_files if v.stem in EXCLUDED_STEMS]
    not_extracted = [v for v in video_files
                      if v.stem not in existing_stems and v.stem not in EXCLUDED_STEMS]

    print(f"\n【影片層級】來源影片總數：{len(video_files)}")
    print(f"  永久排除 (EXCLUDED_STEMS)：{len(excluded_list)} 支")
    for v in excluded_list:
        print(f"    - {v.name}")
    print(f"  尚未提取成 JSON（不在排除清單，也還沒跑過模式 1/3）：{len(not_extracted)} 支")
    for v in not_extracted:
        print(f"    - {v.name}")

    # ── 幀層級 ──────────────────────────────────────────────
    json_files = sorted(iter_skeleton_files(OUTPUT_FOLDER), key=_natural_sort_key)
    if not json_files:
        print(f"\n[Warning] {OUTPUT_FOLDER} 底下沒有任何 skeleton JSON，略過幀層級檢查。")
        return

    print(f"\n【幀層級】掃描 {len(json_files)} 個 skeleton JSON ...")
    sep = "─" * 72
    print(sep)

    # 視窗層級所需的切窗設定（找不到就跳過，不影響幀層級報告）
    wcfg = _load_window_filter_config()
    if wcfg:
        print(f"  ✓ 視窗設定來源: {wcfg['path']}"
              f"  T={wcfg['sequence_length']}  stride={wcfg['window_stride']}"
              f"  strict={wcfg['strict_filter']}  max_no_detect={wcfg['max_no_detect']}")
        print(sep)

    total_frames_all = 0
    total_unannotated_all = 0
    total_undetected_all = 0
    fully_discarded_files = []   # 全部幀都是 unannotated，對訓練完全沒貢獻
    per_file_rows = []

    from collections import Counter as _Counter
    window_too_short_videos = []      # 幀數 < sequence_length，訓練時整支跳過
    total_windows_all       = 0
    kept_windows_all        = 0
    discard_reason_totals   = _Counter()
    per_video_window_runs   = []      # [(video_name, runs, n_total, n_kept, n_discarded), ...]

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            print(f"  ⚠ 無法讀取 {jf.name}：{e}")
            continue

        frames = d.get('frames', [])
        n_total = len(frames)
        if n_total == 0:
            continue

        n_unannotated = sum(1 for fr in frames if fr.get('label', 'unannotated') == 'unannotated')
        n_undetected  = sum(1 for fr in frames if not fr.get('detected', False))

        total_frames_all      += n_total
        total_unannotated_all += n_unannotated
        total_undetected_all  += n_undetected

        if n_unannotated == n_total:
            fully_discarded_files.append(jf.name)

        per_file_rows.append({
            "name": jf.name,
            "total": n_total,
            "unannotated": n_unannotated,
            "undetected": n_undetected,
        })

        if wcfg:
            windows = _classify_windows(frames, wcfg)
            if windows is None:
                window_too_short_videos.append((jf.stem, n_total))
            else:
                n_kept = sum(1 for _, kept, _ in windows if kept)
                n_disc = len(windows) - n_kept
                total_windows_all += len(windows)
                kept_windows_all  += n_kept
                for _, kept, reason in windows:
                    if not kept:
                        discard_reason_totals[reason] += 1
                if n_disc:
                    runs = _merge_discarded_runs(windows, frames, wcfg['sequence_length'])
                    per_video_window_runs.append((jf.stem, runs, len(windows), n_kept, n_disc))

    for row in per_file_rows:
        pct_unannotated = row["unannotated"] / row["total"] * 100 if row["total"] else 0
        pct_undetected  = row["undetected"] / row["total"] * 100 if row["total"] else 0
        flag = "  ⚠ 全檔未標注，對訓練無貢獻" if row["unannotated"] == row["total"] else ""
        print(f"  {row['name']:<40}  未標註 {row['unannotated']:>5}/{row['total']:<5} ({pct_unannotated:5.1f}%)"
              f"   未偵測 {row['undetected']:>5} ({pct_undetected:5.1f}%){flag}")

    print(sep)
    pct_unannotated_all = total_unannotated_all / total_frames_all * 100 if total_frames_all else 0
    pct_undetected_all  = total_undetected_all / total_frames_all * 100 if total_frames_all else 0
    print(f"  合計幀數：{total_frames_all}")
    print(f"  未標註（訓練時自動過濾）：{total_unannotated_all} ({pct_unannotated_all:.1f}%)")
    print(f"  未偵測到貓（keypoints 為空）：{total_undetected_all} ({pct_undetected_all:.1f}%)")

    if fully_discarded_files:
        print(f"\n  ⚠ 有 {len(fully_discarded_files)} 個檔案全部幀都是 unannotated，對訓練完全沒有貢獻：")
        for name in fully_discarded_files:
            print(f"    - {name}")

    # ── 視窗層級：哪些訓練用滑動視窗被丟棄、從影片哪個位置切出來 ──────────────
    if wcfg:
        print(f"\n【視窗層級】依訓練切窗參數重演每支影片的滑動視窗判斷 ...")
        print(sep)
        for video_name, runs, n_win, n_kept, n_disc in per_video_window_runs:
            pct_disc = n_disc / n_win * 100 if n_win else 0
            print(f"  {video_name}.json  候選視窗 {n_win}  保留 {n_kept}  捨棄 {n_disc} ({pct_disc:.1f}%)")
            for r in runs:
                print(f"    ✗ {r['reason']:<20} "
                      f"video frame {r['start_orig_fid']}~{r['end_orig_fid']}"
                      f"  ({r['start_ts']:.2f}s~{r['end_ts']:.2f}s)"
                      f"  x{r['n_windows']} window")
        print(sep)
        print(f"  合計候選視窗：{total_windows_all}")
        if total_windows_all:
            print(f"  保留：{kept_windows_all} ({kept_windows_all/total_windows_all*100:.1f}%)"
                  f"  捨棄：{total_windows_all - kept_windows_all}"
                  f" ({(total_windows_all - kept_windows_all)/total_windows_all*100:.1f}%)")
            for reason, cnt in discard_reason_totals.most_common():
                print(f"    - {reason}: {cnt}")
        if window_too_short_videos:
            detail = ', '.join(f"{v}({n}幀)" for v, n in window_too_short_videos)
            print(f"  ⚠ {len(window_too_short_videos)} 支影片幀數 < sequence_length="
                  f"{wcfg['sequence_length']}，訓練時整支跳過（未計入以上視窗統計）: {detail}")

    print("\n✓ 檢查完成（純讀取，未修改任何檔案）。")


def warn_unmarked(folder=None):
    """列出尚未標記區段的骨架；有的話提醒，0_train_gcn.py 會因此拒絕訓練。"""
    folder = folder or OUTPUT_FOLDER
    if not Path(folder).exists():
        return []
    unmarked = find_unmarked_skeletons(folder)
    if unmarked:
        print(f"\n⚠ 警告：{len(unmarked)} 筆骨架尚未標記區段（片段或整段），訓練會被擋下。")
        print(format_unmarked(unmarked))
        print("  → 模式 2 按 u 列出、b 批次勾選整段有效，或逐筆標記片段。")
    return unmarked


if __name__ == "__main__":
    warn_unmarked()
    print("\n==== Cat Skeleton 批次推論/手動標註 ====")
    print("1. 批次推論五個資料夾影片 (YOLO-Pose)  [增量，跳過已有 JSON]")
    print("   → 新抽的骨架尚未標記，需到模式 2 勾選整段或逐段標記後才能訓練")
    print("2. 資料標記管理：尚未標記清單／批次勾選整段有效／逐段標記")
    print("   → 預設只列尚未人工確認，可篩選類別、搜尋檔名；b 開啟批次勾選視窗")
    print("3. 重新推論骨架（新模型），依時間還原既有 action_intervals 與 frame label")
    print("   → YOLO 模型更換或 fps 補償邏輯更新後皆可使用，標注依 timestamp 對應不受幀數變動影響")
    print("4. 檢查有哪些影片／幀/訓練視窗被捨棄或過濾（未進入訓練資料）")
    print("   → 純讀取報告，不修改任何檔案；視窗層級會標出被丟棄的 window 從影片哪個 frame/秒數切出來")
    print("5. 重新推論骨架（新模型），但只針對指定的部分檔案，標籤覆蓋邏輯同模式 3")
    print("   → 只想修正少數幾支骨架抽取結果不佳的影片時使用，不用重跑整個資料集")
    print("6. 標註獨立測試集影片（主要測試資料夾，輸出到獨立的 TEST_OUTPUT_FOLDER）")
    print("   → 進去後再選一次：1.只抽骨架 2.只標註 3.兩者都做；跟訓練資料完全分開路徑")
    print("     （不會誤觸/混進 skeletons/），結束後自動印出報告")
    print("7. 檢視獨立測試集目前的標註結果（純讀取，不開影片視窗，隨時可執行）")
    print("   → 回答「被打上了什麼標籤」：終端列表（含每個區間起訖影片時間，小數1位）")
    print("     + 同步輸出 CSV 到 TEST_OUTPUT_FOLDER")
    mode = input("請選擇模式 (1/2/3/4/5/6/7): ").strip()
    if mode == '1':
        process_all_videos()
        warn_unmarked()
    elif mode == '2':
        manual_action_labeling()
        warn_unmarked()
    elif mode == '3':
        reextract_preserve_labels()
        warn_unmarked()
    elif mode == '4':
        check_discarded_files()
    elif mode == '5':
        reextract_preserve_labels_selected()
        warn_unmarked()
    elif mode == '6':
        label_test_set()
    elif mode == '7':
        review_test_labels()
    else:
        print("✗ 未選擇正確模式，程式結束。")
