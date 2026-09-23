"""
批次影片行為分類與歸檔工具
=======================================================
讀取 SOURCE_FOLDER 底下所有影片，用 YOLO-Pose + ST-GCN 做逐幀行為推論
（與 test_video_inference_ema copy.py 相同的偵測/分類邏輯，含 EMA 平滑），
統計整支影片裡各行為類別被分類到的次數，取次數最多的類別，
把該影片檔案搬進對應的行為資料夾（walk/lick/scratch/shake/stop 五選一）。

分類歸檔階段無視窗、背景批次執行：python 1_classify_and_sort_videos.py
歸檔完成後會跳出「批次序號命名」視窗（_rename_gui.py，以獨立子行程啟動，壞掉也不影響分類），列出五個行為資料夾各自的檔案，
可對選定類別的檔案依「<行為>_<序號>」批次改名（起始序號預設接在資料夾內既有最大序號之後，
目標檔名被未選取的檔案佔用時會擋下，不會覆蓋其他來源的影片）；直接關閉視窗＝跳過。
歸檔時若目的地資料夾已有同名檔案，該影片會保留在原資料夾並印出警告，不會覆蓋。
"""
import os
import sys
import json
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import YOLOConfig as _YOLOConfig
from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
    get_in_channels_for_mode,
)
from utils.constants import BEHAVIOR_CLASSES
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

# ==================== 設定 ====================
SOURCE_FOLDER = r"C:\Users\homec\Downloads\再看看"  # TODO: 待分類影片所在資料夾（單層，不含子資料夾）

# 五個行為資料夾建立在「來源資料夾」底下的 class/（跟 1_classify_and_sort_images.py 的 image_sort/ 同一套設計），
# 只需要改上面的 SOURCE_FOLDER；真的想放到別處，把這行改成絕對路徑字串即可。
DEST_BASE = os.path.join(SOURCE_FOLDER, "class")
DEST_FOLDERS = {name: os.path.join(DEST_BASE, name) for name in BEHAVIOR_CLASSES}  # walk/lick/scratch/shake/stop

YOLO_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "yolo_models" / "v11s_150.pt")

# 若設定 YOLO_MODEL_PATH 環境變數，優先使用該模型路徑（覆蓋上面寫死的 YOLO_MODEL_PATH，
# 對應 settings_window.py 的「🧠 模型路徑」欄位）
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()
if _env_yolo_model:
    YOLO_MODEL_PATH = _env_yolo_model

# 相對於這支腳本的位置（paper/tools/ → 專案根目錄 → stgcn_models/），不寫死磁碟機與使用者資料夾
STGCN_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "stgcn_models" / "run_124_xy_conf_v_bone_att_on" / "124_best_model.pth")
import os as _os
_env_stgcn_model = _os.getenv("CAT_MONITORING_STGCN_MODEL", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫；環境變數名同 config.py 的 ModelPaths.STGCN_MODEL
if _env_stgcn_model:
    STGCN_MODEL_PATH = _env_stgcn_model
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = _YOLOConfig.IMAGE_SIZE  # 跟主系統同步（設定視窗 yolo.image_size／環境變數 CAT_MONITORING_YOLO_IMAGE_SIZE，預設 640）
YOLO_CONF_THRESHOLD = 0.5  # YOLO bbox 偵測信心門檻（不是關鍵點 kp 信心門檻）
import os as _os
_env_yolo_conf = _os.getenv("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫這個 bbox 信心門檻；環境變數名同 config.py 的 YOLOConfig.CONFIDENCE_THRESHOLD
if _env_yolo_conf:
    try:
        YOLO_CONF_THRESHOLD = float(_env_yolo_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD={_env_yolo_conf!r} 不是數字，沿用預設 {YOLO_CONF_THRESHOLD}")

STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
STGCN_FEATURE_MODE = "xy"  # checkpoint 讀取失敗時的 fallback 值；正常情況下 load_models() 會依
                            # checkpoint 的 bn_input 通道數自動校正為實際的 feature mode（見下方）
_FEATURE_MODE_MAP = {
    "xyconf": "xy_conf",
    "xyv_conf": "xy_conf_v",
    "xyv_conf_bone": "xy_conf_v_bone",
    "xyv_conf_bone_bone_motion": "xy_conf_v_bone_bmotion",
    "xyv_conf_bone_bmotion": "xy_conf_v_bone_bmotion",
    "xyvconf": "xy_conf_v",
    "xyvconfbone": "xy_conf_v_bone",
    "xyvconfbonebmotion": "xy_conf_v_bone_bmotion",
}
STGCN_FEATURE_MODE = _FEATURE_MODE_MAP.get(STGCN_FEATURE_MODE, STGCN_FEATURE_MODE)

BEHAVIOR_MIN_CONFIDENCE = _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
TARGET_MODEL_FPS = 30.0
ENABLE_FPS_DOWNSAMPLE = True   # 來源 fps 高於 30 時跳幀降採樣，統一模型時基
CLASSIFY_STRIDE = 2            # 每幾個處理幀分類一次
EMA_ALPHA = 1.0                # 須與訓練時 KP_EMA_ALPHA 保持一致
CLASSIFY_COUNT_THRESHOLD = 90  # 任一行為累計分類次數（不需連續）達此值，立刻停止推論並歸檔至該行為

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}

# 批次序號命名視窗是「完全獨立的選用擴充」：以獨立子行程執行 _rename_gui.py，本腳本不 import 它，
# 也不共用任何程式碼。它壞掉（語法錯誤／檔案遺失／例外／當機）都只會讓子行程失敗，這裡只印警告，
# 不影響已完成的分類與歸檔。（另一支 1_classify_and_sort_images.py 有一份刻意獨立維護的相同啟動器，
# 不抽成共用模組，以免兩支腳本又因為共用程式碼而互相牽連。）
_RENAME_GUI_SCRIPT = Path(__file__).resolve().parent / "_rename_gui.py"


def run_optional_rename_window(dest_folders, exts, fresh_files):
    try:
        payload = json.dumps({
            "dest_folders": {name: str(path) for name, path in dest_folders.items()},
            "exts": sorted(exts),
            "fresh_files": [str(p) for p in fresh_files],
        })
        result = subprocess.run([sys.executable, str(_RENAME_GUI_SCRIPT)], input=payload, text=True)
        if result.returncode != 0:
            print(f"⚠ 批次命名視窗異常結束（exit code {result.returncode}），已略過；分類結果不受影響")
    except Exception as exc:  # noqa: BLE001 — 選用功能，任何錯誤都不該影響本腳本
        print(f"⚠ 無法啟動批次命名視窗（{type(exc).__name__}: {exc}），已略過；分類結果不受影響")


def list_videos(folder):
    p = Path(folder)
    return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS)


def load_models():
    """載入 YOLO-Pose 與 ST-GCN，並依 checkpoint 的 bn_input 通道數自動校正 feature_mode。"""
    feature_mode = STGCN_FEATURE_MODE
    in_channels = None
    try:
        ck_channel_map = {2: 'xy', 3: 'xy_conf', 5: 'xy_conf_v', 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}
        import torch
        if os.path.exists(STGCN_MODEL_PATH):
            try:
                ck = torch.load(STGCN_MODEL_PATH, map_location='cpu', weights_only=True)
            except Exception:
                ck = torch.load(STGCN_MODEL_PATH, map_location='cpu')
            state_dict = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
            if isinstance(state_dict, dict) and 'bn_input.weight' in state_dict:
                ck_in_ch = int(state_dict['bn_input.weight'].shape[0])
                expected_ch = get_in_channels_for_mode(feature_mode)
                if ck_in_ch != expected_ch and ck_in_ch in ck_channel_map:
                    feature_mode = ck_channel_map[ck_in_ch]
                    print(f"⚠ checkpoint bn_input channels={ck_in_ch} 與 feature_mode 不符，自動改用 {feature_mode}")
                in_channels = ck_in_ch
    except Exception as e:
        print(f"⚠ 無法從 checkpoint 推斷通道數，改用 feature_mode 預設通道: {e}")

    if in_channels is None:
        in_channels = get_in_channels_for_mode(feature_mode)

    keypoint_detector = KeypointDetector(
        YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD
    )
    behavior_classifier = BehaviorClassifier(
        STGCN_MODEL_PATH, device=INFERENCE_DEVICE, sequence_length=SEQUENCE_LENGTH,
        normalize=STGCN_NORMALIZE, feature_mode=feature_mode, in_channels=in_channels,
    )
    return keypoint_detector, behavior_classifier, feature_mode


def classify_video(video_path, keypoint_detector, behavior_classifier, feature_mode):
    """跑影片直到任一行為累計分類次數達 CLASSIFY_COUNT_THRESHOLD（提前停止）或播完整支影片，
    回傳 (每個行為類別的分類次數 list[len(BEHAVIOR_CLASSES)], 已分類幀數, 提前達標的行為id或None)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ 無法開啟影片: {video_path.name}")
        return None
    keypoint_detector.reset_track()  # 新影片開始，避免延續上一支影片鎖定的貓

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 1:
        source_fps = TARGET_MODEL_FPS
    frame_step = 1
    if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
        frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))

    model_joints = getattr(behavior_classifier.model, 'num_joints', 17)
    keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
    ema_kpts = None
    class_counts = [0] * len(BEHAVIOR_CLASSES)
    sampled_frames = 0
    reached_behavior = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kpts, kpt_conf, _bbox, _conf = keypoint_detector.detect(frame)

        if kpts is not None:
            ema_kpts = kpts.copy() if ema_kpts is None else (EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts)
            kpts = ema_kpts.copy()

            keypoints_buffer.append((kpts, kpt_conf))
            sampled_frames += 1

            if len(keypoints_buffer) >= SEQUENCE_LENGTH and (sampled_frames % CLASSIFY_STRIDE == 0):
                kpts_arr = np.array([item[0] for item in keypoints_buffer])
                conf_arr = np.array([item[1] for item in keypoints_buffer])

                if model_joints < 17:
                    kpts_arr = kpts_arr[:, :model_joints, :]
                    conf_arr = conf_arr[:, :model_joints]

                seq_array = interpolate_missing(kpts_arr, conf_arr, threshold=0.0)
                if STGCN_NORMALIZE:
                    seq_array = flip_normalize(seq_array)
                    seq_array = orientation_normalize(seq_array)
                    seq_array = normalize_skeleton_coords(seq_array)
                seq_features = build_feature_tensor(seq_array, conf_arr, feature_mode)
                pred_id, pred_conf, _probs = behavior_classifier.model.predict(seq_features, precomputed=True)

                if pred_id is not None and pred_conf >= BEHAVIOR_MIN_CONFIDENCE:
                    class_counts[int(pred_id)] += 1
                    if class_counts[int(pred_id)] >= CLASSIFY_COUNT_THRESHOLD:
                        reached_behavior = int(pred_id)
        else:
            ema_kpts = None

        if reached_behavior is not None:
            break

        for _ in range(frame_step - 1):
            if not cap.grab():
                break

    cap.release()
    return class_counts, sampled_frames, reached_behavior


def main():
    src = Path(SOURCE_FOLDER)
    if not src.is_dir():
        print(f"❌ 來源資料夾不存在: {SOURCE_FOLDER}")
        return

    videos = list_videos(src)
    if not videos:
        print(f"❌ 找不到影片: {SOURCE_FOLDER}")
        return

    for name in BEHAVIOR_CLASSES:
        Path(DEST_FOLDERS[name]).mkdir(parents=True, exist_ok=True)

    print(f"待分類影片共 {len(videos)} 部，來源: {SOURCE_FOLDER}")
    print("初始化模型...")
    keypoint_detector, behavior_classifier, feature_mode = load_models()
    print(f"特徵模式: {feature_mode}\n")

    moved_files = []  # 這次剛歸檔進各行為資料夾的檔案，給命名視窗加「＊」標記用
    for idx, video_path in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] {video_path.name}")
        result = classify_video(video_path, keypoint_detector, behavior_classifier, feature_mode)
        if result is None:
            continue
        class_counts, sampled_frames, reached_behavior = result

        counts_str = "  ".join(f"{cls}:{cnt}" for cls, cnt in zip(BEHAVIOR_CLASSES, class_counts))
        print(f"  取樣幀數={sampled_frames}  分類次數 [{counts_str}]")

        if reached_behavior is not None:
            print(f"  ✓ 已累計達 {CLASSIFY_COUNT_THRESHOLD} 次 [{BEHAVIOR_CLASSES[reached_behavior]}]，提前停止推論")
        elif sum(class_counts) == 0:
            print(f"  ⚠ 全片無高信心分類結果，跳過歸檔（保留於原資料夾）")
            continue

        chosen = BEHAVIOR_CLASSES[reached_behavior if reached_behavior is not None else int(np.argmax(class_counts))]
        dest_path = Path(DEST_FOLDERS[chosen]) / video_path.name
        if dest_path.exists():
            # Windows 上 shutil.move 遇到同名目的檔會直接覆蓋（已實測），可能蓋掉其他來源的影片
            print(f"  ⚠ [{chosen.upper()}] 資料夾已有同名檔案 {dest_path.name}，為避免覆蓋，保留於原資料夾\n")
            continue
        shutil.move(str(video_path), str(dest_path))
        moved_files.append(dest_path)
        print(f"  → 歸類為 [{chosen.upper()}]，已搬移至 {dest_path}\n")

    print(f"✓ 全部影片處理完成（本次歸檔 {len(moved_files)} 部）。")
    print("開啟批次序號命名視窗（直接關閉視窗即跳過）...")
    run_optional_rename_window(DEST_FOLDERS, SUPPORTED_VIDEO_EXTS, moved_files)
    print("結束。")


if __name__ == "__main__":
    main()
