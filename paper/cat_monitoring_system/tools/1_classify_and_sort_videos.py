"""
批次影片行為分類與歸檔工具
=======================================================
讀取 SOURCE_FOLDER 底下所有影片，用 YOLO-Pose + ST-GCN 做逐幀行為推論
（與 test_video_inference_ema copy.py 相同的偵測/分類邏輯，含 EMA 平滑），
統計整支影片裡各行為類別被分類到的次數，取次數最多的類別，
把該影片檔案搬進對應的行為資料夾（walk/lick/scratch/shake/stop 五選一）。

單模式、無視窗、背景批次執行：python classify_and_sort_videos.py
"""
import os
import sys
import shutil
import numpy as np
import cv2
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
SOURCE_FOLDER = r"C:\Users\homec\Downloads\istock"  # TODO: 待分類影片所在資料夾（單層，不含子資料夾）

DEST_BASE = r"C:\Users\homec\Downloads\istock\class"  # TODO: 五個行為資料夾將建立於此路徑下
DEST_FOLDERS = {name: rf"{DEST_BASE}\{name}" for name in BEHAVIOR_CLASSES}  # walk/lick/scratch/shake/stop

YOLO_MODEL_PATH = r"C:\AI_Project\cat_pose\v11s_119.pt"
STGCN_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_results\run_105_xy_conf_v_bone_att_on\105_best_model.pth"
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5

STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
_raw_stgcn_mode = os.getenv("STGCN_FEATURE_MODE", "xy")
STGCN_FEATURE_MODE = str(_raw_stgcn_mode).strip().lower()
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
        shutil.move(str(video_path), str(dest_path))
        print(f"  → 歸類為 [{chosen.upper()}]，已搬移至 {dest_path}\n")

    print("✓ 全部影片處理完成。")


if __name__ == "__main__":
    main()
