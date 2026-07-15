"""
ST-GCN Training Script for Cat Behavior Classification
=======================================================
This script trains a Spatial-Temporal Graph Convolutional Network (ST-GCN)
to classify cat behaviors: walk, lick, scratch, shake, stop

Input: Skeleton JSON files from ./data/skeletons/
Output: Trained model saved to ./models/stgcn_best.pth

Author: Generated for Cat Pose Analysis
Date: 2026-01-29
"""

import os
import json
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from datetime import datetime, timezone
import sys

# 這支腳本現在放在 cat_monitoring_system/tools/ 底下，parent.parent 才是
# cat_monitoring_system/（models/ 等套件所在目錄），跟其餘診斷腳本共用同一套
# sys.path 設定慣例。
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stgcn_model import (
    STGCN,
    interpolate_missing,
    orientation_normalize,
    flip_normalize,
    normalize_skeleton_coords,
    get_in_channels_for_mode,
    build_feature_tensor as shared_build_feature_tensor,
)

# ==================== Path Config（絕對路徑統一於此管理） ====================
# 設定檔絕對路徑集中在此常數；可用 STGCN_CONFIG_PATH 環境變數覆寫
DEFAULT_CONFIG_PATH = r'C:\ai_project\paper\cat_monitoring_system\stgcn_config.yaml'

# ==================== EMA Helper ====================
class ModelEMA:
    """
    Maintains Exponential Moving Average (EMA) of model parameters.
    """
    def __init__(self, model, model_config, decay=0.999):
        self.device = next(model.parameters()).device
        self.ema = self._clone_model(type(model), model_config).to(self.device)
        self.ema.load_state_dict(model.state_dict(), strict=True)
        self.ema.eval()
        self.decay = decay

    def _clone_model(self, model_class, model_config):
        # model_config 必須包含所有 __init__ 參數
        ema_model = model_class(
            num_classes=model_config['num_classes'],
            in_channels=model_config['in_channels'],
            num_joints=model_config['num_joints'],
            spatial_kernel_size=model_config.get('spatial_kernel_size', 3),
            temporal_kernel_size=model_config.get('temporal_kernel_size', 9),
            num_layers=model_config.get('num_layers', 3),
            input_dropout=model_config.get('input_dropout', 0.05),
            block_dropout=model_config.get('block_dropout', 0.15),
            final_dropout=model_config.get('final_dropout', 0.5),
            use_attention=model_config.get('use_attention', True),
            joint_prior_weights=model_config.get('joint_prior_weights', None),
        )
        return ema_model

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach().to(ema_v.device)
            # BatchNorm running stats 與 num_batches_tracked 直接複製，不做 EMA
            # 否則 running_mean/running_var 會嚴重滯後，推論時輸出近均勻分布
            if not torch.is_floating_point(model_v) or k.endswith(('running_mean', 'running_var')):
                ema_v.copy_(model_v)
            else:
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

    def to(self, device):
        self.ema.to(device)
        self.device = device
        return self

# ==================== Configuration ====================
# Load external config file (JSON or YAML). The config file is required;
# the script will raise RuntimeError if it is missing or unreadable.
# To use a custom path, set the STGCN_CONFIG_PATH environment variable.
def _load_external_config():
    config_path = os.getenv('STGCN_CONFIG_PATH', DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        # allow config placed next to this script
        local_path = Path(__file__).parent / 'stgcn_config.yaml'
        if local_path.exists():
            config_path = str(local_path)
        else:
            return None

    try:
        if config_path.lower().endswith(('.yml', '.yaml')):
            try:
                import yaml
            except Exception:
                print(f"⚠ YAML config requested but PyYAML not installed. Skipping {config_path}.")
                return None
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f) or {}
        if not isinstance(loaded, dict):
            print(f"⚠ Config file {config_path} did not contain a mapping/object; ignoring.")
            return None
        print(f"✓ Loaded external config from: {config_path}")
        return loaded
    except Exception as e:
        print(f"⚠ Failed to load config file {config_path}: {e}")
        return None


_external = _load_external_config()
# Enforce strict config-only mode: require an external config file and
# fail fast if missing or incomplete. This removes reliance on script
# defaults or environment variables.
if not _external:
    raise RuntimeError(
        "Missing configuration file: create 'stgcn_config.yaml' next to the script "
        "or set STGCN_CONFIG_PATH to point to a valid YAML/JSON config."
    )

# Use external config exclusively
CONFIG = _external

# Validate required top-level keys are present
required_keys = [
    'SKELETON_DATA_FOLDER', 'RESULTS_FOLDER',
    'STRICT_WINDOW_FILTER', 'KP_EMA_ALPHA',
    'NUM_CLASSES', 'BEHAVIOR_PREFIXES', 'NUM_JOINTS', 'SEQUENCE_LENGTH', 'WINDOW_STRIDE',
    'SPATIAL_KERNEL_SIZE', 'TEMPORAL_KERNEL_SIZE', 'NUM_STGCN_LAYERS',
    'INPUT_DROPOUT', 'BLOCK_DROPOUT', 'FINAL_DROPOUT',
    'FEATURE_MODE', 'RUN_ABLATION_STUDY', 'ABLATION_MODES',
    'SPATIAL_ROTATE_DEG', 'SPATIAL_SCALE_MIN', 'SPATIAL_SCALE_MAX', 'SPATIAL_TRANSLATE_STD',
    'SPATIAL_CROP_PROB', 'SPATIAL_OCCLUSION_PROB', 'SPATIAL_OCCLUSION_MAX_JOINTS',
    'BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY',
    'EARLY_STOP_PATIENCE', 'TRAIN_TEST_SPLIT', 'RANDOM_SEED', 'USE_ATTENTION', 'USE_EMA_FOR_EVAL'
]
missing = [k for k in required_keys if k not in CONFIG]
if missing:
    raise RuntimeError(f"Configuration file is missing required keys: {missing}")

# Unpack to module-level constants (keeps backward compatibility with code below)
SKELETON_DATA_FOLDER = CONFIG['SKELETON_DATA_FOLDER']
RESULTS_FOLDER = CONFIG['RESULTS_FOLDER']
STRICT_WINDOW_FILTER = CONFIG['STRICT_WINDOW_FILTER']
MAX_NO_DETECT_FRAMES = int(CONFIG.get('MAX_NO_DETECT_FRAMES', 2))
KP_EMA_ALPHA = CONFIG['KP_EMA_ALPHA']
NUM_CLASSES = CONFIG['NUM_CLASSES']
BEHAVIOR_PREFIXES = CONFIG['BEHAVIOR_PREFIXES']
NUM_JOINTS = CONFIG['NUM_JOINTS']
# 跟 eval_pose_models.py 的 KEYPOINT_NAMES 同一份骨架定義（joint_id 0~16 依序對應）
KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back",
    "Hip", "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]
# JSON 骨架檔永遠有 17 個關節；NUM_JOINTS 控制實際送入模型的關節數
# 支援兩種模式：17（完整）或 14（忽略 tail_base/tail_mid/tail_tip）
_JSON_NUM_JOINTS = 17
assert NUM_JOINTS in (14, 17), (
    f"NUM_JOINTS 僅支援 17（完整骨架）或 14（忽略尾巴三點），目前值：{NUM_JOINTS}"
)
SEQUENCE_LENGTH = CONFIG['SEQUENCE_LENGTH']
WINDOW_STRIDE   = CONFIG['WINDOW_STRIDE']
SPATIAL_KERNEL_SIZE = CONFIG['SPATIAL_KERNEL_SIZE']
TEMPORAL_KERNEL_SIZE = CONFIG['TEMPORAL_KERNEL_SIZE']
NUM_STGCN_LAYERS = CONFIG['NUM_STGCN_LAYERS']
INPUT_DROPOUT = CONFIG['INPUT_DROPOUT']
BLOCK_DROPOUT = CONFIG['BLOCK_DROPOUT']
FINAL_DROPOUT = CONFIG['FINAL_DROPOUT']
FEATURE_MODE = CONFIG['FEATURE_MODE']
RUN_ABLATION_STUDY = CONFIG['RUN_ABLATION_STUDY']
ABLATION_MODES = CONFIG['ABLATION_MODES']
RUN_SEQLEN_ABLATION = CONFIG.get('RUN_SEQLEN_ABLATION', False)
ABLATION_SEQLENS    = CONFIG.get('ABLATION_SEQLENS', [16, 32])
SPATIAL_ROTATE_DEG = CONFIG['SPATIAL_ROTATE_DEG']
SPATIAL_SCALE_MIN = CONFIG['SPATIAL_SCALE_MIN']
SPATIAL_SCALE_MAX = CONFIG['SPATIAL_SCALE_MAX']
SPATIAL_TRANSLATE_STD = CONFIG['SPATIAL_TRANSLATE_STD']
SPATIAL_CROP_PROB = CONFIG['SPATIAL_CROP_PROB']
SPATIAL_OCCLUSION_PROB = CONFIG['SPATIAL_OCCLUSION_PROB']
SPATIAL_OCCLUSION_MAX_JOINTS = CONFIG['SPATIAL_OCCLUSION_MAX_JOINTS']
BATCH_SIZE = CONFIG['BATCH_SIZE']
NUM_EPOCHS = CONFIG['NUM_EPOCHS']
LEARNING_RATE = CONFIG['LEARNING_RATE']
WEIGHT_DECAY = CONFIG['WEIGHT_DECAY']
EARLY_STOP_PATIENCE = CONFIG['EARLY_STOP_PATIENCE']
TRAIN_TEST_SPLIT = CONFIG['TRAIN_TEST_SPLIT']
RANDOM_SEED = CONFIG['RANDOM_SEED']
USE_ATTENTION = CONFIG['USE_ATTENTION']
USE_EMA_FOR_EVAL = CONFIG.get('USE_EMA_FOR_EVAL', False)
OPTIMIZER = CONFIG.get('OPTIMIZER', 'adam')
RUN_KP_EMA_ABLATION    = CONFIG.get('RUN_KP_EMA_ABLATION', False)
ABLATION_KP_EMA_ALPHAS = CONFIG.get('ABLATION_KP_EMA_ALPHAS', [1.0, 0.9, 0.7, 0.5])
RUN_REG_ABLATION       = CONFIG.get('RUN_REG_ABLATION', False)
REG_ABLATION_CONFIGS   = CONFIG.get('REG_ABLATION_CONFIGS', None)
USE_CB_LOSS = CONFIG.get('USE_CB_LOSS', False)
CB_BETA     = CONFIG.get('CB_BETA', 0.999)
USE_JOINT_PRIOR_WEIGHTS = CONFIG.get('USE_JOINT_PRIOR_WEIGHTS', False)
JOINT_PRIOR_WEIGHTS_CFG = CONFIG.get('JOINT_PRIOR_WEIGHTS', {}) or {}


def _build_joint_prior_weights():
    """把 JOINT_PRIOR_WEIGHTS（{關節名稱: 權重}）轉成長度 NUM_JOINTS 的 tensor，
    預設全部 1.0（不影響行為），只有 config 裡指名的關節會覆寫成指定權重。
    USE_JOINT_PRIOR_WEIGHTS=false 時回傳 None（STGCN 內部視同全 1.0）。"""
    if not USE_JOINT_PRIOR_WEIGHTS:
        return None
    weights = [1.0] * NUM_JOINTS
    name_to_idx = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    for name, w in JOINT_PRIOR_WEIGHTS_CFG.items():
        idx = name_to_idx.get(name)
        if idx is None or idx >= NUM_JOINTS:
            print(f"  ⚠ JOINT_PRIOR_WEIGHTS: 找不到關節 '{name}'（或已被 NUM_JOINTS={NUM_JOINTS} 消融排除），已略過")
            continue
        weights[idx] = float(w)
    print(f"✓ Joint prior weights 已啟用: "
          f"{ {KEYPOINT_NAMES[i]: w for i, w in enumerate(weights) if w != 1.0} }")
    return weights

# 預設的 dropout / label smoothing / batch / learning rate 消融網格：
#   baseline_no_reg — 目前 yaml 的設定（全部關閉，用來對照現況）
#   light_reg       — yaml 裡 "# init:" 註解建議的原始預設值
#   recommended     — 針對小資料集 + WeightedRandomSampler 高變異梯度的建議組合
# 可在 yaml 用 REG_ABLATION_CONFIGS 覆寫（同樣格式的 list of dict）。
DEFAULT_REG_ABLATION_CONFIGS = [
    {'name': 'baseline_no_reg', 'batch_size': 8,  'learning_rate': 0.00005,
     'input_dropout': 0.0,  'block_dropout': 0.0, 'final_dropout': 0.0, 'label_smoothing': 0.0},
    {'name': 'light_reg',       'batch_size': 8,  'learning_rate': 0.00005,
     'input_dropout': 0.02, 'block_dropout': 0.1, 'final_dropout': 0.2, 'label_smoothing': 0.01},
    {'name': 'recommended',     'batch_size': 16, 'learning_rate': 0.0001,
     'input_dropout': 0.02, 'block_dropout': 0.1, 'final_dropout': 0.3, 'label_smoothing': 0.05},
]

# Device configuration (runtime detection remains)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Setup ====================
def setup_directories():
    """Create necessary directories"""
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created")


def _next_run_number(results_folder: str) -> int:
    """掃描 results_folder 下的 run_NNN_* 目錄，回傳下一個可用編號。"""
    import re
    p = Path(results_folder)
    if not p.exists():
        return 1
    pat = re.compile(r'^run_(\d+)')
    max_num = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def _format_duration(seconds: float) -> str:
    """將秒數轉為易讀的時分秒字串，例如 1h 23m 45s。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def get_unique_path(path):
    """回傳不衝突的路徑：若目標已存在，自動補 _001、_002 … 後綴。"""
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ==================== Dataset Class ====================
def temporal_augment(sequence, conf=None, max_shift=2, max_drop=2, jitter_std=0.01):
    T = sequence.shape[0]
    seq = sequence.copy()
    conf_aug = conf.copy() if conf is not None else None
    # random shift
    shift = np.random.randint(-max_shift, max_shift+1)
    if shift > 0:
        seq = np.pad(seq, ((shift,0),(0,0),(0,0)), mode='edge')[:T]
        if conf_aug is not None:
            conf_aug = np.pad(conf_aug, ((shift,0),(0,0)), mode='edge')[:T]
    elif shift < 0:
        seq = np.pad(seq, ((0,-shift),(0,0),(0,0)), mode='edge')[-shift:]
        if conf_aug is not None:
            conf_aug = np.pad(conf_aug, ((0,-shift),(0,0)), mode='edge')[-shift:]
    # random drop
    drop = np.random.randint(0, max_drop+1)
    if drop > 0 and T-drop > 0:
        idx = np.sort(np.random.choice(T, T-drop, replace=False))
        seq = seq[idx]
        seq = np.pad(seq, ((0,T-seq.shape[0]),(0,0),(0,0)), mode='edge')
        if conf_aug is not None:
            conf_aug = conf_aug[idx]
            conf_aug = np.pad(conf_aug, ((0,T-conf_aug.shape[0]),(0,0)), mode='edge')
    # jitter
    seq += np.random.normal(0, jitter_std, seq.shape)
    if conf_aug is None:
        return seq
    return seq, conf_aug


def spatial_augment(
    sequence,
    conf=None,
    rotate_deg=SPATIAL_ROTATE_DEG,
    scale_min=SPATIAL_SCALE_MIN,
    scale_max=SPATIAL_SCALE_MAX,
    translate_std=SPATIAL_TRANSLATE_STD,
    crop_prob=SPATIAL_CROP_PROB,
    occlusion_prob=SPATIAL_OCCLUSION_PROB,
    occlusion_max_joints=SPATIAL_OCCLUSION_MAX_JOINTS,
):
    """Apply conservative spatial augmentation on normalized cat poses."""
    seq = sequence.copy()
    conf_aug = conf.copy() if conf is not None else None

    if seq.size == 0:
        return (seq, conf_aug) if conf_aug is not None else seq

    # Random scale + rotation around the body center (normalized coordinates).
    angle = np.deg2rad(np.random.uniform(-rotate_deg, rotate_deg))
    scale = np.random.uniform(scale_min, scale_max)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=seq.dtype)
    seq = (seq * scale) @ rotation.T

    # Mild translation noise.
    seq += np.random.normal(0.0, translate_std, size=(1, 1, 2)).astype(seq.dtype)

    # Random crop: shrink the visible bounding box slightly and drop points outside.
    if np.random.rand() < crop_prob:
        if conf_aug is not None:
            visible = conf_aug > 0
        else:
            visible = np.ones(seq.shape[:2], dtype=bool)
        if np.any(visible):
            visible_points = seq[visible]
            min_xy = visible_points.min(axis=0)
            max_xy = visible_points.max(axis=0)
            bbox_center = (min_xy + max_xy) / 2.0
            bbox_half = np.maximum((max_xy - min_xy) / 2.0, 1e-3)
            keep_ratio = np.random.uniform(0.82, 0.98)
            keep_half = bbox_half * keep_ratio
            shift = np.random.uniform(-0.10, 0.10, size=2).astype(seq.dtype) * bbox_half
            crop_center = bbox_center + shift
            outside = np.any(np.abs(seq - crop_center[None, None, :]) > keep_half[None, None, :], axis=-1)
            seq[outside] = 0.0
            if conf_aug is not None:
                conf_aug[outside] = 0.0

    # Random joint occlusion across the whole clip.
    if np.random.rand() < occlusion_prob:
        joint_count = seq.shape[1]
        max_joints = max(1, min(occlusion_max_joints, joint_count))
        num_occluded = np.random.randint(1, max_joints + 1)
        occluded_joints = np.random.choice(joint_count, size=num_occluded, replace=False)
        seq[:, occluded_joints, :] = 0.0
        if conf_aug is not None:
            conf_aug[:, occluded_joints] = 0.0

    return (seq, conf_aug) if conf_aug is not None else seq


# 使用共享的特徵構建函數（從 models.stgcn_model 導入）
def build_feature_tensor(sequence_xy, conf_seq, feature_mode):
    return shared_build_feature_tensor(sequence_xy, conf_seq, feature_mode)

class CatSkeletonDataset(Dataset):
    """
    Dataset class for loading skeleton sequences
    """
    
    def __init__(self, skeleton_folder, sequence_length=32,
                 num_joints=17, augment=False, feature_mode="xy_conf_v_bone",
                 window_stride=0, kp_ema_alpha=None):
        self.skeleton_folder = Path(skeleton_folder)
        self.sequence_length = sequence_length
        self.num_joints = num_joints
        self.augment = augment
        self.feature_mode = feature_mode
        self.window_stride = window_stride
        self.kp_ema_alpha = kp_ema_alpha if kp_ema_alpha is not None else KP_EMA_ALPHA
        # idx_to_label: map numeric label -> behaviour name (derived from CONFIG)
        self.idx_to_label = {v: k for k, v in BEHAVIOR_PREFIXES.items()}
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences")
        self._print_class_distribution()
    
    def _load_sequences(self):
        """以逐幀標籤（手動/整段標記）載入並切割成固定長度序列。"""
        from collections import Counter

        sequences = []
        json_files = list(self.skeleton_folder.glob("*.json"))
        # Use centralized behavior prefixes mapping from CONFIG
        name_to_idx = BEHAVIOR_PREFIXES

        # 診斷用：記錄各種「靜默跳過」的情況，避免資料悄悄漏掉卻毫無提示
        skipped_no_label = []      # 影片缺少逐幀 label
        skipped_too_short = []     # 影片總幀數 < sequence_length
        unknown_label_counter = Counter()  # best_label 不在 BEHAVIOR_PREFIXES 的視窗

        for json_file in tqdm(json_files, desc="Loading sequences"):
            video_id = json_file.stem

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            frames = data['frames']

            if not (bool(frames) and 'label' in frames[0]):
                skipped_no_label.append(video_id)
                continue  # 無逐幀標籤，略過

            keypoint_frames   = []
            frame_labels_list = []
            frame_detected    = []   # True = 該幀有 bbox（YOLO 偵測到貓）
            for frame in frames:
                kpts_list = frame.get('keypoints', [])
                if len(kpts_list) == _JSON_NUM_JOINTS:
                    coords = np.array([[kpt['x'], kpt['y']] for kpt in kpts_list])
                    conf   = np.array([kpt.get('conf', 1.0) for kpt in kpts_list])
                else:
                    coords = np.zeros((_JSON_NUM_JOINTS, 2), dtype=np.float32)
                    conf   = np.zeros((_JSON_NUM_JOINTS,),   dtype=np.float32)
                keypoint_frames.append(coords)
                frame_labels_list.append(frame.get('label', 'unannotated'))
                frame_detected.append(frame.get('bbox') is not None)

            keypoint_frames = np.array(keypoint_frames)
            confs = np.array([
                np.array([kpt.get('conf', 1.0) for kpt in frame.get('keypoints', [])])
                if len(frame.get('keypoints', [])) == _JSON_NUM_JOINTS
                else np.zeros((_JSON_NUM_JOINTS,))
                for frame in frames
            ])
            keypoint_frames = interpolate_missing(keypoint_frames, confs)

            # EMA 平滑：在插值補全後、滑動切窗前對整段影片套用，與推論行為一致
            # 必須在切窗前套，否則跨窗的平滑狀態不連貫
            _ema = self.kp_ema_alpha
            if _ema is not None and 0.0 < _ema < 1.0:
                for t in range(1, len(keypoint_frames)):
                    keypoint_frames[t] = (
                        _ema * keypoint_frames[t]
                        + (1.0 - _ema) * keypoint_frames[t - 1]
                    )

            if len(keypoint_frames) < self.sequence_length:
                skipped_too_short.append((video_id, len(keypoint_frames)))
                continue

            stride = self.window_stride
            for start_idx in range(0, len(keypoint_frames) - self.sequence_length + 1, stride):
                sequence        = keypoint_frames[start_idx:start_idx + self.sequence_length]
                window_labels   = frame_labels_list[start_idx:start_idx + self.sequence_length]
                window_detected = frame_detected[start_idx:start_idx + self.sequence_length]

                if STRICT_WINDOW_FILTER:
                    if 'unannotated' in window_labels:
                        continue
                    label_counts = Counter(window_labels)
                    best_label, best_count = label_counts.most_common(1)[0]
                else:
                    annotated_labels = [lbl for lbl in window_labels if lbl != 'unannotated']
                    if not annotated_labels:
                        continue
                    label_counts = Counter(annotated_labels)
                    best_label, _ = label_counts.most_common(1)[0]

                # bbox 缺失過濾：與 STRICT_WINDOW_FILTER 無關，永遠套用
                if window_detected.count(False) > MAX_NO_DETECT_FRAMES:
                    continue

                label_idx = name_to_idx.get(best_label)
                if label_idx is None:
                    unknown_label_counter[best_label] += 1
                    continue

                end_idx = start_idx + self.sequence_length - 1
                sequences.append({
                    'video_id': video_id,
                    'sequence': np.array(sequence),
                    'conf_sequence': np.array(confs[start_idx:start_idx + self.sequence_length]),
                    'label':    label_idx,
                    # 供誤判診斷（見 diagnose_confusion_pair）回推這個 window 對應影片的哪個時間點
                    'start_idx':  start_idx,
                    'start_time': frames[start_idx].get('timestamp'),
                    'end_time':   frames[end_idx].get('timestamp'),
                })

        # ── 資料載入診斷：把靜默跳過的情況印出來，才能核對「有幾支 json → 實際用了幾支」 ──
        contributing_videos = {s['video_id'] for s in sequences}
        all_video_ids = {jf.stem for jf in json_files}
        zero_seq_videos = sorted(
            all_video_ids - set(skipped_no_label) - {v for v, _ in skipped_too_short} - contributing_videos
        )

        print(f"\n[資料載入診斷] JSON 檔案總數: {len(json_files)}，實際貢獻序列的影片數: {len(contributing_videos)}")
        if skipped_no_label:
            print(f"  ⚠ {len(skipped_no_label)} 支影片缺少逐幀 label，已跳過: {', '.join(skipped_no_label)}")
        if skipped_too_short:
            detail = ', '.join(f"{vid}({n}幀)" for vid, n in skipped_too_short)
            print(f"  ⚠ {len(skipped_too_short)} 支影片總幀數 < sequence_length={self.sequence_length}，已跳過: {detail}")
        if unknown_label_counter:
            detail = ', '.join(f"{lbl}x{cnt}" for lbl, cnt in unknown_label_counter.items())
            print(f"  ⚠ {sum(unknown_label_counter.values())} 個視窗因 best_label 不在 BEHAVIOR_PREFIXES 而跳過: {detail}")
        if zero_seq_videos:
            print(f"  ⚠ {len(zero_seq_videos)} 支影片有逐幀 label 但切窗後 0 個有效序列"
                  f"（可能整支都是 unannotated、bbox 缺失過多，或標註區間短於 sequence_length）: {', '.join(zero_seq_videos)}")

        return sequences
    
    def _print_class_distribution(self):
        """Print distribution of classes in dataset"""
        label_counts = {}
        for seq in self.sequences:
            label_idx = seq['label']
            label_name = self.idx_to_label.get(label_idx, str(label_idx))
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        print("\nClass distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a single sequence
        
        Returns:
            tuple: (sequence_tensor, label)
                sequence_tensor: Shape (C, T, V) where C=channels, T=time, V=vertices/joints
                label: Class index
        """
        item = self.sequences[idx]
        sequence = item['sequence']      # Shape: (T, 17, 2)
        conf_seq = item['conf_sequence'] # Shape: (T, 17)
        label = item['label']

        # 消融：忽略尾巴三點時，切掉 joint 14/15/16（tail_base/tail_mid/tail_tip）
        if self.num_joints < _JSON_NUM_JOINTS:
            sequence = sequence[:, :self.num_joints, :]
            conf_seq = conf_seq[:, :self.num_joints]

        # Normalize: flip → orientation → center/scale (shared with inference)
        # flip 先於 orientation：原始座標下 walk 貓的 nose_x/tail_x 差距最大，翻轉決策穩定
        sequence = flip_normalize(sequence)
        sequence = orientation_normalize(sequence)
        sequence = normalize_skeleton_coords(sequence)
        # Temporal augmentation (training only)
        if self.augment:
            sequence, conf_seq = spatial_augment(sequence, conf=conf_seq)
            sequence, conf_seq = temporal_augment(sequence, conf=conf_seq)
        # Build feature channels according to selected mode
        sequence = build_feature_tensor(sequence, conf_seq, self.feature_mode)
        # Convert to tensor and transpose to (C, T, V)
        sequence_tensor = torch.FloatTensor(sequence).permute(2, 0, 1)
        return sequence_tensor, label


# ==================== Training Functions ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")

    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        # (debug first-batch prints removed)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, desc="Val"):
    """
    Validate the model

    Args:
        desc: 純顯示用標籤（例如 "Val" 或 "Train(eval)"），用來在同一個 epoch 內
              區分驗證集跟「未加權訓練集」的印出訊息，避免混淆兩者的 per-class 統計。

    Returns:
        tuple: (average_loss, accuracy, predictions, labels)
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc=f"Validating [{desc}]"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 每類別準確率（類別名稱由 CONFIG['BEHAVIOR_PREFIXES'] 決定）
    from collections import Counter
    # derive ordered label names by index
    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    per_class_correct = Counter()
    per_class_total = Counter()
    for pred, true in zip(all_preds, all_labels):
        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1
    per_class_acc = {
        label_names[c]: f"{per_class_correct[c]}/{per_class_total[c]} "
                        f"({100*per_class_correct[c]/per_class_total[c]:.0f}%)"
        if per_class_total[c] > 0 else "N/A"
        for c in range(NUM_CLASSES)
    }
    print(f"  [{desc}] Per-class acc: {per_class_acc}")
    # Predicted distribution for debugging class collapse
    try:
        from collections import Counter
        pred_counts = Counter(all_preds)
        distrib = {int(k): int(pred_counts.get(k, 0)) for k in range(NUM_CLASSES)}
        print(f"  [{desc}] Predicted distribution: {distrib}")
    except Exception:
        pass

    from sklearn.metrics import precision_recall_fscore_support
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return avg_loss, accuracy, macro_f1, all_preds, all_labels, per_class_f1


# ==================== Main Training Loop ====================
def split_train_val_indices(full_dataset, verbose=True):
    """
    影片級切分（防止滑動窗 data leakage），從 train_model() 抽出來獨立成函式，
    讓不需要重新訓練、只想針對「某個已經訓練好的 checkpoint」跑驗證集診斷
    （例如 diagnose_keypoint_motion）的獨立腳本可以重用同一套切分邏輯，
    不用重複貼一份容易失去同步的程式碼。邏輯與參數（RANDOM_SEED/TRAIN_TEST_SPLIT）
    跟 train_model() 完全相同，只要 full_dataset 的載入參數一致，就能重現同一份切分。

    Returns: (train_indices, val_indices)
    """
    from collections import Counter
    video_label_map = {}
    for seq in full_dataset.sequences:
        vid = seq['video_id']
        video_label_map.setdefault(vid, []).append(seq['label'])

    video_ids = list(video_label_map.keys())

    def label_from_video_id(video_id):
        vid_lower = video_id.lower()
        for prefix, lbl in BEHAVIOR_PREFIXES.items():
            if vid_lower.startswith(prefix):
                return lbl
        # fallback：序列多數決（適用前綴不符的特殊命名）
        return Counter(video_label_map[video_id]).most_common(1)[0][0]

    video_labels = [label_from_video_id(v) for v in video_ids]
    video_label_lookup = {vid: lbl for vid, lbl in zip(video_ids, video_labels)}
    # derive ordered label names by index for logging and distribution prints
    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]

    def print_video_distribution(title, vids):
        from collections import Counter
        counter = Counter(video_label_lookup[v] for v in vids)
        print(f"\n{title} video distribution:")
        for cls_idx, cls_name in enumerate(label_names):
            print(f"  {cls_name}: {counter.get(cls_idx, 0)} videos")

    # 影片數太少時 stratify 會失敗（val 影片數 < 類別數），自動退回 random split
    n_val_vids = max(1, int(len(video_ids) * TRAIN_TEST_SPLIT))
    num_unique_labels = len(set(video_labels))
    use_stratify = n_val_vids >= num_unique_labels
    if not use_stratify and verbose:
        print(f"  ⚠ 影片數不足以 stratify（val={n_val_vids} < classes={num_unique_labels}），改用 random split")
    train_vids, val_vids = train_test_split(
        video_ids,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=video_labels if use_stratify else None
    )

    # 確保驗證集盡量覆蓋所有可切分類別：若某類別有 >=2 支影片且 val 缺席，
    # 從 train 移 1 支該類別影片到 val（必要時再移回 1 支其他類別維持大小）。
    rng = np.random.default_rng(RANDOM_SEED)
    desired_val_size = max(1, int(len(video_ids) * TRAIN_TEST_SPLIT))
    train_vids = list(train_vids)
    val_vids = list(val_vids)

    total_label_counts = Counter(video_labels)
    val_label_counts = Counter(video_label_lookup[v] for v in val_vids)

    for cls in sorted(set(video_labels)):
        if total_label_counts[cls] < 2:
            continue
        if val_label_counts.get(cls, 0) > 0:
            continue

        candidates = [v for v in train_vids if video_label_lookup[v] == cls]
        if not candidates:
            continue

        moved_to_val = candidates[int(rng.integers(len(candidates)))]
        train_vids.remove(moved_to_val)
        val_vids.append(moved_to_val)
        val_label_counts[cls] += 1

        # 盡量維持 val 目標大小
        if len(val_vids) > desired_val_size:
            removable = [
                v for v in val_vids
                if v != moved_to_val
                and val_label_counts[video_label_lookup[v]] > 1
            ]
            if removable:
                moved_back = removable[int(rng.integers(len(removable)))]
                val_vids.remove(moved_back)
                train_vids.append(moved_back)
                val_label_counts[video_label_lookup[moved_back]] -= 1

    train_vids_set = set(train_vids)
    val_vids_set   = set(val_vids)

    train_indices = [i for i, s in enumerate(full_dataset.sequences)
                     if s['video_id'] in train_vids_set]
    val_indices   = [i for i, s in enumerate(full_dataset.sequences)
                     if s['video_id'] in val_vids_set]

    if verbose:
        print(f"  Videos → train: {len(train_vids)}, val: {len(val_vids)}")
        print(f"  Sequences → train: {len(train_indices)}, val: {len(val_indices)}")
        print_video_distribution("Train", train_vids)
        print_video_distribution("Validation", val_vids)

    return train_indices, val_indices


def train_model(feature_mode=FEATURE_MODE, run_name=None, run_number=None,
                shared_models_dir=None, kp_ema_alpha=None, seq_len=None,
                batch_size=None, learning_rate=None,
                input_dropout=None, block_dropout=None, final_dropout=None,
                label_smoothing=None):
    """Main training function.

    batch_size / learning_rate / input_dropout / block_dropout / final_dropout /
    label_smoothing 皆可覆寫 CONFIG 預設值（None 則沿用 yaml 設定），供
    run_regularization_ablation() 做網格消融比較用，不影響單次訓練的既有行為。
    """
    in_channels = get_in_channels_for_mode(feature_mode)
    eff_alpha = kp_ema_alpha if kp_ema_alpha is not None else KP_EMA_ALPHA
    alpha_tag = f"_ema{eff_alpha:.2f}" if (eff_alpha is not None and eff_alpha < 1.0) else ""

    # Sequence-length–dependent parameters (auto-adjusted when seq_len differs from config)
    eff_seq_len       = int(seq_len) if seq_len is not None else SEQUENCE_LENGTH
    stride_ratio      = WINDOW_STRIDE / SEQUENCE_LENGTH          # preserve configured overlap ratio
    eff_window_stride = max(1, int(eff_seq_len * stride_ratio))
    eff_batch_size    = int(batch_size) if batch_size is not None else BATCH_SIZE
    seq_tag           = f"_T{eff_seq_len}" if eff_seq_len != SEQUENCE_LENGTH else ""

    # Regularization / optimizer overrides（用於 run_regularization_ablation）
    eff_lr              = float(learning_rate)   if learning_rate   is not None else LEARNING_RATE
    eff_input_dropout    = float(input_dropout)   if input_dropout   is not None else INPUT_DROPOUT
    eff_block_dropout    = float(block_dropout)   if block_dropout   is not None else BLOCK_DROPOUT
    eff_final_dropout    = float(final_dropout)   if final_dropout   is not None else FINAL_DROPOUT
    eff_label_smoothing  = float(label_smoothing) if label_smoothing is not None else float(CONFIG.get('LABEL_SMOOTHING', 0.0))

    # Fix random seeds for more stable runs (does not guarantee identical across GPUs)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = False

    # Attention 只由 CONFIG['USE_ATTENTION'] 控制，不再讀取環境變數
    use_attention = bool(USE_ATTENTION)
    att_suffix = "att_on" if use_attention else "att_off"

    # 消融研究傳入 run_number 讓同批所有模式共用同一組號；單次訓練自動取下一個
    run_num = run_number if run_number is not None else _next_run_number(RESULTS_FOLDER)
    run_tag = f"{run_num:03d}"
    if run_name:
        run_suffix = f"{run_tag}_{run_name}{alpha_tag}_{att_suffix}"
    else:
        run_suffix = f"{run_tag}_{feature_mode}{alpha_tag}{seq_tag}_{att_suffix}"
    print(f"✓ Run #{run_tag}  ({run_suffix})  KP_EMA_ALPHA={eff_alpha}  T={eff_seq_len}  stride={eff_window_stride}  batch={eff_batch_size}  "
          f"lr={eff_lr}  input_dropout={eff_input_dropout}  block_dropout={eff_block_dropout}  final_dropout={eff_final_dropout}  "
          f"label_smoothing={eff_label_smoothing}")

    run_results_dir = os.path.join(RESULTS_FOLDER, f"run_{run_suffix}")

    # Setup directories
    setup_directories()
    Path(run_results_dir).mkdir(parents=True, exist_ok=True)

    # 每次訓練輸出資料夾都放一份參數設定檔（有效值 + 完整原始 yaml），
    # 之後回頭比較不同 run 時不用再去猜當時到底用了什麼設定。
    # 寫在最前面（資料載入/訓練開始前），即使中途失敗也留得下這份記錄。
    params_snapshot_path = os.path.join(run_results_dir, 'params_snapshot.json')
    params_snapshot = {
        'run_suffix': run_suffix,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'effective_params': {
            'feature_mode': feature_mode,
            'in_channels': in_channels,
            'num_classes': NUM_CLASSES,
            'num_joints': NUM_JOINTS,
            'use_attention': use_attention,
            'kp_ema_alpha': eff_alpha,
            'sequence_length': eff_seq_len,
            'window_stride': eff_window_stride,
            'batch_size': eff_batch_size,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': eff_lr,
            'optimizer': OPTIMIZER,
            'weight_decay': WEIGHT_DECAY,
            'input_dropout': eff_input_dropout,
            'block_dropout': eff_block_dropout,
            'final_dropout': eff_final_dropout,
            'label_smoothing': eff_label_smoothing,
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'train_test_split': TRAIN_TEST_SPLIT,
            'random_seed': RANDOM_SEED,
            'use_ema_for_eval': bool(USE_EMA_FOR_EVAL),
            'use_weighted_sampler': bool(CONFIG.get('USE_WEIGHTED_SAMPLER', True)),
            'strict_window_filter': bool(STRICT_WINDOW_FILTER),
            'max_no_detect_frames': int(CONFIG.get('MAX_NO_DETECT_FRAMES', 2)),
        },
        # 完整原始 yaml 快照：含所有沒被上面 effective_params 覆寫的欄位
        # （spatial augmentation、model topology 等），確保可完整還原這次跑的設定。
        'full_config_snapshot': CONFIG,
    }
    try:
        with open(params_snapshot_path, 'w', encoding='utf-8') as pf:
            json.dump(params_snapshot, pf, ensure_ascii=False, indent=2)
        print(f"✓ Params snapshot saved to: {params_snapshot_path}")
    except Exception as e:
        print(f"⚠ Failed to write params snapshot: {e}")

    # 消融研究：模型權重統一存至共用資料夾，以特徵名區分檔名
    # 單次訓練：存在自己的 run 資料夾內
    if shared_models_dir:
        # seqlen/feature ablation: use run_name when it differs from feature_mode (e.g. "xy_conf_v_bone_T16")
        if run_name and run_name != feature_mode:
            name_part = f"{run_name}{alpha_tag}"
        else:
            name_part = f"{feature_mode}{alpha_tag}{seq_tag}"
        model_filename = f"{run_tag}_{name_part}_{att_suffix}.pth"
        run_model_path = str(get_unique_path(Path(shared_models_dir) / model_filename))
    else:
        run_model_path = str(get_unique_path(Path(run_results_dir) / f"{run_tag}_best_model.pth"))

    # Load dataset (no augmentation)
    print("\nLoading dataset...")
    full_dataset = CatSkeletonDataset(
        SKELETON_DATA_FOLDER,
        sequence_length=eff_seq_len,
        num_joints=NUM_JOINTS,
        augment=False,
        feature_mode=feature_mode,
        window_stride=eff_window_stride,
        kp_ema_alpha=eff_alpha,
    )

    if len(full_dataset) == 0:
        print("✗ No data loaded. Please check your skeleton files and labels.")
        return

    # Prepare run logging (single JSON file with meta + epochs list)
    run_log_path = os.path.join(run_results_dir, 'run_log.json')
    # Write initial run metadata (config snapshot + class distributions)
    initial_meta = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'feature_mode': feature_mode,
        'in_channels': in_channels,
        'num_classes': NUM_CLASSES,
        'num_joints': NUM_JOINTS,
        'random_seed': RANDOM_SEED,
        'use_attention': use_attention,
        'use_ema_for_eval': bool(USE_EMA_FOR_EVAL),
        'kp_ema_alpha': eff_alpha,
        'training_params': {
            'batch_size': eff_batch_size,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': eff_lr,
            'optimizer': OPTIMIZER,
            'weight_decay': WEIGHT_DECAY,
            'input_dropout': eff_input_dropout,
            'block_dropout': eff_block_dropout,
            'final_dropout': eff_final_dropout,
            'label_smoothing': eff_label_smoothing,
        },
        'class_prefixes': BEHAVIOR_PREFIXES,
    }
    run_log_data = {
        'meta': initial_meta,
        'epochs': []
    }
    # write initial file
    try:
        with open(run_log_path, 'w', encoding='utf-8') as lf:
            json.dump(run_log_data, lf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ Failed to write initial run log: {e}")

    # 影片級切分（防止滑動窗 data leakage），邏輯抽到 split_train_val_indices()
    train_indices, val_indices = split_train_val_indices(full_dataset)
    # split_train_val_indices() 只回傳序列索引、不回傳影片 id 清單本身，
    # 這裡從切分結果反推「不重複影片數」給下面的 run_log/history 紀錄用。
    train_video_count = len({full_dataset.sequences[i]['video_id'] for i in train_indices})
    val_video_count = len({full_dataset.sequences[i]['video_id'] for i in val_indices})

    # ── 獨立的 augment 旗標（copy.copy 共享 sequences 但各自持有旗標） ──
    train_base = copy.copy(full_dataset)
    train_base.augment = True
    val_base = copy.copy(full_dataset)
    val_base.augment = False

    train_dataset = torch.utils.data.Subset(train_base, train_indices)
    val_dataset   = torch.utils.data.Subset(val_base,   val_indices)

    # 列印訓練集與驗證集的類別分布
    def print_split_distribution(subset, name):
        from collections import Counter
        labels = [subset.dataset.sequences[i]['label'] for i in subset.indices]
        counter = Counter(labels)
        print(f"\n{name} class distribution:")
        for cls in range(NUM_CLASSES):
            print(f"  {cls}: {counter.get(cls, 0)} sequences")

    print_split_distribution(train_dataset, "Train")
    print_split_distribution(val_dataset, "Validation")

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Feature mode: {feature_mode} (in_channels={in_channels})")
    print(f"✓ Strict window filter: {STRICT_WINDOW_FILTER}")

    # ── 計算類別權重（解決類別不平衡） ──────────────────────────────────
    from collections import Counter
    train_labels_all = [full_dataset.sequences[i]['label'] for i in train_indices]
    label_counts = Counter(train_labels_all)
    total = sum(label_counts.values())
    if USE_CB_LOSS:
        # Class-Balanced Loss（effective number of samples）：
        # effective_num = (1 - beta^n) / (1 - beta)，weight ∝ 1 / effective_num，
        # 再正規化使平均權重為 1，維持跟反頻率公式相近的 loss 量級。
        effective_num = [1.0 - CB_BETA ** label_counts.get(c, 1) for c in range(NUM_CLASSES)]
        cb_weights = [(1.0 - CB_BETA) / max(en, 1e-8) for en in effective_num]
        mean_w = sum(cb_weights) / NUM_CLASSES
        class_weights = torch.tensor(
            [w / mean_w for w in cb_weights],
            dtype=torch.float32
        ).to(DEVICE)
    else:
        class_weights = torch.tensor(
            [total / (NUM_CLASSES * label_counts.get(c, 1)) for c in range(NUM_CLASSES)],
            dtype=torch.float32
        ).to(DEVICE)

    print(f"\n✓ Class weights ({'CB loss, beta=' + str(CB_BETA) if USE_CB_LOSS else 'inverse frequency'}): "
          f"{ {i: round(float(w), 3) for i, w in enumerate(class_weights.cpu())} }")

    # Add class counts to run_log meta and flush to disk
    run_log_data['meta']['class_counts'] = {str(k): int(v) for k, v in label_counts.items()}
    try:
        with open(run_log_path, 'w', encoding='utf-8') as lf:
            json.dump(run_log_data, lf, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠ Failed to update run log meta with class counts: {e}")

    # Create data loaders
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=eff_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # 未加權版本的 train_loader（自然類別分布，no WeightedRandomSampler）：
    # 只用來在每個 epoch 結束後跑一次 validate()，取得跟 val_acc 口徑一致的
    # 「unweighted train acc」，藉此拆穿 train_acc/val_acc 落差有多少是取樣權重造成的假象。
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=eff_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # If dataset is small and imbalanced, optionally use a WeightedRandomSampler
    use_weighted_sampler = CONFIG.get('USE_WEIGHTED_SAMPLER', True)
    if use_weighted_sampler and len(train_dataset) > 0:
        # train_labels_all corresponds to the labels in train_indices order
        sample_weights = [1.0 / max(1, label_counts.get(l, 1)) for l in train_labels_all]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=eff_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if DEVICE.type == 'cuda' else False
        )

    # Create model
    print("\nInitializing ST-GCN model...")
    joint_prior_weights = _build_joint_prior_weights()
    model_config = {
        'num_classes': NUM_CLASSES,
        'in_channels': in_channels,
        'num_joints': NUM_JOINTS,
        'spatial_kernel_size': SPATIAL_KERNEL_SIZE,
        'temporal_kernel_size': TEMPORAL_KERNEL_SIZE,
        'num_layers': NUM_STGCN_LAYERS,
        'input_dropout': eff_input_dropout,
        'block_dropout': eff_block_dropout,
        'final_dropout': eff_final_dropout,
        'use_attention': use_attention,
        'joint_prior_weights': joint_prior_weights,
    }
    model = STGCN(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        num_joints=NUM_JOINTS,
        spatial_kernel_size=SPATIAL_KERNEL_SIZE,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        num_layers=NUM_STGCN_LAYERS,
        input_dropout=eff_input_dropout,
        block_dropout=eff_block_dropout,
        final_dropout=eff_final_dropout,
        use_attention=use_attention,
        joint_prior_weights=joint_prior_weights,
    ).to(DEVICE)

    # EMA
    ema_decay = 0.999
    ema = ModelEMA(model, model_config, decay=ema_decay)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {num_params:,} trainable parameters")

    # Loss function and optimizer
    if eff_label_smoothing and eff_label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=eff_label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt_name = str(OPTIMIZER).strip().lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=eff_lr,
            weight_decay=WEIGHT_DECAY
        )
    elif opt_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=eff_lr,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
    else:
        # default to Adam for backward compatibility
        optimizer = optim.Adam(
            model.parameters(),
            lr=eff_lr,
            weight_decay=WEIGHT_DECAY
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # monitor validation loss (more stable than val accuracy)
        factor=0.5,
        patience=4,
    )

    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    training_start_time = datetime.now(timezone.utc)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_state_dict = None       # 記憶體暫存最佳權重，訓練結束後一次寫檔
    best_val_preds = None
    best_val_labels = None
    best_val_macro_f1 = 0.0
    best_val_per_class_f1 = None
    patience_counter = 0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    train_eval_losses = []          # 未加權（自然分布）train loss，跟 val 同口徑
    train_eval_accs = []            # 未加權（自然分布）train acc，跟 val 同口徑
    val_per_class_f1_history = []   # 每個 epoch 的 val per-class F1，用來看各類別是否同步進步

    epoch = -1  # NUM_EPOCHS=0 時迴圈完全不執行，避免下方 epoch+1 出現 UnboundLocalError
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # EMA update
        ema.update(model)

        # Validate: optionally use EMA shadow weights for evaluation/saving
        # (ema.ema is the cloned EMA model kept on the same device)
        eval_model = ema.ema if USE_EMA_FOR_EVAL else model
        val_loss, val_acc, val_macro_f1, val_preds, val_labels, val_per_class_f1 = validate(eval_model, val_loader, criterion, DEVICE, desc="Val")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_per_class_f1_history.append(val_per_class_f1.tolist())

        # 未加權 train acc：同一份 train_dataset，但不透過 WeightedRandomSampler，
        # 拿來跟 val_acc 對齊比較，拆解 train_acc/val_acc 落差有多少是取樣權重造成的假象
        # （見上一輪討論：WeightedRandomSampler 只套用在訓練，會把稀少類別大量重抽樣，
        # 拉低訓練時回報的 accuracy，跟自然分布的 val_acc 不是同一個口徑）。
        train_eval_loss, train_eval_acc, _, _, _, _ = validate(eval_model, train_eval_loader, criterion, DEVICE, desc="Train(eval)")
        train_eval_losses.append(train_eval_loss)
        train_eval_accs.append(train_eval_acc)

        # Scheduler step (monitor validation loss)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}  (weighted-sampler 口徑)")
        print(f"Train Loss (eval,未加權): {train_eval_loss:.4f} | Train Acc (eval,未加權): {train_eval_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        # Append epoch metrics to run log
        epoch_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_eval_loss': float(train_eval_loss),
            'train_eval_acc': float(train_eval_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_macro_f1': float(val_macro_f1),
            'val_per_class_f1': val_per_class_f1.tolist(),
            'lr': float(optimizer.param_groups[0]['lr']) if optimizer.param_groups else None,
        }
        # Append epoch metrics to in-memory log and flush full JSON to disk
        run_log_data['epochs'].append(epoch_record)
        try:
            with open(run_log_path, 'w', encoding='utf-8') as lf:
                json.dump(run_log_data, lf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Failed to write run log: {e}")

        # Early stopping（acc 相同時以 val_loss 更低為準，確保儲存最佳 checkpoint）
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1
            # 暫存最佳權重（若使用 EMA 做驗證，儲存 EMA 權重；否則儲存模型當前權重）
            if USE_EMA_FOR_EVAL:
                best_state_dict = copy.deepcopy(ema.state_dict())
            else:
                best_state_dict = copy.deepcopy(model.state_dict())  # 暫存於記憶體，不寫檔
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_val_per_class_f1 = val_per_class_f1
            patience_counter = 0
            print(f"  → New best: acc={best_val_acc:.4f}, macro_f1={best_val_macro_f1:.4f}, loss={best_val_loss:.4f} (will save after training)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 訓練結束後一次寫檔，避免訓練中途因磁碟 I/O 造成停頓
    if best_state_dict is not None:
        torch.save(best_state_dict, run_model_path)
        print(f"\n✓ Best model saved to: {run_model_path}")
        print(f"  Best val acc={best_val_acc:.4f}, val macro_f1={best_val_macro_f1:.4f}, val loss={best_val_loss:.4f}")
        plot_confusion_matrix(best_val_labels, best_val_preds, run_results_dir)
        diagnose_confusion_pair(full_dataset, val_indices, best_val_labels, best_val_preds,
                                'lick', 'stop', output_dir=run_results_dir)
        diagnose_keypoint_motion(full_dataset, val_indices, best_val_labels, best_val_preds,
                                 'lick', 'stop', output_dir=run_results_dir)
        # scratch 在驗證集裡幾乎沒有真實誤判（100% 精確率/召回率），diagnose_confusion_pair
        # 沒有意義；但 Table 1（正確分類的兩類逐關節動作幅度比較）本身不需要誤判樣本，
        # 拿 stop（近乎靜止的類別）當比較基準，可以驗證 scratch 的判別訊號集中在哪個關節
        # （例如懷疑的後腳尖 LH_Paw/RH_Paw，而非之前 JOINT_PRIOR_WEIGHTS 誤設的前腳）。
        diagnose_keypoint_motion(full_dataset, val_indices, best_val_labels, best_val_preds,
                                 'scratch', 'stop', output_dir=run_results_dir)

    # 計算訓練總時長
    training_end_time = datetime.now(timezone.utc)
    training_duration_sec = (training_end_time - training_start_time).total_seconds()
    total_epochs_run = epoch + 1
    print(f"✓ Training duration: {_format_duration(training_duration_sec)}  ({total_epochs_run} epochs)")

    # run_log 補齊最終結果與時長後寫盤
    run_log_data['meta']['training_duration_seconds'] = round(training_duration_sec, 1)
    run_log_data['meta']['training_duration_human'] = _format_duration(training_duration_sec)
    run_log_data['meta']['total_epochs_run'] = total_epochs_run
    run_log_data['meta']['final_result'] = {
        'best_val_acc': float(best_val_acc),
        'best_val_macro_f1': float(best_val_macro_f1),
        'best_val_loss': float(best_val_loss),
        'best_val_per_class_f1': best_val_per_class_f1.tolist() if best_val_per_class_f1 is not None else [],
        'model_path': run_model_path,
    }
    try:
        with open(run_log_path, 'w', encoding='utf-8') as lf:
            json.dump(run_log_data, lf, ensure_ascii=False, indent=2)
        print(f"✓ Run log finalised: {run_log_path}")
    except Exception as e:
        print(f"⚠ Failed to finalise run log: {e}")

    # 全域訓練歷程（JSONL 追加，不覆蓋）
    history_path = Path(RESULTS_FOLDER) / 'training_history.jsonl'
    history_entry = {
        'timestamp': training_end_time.isoformat(),
        'run_suffix': run_suffix,
        'feature_mode': feature_mode,
        'kp_ema_alpha': eff_alpha,
        'in_channels': in_channels,
        'sequence_length': eff_seq_len,
        'window_stride': eff_window_stride,
        'use_attention': use_attention,
        'batch_size': eff_batch_size,
        'learning_rate': eff_lr,
        'optimizer': OPTIMIZER,
        'input_dropout': eff_input_dropout,
        'block_dropout': eff_block_dropout,
        'final_dropout': eff_final_dropout,
        'label_smoothing': eff_label_smoothing,
        'train_videos': train_video_count,
        'val_videos': val_video_count,
        'train_sequences': len(train_indices),
        'val_sequences': len(val_indices),
        'total_epochs_run': total_epochs_run,
        'best_val_acc': float(best_val_acc),
        'best_val_macro_f1': float(best_val_macro_f1),
        'best_val_loss': float(best_val_loss),
        'training_duration_seconds': round(training_duration_sec, 1),
        'training_duration_human': _format_duration(training_duration_sec),
        'model_path': run_model_path,
        'results_dir': run_results_dir,
    }
    try:
        with open(history_path, 'a', encoding='utf-8') as hf:
            hf.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
        print(f"✓ Training history appended: {history_path}")
    except Exception as e:
        print(f"⚠ Failed to append training history: {e}")

    # Plot training curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    # Loss curves
    ax1.plot(train_losses, label='Train Loss (weighted-sampler)', linewidth=2)
    ax1.plot(train_eval_losses, label='Train Loss (eval,未加權)', linewidth=2, linestyle='--')
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc (weighted-sampler)', linewidth=2)
    ax2.plot(train_eval_accs, label='Train Acc (eval,未加權，跟 val 同口徑)', linewidth=2, linestyle='--')
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Per-class Val F1 curves（拆解整體 acc 掩蓋掉的類別落差，尤其 shake/scratch 這類稀少類別）
    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    for c, cname in enumerate(label_names):
        ax3.plot([f1s[c] for f1s in val_per_class_f1_history], label=cname, linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Val F1', fontsize=12)
    ax3.set_title('Per-Class Validation F1', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(run_results_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()

    return {
        'feature_mode': feature_mode,
        'kp_ema_alpha': eff_alpha,
        'in_channels': in_channels,
        'best_val_acc': float(best_val_acc),
        'best_val_macro_f1': float(best_val_macro_f1),
        'best_val_loss': float(best_val_loss),
        'best_val_per_class_f1': best_val_per_class_f1.tolist() if best_val_per_class_f1 is not None else [],
        'model_path': run_model_path,
        'results_dir': run_results_dir,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_eval_losses': train_eval_losses,
        'train_eval_accs': train_eval_accs,
        'val_per_class_f1_history': val_per_class_f1_history,
        'total_epochs_run': total_epochs_run,
    }


def plot_confusion_matrix(labels, preds, output_dir):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    class_names = ["walk", "lick", "scratch", "shake", "stop"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


def _vlen(s: str) -> int:
    """字串的視覺寬度（中文/全形 = 2，其餘 = 1），CJK 與英數混排時對齊表格用。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _vlj(s, w: int) -> str:
    """左對齊，依視覺寬度補空格。"""
    s = str(s)
    return s + " " * max(0, w - _vlen(s))


def _vrj(s, w: int) -> str:
    """右對齊，依視覺寬度補空格。"""
    s = str(s)
    return " " * max(0, w - _vlen(s)) + s


def _compute_window_motion(seq_xy: np.ndarray, conf_seq: np.ndarray, conf_threshold: float = 0.3) -> float:
    """
    算一個 window 的平均逐幀關節位移量（只計入信心足夠的關節），數值代表模型
    實際看到的「動態程度」。seq_xy 須先經過跟 __getitem__ 相同的
    flip_normalize/orientation_normalize/normalize_skeleton_coords，才能跨影片
    比較（否則不同鏡頭距離/縮放會讓像素位移不可比）。
    """
    diffs = seq_xy[1:] - seq_xy[:-1]                 # (T-1, J, 2)
    dist  = np.linalg.norm(diffs, axis=-1)            # (T-1, J)
    valid = (conf_seq[1:] > conf_threshold) & (conf_seq[:-1] > conf_threshold)
    if not valid.any():
        return 0.0
    return float(dist[valid].mean())


def _compute_per_joint_motion(seq_xy: np.ndarray, conf_seq: np.ndarray,
                              conf_threshold: float = 0.3) -> np.ndarray:
    """
    跟 _compute_window_motion 邏輯相同，但回傳每個關節各自的平均逐幀位移量
    （shape=(J,)），用來檢驗「動態訊號是不是集中在頭部少數關節」這類假設。
    信心不足、整個 window 都沒有有效幀的關節回傳 NaN（統計時記得用 nanmean 略過）。
    """
    diffs = seq_xy[1:] - seq_xy[:-1]                 # (T-1, J, 2)
    dist  = np.linalg.norm(diffs, axis=-1)            # (T-1, J)
    valid = (conf_seq[1:] > conf_threshold) & (conf_seq[:-1] > conf_threshold)   # (T-1, J)
    out = np.full(dist.shape[1], np.nan, dtype=np.float64)
    for j in range(dist.shape[1]):
        if valid[:, j].any():
            out[j] = dist[valid[:, j], j].mean()
    return out


def _joint_prior_weights_note() -> str:
    """回傳這次訓練當下的 JOINT_PRIOR_WEIGHTS 設定描述，嵌進診斷報告開頭，
    這樣報告本身就能回答「這次是用什麼權重設定跑出來的」，不用回頭翻 config。"""
    if not USE_JOINT_PRIOR_WEIGHTS or not JOINT_PRIOR_WEIGHTS_CFG:
        return "Joint Prior Weights: 未啟用（全部關節=1.0，跟原本行為相同）"
    weights_str = ', '.join(f'{name}={w}' for name, w in JOINT_PRIOR_WEIGHTS_CFG.items())
    return f"Joint Prior Weights: 已啟用 — {weights_str}（其餘關節=1.0）"


def diagnose_confusion_pair(full_dataset, val_indices, val_labels, val_preds,
                            class_a: str, class_b: str, max_examples: int = 8,
                            output_dir: str = None):
    """
    針對兩個容易混淆的類別（例如 lick/stop），量化檢驗「是不是誤判的 window
    本身動態幅度就偏低，跟另一類很像」這個假設：
      1. 分別算出「真的是 A 且分對」「真的是 B 且分對」的動作幅度分布，當作兩類的參考基準。
      2. 再看「真的是 A 卻分成 B」「真的是 B 卻分成 A」這些誤判 window 的動作幅度，
         落在哪一邊——如果誤判的 A→B window 動作幅度明顯偏向 B 的基準，就支持
         「這些 window 本質上動態不足、跟 B 太像」的假設；如果落在 A 的基準附近，
         代表問題不在動態量本身，得往別的方向（例如姿勢相似度、標籤邊界）查。
      3. 依影片彙總誤判 window 數，方便看出是不是集中在少數幾支影片（拖累整體誤判數的
         問題影片），而非該類別普遍容易混淆。
      4. 同時列出每個誤判 window 對應的影片與時間戳，方便手動回放影片核對畫面。

    output_dir: 若提供，把這份報告額外存成
    {output_dir}/confusion_diagnosis_{class_a}_{class_b}.txt（每次訓練都會覆寫成最新結果，
    方便跟該次訓練的其他輸出檔案放在一起留存紀錄，不只印在終端）。
    """
    lines = []
    def _p(s=""):
        lines.append(s)

    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    if class_a not in label_names or class_b not in label_names:
        print(f"  ⚠ diagnose_confusion_pair: {class_a}/{class_b} 不在 BEHAVIOR_PREFIXES 內，略過")
        return
    a_idx, b_idx = label_names.index(class_a), label_names.index(class_b)

    def _motion_for(idx):
        item = full_dataset.sequences[idx]
        seq  = item['sequence']
        conf_seq = item['conf_sequence']
        if full_dataset.num_joints < seq.shape[1]:
            seq = seq[:, :full_dataset.num_joints, :]
            conf_seq = conf_seq[:, :full_dataset.num_joints]
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        return _compute_window_motion(seq, conf_seq)

    def _fmt_time(t):
        return f"{t:.2f}s" if t is not None else "n/a"

    correct_a, correct_b = [], []
    confused_a_to_b, confused_b_to_a = [], []
    for i, (true, pred) in enumerate(zip(val_labels, val_preds)):
        if true == a_idx and pred == a_idx:
            correct_a.append(_motion_for(val_indices[i]))
        elif true == b_idx and pred == b_idx:
            correct_b.append(_motion_for(val_indices[i]))
        elif true == a_idx and pred == b_idx:
            confused_a_to_b.append(val_indices[i])
        elif true == b_idx and pred == a_idx:
            confused_b_to_a.append(val_indices[i])

    from collections import Counter
    SEP = '─' * 70

    _p(f"\n{'='*70}")
    _p(f"  混淆診斷：{class_a} ↔ {class_b}")
    _p(f"  {_joint_prior_weights_note()}")
    _p(f"{'='*70}")

    # ── 動作幅度統計表 ──────────────────────────────────────────────────
    _p("\n【動作幅度統計】")
    _p(f"  {_vlj('分類', 20)} {_vrj('n', 6)} {_vrj('mean', 10)} {_vrj('median', 10)}")
    _p(f"  {SEP}")

    def _stat_row(name, vals):
        if not vals:
            _p(f"  {_vlj(name, 20)} {_vrj('—', 6)} {_vrj('—', 10)} {_vrj('—', 10)}")
            return
        arr = np.array(vals)
        _p(f"  {_vlj(name, 20)} {_vrj(len(arr), 6)} {_vrj(f'{arr.mean():.4f}', 10)} "
           f"{_vrj(f'{np.median(arr):.4f}', 10)}")

    _stat_row(f"正確分類的 {class_a}", correct_a)
    _stat_row(f"正確分類的 {class_b}", correct_b)

    all_detail_rows = []   # 給 CSV 用：(direction, video_id, start_time, end_time, start_idx, motion)

    def _report_confused(indices, true_name, pred_name):
        if not indices:
            _stat_row(f"誤判 {true_name}→{pred_name}", [])
            return
        motions = [_motion_for(idx) for idx in indices]
        _stat_row(f"誤判 {true_name}→{pred_name}", motions)

        direction = f"{true_name}→{pred_name}"
        for idx, m in zip(indices, motions):
            item = full_dataset.sequences[idx]
            all_detail_rows.append((direction, item['video_id'], item.get('start_time'),
                                    item.get('end_time'), item.get('start_idx'), m))
        return motions

    motions_a_to_b = _report_confused(confused_a_to_b, class_a, class_b)
    motions_b_to_a = _report_confused(confused_b_to_a, class_b, class_a)

    ref_a = np.median(correct_a) if correct_a else None
    ref_b = np.median(correct_b) if correct_b else None
    for direction, motions in ((f"{class_a}→{class_b}", motions_a_to_b),
                               (f"{class_b}→{class_a}", motions_b_to_a)):
        if motions and ref_a is not None and ref_b is not None:
            # 用 median（而非 mean）判斷比較接近哪一類的基準：mean 在樣本數不多、
            # 分布右偏（少數離群值把平均拉高）時容易誤導結論，median 對離群值較穩健。
            med_m = np.median(motions)
            closer_to = class_a if abs(med_m - ref_a) < abs(med_m - ref_b) else class_b
            skew_note = ""
            if abs(np.mean(motions) - med_m) > 0.3 * max(med_m, 1e-9):
                skew_note = f"（注意：mean={np.mean(motions):.4f} 跟 median 差距較大，分布可能右偏，判斷以 median 為準）"
            _p(f"    → 誤判 {direction} 的動作幅度中位數較接近「{closer_to}」的基準"
                f"（{class_a} median={ref_a:.4f}, {class_b} median={ref_b:.4f}）{skew_note}")

    # ── 依影片彙總表 ────────────────────────────────────────────────────
    for indices, true_name, pred_name in ((confused_a_to_b, class_a, class_b),
                                          (confused_b_to_a, class_b, class_a)):
        if not indices:
            continue
        video_counts = Counter(full_dataset.sequences[idx]['video_id'] for idx in indices)
        if len(video_counts) <= 1:
            continue
        _p(f"\n【{true_name}→{pred_name} 誤判依影片彙總】（共 {len(video_counts)} 支影片）")
        _p(f"  {_vlj('影片', 24)} {_vrj('誤判數', 8)}")
        _p(f"  {SEP}")
        for vid, cnt in video_counts.most_common():
            _p(f"  {_vlj(vid, 24)} {_vrj(cnt, 8)}")

    # ── 逐筆明細表 ──────────────────────────────────────────────────────
    for indices, true_name, pred_name in ((confused_a_to_b, class_a, class_b),
                                          (confused_b_to_a, class_b, class_a)):
        if not indices:
            continue
        motions = [_motion_for(idx) for idx in indices]
        _p(f"\n【{true_name}→{pred_name} 誤判明細】（依動作幅度排序，最多列 {max_examples} 筆）")
        _p(f"  {_vlj('影片', 24)} {_vlj('時間區間', 22)} {_vrj('start_idx', 10)} {_vrj('動作幅度', 10)}")
        _p(f"  {SEP}")
        for idx, m in sorted(zip(indices, motions), key=lambda x: x[1])[:max_examples]:
            item = full_dataset.sequences[idx]
            time_range = f"{_fmt_time(item.get('start_time'))} ~ {_fmt_time(item.get('end_time'))}"
            _p(f"  {_vlj(item['video_id'], 24)} {_vlj(time_range, 22)} "
               f"{_vrj(item.get('start_idx'), 10)} {_vrj(f'{m:.4f}', 10)}")

    _p(f"\n{'='*70}\n")

    print('\n'.join(lines))
    if output_dir:
        # 跟 diagnose_keypoint_motion() 共用同一份報告檔（{a}_{b}_diagnosis.txt），
        # 這個函式先呼叫、用 'w' 建立/覆寫，diagnose_keypoint_motion() 之後接著用
        # 'a' 附加進同一個檔案，兩者互補（動作幅度總量+位置 vs 逐關節拆解），
        # 合成一份檔案就不用來回對照兩個檔案。
        out_path = os.path.join(output_dir, f'{class_a}_{class_b}_diagnosis.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✓ 混淆診斷已存檔: {out_path}")

        if all_detail_rows:
            import csv
            csv_path = os.path.join(output_dir, f'{class_a}_{class_b}_diagnosis.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['direction', 'video_id', 'start_time', 'end_time', 'start_idx', 'motion'])
                for row in sorted(all_detail_rows, key=lambda r: r[-1]):
                    w.writerow(row)
            print(f"✓ 混淆診斷明細（全部 {len(all_detail_rows)} 筆）已存成 CSV: {csv_path}")


def diagnose_keypoint_motion(full_dataset, val_indices, val_labels, val_preds,
                             class_a: str, class_b: str,
                             opening_frame_threshold: int = 48,
                             output_dir: str = None):
    """
    diagnose_confusion_pair() 的逐關節版本，驗證「判別力是否集中在頭部少數關節」
    這個假設（例如 lick/stop 混淆懷疑是頭部局部運動被全域圖卷積稀釋造成）：
      1. 分別算出「正確分類的 A／B」的逐關節動作幅度平均值，兩者差異最大的關節
         就是理論上最有判別力的關節（驗證 Nose/耳朵是否名列前茅）。
      2. 針對 B→A 誤判 window，依 start_idx 是否落在影片開頭
         （< opening_frame_threshold 幀）分成兩組分別看逐關節動作幅度——
         如果排除開頭那組後，剩下的誤判仍呈現「頭部關節動態接近 A、其餘關節接近 B」
         的訊號，才真正支持「局部頭部運動被稀釋」這個假設；如果訊號主要來自開頭那組，
         代表誤判主力其實是別的原因（例如入鏡安頓、偵測未穩定），該分開論述。

    opening_frame_threshold: 幾幀以內視為「影片開頭」，預設 48 幀（≈1.6s@30fps），
    依先前診斷觀察到的誤判聚集範圍抓的門檻，可依實際狀況調整。
    """
    lines = []
    def _p(s=""):
        lines.append(s)

    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    if class_a not in label_names or class_b not in label_names:
        print(f"  ⚠ diagnose_keypoint_motion: {class_a}/{class_b} 不在 BEHAVIOR_PREFIXES 內，略過")
        return
    a_idx, b_idx = label_names.index(class_a), label_names.index(class_b)

    def _joint_motion_for(idx):
        item = full_dataset.sequences[idx]
        seq  = item['sequence']
        conf_seq = item['conf_sequence']
        if full_dataset.num_joints < seq.shape[1]:
            seq = seq[:, :full_dataset.num_joints, :]
            conf_seq = conf_seq[:, :full_dataset.num_joints]
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        return _compute_per_joint_motion(seq, conf_seq)

    n_joints = full_dataset.num_joints
    joint_names = KEYPOINT_NAMES[:n_joints]

    correct_a_j, correct_b_j = [], []
    confused_a_to_b_idx, confused_b_to_a_idx = [], []
    for i, (true, pred) in enumerate(zip(val_labels, val_preds)):
        if true == a_idx and pred == a_idx:
            correct_a_j.append(_joint_motion_for(val_indices[i]))
        elif true == b_idx and pred == b_idx:
            correct_b_j.append(_joint_motion_for(val_indices[i]))
        elif true == a_idx and pred == b_idx:
            confused_a_to_b_idx.append(val_indices[i])
        elif true == b_idx and pred == a_idx:
            confused_b_to_a_idx.append(val_indices[i])

    with np.errstate(invalid='ignore'):
        mean_a = np.nanmean(np.stack(correct_a_j), axis=0) if correct_a_j else np.full(n_joints, np.nan)
        mean_b = np.nanmean(np.stack(correct_b_j), axis=0) if correct_b_j else np.full(n_joints, np.nan)

    SEP = '─' * 70
    _p(f"\n{'='*70}")
    _p(f"  逐關節動作幅度診斷：{class_a} ↔ {class_b}")
    _p(f"  {_joint_prior_weights_note()}")
    _p(f"{'='*70}")

    _p(f"\n【正確分類的 {class_a} vs {class_b}：逐關節動作幅度】"
       f"（依差異排序，最上面的關節理論上判別力最高）")
    _p(f"  {_vlj('關節', 12)} {_vrj(class_a+'均值', 12)} {_vrj(class_b+'均值', 12)} {_vrj('差異', 10)}")
    _p(f"  {SEP}")
    order = np.argsort(-np.nan_to_num(mean_a - mean_b, nan=-np.inf))
    for j in order:
        va, vb = mean_a[j], mean_b[j]
        diff_s = f"{va-vb:+.4f}" if not (np.isnan(va) or np.isnan(vb)) else "n/a"
        va_s = f"{va:.4f}" if not np.isnan(va) else "n/a"
        vb_s = f"{vb:.4f}" if not np.isnan(vb) else "n/a"
        _p(f"  {_vlj(joint_names[j], 12)} {_vrj(va_s, 12)} {_vrj(vb_s, 12)} {_vrj(diff_s, 10)}")

    def _closer_label(v, ref_a, ref_b):
        if np.isnan(v) or np.isnan(ref_a) or np.isnan(ref_b):
            return "n/a"
        return class_a if abs(v - ref_a) < abs(v - ref_b) else class_b

    def _report_group(idxs, group_name):
        if not idxs:
            _p(f"\n  {group_name}：無樣本")
            return
        arrs = [_joint_motion_for(idx) for idx in idxs]
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mean_g = np.nanmean(np.stack(arrs), axis=0)
        _p(f"\n【{group_name}】（n={len(idxs)}）")
        _p(f"  {_vlj('關節', 12)} {_vrj('此組均值', 12)} {_vrj(class_a+'基準', 12)} "
           f"{_vrj(class_b+'基準', 12)} {_vrj('較接近', 8)}")
        _p(f"  {SEP}")
        for j in range(n_joints):
            v = mean_g[j]
            v_s = f"{v:.4f}" if not np.isnan(v) else "n/a"
            _p(f"  {_vlj(joint_names[j], 12)} {_vrj(v_s, 12)} {_vrj(f'{mean_a[j]:.4f}', 12)} "
               f"{_vrj(f'{mean_b[j]:.4f}', 12)} {_vrj(_closer_label(v, mean_a[j], mean_b[j]), 8)}")

    _report_group(confused_a_to_b_idx, f"誤判 {class_a}→{class_b}：逐關節動作幅度")

    opening_idx = [idx for idx in confused_b_to_a_idx
                  if full_dataset.sequences[idx].get('start_idx', 0) < opening_frame_threshold]
    other_idx = [idx for idx in confused_b_to_a_idx if idx not in opening_idx]
    _p(f"\n【誤判 {class_b}→{class_a}：依 start_idx < {opening_frame_threshold} 幀（影片開頭）分群】")
    _p(f"  開頭組 n={len(opening_idx)}，非開頭組 n={len(other_idx)}")
    _report_group(opening_idx, f"誤判 {class_b}→{class_a}（開頭組）：逐關節動作幅度")
    _report_group(other_idx, f"誤判 {class_b}→{class_a}（非開頭組）：逐關節動作幅度 ← 排除開頭雜訊後的真實訊號")

    _p(f"\n{'='*70}\n")

    print('\n'.join(lines))
    if output_dir:
        # 附加進 diagnose_confusion_pair() 已建立的同一份報告檔，兩份診斷合成一份，
        # 不需要另外開一個檔案來回對照。
        out_path = os.path.join(output_dir, f'{class_a}_{class_b}_diagnosis.txt')
        mode = 'a' if os.path.exists(out_path) else 'w'
        with open(out_path, mode, encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✓ 逐關節動作幅度診斷已附加存檔: {out_path}")


def diagnose_keypoint_motion_groundtruth(full_dataset, indices, class_a: str, class_b: str,
                                         output_dir: str = None):
    """
    diagnose_keypoint_motion() 的「純資料版」：完全不需要模型/預測結果，只依
    ground truth 標籤篩選 window，比較兩個類別「全部」樣本（不是只挑模型判對
    的樣本）的逐關節動作幅度。

    動機：diagnose_keypoint_motion() 的 Table 1 只統計「模型判對」的樣本，而
    模型是拿全部行為類別一起訓練出來的——如果訓練時的類別交互作用讓「判對的
    scratch」剛好偏向某種特定樣態（不是 scratch 的真實全貌），那 Table 1 看到
    的關節排名就可能是訓練造成的假象，不是資料本身的特性。這個函式跳過模型，
    直接用標註標籤挑出「全部」屬於該類別的 window 來算，結果如果跟
    diagnose_keypoint_motion() 的 Table 1 一致，就能證明先前的結論不是模型
    5 類別聯合訓練造成的選樣偏差；如果不一致，代表選樣偏差確實存在，要以這份
    「純資料版」結果為準。

    indices: 要涵蓋的 sequence index 清單（例如 val_indices，或
    range(len(full_dataset.sequences)) 涵蓋全部資料以取得最大樣本數——這裡沒有
    train/val 洩漏疑慮，因為完全不涉及任何模型評估，用全部資料統計最穩定）。
    """
    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
    if class_a not in label_names or class_b not in label_names:
        print(f"  ⚠ diagnose_keypoint_motion_groundtruth: {class_a}/{class_b} 不在 BEHAVIOR_PREFIXES 內，略過")
        return
    a_idx, b_idx = label_names.index(class_a), label_names.index(class_b)

    def _joint_motion_for(idx):
        item = full_dataset.sequences[idx]
        seq  = item['sequence']
        conf_seq = item['conf_sequence']
        if full_dataset.num_joints < seq.shape[1]:
            seq = seq[:, :full_dataset.num_joints, :]
            conf_seq = conf_seq[:, :full_dataset.num_joints]
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        return _compute_per_joint_motion(seq, conf_seq)

    n_joints = full_dataset.num_joints
    joint_names = KEYPOINT_NAMES[:n_joints]

    a_joints, b_joints = [], []
    for idx in indices:
        lbl = full_dataset.sequences[idx]['label']
        if lbl == a_idx:
            a_joints.append(_joint_motion_for(idx))
        elif lbl == b_idx:
            b_joints.append(_joint_motion_for(idx))

    with np.errstate(invalid='ignore'):
        mean_a = np.nanmean(np.stack(a_joints), axis=0) if a_joints else np.full(n_joints, np.nan)
        mean_b = np.nanmean(np.stack(b_joints), axis=0) if b_joints else np.full(n_joints, np.nan)

    lines = []
    def _p(s=""):
        lines.append(s)

    SEP = '─' * 70
    _p(f"\n{'='*70}")
    _p(f"  逐關節動作幅度診斷（純資料版，不經模型）：{class_a} ↔ {class_b}")
    _p(f"  {_joint_prior_weights_note()}")
    _p(f"{'='*70}")
    _p(f"\n【全部 {class_a}（n={len(a_joints)}） vs 全部 {class_b}（n={len(b_joints)}）：逐關節動作幅度】")
    _p(f"  （依 ground truth 標籤挑樣本，不經模型分類，驗證 Table 1 排名是否為訓練選樣偏差）")
    _p(f"  {_vlj('關節', 12)} {_vrj(class_a+'均值', 12)} {_vrj(class_b+'均值', 12)} {_vrj('差異', 10)}")
    _p(f"  {SEP}")
    order = np.argsort(-np.nan_to_num(mean_a - mean_b, nan=-np.inf))
    for j in order:
        va, vb = mean_a[j], mean_b[j]
        diff_s = f"{va-vb:+.4f}" if not (np.isnan(va) or np.isnan(vb)) else "n/a"
        va_s = f"{va:.4f}" if not np.isnan(va) else "n/a"
        vb_s = f"{vb:.4f}" if not np.isnan(vb) else "n/a"
        _p(f"  {_vlj(joint_names[j], 12)} {_vrj(va_s, 12)} {_vrj(vb_s, 12)} {_vrj(diff_s, 10)}")
    _p(f"\n{'='*70}\n")

    print('\n'.join(lines))
    if output_dir:
        out_path = os.path.join(output_dir, f'{class_a}_{class_b}_diagnosis.txt')
        mode = 'a' if os.path.exists(out_path) else 'w'
        with open(out_path, mode, encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"✓ 純資料版逐關節動作幅度診斷已附加存檔: {out_path}")


def run_ablation_study(modes=None):
    """Run feature ablation and export a comparison CSV."""
    # If CONFIG specifies a single feature mode to run, use it.
    selected = CONFIG.get('SELECT_FEATURE_MODE')
    if selected:
        if selected not in ABLATION_MODES:
            print(f"⚠ SELECT_FEATURE_MODE='{selected}' is not a known mode. Falling back to ABLATION_MODES.")
            modes = modes or ABLATION_MODES
        else:
            modes = [selected]
    else:
        modes = modes or ABLATION_MODES

    # 同一批消融研究共用同一個組號，所有模式的結果資料夾都用這個號碼
    shared_run_num = _next_run_number(RESULTS_FOLDER)
    shared_tag = f"{shared_run_num:03d}"
    att_suffix = "att_on" if USE_ATTENTION else "att_off"
    shared_models_dir = Path(RESULTS_FOLDER) / f"run_{shared_tag}_models_{att_suffix}"
    shared_models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ 消融研究組號 #{shared_tag}  ({len(modes)} 種特徵模式)")
    print(f"  模型權重目錄: {shared_models_dir}")

    summary = []
    print("\n" + "=" * 70)
    print("Feature Ablation Study")
    print("=" * 70)
    for mode in modes:
        print("\n" + "-" * 70)
        print(f"Running mode: {mode}")
        print("-" * 70)
        result = train_model(feature_mode=mode, run_name=mode,
                             run_number=shared_run_num,
                             shared_models_dir=str(shared_models_dir))
        if result is not None:
            summary.append(result)

    if not summary:
        print("✗ No ablation results generated.")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Ablation summary CSV（含時戳，不覆蓋） ────────────────────────────
    import csv
    summary_csv = str(get_unique_path(Path(RESULTS_FOLDER) / f'ablation_summary_{ts}.csv'))
    csv_fields = [
        'feature_mode', 'in_channels', 'best_val_acc',
        'best_val_macro_f1', 'best_val_loss', 'total_epochs_run',
        'model_path', 'results_dir',
    ]
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 70)
    print("Ablation Summary")
    print("=" * 70)
    for rec in summary:
        print(
            f"{rec['feature_mode']:>22s} | C={rec['in_channels']} | "
            f"Acc={rec['best_val_acc']:.4f} | Macro-F1={rec['best_val_macro_f1']:.4f} | "
            f"Loss={rec['best_val_loss']:.4f}"
        )
    print(f"\n✓ Summary CSV saved to: {summary_csv}")

    # ── 模式短標籤 ─────────────────────────────────────────────────────────
    _mode_short = {
        'xy': 'E0', 'xy_conf': 'E1', 'xy_conf_v': 'E2',
        'xy_conf_v_bone': 'E3', 'xy_conf_v_bone_bmotion': 'E4',
    }
    label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]

    # ── Plot A：ablation_result_comparison（Acc/F1 折線 + 每類 F1 長條） ──
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        modes_run   = [r['feature_mode'] for r in summary]
        short_labels = [_mode_short.get(m, m) for m in modes_run]
        accs = [r['best_val_acc']     for r in summary]
        f1s  = [r['best_val_macro_f1'] for r in summary]

        x = np.arange(len(modes_run))
        ax1.plot(x, accs, 'o-',  linewidth=2, markersize=8, label='Val Accuracy')
        ax1.plot(x, f1s,  's--', linewidth=2, markersize=8, label='Macro-F1')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels)
        ax1.set_ylim(0, 1.08)
        ax1.set_xlabel('Feature Mode', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Accuracy & Macro-F1 per Feature Mode', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for xi, (a, f) in enumerate(zip(accs, f1s)):
            ax1.annotate(f'{a:.3f}', (xi, a), textcoords='offset points', xytext=(0, 8),  ha='center', fontsize=8)
            ax1.annotate(f'{f:.3f}', (xi, f), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8)

        # Per-class F1 grouped bar chart
        has_per_class = any(r.get('best_val_per_class_f1') for r in summary)
        if has_per_class:
            n_modes   = len(summary)
            n_classes = len(label_names)
            bar_w = 0.8 / n_modes
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for mi, rec in enumerate(summary):
                pcf1 = rec.get('best_val_per_class_f1') or [0.0] * n_classes
                xs = np.arange(n_classes) + mi * bar_w - (n_modes - 1) * bar_w / 2
                ax2.bar(xs, pcf1, width=bar_w, color=colors[mi % len(colors)],
                        label=_mode_short.get(rec['feature_mode'], rec['feature_mode']))
            ax2.set_xticks(np.arange(n_classes))
            ax2.set_xticklabels(label_names, rotation=20, ha='right')
            ax2.set_ylim(0, 1.08)
            ax2.set_xlabel('Class', fontsize=12)
            ax2.set_ylabel('F1 Score', fontsize=12)
            ax2.set_title('Per-Class F1 Score by Feature Mode', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Per-class F1 not available',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        cmp_path = str(get_unique_path(Path(RESULTS_FOLDER) / f'ablation_result_comparison_{ts}.png'))
        plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
        print(f"✓ Result comparison saved to: {cmp_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot result comparison: {e}")

    # ── Plot B：ablation_convergence_comparison（val_loss + val_acc vs epoch） ──
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for ci, rec in enumerate(summary):
            short = _mode_short.get(rec['feature_mode'], rec['feature_mode'])
            col   = colors[ci % len(colors)]
            vl = rec.get('val_losses', [])
            va = rec.get('val_accs',   [])
            if vl:
                ax1.plot(range(1, len(vl) + 1), vl, '-o', linewidth=2,
                         markersize=4, color=col, label=short)
            if va:
                ax2.plot(range(1, len(va) + 1), va, '-o', linewidth=2,
                         markersize=4, color=col, label=short)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Val Loss', fontsize=12)
        ax1.set_title('Validation Loss Convergence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Val Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy Convergence', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        conv_path = str(get_unique_path(Path(RESULTS_FOLDER) / f'ablation_convergence_comparison_{ts}.png'))
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence comparison saved to: {conv_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot convergence comparison: {e}")


# ==================== KP EMA Alpha Ablation ====================
def run_kp_ema_ablation(alphas=None):
    """對不同 KP_EMA_ALPHA 做消融實驗，固定使用 FEATURE_MODE。"""
    alphas = alphas or list(ABLATION_KP_EMA_ALPHAS)
    shared_run_num = _next_run_number(RESULTS_FOLDER)
    shared_tag     = f"{shared_run_num:03d}"
    att_suffix     = "att_on" if USE_ATTENTION else "att_off"
    shared_models_dir = Path(RESULTS_FOLDER) / f"run_{shared_tag}_ema_ablation_{att_suffix}"
    shared_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ KP EMA Alpha 消融研究 #{shared_tag}  ({len(alphas)} 種 alpha)")
    print(f"  Feature mode : {FEATURE_MODE}")
    print(f"  Alphas       : {alphas}")
    print(f"  模型目錄     : {shared_models_dir}")

    summary = []
    for alpha in alphas:
        print(f"\n{'─'*70}")
        print(f"KP_EMA_ALPHA = {alpha}")
        print(f"{'─'*70}")
        result = train_model(
            feature_mode=FEATURE_MODE,
            kp_ema_alpha=alpha,
            run_number=shared_run_num,
            shared_models_dir=str(shared_models_dir),
        )
        if result is not None:
            summary.append(result)

    if not summary:
        print("✗ No KP EMA ablation results generated.")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Summary CSV
    import csv
    summary_csv = str(get_unique_path(
        Path(RESULTS_FOLDER) / f'kp_ema_ablation_summary_{ts}.csv'
    ))
    csv_fields = [
        'kp_ema_alpha', 'feature_mode', 'in_channels',
        'best_val_acc', 'best_val_macro_f1', 'best_val_loss',
        'total_epochs_run', 'model_path', 'results_dir',
    ]
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*70}")
    print("KP EMA Alpha Ablation Summary")
    print(f"{'='*70}")
    for rec in summary:
        print(
            f"  alpha={rec['kp_ema_alpha']:.2f} | "
            f"Acc={rec['best_val_acc']:.4f} | "
            f"Macro-F1={rec['best_val_macro_f1']:.4f} | "
            f"Loss={rec['best_val_loss']:.4f}"
        )
    print(f"\n✓ Summary CSV: {summary_csv}")

    # Plot: alpha vs acc / f1
    try:
        alphas_run  = [r['kp_ema_alpha'] for r in summary]
        accs = [r['best_val_acc']      for r in summary]
        f1s  = [r['best_val_macro_f1'] for r in summary]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'KP EMA Alpha Ablation  [{FEATURE_MODE}]',
                     fontsize=13, fontweight='bold')

        # Left: line chart (alpha vs score)
        ax1 = axes[0]
        ax1.plot(alphas_run, accs, 'o-',  linewidth=2, markersize=8, label='Val Accuracy')
        ax1.plot(alphas_run, f1s,  's--', linewidth=2, markersize=8, label='Macro-F1')
        ax1.set_xlabel('KP EMA Alpha  (← stronger smoothing)', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Score vs EMA Alpha', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.08)
        ax1.invert_xaxis()   # 右 = 不平滑(1.0)，左 = 強平滑
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for al, a, f in zip(alphas_run, accs, f1s):
            ax1.annotate(f'{a:.3f}', (al, a), textcoords='offset points',
                         xytext=(0, 8),   ha='center', fontsize=9)
            ax1.annotate(f'{f:.3f}', (al, f), textcoords='offset points',
                         xytext=(0, -14), ha='center', fontsize=9)

        # Right: per-class F1 grouped bar chart
        ax2 = axes[1]
        label_names = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]
        has_per_class = any(r.get('best_val_per_class_f1') for r in summary)
        if has_per_class:
            n_modes   = len(summary)
            n_classes = len(label_names)
            bar_w  = 0.8 / n_modes
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for mi, rec in enumerate(summary):
                pcf1 = rec.get('best_val_per_class_f1') or [0.0] * n_classes
                xs   = np.arange(n_classes) + mi * bar_w - (n_modes - 1) * bar_w / 2
                ax2.bar(xs, pcf1, width=bar_w, color=colors[mi % len(colors)],
                        label=f"α={rec['kp_ema_alpha']:.2f}")
            ax2.set_xticks(np.arange(n_classes))
            ax2.set_xticklabels(label_names, rotation=20, ha='right')
            ax2.set_ylim(0, 1.08)
            ax2.set_ylabel('F1 Score', fontsize=11)
            ax2.set_title('Per-Class F1 by Alpha', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Per-class F1 not available',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plot_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'kp_ema_ablation_comparison_{ts}.png'
        ))
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ablation plot: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot KP EMA ablation: {e}")

    # Plot B：kp_ema_ablation_convergence（val_loss + val_acc vs epoch, one line per alpha）
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'KP EMA Alpha Ablation — Convergence  [{FEATURE_MODE}]',
                     fontsize=13, fontweight='bold')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for ci, rec in enumerate(summary):
            label = f"α={rec['kp_ema_alpha']:.2f}"
            col   = colors[ci % len(colors)]
            vl    = rec.get('val_losses', [])
            va    = rec.get('val_accs',   [])
            if vl:
                ax1.plot(range(1, len(vl) + 1), vl, '-o', linewidth=2,
                         markersize=4, color=col, label=label)
            if va:
                ax2.plot(range(1, len(va) + 1), va, '-o', linewidth=2,
                         markersize=4, color=col, label=label)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Val Loss', fontsize=12)
        ax1.set_title('Validation Loss Convergence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Val Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy Convergence', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        conv_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'kp_ema_ablation_convergence_{ts}.png'
        ))
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence comparison saved to: {conv_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot KP EMA convergence: {e}")


# ==================== Sequence Length Ablation ====================
def run_seqlen_ablation(seq_lens=None):
    """對不同 SEQUENCE_LENGTH 做消融實驗，固定使用 FEATURE_MODE。
    Window stride 與 batch size 依序列長度比例自動調整。
    """
    seq_lens = list(seq_lens or ABLATION_SEQLENS)
    shared_run_num = _next_run_number(RESULTS_FOLDER)
    shared_tag     = f"{shared_run_num:03d}"
    att_suffix     = "att_on" if USE_ATTENTION else "att_off"
    shared_models_dir = Path(RESULTS_FOLDER) / f"run_{shared_tag}_seqlen_ablation_{att_suffix}"
    shared_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Sequence Length 消融研究 #{shared_tag}  ({len(seq_lens)} 種序列長度: {seq_lens})")
    print(f"  特徵模式: {FEATURE_MODE}  基準設定: T={SEQUENCE_LENGTH}, stride={WINDOW_STRIDE}, batch={BATCH_SIZE}")
    print(f"  模型權重目錄: {shared_models_dir}")

    summary = []
    print("\n" + "=" * 70)
    print("Sequence Length Ablation Study")
    print("=" * 70)
    for sl in seq_lens:
        stride_ratio  = WINDOW_STRIDE / SEQUENCE_LENGTH
        eff_stride    = max(1, int(sl * stride_ratio))
        eff_batch     = BATCH_SIZE
        run_label     = f"{FEATURE_MODE}_T{sl}"
        print(f"\n{'─'*70}")
        print(f"T={sl}  stride={eff_stride}  batch={eff_batch}  (label: {run_label})")
        print(f"{'─'*70}")
        result = train_model(
            feature_mode=FEATURE_MODE,
            run_name=run_label,
            run_number=shared_run_num,
            shared_models_dir=str(shared_models_dir),
            seq_len=sl,
        )
        if result is not None:
            result['seq_len'] = sl
            result['window_stride'] = eff_stride
            result['eff_batch_size'] = eff_batch
            summary.append(result)

    if not summary:
        print("✗ No sequence length ablation results generated.")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Summary CSV
    import csv
    summary_csv = str(get_unique_path(
        Path(RESULTS_FOLDER) / f'seqlen_ablation_summary_{ts}.csv'
    ))
    csv_fields = [
        'seq_len', 'window_stride', 'eff_batch_size', 'feature_mode',
        'best_val_acc', 'best_val_macro_f1', 'best_val_loss',
        'total_epochs_run', 'model_path', 'results_dir',
    ]
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*70}")
    print("Sequence Length Ablation Summary")
    print(f"{'='*70}")
    for rec in summary:
        print(
            f"  T={rec['seq_len']}  stride={rec['window_stride']}  batch={rec['eff_batch_size']} | "
            f"Acc={rec['best_val_acc']:.4f} | "
            f"Macro-F1={rec['best_val_macro_f1']:.4f} | "
            f"Loss={rec['best_val_loss']:.4f}"
        )
    print(f"\n✓ Summary CSV: {summary_csv}")

    # Plot: T vs acc/f1 (bar) + convergence curves
    try:
        seq_lens_run = [r['seq_len']           for r in summary]
        accs         = [r['best_val_acc']      for r in summary]
        f1s          = [r['best_val_macro_f1'] for r in summary]
        label_names  = [name for name, _ in sorted(BEHAVIOR_PREFIXES.items(), key=lambda kv: kv[1])]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Sequence Length Ablation  [{FEATURE_MODE}]',
                     fontsize=13, fontweight='bold')

        # Left: grouped bar Acc / Macro-F1
        ax1 = axes[0]
        x   = np.arange(len(seq_lens_run))
        w   = 0.35
        bars_acc = ax1.bar(x - w/2, accs, w, label='Val Accuracy', color='steelblue')
        bars_f1  = ax1.bar(x + w/2, f1s,  w, label='Macro-F1',    color='darkorange')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'T={sl}' for sl in seq_lens_run])
        ax1.set_ylim(0, 1.08)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Accuracy & Macro-F1 by Sequence Length', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars_acc, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)
        for bar, v in zip(bars_f1, f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)

        # Right: per-class F1 grouped bar
        ax2 = axes[1]
        has_per_class = any(r.get('best_val_per_class_f1') for r in summary)
        if has_per_class:
            n_modes   = len(summary)
            n_classes = len(label_names)
            bar_w     = 0.8 / n_modes
            colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for mi, rec in enumerate(summary):
                pcf1 = rec.get('best_val_per_class_f1') or [0.0] * n_classes
                xs   = np.arange(n_classes) + mi * bar_w - (n_modes - 1) * bar_w / 2
                ax2.bar(xs, pcf1, width=bar_w, color=colors[mi % len(colors)],
                        label=f"T={rec['seq_len']}")
            ax2.set_xticks(np.arange(n_classes))
            ax2.set_xticklabels(label_names, rotation=20, ha='right')
            ax2.set_ylim(0, 1.08)
            ax2.set_ylabel('F1 Score', fontsize=11)
            ax2.set_title('Per-Class F1 by Sequence Length', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Per-class F1 not available',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plot_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'seqlen_ablation_comparison_{ts}.png'
        ))
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ablation plot: {plot_path}")
        plt.close()

        # Convergence curves (val_acc per epoch for each T)
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))
        fig2.suptitle(f'Convergence  [{FEATURE_MODE}]', fontsize=13, fontweight='bold')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for ci, rec in enumerate(summary):
            c = colors[ci % len(colors)]
            lbl = f"T={rec['seq_len']}"
            ax3.plot(rec.get('val_losses', []),  color=c, linewidth=2, label=lbl)
            ax4.plot(rec.get('val_accs',   []),  color=c, linewidth=2, label=lbl)
        for ax, title, ylabel in [
            (ax3, 'Val Loss',     'Loss'),
            (ax4, 'Val Accuracy', 'Accuracy'),
        ]:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        conv_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'seqlen_ablation_convergence_{ts}.png'
        ))
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence plot: {conv_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot sequence length ablation: {e}")


# ==================== Regularization / Optimizer Ablation ====================
def run_regularization_ablation(configs=None):
    """
    對 dropout（input/block/final）、label smoothing、batch size、learning rate
    做小型網格消融，固定 FEATURE_MODE、SEQUENCE_LENGTH 不變。

    動機：train/val loss 出現裂口（train loss 持續下降、val loss 打平）是過擬合訊號，
    但 dropout=0 時到底該調多少、batch/LR 要不要一起動，光憑感覺猜不如直接跑幾組
    候選值比較。這裡用 train_eval_losses（未加權、跟 val 同口徑的 train loss，
    見 CatSkeletonDataset/train_eval_loader）算出 final_train_val_loss_gap，
    直接量化「過擬合有沒有變輕」，而不是只看 best_val_acc 一個數字。
    """
    configs = list(configs or REG_ABLATION_CONFIGS or DEFAULT_REG_ABLATION_CONFIGS)
    shared_run_num = _next_run_number(RESULTS_FOLDER)
    shared_tag     = f"{shared_run_num:03d}"
    att_suffix     = "att_on" if USE_ATTENTION else "att_off"
    shared_models_dir = Path(RESULTS_FOLDER) / f"run_{shared_tag}_reg_ablation_{att_suffix}"
    shared_models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Regularization/Optimizer 消融研究 #{shared_tag}  ({len(configs)} 組候選設定)")
    print(f"  特徵模式: {FEATURE_MODE}  T={SEQUENCE_LENGTH}  stride={WINDOW_STRIDE}")
    print(f"  模型權重目錄: {shared_models_dir}")

    summary = []
    print("\n" + "=" * 70)
    print("Regularization / Optimizer Ablation Study")
    print("=" * 70)
    for cfg in configs:
        run_label = f"{FEATURE_MODE}_{cfg['name']}"
        print(f"\n{'─'*70}")
        print(f"[{cfg['name']}]  batch={cfg['batch_size']}  lr={cfg['learning_rate']}  "
              f"input_dropout={cfg.get('input_dropout', 0.0)}  block_dropout={cfg.get('block_dropout', 0.0)}  "
              f"final_dropout={cfg.get('final_dropout', 0.0)}  label_smoothing={cfg.get('label_smoothing', 0.0)}")
        print(f"{'─'*70}")
        result = train_model(
            feature_mode=FEATURE_MODE,
            run_name=run_label,
            run_number=shared_run_num,
            shared_models_dir=str(shared_models_dir),
            batch_size=cfg['batch_size'],
            learning_rate=cfg['learning_rate'],
            input_dropout=cfg.get('input_dropout'),
            block_dropout=cfg.get('block_dropout'),
            final_dropout=cfg.get('final_dropout'),
            label_smoothing=cfg.get('label_smoothing'),
        )
        if result is not None:
            result['config_name']     = cfg['name']
            result['batch_size']      = cfg['batch_size']
            result['learning_rate']   = cfg['learning_rate']
            result['input_dropout']   = cfg.get('input_dropout', 0.0)
            result['block_dropout']   = cfg.get('block_dropout', 0.0)
            result['final_dropout']   = cfg.get('final_dropout', 0.0)
            result['label_smoothing'] = cfg.get('label_smoothing', 0.0)
            # 過擬合幅度：最後一個 epoch 的 val loss 減掉「未加權 train loss」，越接近 0 越健康
            train_eval_losses = result.get('train_eval_losses') or []
            val_losses_hist   = result.get('val_losses') or []
            if train_eval_losses and val_losses_hist:
                result['final_train_val_loss_gap'] = float(val_losses_hist[-1] - train_eval_losses[-1])
            else:
                result['final_train_val_loss_gap'] = None
            summary.append(result)

    if not summary:
        print("✗ No regularization ablation results generated.")
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Summary CSV
    import csv
    summary_csv = str(get_unique_path(
        Path(RESULTS_FOLDER) / f'reg_ablation_summary_{ts}.csv'
    ))
    csv_fields = [
        'config_name', 'batch_size', 'learning_rate',
        'input_dropout', 'block_dropout', 'final_dropout', 'label_smoothing',
        'best_val_acc', 'best_val_macro_f1', 'best_val_loss',
        'final_train_val_loss_gap', 'total_epochs_run', 'model_path', 'results_dir',
    ]
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n{'='*70}")
    print("Regularization Ablation Summary")
    print(f"{'='*70}")
    for rec in summary:
        gap = rec.get('final_train_val_loss_gap')
        gap_str = f"{gap:+.4f}" if gap is not None else "N/A"
        print(
            f"  [{rec['config_name']:<16}]  batch={rec['batch_size']:<3} lr={rec['learning_rate']:<8} | "
            f"Acc={rec['best_val_acc']:.4f} | Macro-F1={rec['best_val_macro_f1']:.4f} | "
            f"Loss={rec['best_val_loss']:.4f} | TrainValGap={gap_str}"
        )
    print(f"\n✓ Summary CSV: {summary_csv}")

    # Plot: config vs acc/f1 + train/val loss gap 條狀圖，以及逐 epoch 收斂曲線
    try:
        names       = [r['config_name']        for r in summary]
        accs        = [r['best_val_acc']       for r in summary]
        f1s         = [r['best_val_macro_f1']  for r in summary]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Regularization / Optimizer Ablation  [{FEATURE_MODE}]',
                     fontsize=13, fontweight='bold')

        ax1 = axes[0]
        x   = np.arange(len(names))
        w   = 0.35
        bars_acc = ax1.bar(x - w/2, accs, w, label='Val Accuracy', color='steelblue')
        bars_f1  = ax1.bar(x + w/2, f1s,  w, label='Macro-F1',    color='darkorange')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha='right')
        ax1.set_ylim(0, 1.08)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Accuracy & Macro-F1 by Config', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars_acc, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)
        for bar, v in zip(bars_f1, f1s):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)

        ax2  = axes[1]
        gaps = [r.get('final_train_val_loss_gap') or 0.0 for r in summary]
        colors_gap = ['#C44E52' if g > 0 else '#55A868' for g in gaps]
        bars_gap = ax2.bar(x, gaps, color=colors_gap)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=15, ha='right')
        ax2.axhline(0, color='black', lw=0.8)
        ax2.set_ylabel('Val Loss − Train(eval,未加權) Loss', fontsize=11)
        ax2.set_title('Final Train/Val Loss Gap（越接近 0 代表過擬合越輕）', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars_gap, gaps):
            ax2.text(bar.get_x() + bar.get_width()/2, v + (0.01 if v >= 0 else -0.03),
                      f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plot_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'reg_ablation_comparison_{ts}.png'
        ))
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Ablation plot: {plot_path}")
        plt.close()

        # Convergence curves：val loss（實線）vs train(eval,未加權) loss（虛線），每組設定一個顏色
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))
        fig2.suptitle(f'Convergence  [{FEATURE_MODE}]', fontsize=13, fontweight='bold')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for ci, rec in enumerate(summary):
            c   = colors[ci % len(colors)]
            lbl = rec['config_name']
            ax3.plot(rec.get('val_losses', []), color=c, linewidth=2, label=f'{lbl} (val)')
            ax3.plot(rec.get('train_eval_losses', []), color=c, linewidth=1,
                     linestyle='--', label=f'{lbl} (train,未加權)')
            ax4.plot(rec.get('val_accs', []), color=c, linewidth=2, label=lbl)
        for ax, title, ylabel in [
            (ax3, 'Val vs Train(eval,未加權) Loss', 'Loss'),
            (ax4, 'Val Accuracy',                    'Accuracy'),
        ]:
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        conv_path = str(get_unique_path(
            Path(RESULTS_FOLDER) / f'reg_ablation_convergence_{ts}.png'
        ))
        plt.savefig(conv_path, dpi=150, bbox_inches='tight')
        print(f"✓ Convergence plot: {conv_path}")
        plt.close()
    except Exception as e:
        print(f"⚠ Failed to plot regularization ablation: {e}")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    ran = False
    if RUN_KP_EMA_ABLATION:
        run_kp_ema_ablation()
        ran = True
    if RUN_ABLATION_STUDY:
        run_ablation_study()
        ran = True
    if RUN_SEQLEN_ABLATION:
        run_seqlen_ablation()
        ran = True
    if RUN_REG_ABLATION:
        run_regularization_ablation()
        ran = True
    if not ran:
        train_model(feature_mode=FEATURE_MODE)

