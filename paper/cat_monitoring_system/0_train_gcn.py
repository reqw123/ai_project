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


from models.stgcn_model import (
    STGCN,
    interpolate_missing,
    orientation_normalize,
    flip_normalize,
    normalize_skeleton_coords,
    add_velocity_feature,
    compute_bone_feature,
    get_in_channels_for_mode,
    build_feature_tensor as shared_build_feature_tensor,
)

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
            use_attention=model_config.get('use_attention', True)
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
    config_path = os.getenv('STGCN_CONFIG_PATH', r'C:\paper\cat_monitoring_system\stgcn_config.yaml')
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
                 num_joints=17, augment=False, feature_mode="xyv",
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

        for json_file in tqdm(json_files, desc="Loading sequences"):
            video_id = json_file.stem

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            frames = data['frames']

            if not (bool(frames) and 'label' in frames[0]):
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
                    continue

                sequences.append({
                    'video_id': video_id,
                    'sequence': np.array(sequence),
                    'conf_sequence': np.array(confs[start_idx:start_idx + self.sequence_length]),
                    'label':    label_idx,
                })

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


def validate(model, dataloader, criterion, device):
    """
    Validate the model

    Returns:
        tuple: (average_loss, accuracy, predictions, labels)
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
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
    print(f"  Per-class acc: {per_class_acc}")
    # Predicted distribution for debugging class collapse
    try:
        from collections import Counter
        pred_counts = Counter(all_preds)
        distrib = {int(k): int(pred_counts.get(k, 0)) for k in range(NUM_CLASSES)}
        print(f"  Predicted distribution: {distrib}")
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
def train_model(feature_mode=FEATURE_MODE, run_name=None, run_number=None,
                shared_models_dir=None, kp_ema_alpha=None, seq_len=None):
    """Main training function"""
    in_channels = get_in_channels_for_mode(feature_mode)
    eff_alpha = kp_ema_alpha if kp_ema_alpha is not None else KP_EMA_ALPHA
    alpha_tag = f"_ema{eff_alpha:.2f}" if (eff_alpha is not None and eff_alpha < 1.0) else ""

    # Sequence-length–dependent parameters (auto-adjusted when seq_len differs from config)
    eff_seq_len       = int(seq_len) if seq_len is not None else SEQUENCE_LENGTH
    stride_ratio      = WINDOW_STRIDE / SEQUENCE_LENGTH          # preserve configured overlap ratio
    eff_window_stride = max(1, int(eff_seq_len * stride_ratio))
    eff_batch_size    = BATCH_SIZE
    seq_tag           = f"_T{eff_seq_len}" if eff_seq_len != SEQUENCE_LENGTH else ""

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
    print(f"✓ Run #{run_tag}  ({run_suffix})  KP_EMA_ALPHA={eff_alpha}  T={eff_seq_len}  stride={eff_window_stride}  batch={eff_batch_size}")

    run_results_dir = os.path.join(RESULTS_FOLDER, f"run_{run_suffix}")

    # Setup directories
    setup_directories()
    Path(run_results_dir).mkdir(parents=True, exist_ok=True)
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
        run_model_path = str(get_unique_path(Path(run_results_dir) / "best_model.pth"))

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
            'learning_rate': LEARNING_RATE,
            'optimizer': OPTIMIZER,
            'weight_decay': WEIGHT_DECAY,
            'input_dropout': INPUT_DROPOUT,
            'block_dropout': BLOCK_DROPOUT,
            'final_dropout': FINAL_DROPOUT,
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

    # ── 影片級切分（防止滑動窗 data leakage） ──────────────────────────
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
    if not use_stratify:
        print(f"  ⚠ 影片數不足以 stratify（val={n_val_vids} < classes={num_unique_labels}），改用 random split")
    train_vids, val_vids = train_test_split(
        video_ids,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=video_labels if use_stratify else None
    )

    # 確保驗證集盡量覆蓋所有可切分類別：若某類別有 >=2 支影片且 val 缺席，
    # 從 train 移 1 支該類別影片到 val（必要時再移回 1 支其他類別維持大小）。
    from collections import Counter
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

    print(f"  Videos → train: {len(train_vids)}, val: {len(val_vids)}")
    print(f"  Sequences → train: {len(train_indices)}, val: {len(val_indices)}")
    print_video_distribution("Train", train_vids)
    print_video_distribution("Validation", val_vids)

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
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * label_counts.get(c, 1)) for c in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(DEVICE)

    print(f"\n✓ Class weights: { {i: round(float(w), 3) for i, w in enumerate(class_weights.cpu())} }")

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
    model_config = {
        'num_classes': NUM_CLASSES,
        'in_channels': in_channels,
        'num_joints': NUM_JOINTS,
        'spatial_kernel_size': SPATIAL_KERNEL_SIZE,
        'temporal_kernel_size': TEMPORAL_KERNEL_SIZE,
        'num_layers': NUM_STGCN_LAYERS,
        'input_dropout': INPUT_DROPOUT,
        'block_dropout': BLOCK_DROPOUT,
        'final_dropout': FINAL_DROPOUT,
        'use_attention': use_attention,
    }
    model = STGCN(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        num_joints=NUM_JOINTS,
        spatial_kernel_size=SPATIAL_KERNEL_SIZE,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        num_layers=NUM_STGCN_LAYERS,
        input_dropout=INPUT_DROPOUT,
        block_dropout=BLOCK_DROPOUT,
        final_dropout=FINAL_DROPOUT,
        use_attention=use_attention,
    ).to(DEVICE)

    # EMA
    ema_decay = 0.999
    ema = ModelEMA(model, model_config, decay=ema_decay)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {num_params:,} trainable parameters")

    # Loss function and optimizer
    label_smoothing = float(CONFIG.get('LABEL_SMOOTHING', 0.0))
    if label_smoothing and label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    opt_name = str(OPTIMIZER).strip().lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    elif opt_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
    else:
        # default to Adam for backward compatibility
        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
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

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # EMA update
        ema.update(model)

        # Validate: optionally use EMA shadow weights for evaluation/saving
        if USE_EMA_FOR_EVAL:
            # ema.ema is the cloned EMA model kept on the same device
            val_loss, val_acc, val_macro_f1, val_preds, val_labels, val_per_class_f1 = validate(ema.ema, val_loader, criterion, DEVICE)
        else:
            val_loss, val_acc, val_macro_f1, val_preds, val_labels, val_per_class_f1 = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler step (monitor validation loss)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        # Append epoch metrics to run log
        epoch_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_macro_f1': float(val_macro_f1),
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
        'learning_rate': LEARNING_RATE,
        'optimizer': OPTIMIZER,
        'label_smoothing': float(CONFIG.get('LABEL_SMOOTHING', 0.0)),
        'train_videos': len(train_vids),
        'val_videos': len(val_vids),
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
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
    if not ran:
        train_model(feature_mode=FEATURE_MODE)

