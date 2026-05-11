"""
ST-GCN Training Script for Cat Behavior Classification
=======================================================
This script trains a Spatial-Temporal Graph Convolutional Network (ST-GCN)
to classify cat behaviors: normal, scratch, headshake, lick

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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
            num_layers=model_config.get('num_layers', 3)
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
SKELETON_DATA_FOLDER = r"C:\AI_Project\cat_pose\gcn_pose\skeletons"
MODEL_SAVE_PATH = r"C:\AI_Project\cat_pose\gcn_pose\models\stgcn_best.pth"
RESULTS_FOLDER = r"C:\AI_Project\cat_monitoring_system"

# 滑動窗純度門殿：窗口內同一標籤的帧數比例短於此即跳過（mode 2 標記語料專用）
LABEL_PURITY_THRESHOLD = 0.8
# Relaxed window policy: include any window that contains labeled frames.
# Set STGCN_STRICT_WINDOW_FILTER=1 to restore old strict behavior.
STRICT_WINDOW_FILTER = os.getenv("STGCN_STRICT_WINDOW_FILTER", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

# 關鍵點 EMA 平滑（與推論腳本 test_video_inference_ema.py 的 EMA_ALPHA 保持一致）
# 設為 None 或 1.0 可完全停用
KP_EMA_ALPHA = 1.0
# Model hyperparameters
NUM_CLASSES = 4  # walk, lick, scratch, shake
# 檔名前綴 → label index（需與 NUM_CLASSES 順序一致）
BEHAVIOR_PREFIXES = {
    'walk':    0,
    'lick':    1,
    'scratch': 2,
    'shake':   3,
}
NUM_JOINTS = 17  # custom cat skeleton keypoints
IN_CHANNELS = 4  # x, y, vx, vy
SEQUENCE_LENGTH = 16  # Number of frames per sequence
SPATIAL_KERNEL_SIZE = 3
TEMPORAL_KERNEL_SIZE = 9
NUM_STGCN_LAYERS = 3

# Feature mode:
#   xyv            -> x,y,vx,vy (baseline)
#   xyv_conf       -> x,y,conf,vx,vy
#   xyv_bone       -> x,y,vx,vy,bone_x,bone_y
#   xyv_conf_bone  -> x,y,conf,vx,vy,bone_x,bone_y
FEATURE_MODE = os.getenv("STGCN_FEATURE_MODE", "xyv").strip().lower()
RUN_ABLATION_STUDY = os.getenv("STGCN_RUN_ABLATION", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ABLATION_MODES = ["xyv", "xyv_conf", "xyv_bone", "xyv_conf_bone"]

# Training hyperparameters
BATCH_SIZE = int(os.getenv("STGCN_BATCH_SIZE", "8"))
NUM_EPOCHS = int(os.getenv("STGCN_NUM_EPOCHS", "50"))
LEARNING_RATE = float(os.getenv("STGCN_LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("STGCN_WEIGHT_DECAY", "0.0001"))
EARLY_STOP_PATIENCE = int(os.getenv("STGCN_EARLY_STOP", "10"))
TRAIN_TEST_SPLIT = float(os.getenv("STGCN_TRAIN_TEST_SPLIT", "0.2"))
RANDOM_SEED = int(os.getenv("STGCN_RANDOM_SEED", "42"))

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Setup ====================
def setup_directories():
    """Create necessary directories"""
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(MODEL_SAVE_PATH)).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created")


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


# 使用共享的特徵構建函數（從 models.stgcn_model 導入）
def build_feature_tensor(sequence_xy, conf_seq, feature_mode, label=None):
    """包裝函數，用於與既有代碼兼容（label 參數已廢棄）"""
    return shared_build_feature_tensor(sequence_xy, conf_seq, feature_mode)

class CatSkeletonDataset(Dataset):
    """
    Dataset class for loading skeleton sequences
    """
    
    def __init__(self, skeleton_folder, sequence_length=32,
                 in_channels=2, num_joints=17, augment=False, feature_mode="xyv"):
        self.skeleton_folder = Path(skeleton_folder)
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.augment = augment
        self.feature_mode = feature_mode
        self.idx_to_label = {v: str(v) for v in range(4)}
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences")
        self._print_class_distribution()
    
    def _load_sequences(self):
        """以逐幀標籤（手動/整段標記）載入並切割成固定長度序列。"""
        from collections import Counter

        sequences = []
        json_files = list(self.skeleton_folder.glob("*.json"))
        name_to_idx = {'walk': 0, 'lick': 1, 'scratch': 2, 'shake': 3}

        for json_file in tqdm(json_files, desc="Loading sequences"):
            video_id = json_file.stem

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            frames = data['frames']

            if not (bool(frames) and 'label' in frames[0]):
                continue  # 無逐幀標籤，略過

            keypoint_frames   = []
            frame_labels_list = []
            for frame in frames:
                kpts_list = frame.get('keypoints', [])
                if len(kpts_list) == self.num_joints:
                    coords = np.array([[kpt['x'], kpt['y']] for kpt in kpts_list])
                    conf   = np.array([kpt.get('conf', 1.0) for kpt in kpts_list])
                else:
                    coords = np.zeros((self.num_joints, 2), dtype=np.float32)
                    conf   = np.zeros((self.num_joints,),  dtype=np.float32)
                keypoint_frames.append(coords)
                frame_labels_list.append(frame.get('label', 'unannotated'))

            keypoint_frames = np.array(keypoint_frames)
            confs = np.array([
                np.array([kpt.get('conf', 1.0) for kpt in frame.get('keypoints', [])])
                if len(frame.get('keypoints', [])) == self.num_joints
                else np.zeros((self.num_joints,))
                for frame in frames
            ])
            keypoint_frames = interpolate_missing(keypoint_frames, confs)

            # EMA 平滑：在插值補全後、滑動切窗前對整段影片套用，與推論行為一致
            # 必須在切窗前套，否則跨窗的平滑狀態不連貫
            if KP_EMA_ALPHA is not None and 0.0 < KP_EMA_ALPHA < 1.0:
                for t in range(1, len(keypoint_frames)):
                    keypoint_frames[t] = (
                        KP_EMA_ALPHA * keypoint_frames[t]
                        + (1.0 - KP_EMA_ALPHA) * keypoint_frames[t - 1]
                    )

            if len(keypoint_frames) < self.sequence_length:
                continue

            stride = self.sequence_length // 2
            for start_idx in range(0, len(keypoint_frames) - self.sequence_length + 1, stride):
                sequence      = keypoint_frames[start_idx:start_idx + self.sequence_length]
                window_labels = frame_labels_list[start_idx:start_idx + self.sequence_length]

                if STRICT_WINDOW_FILTER:
                    # Old behavior: drop mixed windows with any unannotated frame.
                    if 'unannotated' in window_labels:
                        continue
                    label_counts = Counter(window_labels)
                    best_label, best_count = label_counts.most_common(1)[0]
                    if best_count / self.sequence_length < LABEL_PURITY_THRESHOLD:
                        continue  # 行為轉換邊界，跳過
                else:
                    # Relaxed behavior: keep any window with at least one annotated frame.
                    annotated_labels = [lbl for lbl in window_labels if lbl != 'unannotated']
                    if not annotated_labels:
                        continue
                    label_counts = Counter(annotated_labels)
                    best_label, _ = label_counts.most_common(1)[0]

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
            label_name = self.idx_to_label[label_idx]
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
        sequence = item['sequence']  # Shape: (T, V, C)
        conf_seq = item['conf_sequence']  # Shape: (T, V)
        label = item['label']
        
        # Normalize: flip → orientation → center/scale (shared with inference)
        # flip 先於 orientation：原始座標下 walk 貓的 nose_x/tail_x 差距最大，翻轉決策穩定
        sequence = flip_normalize(sequence)
        sequence = orientation_normalize(sequence)
        sequence = normalize_skeleton_coords(sequence)
        # Temporal augmentation (training only)
        if self.augment:
            sequence, conf_seq = temporal_augment(sequence, conf=conf_seq)
        # Build feature channels according to selected mode
        sequence = build_feature_tensor(sequence, conf_seq, self.feature_mode, label=label)
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
    
    # 每類別準確率
    from collections import Counter
    label_names = ['walk', 'lick', 'scratch', 'shake']
    per_class_correct = Counter()
    per_class_total   = Counter()
    for pred, true in zip(all_preds, all_labels):
        per_class_total[true] += 1
        if pred == true:
            per_class_correct[true] += 1
    per_class_acc = {
        label_names[c]: f"{per_class_correct[c]}/{per_class_total[c]} "
                        f"({100*per_class_correct[c]/per_class_total[c]:.0f}%)"
        if per_class_total[c] > 0 else "N/A"
        for c in range(4)
    }
    print(f"  Per-class acc: {per_class_acc}")

    return avg_loss, accuracy, macro_f1, all_preds, all_labels


# ==================== Main Training Loop ====================
def train_model(feature_mode=FEATURE_MODE, run_name=None):
    """Main training function"""
    in_channels = get_in_channels_for_mode(feature_mode)
    
    # 自動添加 att_on/att_off / shake_on/off 後綴（保留 run_name 時仍自動加尾綴，避免覆寫）
    use_attention = os.getenv('STGCN_USE_ATTENTION','1').strip().lower() in {"1", "true", "yes", "y", "on"}
    att_suffix = "att_on" if use_attention else "att_off"
    use_shake_head = os.getenv('STGCN_USE_SHAKE_HEAD','1').strip().lower() in {"1", "true", "yes", "y", "on"}
    shake_suffix = "shake_on" if use_shake_head else "shake_off"
    
    if run_name:
        # 若使用者已指定 run_name，在其後添加 att_on/att_off 與 shake_on/off
        run_suffix = f"{run_name}_{att_suffix}_{shake_suffix}"
    else:
        # 若未指定，使用 feature_mode + att_suffix + shake_suffix
        run_suffix = f"{feature_mode}_{att_suffix}_{shake_suffix}"
    
    run_results_dir = os.path.join(RESULTS_FOLDER, f"run_{run_suffix}")
    model_root = Path(MODEL_SAVE_PATH)
    run_model_path = str(model_root.with_name(f"{model_root.stem}_{run_suffix}{model_root.suffix}"))

    # Setup directories
    setup_directories()
    Path(run_results_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset (no augmentation)
    print("\nLoading dataset...")
    full_dataset = CatSkeletonDataset(
        SKELETON_DATA_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        in_channels=in_channels,
        num_joints=NUM_JOINTS,
        augment=False,
        feature_mode=feature_mode,
    )

    if len(full_dataset) == 0:
        print("✗ No data loaded. Please check your skeleton files and labels.")
        return

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
    label_names = ['walk', 'lick', 'scratch', 'shake']

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

    # Create data loaders
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
        'num_layers': NUM_STGCN_LAYERS
    }
    model = STGCN(
        num_classes=NUM_CLASSES,
        in_channels=in_channels,
        num_joints=NUM_JOINTS,
        spatial_kernel_size=SPATIAL_KERNEL_SIZE,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        num_layers=NUM_STGCN_LAYERS
    ).to(DEVICE)

    # EMA
    ema_decay = 0.999
    ema = ModelEMA(model, model_config, decay=ema_decay)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {num_params:,} trainable parameters")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_state_dict = None       # 記憶體暫存最佳權重，訓練結束後一次寫檔
    best_val_preds = None
    best_val_labels = None
    best_val_macro_f1 = 0.0
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

        # ── 直接用真實模型驗證（反映真實學習進度）──────────────────────
        # EMA 僅用於儲存最佳 checkpoint，不用於每 epoch 的 loss 監控
        val_loss, val_acc, val_macro_f1, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler step
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

        # Early stopping（acc 相同時以 val_loss 更低為準，確保儲存最佳 checkpoint）
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1
            best_state_dict = copy.deepcopy(model.state_dict())  # 暫存於記憶體，不寫檔
            best_val_preds = val_preds
            best_val_labels = val_labels
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
        'in_channels': in_channels,
        'best_val_acc': float(best_val_acc),
        'best_val_macro_f1': float(best_val_macro_f1),
        'best_val_loss': float(best_val_loss),
        'model_path': run_model_path,
        'results_dir': run_results_dir,
    }


def plot_confusion_matrix(labels, preds, output_dir):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    class_names = ["walk", "lick", "scratch", "shake"]
    
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
    modes = modes or ABLATION_MODES
    summary = []
    print("\n" + "=" * 70)
    print("Feature Ablation Study")
    print("=" * 70)
    for mode in modes:
        print("\n" + "-" * 70)
        print(f"Running mode: {mode}")
        print("-" * 70)
        result = train_model(feature_mode=mode, run_name=mode)
        if result is not None:
            summary.append(result)

    if not summary:
        print("✗ No ablation results generated.")
        return

    summary_csv = os.path.join(RESULTS_FOLDER, 'ablation_summary.csv')
    import csv
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'feature_mode',
                'in_channels',
                'best_val_acc',
                'best_val_macro_f1',
                'best_val_loss',
                'model_path',
                'results_dir',
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 70)
    print("Ablation Summary")
    print("=" * 70)
    for rec in summary:
        print(
            f"{rec['feature_mode']:>14s} | C={rec['in_channels']} | "
            f"Acc={rec['best_val_acc']:.4f} | Macro-F1={rec['best_val_macro_f1']:.4f} | "
            f"Loss={rec['best_val_loss']:.4f}"
        )
    print(f"\n✓ Summary CSV saved to: {summary_csv}")


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    if RUN_ABLATION_STUDY:
        run_ablation_study()
    else:
        train_model(feature_mode=FEATURE_MODE)

