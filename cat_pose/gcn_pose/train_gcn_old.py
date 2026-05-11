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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
            adjacency_matrix=model_config['adjacency_matrix'] if 'adjacency_matrix' in model_config else model_config['adj_matrix'],
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
            if not torch.is_floating_point(model_v):
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
SKELETON_DATA_FOLDER = r"C:\cat_pose\gcn_pose\skeletons"
LABELS_FILE = r"C:\cat_pose\gcn_pose\labels.json"  # Format: {"video_id": "label"}
MODEL_SAVE_PATH = r"C:\cat_pose\gcn_pose\models\stgcn_best.pth"
RESULTS_FOLDER = r"C:\cat_pose\gcn_pose\results"

NORMAL_LABEL = 0  # 明確定義 normal label
# Model hyperparameters
NUM_CLASSES = 4  # walk, lying, lick, shake
NUM_JOINTS = 17  # COCO-Pose has 17 keypoints
IN_CHANNELS = 2  # x, y coordinates (can be 3 if including confidence)
SEQUENCE_LENGTH = 32  # Number of frames per sequence
SPATIAL_KERNEL_SIZE = 3
TEMPORAL_KERNEL_SIZE = 9
NUM_STGCN_LAYERS = 3

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 40 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOP_PATIENCE = 15
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Setup ====================
def setup_directories():
    """Create necessary directories"""
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(MODEL_SAVE_PATH)).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created")


# ==================== Skeleton Graph Adjacency Matrix ====================
def get_skeleton_adjacency_matrix(num_joints=17):
    """
    Create adjacency matrix for skeleton graph connections
    Based on COCO-17 keypoint topology (adapted for cat pose)
    
    Keypoint indices (cat pose, 17 keypoints):
    0-2: Head (nose, left_ear, right_ear)
    3-5: Body (neck, upper_body, lower_body)
    6-7: Left front leg
    8-9: Right front leg
    10-11: Left hind leg
    12-13: Right hind leg
    14-16: Tail
    
    Returns:
        numpy array: Adjacency matrix (num_joints x num_joints)
    """
    # Define skeleton connections (edges)
    connections = [
        # Head
        (0, 1), (0, 2), (1, 2),
        # Body
        (0, 3), (3, 4), (4, 5),
        # Front limbs
        (3, 6), (6, 7), (3, 8), (8, 9),
        # Hind limbs
        (5, 10), (10, 11), (5, 12), (12, 13),
        # Tail
        (5, 14), (14, 15), (15, 16)
    ]
    
    # Create adjacency matrix
    adj_matrix = np.zeros((num_joints, num_joints), dtype=np.float32)
    
    # Add self-connections (identity matrix)
    adj_matrix += np.eye(num_joints)
    
    # Add edges
    for i, j in connections:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Undirected graph
    
    return adj_matrix


def normalize_adjacency_matrix(adj_matrix):
    """
    Normalize adjacency matrix using symmetric normalization
    D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj_matrix: Adjacency matrix
    
    Returns:
        Normalized adjacency matrix
    """
    # Calculate degree matrix
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    
    # Create diagonal matrix
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    return normalized.astype(np.float32)


# ==================== Dataset Class ====================
class CatSkeletonDataset(Dataset):
    """
    Dataset class for loading skeleton sequences
    """
    
    def __init__(self, skeleton_folder, labels_dict, sequence_length=32, 
                 in_channels=2, num_joints=17, augment=False):
        """
        Initialize dataset
        
        Args:
            skeleton_folder: Path to folder containing skeleton JSON files
            labels_dict: Dictionary mapping video_id to label
            sequence_length: Number of frames per sequence
            in_channels: Number of input channels (2 for x,y; 3 for x,y,conf)
            num_joints: Number of skeleton joints
            augment: Whether to apply data augmentation
        """
        self.skeleton_folder = Path(skeleton_folder)
        self.labels_dict = labels_dict
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.num_joints = num_joints
        self.augment = augment
        
        # Label encoding
        # Accept both int and str labels for compatibility with labels.json
        self.label_to_idx = {
            0: 0, "0": 0,
            1: 1, "1": 1,
            2: 2, "2": 2,
            3: 3, "3": 3
        }
        self.idx_to_label = {v: str(v) for v in range(4)}
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} sequences")
        self._print_class_distribution()
    
    def _load_sequences(self):
        """
        Load and segment skeleton data into fixed-length sequences.
        支援 action_intervals 標註格式，window 根據 frame 是否落在行為區間自動決定 label。
        若無 action_intervals，則 fallback 為原本的影片級標註。
        """
        sequences = []
        json_files = list(self.skeleton_folder.glob("*.json"))
        for json_file in tqdm(json_files, desc="Loading sequences"):
            video_id = json_file.stem
            # Check if we have a label for this video
            if video_id not in self.labels_dict:
                continue
            label_info = self.labels_dict[video_id]
            # 支援 action_intervals 格式（list of [start,end] 或 list of dict）
            action_intervals = None
            action_interval_labels = None
            video_level_label = None
            # labels.json 支援 {"video_id": {"label": 2, "action_intervals": [[120,170],[310,360]]}} 或 {"video_id": 2}
            if isinstance(label_info, dict):
                video_level_label = label_info.get("label", None)
                action_intervals = label_info.get("action_intervals", None)
                # 支援新版 action_intervals: list of dict with label/start/end
                if action_intervals and isinstance(action_intervals, list) and isinstance(action_intervals[0], dict):
                    # 轉成 [(start, end, label)]
                    action_interval_labels = [(d['start'], d['end'], d.get('label', video_level_label)) for d in action_intervals]
                    # 也保留舊格式給後續 fallback
                    action_intervals = [(d['start'], d['end']) for d in action_intervals]
            else:
                video_level_label = label_info
            # Try both int and str for label lookup
            if video_level_label not in self.label_to_idx:
                try:
                    label_int = int(video_level_label)
                except Exception:
                    label_int = None
                if label_int is not None and label_int in self.label_to_idx:
                    video_level_label_idx = self.label_to_idx[label_int]
                else:
                    print(f"Warning: Unknown label '{video_level_label}' for video {video_id}")
                    continue
            else:
                video_level_label_idx = self.label_to_idx[video_level_label]
            # Load skeleton data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            frames = data['frames']
            # Extract keypoint sequences (only from frames with detection)
            keypoint_frames = []
            frame_indices = []
            for idx, frame in enumerate(frames):
                if frame['detected'] and len(frame['keypoints']) > 0:
                    kpts = frame['keypoints']
                    if len(kpts) != self.num_joints:
                        continue  # Skip frames with wrong number of keypoints
                    if self.in_channels == 2:
                        coords = np.array([[kpt['x'], kpt['y']] for kpt in kpts])
                    else:  # in_channels == 3
                        coords = np.array([[kpt['x'], kpt['y'], kpt['conf']] for kpt in kpts])
                    keypoint_frames.append(coords)
                    # 用真實 frame_id 對齊物理時間軸
                    frame_indices.append(frame.get('frame_id', idx))
            if len(keypoint_frames) < self.sequence_length:
                continue  # Skip videos that are too short
            # Segment into sequences with sliding window
            stride = self.sequence_length // 2  # 50% overlap
            for start_idx in range(0, len(keypoint_frames) - self.sequence_length + 1, stride):
                sequence = keypoint_frames[start_idx:start_idx + self.sequence_length]
                sequence = np.array(sequence)  # Shape: (T, V, C)
                window_frame_indices = frame_indices[start_idx:start_idx + self.sequence_length]
                # 預設用影片級標註
                label_idx = video_level_label_idx
                # 若有 action_intervals，則根據 overlap 決定 label
                if (action_intervals is not None and len(action_intervals) > 0):
                    # 新格式: action_interval_labels = [(start, end, label)]
                    if action_interval_labels is not None:
                        # 統計每個 label 在 window 中出現的 frame 數
                        label_frame_count = {}
                        for start, end, label in action_interval_labels:
                            count = sum((start <= fidx <= end) for fidx in window_frame_indices)
                            if count > 0:
                                label_frame_count[label] = label_frame_count.get(label, 0) + count
                        if label_frame_count:
                            # 取出現最多的 label，且其 frame 數超過 50%
                            best_label, best_count = max(label_frame_count.items(), key=lambda x: x[1])
                            if best_count / self.sequence_length > 0.5:
                                # 支援 label 為字串
                                if best_label in self.label_to_idx:
                                    label_idx = self.label_to_idx[best_label]
                                else:
                                    try:
                                        label_idx = self.label_to_idx[int(best_label)]
                                    except Exception:
                                        label_idx = video_level_label_idx
                            else:
                                label_idx = NORMAL_LABEL  # normal
                        else:
                            label_idx = NORMAL_LABEL  # normal
                    else:
                        # 舊格式: list of [start, end]
                        overlap_count = 0
                        for interval in action_intervals:
                            start, end = interval
                            overlap_count += sum((start <= fidx <= end) for fidx in window_frame_indices)
                        overlap_ratio = overlap_count / self.sequence_length
                        if overlap_ratio > 0.5:
                            action_label = label_info.get("action_label", video_level_label) if isinstance(label_info, dict) else video_level_label
                            if action_label not in self.label_to_idx:
                                try:
                                    action_label_int = int(action_label)
                                except Exception:
                                    action_label_int = None
                                if action_label_int is not None and action_label_int in self.label_to_idx:
                                    label_idx = self.label_to_idx[action_label_int]
                                else:
                                    label_idx = video_level_label_idx
                            else:
                                label_idx = self.label_to_idx[action_label]
                        else:
                            label_idx = NORMAL_LABEL  # normal
                sequences.append({
                    'video_id': video_id,
                    'sequence': sequence,
                    'label': label_idx
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
        label = item['label']
        
        # Normalize coordinates (optional but recommended)
        sequence = self._normalize_skeleton(sequence)
        
        # Apply augmentation if enabled
        if self.augment:
            sequence = self._augment_sequence(sequence)
        
        # Convert to tensor and transpose to (C, T, V)
        sequence_tensor = torch.FloatTensor(sequence).permute(2, 0, 1)
        
        return sequence_tensor, label
    
    def _normalize_skeleton(self, sequence):
        """
        Normalize skeleton coordinates
        Center at origin and scale by body size
        """
        # Use body center (joint 4) as reference point
        center_joint = 4
        
        # Center the skeleton
        sequence_centered = sequence.copy()
        for t in range(len(sequence)):
            center = sequence[t, center_joint, :2]  # x, y of center joint
            sequence_centered[t, :, :2] -= center
        
        # Scale by body size (distance between neck and lower body)
        neck_joint = 3
        lower_body_joint = 5
        
        body_sizes = []
        for t in range(len(sequence)):
            body_size = np.linalg.norm(
                sequence[t, neck_joint, :2] - sequence[t, lower_body_joint, :2]
            )
            body_sizes.append(body_size)
        
        avg_body_size = np.mean(body_sizes)
        if avg_body_size > 1e-6:
            sequence_centered[:, :, :2] /= avg_body_size
        
        return sequence_centered
    
    def _augment_sequence(self, sequence):
        """
        Apply data augmentation
        - Random rotation
        - Random scaling
        - Random temporal crop
        """
        sequence_aug = sequence.copy()
        
        # Random rotation (around z-axis, in 2D it's just rotation in x-y plane)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-30, 30) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            for t in range(len(sequence_aug)):
                sequence_aug[t, :, :2] = sequence_aug[t, :, :2] @ rotation_matrix.T
        
        # Random scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            sequence_aug[:, :, :2] *= scale
        
        # Add small random noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, sequence_aug[:, :, :2].shape)
            sequence_aug[:, :, :2] += noise
        
        return sequence_aug


# ==================== ST-GCN Model ====================
class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution Layer
    Performs convolution over graph structure (skeleton joints)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, adjacency_matrix):
        super(SpatialGraphConv, self).__init__()
        
        self.kernel_size = kernel_size  # Number of spatial kernels
        
        # Convert adjacency matrix to tensor
        self.register_buffer('A', torch.FloatTensor(adjacency_matrix))
        
        # Learnable spatial kernels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1
        )
        
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)
               N: batch size, C: channels, T: time, V: vertices/joints
        
        Returns:
            Output tensor of shape (N, C_out, T, V)
        """
        N, C, T, V = x.size()
        
        # Apply spatial convolution
        x = self.conv(x)  # (N, C_out * K, T, V)
        
        # Reshape for graph convolution
        x = x.view(N, self.kernel_size, -1, T, V)  # (N, K, C_out, T, V)
        
        # Graph convolution: multiply with adjacency matrix
        # For simplicity, we use the same adjacency for all kernels
        x = torch.einsum('nkctv,vw->nkctw', x, self.A)  # (N, K, C_out, T, V)
        
        # Sum over kernels
        x = x.sum(dim=1)  # (N, C_out, T, V)
        
        # Batch normalization and activation
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class TemporalConv(nn.Module):
    """
    Temporal Convolution Layer
    Performs convolution over time dimension
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(TemporalConv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)
        
        Returns:
            Output tensor of shape (N, C_out, T, V)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class STGCNBlock(nn.Module):
    """
    ST-GCN Block: Spatial-Temporal Graph Convolution Block
    Consists of spatial graph conv followed by temporal conv
    """
    
    def __init__(self, in_channels, out_channels, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, stride=1,
                 residual=True):
        super(STGCNBlock, self).__init__()
        
        self.residual = residual
        
        # Spatial graph convolution
        self.sgc = SpatialGraphConv(
            in_channels,
            out_channels,
            spatial_kernel_size,
            adjacency_matrix
        )
        
        # Temporal convolution
        self.tcn = TemporalConv(
            out_channels,
            out_channels,
            temporal_kernel_size,
            stride
        )
        
        # Residual connection
        if not residual:
            self.residual_conv = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual_conv = lambda x: x
        else:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)
        
        Returns:
            Output tensor of shape (N, C_out, T, V)
        """
        res = self.residual_conv(x)
        x = self.sgc(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network
    For skeleton-based action recognition
    """
    
    def __init__(self, num_classes, in_channels, num_joints, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, num_layers=3):
        super(STGCN, self).__init__()
        
        # Normalize adjacency matrix
        adj_matrix = normalize_adjacency_matrix(adjacency_matrix)
        
        # Input batch normalization
        self.bn_input = nn.BatchNorm2d(in_channels)
        
        # ST-GCN layers
        self.stgcn_layers = nn.ModuleList()
        
        # Layer 1: in_channels -> 64
        self.stgcn_layers.append(
            STGCNBlock(in_channels, 64, adj_matrix, 
                      spatial_kernel_size, temporal_kernel_size, stride=1)
        )
        
        # Layer 2: 64 -> 128
        self.stgcn_layers.append(
            STGCNBlock(64, 128, adj_matrix,
                      spatial_kernel_size, temporal_kernel_size, stride=2)
        )
        
        # Layer 3+: 128 -> 128
        for _ in range(num_layers - 2):
            self.stgcn_layers.append(
                STGCNBlock(128, 128, adj_matrix,
                          spatial_kernel_size, temporal_kernel_size, stride=1)
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected classification head
        self.fc = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, C, T, V)
               N: batch size, C: channels, T: time steps, V: joints
        
        Returns:
            Output logits of shape (N, num_classes)
        """
        # Input normalization
        x = self.bn_input(x)
        
        # ST-GCN layers
        for layer in self.stgcn_layers:
            x = layer(x)
        
        # Global pooling: (N, C, T, V) -> (N, C, 1, 1)
        x = self.global_pool(x)
        
        # Flatten: (N, C, 1, 1) -> (N, C)
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x


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
    
    return avg_loss, accuracy, all_preds, all_labels


# ==================== Main Training Loop ====================
def train_model():
    """Main training function"""

    # Setup directories
    setup_directories()

    # Load labels
    if not os.path.exists(LABELS_FILE):
        print(f"✗ Labels file not found: {LABELS_FILE}")
        return
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)

    # Get adjacency matrix
    adj_matrix = get_skeleton_adjacency_matrix(NUM_JOINTS)

    # Load dataset (no augmentation)
    print("\nLoading dataset...")
    full_dataset = CatSkeletonDataset(
        SKELETON_DATA_FOLDER,
        labels_dict,
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        num_joints=NUM_JOINTS,
        augment=False
    )

    if len(full_dataset) == 0:
        print("✗ No data loaded. Please check your skeleton files and labels.")
        return

    # Split into train and validation
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=[full_dataset[i][1] for i in range(len(full_dataset))]
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # 列印訓練集與驗證集的類別分布
    def print_split_distribution(subset, name):
        from collections import Counter
        labels = [full_dataset[i][1] for i in subset.indices]
        counter = Counter(labels)
        print(f"\n{name} class distribution:")
        for cls in range(NUM_CLASSES):
            print(f"  {cls}: {counter.get(cls, 0)} sequences")

    print_split_distribution(train_dataset, "Train")
    print_split_distribution(val_dataset, "Validation")

    # Enable augmentation for training set only
    full_dataset.augment = False
    train_dataset.dataset.augment = True
    val_dataset.dataset.augment = False

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Create model
    print("\nInitializing ST-GCN model...")
    model_config = {
        'num_classes': NUM_CLASSES,
        'in_channels': IN_CHANNELS,
        'num_joints': NUM_JOINTS,
        'adjacency_matrix': adj_matrix,
        'spatial_kernel_size': SPATIAL_KERNEL_SIZE,
        'temporal_kernel_size': TEMPORAL_KERNEL_SIZE,
        'num_layers': NUM_STGCN_LAYERS
    }
    model = STGCN(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        num_joints=NUM_JOINTS,
        adjacency_matrix=adj_matrix,
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
    criterion = nn.CrossEntropyLoss()
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

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Scheduler step
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✓ Best model saved to: {MODEL_SAVE_PATH}")
            # Save confusion matrix
            plot_confusion_matrix(val_labels, val_preds)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

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
    save_path = os.path.join(RESULTS_FOLDER, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(labels, preds):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(labels, preds)
    class_names = ["walk", "lying", "lick", "shake"]
    
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
    save_path = os.path.join(RESULTS_FOLDER, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


# ==================== Main Entry Point ====================
if __name__ == "__main__":
    train_model()

