"""
ST-GCN 模型類別
優化自原 cat_behavior_stgcn.py
"""
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os

# 骨架鄰接矩陣定義
# ...existing code from get_adjacency_matrix and normalize_adjacency_matrix...

def get_adjacency_matrix():
    edges = [
        (0, 1), (0, 2), (1, 2),
        (0, 3), (3, 4), (4, 5),
        (3, 6), (6, 7), (3, 8), (8, 9),
        (5, 10), (10, 11), (5, 12), (12, 13),
        (5, 14), (14, 15), (15, 16)
    ]
    adj_matrix = np.zeros((17, 17), dtype=np.float32)
    for i, j in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    for i in range(17):
        adj_matrix[i, i] = 1
    return adj_matrix

def get_stgcn_partition_adjacency(num_joints=17):
    # Partition: root (self), close (1-hop), further (2-hop)
    A = get_adjacency_matrix()
    A_root = np.eye(num_joints, dtype=np.float32)
    A_close = (A > 0).astype(np.float32) - A_root
    A2 = np.linalg.matrix_power(A, 2)
    A_further = ((A2 > 0).astype(np.float32) - (A > 0).astype(np.float32))
    return [A_root, A_close, A_further]

def normalize_adjacency_matrix(adj_matrix):
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return normalized.astype(np.float32)

# ==================== 骨架前處理函數（訓練與推論共用） ====================
def interpolate_missing(sequence, conf, threshold=0.1):
    """sequence: (T, V, 2), conf: (T, V) — 時間軸插尼補全低信心關節點"""
    seq = sequence.copy()
    for v in range(seq.shape[1]):
        valid = conf[:, v] > threshold
        idx = np.arange(seq.shape[0])
        if not np.any(valid):
            seq[:, v, :] = 0
        else:
            seq[:, v, 0] = np.interp(idx, idx[valid], seq[valid, v, 0])
            seq[:, v, 1] = np.interp(idx, idx[valid], seq[valid, v, 1])
    return seq

def orientation_normalize(sequence):
    """sequence: (T, V, 2) — 將 mid_back(4)→hip(5) 軸旋轉至 y 軸正向。
    使用 mid_back/hip 而非 nose/hip，因為 nose 常被遮蔽導致旋轉角噪音過大。
    此函數在 flip_normalize 之後呼叫：flip 已確保 mid_back 在 hip 右側，
    旋轉後 mid_back 始終在 hip 上方（y 軸正向），骨架朝向一致。
    """
    seq = sequence.copy()
    mid_back = seq[:, 4]
    hip      = seq[:, 5]
    axis  = hip - mid_back                           # mid_back→hip 方向
    angles = np.arctan2(axis[:, 1], axis[:, 0])
    target_angle = np.pi / 2                          # 目標：軸朝向 y 軸正向
    rot_angles = target_angle - angles
    center = mid_back                                 # 以 mid_back 為旋轉中心
    for t in range(seq.shape[0]):
        R = np.array([
            [np.cos(rot_angles[t]), -np.sin(rot_angles[t])],
            [np.sin(rot_angles[t]),  np.cos(rot_angles[t])]
        ])
        seq[t] = (seq[t] - center[t]) @ R.T + center[t]
    return seq

def flip_normalize(sequence):
    """sequence: (T, V, 2) — 在原始座標下確保 mid_back(4) 在 hip(5) 右側（序列級多數決），
    必須在 orientation_normalize 之前呼叫。
    使用 mid_back/hip 而非 nose/tail，因為鼻子常被遮蔽或消失，mid_back/hip 是穩定的軀幹中央關節。"""
    seq = sequence.copy()
    mid_back_x = seq[:, 4, 0]
    hip_x      = seq[:, 5, 0]
    # 只用兩個關節都有效（非零）的幀做多數決，避免遮蔽幀污染決策
    valid = (mid_back_x != 0) | (hip_x != 0)
    if valid.sum() > 0:
        should_flip = np.mean(mid_back_x[valid] < hip_x[valid]) > 0.5
    else:
        should_flip = False
    if should_flip:
        for t in range(seq.shape[0]):
            mid_x = (seq[t, 4, 0] + seq[t, 5, 0]) / 2
            seq[t, :, 0] = 2 * mid_x - seq[t, :, 0]
    return seq

def normalize_skeleton_coords(sequence, center_joint=4, chest_joint=3, lower_body_joint=5):
    """sequence: (T, V, 2) — 中心化並以體型縮放
    center_joint=4 (Mid_Back), chest_joint=3 (Chest/前胸), lower_body_joint=5 (Hip)"""
    seq = sequence.copy()
    seq[:, :, :2] -= seq[:, center_joint:center_joint+1, :2]
    body_sizes = np.linalg.norm(
        seq[:, chest_joint, :2] - seq[:, lower_body_joint, :2], axis=1
    )
    avg_body_size = np.mean(body_sizes)
    if avg_body_size > 1e-6:
        seq[:, :, :2] /= avg_body_size
    return seq

def add_velocity_feature(sequence):
    """sequence: (T, V, 2) → (T, V, 4) 加入速度特徵"""
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return np.concatenate([sequence, velocity], axis=-1)


# ==================== 共享特徵構建函數（訓練與推論用） ====================
def compute_bone_feature(sequence):
    """sequence: (T, V, 2) -> bone_xy: (T, V, 2)
    骨架拓撲（COCO 17點貓骨架）：父節點索引
    """
    parents = np.array([0, 0, 0, 0, 3, 4, 3, 6, 3, 8, 5, 10, 5, 12, 5, 14, 15], dtype=np.int64)
    bone = sequence - sequence[:, parents, :]
    bone[:, 0, :] = 0.0
    return bone


def get_in_channels_for_mode(feature_mode):
    """根據特徵模式回傳對應的通道數"""
    mode = feature_mode.strip().lower()
    if mode == "xyv":
        return 4  # x, y, vx, vy
    if mode == "xyv_conf":
        return 5  # x, y, conf, vx, vy
    if mode == "xyv_bone":
        return 6  # x, y, vx, vy, bone_x, bone_y
    if mode == "xyv_conf_bone":
        return 7  # x, y, conf, vx, vy, bone_x, bone_y
    raise ValueError(f"Unknown feature mode: {feature_mode}")


def build_feature_tensor(sequence_xy, conf_seq, feature_mode):
    """
    共享的特徵構建函數（訓練與推論皆用）
    
    Args:
        sequence_xy: (T, V, 2) 已正規化的關鍵點座標
        conf_seq:    (T, V) 信心值序列
        feature_mode: "xyv" | "xyv_conf" | "xyv_bone" | "xyv_conf_bone"
    
    Returns:
        (T, V, C) 特徵張量，通道順序始終為：x, y, conf, vx, vy, bone_x, bone_y
    """
    mode = feature_mode.strip().lower()
    
    # 計算基礎特徵
    velocity = np.zeros_like(sequence_xy)
    velocity[1:] = sequence_xy[1:] - sequence_xy[:-1]
    
    conf_channel = conf_seq.astype(sequence_xy.dtype, copy=False)[..., None]  # (T, V, 1)
    bone_xy = compute_bone_feature(sequence_xy).astype(sequence_xy.dtype, copy=False)  # (T, V, 2)
    
    # 根據模式組合特徵通道
    if mode == "xyv":
        # (T, V, 4): x, y, vx, vy
        return np.concatenate([sequence_xy, velocity], axis=-1)
    
    elif mode == "xyv_conf":
        # (T, V, 5): x, y, conf, vx, vy
        return np.concatenate([sequence_xy, conf_channel, velocity], axis=-1)
    
    elif mode == "xyv_bone":
        # (T, V, 6): x, y, vx, vy, bone_x, bone_y
        return np.concatenate([sequence_xy, velocity, bone_xy], axis=-1)
    
    elif mode == "xyv_conf_bone":
        # (T, V, 7): x, y, conf, vx, vy, bone_x, bone_y
        base = np.concatenate([sequence_xy, conf_channel, velocity], axis=-1)  # (T, V, 5)
        return np.concatenate([base, bone_xy], axis=-1)  # (T, V, 7)
    
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")


# ==================== 關鍵點重要性加權 ====================
# 目的：降低非核心關鍵點對行為判斷的稀釋，特別讓 shake 更聚焦在鼻子與雙耳。
DEFAULT_JOINT_IMPORTANCE = np.array([
    1.70,  # 0 Nose
    1.65,  # 1 Left_Ear
    1.65,  # 2 Right_Ear
    1.15,  # 3 Chest
    0.95,  # 4 Mid_Back
    1.05,  # 5 Hip
    0.85,  # 6 LF_Elbow
    0.85,  # 7 LF_Paw
    0.85,  # 8 RF_Elbow
    0.85,  # 9 RF_Paw
    0.80,  # 10 LH_Knee
    0.80,  # 11 LH_Paw
    0.80,  # 12 RH_Knee
    0.80,  # 13 RH_Paw
    0.75,  # 14 Tail_Root
    0.75,  # 15 Tail_Mid
    0.75,  # 16 Tail_Tip
], dtype=np.float32)

SHAKE_FOCUS_JOINT_IMPORTANCE = np.array([
    2.00,  # 0 Nose
    1.90,  # 1 Left_Ear
    1.90,  # 2 Right_Ear
    0.90,  # 3 Chest
    0.70,  # 4 Mid_Back
    0.80,  # 5 Hip
    0.55,  # 6 LF_Elbow
    0.55,  # 7 LF_Paw
    0.55,  # 8 RF_Elbow
    0.55,  # 9 RF_Paw
    0.50,  # 10 LH_Knee
    0.50,  # 11 LH_Paw
    0.50,  # 12 RH_Knee
    0.50,  # 13 RH_Paw
    0.45,  # 14 Tail_Root
    0.45,  # 15 Tail_Mid
    0.45,  # 16 Tail_Tip
], dtype=np.float32)

HEAD_FOCUSED_JOINT_PRIOR = np.array([
    2.20,  # 0 Nose
    2.00,  # 1 Left_Ear
    2.00,  # 2 Right_Ear
    1.00,  # 3 Chest
    0.85,  # 4 Mid_Back
    0.90,  # 5 Hip
    0.65,  # 6 LF_Elbow
    0.65,  # 7 LF_Paw
    0.65,  # 8 RF_Elbow
    0.65,  # 9 RF_Paw
    0.60,  # 10 LH_Knee
    0.60,  # 11 LH_Paw
    0.60,  # 12 RH_Knee
    0.60,  # 13 RH_Paw
    0.55,  # 14 Tail_Root
    0.55,  # 15 Tail_Mid
    0.55,  # 16 Tail_Tip
], dtype=np.float32)


def apply_joint_importance(sequence, focus="default"):
    """對每個關鍵點套用固定重要性權重。"""
    seq = np.asarray(sequence)
    if seq.ndim != 3 or seq.shape[1] < 17:
        return seq.copy()

    weights = DEFAULT_JOINT_IMPORTANCE if str(focus).lower() != "shake" else SHAKE_FOCUS_JOINT_IMPORTANCE
    weights = weights[:seq.shape[1]].astype(seq.dtype, copy=False)
    return seq * weights[None, :, None]


def get_joint_attention_prior(num_joints=17):
    """回傳頭部優先的關節先驗權重，供 attention 使用。"""
    prior = HEAD_FOCUSED_JOINT_PRIOR[:num_joints].astype(np.float32, copy=False)
    if prior.shape[0] < num_joints:
        pad = np.ones((num_joints - prior.shape[0],), dtype=np.float32)
        prior = np.concatenate([prior, pad], axis=0)
    prior = prior / max(float(np.mean(prior)), 1e-6)
    return prior

# ST-GCN 網絡元件
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, adj_matrices):
        super().__init__()
        self.K = len(adj_matrices)
        self.register_buffer('A', torch.stack([torch.tensor(a, dtype=torch.float32) for a in adj_matrices]))
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.K, -1, T, V)
        # 批次處理版本：更高效且與 train_gcn.py 一致
        x = torch.einsum('nkctv,kvw->nkctw', x, self.A)  # (N, K, C_out, T, V)
        x = x.sum(dim=1)  # (N, C_out, T, V)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.sgc = SpatialGraphConv(in_channels, out_channels, spatial_kernel_size, adjacency_matrix)
        self.tcn = TemporalConv(out_channels, out_channels, temporal_kernel_size, stride)
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
        res = self.residual_conv(x)
        x = self.sgc(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x

class STGCN(nn.Module):
    def __init__(self, num_classes=4, in_channels=4, num_joints=17,
                 spatial_kernel_size=3, temporal_kernel_size=9, num_layers=3):
        super().__init__()
        adj_matrices = get_stgcn_partition_adjacency(num_joints)
        adj_matrices = [normalize_adjacency_matrix(a) for a in adj_matrices]
        self.bn_input = nn.BatchNorm2d(in_channels)
        # 可學習的 per-sample per-frame joint attention 模組
        # 保持為通用 attention，不直接偏置任何類別，避免影響 walk / lick / scratch
        class JointAttention(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                # x: (N, C, T, V) -> conv -> (N, 1, T, V)
                w = self.conv(x)
                return self.sigmoid(w)

        self.joint_attention = JointAttention(in_channels)
        # Allow disabling attention via environment variable for experiments
        use_attention = os.getenv('STGCN_USE_ATTENTION', '1').strip()
        if use_attention.lower() in {'0', 'false', 'no', 'off'}:
            try:
                self.joint_attention = nn.Identity()
            except Exception:
                pass

        # Shake-only head branch: only affects the shake logit, leaving other classes untouched.
        use_shake_head = os.getenv('STGCN_USE_SHAKE_HEAD', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        self.use_shake_head = use_shake_head
        self.shake_class_idx = 3
        self.shake_bias_scale = float(os.getenv('STGCN_SHAKE_BIAS_SCALE', '0.5'))
        if self.use_shake_head:
            head_mask = torch.zeros(num_joints, dtype=torch.float32)
            head_mask[:3] = 1.0  # Nose / Left_Ear / Right_Ear
            self.register_buffer('shake_head_mask', head_mask.view(1, 1, 1, num_joints))
            self.shake_head_branch = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.shake_head_fc = nn.Linear(16, 1)
        else:
            self.shake_head_branch = None
            self.shake_head_fc = None
        self.stgcn_layers = nn.ModuleList()
        self.stgcn_layers.append(
            STGCNBlock(in_channels, 64, adj_matrices, spatial_kernel_size, temporal_kernel_size, stride=1)
        )
        self.stgcn_layers.append(
            STGCNBlock(64, 128, adj_matrices, spatial_kernel_size, temporal_kernel_size, stride=2)
        )
        for _ in range(num_layers - 2):
            self.stgcn_layers.append(
                STGCNBlock(128, 128, adj_matrices, spatial_kernel_size, temporal_kernel_size, stride=1)
            )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        bn_x = self.bn_input(x)
        # 由 JointAttention 決定每個樣本、每個關節的縮放係數並套用
        try:
            attn = self.joint_attention(bn_x)
            x = bn_x * attn
        except Exception:
            # 若模型是舊版 checkpoint，joint_attention 可能尚未被訓練或有 shape 差異，
            # 我們容錯處理：若出錯則跳過 attention，不阻斷推論/訓練流程。
            x = bn_x

        shake_bias = None
        if self.use_shake_head and self.shake_head_branch is not None and self.shake_head_fc is not None:
            head_x = bn_x * self.shake_head_mask[:, :, :, :bn_x.shape[-1]]
            head_feat = self.shake_head_branch(head_x)
            head_feat = head_feat.view(head_feat.size(0), -1)
            shake_bias = torch.tanh(self.shake_head_fc(head_feat)) * self.shake_bias_scale
        for layer in self.stgcn_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)

        # 只對 shake 類別增加 head-only 修正，不影響其他三類別的 logits
        if shake_bias is not None:
            logits[:, self.shake_class_idx] = logits[:, self.shake_class_idx] + shake_bias.squeeze(1)
        return logits

# 包裝器
class CatBehaviorSTGCN:
    def __init__(self, model_path, device='cuda', sequence_length=32, num_classes=4, normalize=True, feature_mode='xyv', in_channels=None):
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.normalize = normalize
        self.feature_mode = feature_mode.strip().lower()
        self.in_channels = int(in_channels) if in_channels is not None else None

        checkpoint = None
        state_dict = None
        if Path(model_path).exists():
            # Use safe loading when available (PyTorch experimental 'weights_only' flag).
            try:
                sig = inspect.signature(torch.load)
                if 'weights_only' in sig.parameters:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device)
            except Exception:
                # Fallback to normal load if signature inspection fails
                checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

        if self.in_channels is None and state_dict is not None and 'bn_input.weight' in state_dict:
            self.in_channels = int(state_dict['bn_input.weight'].shape[0])
        if self.in_channels is None:
            self.in_channels = 4

        expected_channels = get_in_channels_for_mode(self.feature_mode)
        if self.in_channels != expected_channels:
            raise ValueError(
                f"feature_mode={self.feature_mode} 對應 {expected_channels} channels，但目前 in_channels={self.in_channels}; "
                "請確認訓練與推論設定一致"
            )

        self.model = STGCN(
            num_classes=num_classes,
            in_channels=self.in_channels,
            num_joints=17,
            spatial_kernel_size=3,
            temporal_kernel_size=9,
            num_layers=3
        ).to(self.device)
        if state_dict is not None:
            # 允許舊版 checkpoint 在沒有 joint_attention 權重時載入（strict=False）
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ ST-GCN 模型已載入: {model_path}")
            print(f"  in_channels={self.in_channels}, feature_mode={self.feature_mode}")
        else:
            print(f"⚠ 警告：模型檔案未找到 {model_path}")
        self.model.eval()
    def normalize_keypoints(self, keypoints_sequence):
        # flip 必須在 orientation 之前：原始座標下 walk 貓的 nose_x 與 tail_x 差距大，翻轉決策穩定
        seq = flip_normalize(keypoints_sequence)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        seq = add_velocity_feature(seq)
        return seq
    def predict(self, keypoints_sequence, precomputed=False):
        if keypoints_sequence.shape[0] < self.sequence_length:
            return None, 0.0, np.zeros(self.num_classes)

        seq_window = keypoints_sequence[-self.sequence_length:].copy()

        if precomputed:
            if seq_window.ndim != 3:
                raise ValueError(f"precomputed=True 時輸入需為 (T,V,C)，目前 shape={seq_window.shape}")
            if seq_window.shape[-1] != self.in_channels:
                raise ValueError(
                    f"預計 in_channels={self.in_channels}，但收到 C={seq_window.shape[-1]}，請檢查訓練/推論特徵通道是否一致"
                )
            seq_features = seq_window
        else:
            if self.in_channels != 4:
                raise ValueError(
                    f"當前模型 in_channels={self.in_channels}，需提供 precomputed=True 的特徵輸入"
                )

            seq = seq_window[:, :, :2]
            if self.normalize:
                seq_features = self.normalize_keypoints(seq)
            else:
                seq_features = add_velocity_feature(seq)
                print("[DEBUG] 正規化已停用 — 使用原始座標")

        seq_tensor = torch.FloatTensor(seq_features).permute(2, 0, 1).unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        behavior_id = int(np.argmax(probs))
        confidence = float(probs[behavior_id])
        return behavior_id, confidence, probs
    def __call__(self, keypoints_sequence):
        return self.predict(keypoints_sequence)

def load_stgcn_model(model_path, device='cuda', sequence_length=32, normalize=True, feature_mode='xyv', in_channels=None):
    return CatBehaviorSTGCN(
        model_path,
        device=device,
        sequence_length=sequence_length,
        normalize=normalize,
        feature_mode=feature_mode,
        in_channels=in_channels,
    )
