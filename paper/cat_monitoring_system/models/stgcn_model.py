"""
ST-GCN 模型類別
優化自原 cat_behavior_stgcn.py
"""
import logging
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

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
    # Directional partition (PYSKL-style): center = mid_back (node 4).
    # BFS hop distances from center determine whether each neighbor is
    # centripetal (toward trunk) or centrifugal (toward extremities).
    # Normalization is applied by the caller (normalize_adjacency_matrix).
    center_node = 4

    A_full = get_adjacency_matrix()[:num_joints, :num_joints]
    A_edges = A_full.copy()
    np.fill_diagonal(A_edges, 0)  # remove self-loops for BFS

    # BFS from center_node
    dist = np.full(num_joints, np.inf)
    dist[center_node] = 0.0
    queue = [center_node]
    while queue:
        node = queue.pop(0)
        for nb in np.where(A_edges[node] > 0)[0]:
            if np.isinf(dist[nb]):
                dist[nb] = dist[node] + 1
                queue.append(nb)

    A_self = np.eye(num_joints, dtype=np.float32)
    A_centripetal = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_centrifugal = np.zeros((num_joints, num_joints), dtype=np.float32)

    for i in range(num_joints):
        for j in range(num_joints):
            if i == j or A_edges[i, j] == 0:
                continue
            # For edge (i→j): j is centripetal to i when j is closer to center
            if dist[j] < dist[i]:
                A_centripetal[j, i] = 1.0
            else:  # dist[j] >= dist[i]: same distance counts as centrifugal (per PYSKL)
                A_centrifugal[j, i] = 1.0

    return [A_self, A_centripetal, A_centrifugal]

def normalize_adjacency_matrix(adj_matrix):
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    normalized = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    return normalized.astype(np.float32)


def _normalize_temporal_kernel_sizes(temporal_kernel_size):
    if isinstance(temporal_kernel_size, (list, tuple, np.ndarray)):
        kernel_sizes = tuple(int(k) for k in temporal_kernel_size)
    else:
        kernel_sizes = (int(temporal_kernel_size),)
    if len(kernel_sizes) == 0:
        raise ValueError("temporal_kernel_size must contain at least one kernel size")
    return kernel_sizes

# ==================== 骨架前處理函數（訓練與推論共用） ====================
def interpolate_missing(sequence, conf, threshold=0.1):
    # threshold=0.1 保留弱訊號供時間插值；動作分數過濾使用更嚴格的 AnomalyDetectionConfig.KP_CONF_THRES=0.5
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
    valid = (mid_back_x != 0) & (hip_x != 0)
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
    _parents_17 = np.array([0, 0, 0, 0, 3, 4, 3, 6, 3, 8, 5, 10, 5, 12, 5, 14, 15], dtype=np.int64)
    parents = _parents_17[:sequence.shape[1]]
    bone = sequence - sequence[:, parents, :]
    bone[:, 0, :] = 0.0
    return bone


def compute_bone_motion_feature(sequence):
    """sequence: (T, V, 2) -> bone_motion_xy: (T, V, 2)
    骨向量的時間差分特徵，第一幀為 0。
    """
    bone = compute_bone_feature(sequence)
    bone_motion = np.zeros_like(bone)
    bone_motion[1:] = bone[1:] - bone[:-1]
    return bone_motion


# 五種特徵模式，名稱直接反映所含特徵標籤：
#   xy=位置, conf=信心值, v=速度, bone=骨架向量, bmotion=骨架位移
FEATURE_MODE_CHANNELS = {
    "xy":                     2,   # x, y
    "xy_conf":                3,   # x, y, conf
    "xy_conf_v":              5,   # x, y, conf, vx, vy
    "xy_conf_v_bone":         7,   # x, y, conf, vx, vy, bone_x, bone_y
    "xy_conf_v_bone_bmotion": 9,   # x, y, conf, vx, vy, bone_x, bone_y, bone_mx, bone_my
}


def get_in_channels_for_mode(feature_mode):
    """根據特徵模式回傳對應的通道數"""
    mode = feature_mode.strip().lower()
    if mode in FEATURE_MODE_CHANNELS:
        return FEATURE_MODE_CHANNELS[mode]
    raise ValueError(f"Unknown feature mode: {feature_mode!r}. 支援模式: {list(FEATURE_MODE_CHANNELS)}")


def build_feature_tensor(sequence_xy, conf_seq, feature_mode):
    """
    共享的特徵構建函數（訓練與推論皆用）

    Args:
        sequence_xy:  (T, V, 2) 已正規化的關鍵點座標
        conf_seq:     (T, V)   信心值序列
        feature_mode: "xy" | "xy_conf" | "xy_conf_v" | "xy_conf_v_bone" | "xy_conf_v_bone_bmotion"

    Returns:
        (T, V, C) 特徵張量，通道順序依模式而定；
        骨向量與骨 motion 均建立在正規化後座標上
    """
    mode = feature_mode.strip().lower()

    velocity = np.zeros_like(sequence_xy)
    velocity[1:] = sequence_xy[1:] - sequence_xy[:-1]

    conf_channel   = conf_seq.astype(sequence_xy.dtype, copy=False)[..., None]           # (T,V,1)
    bone_xy        = compute_bone_feature(sequence_xy).astype(sequence_xy.dtype, copy=False)       # (T,V,2)
    bone_motion_xy = compute_bone_motion_feature(sequence_xy).astype(sequence_xy.dtype, copy=False) # (T,V,2)

    if mode == "xy":
        # 2ch: x, y
        return sequence_xy.copy()

    elif mode == "xy_conf":
        # 3ch: x, y, conf
        return np.concatenate([sequence_xy, conf_channel], axis=-1)

    elif mode == "xy_conf_v":
        # 5ch: x, y, conf, vx, vy
        return np.concatenate([sequence_xy, conf_channel, velocity], axis=-1)

    elif mode == "xy_conf_v_bone":
        # 7ch: x, y, conf, vx, vy, bone_x, bone_y
        base = np.concatenate([sequence_xy, conf_channel, velocity], axis=-1)
        return np.concatenate([base, bone_xy], axis=-1)

    elif mode == "xy_conf_v_bone_bmotion":
        # 9ch: x, y, conf, vx, vy, bone_x, bone_y, bone_mx, bone_my
        base = np.concatenate([sequence_xy, conf_channel, velocity], axis=-1)
        return np.concatenate([base, bone_xy, bone_motion_xy], axis=-1)

    else:
        raise ValueError(f"Unknown feature mode: {feature_mode!r}. 支援模式: {list(FEATURE_MODE_CHANNELS)}")


# Module-level JointAttention: 1x1 conv -> sigmoid producing (N,1,T,V)
class JointAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        w = self.conv(x)
        return self.sigmoid(w)

class _ZeroResidual(nn.Module):
    """殘差分支回傳全零張量——用於 residual=False 時，可序列化且設備感知。"""
    def forward(self, x):
        return torch.zeros_like(x)


# ST-GCN 網絡元件
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, adj_matrices):
        super().__init__()
        self.K = len(adj_matrices)
        self.register_buffer('A', torch.stack([torch.tensor(a, dtype=torch.float32) for a in adj_matrices]))
        # Learnable importance per graph partition; low-cost and often effective.
        self.partition_importance = nn.Parameter(torch.ones(self.K, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.K, -1, T, V)
        # 批次處理版本：更高效且與 train_gcn.py 一致
        x = torch.einsum('nkctv,kvw->nkctw', x, self.A)  # (N, K, C_out, T, V)
        x = x * self.partition_importance.view(1, self.K, 1, 1, 1)
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


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 9), stride=1):
        super().__init__()
        self.kernel_sizes = _normalize_temporal_kernel_sizes(kernel_sizes)
        self.branches = nn.ModuleList([
            TemporalConv(in_channels, out_channels, kernel_size, stride=stride)
            for kernel_size in self.kernel_sizes
        ])
        self.branch_logits = nn.Parameter(torch.zeros(len(self.branches), dtype=torch.float32))

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        branch_weights = torch.softmax(self.branch_logits, dim=0)
        x = torch.zeros_like(branch_outputs[0])
        for weight, branch_output in zip(branch_weights, branch_outputs):
            x = x + weight * branch_output
        return x

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix,
                 spatial_kernel_size=3, temporal_kernel_size=9, stride=1, residual=True,
                 dropout=0.15):
        super().__init__()
        self.residual = residual
        self.sgc = SpatialGraphConv(in_channels, out_channels, spatial_kernel_size, adjacency_matrix)
        self.tcn = MultiScaleTemporalConv(out_channels, out_channels, temporal_kernel_size, stride)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        if not residual:
            self.residual_conv = _ZeroResidual()
        elif in_channels == out_channels and stride == 1:
            self.residual_conv = nn.Identity()
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
        x = self.dropout(x)
        return x

class STGCN(nn.Module):
    def __init__(self, num_classes=5, in_channels=4, num_joints=17,
                 spatial_kernel_size=3, temporal_kernel_size=(3, 5, 9), num_layers=3,
                 input_dropout=0.05, block_dropout=0.15, final_dropout=0.5,
                 use_attention=True):
        super().__init__()
        adj_matrices = get_stgcn_partition_adjacency(num_joints)
        adj_matrices = [normalize_adjacency_matrix(a) for a in adj_matrices]
        self.bn_input = nn.BatchNorm2d(in_channels)
        self.input_dropout = nn.Dropout2d(input_dropout) if input_dropout and input_dropout > 0 else nn.Identity()
        # 可學習的 per-sample per-frame joint attention 模組（模組層級定義）
        self.use_attention = bool(use_attention)
        self.joint_attention = JointAttention(in_channels) if self.use_attention else nn.Identity()

        # No per-class head or bias terms here — all classes treated equally.
        self.stgcn_layers = nn.ModuleList()
        self.stgcn_layers.append(
            STGCNBlock(
                in_channels, 64, adj_matrices,
                spatial_kernel_size, temporal_kernel_size, stride=1,
                dropout=block_dropout
            )
        )
        self.stgcn_layers.append(
            STGCNBlock(
                64, 128, adj_matrices,
                spatial_kernel_size, temporal_kernel_size, stride=2,
                dropout=block_dropout
            )
        )
        for _ in range(num_layers - 2):
            self.stgcn_layers.append(
                STGCNBlock(
                    128, 128, adj_matrices,
                    spatial_kernel_size, temporal_kernel_size, stride=1,
                    dropout=block_dropout
                )
            )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(final_dropout) if final_dropout and final_dropout > 0 else nn.Identity()
    def forward(self, x):
        bn_x = self.bn_input(x)
        # 由 JointAttention 決定每個樣本、每個關節的縮放係數並套用
        if self.use_attention:
            attn = self.joint_attention(bn_x)
            x = bn_x * attn
        else:
            x = bn_x

        x = self.input_dropout(x)

        # Shake-specific head removed; no per-class bias will be applied.
        for layer in self.stgcn_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)

        # No class-specific logit adjustments applied.
        return logits

# 包裝器
class CatBehaviorSTGCN:
    def __init__(self, model_path, device='cuda', sequence_length=32, num_classes=5, normalize=True, feature_mode='xy', in_channels=None, use_attention=None):
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
            except Exception as _e:
                logging.debug("weights_only load failed (%s), retrying without flag", _e)
                checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

        if self.in_channels is None and state_dict is not None and 'bn_input.weight' in state_dict:
            self.in_channels = int(state_dict['bn_input.weight'].shape[0])
        if self.in_channels is None:
            self.in_channels = get_in_channels_for_mode(self.feature_mode)

        # 從 checkpoint 自動偵測 num_classes，避免模型預設值與訓練 checkpoint 不符
        if state_dict is not None and 'fc.weight' in state_dict:
            ckpt_num_classes = int(state_dict['fc.weight'].shape[0])
            if ckpt_num_classes != num_classes:
                print(f"✓ 依 checkpoint 自動調整 num_classes: {num_classes} → {ckpt_num_classes}")
                num_classes = ckpt_num_classes
        self.num_classes = num_classes

        expected_channels = get_in_channels_for_mode(self.feature_mode)
        if self.in_channels != expected_channels:
            _ch_to_mode = {v: k for k, v in FEATURE_MODE_CHANNELS.items()}
            if self.in_channels in _ch_to_mode:
                auto_mode = _ch_to_mode[self.in_channels]
                print(f"⚠ feature_mode={self.feature_mode} 對應 {expected_channels} channels，"
                      f"但 checkpoint in_channels={self.in_channels}，自動調整為 {auto_mode}")
                self.feature_mode = auto_mode
            else:
                raise ValueError(
                    f"feature_mode={self.feature_mode} 對應 {expected_channels} channels，但目前 in_channels={self.in_channels}; "
                    "請確認訓練與推論設定一致"
                )

        if use_attention is None:
            if state_dict is not None:
                has_attention_weights = any(k.startswith('joint_attention.') for k in state_dict.keys())
                use_attention = bool(has_attention_weights)
                print(f"✓ 依 checkpoint 自動判定 attention: {'啟用' if use_attention else '停用'}")
            else:
                use_attention = True

        # 從 checkpoint 自動偵測 num_joints（adjacency buffer shape：K × V × V）
        _adj_key = 'stgcn_layers.0.sgc.A'
        if state_dict is not None and _adj_key in state_dict:
            self.num_joints = int(state_dict[_adj_key].shape[-1])
            if self.num_joints != 17:
                print(f"✓ 依 checkpoint 自動調整 num_joints: 17 → {self.num_joints}")
        else:
            self.num_joints = 17

        # 從 checkpoint 偵測 TCN 分支數，避免新預設 (3,5,9) 與舊單分支 checkpoint 架構不符
        _tcn_branch_key = 'stgcn_layers.0.tcn.branches.{}.conv.weight'
        if state_dict is not None:
            num_tcn_branches = sum(
                1 for i in range(10) if _tcn_branch_key.format(i) in state_dict
            )
            if num_tcn_branches == 0:
                num_tcn_branches = 1
            _default_kernels = (3, 5, 9)
            if num_tcn_branches != len(_default_kernels):
                _ckpt_kernels = tuple(9 for _ in range(num_tcn_branches))
                print(f"✓ 依 checkpoint 自動調整 TCN 分支數: {num_tcn_branches}（kernel={_ckpt_kernels}）")
                temporal_kernel_size_ckpt = _ckpt_kernels
            else:
                temporal_kernel_size_ckpt = _default_kernels
        else:
            temporal_kernel_size_ckpt = (3, 5, 9)

        self.model = STGCN(
            num_classes=num_classes,
            in_channels=self.in_channels,
            num_joints=self.num_joints,
            spatial_kernel_size=3,
            temporal_kernel_size=temporal_kernel_size_ckpt,
            num_layers=3,
            use_attention=use_attention,
        ).to(self.device)
        if state_dict is not None:
            load_result = self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ ST-GCN 模型已載入: {model_path}")
            print(f"  in_channels={self.in_channels}, feature_mode={self.feature_mode}")
            non_attn_missing = [k for k in load_result.missing_keys if 'joint_attention' not in k]
            if non_attn_missing:
                print(f"  ⚠ checkpoint 缺少非 attention 層的 key: {non_attn_missing}")
            if load_result.unexpected_keys:
                print(f"  ⚠ checkpoint 含有未預期的 key: {load_result.unexpected_keys}")
        else:
            print(f"⚠ 警告：模型檔案未找到 {model_path}")
        self.model.eval()
    def normalize_keypoints(self, keypoints_sequence):
        # flip 必須在 orientation 之前：原始座標下使用 mid_back(4) 與 hip(5) 的 x 差距做多數決，翻轉決策穩定
        seq = flip_normalize(keypoints_sequence)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        seq = add_velocity_feature(seq)
        return seq
    def predict(self, keypoints_sequence, precomputed=False):
        if keypoints_sequence.shape[0] < self.sequence_length:
            return None, 0.0, [0.0] * self.num_classes

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
                logging.debug("ST-GCN normalize disabled — using raw coordinates")

        seq_tensor = torch.FloatTensor(seq_features).permute(2, 0, 1).unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        behavior_id = int(np.argmax(probs))
        confidence = float(probs[behavior_id])
        return behavior_id, confidence, probs.tolist()
    def __call__(self, keypoints_sequence):
        return self.predict(keypoints_sequence)

