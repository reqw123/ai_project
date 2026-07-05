# ==============================================================================
# Cat Pose Analysis System
# ==============================================================================
# Preprocessing pipeline:
#   YOLO keypoints
#       ↓  kpts_to_relative()      — scale normalization (bbox)
#       ↓  rotation_normalize()    — rotation normalization (chest→hip → Y-axis)
#       ↓  flip_skeleton_if_needed() — left-right flip normalization
#       ↓  EMA temporal smoothing  — per-frame online smoothing
#       ↓  trajectory extraction   — per-keypoint X-Y trajectory
#       ↓  statistics analysis     — distribution / errorbar / line plots
#       ↓  ST-GCN dataset         — train_data.npy  (N, T, 17, 2)
#                                    train_label.npy (N,)
# ==============================================================================

from ultralytics import YOLO
import cv2
import numpy as np
import csv
import glob
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ==============================================================================
# 1. Keypoint Index (17 points)
# ==============================================================================
# 0  nose          1  left_ear_tip    2  right_ear_tip
# 3  chest         4  mid_back        5  hip
# 6  lf_elbow      7  lf_paw          8  rf_elbow        9  rf_paw
# 10 lh_knee       11 lh_paw          12 rh_knee         13 rh_paw
# 14 tail_base     15 tail_mid        16 tail_tip

NUM_KPTS = 17

KEYPOINT_NAMES = [
    "nose", "left_ear_tip", "right_ear_tip",
    "chest", "mid_back", "hip",
    "lf_elbow", "lf_paw", "rf_elbow", "rf_paw",
    "lh_knee",  "lh_paw",  "rh_knee",  "rh_paw",
    "tail_base", "tail_mid", "tail_tip",
]

# Rotation reference: chest(3) → hip(5)
ROT_REF_A = 3   # chest (anchor)
ROT_REF_B = 5   # hip   (direction)

# Symmetric keypoint pairs for left-right flip
FLIP_PAIRS = [(1, 2), (6, 8), (7, 9), (10, 12), (11, 13)]

# ==============================================================================
# 2. Configuration
# ==============================================================================
MODEL_PATH     = r"C:\cat_pose\v11s_10.pt"
TEST_DIR       = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\ai貓圖"
OUTPUT_DIR     = r"C:\cat_pose\output"
STATISTICS_DIR = os.path.join(OUTPUT_DIR, "statistics")
STGCN_DIR      = os.path.join(OUTPUT_DIR, "stgcn_dataset")

CONF_THR      = 0.3    # keypoint confidence threshold
DETECT_CONF   = 0.4    # YOLO detection confidence
LOW_KPT_RATIO = 0.4    # low-quality frame threshold (ratio of low-conf kpts)
EMA_ALPHA     = 0.3    # EMA smoothing factor  (0 < α ≤ 1)
SEQUENCE_LEN  = 30     # frames per ST-GCN sample (T)

# ==============================================================================
# 3. Colors (BGR) and Skeleton Links
# ==============================================================================
BLUE       = (255,   0,   0)
COLOR_HEAD = (255, 255,   0)
COLOR_BODY = (  0, 255,   0)
COLOR_LIMB = (  0, 150, 255)
COLOR_TAIL = (255,   0, 255)
COLOR_KPT  = (  0,   0, 255)   # normal keypoint
COLOR_ABN  = (  0, 255, 255)   # uncertain keypoint (0.3 < conf ≤ 0.5)

HEAD_LINKS  = [(0, 1), (0, 2), (1, 2)]
BODY_LINKS  = [(0, 3), (3, 4), (4, 5)]
FRONT_LIMBS = [(3, 6), (6, 7), (3, 8), (8, 9)]
HIND_LIMBS  = [(5, 10), (10, 11), (5, 12), (12, 13)]
TAIL_LINKS  = [(5, 14), (14, 15), (15, 16)]

ALL_GROUPS = [
    (HEAD_LINKS,  COLOR_HEAD),
    (BODY_LINKS,  COLOR_BODY),
    (FRONT_LIMBS, COLOR_LIMB),
    (HIND_LIMBS,  COLOR_LIMB),
    (TAIL_LINKS,  COLOR_TAIL),
]

# Bone links (parent → child) used by ST-GCN feature extensions
BONE_LINKS = (
    HEAD_LINKS + BODY_LINKS + FRONT_LIMBS + HIND_LIMBS + TAIL_LINKS
)

# ==============================================================================
# 4. Normalization Functions
# ==============================================================================

def kpts_to_relative(kpts_xy, bbox):
    """
    Scale normalization: convert pixel keypoints to bbox-centered,
    bbox-size-normalized coordinates.

    Args:
        kpts_xy : np.ndarray (17, 2)  — pixel coordinates
        bbox    : array-like [x1, y1, x2, y2]

    Returns:
        kpts_rel: np.ndarray (17, 2)  — values ≈ [-0.5, 0.5]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = max(x2 - x1, 1e-6)
    h  = max(y2 - y1, 1e-6)
    kpts_rel = np.array([[(x - cx) / w, (y - cy) / h] for x, y in kpts_xy])
    return kpts_rel


def rotation_normalize(kpts_rel, ref_a=ROT_REF_A, ref_b=ROT_REF_B):
    """
    Rotation normalization: rotate the skeleton so that the
    chest(3) → hip(5) axis aligns with the Y-axis. Centers at chest.

    Args:
        kpts_rel: np.ndarray (17, 2)
        ref_a   : anchor joint (chest = 3)
        ref_b   : direction joint (hip = 5)

    Returns:
        kpts_rot: np.ndarray (17, 2)
    """
    v = kpts_rel[ref_b] - kpts_rel[ref_a]
    angle = np.arctan2(v[0], v[1])          # angle needed to align to Y-axis
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
    kpts_centered = kpts_rel - kpts_rel[ref_a]
    return kpts_centered @ rot.T


def flip_skeleton_if_needed(kpts_rot):
    """
    Left-right flip normalization: make the cat always face right.

    Decision rule:
        nose_x < tail_base_x  →  cat faces left  →  flip

    Flip method:
        x = -x
        swap symmetric pairs: 1↔2, 6↔8, 7↔9, 10↔12, 11↔13

    Args:
        kpts_rot : np.ndarray (17, 2)

    Returns:
        kpts_flip: np.ndarray (17, 2)
        flipped  : bool
    """
    nose_x      = kpts_rot[0,  0]
    tail_base_x = kpts_rot[14, 0]

    if nose_x < tail_base_x:
        kpts_flip = kpts_rot.copy()
        kpts_flip[:, 0] = -kpts_flip[:, 0]
        for i, j in FLIP_PAIRS:
            kpts_flip[i], kpts_flip[j] = kpts_flip[j].copy(), kpts_flip[i].copy()
        return kpts_flip, True

    return kpts_rot.copy(), False


# ==============================================================================
# 5. Temporal Smoothing (EMA)
# ==============================================================================

def ema_smooth(seq, alpha=EMA_ALPHA):
    """
    Exponential Moving Average smoothing for a 1-D list.

    Formula: smoothed[t] = α * seq[t] + (1-α) * smoothed[t-1]

    Args:
        seq  : list[float]
        alpha: smoothing factor (higher = less smoothing)

    Returns:
        list[float]
    """
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return out


def ema_smooth_2d(seq_2d, alpha=EMA_ALPHA):
    """EMA smoothing for a list of (x, y) tuples."""
    if not seq_2d:
        return []
    out = [seq_2d[0]]
    for pt in seq_2d[1:]:
        px = alpha * pt[0] + (1 - alpha) * out[-1][0]
        py = alpha * pt[1] + (1 - alpha) * out[-1][1]
        out.append((px, py))
    return out


# ==============================================================================
# 6. Quality Check
# ==============================================================================

def count_low_conf_kpts(kpts_conf, thr=CONF_THR):
    """Return the number of keypoints with confidence ≤ thr."""
    return int(np.sum(kpts_conf <= thr))


# ==============================================================================
# 7. Visualization
# ==============================================================================

def draw_fancy_pose(frame, box, kpts):
    """
    Overlay bounding box, skeleton links, and keypoints on the frame.

    Colour coding:
        Blue   — bounding box
        Cyan   — head links
        Green  — body links
        Orange — limb links
        Purple — tail links
        Red    — keypoint confidence > 0.5
        Yellow — keypoint confidence 0.3–0.5  (uncertain, highlighted)
    """
    kpts_xy   = kpts.xy[0].cpu().numpy().astype(int)
    kpts_conf = kpts.conf[0].cpu().numpy()

    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), BLUE, 2)
    cv2.putText(frame, f"Cat {float(box.conf[0]):.2f}", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)

    for links, color in ALL_GROUPS:
        for a, b in links:
            if a < len(kpts_conf) and b < len(kpts_conf):
                if kpts_conf[a] > CONF_THR and kpts_conf[b] > CONF_THR:
                    cv2.line(frame, tuple(kpts_xy[a]), tuple(kpts_xy[b]), color, 2)

    for i, (x, y) in enumerate(kpts_xy):
        if kpts_conf[i] > CONF_THR:
            node_color = COLOR_KPT if kpts_conf[i] > 0.5 else COLOR_ABN
            cv2.circle(frame, (x, y), 5, node_color, -1)
            cv2.putText(frame, str(i), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, node_color, 1)


# ==============================================================================
# 8. Trajectory Visualization
# ==============================================================================

def plot_kpt_xy_trajectory(traj_dict, output_dir):
    """
    Save one X-Y scatter+line plot per keypoint.

    Args:
        traj_dict : dict {kpt_idx: list of (x, y)}  — after full normalization
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, pts in traj_dict.items():
        if not pts:
            continue
        xs   = [p[0] for p in pts]
        ys   = [p[1] for p in pts]
        name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kpt{idx}"

        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=8, alpha=0.6, label=f"kpt{idx} {name}")
        plt.plot(xs, ys, linewidth=0.8, alpha=0.5)
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.title(f"Keypoint {idx} [{name}] X-Y Trajectory\n"
                  "(scale + rotation + flip normalized, EMA smoothed)")
        plt.xlabel("X (normalized)")
        plt.ylabel("Y (normalized)")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"kpt_{idx:02d}_xy_traj.png"))
        plt.close()


# ==============================================================================
# 9. Statistics Collector
# ==============================================================================

class KptStatsCollector:
    """
    Accumulate normalized keypoint positions frame-by-frame and
    produce distribution / error-bar / line plots.
    """

    def __init__(self, num_kpts=NUM_KPTS):
        self.num_kpts = num_kpts
        self.xs = [[] for _ in range(num_kpts)]
        self.ys = [[] for _ in range(num_kpts)]
        self.ws = [[] for _ in range(num_kpts)]   # confidence weights

    def add(self, kpts_norm, kpts_conf, conf_thr=CONF_THR):
        """Add one frame's fully-normalized keypoints."""
        for i, (xy, conf) in enumerate(zip(kpts_norm, kpts_conf)):
            if conf > conf_thr:
                self.xs[i].append(float(xy[0]))
                self.ys[i].append(float(xy[1]))
                self.ws[i].append(float(conf))

    def apply_ema_smoothing(self, alpha=EMA_ALPHA):
        """Apply EMA smoothing to accumulated position sequences."""
        for i in range(self.num_kpts):
            self.xs[i] = ema_smooth(self.xs[i], alpha)
            self.ys[i] = ema_smooth(self.ys[i], alpha)

    def get_stats(self):
        """Return confidence-weighted mean and std for each keypoint."""
        stats = []
        for i in range(self.num_kpts):
            x = np.array(self.xs[i])
            y = np.array(self.ys[i])
            w = np.array(self.ws[i])
            if len(x) > 0:
                mx = np.average(x, weights=w)
                my = np.average(y, weights=w)
                sx = np.sqrt(np.average((x - mx) ** 2, weights=w))
                sy = np.sqrt(np.average((y - my) ** 2, weights=w))
                stats.append({"mean_x": mx, "std_x": sx,
                               "mean_y": my, "std_y": sy, "count": len(x)})
            else:
                stats.append({"mean_x": None, "std_x": None,
                               "mean_y": None, "std_y": None, "count": 0})
        return stats

    def plot_distribution(self, output_dir):
        """Scatter plot: position distribution of all keypoints."""
        plt.figure(figsize=(12, 8))
        for i in range(self.num_kpts):
            if self.xs[i]:
                name = KEYPOINT_NAMES[i]
                plt.scatter(self.xs[i], self.ys[i], s=10, label=f"{i}:{name}")
        plt.gca().invert_yaxis()
        plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        plt.title("Keypoint Position Distribution\n"
                  "(scale + rotation + flip normalized, EMA smoothed)")
        plt.xlabel("X (normalized)")
        plt.ylabel("Y (normalized)")
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1),
                   loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "keypoint_distribution.png"))
        plt.close()

    def plot_errorbar(self, output_dir):
        """Error-bar plot: mean ± std per keypoint."""
        stats  = self.get_stats()
        idx    = np.arange(self.num_kpts)
        mean_x = [s["mean_x"] or 0 for s in stats]
        std_x  = [s["std_x"]  or 0 for s in stats]
        mean_y = [s["mean_y"] or 0 for s in stats]
        std_y  = [s["std_y"]  or 0 for s in stats]

        plt.figure(figsize=(14, 5))
        plt.errorbar(idx, mean_x, yerr=std_x, fmt="o-", capsize=4, label="X mean±std")
        plt.errorbar(idx, mean_y, yerr=std_y, fmt="s-", capsize=4, label="Y mean±std")
        plt.xticks(idx,
                   [f"{i}\n{KEYPOINT_NAMES[i]}" for i in idx],
                   fontsize=7, rotation=45)
        plt.title("Keypoint Mean and Std (normalized)")
        plt.xlabel("Keypoint")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "keypoint_errorbar.png"))
        plt.close()

    def plot_line(self, output_dir):
        """Line plot: X and Y positions over frame index per keypoint."""
        for i in range(self.num_kpts):
            if not self.xs[i]:
                continue
            name = KEYPOINT_NAMES[i]
            plt.figure(figsize=(10, 4))
            plt.plot(self.xs[i], label="X (normalized)", color="steelblue")
            plt.plot(self.ys[i], label="Y (normalized)", color="tomato")
            plt.title(f"Keypoint {i} [{name}] Position over Frames (EMA smoothed)")
            plt.xlabel("Frame Index")
            plt.ylabel("Normalized Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"keypoint_{i:02d}_line.png"))
            plt.close()


# ==============================================================================
# 10. YOLO Ground-Truth Label Loader  (optional, for evaluation)
# ==============================================================================

def load_yolo_v8_labels(label_dir, img_dir, img_exts=(".jpg", ".png")):
    """
    Load YOLO v8 keypoint annotation labels.

    Returns:
        dict {img_name: [np.ndarray (17, 3)]}  — [x_px, y_px, visibility]
    """
    img_size_map = {}
    for ext in img_exts:
        for img_path in glob.glob(os.path.join(img_dir, f"*{ext}")):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                img_size_map[img_name] = (img.shape[1], img.shape[0])

    gt_dict = {}
    for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
        base     = os.path.splitext(os.path.basename(label_path))[0]
        img_name = None
        for ext in img_exts:
            if base + ext in img_size_map:
                img_name = base + ext
                break
        if img_name is None:
            continue

        w, h = img_size_map[img_name]
        with open(label_path) as f:
            lines = f.readlines()

        gt_kpts_list = []
        for line in lines:
            arr = line.strip().split()
            if len(arr) < 3 + NUM_KPTS * 3:
                continue
            kpts = []
            for k in range(NUM_KPTS):
                kx = float(arr[3 + k * 3])     * w
                ky = float(arr[3 + k * 3 + 1]) * h
                kv = int(  arr[3 + k * 3 + 2])
                kpts.append([kx, ky, kv])
            gt_kpts_list.append(np.array(kpts))
        gt_dict[img_name] = gt_kpts_list

    return gt_dict


# ==============================================================================
# 11. ST-GCN Dataset Builder
# ==============================================================================

class STGCNDatasetBuilder:
    """
    Build an ST-GCN compatible skeleton sequence dataset.

    Output format
    -------------
    train_data.npy  : float32  (N, T, 17, 2)
    train_label.npy : int64    (N,)

        N — number of samples
        T — SEQUENCE_LEN (frames per sample)
        V — 17 keypoints
        C — 2 (x, y)

    Future feature extensions (static methods provided)
    -------------------
    compute_velocity() — frame-to-frame displacement  (N, T, 17, 2)
    compute_bone()     — parent→child bone vectors     (N, T, B, 2)
    compute_angle()    — joint angle between bones     (N, T, 17)
    """

    def __init__(self, sequence_len=SEQUENCE_LEN):
        self.sequence_len = sequence_len
        self._frame_buf   = []    # list of (17, 2) — current sequence buffer
        self._samples     = []    # list of np.ndarray (T, 17, 2)
        self._label_list  = []    # list of int

    def add_frame(self, kpts_norm):
        """
        Buffer one normalized frame.

        Args:
            kpts_norm: np.ndarray (17, 2) — fully normalized keypoints
        """
        self._frame_buf.append(kpts_norm.copy())

    def flush_sequence(self, label=0):
        """
        Convert the buffered frames into fixed-length samples (sliding window,
        non-overlapping, step = T). Remaining frames are zero-padded.

        Args:
            label: action class label (int)
        """
        if not self._frame_buf:
            return

        frames = np.stack(self._frame_buf, axis=0).astype(np.float32)  # (F, 17, 2)
        F, T   = len(frames), self.sequence_len

        if F >= T:
            # Non-overlapping segments of length T
            for start in range(0, F - T + 1, T):
                self._samples.append(frames[start: start + T])
                self._label_list.append(label)
        else:
            # Zero-pad short sequence
            pad = np.zeros((T - F, NUM_KPTS, 2), dtype=np.float32)
            self._samples.append(np.concatenate([frames, pad], axis=0))
            self._label_list.append(label)

        self._frame_buf = []   # reset buffer

    def save(self, output_dir, split="train"):
        """
        Save dataset files.

        Files created:
            {output_dir}/{split}_data.npy   shape (N, T, 17, 2)
            {output_dir}/{split}_label.npy  shape (N,)
        """
        os.makedirs(output_dir, exist_ok=True)
        if not self._samples:
            print("⚠️  No ST-GCN samples to save.")
            return

        data  = np.stack(self._samples, axis=0).astype(np.float32)
        label = np.array(self._label_list, dtype=np.int64)

        np.save(os.path.join(output_dir, f"{split}_data.npy"),  data)
        np.save(os.path.join(output_dir, f"{split}_label.npy"), label)
        print(f"💾 ST-GCN dataset saved to {output_dir}")
        print(f"   {split}_data.npy  : {data.shape}")
        print(f"   {split}_label.npy : {label.shape}")

    # ------------------------------------------------------------------
    # Feature extension helpers (ready for future use)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_velocity(data):
        """
        Velocity feature: frame-to-frame displacement.

        Args:
            data: np.ndarray (N, T, 17, 2)

        Returns:
            velocity: np.ndarray (N, T, 17, 2)  — first frame = zeros
        """
        vel = np.zeros_like(data)
        vel[:, 1:] = data[:, 1:] - data[:, :-1]
        return vel

    @staticmethod
    def compute_bone(data, bone_links=BONE_LINKS):
        """
        Bone feature: displacement vector from parent to child joint.

        Args:
            data      : np.ndarray (N, T, 17, 2)
            bone_links: list of (parent_idx, child_idx)

        Returns:
            bone: np.ndarray (N, T, B, 2)
        """
        bones = [data[:, :, b] - data[:, :, a] for a, b in bone_links]
        return np.stack(bones, axis=2)

    @staticmethod
    def compute_angle(data, bone_links=BONE_LINKS):
        """
        Joint angle feature: angle between incoming and outgoing bone vectors.

        Args:
            data      : np.ndarray (N, T, 17, 2)
            bone_links: list of (parent_idx, child_idx)

        Returns:
            angles: np.ndarray (N, T, 17) — radians, 0 if no adjacent bone
        """
        N, T, V, _ = data.shape
        angles     = np.zeros((N, T, V), dtype=np.float32)
        parent_map = {b: a for a, b in bone_links}

        for v in range(V):
            if v not in parent_map:
                continue
            p      = parent_map[v]
            vec_in = data[:, :, v] - data[:, :, p]           # (N, T, 2)
            for a, b in bone_links:
                if a != v:
                    continue
                vec_out  = data[:, :, b] - data[:, :, v]     # (N, T, 2)
                dot      = np.sum(vec_in * vec_out, axis=-1)
                norm_in  = np.linalg.norm(vec_in,  axis=-1) + 1e-8
                norm_out = np.linalg.norm(vec_out, axis=-1) + 1e-8
                angles[:, :, v] = np.arccos(
                    np.clip(dot / (norm_in * norm_out), -1.0, 1.0))

        return angles


# ==============================================================================
# 12. Main Pipeline
# ==============================================================================

if __name__ == "__main__":

    # ── Setup output directories ─────────────────────────────────────────────
    for d in [OUTPUT_DIR, STATISTICS_DIR, STGCN_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print("🔄 Loading model...")
    model = YOLO(MODEL_PATH)

    img_list = sorted(
        p for p in Path(TEST_DIR).glob("*.*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not img_list:
        print(f"❌ No images found in {TEST_DIR}")
        raise SystemExit(1)

    print(f"✅ Found {len(img_list)} images. Starting inference...\n")

    # ── Collectors ───────────────────────────────────────────────────────────
    kpt_stats     = KptStatsCollector(num_kpts=NUM_KPTS)
    traj_pos      = {i: [] for i in range(NUM_KPTS)}
    stgcn_builder = STGCNDatasetBuilder(sequence_len=SEQUENCE_LEN)
    low_quality_records = []

    # Online EMA state: (17, 2) running average across frames
    ema_state: np.ndarray | None = None

    # ── Per-image inference loop ─────────────────────────────────────────────
    for img_path in img_list:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = model.predict(frame, conf=DETECT_CONF, verbose=False)

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            for det_i in range(len(r.boxes)):
                kpts_conf = r.keypoints[det_i].conf[0].cpu().numpy()  # (17,)
                kpts_xy   = r.keypoints[det_i].xy[0].cpu().numpy()    # (17, 2)
                bbox      = r.boxes[det_i].xyxy[0].cpu().numpy()      # [x1,y1,x2,y2]

                # ── Step 1: Scale normalization ──────────────────────────────
                kpts_rel  = kpts_to_relative(kpts_xy, bbox)            # (17, 2)

                # ── Step 2: Rotation normalization ───────────────────────────
                kpts_rot  = rotation_normalize(kpts_rel)               # (17, 2)

                # ── Step 3: Left-right flip normalization ────────────────────
                kpts_norm, flipped = flip_skeleton_if_needed(kpts_rot) # (17, 2)

                # ── Step 4: Online EMA temporal smoothing ────────────────────
                if ema_state is None:
                    ema_state = kpts_norm.copy()
                else:
                    ema_state = EMA_ALPHA * kpts_norm + (1 - EMA_ALPHA) * ema_state
                kpts_final = ema_state.copy()                          # (17, 2)

                # ── Step 5: Quality check ────────────────────────────────────
                low_count = count_low_conf_kpts(kpts_conf)
                low_ratio = low_count / NUM_KPTS
                if low_ratio > LOW_KPT_RATIO:
                    low_quality_records.append({
                        "filename"     : img_path.name,
                        "low_kpt_count": low_count,
                        "total_kpts"   : NUM_KPTS,
                        "low_ratio"    : f"{low_ratio:.2f}",
                        "flipped"      : flipped,
                    })

                # ── Step 6: Statistics & trajectory collection ────────────────
                kpt_stats.add(kpts_final, kpts_conf)
                for k, (px, py) in enumerate(kpts_final):
                    traj_pos[k].append((float(px), float(py)))

                # ── Step 7: ST-GCN frame buffer ───────────────────────────────
                stgcn_builder.add_frame(kpts_final)

                # ── Draw (original pixel coords) ──────────────────────────────
                draw_fancy_pose(frame, r.boxes[det_i], r.keypoints[det_i])

        cv2.imwrite(str(Path(OUTPUT_DIR) / img_path.name), frame)
        print(f"  Done: {img_path.name}")

    # ── Post-processing: EMA smoothing on statistics ─────────────────────────
    kpt_stats.apply_ema_smoothing()

    # ── Statistics plots ──────────────────────────────────────────────────────
    print("\n📊 Generating statistics plots...")
    kpt_stats.plot_distribution(STATISTICS_DIR)
    kpt_stats.plot_errorbar(STATISTICS_DIR)
    kpt_stats.plot_line(STATISTICS_DIR)

    # ── Trajectory plots (EMA smoothed) ──────────────────────────────────────
    print("📈 Generating trajectory plots...")
    for k in range(NUM_KPTS):
        traj_pos[k] = ema_smooth_2d(traj_pos[k])
    plot_kpt_xy_trajectory(traj_pos, STATISTICS_DIR)

    # ── Low quality CSV ───────────────────────────────────────────────────────
    if low_quality_records:
        csv_path = Path(OUTPUT_DIR) / "low_quality_images.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["filename", "low_kpt_count", "total_kpts",
                          "low_ratio", "flipped"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(low_quality_records)
        print(f"\n⚠️  Low quality frames: {csv_path}  "
              f"({len(low_quality_records)} records)")
    else:
        print("\n✅ No low quality frames detected.")

    # ── ST-GCN dataset ────────────────────────────────────────────────────────
    print("\n💾 Generating ST-GCN dataset...")
    stgcn_builder.flush_sequence(label=0)   # label=0: single action class
    stgcn_builder.save(STGCN_DIR, split="train")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n🎉 Pipeline complete!")
    print(f"   Inference images : {OUTPUT_DIR}")
    print(f"   Statistics plots : {STATISTICS_DIR}")
    print(f"   ST-GCN dataset   : {STGCN_DIR}")