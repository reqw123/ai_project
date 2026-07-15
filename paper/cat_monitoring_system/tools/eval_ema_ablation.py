"""
Compare ST-GCN models trained with different KP EMA alpha values on labeled behavior videos.

Each model was trained with a different keypoint temporal smoothing coefficient (alpha).
Inference here applies the same alpha as training so evaluation conditions are consistent.

Output: <HARD_OUTPUT_DIR>/ema_ablation_NNN/
  ema_ablation_accuracy.png    — grouped bar chart (per-class + overall)
  ema_ablation_confusion.png   — confusion matrices (one per model)
  ema_ablation_prob_hist.png   — true-class probability histograms
  ema_ablation_summary.csv     — all metrics for all alphas
"""
import re
import csv
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys

# 這支腳本現在放在 cat_monitoring_system/tools/ 底下，parent.parent 才是
# cat_monitoring_system/（detectors/models/utils 等套件所在目錄）。
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
)
from utils.constants import BEHAVIOR_CLASSES
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── Hardcoded model definitions ───────────────────────────────────────────────
# ↓ 重新訓練後只需改此一行 run 號碼，下方路徑全部自動更新
_RUN_TAG = "075"
_BASE    = rf"C:\Users\homec\Downloads\stgcn_results\run_{_RUN_TAG}_ema_ablation_att_on"
MODELS = [
    {'path': rf"{_BASE}\{_RUN_TAG}_xy_conf_v_bone_att_on.pth",          'alpha': 1.0, 'name': 'α=1.00 (no EMA)'},
    {'path': rf"{_BASE}\{_RUN_TAG}_xy_conf_v_bone_ema0.90_att_on.pth",  'alpha': 0.9, 'name': 'α=0.90'},
    {'path': rf"{_BASE}\{_RUN_TAG}_xy_conf_v_bone_ema0.70_att_on.pth",  'alpha': 0.7, 'name': 'α=0.70'},
    {'path': rf"{_BASE}\{_RUN_TAG}_xy_conf_v_bone_ema0.50_att_on.pth",  'alpha': 0.5, 'name': 'α=0.50'},
    {'path': rf"{_BASE}\{_RUN_TAG}_xy_conf_v_bone_ema0.30_att_on.pth",  'alpha': 0.3, 'name': 'α=0.30'},
]

# ── Inference settings ────────────────────────────────────────────────────────
HARD_VIDEO_WALK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk"
HARD_VIDEO_LICK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick"
HARD_VIDEO_SCRATCH_DIR = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch"
HARD_VIDEO_SHAKE_DIR   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake"
HARD_VIDEO_STOP_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\stop"
HARD_OUTPUT_DIR        = r"C:\paper\cat_monitoring_system\eval_results"

DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\aug_8.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 2
DEFAULT_DEVICE  = 'cuda'

EVENT_MIN_WINDOWS    = 3
EVENT_MIN_RATIO      = 0.30
PROB_EVENT_THRESHOLD = 0.40

# ── Colour palette (one per model) ────────────────────────────────────────────
_PALETTE = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0', '#00BCD4']

CH_TO_FEATURE = {2: 'xy', 3: 'xy_conf', 5: 'xy_conf_v',
                 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}

_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg', '.webm'}


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def _next_run_number(out_root: str) -> int:
    p = Path(out_root)
    if not p.exists():
        return 1
    pat = re.compile(r'^ema_ablation_(\d+)')
    max_num = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def infer_bn_input_channels(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    ck = torch.load(model_path, map_location='cpu')
    sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    if isinstance(sd, dict):
        for k, v in sd.items():
            if k.endswith('bn_input.weight'):
                return int(v.shape[0])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════
def _apply_ema(preds, alpha):
    if alpha >= 1.0 or not preds:
        return preds
    smoothed = []
    ema = None
    for p in preds:
        raw = np.array(p['probs'], dtype=np.float32)
        ema = raw if ema is None else alpha * raw + (1.0 - alpha) * ema
        new_pred = int(np.argmax(ema))
        smoothed.append({**p, 'probs': ema.tolist(), 'pred': new_pred, 'conf': float(ema[new_pred])})
    return smoothed


def evaluate_video(video_path, kp_detector, classifier, feature_mode,
                   sequence_length=16, classify_stride=2, ema_alpha=1.0):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kp_detector.reset_track()  # 新影片開始，避免延續上一支影片鎖定的貓
    buf = deque(maxlen=sequence_length)
    preds = []
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        kpts, kpt_conf, _, _ = kp_detector.detect(frame)
        buf.append((kpts, kpt_conf) if kpts is not None else (None, None))
        if len(buf) < sequence_length or frame_idx % classify_stride != 0:
            continue
        kpts_arr = np.array([item[0] if item[0] is not None else np.zeros((17, 2), np.float32) for item in buf])
        conf_arr = np.array([item[1] if item[1] is not None else np.zeros((17,), np.float32) for item in buf])
        _mj = getattr(classifier.model, 'num_joints', 17)
        if _mj < 17:
            kpts_arr = kpts_arr[:, :_mj, :]
            conf_arr = conf_arr[:, :_mj]
        seq = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        feats = build_feature_tensor(seq, conf_arr, feature_mode)
        pred_id, pred_conf, pred_probs = classifier.model.predict(feats, precomputed=True)
        if pred_id is None:
            pred_id, pred_conf, pred_probs = -1, 0.0, [0.0] * len(BEHAVIOR_CLASSES)
        preds.append({'frame': frame_idx, 'time': round(frame_idx / fps, 3),
                      'pred': int(pred_id), 'conf': float(pred_conf),
                      'probs': [float(x) for x in pred_probs]})
    cap.release()
    return _apply_ema(preds, ema_alpha)


def evaluate_folder(folder_path, kp_detector, classifier, feature_mode,
                    sequence_length=16, classify_stride=2, ema_alpha=1.0):
    folder = Path(folder_path)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        print(f"    ⚠ No videos in: {folder}")
        return []
    results = []
    for vid in videos:
        print(f"    {vid.name} ...", end=' ', flush=True)
        preds = evaluate_video(vid, kp_detector, classifier, feature_mode,
                               sequence_length, classify_stride, ema_alpha)
        print(f"{len(preds)} windows")
        results.append((vid.name, preds))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_by_class: dict) -> dict:
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class = {i: {'accuracy': 0.0, 'top2_accuracy': 0.0,
                     'avg_true_prob': 0.0, 'max_true_prob': 0.0,
                     'event_rate': 0.0, 'prob_event_rate': 0.0,
                     'n_correct': 0, 'n_windows': 0,
                     'n_videos': 0, 'n_videos_detected': 0, 'n_videos_prob_detected': 0}
                 for i in range(n_cls)}

    for cls_idx, vid_preds_list in preds_by_class.items():
        preds = [p for vid_preds in vid_preds_list for p in vid_preds]
        if not preds:
            continue
        probs      = np.array([p['probs'] for p in preds])
        actual_cls = probs.shape[1]
        pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int), 0, actual_cls - 1)
        probs_cls  = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(len(preds))
        n_correct  = int((pred_ids == cls_idx).sum())
        top2_ids   = np.argsort(probs, axis=1)[:, -2:]
        n_top2     = int(np.any(top2_ids == cls_idx, axis=1).sum())

        n_videos, n_vid_det, n_vid_prob_det = len(vid_preds_list), 0, 0
        for vid_preds in vid_preds_list:
            if not vid_preds:
                continue
            vp = np.clip(np.array([p['pred'] for p in vid_preds], dtype=int), 0, actual_cls - 1)
            vc, vn = int((vp == cls_idx).sum()), len(vid_preds)
            if (vc / vn) >= EVENT_MIN_RATIO or vc >= EVENT_MIN_WINDOWS:
                n_vid_det += 1
            vpc = np.array([p['probs'][cls_idx] for p in vid_preds if cls_idx < len(p['probs'])])
            if len(vpc) > 0 and int((vpc >= PROB_EVENT_THRESHOLD).sum()) >= EVENT_MIN_WINDOWS:
                n_vid_prob_det += 1

        per_class[cls_idx] = {
            'accuracy':               float(n_correct / len(preds)),
            'top2_accuracy':          float(n_top2 / len(preds)),
            'avg_true_prob':          float(probs_cls.mean()),
            'max_true_prob':          float(probs_cls.max()),
            'event_rate':             float(n_vid_det / n_videos) if n_videos else 0.0,
            'prob_event_rate':        float(n_vid_prob_det / n_videos) if n_videos else 0.0,
            'n_correct':              n_correct,
            'n_windows':              len(preds),
            'n_videos':               n_videos,
            'n_videos_detected':      n_vid_det,
            'n_videos_prob_detected': n_vid_prob_det,
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    evaluated = [i for i in range(n_cls) if per_class[i]['n_windows'] > 0]
    top2_acc  = float(np.mean([per_class[i]['top2_accuracy'] for i in evaluated])) if evaluated else 0.0
    ev_rate   = float(np.mean([per_class[i]['event_rate'] for i in evaluated])) if evaluated else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'accuracy':             float(accuracy_score(y_true, y_pred)),
            'top2_accuracy':        top2_acc,
            'macro_f1':             float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'event_detection_rate': ev_rate,
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(all_metrics, names, colors, classes, out_path):
    """
    2 行 grouped bar chart：
      Row 0 — Discrete Accuracy  (argmax == true label)
      Row 1 — Avg True-Class Probability
    每個類別 + Overall，每個 alpha 一個 bar。
    """
    n_models = len(all_metrics)
    n_cls    = len(classes)
    x_labels = classes + ['Overall']
    x        = np.arange(len(x_labels))
    total_w  = 0.72
    bar_w    = total_w / n_models
    offsets  = np.linspace(-total_w / 2 + bar_w / 2, total_w / 2 - bar_w / 2, n_models)

    def _vals(metrics, key):
        pc  = metrics['per_class']
        out = [pc[i].get(key, 0.0) for i in range(n_cls)]
        if key == 'accuracy':
            out.append(metrics['overall']['accuracy'])
        else:
            out.append(float(np.mean([pc[i].get(key, 0.0) for i in range(n_cls)])))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle('KP EMA Alpha Ablation — Inference Evaluation', fontsize=13, fontweight='bold')

    for ax, metric_key, title, ylabel in [
        (axes[0], 'accuracy',      'Discrete Accuracy  (argmax == true label)', 'Accuracy'),
        (axes[1], 'avg_true_prob', 'Avg True-Class Probability  (model conviction)', 'Avg Prob'),
    ]:
        for mi, (metrics, name, col) in enumerate(zip(all_metrics, names, colors)):
            vals = _vals(metrics, metric_key)
            bars = ax.bar(x + offsets[mi], vals, bar_w, label=name, color=col, alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f'{h:.1%}', ha='center', va='bottom', fontsize=7)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {Path(out_path).name}")


def plot_confusion_matrices(all_metrics, names, colors, classes, out_path):
    n = len(all_metrics)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), constrained_layout=True)
    fig.suptitle('Confusion Matrices — KP EMA Alpha Ablation', fontsize=13, fontweight='bold')
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    try:
        import seaborn as sns
        _sns = True
    except ImportError:
        _sns = False

    for ax, metrics, name, col in zip(axes_flat, all_metrics, names, colors):
        cm = metrics['confusion_matrix']
        if _sns:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, cbar=False, linewidths=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=30, ha='right')
            ax.set_yticklabels(classes)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)
        avg_recall = float(np.mean([cm[i, i] / max(cm[i].sum(), 1) for i in range(len(classes))]))
        ax.set_title(f'{name}\nAvg Recall={avg_recall:.1%}', fontsize=10, fontweight='bold', color=col)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {Path(out_path).name}")


def plot_prob_histograms(all_flat_preds, names, colors, classes, out_path):
    n_cls    = len(classes)
    n_models = len(all_flat_preds)
    fig, axes = plt.subplots(n_cls, n_models,
                              figsize=(3.5 * n_models, 2.8 * n_cls), constrained_layout=True)
    fig.suptitle('True-Class Probability Histogram — KP EMA Alpha Ablation',
                 fontsize=12, fontweight='bold')
    if n_cls == 1:
        axes = [axes]
    if n_models == 1:
        axes = [[ax] for ax in axes]

    bins = np.linspace(0, 1, 21)
    for i, cls in enumerate(classes):
        for j, (flat, name, col) in enumerate(zip(all_flat_preds, names, colors)):
            ax = axes[i][j]
            if i not in flat or not flat[i]:
                ax.set_visible(False)
                continue
            probs_cls = np.array([p['probs'][i] for p in flat[i] if i < len(p['probs'])])
            ax.hist(probs_cls, bins=bins, color=col, alpha=0.85, edgecolor='white')
            ax.axvline(float(probs_cls.mean()), color='k', linestyle='--',
                       linewidth=1.2, label=f'mean={probs_cls.mean():.2f}')
            ax.set_xlim(0, 1)
            ax.set_title(f'[{cls}] {name}', fontsize=8, fontweight='bold')
            ax.set_xlabel('True-class prob', fontsize=7)
            ax.set_ylabel('Windows', fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {Path(out_path).name}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════════
def save_summary_csv(all_metrics, names, classes, out_path):
    header = ['metric'] + names
    rows   = [header]

    def _row(label, vals):
        rows.append([label] + [f'{v:.4f}' for v in vals])

    _row('overall_accuracy',   [m['overall']['accuracy']    for m in all_metrics])
    _row('overall_top2_acc',   [m['overall']['top2_accuracy'] for m in all_metrics])
    _row('overall_macro_f1',   [m['overall']['macro_f1']    for m in all_metrics])
    _row('overall_event_rate', [m['overall']['event_detection_rate'] for m in all_metrics])
    rows.append([])

    for i, cls in enumerate(classes):
        for key in ('accuracy', 'top2_accuracy', 'avg_true_prob', 'max_true_prob',
                    'event_rate', 'prob_event_rate'):
            _row(f'{cls}_{key}', [m['per_class'][i].get(key, 0.0) for m in all_metrics])
        rows.append([f'{cls}_n_videos_detected'] +
                    [f"{m['per_class'][i]['n_videos_detected']}/{m['per_class'][i]['n_videos']}"
                     for m in all_metrics])
        rows.append([f'{cls}_n_windows'] +
                    [str(m['per_class'][i]['n_windows']) for m in all_metrics])
        rows.append([])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {Path(out_path).name}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description='KP EMA alpha ablation inference evaluation.')
    parser.add_argument('--yolo',             default=DEFAULT_YOLO)
    parser.add_argument('--imgsz',            type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument('--conf',             type=float, default=DEFAULT_CONF)
    parser.add_argument('--sequence_length',  type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument('--classify_stride',  type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--device',           default=DEFAULT_DEVICE)
    parser.add_argument('--output',           default=HARD_OUTPUT_DIR)
    args = parser.parse_args()

    out_root = Path(args.output)
    run_num  = _next_run_number(str(out_root))
    out_dir  = out_root / f"ema_ablation_{run_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names  = [m['name']  for m in MODELS]
    alphas = [m['alpha'] for m in MODELS]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(MODELS))]

    sep = '=' * 64
    print(f"\n{sep}")
    print(f"  KP EMA Alpha Ablation — Inference Evaluation  #{run_num:03d}")
    for m, col in zip(MODELS, colors):
        print(f"  {m['name']:<20}  {Path(m['path']).name}")
    print(f"  Output: {out_dir}")
    print(f"{sep}")

    # Load YOLO
    print('\n[Loading YOLO]')
    kp_det = KeypointDetector(args.yolo, device=args.device,
                               imgsz=args.imgsz, conf_thres=args.conf)

    # Load classifiers
    print('\n[Loading ST-GCN models]')
    classifiers = []
    feature_modes = []
    for m in MODELS:
        bn_ch = infer_bn_input_channels(m['path'])
        fm    = CH_TO_FEATURE.get(bn_ch, 'xy')
        print(f"  ✓ {Path(m['path']).name}  → {fm} ({bn_ch} ch)  α={m['alpha']}")
        clf = BehaviorClassifier(
            m['path'], device=args.device,
            sequence_length=args.sequence_length,
            normalize=True, feature_mode=fm, in_channels=bn_ch,
        )
        classifiers.append(clf)
        feature_modes.append(fm)

    # Video folders
    _all_dirs = [
        (HARD_VIDEO_WALK_DIR,    0),
        (HARD_VIDEO_LICK_DIR,    1),
        (HARD_VIDEO_SCRATCH_DIR, 2),
        (HARD_VIDEO_SHAKE_DIR,   3),
        (HARD_VIDEO_STOP_DIR,    4),
    ]
    dirs    = [(p, idx) for p, idx in _all_dirs if p and Path(p).is_dir()]
    skipped = [(BEHAVIOR_CLASSES[idx], p) for p, idx in _all_dirs
               if not p or not Path(p).is_dir()]
    for cls_name, p in skipped:
        print(f"  ⚠ 跳過 [{cls_name}]" + (f" 找不到資料夾: {p}" if p else " 路徑未設定"))

    # Inference — all models, all class folders
    all_preds = [{} for _ in MODELS]   # all_preds[model_idx][cls_idx] = [[pred,...],...]

    for dir_path, cls_idx in dirs:
        cls_name = BEHAVIOR_CLASSES[cls_idx]
        print(f"\n[{cls_name.upper()}]  {Path(dir_path).name}/")
        for mi, (clf, fm, m, name) in enumerate(zip(classifiers, feature_modes, MODELS, names)):
            print(f"  ▶ {name}")
            results = evaluate_folder(dir_path, kp_det, clf, fm,
                                      args.sequence_length, args.classify_stride,
                                      ema_alpha=m['alpha'])
            all_preds[mi][cls_idx] = [p for _, p in results]
            nw = sum(len(v) for v in all_preds[mi][cls_idx])
            print(f"    → {len(all_preds[mi][cls_idx])} videos  {nw} windows")

    # Metrics
    all_metrics = [compute_metrics(preds) for preds in all_preds]

    # Console summary
    print(f"\n{sep}")
    col_w = max(len(n) for n in names)
    print(f"  {'Metric':<28}  " + "  ".join(f"{n:>{col_w}}" for n in names))
    print(f"  {'-' * (28 + (col_w + 2) * len(names))}")

    def _row_print(label, vals):
        best = max(vals)
        parts = []
        for v in vals:
            marker = ' ★' if abs(v - best) < 0.001 else '  '
            parts.append(f"{v:>{col_w}.4f}{marker}")
        print(f"  {label:<28}  " + "  ".join(parts))

    _row_print('Overall Accuracy',    [m['overall']['accuracy']             for m in all_metrics])
    _row_print('Overall Macro-F1',    [m['overall']['macro_f1']             for m in all_metrics])
    _row_print('Overall Top-2 Acc',   [m['overall']['top2_accuracy']        for m in all_metrics])
    _row_print('Event Detection Rate',[m['overall']['event_detection_rate'] for m in all_metrics])
    print()
    for i, cls in enumerate(BEHAVIOR_CLASSES):
        if all(all_preds[mi].get(i) is None or len(all_preds[mi].get(i, [])) == 0
               for mi in range(len(MODELS))):
            continue
        _row_print(f'{cls} accuracy',      [m['per_class'][i]['accuracy']      for m in all_metrics])
        _row_print(f'{cls} avg_true_prob',  [m['per_class'][i]['avg_true_prob'] for m in all_metrics])
        _row_print(f'{cls} event_rate',     [m['per_class'][i]['event_rate']    for m in all_metrics])
    print(sep)

    # Flat preds for histogram
    all_flat = []
    for mi in range(len(MODELS)):
        flat = {cls_idx: [p for vid in vids for p in vid]
                for cls_idx, vids in all_preds[mi].items()}
        all_flat.append(flat)

    # Save outputs
    print('\n[Saving]')
    save_summary_csv(all_metrics, names, BEHAVIOR_CLASSES, out_dir / 'ema_ablation_summary.csv')
    plot_accuracy_comparison(all_metrics, names, colors, BEHAVIOR_CLASSES,
                             out_dir / 'ema_ablation_accuracy.png')
    plot_confusion_matrices(all_metrics, names, colors, BEHAVIOR_CLASSES,
                            out_dir / 'ema_ablation_confusion.png')
    plot_prob_histograms(all_flat, names, colors, BEHAVIOR_CLASSES,
                         out_dir / 'ema_ablation_prob_hist.png')
    print(f'\n✓ All results saved to: {out_dir}')


if __name__ == '__main__':
    main()
