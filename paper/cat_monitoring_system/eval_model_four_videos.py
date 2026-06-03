"""
Compare two ST-GCN models on up to five labeled videos (walk, lick, scratch, shake, stop).

Metrics (per model, per class):
  1. Discrete accuracy     — argmax(probs) == true_label  (硬指標)
  2. Avg true-class prob   — mean(probs[:, true_class])   (軟指標：信心度)
  3. Macro F1 & confusion matrix                           (分類輪廓)

Output directory: <output>/comparison_NNN_<nameA>_vs_<nameB>/
  NNN = next sequential number, consistent with 0_train_gcn.py convention

Files:
  <nameA>_<class>_preds.csv  <nameB>_<class>_preds.csv
  comparison_summary.csv
  accuracy_comparison.png    (grouped bar: discrete acc + avg true-class prob)
  confusion_matrices.png     (side-by-side heatmaps)
"""
import argparse
import csv
import re
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ── Channel → feature mode ────────────────────────────────────────────────
CH_TO_FEATURE = {
    4: 'xy_v',
    5: 'xy_conf_v',
    7: 'xy_conf_v_bone',
    9: 'xy_conf_v_bone_bmotion',
}

# ── Default / hardcoded paths ─────────────────────────────────────────────
DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\v11s_90.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 2
DEFAULT_DEVICE  = 'cuda'

HARD_MODEL_A    = r"C:\Users\homec\Downloads\stgcn_best_020_xy_v_att_on.pth"
HARD_MODEL_B    = r"C:\Users\homec\Downloads\stgcn_best_021_xy_v_att_on.pth"
HARD_NAME_A     = None   # None → auto-derived from filename
HARD_NAME_B     = None
HARD_VIDEO_WALK    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk\walk_5.mp4"
HARD_VIDEO_LICK    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\lick9.mp4"
HARD_VIDEO_SCRATCH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\scratch\scratch_ok (2).mp4"
HARD_VIDEO_SHAKE   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\shake\shake_15.mp4"
HARD_VIDEO_STOP    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\stop\6月2日(1).mp4"   # 留空表示跳過 stop 類別評估
HARD_OUTPUT_DIR    = r"C:\ai_project\paper\cat_monitoring_system\eval_results"

# ── Visual style ──────────────────────────────────────────────────────────
_COL_A = '#2196F3'   # blue   — model A
_COL_B = '#FF9800'   # orange — model B


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def _next_comparison_number(out_root: str) -> int:
    """掃描 out_root/comparison_NNN_* 目錄，回傳下一個可用編號。"""
    p = Path(out_root)
    if not p.exists():
        return 1
    pat = re.compile(r'^comparison_(\d+)')
    max_num = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def _short_name(model_path: str) -> str:
    """從模型路徑提取簡短名稱，優先取 run number + mode 部分。"""
    stem = Path(model_path).stem          # e.g. stgcn_best_001_xy_v_att_on
    parts = stem.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) >= 3:
            return '_'.join(parts[i:])    # e.g. 001_xy_v_att_on
    return stem[-28:] if len(stem) > 28 else stem


def infer_bn_input_channels(model_path: str):
    """從 checkpoint 的 bn_input.weight 推斷輸入通道數。"""
    try:
        ck = torch.load(model_path, map_location='cpu')
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        if isinstance(sd, dict):
            for k, v in sd.items():
                if k.endswith('bn_input.weight'):
                    return int(v.shape[0])
    except Exception as e:
        print(f"  ⚠ channel inference failed for {model_path}: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_video(video_path, kp_detector, classifier, feature_mode,
                   sequence_length=16, classify_stride=2):
    """
    對單一影片執行逐幀推論。
    回傳 list of dicts: {frame, time, pred, conf, probs}
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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

        kpts_arr = np.array([
            item[0] if item[0] is not None else np.zeros((17, 2), np.float32)
            for item in buf
        ])
        conf_arr = np.array([
            item[1] if item[1] is not None else np.zeros((17,), np.float32)
            for item in buf
        ])
        seq = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        feats = build_feature_tensor(seq, conf_arr, feature_mode)

        pred_id, pred_conf, pred_probs = classifier.classify(feats, precomputed=True)
        if pred_id is None:
            pred_id, pred_conf, pred_probs = -1, 0.0, [0.0] * len(BEHAVIOR_CLASSES)

        preds.append({
            'frame': frame_idx,
            'time':  round(frame_idx / fps, 3),
            'pred':  int(pred_id),
            'conf':  float(pred_conf),
            'probs': [float(x) for x in pred_probs],
        })

    cap.release()
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_by_class: dict) -> dict:
    """
    preds_by_class: {class_idx: [pred_dict, ...]}

    Per-class metrics:
      accuracy      — discrete accuracy (argmax == true label)
      avg_true_prob — mean predicted probability for the true class
      n_windows     — total inference windows used

    Overall metrics:
      accuracy, macro_f1, confusion_matrix
    """
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    # 預先為所有類別初始化，避免未提供影片的類別造成 KeyError
    per_class = {i: {'accuracy': 0.0, 'avg_true_prob': 0.0, 'n_windows': 0}
                 for i in range(n_cls)}

    for cls_idx, preds in preds_by_class.items():
        if not preds:
            continue

        probs      = np.array([p['probs'] for p in preds])   # (N, C_actual)
        actual_cls = probs.shape[1]   # 模型實際輸出類別數，可能 < n_cls
        pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int),
                             0, actual_cls - 1)

        # 若 cls_idx 超出模型輸出範圍（舊模型缺少新類別），avg_true_prob 設 0
        avg_tp = float(probs[:, cls_idx].mean()) if cls_idx < actual_cls else 0.0

        per_class[cls_idx] = {
            'accuracy':      float((pred_ids == cls_idx).mean()),
            'avg_true_prob': avg_tp,
            'n_windows':     len(preds),
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    return {
        'per_class': per_class,
        'overall': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(metrics_a, metrics_b, name_a, name_b, classes, out_path):
    """
    雙列 grouped bar chart：
      Row 0 — Discrete Accuracy   (硬指標：argmax 是否正確)
      Row 1 — Avg True-Class Prob (軟指標：模型對正確類別的平均信心)

    每個 group 右側標示贏家（★）。
    """
    n_cls = len(classes)
    x_labels = classes + ['Overall']
    x = np.arange(len(x_labels))
    w = 0.33

    def _vals(metrics, key):
        pc = metrics['per_class']
        out = [pc[i][key] for i in range(n_cls)]
        # Overall column
        if key == 'accuracy':
            out.append(metrics['overall']['accuracy'])
        else:
            out.append(float(np.mean([pc[i]['avg_true_prob'] for i in range(n_cls)])))
        return out

    acc_a  = _vals(metrics_a, 'accuracy')
    acc_b  = _vals(metrics_b, 'accuracy')
    prob_a = _vals(metrics_a, 'avg_true_prob')
    prob_b = _vals(metrics_b, 'avg_true_prob')

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        f'ST-GCN Model Comparison\n'
        f'A: {name_a}   |   B: {name_b}',
        fontsize=12, fontweight='bold'
    )

    def _draw(ax, vals_a, vals_b, title, ylabel):
        bars_a = ax.bar(x - w / 2, vals_a, w, label=f'A: {name_a}', color=_COL_A, alpha=0.88)
        bars_b = ax.bar(x + w / 2, vals_b, w, label=f'B: {name_b}', color=_COL_B, alpha=0.88)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        # Value labels on bars
        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f'{h:.1%}', ha='center', va='bottom', fontsize=8)

        # Winner star above each group
        for i in range(len(x_labels)):
            va, vb = vals_a[i], vals_b[i]
            if abs(va - vb) < 0.005:
                continue
            winner_x = x[i] - w / 2 if va > vb else x[i] + w / 2
            winner_h = max(va, vb)
            col = _COL_A if va > vb else _COL_B
            ax.text(winner_x, winner_h + 0.055, '★',
                    ha='center', va='bottom', fontsize=11, color=col)

        # Δ delta labels between bars
        for i in range(len(x_labels)):
            va, vb = vals_a[i], vals_b[i]
            diff = abs(va - vb)
            if diff < 0.005:
                label = '='
            else:
                label = f'Δ{diff:.1%}'
            ax.text(x[i], 1.09, label, ha='center', va='bottom',
                    fontsize=8, color='#555')

    _draw(axes[0], acc_a,  acc_b,
          'Discrete Accuracy  (argmax == true label)',
          'Accuracy')
    _draw(axes[1], prob_a, prob_b,
          'Avg True-Class Probability  (model conviction)',
          'Avg Prob')

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_confusion_matrices(cm_a, cm_b, name_a, name_b, classes, out_path):
    """兩個 confusion matrix 並排，支援 seaborn（可選）。"""
    try:
        import seaborn as sns
        _sns = True
    except ImportError:
        _sns = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')

    for ax, cm, name, col in zip(axes, [cm_a, cm_b], [name_a, name_b], [_COL_A, _COL_B]):
        if _sns:
            # Normalize to show row-wise recall
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, cbar=True, linewidths=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=30, ha='right')
            ax.set_yticklabels(classes)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black',
                            fontsize=12)

        # Per-class recall on diagonal annotation
        diag_recall = [cm[i, i] / max(cm[i].sum(), 1) for i in range(len(classes))]
        avg_recall = float(np.mean(diag_recall))

        ax.set_title(f'{name}\nAvg Recall = {avg_recall:.1%}', fontsize=11,
                     fontweight='bold', color=col)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV outputs
# ═══════════════════════════════════════════════════════════════════════════
def save_summary_csv(metrics_a, metrics_b, name_a, name_b, classes, out_path):
    """對比摘要 CSV：每行一個指標，含贏家欄位。"""
    rows = [['metric', name_a, name_b, 'winner', 'delta']]

    def _row(label, va, vb):
        delta = abs(va - vb)
        winner = '=' if delta < 0.001 else (name_a if va >= vb else name_b)
        rows.append([label, f'{va:.4f}', f'{vb:.4f}', winner, f'{delta:.4f}'])

    _row('overall_accuracy', metrics_a['overall']['accuracy'], metrics_b['overall']['accuracy'])
    _row('overall_macro_f1', metrics_a['overall']['macro_f1'], metrics_b['overall']['macro_f1'])
    rows.append([])   # blank separator

    for i, cls in enumerate(classes):
        pa = metrics_a['per_class'][i]
        pb = metrics_b['per_class'][i]
        _row(f'{cls}_accuracy',      pa['accuracy'],      pb['accuracy'])
        _row(f'{cls}_avg_true_prob', pa['avg_true_prob'], pb['avg_true_prob'])
        rows.append([f'{cls}_n_windows', pa['n_windows'], pb['n_windows'], '-', '-'])
        rows.append([])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out_path.name}")


def save_preds_csv(preds_by_class: dict, model_name: str, classes, out_dir: Path):
    """儲存每類別的逐窗口預測結果。"""
    for cls_idx, preds in preds_by_class.items():
        cls_name = classes[cls_idx]
        out = out_dir / f"{model_name}_{cls_name}_preds.csv"
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'time', 'pred_class', 'conf'] +
                       [f'prob_{c}' for c in classes])
            for p in preds:
                w.writerow([p['frame'], p['time'],
                            classes[p['pred']] if 0 <= p['pred'] < len(classes) else 'none',
                            f"{p['conf']:.4f}"] +
                           [f"{x:.4f}" for x in p['probs']])


# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
def print_final_summary(metrics_a, metrics_b, name_a, name_b):
    """執行結束後在終端列印人類可讀的對比分析。"""
    classes = BEHAVIOR_CLASSES
    n_cls   = len(classes)

    oa = metrics_a['overall']['accuracy']
    ob = metrics_b['overall']['accuracy']
    fa = metrics_a['overall']['macro_f1']
    fb = metrics_b['overall']['macro_f1']

    per_a = metrics_a['per_class']
    per_b = metrics_b['per_class']

    BOX  = '═' * 62
    SEP  = '─' * 62
    NL   = ''

    def _winner_str(va, vb, na, nb):
        d = abs(va - vb)
        if d < 0.001:
            return '='
        return f'→ {na}  +{d:.1%}' if va > vb else f'→ {nb}  +{d:.1%}'

    lines = [
        NL,
        f'╔{BOX}╗',
        f'║{"  FINAL COMPARISON SUMMARY":^62}║',
        f'╚{BOX}╝',
        NL,
        '● Overall Performance',
        f'  {"Metric":<14}  {"Model A":>14}  {"Model B":>14}  Result',
        f'  {SEP}',
        f'  {"Accuracy":<14}  {oa:>14.1%}  {ob:>14.1%}  {_winner_str(oa, ob, name_a, name_b)}',
        f'  {"Macro F1":<14}  {fa:>14.1%}  {fb:>14.1%}  {_winner_str(fa, fb, name_a, name_b)}',
        NL,
    ]

    # Overall winner verdict
    a_pts = (1 if oa > ob else 0) + (1 if fa > fb else 0)
    b_pts = (1 if ob > oa else 0) + (1 if fb > fa else 0)
    if a_pts > b_pts:
        verdict = f'★ {name_a} wins both overall metrics.'
    elif b_pts > a_pts:
        verdict = f'★ {name_b} wins both overall metrics.'
    else:
        verdict = f'★ Split: {name_a} leads Accuracy, {name_b} leads Macro-F1.' \
                  if oa > ob else f'★ Split: {name_b} leads Accuracy, {name_a} leads Macro-F1.'
    lines += [f'  {verdict}', NL]

    # Per-class accuracy breakdown
    lines += ['● Per-Class Accuracy  (argmax == true label)', f'  {SEP}']
    a_class_wins, b_class_wins = 0, 0
    class_gaps = []
    for i, cls in enumerate(classes):
        va = per_a[i]['accuracy']
        vb = per_b[i]['accuracy']
        d  = abs(va - vb)
        class_gaps.append((d, cls, va, vb))
        if abs(va - vb) < 0.005:
            tag = '  ='
        elif va > vb:
            a_class_wins += 1
            tag = f'  → {name_a}  +{d:.1%}'
        else:
            b_class_wins += 1
            tag = f'  → {name_b}  +{d:.1%}'
        lines.append(f'  {cls:<10}  A={va:>6.1%}  B={vb:>6.1%}  {tag}')
    lines.append(NL)
    n_evaluated = sum(1 for i in range(n_cls) if per_a[i]['n_windows'] > 0)
    lines.append(f'  Class wins → {name_a}: {a_class_wins}/{n_evaluated}   {name_b}: {b_class_wins}/{n_evaluated}')
    lines.append(NL)

    # Per-class avg true-class probability
    lines += ['● Avg True-Class Probability  (model conviction)', f'  {SEP}']
    for i, cls in enumerate(classes):
        va = per_a[i]['avg_true_prob']
        vb = per_b[i]['avg_true_prob']
        d  = abs(va - vb)
        if abs(va - vb) < 0.005:
            tag = '='
        else:
            tag = f'→ {name_a} +{d:.1%}' if va > vb else f'→ {name_b} +{d:.1%}'
        lines.append(f'  {cls:<10}  A={va:>6.1%}  B={vb:>6.1%}  {tag}')
    lines.append(NL)

    # Biggest gap class
    class_gaps.sort(reverse=True)
    biggest_gap, biggest_cls, gap_va, gap_vb = class_gaps[0]
    gap_winner = name_a if gap_va > gap_vb else name_b
    lines.append(f'● Biggest gap: [{biggest_cls}]  Δ={biggest_gap:.1%}  ({gap_winner} leads)')

    # Strongest / weakest per model
    acc_a = [(per_a[i]['accuracy'], classes[i]) for i in range(n_cls)]
    acc_b = [(per_b[i]['accuracy'], classes[i]) for i in range(n_cls)]
    best_a  = max(acc_a);  worst_a = min(acc_a)
    best_b  = max(acc_b);  worst_b = min(acc_b)
    lines += [
        NL,
        '● Per-model profile',
        f'  {name_a:<20}  best={best_a[1]}({best_a[0]:.1%})  worst={worst_a[1]}({worst_a[0]:.1%})',
        f'  {name_b:<20}  best={best_b[1]}({best_b[0]:.1%})  worst={worst_b[1]}({worst_b[0]:.1%})',
        NL,
    ]

    # Recommendation
    lines.append('● Recommendation')
    if a_pts > b_pts:
        lines.append(f'  {name_a} is the stronger model overall.')
    elif b_pts > a_pts:
        lines.append(f'  {name_b} is the stronger model overall.')
    else:
        lines.append(f'  Both models are comparable overall; see per-class breakdown for use-case selection.')
    if biggest_gap >= 0.10:
        lines.append(f'  Note: [{biggest_cls}] shows a large gap (Δ{biggest_gap:.1%}); '
                     f'prioritize {gap_winner} if this class is critical.')
    lines.append(NL)

    print('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Compare two ST-GCN models on four labeled videos.'
    )
    parser.add_argument('--model_a',         default=HARD_MODEL_A)
    parser.add_argument('--model_b',         default=HARD_MODEL_B)
    parser.add_argument('--name_a',          default=HARD_NAME_A)
    parser.add_argument('--name_b',          default=HARD_NAME_B)
    parser.add_argument('--video_walk',      default=HARD_VIDEO_WALK)
    parser.add_argument('--video_lick',      default=HARD_VIDEO_LICK)
    parser.add_argument('--video_scratch',   default=HARD_VIDEO_SCRATCH)
    parser.add_argument('--video_shake',     default=HARD_VIDEO_SHAKE)
    parser.add_argument('--video_stop',      default=HARD_VIDEO_STOP,
                        help='stop 類別影片路徑（留空則跳過此類別）')
    parser.add_argument('--yolo',            default=DEFAULT_YOLO)
    parser.add_argument('--imgsz',           type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument('--conf',            type=float, default=DEFAULT_CONF)
    parser.add_argument('--output',          default=HARD_OUTPUT_DIR)
    parser.add_argument('--sequence_length', type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument('--classify_stride', type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--device',          default=DEFAULT_DEVICE)
    args = parser.parse_args()

    name_a = args.name_a or _short_name(args.model_a)
    name_b = args.name_b or _short_name(args.model_b)

    # Sequential output directory
    out_root = Path(args.output)
    run_num  = _next_comparison_number(str(out_root))
    run_tag  = f"{run_num:03d}"
    out_dir  = out_root / f"comparison_{run_tag}_{name_a}_vs_{name_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  Comparison #{run_tag}")
    print(f"  Model A : {name_a}")
    print(f"  Model B : {name_b}")
    print(f"  Output  : {out_dir}")
    print(f"{sep}")

    # Load detectors
    print('\n[Loading YOLO]')
    kp_det = KeypointDetector(args.yolo, device=args.device,
                               imgsz=args.imgsz, conf_thres=args.conf)

    def _load_classifier(model_path, device):
        bn_ch = infer_bn_input_channels(model_path)
        fm    = CH_TO_FEATURE.get(bn_ch, 'xy_v')
        if bn_ch not in CH_TO_FEATURE:
            print(f"  ⚠ {Path(model_path).name}: cannot infer feature_mode "
                  f"(bn_ch={bn_ch}), defaulting to 'xy_v'")
        else:
            print(f"  ✓ {Path(model_path).name} → {fm} ({bn_ch} ch)")
        clf = BehaviorClassifier(
            model_path, device=device,
            sequence_length=args.sequence_length,
            normalize=True, feature_mode=fm, in_channels=bn_ch,
        )
        return clf, fm

    print('\n[Loading ST-GCN models]')
    clf_a, fm_a = _load_classifier(args.model_a, args.device)
    clf_b, fm_b = _load_classifier(args.model_b, args.device)

    _all_videos = [
        (args.video_walk,    0),
        (args.video_lick,    1),
        (args.video_scratch, 2),
        (args.video_shake,   3),
        (args.video_stop,    4),
    ]
    # 路徑為空或檔案不存在時跳過，避免 KeyError 也支援尚無 stop 資料的情況
    videos = [(p, idx) for p, idx in _all_videos if p and Path(p).exists()]
    skipped = [(BEHAVIOR_CLASSES[idx], p) for p, idx in _all_videos
               if not p or not Path(p).exists()]
    if skipped:
        for cls_name, p in skipped:
            reason = '（路徑未設定）' if not p else f'（找不到檔案: {p}）'
            print(f"  ⚠ 跳過 [{cls_name}] {reason}")

    # ── Inference ──────────────────────────────────────────────────────────
    preds_a: dict = {}
    preds_b: dict = {}

    for vid_path, cls_idx in videos:
        cls_name = BEHAVIOR_CLASSES[cls_idx]
        print(f"\n[{cls_name.upper()}]  {Path(vid_path).name}")
        print(f"  A ({name_a}) ...", end=' ', flush=True)
        preds_a[cls_idx] = evaluate_video(
            vid_path, kp_det, clf_a, fm_a,
            args.sequence_length, args.classify_stride
        )
        print(f"{len(preds_a[cls_idx])} windows")

        print(f"  B ({name_b}) ...", end=' ', flush=True)
        preds_b[cls_idx] = evaluate_video(
            vid_path, kp_det, clf_b, fm_b,
            args.sequence_length, args.classify_stride
        )
        print(f"{len(preds_b[cls_idx])} windows")

    # ── Compute metrics ─────────────────────────────────────────────────────
    metrics_a = compute_metrics(preds_a)
    metrics_b = compute_metrics(preds_b)

    # ── Console summary ─────────────────────────────────────────────────────
    col_w = max(len(name_a), len(name_b), 10)
    hdr = f"  {'Metric':<30}  {name_a:>{col_w}}  {name_b:>{col_w}}  Winner"
    print(f"\n{sep}\n{hdr}\n  {'-' * (len(hdr) - 2)}")

    def _row(label, va, vb):
        diff = abs(va - vb)
        if diff < 0.001:
            winner = '='
        else:
            winner = f'+A Δ{diff:.1%}' if va > vb else f'+B Δ{diff:.1%}'
        print(f"  {label:<30}  {va:>{col_w}.4f}  {vb:>{col_w}.4f}  {winner}")

    _row('Overall Accuracy',    metrics_a['overall']['accuracy'],  metrics_b['overall']['accuracy'])
    _row('Overall Macro-F1',    metrics_a['overall']['macro_f1'],  metrics_b['overall']['macro_f1'])
    print()
    evaluated_cls = {idx for _, idx in videos}
    for i, cls in enumerate(BEHAVIOR_CLASSES):
        if i not in evaluated_cls:
            continue
        pa, pb = metrics_a['per_class'][i], metrics_b['per_class'][i]
        _row(f'{cls:<8} accuracy',      pa['accuracy'],      pb['accuracy'])
        _row(f'{cls:<8} avg_true_prob', pa['avg_true_prob'], pb['avg_true_prob'])

    print(sep)

    # ── Save outputs ─────────────────────────────────────────────────────────
    print('\n[Saving]')
    save_preds_csv(preds_a, name_a, BEHAVIOR_CLASSES, out_dir)
    save_preds_csv(preds_b, name_b, BEHAVIOR_CLASSES, out_dir)
    save_summary_csv(
        metrics_a, metrics_b, name_a, name_b,
        BEHAVIOR_CLASSES, out_dir / 'comparison_summary.csv'
    )
    plot_accuracy_comparison(
        metrics_a, metrics_b, name_a, name_b,
        BEHAVIOR_CLASSES, out_dir / 'accuracy_comparison.png'
    )
    plot_confusion_matrices(
        metrics_a['confusion_matrix'], metrics_b['confusion_matrix'],
        name_a, name_b, BEHAVIOR_CLASSES,
        out_dir / 'confusion_matrices.png'
    )

    print(f'\n✓ All results saved to: {out_dir}')

    print_final_summary(metrics_a, metrics_b, name_a, name_b)


if __name__ == '__main__':
    main()
