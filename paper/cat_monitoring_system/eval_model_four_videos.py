"""
Compare two ST-GCN models on up to five labeled videos (walk, lick, scratch, shake, stop).

Metrics (per model, per class):
  1. Discrete accuracy     — argmax(probs) == true_label       (硬指標)
  2. Avg true-class prob   — mean(probs[:, true_class])        (軟指標：平均信心)
  3. Max true-class prob   — max(probs[:, true_class])         (峰值信號：事件偵測能力)
  4. Event detected        — n_correct >= EVENT_MIN_WINDOWS    (影片層級偵測，對含靜止段影片最公平)
  5. Macro F1 & confusion matrix                                (分類輪廓)

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
DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\v11s_94.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 2
EVENT_MIN_WINDOWS = 3   # 事件偵測門檻：一部影片中 ≥ N 個 window 預測正確即視為偵測到該行為
DEFAULT_DEVICE  = 'cuda'

HARD_MODEL_A    = r"C:\Users\homec\Downloads\stgcn_best_033_xy_v_att_on.pth"
HARD_MODEL_B    = r"C:\Users\homec\Downloads\stgcn_best_032_xy_v_att_on.pth"
HARD_NAME_A     = None   # None → auto-derived from filename
HARD_NAME_B     = None
HARD_VIDEO_WALK    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk\walk1.mp4"
HARD_VIDEO_LICK    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\lick\lick_9.mp4"
HARD_VIDEO_SCRATCH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\scratch\scratch_33.mp4"
HARD_VIDEO_SHAKE   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\shake\shake_15.mp4"
HARD_VIDEO_STOP    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\stop\stop_35.mp4"   # 留空表示跳過 stop 類別評估
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
def compute_metrics(preds_by_class: dict,
                    event_min_windows: int = EVENT_MIN_WINDOWS) -> dict:
    """
    preds_by_class: {class_idx: [pred_dict, ...]}

    Per-class metrics:
      accuracy        — discrete accuracy (argmax == true label)
      avg_true_prob   — mean predicted probability for the true class
      max_true_prob   — peak predicted probability for the true class
      event_detected  — True if n_correct >= event_min_windows
      n_correct       — number of windows correctly predicted
      n_windows       — total inference windows used

    Overall metrics:
      accuracy, macro_f1, event_detection_rate, confusion_matrix
    """
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class = {i: {'accuracy': 0.0, 'avg_true_prob': 0.0, 'max_true_prob': 0.0,
                     'event_detected': False, 'n_correct': 0, 'n_windows': 0}
                 for i in range(n_cls)}

    for cls_idx, preds in preds_by_class.items():
        if not preds:
            continue

        probs      = np.array([p['probs'] for p in preds])
        actual_cls = probs.shape[1]
        pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int),
                             0, actual_cls - 1)

        probs_cls = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(len(preds))
        n_correct = int((pred_ids == cls_idx).sum())

        per_class[cls_idx] = {
            'accuracy':       float(n_correct / len(preds)),
            'avg_true_prob':  float(probs_cls.mean()),
            'max_true_prob':  float(probs_cls.max()),
            'event_detected': n_correct >= event_min_windows,
            'n_correct':      n_correct,
            'n_windows':      len(preds),
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    evaluated = [i for i in range(n_cls) if per_class[i]['n_windows'] > 0]
    event_rate = float(np.mean([per_class[i]['event_detected'] for i in evaluated])) \
                 if evaluated else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'accuracy':             float(accuracy_score(y_true, y_pred)),
            'macro_f1':             float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'event_detection_rate': event_rate,
        },
        'confusion_matrix':  confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
        'event_min_windows': event_min_windows,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(metrics_a, metrics_b, name_a, name_b, classes, out_path):
    """
    三列 grouped bar chart：
      Row 0 — Discrete Accuracy         (硬指標：argmax 是否正確) + ✓/✗ 事件偵測標記
      Row 1 — Avg True-Class Prob       (軟指標：平均信心)
      Row 2 — Max True-Class Prob       (峰值信號：整部影片最高信心，反映事件偵測能力)
    """
    n_cls  = len(classes)
    x_labels = classes + ['Overall']
    x      = np.arange(len(x_labels))
    w      = 0.33
    ev_min = metrics_a.get('event_min_windows', EVENT_MIN_WINDOWS)

    def _vals(metrics, key):
        pc  = metrics['per_class']
        out = [pc[i].get(key, 0.0) for i in range(n_cls)]
        if key == 'accuracy':
            out.append(metrics['overall']['accuracy'])
        else:
            out.append(float(np.mean([pc[i].get(key, 0.0) for i in range(n_cls)])))
        return out

    def _evt(metrics):
        pc  = metrics['per_class']
        out = [pc[i].get('event_detected', False) for i in range(n_cls)]
        # Overall ✓ = 所有已評估類別都偵測到
        out.append(all(pc[i].get('event_detected', False)
                       for i in range(n_cls) if pc[i].get('n_windows', 0) > 0))
        return out

    acc_a  = _vals(metrics_a, 'accuracy')
    acc_b  = _vals(metrics_b, 'accuracy')
    prob_a = _vals(metrics_a, 'avg_true_prob')
    prob_b = _vals(metrics_b, 'avg_true_prob')
    max_a  = _vals(metrics_a, 'max_true_prob')
    max_b  = _vals(metrics_b, 'max_true_prob')
    evt_a  = _evt(metrics_a)
    evt_b  = _evt(metrics_b)

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    fig.suptitle(
        f'ST-GCN Model Comparison\n'
        f'A: {name_a}   |   B: {name_b}',
        fontsize=12, fontweight='bold'
    )

    def _draw(ax, vals_a, vals_b, title, ylabel, evt_a_=None, evt_b_=None):
        bars_a = ax.bar(x - w / 2, vals_a, w, label=f'A: {name_a}', color=_COL_A, alpha=0.88)
        bars_b = ax.bar(x + w / 2, vals_b, w, label=f'B: {name_b}', color=_COL_B, alpha=0.88)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.28)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        for bar in list(bars_a) + list(bars_b):
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f'{h:.1%}', ha='center', va='bottom', fontsize=8)

        for i in range(len(x_labels)):
            va, vb = vals_a[i], vals_b[i]
            if abs(va - vb) >= 0.005:
                winner_x = x[i] - w / 2 if va > vb else x[i] + w / 2
                col      = _COL_A if va > vb else _COL_B
                ax.text(winner_x, max(va, vb) + 0.055, '★',
                        ha='center', va='bottom', fontsize=11, color=col)

        for i in range(len(x_labels)):
            va, vb = vals_a[i], vals_b[i]
            diff  = abs(va - vb)
            label = '=' if diff < 0.005 else f'Δ{diff:.1%}'
            ax.text(x[i], 1.10, label, ha='center', va='bottom',
                    fontsize=8, color='#555')

        # 事件偵測標記 ✓/✗
        if evt_a_ is not None:
            for i, det in enumerate(evt_a_):
                sym, col = ('✓', _COL_A) if det else ('✗', '#bbb')
                ax.text(x[i] - w / 2, 1.18, sym, ha='center', va='bottom',
                        fontsize=10, color=col, fontweight='bold')
        if evt_b_ is not None:
            for i, det in enumerate(evt_b_):
                sym, col = ('✓', _COL_B) if det else ('✗', '#bbb')
                ax.text(x[i] + w / 2, 1.18, sym, ha='center', va='bottom',
                        fontsize=10, color=col, fontweight='bold')

    _draw(axes[0], acc_a, acc_b,
          f'Discrete Accuracy  (argmax == true label)\n'
          f'✓/✗ = event detected (≥{ev_min} correct windows per clip)',
          'Accuracy', evt_a_=evt_a, evt_b_=evt_b)
    _draw(axes[1], prob_a, prob_b,
          'Avg True-Class Probability  (model conviction)',
          'Avg Prob')
    _draw(axes[2], max_a, max_b,
          'Max True-Class Probability  (peak event signal)',
          'Max Prob')

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

    _row('overall_accuracy',             metrics_a['overall']['accuracy'],             metrics_b['overall']['accuracy'])
    _row('overall_macro_f1',             metrics_a['overall']['macro_f1'],             metrics_b['overall']['macro_f1'])
    _row('overall_event_detection_rate', metrics_a['overall']['event_detection_rate'], metrics_b['overall']['event_detection_rate'])
    rows.append([])   # blank separator

    for i, cls in enumerate(classes):
        pa = metrics_a['per_class'][i]
        pb = metrics_b['per_class'][i]
        _row(f'{cls}_accuracy',       pa['accuracy'],      pb['accuracy'])
        _row(f'{cls}_avg_true_prob',  pa['avg_true_prob'], pb['avg_true_prob'])
        _row(f'{cls}_max_true_prob',  pa['max_true_prob'], pb['max_true_prob'])
        rows.append([f'{cls}_event_detected', str(pa['event_detected']), str(pb['event_detected']), '-', '-'])
        rows.append([f'{cls}_n_correct',  pa['n_correct'],  pb['n_correct'],  '-', '-'])
        rows.append([f'{cls}_n_windows',  pa['n_windows'],  pb['n_windows'],  '-', '-'])
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
# Composite scoring
# ═══════════════════════════════════════════════════════════════════════════
def _score_comparison(metrics_a, metrics_b, name_a, name_b) -> tuple:
    """
    多指標加權 Borda 計分，衡量兩個模型的綜合強弱。

    每個指標勝者得 W 分，差距 < 0.2% 視為平手（各得 W/2 分）。
    加權設計依據文獻慣例：
      Macro F1 / Min-class acc 各 W=2（最重要，行為辨識首選 + 木桶原則）
      其餘各 W=1（輔助判斷）

    參考：
      Macro F1 作為行為辨識主指標 —— NTU RGB+D, Kinetics benchmark 慣例
      Min-class accuracy (木桶原則) —— 部署可靠性評估
      Avg true-class prob            —— 對應 ECE 概念, Guo et al. ICML 2017
      Borda count 多指標排名          —— Kittler et al. 1998

    Returns:
        (score_a, score_b, breakdown_list)
    """
    n_cls     = len(BEHAVIOR_CLASSES)
    per_a     = metrics_a['per_class']
    per_b     = metrics_b['per_class']
    evaluated = [i for i in range(n_cls) if per_a[i]['n_windows'] > 0]
    ev_min    = metrics_a.get('event_min_windows', EVENT_MIN_WINDOWS)

    candidate_metrics = []

    # ── 1. Overall Accuracy  (W=1) ─────────────────────────────────────────
    # 最直觀的硬指標，argmax 是否正確；對類別不平衡不敏感。
    candidate_metrics.append(dict(
        label='Overall Accuracy', weight=1, higher_better=True,
        note='argmax 正確率，直觀但不考慮類別平衡',
        va=metrics_a['overall']['accuracy'],
        vb=metrics_b['overall']['accuracy'],
    ))

    # ── 2. Macro F1  (W=2) ─────────────────────────────────────────────────
    # 行為辨識論文的首選主指標（NTU RGB+D、Kinetics 慣例）。
    # 分別計算每類 F1 再平均，對少數類別（shake、scratch）更公平。
    candidate_metrics.append(dict(
        label='Macro F1', weight=2, higher_better=True,
        note='類別平均 F1，行為辨識首選指標，兼顧 Precision + Recall',
        va=metrics_a['overall']['macro_f1'],
        vb=metrics_b['overall']['macro_f1'],
    ))

    # ── 3. Avg True-Class Probability  (W=1) ───────────────────────────────
    # 軟指標：即使 argmax 相同，信心越高的模型部署時越穩定可靠。
    # 概念上對應 ECE（Expected Calibration Error）中的信心對齊程度。
    if evaluated:
        candidate_metrics.append(dict(
            label='Avg True-Class Prob', weight=1, higher_better=True,
            note='對正確類別的平均預測機率，反映校準度與部署穩定性（對應 ECE）',
            va=float(np.mean([per_a[i]['avg_true_prob'] for i in evaluated])),
            vb=float(np.mean([per_b[i]['avg_true_prob'] for i in evaluated])),
        ))

    # ── 4. Min Per-Class Accuracy  (W=2) ───────────────────────────────────
    # 「木桶原則」：最弱的類別決定系統下限。
    # 部署後若某類完全失效，其他類別再高也無意義。
    if evaluated:
        candidate_metrics.append(dict(
            label='Min Class Accuracy', weight=2, higher_better=True,
            note='最差類別準確率，衡量部署穩健性（木桶原則，最脆弱的一環）',
            va=float(min(per_a[i]['accuracy'] for i in evaluated)),
            vb=float(min(per_b[i]['accuracy'] for i in evaluated)),
        ))

    # ── 5. Std Per-Class Accuracy  (W=1, lower is better) ──────────────────
    # 均衡性：各類表現差距小，代表模型對所有行為都一致有效。
    if len(evaluated) > 1:
        candidate_metrics.append(dict(
            label='Std Class Accuracy', weight=1, higher_better=False,
            note='各類準確率標準差，越小代表各類表現越均衡（↓ lower is better）',
            va=float(np.std([per_a[i]['accuracy'] for i in evaluated])),
            vb=float(np.std([per_b[i]['accuracy'] for i in evaluated])),
        ))

    # ── 6. Event Detection Rate  (W=2) ─────────────────────────────────────
    # 影片層級偵測率：部署場景最實用的指標。
    # 一部影片中只要出現 ≥ ev_min 個正確 window，即視為成功偵測到該行為事件。
    # 對「甩頭影片大部分是靜止」這類場景特別公平，不因靜止段拉低整體 accuracy。
    candidate_metrics.append(dict(
        label='Event Detection Rate', weight=2, higher_better=True,
        note=f'影片層級偵測率，≥{ev_min} windows 正確即算偵測成功（對含靜止段的影片最公平）',
        va=metrics_a['overall']['event_detection_rate'],
        vb=metrics_b['overall']['event_detection_rate'],
    ))

    # ── Borda 加權計分 ─────────────────────────────────────────────────────
    TIE_THRESHOLD = 0.002   # 差距 < 0.2% 視為平手
    score_a = score_b = 0.0
    breakdown = []

    for m in candidate_metrics:
        va, vb, w = m['va'], m['vb'], m['weight']
        diff   = abs(va - vb)
        wins_a = (va > vb) if m['higher_better'] else (va < vb)

        if diff < TIE_THRESHOLD:
            pts_a = pts_b = w / 2
            result = '='
        elif wins_a:
            pts_a, pts_b = float(w), 0.0
            result = f'→ {name_a}'
        else:
            pts_a, pts_b = 0.0, float(w)
            result = f'→ {name_b}'

        score_a += pts_a
        score_b += pts_b
        breakdown.append({**m, 'pts_a': pts_a, 'pts_b': pts_b, 'result': result})

    return score_a, score_b, breakdown


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

    # 加權 Borda 綜合計分
    score_a, score_b, score_breakdown = _score_comparison(metrics_a, metrics_b, name_a, name_b)
    max_score = sum(m['weight'] for m in score_breakdown)
    if score_a > score_b:
        verdict = f'★ {name_a} leads on composite score  ({score_a:.1f} / {max_score:.0f} pts)'
    elif score_b > score_a:
        verdict = f'★ {name_b} leads on composite score  ({score_b:.1f} / {max_score:.0f} pts)'
    else:
        verdict = f'★ Tie  ({score_a:.1f} / {max_score:.0f} pts each)'
    lines += [f'  {verdict}', NL]

    # Event detection summary
    ev_min = metrics_a.get('event_min_windows', EVENT_MIN_WINDOWS)
    edr_a  = metrics_a['overall']['event_detection_rate']
    edr_b  = metrics_b['overall']['event_detection_rate']
    lines += [
        f'● Event Detection  (≥{ev_min} correct windows per clip = detected)',
        f'  {SEP}',
    ]
    for i, cls in enumerate(classes):
        pa, pb = per_a[i], per_b[i]
        if pa['n_windows'] == 0 and pb['n_windows'] == 0:
            continue
        da = '✓' if pa.get('event_detected', False) else '✗'
        db = '✓' if pb.get('event_detected', False) else '✗'
        nc_a, nc_b = pa.get('n_correct', 0), pb.get('n_correct', 0)
        lines.append(
            f'  {cls:<10}  A={da} ({nc_a:3d}/{pa["n_windows"]:3d})  '
            f'B={db} ({nc_b:3d}/{pb["n_windows"]:3d})'
        )
    lines.append(f'  Detection rate → A={edr_a:.0%}  B={edr_b:.0%}  '
                 f'{_winner_str(edr_a, edr_b, name_a, name_b)}')
    lines.append(NL)

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

    # 加權 Borda 計分明細
    lines += ['● Composite Score  (Weighted Borda Count)', f'  {SEP}']
    lines.append(f'  {"Metric":<26}  W  {"A":>7}  {"B":>7}  {"A pts":>6}  {"B pts":>6}  Result')
    for m in score_breakdown:
        dir_hint = '↑' if m['higher_better'] else '↓'
        lines.append(
            f'  {m["label"]:<26}  {m["weight"]}  '
            f'{m["va"]:>7.3f}  {m["vb"]:>7.3f}  '
            f'{m["pts_a"]:>6.1f}  {m["pts_b"]:>6.1f}  '
            f'{dir_hint} {m["result"]}'
        )
        lines.append(f'    └ {m["note"]}')
    lines += [f'  {SEP}',
              f'  {"TOTAL":<26}     {"":>7}  {"":>7}  {score_a:>6.1f}  {score_b:>6.1f}',
              NL]

    # Recommendation
    lines.append('● Recommendation')
    if score_a > score_b:
        lines.append(f'  {name_a} is the stronger model  ({score_a:.1f} vs {score_b:.1f} / {max_score:.0f} pts).')
    elif score_b > score_a:
        lines.append(f'  {name_b} is the stronger model  ({score_b:.1f} vs {score_a:.1f} / {max_score:.0f} pts).')
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
