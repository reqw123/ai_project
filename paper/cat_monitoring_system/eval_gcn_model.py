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
    2: 'xy',
    3: 'xy_conf',
    5: 'xy_conf_v',
    7: 'xy_conf_v_bone',
    9: 'xy_conf_v_bone_bmotion',
}

# ── Default / hardcoded paths ─────────────────────────────────────────────
DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\v11s_117.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 2
EVENT_MIN_WINDOWS = 3   # 保留作備用下限；實際以比例門檻為主
EVENT_MIN_RATIO   = 0.30  # 事件偵測（比例門檻）：正確 window 數 / 總 window 數 ≥ 此值即視為偵測成功
PROB_EVENT_THRESHOLD = 0.40  # 機率門檻型事件偵測：true-class prob ≥ 此值的 window 達 EVENT_MIN_WINDOWS 即算偵測
DEFAULT_DEVICE  = 'cuda'

HARD_MODEL_A    = r"C:\Users\homec\Downloads\stgcn_results\run_087_xy_conf_v_bone_att_on\087_best_model.pth"
HARD_MODEL_B    = r"C:\Users\homec\Downloads\stgcn_results\run_093_xy_conf_v_bone_att_on\093_best_model.pth"
HARD_NAME_A     = None   # None → auto-derived from filename
HARD_NAME_B     = None
HARD_EMA_ALPHA_A = 1.0   # 1.0 = 不平滑；< 1.0 = EMA 平滑（例如 0.5）
HARD_EMA_ALPHA_B = 1.0
HARD_SEQ_LEN_A   = 16    # 模型 A 訓練時使用的序列長度（影響 ring buffer 大小）
HARD_SEQ_LEN_B   = 16    # 模型 B 訓練時使用的序列長度；seqlen 消融時改為 32
HARD_VIDEO_WALK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk"
HARD_VIDEO_LICK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick"
HARD_VIDEO_SCRATCH_DIR = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch"
HARD_VIDEO_SHAKE_DIR   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake"
HARD_VIDEO_STOP_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\stop"
HARD_OUTPUT_DIR        = r"C:\paper\cat_monitoring_system\eval_results"

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
def _apply_ema(preds: list, alpha: float) -> list:
    """
    對 preds 序列的 probs 套用 EMA 平滑。
    alpha=1.0 → 不平滑（原始值）；alpha=0.5 → 半衰期平滑。
    pred/conf 會從平滑後的 probs 重新計算。
    """
    if alpha >= 1.0 or not preds:
        return preds
    smoothed = []
    ema = None
    for p in preds:
        raw = np.array(p['probs'], dtype=np.float32)
        ema = raw if ema is None else alpha * raw + (1.0 - alpha) * ema
        new_pred = int(np.argmax(ema))
        smoothed.append({
            **p,
            'probs': ema.tolist(),
            'pred':  new_pred,
            'conf':  float(ema[new_pred]),
        })
    return smoothed


def evaluate_video(video_path, kp_detector, classifier, feature_mode,
                   sequence_length=16, classify_stride=2, ema_alpha=1.0):
    """
    對單一影片執行逐幀推論。
    ema_alpha: EMA 平滑係數，1.0 = 不平滑，0.5 = 半衰期平滑。
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

        _model_joints = getattr(classifier.model, 'num_joints', 17)
        if _model_joints < 17:
            kpts_arr = kpts_arr[:, :_model_joints, :]
            conf_arr = conf_arr[:, :_model_joints]

        seq = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        feats = build_feature_tensor(seq, conf_arr, feature_mode)

        pred_id, pred_conf, pred_probs = classifier.model.predict(feats, precomputed=True)
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
    return _apply_ema(preds, ema_alpha)


_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg', '.webm'}


def evaluate_folder(folder_path, kp_detector, classifier, feature_mode,
                    sequence_length=16, classify_stride=2, ema_alpha=1.0):
    """
    對資料夾內所有影片執行推論。
    回傳 list of (filename, preds_list)，每部影片一個元素。
    """
    folder = Path(folder_path)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        print(f"    ⚠ No videos found in: {folder}")
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
def compute_metrics(preds_by_class: dict,
                    event_min_windows: int = EVENT_MIN_WINDOWS,
                    event_min_ratio: float = EVENT_MIN_RATIO,
                    prob_event_threshold: float = PROB_EVENT_THRESHOLD) -> dict:
    """
    preds_by_class: {class_idx: [[pred_dict, ...], ...]}
      外層 list = 每部影片；內層 list = 該影片的每個 window。

    Per-class metrics（window 層級指標跨所有影片合併計算）：
      accuracy, top2_accuracy, avg_true_prob, max_true_prob
    Per-class 事件偵測（逐影片判斷，回傳偵測率）：
      event_rate          — 比例門檻：各影片獨立判斷後取平均（偵測成功影片數 / 總影片數）
      prob_event_rate     — 機率門檻：同上
      n_videos, n_videos_detected, n_videos_prob_detected
    """
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class = {i: {'accuracy': 0.0, 'top2_accuracy': 0.0,
                     'avg_true_prob': 0.0, 'max_true_prob': 0.0,
                     'event_detected': False, 'prob_event_detected': False,
                     'event_rate': 0.0, 'prob_event_rate': 0.0,
                     'n_correct': 0, 'n_prob_hit': 0, 'n_windows': 0,
                     'n_videos': 0, 'n_videos_detected': 0, 'n_videos_prob_detected': 0}
                 for i in range(n_cls)}

    for cls_idx, vid_preds_list in preds_by_class.items():
        # 合併所有影片 windows 用於 window 層級指標
        preds = [p for vid_preds in vid_preds_list for p in vid_preds]
        if not preds:
            continue

        probs      = np.array([p['probs'] for p in preds])
        actual_cls = probs.shape[1]
        pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int),
                             0, actual_cls - 1)
        probs_cls  = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(len(preds))
        n_correct  = int((pred_ids == cls_idx).sum())
        # NOTE (shake / scratch): Discrete accuracy and avg true-class probability are
        # artificially deflated for impulsive behaviors such as "shake" (head shake),
        # which typically lasts only 0.5–1 s and accounts for roughly 1/10 of the
        # clip duration; the remaining ~90 % of windows contain "stop" and are counted
        # as misclassifications even though the model predicts them correctly.
        # Use max_true_prob and event_detected as the primary validity metrics for
        # these classes — they reflect whether the model fires during the event peak,
        # regardless of how many non-event windows surround it.

        top2_ids = np.argsort(probs, axis=1)[:, -2:]
        n_top2   = int(np.any(top2_ids == cls_idx, axis=1).sum())

        # 逐影片事件偵測
        n_videos = len(vid_preds_list)
        n_vid_detected = 0
        n_vid_prob_detected = 0
        for vid_preds in vid_preds_list:
            if not vid_preds:
                continue
            vp = np.clip(np.array([p['pred'] for p in vid_preds], dtype=int), 0, actual_cls - 1)
            vc = int((vp == cls_idx).sum())
            vn = len(vid_preds)
            if (vc / vn) >= event_min_ratio or vc >= event_min_windows:
                n_vid_detected += 1
            vpc = np.array([p['probs'][cls_idx] for p in vid_preds if cls_idx < len(p['probs'])])
            if len(vpc) > 0 and int((vpc >= prob_event_threshold).sum()) >= event_min_windows:
                n_vid_prob_detected += 1

        event_rate      = n_vid_detected / n_videos if n_videos > 0 else 0.0
        prob_event_rate = n_vid_prob_detected / n_videos if n_videos > 0 else 0.0
        n_prob_hit      = int((probs_cls >= prob_event_threshold).sum())

        per_class[cls_idx] = {
            'accuracy':               float(n_correct / len(preds)),
            'top2_accuracy':          float(n_top2 / len(preds)),
            'avg_true_prob':          float(probs_cls.mean()),
            'max_true_prob':          float(probs_cls.max()),
            'event_detected':         n_vid_detected > 0,
            'prob_event_detected':    n_vid_prob_detected > 0,
            'event_rate':             float(event_rate),
            'prob_event_rate':        float(prob_event_rate),
            'n_correct':              n_correct,
            'n_prob_hit':             n_prob_hit,
            'n_windows':              len(preds),
            'n_videos':               n_videos,
            'n_videos_detected':      n_vid_detected,
            'n_videos_prob_detected': n_vid_prob_detected,
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    evaluated       = [i for i in range(n_cls) if per_class[i]['n_windows'] > 0]
    event_rate      = float(np.mean([per_class[i]['event_rate']      for i in evaluated])) if evaluated else 0.0
    prob_event_rate = float(np.mean([per_class[i]['prob_event_rate'] for i in evaluated])) if evaluated else 0.0
    top2_acc        = float(np.mean([per_class[i]['top2_accuracy']   for i in evaluated])) if evaluated else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'accuracy':                 float(accuracy_score(y_true, y_pred)),
            'top2_accuracy':            top2_acc,
            'macro_f1':                 float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'event_detection_rate':     event_rate,
            'prob_event_detection_rate': prob_event_rate,
        },
        'confusion_matrix':      confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
        'event_min_windows':     event_min_windows,
        'event_min_ratio':       event_min_ratio,
        'prob_event_threshold':  prob_event_threshold,
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


def plot_prob_histograms(preds_a, preds_b, name_a, name_b, classes, out_path):
    """
    ⑦ True-class probability histogram（每個類別一列，A/B 並排）
    橫軸：true-class prob 0~1，縱軸：window 數量。
    可立即看出 Softmax collapse（機率全堆在同一處）或分布健康與否。
    """
    n_cls = len(classes)
    fig, axes = plt.subplots(n_cls, 2, figsize=(11, 2.8 * n_cls), constrained_layout=True)
    fig.suptitle('True-Class Probability Histogram\n(each row = one behavior class)',
                 fontsize=12, fontweight='bold')
    if n_cls == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)
    for i, cls in enumerate(classes):
        for j, (preds, name, col) in enumerate(
            [(preds_a, name_a, _COL_A), (preds_b, name_b, _COL_B)]
        ):
            ax = axes[i][j]
            if i not in preds or not preds[i]:
                ax.set_visible(False)
                continue
            probs_cls = np.array([p['probs'][i] for p in preds[i]
                                  if i < len(p['probs'])])
            ax.hist(probs_cls, bins=bins, color=col, alpha=0.85, edgecolor='white')
            ax.axvline(float(probs_cls.mean()), color='k', linestyle='--',
                       linewidth=1.2, label=f'mean={probs_cls.mean():.2f}')
            ax.set_xlim(0, 1)
            ax.set_title(f'[{cls}]  {name}', fontsize=9, fontweight='bold')
            ax.set_xlabel('True-class probability', fontsize=8)
            ax.set_ylabel('Windows', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV outputs
# ═══════════════════════════════════════════════════════════════════════════
def save_summary_csv(metrics_a, metrics_b, name_a, name_b, classes, out_path):
    """對比摘要 CSV：每行一個指標，含贏家欄位。primary 欄位標示指定的主指標
    （event_detection_rate，比例門檻事件偵測率），其餘指標僅供參考。"""
    rows = [['metric', name_a, name_b, 'winner', 'delta', 'primary']]

    def _row(label, va, vb, primary=False):
        delta = abs(va - vb)
        winner = '=' if delta < 0.001 else (name_a if va >= vb else name_b)
        rows.append([label, f'{va:.4f}', f'{vb:.4f}', winner, f'{delta:.4f}', 'YES' if primary else ''])

    _row('overall_event_det_rate_ratio',      metrics_a['overall']['event_detection_rate'],      metrics_b['overall']['event_detection_rate'], primary=True)
    _row('overall_accuracy',                  metrics_a['overall']['accuracy'],                  metrics_b['overall']['accuracy'])
    _row('overall_top2_accuracy',             metrics_a['overall']['top2_accuracy'],             metrics_b['overall']['top2_accuracy'])
    _row('overall_macro_f1',                  metrics_a['overall']['macro_f1'],                  metrics_b['overall']['macro_f1'])
    _row('overall_event_det_rate_prob',       metrics_a['overall']['prob_event_detection_rate'], metrics_b['overall']['prob_event_detection_rate'])
    rows.append([])   # blank separator

    for i, cls in enumerate(classes):
        pa = metrics_a['per_class'][i]
        pb = metrics_b['per_class'][i]
        _row(f'{cls}_event_rate',         pa['event_rate'],      pb['event_rate'], primary=True)
        _row(f'{cls}_accuracy',           pa['accuracy'],      pb['accuracy'])
        _row(f'{cls}_top2_accuracy',      pa['top2_accuracy'], pb['top2_accuracy'])
        _row(f'{cls}_avg_true_prob',      pa['avg_true_prob'], pb['avg_true_prob'])
        _row(f'{cls}_max_true_prob',      pa['max_true_prob'], pb['max_true_prob'])
        _row(f'{cls}_prob_event_rate',    pa['prob_event_rate'], pb['prob_event_rate'])
        rows.append([f'{cls}_n_videos_detected',
                     f"{pa['n_videos_detected']}/{pa['n_videos']}",
                     f"{pb['n_videos_detected']}/{pb['n_videos']}", '-', '-'])
        rows.append([f'{cls}_prob_videos_detected',
                     f"{pa['n_videos_prob_detected']}/{pa['n_videos']}",
                     f"{pb['n_videos_prob_detected']}/{pb['n_videos']}", '-', '-'])
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

    # ── 主指標：Event Detection Rate（你指定在意的偵測率，獨立於下方複合分數）──
    ev_min = metrics_a.get('event_min_windows', EVENT_MIN_WINDOWS)
    edr_a  = metrics_a['overall']['event_detection_rate']
    edr_b  = metrics_b['overall']['event_detection_rate']
    primary_diff   = abs(edr_a - edr_b)
    primary_winner = ('=' if primary_diff < 0.005
                       else (name_a if edr_a > edr_b else name_b))

    lines = [
        NL,
        f'╔{BOX}╗',
        f'║{"  FINAL COMPARISON SUMMARY":^62}║',
        f'╚{BOX}╝',
        NL,
        f'★★★ 主指標：Event Detection Rate  (ratio≥{EVENT_MIN_RATIO:.0%} or ≥{ev_min} windows) ★★★',
        f'  {"Class":<10}  {"A rate":>8}  {"A clips":>9}  {"B rate":>8}  {"B clips":>9}  Winner',
        f'  {SEP}',
    ]
    for i, cls in enumerate(classes):
        pa, pb = per_a[i], per_b[i]
        if pa['n_windows'] == 0 and pb['n_windows'] == 0:
            continue
        ra, rb = pa['event_rate'], pb['event_rate']
        da = f"{pa['n_videos_detected']}/{pa['n_videos']}"
        db = f"{pb['n_videos_detected']}/{pb['n_videos']}"
        d  = abs(ra - rb)
        w  = '=' if d < 0.005 else (name_a if ra > rb else name_b)
        lines.append(f'  {cls:<10}  {ra:>8.0%}  {da:>9}  {rb:>8.0%}  {db:>9}  {w}')
    lines += [
        f'  {SEP}',
        f'  {"Overall":<10}  {edr_a:>8.0%}  {"":>9}  {edr_b:>8.0%}  {"":>9}  {primary_winner}',
        NL,
    ]
    if primary_winner == '=':
        lines.append(f'  ★ 主指標結論：兩模型偵測率相近（Δ{primary_diff:.1%}）')
    else:
        lines.append(f'  ★ 主指標結論：{primary_winner} 偵測率較高（Δ{primary_diff:.1%}）'
                     f'— 依你指定的主指標，優先選它')
    lines += [
        NL,
        '● [參考，非主指標] Overall Performance  (Accuracy / Macro F1)',
        f'  {"Metric":<14}  {"Model A":>14}  {"Model B":>14}  Result',
        f'  {SEP}',
        f'  {"Accuracy":<14}  {oa:>14.1%}  {ob:>14.1%}  {_winner_str(oa, ob, name_a, name_b)}',
        f'  {"Macro F1":<14}  {fa:>14.1%}  {fb:>14.1%}  {_winner_str(fa, fb, name_a, name_b)}',
        NL,
    ]

    # [參考] 加權 Borda 綜合計分——混入了 F1/Min-class Acc 等你不特別在意的指標，非主指標
    score_a, score_b, score_breakdown = _score_comparison(metrics_a, metrics_b, name_a, name_b)
    max_score = sum(m['weight'] for m in score_breakdown)
    if score_a > score_b:
        composite_winner = name_a
        verdict = f'[參考] {name_a} leads on composite score  ({score_a:.1f} / {max_score:.0f} pts)'
    elif score_b > score_a:
        composite_winner = name_b
        verdict = f'[參考] {name_b} leads on composite score  ({score_b:.1f} / {max_score:.0f} pts)'
    else:
        composite_winner = '='
        verdict = f'[參考] Tie  ({score_a:.1f} / {max_score:.0f} pts each)'
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

    # 加權 Borda 計分明細
    lines += ['● [參考] Composite Score  (Weighted Borda Count — 非主指標)', f'  {SEP}']
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

    # Recommendation — 以你指定的主指標（Event Detection Rate）為準，複合分數僅供參考
    lines.append('● Recommendation')
    if primary_winner == '=':
        lines.append(f'  主指標（偵測率）打平；請改看上方 [參考] 複合分數或逐類別表現決定。')
    else:
        lines.append(f'  依主指標（Event Detection Rate），{primary_winner} 較適合部署'
                     f'（Δ{primary_diff:.1%}）。')
    if biggest_gap >= 0.10:
        lines.append(f'  Note: [{biggest_cls}] shows a large gap (Δ{biggest_gap:.1%}); '
                     f'prioritize {gap_winner} if this class is critical.')
    if primary_winner != '=' and composite_winner != '=' and primary_winner != composite_winner:
        lines.append(f'  ⚠ 注意：[參考] 複合分數建議 {composite_winner}，但主指標（偵測率）建議 {primary_winner}'
                     f'——兩者不一致時請以主指標為準（複合分數混入了 F1/Accuracy 等你不特別在意的項目）。')
    lines.append(NL)

    print('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Compare two ST-GCN models on behavior video folders.'
    )
    parser.add_argument('--model_a',             default=HARD_MODEL_A)
    parser.add_argument('--model_b',             default=HARD_MODEL_B)
    parser.add_argument('--name_a',              default=HARD_NAME_A)
    parser.add_argument('--name_b',              default=HARD_NAME_B)
    parser.add_argument('--video_walk_dir',      default=HARD_VIDEO_WALK_DIR)
    parser.add_argument('--video_lick_dir',      default=HARD_VIDEO_LICK_DIR)
    parser.add_argument('--video_scratch_dir',   default=HARD_VIDEO_SCRATCH_DIR)
    parser.add_argument('--video_shake_dir',     default=HARD_VIDEO_SHAKE_DIR)
    parser.add_argument('--video_stop_dir',      default=HARD_VIDEO_STOP_DIR,
                        help='stop 類別資料夾（留空則跳過）')
    parser.add_argument('--yolo',                default=DEFAULT_YOLO)
    parser.add_argument('--imgsz',               type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument('--conf',                type=float, default=DEFAULT_CONF)
    parser.add_argument('--output',              default=HARD_OUTPUT_DIR)
    parser.add_argument('--sequence_length',     type=int,   default=DEFAULT_SEQ_LEN,
                        help='共用序列長度（當兩模型相同時使用）')
    parser.add_argument('--seq_len_a',           type=int,   default=None,
                        help='模型 A 的序列長度（覆蓋 --sequence_length；None → 讀 HARD_SEQ_LEN_A）')
    parser.add_argument('--seq_len_b',           type=int,   default=None,
                        help='模型 B 的序列長度（覆蓋 --sequence_length；None → 讀 HARD_SEQ_LEN_B）')
    parser.add_argument('--classify_stride',     type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--device',              default=DEFAULT_DEVICE)
    args = parser.parse_args()

    # EMA alpha 直接從硬編碼常數讀取
    ema_a = HARD_EMA_ALPHA_A
    ema_b = HARD_EMA_ALPHA_B

    # 序列長度：CLI 明確指定 > 硬編碼常數 > --sequence_length 全域值
    seq_len_a = args.seq_len_a if args.seq_len_a is not None else HARD_SEQ_LEN_A
    seq_len_b = args.seq_len_b if args.seq_len_b is not None else HARD_SEQ_LEN_B

    name_a = args.name_a or _short_name(args.model_a)
    name_b = args.name_b or _short_name(args.model_b)

    # 若 alpha 不同，名稱後附加 EMA 標記，讓圖表/CSV 一眼看出差異
    label_a = f"{name_a}[ema={ema_a}]" if ema_a < 1.0 else name_a
    label_b = f"{name_b}[ema={ema_b}]" if ema_b < 1.0 else name_b
    # 序列長度不同時也附加 T 標記
    if seq_len_a != seq_len_b:
        label_a = f"{label_a}[T{seq_len_a}]"
        label_b = f"{label_b}[T{seq_len_b}]"

    # Sequential output directory
    out_root = Path(args.output)
    run_num  = _next_comparison_number(str(out_root))
    run_tag  = f"{run_num:03d}"
    out_dir  = out_root / f"comparison_{run_tag}_{label_a}_vs_{label_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  Comparison #{run_tag}")
    print(f"  Model A : {label_a}  (EMA α={ema_a}  T={seq_len_a})")
    print(f"  Model B : {label_b}  (EMA α={ema_b}  T={seq_len_b})")
    print(f"  Output  : {out_dir}")
    print(f"{sep}")

    # Load detectors
    print('\n[Loading YOLO]')
    kp_det = KeypointDetector(args.yolo, device=args.device,
                               imgsz=args.imgsz, conf_thres=args.conf)

    def _load_classifier(model_path, device, seq_len):
        bn_ch = infer_bn_input_channels(model_path)
        fm    = CH_TO_FEATURE.get(bn_ch, 'xy')
        if bn_ch not in CH_TO_FEATURE:
            print(f"  ⚠ {Path(model_path).name}: cannot infer feature_mode "
                  f"(bn_ch={bn_ch}), defaulting to 'xy'")
        else:
            print(f"  ✓ {Path(model_path).name} → {fm} ({bn_ch} ch)  T={seq_len}")
        clf = BehaviorClassifier(
            model_path, device=device,
            sequence_length=seq_len,
            normalize=True, feature_mode=fm, in_channels=bn_ch,
        )
        return clf, fm

    print('\n[Loading ST-GCN models]')
    clf_a, fm_a = _load_classifier(args.model_a, args.device, seq_len_a)
    clf_b, fm_b = _load_classifier(args.model_b, args.device, seq_len_b)

    _all_dirs = [
        (args.video_walk_dir,    0),
        (args.video_lick_dir,    1),
        (args.video_scratch_dir, 2),
        (args.video_shake_dir,   3),
        (args.video_stop_dir,    4),
    ]
    # 資料夾不存在或未設定時跳過
    dirs = [(p, idx) for p, idx in _all_dirs if p and Path(p).is_dir()]
    skipped = [(BEHAVIOR_CLASSES[idx], p) for p, idx in _all_dirs
               if not p or not Path(p).is_dir()]
    if skipped:
        for cls_name, p in skipped:
            reason = '（路徑未設定）' if not p else f'（找不到資料夾: {p}）'
            print(f"  ⚠ 跳過 [{cls_name}] {reason}")

    # ── Inference ──────────────────────────────────────────────────────────
    # preds_*[cls_idx] = [[pred_dict, ...], ...]  （外層 = 每部影片）
    preds_a: dict = {}
    preds_b: dict = {}

    for dir_path, cls_idx in dirs:
        cls_name = BEHAVIOR_CLASSES[cls_idx]
        print(f"\n[{cls_name.upper()}]  {Path(dir_path).name}/")

        print(f"  ▶ A ({label_a})")
        preds_a[cls_idx] = [p for _, p in evaluate_folder(
            dir_path, kp_det, clf_a, fm_a,
            seq_len_a, args.classify_stride, ema_alpha=ema_a
        )]

        print(f"  ▶ B ({label_b})")
        preds_b[cls_idx] = [p for _, p in evaluate_folder(
            dir_path, kp_det, clf_b, fm_b,
            seq_len_b, args.classify_stride, ema_alpha=ema_b
        )]

        na = sum(len(v) for v in preds_a[cls_idx])
        nb = sum(len(v) for v in preds_b[cls_idx])
        nv = len(preds_a[cls_idx])
        print(f"  → {nv} videos  |  A={na} windows  B={nb} windows")

    # ── Compute metrics ─────────────────────────────────────────────────────
    metrics_a = compute_metrics(preds_a)
    metrics_b = compute_metrics(preds_b)

    # ── Console summary ─────────────────────────────────────────────────────
    col_w = max(len(label_a), len(label_b), 10)
    hdr = f"  {'Metric':<30}  {label_a:>{col_w}}  {label_b:>{col_w}}  Winner"
    print(f"\n{sep}\n{hdr}\n  {'-' * (len(hdr) - 2)}")

    def _row(label, va, vb):
        diff = abs(va - vb)
        winner = '=' if diff < 0.001 else (f'+A Δ{diff:.1%}' if va > vb else f'+B Δ{diff:.1%}')
        print(f"  {label:<30}  {va:>{col_w}.4f}  {vb:>{col_w}.4f}  {winner}")

    _row('Overall Accuracy',           metrics_a['overall']['accuracy'],                 metrics_b['overall']['accuracy'])
    _row('Overall Top-2 Accuracy',     metrics_a['overall']['top2_accuracy'],            metrics_b['overall']['top2_accuracy'])
    _row('Overall Macro-F1',           metrics_a['overall']['macro_f1'],                 metrics_b['overall']['macro_f1'])
    _row('Event Det. Rate (ratio)',    metrics_a['overall']['event_detection_rate'],      metrics_b['overall']['event_detection_rate'])
    _row('Event Det. Rate (prob thr)', metrics_a['overall']['prob_event_detection_rate'], metrics_b['overall']['prob_event_detection_rate'])
    print()
    evaluated_cls = {idx for _, idx in dirs}
    for i, cls in enumerate(BEHAVIOR_CLASSES):
        if i not in evaluated_cls:
            continue
        pa, pb = metrics_a['per_class'][i], metrics_b['per_class'][i]
        _row(f'{cls:<8} accuracy',      pa['accuracy'],      pb['accuracy'])
        _row(f'{cls:<8} top2_accuracy', pa['top2_accuracy'], pb['top2_accuracy'])
        _row(f'{cls:<8} avg_true_prob', pa['avg_true_prob'], pb['avg_true_prob'])
        _row(f'{cls:<8} event_rate',    pa['event_rate'],    pb['event_rate'])
        print(f"  {cls:<8} clips detected  "
              f"A={pa['n_videos_detected']}/{pa['n_videos']}  "
              f"B={pb['n_videos_detected']}/{pb['n_videos']}  "
              f"(prob≥{PROB_EVENT_THRESHOLD:.2f}: "
              f"A={pa['n_videos_prob_detected']}/{pa['n_videos']}  "
              f"B={pb['n_videos_prob_detected']}/{pb['n_videos']})")

    print(sep)

    # ── Save outputs ─────────────────────────────────────────────────────────
    # plot_prob_histograms / save_preds_csv 需要 flat list，先在此展開
    flat_a = {cls_idx: [p for vid in vids for p in vid]
              for cls_idx, vids in preds_a.items()}
    flat_b = {cls_idx: [p for vid in vids for p in vid]
              for cls_idx, vids in preds_b.items()}

    print('\n[Saving]')
    save_preds_csv(flat_a, label_a, BEHAVIOR_CLASSES, out_dir)
    save_preds_csv(flat_b, label_b, BEHAVIOR_CLASSES, out_dir)
    save_summary_csv(
        metrics_a, metrics_b, label_a, label_b,
        BEHAVIOR_CLASSES, out_dir / 'comparison_summary.csv'
    )
    plot_accuracy_comparison(
        metrics_a, metrics_b, label_a, label_b,
        BEHAVIOR_CLASSES, out_dir / 'accuracy_comparison.png'
    )
    plot_confusion_matrices(
        metrics_a['confusion_matrix'], metrics_b['confusion_matrix'],
        label_a, label_b, BEHAVIOR_CLASSES,
        out_dir / 'confusion_matrices.png'
    )
    plot_prob_histograms(
        flat_a, flat_b, label_a, label_b,
        BEHAVIOR_CLASSES, out_dir / 'prob_histograms.png'
    )

    print(f'\n✓ All results saved to: {out_dir}')

    print_final_summary(metrics_a, metrics_b, label_a, label_b)


if __name__ == '__main__':
    main()
