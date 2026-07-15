"""
Run ONE ST-GCN weight against all five behavior video folders and identify
which individual videos the model performs worst on.

Unlike eval_model_four_videos.py (A vs B model comparison), this script
evaluates a single model and ranks per-video accuracy within/across class
folders, explicitly calling out the worst-performing video(s) so they can
be inspected or re-labeled.

Missing / unreadable class folders are skipped with a warning; at least
one folder must be found or the script aborts (RuntimeError). Fully
headless (matplotlib Agg, no display window) — safe to run in the
background.

Output directory: <output>/single_eval_NNN_<name>/
  video_breakdown.csv     — every video's accuracy/confidence/event-detected, worst first
  per_class_accuracy.png  — per class, videos sorted worst → best (red = below threshold)
  confusion_matrix.png
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
# 影片檔名含中文（例如「7月3日(1).mp4」），預設字型無中文字形會變成方框，
# 這裡指定系統已安裝的中文字型，讓 per_class_accuracy.png 的 Y 軸檔名正常顯示。
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'MingLiU', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

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

# ── Channel → feature mode ────────────────────────────────────────────────
CH_TO_FEATURE = {
    2: 'xy',
    3: 'xy_conf',
    5: 'xy_conf_v',
    7: 'xy_conf_v_bone',
    9: 'xy_conf_v_bone_bmotion',
}

# ── Default / hardcoded paths ─────────────────────────────────────────────
DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\v11s_114.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 2
EVENT_MIN_WINDOWS = 3     # 保留作備用下限；實際以比例門檻為主
EVENT_MIN_RATIO   = 0.30  # 事件偵測（比例門檻）：正確 window 數 / 總 window 數 ≥ 此值即視為偵測成功
DEFAULT_DEVICE  = 'cuda'

HARD_MODEL_PATH = r"C:\Users\homec\Downloads\stgcn_results\run_081_models_att_on\081_xy_conf_v_bone_att_on.pth"
HARD_NAME       = None   # None → auto-derived from filename
HARD_EMA_ALPHA  = 1.0    # 1.0 = 不平滑；< 1.0 = EMA 平滑（例如 0.5），須與訓練時 KP_EMA_ALPHA 一致
HARD_SEQ_LEN    = 16     # 模型訓練時使用的序列長度（影響 ring buffer 大小）
HARD_VIDEO_WALK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk"
HARD_VIDEO_LICK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick"
HARD_VIDEO_SCRATCH_DIR = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch"
HARD_VIDEO_SHAKE_DIR   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake"
HARD_VIDEO_STOP_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\stop"
HARD_OUTPUT_DIR        = r"C:\paper\cat_monitoring_system\eval_results"

POOR_ACC_THRESHOLD       = 0.50  # 低於此準確率的影片在圖表/摘要中標記為表現不佳
POOR_KPT_CONF_THRESHOLD  = 0.50  # 平均關鍵點信心低於此值，懷疑是 YOLO 偵測品質問題而非分類器問題
POOR_CAT_RATIO_THRESHOLD = 0.70  # 貓咪偵測到的幀數比例低於此值，同上
WORST_LIST_TOP_N    = 10   # 全域最差影片列表最多列出幾支

_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg', '.webm'}
_COL_OK   = '#4CAF50'
_COL_POOR = '#E53935'


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def _next_run_number(out_root: str) -> int:
    """掃描 out_root/single_eval_NNN_* 目錄，回傳下一個可用編號。"""
    p = Path(out_root)
    if not p.exists():
        return 1
    pat = re.compile(r'^single_eval_(\d+)')
    max_num = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def _short_name(model_path: str) -> str:
    """從模型路徑提取簡短名稱，優先取 run number + mode 部分。"""
    stem = Path(model_path).stem
    parts = stem.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) >= 3:
            return '_'.join(parts[i:])
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
    """對 preds 序列的 probs 套用 EMA 平滑；alpha=1.0 = 不平滑。"""
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

    回傳 (preds, detection_stats)：
      preds:            list of dicts: {frame, time, pred, conf, probs}
      detection_stats:  {total_frames, frames_with_cat, frames_without_cat,
                         cat_detected_ratio, mean_kpt_conf}
                         — 用於區分「YOLO 沒偵測到骨架」與「ST-GCN 判斷錯誤」兩種失敗原因
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kp_detector.reset_track()  # 新影片開始，避免延續上一支影片鎖定的貓

    buf = deque(maxlen=sequence_length)
    preds = []
    frame_idx = -1
    total_frames = 0
    frames_with_cat = 0
    kpt_conf_sum = 0.0
    kpt_conf_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        total_frames += 1
        kpts, kpt_conf, _, _ = kp_detector.detect(frame)
        buf.append((kpts, kpt_conf) if kpts is not None else (None, None))

        if kpts is not None:
            frames_with_cat += 1
            if kpt_conf is not None and len(kpt_conf) > 0:
                kpt_conf_sum += float(np.mean(kpt_conf))
                kpt_conf_frame_count += 1

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
    detection_stats = {
        'total_frames':       total_frames,
        'frames_with_cat':    frames_with_cat,
        'frames_without_cat': total_frames - frames_with_cat,
        'cat_detected_ratio': (frames_with_cat / total_frames) if total_frames > 0 else 0.0,
        'mean_kpt_conf':      (kpt_conf_sum / kpt_conf_frame_count) if kpt_conf_frame_count > 0 else 0.0,
    }
    return _apply_ema(preds, ema_alpha), detection_stats


def evaluate_folder(folder_path, kp_detector, classifier, feature_mode,
                    sequence_length=16, classify_stride=2, ema_alpha=1.0):
    """對資料夾內所有影片執行推論，回傳 list of (filename, preds_list, detection_stats)。"""
    folder = Path(folder_path)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        print(f"    ⚠ No videos found in: {folder}")
        return []
    results = []
    for vid in videos:
        print(f"    {vid.name} ...", end=' ', flush=True)
        preds, det_stats = evaluate_video(vid, kp_detector, classifier, feature_mode,
                                          sequence_length, classify_stride, ema_alpha)
        print(f"{len(preds)} windows  "
              f"(cat detected {det_stats['frames_with_cat']}/{det_stats['total_frames']} frames, "
              f"mean_kpt_conf={det_stats['mean_kpt_conf']:.2f})")
        results.append((vid.name, preds, det_stats))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_video_metrics(video_name, cls_idx, preds, detection_stats=None,
                          event_min_windows=EVENT_MIN_WINDOWS,
                          event_min_ratio=EVENT_MIN_RATIO):
    """單一影片的準確率／信心／事件偵測／偵測品質指標（true label = 該影片所在的類別資料夾）。"""
    det = detection_stats or {}
    det_fields = {
        'total_frames':       det.get('total_frames', 0),
        'frames_with_cat':    det.get('frames_with_cat', 0),
        'frames_without_cat': det.get('frames_without_cat', 0),
        'cat_detected_ratio': det.get('cat_detected_ratio', 0.0),
        'mean_kpt_conf':      det.get('mean_kpt_conf', 0.0),
    }
    n_windows = len(preds)
    if n_windows == 0:
        return {
            'video': video_name, 'class': BEHAVIOR_CLASSES[cls_idx], 'class_idx': cls_idx,
            'n_windows': 0, 'n_correct': 0, 'accuracy': 0.0,
            'avg_true_prob': 0.0, 'max_true_prob': 0.0, 'event_detected': False,
            **det_fields,
        }
    probs      = np.array([p['probs'] for p in preds])
    actual_cls = probs.shape[1]
    pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int), 0, actual_cls - 1)
    probs_cls  = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(n_windows)
    n_correct  = int((pred_ids == cls_idx).sum())
    accuracy   = n_correct / n_windows
    event_detected = (accuracy >= event_min_ratio) or (n_correct >= event_min_windows)
    return {
        'video': video_name, 'class': BEHAVIOR_CLASSES[cls_idx], 'class_idx': cls_idx,
        'n_windows': n_windows, 'n_correct': n_correct,
        'accuracy': float(accuracy),
        'avg_true_prob': float(probs_cls.mean()),
        'max_true_prob': float(probs_cls.max()),
        'event_detected': bool(event_detected),
        **det_fields,
    }


def compute_overall_metrics(preds_by_class: dict):
    """跨影片彙整成 overall accuracy / macro-F1 / per-class accuracy / confusion matrix。"""
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class_correct = {i: 0 for i in range(n_cls)}
    per_class_total    = {i: 0 for i in range(n_cls)}

    for cls_idx, vid_preds_list in preds_by_class.items():
        for _, preds, _ in vid_preds_list:
            if not preds:
                continue
            actual_cls = len(preds[0]['probs'])
            pred_ids = np.clip(np.array([p['pred'] for p in preds], dtype=int), 0, actual_cls - 1)
            all_true.extend([cls_idx] * len(preds))
            all_pred.extend(pred_ids.tolist())
            per_class_total[cls_idx]   += len(preds)
            per_class_correct[cls_idx] += int((pred_ids == cls_idx).sum())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    per_class_accuracy = {
        i: (per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0)
        for i in range(n_cls)
    }
    return {
        'accuracy':  float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        'macro_f1':  float(f1_score(y_true, y_pred, average='macro', zero_division=0)) if len(y_true) else 0.0,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': (confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
                             if len(y_true) else np.zeros((n_cls, n_cls), dtype=int)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_class_accuracy(video_metrics: list, classes, out_path):
    """每個類別一個子圖，該類別的影片依準確率由低到高排序（最差在最上面）。"""
    by_class = {c: [] for c in classes}
    for vm in video_metrics:
        by_class[vm['class']].append(vm)

    present = [c for c in classes if by_class[c]]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 2.2 * len(present)), constrained_layout=True)
    if len(present) == 1:
        axes = [axes]
    fig.suptitle('Per-Video Accuracy by Class  (worst → best, top to bottom)',
                 fontsize=13, fontweight='bold')

    for ax, cls in zip(axes, present):
        rows = sorted(by_class[cls], key=lambda r: r['accuracy'])
        names = [r['video'] for r in rows]
        accs  = [r['accuracy'] for r in rows]
        colors = [_COL_POOR if a < POOR_ACC_THRESHOLD else _COL_OK for a in accs]
        y = np.arange(len(rows))
        ax.barh(y, accs, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.axvline(POOR_ACC_THRESHOLD, color='#555', linestyle='--', linewidth=1)
        ax.set_title(f'[{cls}]', fontsize=10, fontweight='bold')
        ax.set_xlabel('Accuracy', fontsize=8)
        for yi, a in zip(y, accs):
            ax.text(a + 0.01, yi, f'{a:.0%}', va='center', fontsize=7)
        ax.grid(axis='x', alpha=0.25, linestyle='--')

    plt.savefig(out_path, dpi=170, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_confusion_matrix(cm, classes, out_path):
    try:
        import seaborn as sns
        _sns = True
    except ImportError:
        _sns = False

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    if _sns:
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
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=11)

    diag_recall = [cm[i, i] / max(cm[i].sum(), 1) for i in range(len(classes))]
    ax.set_title(f'Confusion Matrix\nAvg Recall = {np.mean(diag_recall):.1%}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV / console reporting
# ═══════════════════════════════════════════════════════════════════════════
def save_video_breakdown_csv(video_metrics: list, out_path):
    """每支影片一行，依準確率由低到高排序（最差在最上面），方便直接開檔案看。

    encoding='utf-8-sig'：影片檔名常含中文（例如「7月3日(1).mp4」），
    不加 BOM 的話 Windows Excel 用預設編碼開啟會顯示亂碼；加了 BOM 才能讓
    Excel 正確辨識為 UTF-8（與專案內其他報表 CSV 的慣例一致）。
    """
    rows_sorted = sorted(video_metrics, key=lambda r: r['accuracy'])
    header = ['rank_worst_first', 'class', 'video', 'n_windows', 'n_correct',
              'accuracy', 'avg_true_prob', 'max_true_prob', 'event_detected',
              'total_frames', 'frames_with_cat', 'frames_without_cat',
              'cat_detected_ratio', 'mean_kpt_conf', 'suspected_detection_issue']
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(rows_sorted, 1):
            suspected_detection_issue = (
                r.get('mean_kpt_conf', 0.0) < POOR_KPT_CONF_THRESHOLD
                or r.get('cat_detected_ratio', 0.0) < POOR_CAT_RATIO_THRESHOLD
            )
            w.writerow([i, r['class'], r['video'], r['n_windows'], r['n_correct'],
                       f"{r['accuracy']:.4f}", f"{r['avg_true_prob']:.4f}",
                       f"{r['max_true_prob']:.4f}", r['event_detected'],
                       r.get('total_frames', 0), r.get('frames_with_cat', 0),
                       r.get('frames_without_cat', 0),
                       f"{r.get('cat_detected_ratio', 0.0):.4f}",
                       f"{r.get('mean_kpt_conf', 0.0):.4f}",
                       suspected_detection_issue])
    print(f"  ✓ {out_path.name}")


def print_worst_videos_report(video_metrics: list, overall_metrics: dict, model_name: str):
    """終端列印明確標出「哪支影片表現最差」的診斷報告。"""
    classes = BEHAVIOR_CLASSES
    BOX = '═' * 62
    SEP = '─' * 62

    rows_sorted = sorted(video_metrics, key=lambda r: r['accuracy'])
    worst = rows_sorted[0]

    lines = [
        '',
        f'╔{BOX}╗',
        f'║{"  WORST-VIDEO DIAGNOSTIC REPORT":^62}║',
        f'╚{BOX}╝',
        '',
        f'Model: {model_name}',
        '',
        '● Overall Performance',
        f'  {SEP}',
        f'  Accuracy   {overall_metrics["accuracy"]:.1%}',
        f'  Macro F1   {overall_metrics["macro_f1"]:.1%}',
        '',
        '● Per-Class Accuracy',
        f'  {SEP}',
    ]
    for i, cls in enumerate(classes):
        acc = overall_metrics['per_class_accuracy'].get(i)
        if acc is None:
            continue
        flag = '  ⚠ POOR' if acc < POOR_ACC_THRESHOLD else ''
        lines.append(f'  {cls:<10}  {acc:>6.1%}{flag}')

    def _is_poor_detection(r):
        return (r.get('mean_kpt_conf', 0.0) < POOR_KPT_CONF_THRESHOLD
                or r.get('cat_detected_ratio', 0.0) < POOR_CAT_RATIO_THRESHOLD)

    worst_det_hint = ('  ⚠ 骨架偵測品質差（可能是 YOLO 問題，非分類器問題）'
                      if _is_poor_detection(worst) else '')
    lines += ['', f'★ OVERALL WORST VIDEO: [{worst["class"]}] {worst["video"]}',
              f'   accuracy={worst["accuracy"]:.1%}  ({worst["n_correct"]}/{worst["n_windows"]} windows)  '
              f'avg_true_prob={worst["avg_true_prob"]:.1%}  event_detected={worst["event_detected"]}',
              f'   cat_detected={worst.get("cat_detected_ratio", 0.0):.1%} of frames  '
              f'mean_kpt_conf={worst.get("mean_kpt_conf", 0.0):.1%}{worst_det_hint}',
              '']

    lines += [f'● Worst-Performing Videos  (bottom {min(WORST_LIST_TOP_N, len(rows_sorted))} of {len(rows_sorted)}, ranked)',
              f'  {SEP}']
    for i, r in enumerate(rows_sorted[:WORST_LIST_TOP_N], 1):
        tag = ' 🔻' if r['accuracy'] < POOR_ACC_THRESHOLD else ''
        undetected = '' if r['event_detected'] else '  [未偵測到事件]'
        det_hint = '  [疑似偵測品質差]' if _is_poor_detection(r) else ''
        lines.append(
            f'  {i:>2}. [{r["class"]:<8}] {r["video"]:<42} '
            f'acc={r["accuracy"]:>6.1%}  conf={r["avg_true_prob"]:>6.1%}{undetected}{det_hint}{tag}'
        )
    lines.append('')

    # 逐類別最差影片（明確點名）
    lines += ['● Worst Video Per Class', f'  {SEP}']
    by_class = {}
    for r in video_metrics:
        by_class.setdefault(r['class'], []).append(r)
    for cls in classes:
        if cls not in by_class:
            continue
        w = min(by_class[cls], key=lambda r: r['accuracy'])
        det_hint = '  ⚠ 疑似偵測品質差' if _is_poor_detection(w) else ''
        lines.append(f'  {cls:<10}  → {w["video"]}  (acc={w["accuracy"]:.1%}, '
                     f'{w["n_correct"]}/{w["n_windows"]} windows){det_hint}')
    lines.append('')

    # shake/scratch 提醒：脈衝式行為 accuracy 天生偏低，需搭配 event_detected 判讀
    if any(r['class'] in ('shake', 'scratch') for r in video_metrics):
        lines += [
            '● Note',
            '  shake/scratch 屬脈衝式短暫動作，若整支影片大部分是靜止片段，',
            '  window 層級 accuracy 會被天生拉低；判斷是否「真的表現不好」時，',
            '  請同時參考 event_detected 欄位（True = 至少捕捉到一段動作事件）。',
            '  [疑似偵測品質差] 代表平均關鍵點信心 < '
            f'{POOR_KPT_CONF_THRESHOLD:.0%} 或偵測到貓的幀數比例 < {POOR_CAT_RATIO_THRESHOLD:.0%}，',
            '  這種情況低分可能來自 YOLO 骨架偵測失敗，而非 ST-GCN 分類器誤判，建議優先排查。',
            '',
        ]

    print('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a single ST-GCN model on behavior video folders '
                    'and identify the worst-performing videos.'
    )
    parser.add_argument('--model',               default=HARD_MODEL_PATH)
    parser.add_argument('--name',                default=HARD_NAME)
    parser.add_argument('--video_walk_dir',      default=HARD_VIDEO_WALK_DIR)
    parser.add_argument('--video_lick_dir',      default=HARD_VIDEO_LICK_DIR)
    parser.add_argument('--video_scratch_dir',   default=HARD_VIDEO_SCRATCH_DIR)
    parser.add_argument('--video_shake_dir',     default=HARD_VIDEO_SHAKE_DIR)
    parser.add_argument('--video_stop_dir',      default=HARD_VIDEO_STOP_DIR)
    parser.add_argument('--yolo',                default=DEFAULT_YOLO)
    parser.add_argument('--imgsz',               type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument('--conf',                type=float, default=DEFAULT_CONF)
    parser.add_argument('--output',              default=HARD_OUTPUT_DIR)
    parser.add_argument('--sequence_length',     type=int,   default=HARD_SEQ_LEN)
    parser.add_argument('--classify_stride',     type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--ema_alpha',           type=float, default=HARD_EMA_ALPHA)
    parser.add_argument('--device',              default=DEFAULT_DEVICE)
    args = parser.parse_args()

    name = args.name or _short_name(args.model)
    label = f"{name}[ema={args.ema_alpha}]" if args.ema_alpha < 1.0 else name

    # ── 資料夾解析：讀不到的跳過；至少要有一個讀得到 ─────────────────────
    _all_dirs = [
        (args.video_walk_dir,    0),
        (args.video_lick_dir,    1),
        (args.video_scratch_dir, 2),
        (args.video_shake_dir,   3),
        (args.video_stop_dir,    4),
    ]
    dirs = [(p, idx) for p, idx in _all_dirs if p and Path(p).is_dir()]
    skipped = [(BEHAVIOR_CLASSES[idx], p) for p, idx in _all_dirs
               if not p or not Path(p).is_dir()]
    for cls_name, p in skipped:
        reason = '（路徑未設定）' if not p else f'（找不到資料夾: {p}）'
        print(f"  ⚠ 跳過 [{cls_name}] {reason}")
    if not dirs:
        raise RuntimeError("找不到任何可用的行為資料夾，至少需要一個有效路徑才能執行評估。")

    # Sequential output directory
    out_root = Path(args.output)
    run_num  = _next_run_number(str(out_root))
    out_dir  = out_root / f"single_eval_{run_num:03d}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = '=' * 62
    print(f"\n{sep}")
    print(f"  Single-Model Worst-Video Evaluation #{run_num:03d}")
    print(f"  Model  : {label}  (EMA α={args.ema_alpha}  T={args.sequence_length})")
    print(f"  Output : {out_dir}")
    print(f"{sep}")

    # ── Load detectors ────────────────────────────────────────────────────
    print('\n[Loading YOLO]')
    kp_det = KeypointDetector(args.yolo, device=args.device,
                               imgsz=args.imgsz, conf_thres=args.conf)

    print('\n[Loading ST-GCN model]')
    bn_ch = infer_bn_input_channels(args.model)
    fm    = CH_TO_FEATURE.get(bn_ch, 'xy')
    if bn_ch not in CH_TO_FEATURE:
        print(f"  ⚠ {Path(args.model).name}: cannot infer feature_mode "
              f"(bn_ch={bn_ch}), defaulting to 'xy'")
    else:
        print(f"  ✓ {Path(args.model).name} → {fm} ({bn_ch} ch)  T={args.sequence_length}")
    clf = BehaviorClassifier(
        args.model, device=args.device,
        sequence_length=args.sequence_length,
        normalize=True, feature_mode=fm, in_channels=bn_ch,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    preds_by_class: dict = {}   # cls_idx -> [(video_name, preds, detection_stats), ...]
    for dir_path, cls_idx in dirs:
        cls_name = BEHAVIOR_CLASSES[cls_idx]
        print(f"\n[{cls_name.upper()}]  {Path(dir_path).name}/")
        preds_by_class[cls_idx] = evaluate_folder(
            dir_path, kp_det, clf, fm,
            args.sequence_length, args.classify_stride, ema_alpha=args.ema_alpha
        )
        nw = sum(len(p) for _, p, _ in preds_by_class[cls_idx])
        print(f"  → {len(preds_by_class[cls_idx])} videos  |  {nw} windows")

    # ── Per-video metrics ─────────────────────────────────────────────────
    video_metrics = []
    for cls_idx, vid_preds_list in preds_by_class.items():
        for video_name, preds, det_stats in vid_preds_list:
            video_metrics.append(compute_video_metrics(video_name, cls_idx, preds, det_stats))

    if not video_metrics:
        raise RuntimeError("所有已找到的資料夾內都沒有可用的影片，無法產生評估結果。")

    overall_metrics = compute_overall_metrics(preds_by_class)

    # ── Save outputs ─────────────────────────────────────────────────────
    print('\n[Saving]')
    save_video_breakdown_csv(video_metrics, out_dir / 'video_breakdown.csv')
    plot_per_class_accuracy(video_metrics, BEHAVIOR_CLASSES, out_dir / 'per_class_accuracy.png')
    plot_confusion_matrix(overall_metrics['confusion_matrix'], BEHAVIOR_CLASSES,
                          out_dir / 'confusion_matrix.png')
    print(f'\n✓ All results saved to: {out_dir}')

    print_worst_videos_report(video_metrics, overall_metrics, label)


if __name__ == '__main__':
    main()
