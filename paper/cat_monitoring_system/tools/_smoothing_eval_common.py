"""共用評估核心，供 eval_accuracy_smoothing_compare.py 使用。

架構沿用 eval_ema_ablation.py 的既有慣例（labeled 影片資料夾、
KeypointDetector + BehaviorClassifier、accuracy/top2/macro-F1/event-detection
-rate 指標、CSV + PNG 輸出、跑號自動遞增的輸出目錄）。差異：
eval_ema_ablation.py 比較的是「不同 checkpoint」（各自用不同 KP_EMA_ALPHA
訓練），這裡比較的是「同一個 checkpoint、關鍵點前處理方式不同」——見
docs/YOLO-Pose應用文獻與專案優化建議.md「深入」章節的路線 A（Kalman filter
後處理，不需要重新訓練）。

**一次跑完多個前處理設定**：`evaluate_video_multi()` 對每支影片只呼叫一次
YOLO 偵測（偵測結果不受關鍵點前處理方式影響），偵測到的原始關鍵點序列
會分別餵給每個設定的 `preprocess_fn` 再各自跑 ST-GCN 分類——YOLO 通常是
整條管線裡最貴的部分，這樣比「每個設定各自重跑一次影片」快，也保證每個
設定看到的是完全相同的 YOLO 偵測結果，比較才公平。

**重要提醒（誠實揭露，別被自己的腳本騙了）**：目前部署的 checkpoint 是用
`interpolate_missing`（不平滑，`KP_EMA_ALPHA=1.0`）訓練出來的。用這裡的
Kalman 平滑去餵同一個 checkpoint，屬於「推論時分布偏移」的實測，不是
「公平重訓後比較」——結果只能回答「不重新訓練、單純換前處理，有沒有幫助」
這個問題，不能回答「Kalman 訓練出來的模型會不會更好」（那需要真的用
Kalman 平滑過的資料重新訓練，見 docs 的路線 B）。
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from models.stgcn_model import (
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
)
from utils.constants import BEHAVIOR_CLASSES

EVENT_MIN_WINDOWS = 3
EVENT_MIN_RATIO = 0.30
PROB_EVENT_THRESHOLD = 0.40

CH_TO_FEATURE = {
    2: "xy",
    3: "xy_conf",
    5: "xy_conf_v",
    7: "xy_conf_v_bone",
    9: "xy_conf_v_bone_bmotion",
}

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
_PALETTE = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def next_run_number(out_root: str, prefix: str) -> int:
    import re

    p = Path(out_root)
    if not p.exists():
        return 1
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)")
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
    ck = torch.load(model_path, map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    if isinstance(sd, dict):
        for k, v in sd.items():
            if k.endswith("bn_input.weight"):
                return int(v.shape[0])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inference — 一次跑完全部前處理設定
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_video_multi(
    video_path,
    kp_detector,
    classifier,
    feature_mode,
    preprocess_fns: dict,
    sequence_length=16,
    classify_stride=2,
):
    """對一支影片跑完整推論，同時評估 ``preprocess_fns`` 裡的每個設定。

    ``preprocess_fns``: ``{config_name: fn(kpts_arr, conf_arr) -> seq_array}``

    回傳 ``{config_name: [pred, ...]}``——YOLO 偵測只跑一次，每個設定各自
    只多花「關鍵點前處理 + ST-GCN 分類」這兩步的成本，不用重跑一次 YOLO。
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kp_detector.reset_track()  # 新影片開始，避免延續上一支影片鎖定的貓
    buf = deque(maxlen=sequence_length)
    preds = {name: [] for name in preprocess_fns}
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
        kpts_arr = np.array(
            [item[0] if item[0] is not None else np.zeros((17, 2), np.float32) for item in buf]
        )
        conf_arr = np.array(
            [item[1] if item[1] is not None else np.zeros((17,), np.float32) for item in buf]
        )
        _mj = getattr(classifier.model, "num_joints", 17)
        if _mj < 17:
            kpts_arr = kpts_arr[:, :_mj, :]
            conf_arr = conf_arr[:, :_mj]

        for name, preprocess_fn in preprocess_fns.items():
            seq = preprocess_fn(kpts_arr, conf_arr)
            seq = flip_normalize(seq)
            seq = orientation_normalize(seq)
            seq = normalize_skeleton_coords(seq)
            feats = build_feature_tensor(seq, conf_arr, feature_mode)
            pred_id, pred_conf, pred_probs = classifier.model.predict(feats, precomputed=True)
            if pred_id is None:
                pred_id, pred_conf, pred_probs = -1, 0.0, [0.0] * len(BEHAVIOR_CLASSES)
            preds[name].append(
                {
                    "frame": frame_idx,
                    "time": round(frame_idx / fps, 3),
                    "pred": int(pred_id),
                    "conf": float(pred_conf),
                    "probs": [float(x) for x in pred_probs],
                }
            )
    cap.release()
    return preds


def evaluate_folder_multi(
    folder_path, kp_detector, classifier, feature_mode, preprocess_fns,
    sequence_length=16, classify_stride=2,
):
    """回傳 ``{config_name: [(video_name, preds), ...]}``"""
    folder = Path(folder_path)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        print(f"    ⚠ No videos in: {folder}")
        return {name: [] for name in preprocess_fns}
    results = {name: [] for name in preprocess_fns}
    for vid in videos:
        print(f"    {vid.name} ...", end=" ", flush=True)
        preds_by_config = evaluate_video_multi(
            vid, kp_detector, classifier, feature_mode, preprocess_fns,
            sequence_length, classify_stride,
        )
        n_windows = len(next(iter(preds_by_config.values()), []))
        print(f"{n_windows} windows × {len(preprocess_fns)} configs")
        for name, preds in preds_by_config.items():
            results[name].append((vid.name, preds))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_by_class: dict) -> dict:
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class = {
        i: {
            "accuracy": 0.0, "top2_accuracy": 0.0, "avg_true_prob": 0.0,
            "max_true_prob": 0.0, "event_rate": 0.0, "prob_event_rate": 0.0,
            "n_correct": 0, "n_windows": 0, "n_videos": 0,
            "n_videos_detected": 0, "n_videos_prob_detected": 0,
        }
        for i in range(n_cls)
    }

    for cls_idx, vid_preds_list in preds_by_class.items():
        preds = [p for vid_preds in vid_preds_list for p in vid_preds]
        if not preds:
            continue
        probs = np.array([p["probs"] for p in preds])
        actual_cls = probs.shape[1]
        pred_ids = np.clip(np.array([p["pred"] for p in preds], dtype=int), 0, actual_cls - 1)
        probs_cls = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(len(preds))
        n_correct = int((pred_ids == cls_idx).sum())
        top2_ids = np.argsort(probs, axis=1)[:, -2:]
        n_top2 = int(np.any(top2_ids == cls_idx, axis=1).sum())

        n_videos, n_vid_det, n_vid_prob_det = len(vid_preds_list), 0, 0
        for vid_preds in vid_preds_list:
            if not vid_preds:
                continue
            vp = np.clip(np.array([p["pred"] for p in vid_preds], dtype=int), 0, actual_cls - 1)
            vc, vn = int((vp == cls_idx).sum()), len(vid_preds)
            if (vc / vn) >= EVENT_MIN_RATIO or vc >= EVENT_MIN_WINDOWS:
                n_vid_det += 1
            vpc = np.array([p["probs"][cls_idx] for p in vid_preds if cls_idx < len(p["probs"])])
            if len(vpc) > 0 and int((vpc >= PROB_EVENT_THRESHOLD).sum()) >= EVENT_MIN_WINDOWS:
                n_vid_prob_det += 1

        per_class[cls_idx] = {
            "accuracy": float(n_correct / len(preds)),
            "top2_accuracy": float(n_top2 / len(preds)),
            "avg_true_prob": float(probs_cls.mean()),
            "max_true_prob": float(probs_cls.max()),
            "event_rate": float(n_vid_det / n_videos) if n_videos else 0.0,
            "prob_event_rate": float(n_vid_prob_det / n_videos) if n_videos else 0.0,
            "n_correct": n_correct,
            "n_windows": len(preds),
            "n_videos": n_videos,
            "n_videos_detected": n_vid_det,
            "n_videos_prob_detected": n_vid_prob_det,
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    evaluated = [i for i in range(n_cls) if per_class[i]["n_windows"] > 0]
    top2_acc = float(np.mean([per_class[i]["top2_accuracy"] for i in evaluated])) if evaluated else 0.0
    ev_rate = float(np.mean([per_class[i]["event_rate"] for i in evaluated])) if evaluated else 0.0

    return {
        "per_class": per_class,
        "overall": {
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
            "top2_accuracy": top2_acc,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
            "event_detection_rate": ev_rate,
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output — 多設定並排比較
# ═══════════════════════════════════════════════════════════════════════════
def save_comparison_csv(all_metrics, names, classes, out_path):
    import csv

    header = ["metric"] + names
    rows = [header]

    def _row(label, vals):
        rows.append([label] + [f"{v:.4f}" for v in vals])

    _row("overall_accuracy", [m["overall"]["accuracy"] for m in all_metrics])
    _row("overall_top2_acc", [m["overall"]["top2_accuracy"] for m in all_metrics])
    _row("overall_macro_f1", [m["overall"]["macro_f1"] for m in all_metrics])
    _row("overall_event_rate", [m["overall"]["event_detection_rate"] for m in all_metrics])
    rows.append([])

    for i, cls in enumerate(classes):
        for key in ("accuracy", "top2_accuracy", "avg_true_prob", "max_true_prob", "event_rate", "prob_event_rate"):
            _row(f"{cls}_{key}", [m["per_class"][i].get(key, 0.0) for m in all_metrics])
        rows.append(
            [f"{cls}_n_videos_detected"]
            + [f"{m['per_class'][i]['n_videos_detected']}/{m['per_class'][i]['n_videos']}" for m in all_metrics]
        )
        rows.append([f"{cls}_n_windows"] + [str(m["per_class"][i]["n_windows"]) for m in all_metrics])
        rows.append([])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {Path(out_path).name}")


def plot_accuracy_comparison(all_metrics, names, colors, classes, out_path):
    """Grouped bar chart：每個類別 + Overall，每個設定一個 bar，並排比較。"""
    n_models = len(all_metrics)
    n_cls = len(classes)
    x_labels = classes + ["Overall"]
    x = np.arange(len(x_labels))
    total_w = 0.72
    bar_w = total_w / n_models
    offsets = np.linspace(-total_w / 2 + bar_w / 2, total_w / 2 - bar_w / 2, n_models)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    fig.suptitle("Keypoint Smoothing Comparison — Accuracy", fontsize=13, fontweight="bold")

    for mi, (metrics, name, col) in enumerate(zip(all_metrics, names, colors)):
        pc = metrics["per_class"]
        vals = [pc[i]["accuracy"] for i in range(n_cls)] + [metrics["overall"]["accuracy"]]
        bars = ax.bar(x + offsets[mi], vals, bar_w, label=name, color=col, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.1%}",
                         ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {Path(out_path).name}")


def plot_confusion_comparison(all_metrics, names, colors, classes, out_path):
    n = len(all_metrics)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), constrained_layout=True)
    fig.suptitle("Confusion Matrices — Keypoint Smoothing Comparison", fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    try:
        import seaborn as sns

        _sns = True
    except ImportError:
        _sns = False

    for ax, metrics, name, col in zip(axes_flat, all_metrics, names, colors):
        cm = metrics["confusion_matrix"]
        if _sns:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=classes,
                        yticklabels=classes, ax=ax, cbar=False, linewidths=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=30, ha="right")
            ax.set_yticklabels(classes)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        avg_recall = float(np.mean([cm[i, i] / max(cm[i].sum(), 1) for i in range(len(classes))]))
        ax.set_title(f"{name}\nAvg Recall={avg_recall:.1%}", fontsize=10, fontweight="bold", color=col)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {Path(out_path).name}")


def print_comparison_summary(all_metrics, names, classes):
    sep = "=" * 70
    print(f"\n{sep}")
    col_w = max(len(n) for n in names)
    print(f"  {'Metric':<24}  " + "  ".join(f"{n:>{col_w}}" for n in names))
    print(f"  {'-' * (24 + (col_w + 2) * len(names))}")

    def _row(label, vals):
        best = max(vals)
        parts = []
        for v in vals:
            marker = " ★" if abs(v - best) < 0.001 else "  "
            parts.append(f"{v:>{col_w}.4f}{marker}")
        print(f"  {label:<24}  " + "  ".join(parts))

    _row("Overall Accuracy", [m["overall"]["accuracy"] for m in all_metrics])
    _row("Overall Macro-F1", [m["overall"]["macro_f1"] for m in all_metrics])
    _row("Overall Top-2 Acc", [m["overall"]["top2_accuracy"] for m in all_metrics])
    _row("Event Detection Rate", [m["overall"]["event_detection_rate"] for m in all_metrics])
    print()
    for i, cls in enumerate(classes):
        if all(m["per_class"][i]["n_windows"] == 0 for m in all_metrics):
            continue
        _row(f"{cls} accuracy", [m["per_class"][i]["accuracy"] for m in all_metrics])
        _row(f"{cls} avg_true_prob", [m["per_class"][i]["avg_true_prob"] for m in all_metrics])
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════
# 主流程：一次跑完全部設定的比較
# ═══════════════════════════════════════════════════════════════════════════
def run_comparison(
    model_path,
    yolo_path,
    video_dirs,
    configs,
    output_root,
    output_prefix,
    imgsz=640,
    bbox_conf_thres=0.5,
    sequence_length=16,
    classify_stride=2,
    device="cuda",
):
    """一口氣跑完 ``configs`` 裡的每個關鍵點前處理設定，輸出並排比較結果。

    Parameters
    ----------
    video_dirs : dict[class_name -> folder_path or None]
    configs : list[{"name": str, "preprocess_fn": callable}]
        每個設定的顯示名稱跟前處理函式，直接寫在呼叫端的模組層級常數裡
        管理（見 eval_accuracy_smoothing_compare.py 的 SMOOTHING_CONFIGS）。
    """
    out_root = Path(output_root)
    run_num = next_run_number(str(out_root), output_prefix)
    out_dir = out_root / f"{output_prefix}_{run_num:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [c["name"] for c in configs]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(configs))]
    preprocess_fns = {c["name"]: c["preprocess_fn"] for c in configs}

    sep = "=" * 64
    print(f"\n{sep}\n  Keypoint Smoothing Comparison  #{run_num:03d}\n  Model: {model_path}")
    for n in names:
        print(f"  · {n}")
    print(f"  Output: {out_dir}\n{sep}")

    print("\n[Loading YOLO]")
    kp_det = KeypointDetector(yolo_path, device=device, imgsz=imgsz, bbox_conf_thres=bbox_conf_thres)

    print("\n[Loading ST-GCN model]")
    bn_ch = infer_bn_input_channels(model_path)
    feature_mode = CH_TO_FEATURE.get(bn_ch, "xy")
    print(f"  ✓ {Path(model_path).name}  → {feature_mode} ({bn_ch} ch)")
    classifier = BehaviorClassifier(
        model_path, device=device, sequence_length=sequence_length,
        normalize=True, feature_mode=feature_mode, in_channels=bn_ch,
    )

    # all_preds[config_name][cls_idx] = [[pred,...], ...]（每支影片一個 list）
    all_preds = {name: {} for name in names}
    for cls_idx, cls_name in enumerate(BEHAVIOR_CLASSES):
        dir_path = video_dirs.get(cls_name)
        if not dir_path or not Path(dir_path).is_dir():
            print(f"  ⚠ 跳過 [{cls_name}]" + (f" 找不到資料夾: {dir_path}" if dir_path else " 路徑未設定"))
            continue
        print(f"\n[{cls_name.upper()}]  {Path(dir_path).name}/")
        results = evaluate_folder_multi(
            dir_path, kp_det, classifier, feature_mode, preprocess_fns,
            sequence_length, classify_stride,
        )
        for name in names:
            all_preds[name][cls_idx] = [p for _, p in results[name]]
        nw = sum(len(v) for v in all_preds[names[0]][cls_idx])
        print(f"    → {len(all_preds[names[0]][cls_idx])} videos  {nw} windows/config")

    all_metrics = [compute_metrics(all_preds[name]) for name in names]
    print_comparison_summary(all_metrics, names, BEHAVIOR_CLASSES)

    print("\n[Saving]")
    save_comparison_csv(all_metrics, names, BEHAVIOR_CLASSES, out_dir / f"{output_prefix}_summary.csv")
    plot_accuracy_comparison(all_metrics, names, colors, BEHAVIOR_CLASSES, out_dir / f"{output_prefix}_accuracy.png")
    plot_confusion_comparison(all_metrics, names, colors, BEHAVIOR_CLASSES, out_dir / f"{output_prefix}_confusion.png")
    print(f"\n✓ All results saved to: {out_dir}")
    return all_metrics, out_dir
