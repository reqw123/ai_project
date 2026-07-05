#!/usr/bin/env python3
"""
YOLO Pose Model Comparison Benchmark Tool
比較兩個 YOLO Pose 模型在 Benchmark Dataset 上的量化表現
完全背景執行，不開 GUI，輸出 CSV / PNG / TXT / LOG / HTML
"""

import csv
import html as _html
import logging
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back",
    "Hip", "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]
BEHAVIOR_CLASSES = ["walk", "lick", "scratch", "shake", "stop"]
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}

JITTER_CONF_THRESHOLD = 0.3
YOLO_CONF_THRESHOLD   = 0.5
YOLO_IMGSZ            = 640

NOSE_IDX     = [0]
PAW_IDX      = [7, 9, 11, 13]
TAIL_IDX     = [14, 15, 16]
TAIL_TIP_IDX = 16

W_CONFIDENCE = 0.35
W_JITTER     = 0.35
W_MISSING    = 0.30

COLOR_OLD = "#4C72B0"
COLOR_NEW = "#DD8452"


# ─── CJK-aware 對齊輔助 ───────────────────────────────────────────
def _vlen(s: str) -> int:
    """字串的視覺寬度（中文 / 全形 = 2，其餘 = 1）。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

def _vlj(s: str, w: int) -> str:
    """左對齊，依視覺寬度補空格。"""
    return s + " " * max(0, w - _vlen(s))

def _vrj(s: str, w: int) -> str:
    """右對齊，依視覺寬度補空格。"""
    return " " * max(0, w - _vlen(s)) + s


# ═══════════════════════════════════════════════════════
#  設定區  ── 修改這裡，不需要命令列參數
# ═══════════════════════════════════════════════════════
OLD_MODEL_PATH   = r"C:\ai_project\cat_pose\v11s_116.pt"
NEW_MODEL_PATH   = r"C:\ai_project\cat_pose\v11s_117.pt"
BENCHMARK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試"
OUTPUT_DIR       = r"C:\ai_project\compare_results"
INFERENCE_DEVICE = "cuda"   # "cuda" 或 "cpu"
EMA_ALPHA_OLD    = 1.0      # Old model EMA 平滑係數（1.0 = 不平滑；0.5 = 半衰期平滑）
EMA_ALPHA_NEW    = 1.0      # New model EMA 平滑係數
# ═══════════════════════════════════════════════════════


# ─── Logger ───────────────────────────────────────────────────────
def setup_logger(log_dir: Path, run_ts: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"compare_pose_{run_ts}")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_dir / f"run_{run_ts}.log", encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ─── EMA helper ──────────────────────────────────────────────────
def _apply_ema_list(values: list, alpha: float) -> list:
    """對值序列套用 EMA 平滑。alpha=1.0 = 不平滑。"""
    if alpha >= 1.0 or not values:
        return values
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1.0 - alpha) * result[-1])
    return result


# ─── Model Loading ────────────────────────────────────────────────
def load_yolo_model(model_path: str, device: str):
    from ultralytics import YOLO
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO 模型檔案不存在: {model_path}")
    m = YOLO(model_path)
    m.to(device)
    return m


def infer_frame(model, frame: np.ndarray):
    """Return (kpts (17,2), kpt_conf (17,), bbox (4,)) or (None,None,None)."""
    results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, verbose=False)
    if not results or results[0].keypoints is None:
        return None, None, None
    kp = results[0].keypoints
    if kp.xy is None or len(kp.xy) == 0:
        return None, None, None

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        best = int(np.argmax(boxes.conf.cpu().numpy()))
    else:
        best = 0

    if best >= len(kp.xy):
        return None, None, None

    kpts = kp.xy[best].cpu().numpy().astype(np.float32)
    if len(kpts) < 17:
        return None, None, None

    if kp.conf is not None and len(kp.conf) > best:
        kpt_conf = kp.conf[best].cpu().numpy().astype(np.float32)
    else:
        kpt_conf = np.full(17, 0.5, dtype=np.float32)

    bbox = None
    if boxes is not None and len(boxes) > best:
        bbox = boxes.xyxy[best].cpu().numpy()

    return kpts, kpt_conf, bbox


# ─── Video Collection ─────────────────────────────────────────────
def resolve_videos(videos_dir: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for b in BEHAVIOR_CLASSES:
        bdir = videos_dir / b
        if bdir.is_dir():
            result[b] = sorted(
                f for f in bdir.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            )
        else:
            result[b] = []
    return result


# ─── Single Video Processing ──────────────────────────────────────
def process_video(model, video_path: Path, logger: logging.Logger,
                  ema_alpha: float = 1.0) -> Optional[Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open: {video_path}")
        return None

    frames = 0
    frames_with_cat = 0
    frames_without_cat = 0
    frame_confidences: List[float] = []
    kpt_conf_lists = [[] for _ in range(17)]
    jitter_lists   = [[] for _ in range(17)]
    kpt_present    = [0] * 17

    prev_kpts = prev_kpt_conf = None
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        kpts, kpt_conf, bbox = infer_frame(model, frame)

        if kpts is not None:
            frames_with_cat += 1
            for i in range(17):
                c = float(kpt_conf[i])
                kpt_conf_lists[i].append(c)
                if c >= JITTER_CONF_THRESHOLD:
                    kpt_present[i] += 1

            valid = kpt_conf[kpt_conf >= JITTER_CONF_THRESHOLD]
            if len(valid) > 0:
                frame_confidences.append(float(np.mean(valid)))

            if prev_kpts is not None:
                mask = (kpt_conf >= JITTER_CONF_THRESHOLD) & (prev_kpt_conf >= JITTER_CONF_THRESHOLD)
                for i in range(17):
                    if mask[i]:
                        jitter_lists[i].append(float(np.linalg.norm(kpts[i] - prev_kpts[i])))

            prev_kpts, prev_kpt_conf = kpts.copy(), kpt_conf.copy()
        else:
            frames_without_cat += 1
            prev_kpts = prev_kpt_conf = None

    cap.release()
    elapsed = time.time() - t0

    if frames == 0:
        return None

    # 套用 EMA 平滑（alpha < 1.0 時才實際作用）
    frame_confidences = _apply_ema_list(frame_confidences, ema_alpha)
    kpt_conf_lists    = [_apply_ema_list(lst, ema_alpha) for lst in kpt_conf_lists]
    jitter_lists      = [_apply_ema_list(lst, ema_alpha) for lst in jitter_lists]

    all_jitter = [v for lst in jitter_lists for v in lst]

    kpt_missing_ratio = []
    for i in range(17):
        if frames_with_cat > 0:
            kpt_missing_ratio.append((frames_with_cat - kpt_present[i]) / frames_with_cat)
        else:
            kpt_missing_ratio.append(1.0)

    return {
        "frames": frames,
        "frames_with_cat": frames_with_cat,
        "frames_without_cat": frames_without_cat,
        "frame_confidences": frame_confidences,
        "kpt_conf_lists": kpt_conf_lists,
        "jitter_lists": jitter_lists,
        "kpt_missing_ratio": kpt_missing_ratio,
        "inference_time": elapsed,
        "mean_confidence": float(np.mean(frame_confidences)) if frame_confidences else 0.0,
        "std_confidence":  float(np.std(frame_confidences))  if frame_confidences else 0.0,
        "mean_jitter_px":  float(np.mean(all_jitter))        if all_jitter else 0.0,
        "std_jitter_px":   float(np.std(all_jitter))         if all_jitter else 0.0,
        "p95_jitter_px":   float(np.percentile(all_jitter, 95)) if all_jitter else 0.0,
        "max_jitter_px":   float(np.max(all_jitter))         if all_jitter else 0.0,
        "missing_ratio":      float(frames_without_cat / frames),
        "nose_missing_ratio": float(kpt_missing_ratio[0]),
        "paw_missing_ratio":  float(np.mean([kpt_missing_ratio[i] for i in PAW_IDX])),
        "tail_missing_ratio": float(np.mean([kpt_missing_ratio[i] for i in TAIL_IDX])),
        "tail_tip_missing_ratio": float(kpt_missing_ratio[TAIL_TIP_IDX]),
    }


# ─── Aggregation ──────────────────────────────────────────────────
_ZERO_AGG = {
    "mean_confidence": 0.0, "std_confidence": 0.0,
    "mean_jitter_px": 0.0,  "p95_jitter_px": 0.0,
    "missing_ratio": 0.0,   "nose_missing_ratio": 0.0,
    "paw_missing_ratio": 0.0, "tail_missing_ratio": 0.0,
    "tail_tip_missing_ratio": 0.0,
    "kpt_mean_confidence": [0.0] * 17,
    "kpt_std_confidence":  [0.0] * 17,
    "kpt_mean_jitter":     [0.0] * 17,
    "kpt_p95_jitter":      [0.0] * 17,
    "kpt_missing_ratio":   [0.0] * 17,
}


def aggregate(results: List[Dict]) -> Dict:
    valid = [r for r in results if r is not None]
    if not valid:
        return dict(_ZERO_AGG)

    all_confs   = [c for r in valid for c in r["frame_confidences"]]
    all_jitters = [v for r in valid for lst in r["jitter_lists"] for v in lst]
    total_frames = sum(r["frames"] for r in valid)
    total_no_cat = sum(r["frames_without_cat"] for r in valid)

    kpt_conf_all   = [[] for _ in range(17)]
    kpt_jitter_all = [[] for _ in range(17)]
    for r in valid:
        for i in range(17):
            kpt_conf_all[i].extend(r["kpt_conf_lists"][i])
            kpt_jitter_all[i].extend(r["jitter_lists"][i])

    kpt_mean_conf = [float(np.mean(kpt_conf_all[i]))  if kpt_conf_all[i]   else 0.0 for i in range(17)]
    kpt_std_conf  = [float(np.std(kpt_conf_all[i]))   if kpt_conf_all[i]   else 0.0 for i in range(17)]
    kpt_mean_jit  = [float(np.mean(kpt_jitter_all[i])) if kpt_jitter_all[i] else 0.0 for i in range(17)]
    kpt_p95_jit   = [float(np.percentile(kpt_jitter_all[i], 95)) if kpt_jitter_all[i] else 0.0 for i in range(17)]
    kpt_miss      = [float(np.mean([r["kpt_missing_ratio"][i] for r in valid])) for i in range(17)]

    return {
        "mean_confidence": float(np.mean(all_confs))  if all_confs   else 0.0,
        "std_confidence":  float(np.std(all_confs))   if all_confs   else 0.0,
        "mean_jitter_px":  float(np.mean(all_jitters)) if all_jitters else 0.0,
        "p95_jitter_px":   float(np.percentile(all_jitters, 95)) if all_jitters else 0.0,
        "missing_ratio":      float(total_no_cat / total_frames) if total_frames > 0 else 0.0,
        "nose_missing_ratio": float(np.mean([r["nose_missing_ratio"] for r in valid])),
        "paw_missing_ratio":  float(np.mean([r["paw_missing_ratio"]  for r in valid])),
        "tail_missing_ratio": float(np.mean([r["tail_missing_ratio"] for r in valid])),
        "tail_tip_missing_ratio": float(np.mean([r["tail_tip_missing_ratio"] for r in valid])),
        "kpt_mean_confidence": kpt_mean_conf,
        "kpt_std_confidence":  kpt_std_conf,
        "kpt_mean_jitter":     kpt_mean_jit,
        "kpt_p95_jitter":      kpt_p95_jit,
        "kpt_missing_ratio":   kpt_miss,
    }


# ─── Metric Helpers ───────────────────────────────────────────────
def improvement(old, new, higher_is_better=True) -> float:
    if abs(old) < 1e-9:
        return 0.0
    return ((new - old) if higher_is_better else (old - new)) / abs(old) * 100.0


def composite_score(m: Dict) -> float:
    conf_s    = m["mean_confidence"] * 100.0
    jitter_s  = max(0.0, 100.0 - m["mean_jitter_px"] * 2.0)
    missing_s = (1.0 - m["missing_ratio"]) * 100.0
    return W_CONFIDENCE * conf_s + W_JITTER * jitter_s + W_MISSING * missing_s


def behavior_score(old_m: Dict, new_m: Dict) -> float:
    ci = improvement(old_m["mean_confidence"], new_m["mean_confidence"], True)
    ji = improvement(old_m["mean_jitter_px"],  new_m["mean_jitter_px"],  False)
    mi = improvement(old_m["missing_ratio"],   new_m["missing_ratio"],   False)
    return W_CONFIDENCE * ci + W_JITTER * ji + W_MISSING * mi


def best_worst_behavior(class_old: Dict, class_new: Dict) -> Tuple[str, float, str, float]:
    scores = {b: behavior_score(class_old[b], class_new[b])
              for b in BEHAVIOR_CLASSES if class_old.get(b) and class_new.get(b)}
    if not scores:
        return "N/A", 0.0, "N/A", 0.0
    best  = max(scores, key=scores.get)
    worst = min(scores, key=scores.get)
    return best, scores[best], worst, scores[worst]


# ─── CSV Writers ──────────────────────────────────────────────────
def write_video_metrics_csv(csv_dir: Path, results: Dict):
    path = csv_dir / "video_metrics.csv"
    fields = [
        "model", "behavior", "video_name",
        "frames", "frames_with_cat", "frames_without_cat",
        "mean_confidence", "std_confidence",
        "mean_jitter_px", "std_jitter_px", "p95_jitter_px", "max_jitter_px",
        "missing_ratio", "nose_missing_ratio", "paw_missing_ratio", "tail_missing_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mk in ("old", "new"):
            for b in BEHAVIOR_CLASSES:
                for vpath, r in zip(results["videos"][b], results[mk][b]):
                    if r is None:
                        continue
                    w.writerow({
                        "model": mk, "behavior": b,
                        "video_name": Path(vpath).name,
                        "frames": r["frames"],
                        "frames_with_cat": r["frames_with_cat"],
                        "frames_without_cat": r["frames_without_cat"],
                        "mean_confidence": f"{r['mean_confidence']:.4f}",
                        "std_confidence":  f"{r['std_confidence']:.4f}",
                        "mean_jitter_px":  f"{r['mean_jitter_px']:.3f}",
                        "std_jitter_px":   f"{r['std_jitter_px']:.3f}",
                        "p95_jitter_px":   f"{r['p95_jitter_px']:.3f}",
                        "max_jitter_px":   f"{r['max_jitter_px']:.3f}",
                        "missing_ratio":      f"{r['missing_ratio']:.4f}",
                        "nose_missing_ratio": f"{r['nose_missing_ratio']:.4f}",
                        "paw_missing_ratio":  f"{r['paw_missing_ratio']:.4f}",
                        "tail_missing_ratio": f"{r['tail_missing_ratio']:.4f}",
                    })
    return path


def write_keypoint_metrics_csv(csv_dir: Path, agg_old: Dict, agg_new: Dict):
    path = csv_dir / "keypoint_metrics.csv"
    fields = [
        "model", "keypoint_idx", "keypoint_name",
        "mean_confidence", "std_confidence",
        "mean_jitter", "p95_jitter", "missing_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mk, agg in [("old", agg_old), ("new", agg_new)]:
            for i, name in enumerate(KEYPOINT_NAMES):
                w.writerow({
                    "model": mk, "keypoint_idx": i, "keypoint_name": name,
                    "mean_confidence": f"{agg['kpt_mean_confidence'][i]:.4f}",
                    "std_confidence":  f"{agg['kpt_std_confidence'][i]:.4f}",
                    "mean_jitter":     f"{agg['kpt_mean_jitter'][i]:.3f}",
                    "p95_jitter":      f"{agg['kpt_p95_jitter'][i]:.3f}",
                    "missing_ratio":   f"{agg['kpt_missing_ratio'][i]:.4f}",
                })
    return path


def write_class_summary_csv(csv_dir: Path, class_old: Dict, class_new: Dict):
    path = csv_dir / "class_summary.csv"
    fields = [
        "behavior",
        "old_mean_confidence", "new_mean_confidence", "conf_change", "conf_improvement_pct",
        "old_mean_jitter",     "new_mean_jitter",     "jitter_change", "jitter_improvement_pct",
        "old_missing_ratio",   "new_missing_ratio",   "missing_change", "missing_improvement_pct",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b in BEHAVIOR_CLASSES:
            o, n = class_old[b], class_new[b]
            ci = improvement(o["mean_confidence"], n["mean_confidence"], True)
            ji = improvement(o["mean_jitter_px"],  n["mean_jitter_px"],  False)
            mi = improvement(o["missing_ratio"],   n["missing_ratio"],   False)
            w.writerow({
                "behavior": b,
                "old_mean_confidence": f"{o['mean_confidence']:.4f}",
                "new_mean_confidence": f"{n['mean_confidence']:.4f}",
                "conf_change":   f"{n['mean_confidence']-o['mean_confidence']:+.4f}",
                "conf_improvement_pct":    f"{ci:+.2f}%",
                "old_mean_jitter": f"{o['mean_jitter_px']:.3f}",
                "new_mean_jitter": f"{n['mean_jitter_px']:.3f}",
                "jitter_change":   f"{n['mean_jitter_px']-o['mean_jitter_px']:+.3f}",
                "jitter_improvement_pct":  f"{ji:+.2f}%",
                "old_missing_ratio": f"{o['missing_ratio']:.4f}",
                "new_missing_ratio": f"{n['missing_ratio']:.4f}",
                "missing_change":    f"{n['missing_ratio']-o['missing_ratio']:+.4f}",
                "missing_improvement_pct": f"{mi:+.2f}%",
            })
    return path


def write_improvement_summary_csv(csv_dir: Path, agg_old: Dict, agg_new: Dict):
    path = csv_dir / "improvement_summary.csv"
    rows = [
        ("[PRIMARY] P95 Jitter (px)",
         f"{agg_old['p95_jitter_px']:.2f}", f"{agg_new['p95_jitter_px']:.2f}",
         f"{agg_new['p95_jitter_px']-agg_old['p95_jitter_px']:+.3f}",
         improvement(agg_old["p95_jitter_px"], agg_new["p95_jitter_px"], False)),
        ("Mean Confidence",
         f"{agg_old['mean_confidence']:.3f}", f"{agg_new['mean_confidence']:.3f}",
         f"{agg_new['mean_confidence']-agg_old['mean_confidence']:+.3f}",
         improvement(agg_old["mean_confidence"], agg_new["mean_confidence"], True)),
        ("Mean Jitter (px)",
         f"{agg_old['mean_jitter_px']:.2f}", f"{agg_new['mean_jitter_px']:.2f}",
         f"{agg_new['mean_jitter_px']-agg_old['mean_jitter_px']:+.3f}",
         improvement(agg_old["mean_jitter_px"], agg_new["mean_jitter_px"], False)),
        ("Missing Ratio",
         f"{agg_old['missing_ratio']*100:.1f}%", f"{agg_new['missing_ratio']*100:.1f}%",
         f"{(agg_new['missing_ratio']-agg_old['missing_ratio'])*100:+.1f}%",
         improvement(agg_old["missing_ratio"], agg_new["missing_ratio"], False)),
        ("Nose Missing",
         f"{agg_old['nose_missing_ratio']*100:.1f}%", f"{agg_new['nose_missing_ratio']*100:.1f}%",
         f"{(agg_new['nose_missing_ratio']-agg_old['nose_missing_ratio'])*100:+.1f}%",
         improvement(agg_old["nose_missing_ratio"], agg_new["nose_missing_ratio"], False)),
        ("Paw Missing",
         f"{agg_old['paw_missing_ratio']*100:.1f}%", f"{agg_new['paw_missing_ratio']*100:.1f}%",
         f"{(agg_new['paw_missing_ratio']-agg_old['paw_missing_ratio'])*100:+.1f}%",
         improvement(agg_old["paw_missing_ratio"], agg_new["paw_missing_ratio"], False)),
        ("Tail Missing",
         f"{agg_old['tail_missing_ratio']*100:.1f}%", f"{agg_new['tail_missing_ratio']*100:.1f}%",
         f"{(agg_new['tail_missing_ratio']-agg_old['tail_missing_ratio'])*100:+.1f}%",
         improvement(agg_old["tail_missing_ratio"], agg_new["tail_missing_ratio"], False)),
        ("Tail Tip Missing",
         f"{agg_old['tail_tip_missing_ratio']*100:.1f}%", f"{agg_new['tail_tip_missing_ratio']*100:.1f}%",
         f"{(agg_new['tail_tip_missing_ratio']-agg_old['tail_tip_missing_ratio'])*100:+.1f}%",
         improvement(agg_old["tail_tip_missing_ratio"], agg_new["tail_tip_missing_ratio"], False)),
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Old", "New", "Change", "Improvement"])
        for metric, old_s, new_s, change_s, imp_val in rows:
            w.writerow([metric, old_s, new_s, change_s, f"{imp_val:+.2f}%"])
    return path


# ─── Figures ──────────────────────────────────────────────────────
def _save(fig, path: Path):
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confidence_comparison(fig_dir, class_old, class_new, agg_old, agg_new):
    bnames = [b.capitalize() for b in BEHAVIOR_CLASSES]
    x = np.arange(len(BEHAVIOR_CLASSES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, [class_old[b]["mean_confidence"] for b in BEHAVIOR_CLASSES], w,
                label="Old Model", color=COLOR_OLD, alpha=0.85)
    b2 = ax.bar(x + w/2, [class_new[b]["mean_confidence"] for b in BEHAVIOR_CLASSES], w,
                label="New Model", color=COLOR_NEW, alpha=0.85)
    ax.set_title("Mean Confidence by Behavior", fontsize=14, fontweight="bold")
    ax.set_xlabel("Behavior"); ax.set_ylabel("Mean Confidence")
    ax.set_xticks(x); ax.set_xticklabels(bnames); ax.set_ylim(0, 1.15)
    ax.axhline(agg_old["mean_confidence"], color=COLOR_OLD, ls="--", lw=1, alpha=0.6)
    ax.axhline(agg_new["mean_confidence"], color=COLOR_NEW, ls="--", lw=1, alpha=0.6)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir / "confidence_comparison.png")


def plot_jitter_comparison(fig_dir, class_old, class_new, agg_old, agg_new):
    bnames = [b.capitalize() for b in BEHAVIOR_CLASSES]
    x = np.arange(len(BEHAVIOR_CLASSES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, [class_old[b]["mean_jitter_px"] for b in BEHAVIOR_CLASSES], w,
                label="Old Model", color=COLOR_OLD, alpha=0.85)
    b2 = ax.bar(x + w/2, [class_new[b]["mean_jitter_px"] for b in BEHAVIOR_CLASSES], w,
                label="New Model", color=COLOR_NEW, alpha=0.85)
    ax.set_title("Mean Jitter by Behavior", fontsize=14, fontweight="bold")
    ax.set_xlabel("Behavior"); ax.set_ylabel("Mean Jitter (px)  ↓ lower is better")
    ax.set_xticks(x); ax.set_xticklabels(bnames)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir / "jitter_comparison.png")


def plot_missing_ratio_comparison(fig_dir, class_old, class_new, agg_old, agg_new):
    bnames = [b.capitalize() for b in BEHAVIOR_CLASSES]
    x = np.arange(len(BEHAVIOR_CLASSES))
    w = 0.35
    old_v = [class_old[b]["missing_ratio"]*100 for b in BEHAVIOR_CLASSES]
    new_v = [class_new[b]["missing_ratio"]*100 for b in BEHAVIOR_CLASSES]
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, old_v, w, label="Old Model", color=COLOR_OLD, alpha=0.85)
    b2 = ax.bar(x + w/2, new_v, w, label="New Model", color=COLOR_NEW, alpha=0.85)
    ax.set_title("Missing Detection Ratio by Behavior", fontsize=14, fontweight="bold")
    ax.set_xlabel("Behavior"); ax.set_ylabel("Missing Ratio (%)  ↓ lower is better")
    ax.set_xticks(x); ax.set_xticklabels(bnames)
    top = max(max(old_v, default=0), max(new_v, default=0))
    ax.set_ylim(0, top * 1.35 + 5)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir / "missing_ratio_comparison.png")


def plot_17_keypoint_heatmap(fig_dir, agg_old, agg_new):
    x = np.arange(17)
    conf_o = np.array(agg_old["kpt_mean_confidence"])
    conf_n = np.array(agg_new["kpt_mean_confidence"])
    jit_o  = np.array(agg_old["kpt_mean_jitter"])
    jit_n  = np.array(agg_new["kpt_mean_jitter"])
    miss_o = np.array(agg_old["kpt_missing_ratio"]) * 100
    miss_n = np.array(agg_new["kpt_missing_ratio"]) * 100
    conf_imp = np.where(np.abs(conf_o) > 1e-9, (conf_n - conf_o) / np.abs(conf_o) * 100, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("17-Keypoint Analysis", fontsize=14, fontweight="bold")
    short = [n.replace("_", "\n") for n in KEYPOINT_NAMES]
    kw = dict(rotation=45, ha="right", fontsize=6.5)

    ax = axes[0, 0]
    ax.bar(x-0.2, conf_o, 0.4, label="Old", color=COLOR_OLD, alpha=0.85)
    ax.bar(x+0.2, conf_n, 0.4, label="New", color=COLOR_NEW, alpha=0.85)
    ax.set_title("Mean Confidence per Keypoint"); ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(short, **kw)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x-0.2, jit_o, 0.4, label="Old", color=COLOR_OLD, alpha=0.85)
    ax.bar(x+0.2, jit_n, 0.4, label="New", color=COLOR_NEW, alpha=0.85)
    ax.set_title("Mean Jitter per Keypoint (px)")
    ax.set_xticks(x); ax.set_xticklabels(short, **kw)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    data = np.vstack([miss_o, miss_n])
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Old", "New"])
    ax.set_xticks(x); ax.set_xticklabels(short, **kw)
    ax.set_title("Missing Ratio Heatmap (%)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in conf_imp]
    ax.bar(x, conf_imp, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Confidence Improvement per Keypoint (%)")
    ax.set_xticks(x); ax.set_xticklabels(short, **kw)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir / "17_keypoint_heatmap.png")


def plot_tail_paw_analysis(fig_dir, agg_old, agg_new):
    indices = [0, 7, 9, 11, 13, 14, 15, 16]
    labels  = ["Nose\n#0","LF_Paw\n#7","RF_Paw\n#9","LH_Paw\n#11","RH_Paw\n#13",
               "Tail_Root\n#14","Tail_Mid\n#15","Tail_Tip\n#16"]
    x = np.arange(len(indices)); w = 0.35

    old_miss = [agg_old["kpt_missing_ratio"][i]*100 for i in indices]
    new_miss = [agg_new["kpt_missing_ratio"][i]*100 for i in indices]
    old_conf = [agg_old["kpt_mean_confidence"][i] for i in indices]
    new_conf = [agg_new["kpt_mean_confidence"][i] for i in indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Tail & Paw Keypoint Analysis", fontsize=14, fontweight="bold")

    b1 = ax1.bar(x-w/2, old_miss, w, label="Old", color=COLOR_OLD, alpha=0.85)
    b2 = ax1.bar(x+w/2, new_miss, w, label="New", color=COLOR_NEW, alpha=0.85)
    ax1.set_title("Missing Ratio (%)"); ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Missing (%)"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(list(b1)+list(b2), old_miss+new_miss):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    b3 = ax2.bar(x-w/2, old_conf, w, label="Old", color=COLOR_OLD, alpha=0.85)
    b4 = ax2.bar(x+w/2, new_conf, w, label="New", color=COLOR_NEW, alpha=0.85)
    ax2.set_title("Mean Confidence"); ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Confidence"); ax2.set_ylim(0, 1.15); ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, fig_dir / "tail_paw_analysis.png")


def plot_overall_score_radar(fig_dir, agg_old, agg_new):
    categories = ["Confidence", "Jitter", "Missing", "Nose Det.", "Paw Det.", "Tail Det."]
    imps = [
        improvement(agg_old["mean_confidence"],      agg_new["mean_confidence"],      True),
        improvement(agg_old["mean_jitter_px"],       agg_new["mean_jitter_px"],       False),
        improvement(agg_old["missing_ratio"],        agg_new["missing_ratio"],        False),
        improvement(agg_old["nose_missing_ratio"],   agg_new["nose_missing_ratio"],   False),
        improvement(agg_old["paw_missing_ratio"],    agg_new["paw_missing_ratio"],    False),
        improvement(agg_old["tail_missing_ratio"],   agg_new["tail_missing_ratio"],   False),
    ]
    new_vals = [min(100, max(0, 50 + imp/2)) for imp in imps]
    old_vals = [50.0] * len(categories)

    angles = [n / len(categories) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    old_vals += old_vals[:1]
    new_vals += new_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, old_vals, "o-", lw=2, label="Old Model", color=COLOR_OLD)
    ax.fill(angles, old_vals, alpha=0.15, color=COLOR_OLD)
    ax.plot(angles, new_vals, "o-", lw=2, label="New Model", color=COLOR_NEW)
    ax.fill(angles, new_vals, alpha=0.15, color=COLOR_NEW)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_title("Overall Performance Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    _save(fig, fig_dir / "overall_score_radar.png")


def plot_improvement_percent(fig_dir, class_old, class_new):
    bnames = [b.capitalize() for b in BEHAVIOR_CLASSES]
    x = np.arange(len(BEHAVIOR_CLASSES)); w = 0.25
    ci = [improvement(class_old[b]["mean_confidence"], class_new[b]["mean_confidence"], True)  for b in BEHAVIOR_CLASSES]
    ji = [improvement(class_old[b]["mean_jitter_px"],  class_new[b]["mean_jitter_px"],  False) for b in BEHAVIOR_CLASSES]
    mi = [improvement(class_old[b]["missing_ratio"],   class_new[b]["missing_ratio"],   False) for b in BEHAVIOR_CLASSES]

    fig, ax = plt.subplots(figsize=(12, 7))
    b1 = ax.bar(x-w, ci, w, label="Confidence Imp%", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,   ji, w, label="Jitter Imp%",     color="#FF9800", alpha=0.85)
    b3 = ax.bar(x+w, mi, w, label="Missing Imp%",    color="#4CAF50", alpha=0.85)
    ax.set_title("Improvement % per Behavior Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Behavior"); ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x); ax.set_xticklabels(bnames)
    ax.axhline(0, color="black", lw=0.8); ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+(0.3 if h >= 0 else -1.8),
                    f"{h:+.1f}%", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir / "improvement_percent.png")


# ─── Text Reports ─────────────────────────────────────────────────
def write_model_report_txt(report_dir, old_name, new_name, total_videos,
                            agg_old, agg_new, class_old, class_new,
                            overall_score, old_comp, new_comp):
    ci  = improvement(agg_old["mean_confidence"],        agg_new["mean_confidence"],        True)
    ji  = improvement(agg_old["mean_jitter_px"],         agg_new["mean_jitter_px"],         False)
    mi  = improvement(agg_old["missing_ratio"],          agg_new["missing_ratio"],          False)
    tti = improvement(agg_old["tail_tip_missing_ratio"], agg_new["tail_tip_missing_ratio"], False)
    p95i = improvement(agg_old["p95_jitter_px"],         agg_new["p95_jitter_px"],          False)
    best_b, best_s, worst_b, worst_s = best_worst_behavior(class_old, class_new)
    verdict = "NEW MODEL IS BETTER" if overall_score > 0 else "NEW MODEL IS WORSE"
    rec = "✓ Replace old model." if overall_score > 0 else "✗ Keep old model."
    primary_verdict = ("NEW MODEL IS MORE STABLE" if agg_new["p95_jitter_px"] < agg_old["p95_jitter_px"]
                       else "OLD MODEL IS MORE STABLE")

    sep = "================================================="
    lines = [
        sep, "YOLO POSE MODEL COMPARISON REPORT", sep, "",
        "Old Model:", Path(old_name).name, "",
        "New Model:", Path(new_name).name, "",
        "Benchmark Videos:", str(total_videos), "",
        sep, "PRIMARY METRIC: P95 JITTER (tail/occasional-spike stability)", sep, "",
        "P95 Jitter:", f"{agg_old['p95_jitter_px']:.2f} px -> {agg_new['p95_jitter_px']:.2f} px", "",
        "Improvement:", f"{p95i:+.2f} %", "",
        "Primary Result:", primary_verdict, "",
        sep, "[REFERENCE] OVERALL COMPOSITE SCORE (not the primary metric)", sep, "",
        "Overall Score:", f"{overall_score:+.2f} %", "",
        "Result:", verdict, "",
        sep, "CONFIDENCE", sep, "",
        "Mean Confidence:", f"{agg_old['mean_confidence']:.3f} → {agg_new['mean_confidence']:.3f}", "",
        "Improvement:", f"{ci:+.2f} %", "",
        sep, "JITTER", sep, "",
        "Mean Jitter:", f"{agg_old['mean_jitter_px']:.2f} px → {agg_new['mean_jitter_px']:.2f} px", "",
        "Improvement:", f"{ji:+.2f} %", "",
        sep, "MISSING KEYPOINT", sep, "",
        "Missing Ratio:", f"{agg_old['missing_ratio']*100:.1f} % → {agg_new['missing_ratio']*100:.1f} %", "",
        "Improvement:", f"{mi:+.2f} %", "",
        sep, "BIGGEST IMPROVEMENT", sep, "",
        "Behavior:", best_b.capitalize(), "",
        "Improvement:", f"{best_s:+.1f} %", "",
        sep, "BIGGEST REGRESSION", sep, "",
        "Behavior:", worst_b.capitalize(), "",
        "Degradation:", f"{worst_s:+.1f} %", "",
        sep, "TAIL ANALYSIS", sep, "",
        "Tail Tip Missing:",
        f"{agg_old['tail_tip_missing_ratio']*100:.1f} % → {agg_new['tail_tip_missing_ratio']*100:.1f} %", "",
        "Improvement:", f"{tti:+.2f} %", "",
        sep, "RECOMMENDATION", sep, "",
        f"Primary metric (P95 jitter): {primary_verdict}", "",
        f"[Reference] Composite score: {rec}", "", sep,
    ]
    p = report_dir / "model_report.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_final_summary_log(log_dir, old_name, new_name, total_videos,
                             agg_old, agg_new, class_old, class_new,
                             overall_score, old_comp, new_comp):
    ci = improvement(agg_old["mean_confidence"], agg_new["mean_confidence"], True)
    ji = improvement(agg_old["mean_jitter_px"],  agg_new["mean_jitter_px"],  False)
    mi = improvement(agg_old["missing_ratio"],   agg_new["missing_ratio"],   False)
    p95i = improvement(agg_old["p95_jitter_px"], agg_new["p95_jitter_px"],  False)
    best_b, _, worst_b, _ = best_worst_behavior(class_old, class_new)
    sep = "================================================="
    lines = [
        sep, "FINAL SUMMARY", sep, "",
        "Videos Processed:", str(total_videos), "",
        "[PRIMARY] P95 Jitter Improvement:", f"{p95i:+.2f}%", "",
        "[PRIMARY] Decision:",
        "NEW MODEL WINS" if agg_new["p95_jitter_px"] < agg_old["p95_jitter_px"] else "OLD MODEL WINS", "",
        "[Reference] Old Model Composite Score:", f"{old_comp:.2f}", "",
        "[Reference] New Model Composite Score:", f"{new_comp:.2f}", "",
        "[Reference] Overall Improvement:", f"{overall_score:+.2f}%", "",
        "[Reference] Decision:", "NEW MODEL WINS" if overall_score > 0 else "OLD MODEL WINS", "",
        "Confidence:", f"{ci:+.2f}%", "",
        "Jitter (mean):", f"{ji:+.2f}%", "",
        "Missing Ratio:", f"{mi:+.2f}%", "",
        "Largest Improvement:", best_b.capitalize(), "",
        "Largest Regression:", worst_b.capitalize(), "",
        "Finished Successfully.", sep,
    ]
    p = log_dir / "final_summary.log"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ─── HTML Report ──────────────────────────────────────────────────
def write_html_report(report_dir, old_name, new_name, total_videos,
                       agg_old, agg_new, class_old, class_new,
                       overall_score, old_comp, new_comp):
    ci   = improvement(agg_old["mean_confidence"],        agg_new["mean_confidence"],        True)
    ji   = improvement(agg_old["mean_jitter_px"],         agg_new["mean_jitter_px"],         False)
    mi   = improvement(agg_old["missing_ratio"],          agg_new["missing_ratio"],          False)
    tti  = improvement(agg_old["tail_tip_missing_ratio"], agg_new["tail_tip_missing_ratio"], False)
    p95i = improvement(agg_old["p95_jitter_px"],          agg_new["p95_jitter_px"],          False)
    best_b, best_s, worst_b, worst_s = best_worst_behavior(class_old, class_new)
    is_better         = overall_score > 0                              # [參考] 複合分數
    primary_is_better = agg_new["p95_jitter_px"] < agg_old["p95_jitter_px"]  # 主指標：P95 Jitter
    vc = "#4CAF50" if primary_is_better else "#F44336"

    def badge(v):
        c = "#4CAF50" if v >= 0 else "#F44336"
        return f'<span style="color:{c};font-weight:bold">{v:+.2f}%</span>'

    def card(label, imp, old_s, new_s, cls=""):
        c = "#4CAF50" if imp >= 0 else "#F44336"
        return (f'<div class="card {cls}">'
                f'<div class="clabel">{label}</div>'
                f'<div class="cval" style="color:{c}">{imp:+.2f}%</div>'
                f'<div class="csub">{old_s} → {new_s}</div></div>')

    brows = ""
    for b in BEHAVIOR_CLASSES:
        o, n = class_old[b], class_new[b]
        _ci = improvement(o["mean_confidence"], n["mean_confidence"], True)
        _ji = improvement(o["mean_jitter_px"],  n["mean_jitter_px"],  False)
        _mi = improvement(o["missing_ratio"],   n["missing_ratio"],   False)
        _sc = W_CONFIDENCE*_ci + W_JITTER*_ji + W_MISSING*_mi
        brows += (f'<tr><td><b>{b.capitalize()}</b></td>'
                  f'<td>{o["mean_confidence"]:.3f}</td><td>{n["mean_confidence"]:.3f}</td><td>{badge(_ci)}</td>'
                  f'<td>{o["mean_jitter_px"]:.2f}</td><td>{n["mean_jitter_px"]:.2f}</td><td>{badge(_ji)}</td>'
                  f'<td>{o["missing_ratio"]*100:.1f}%</td><td>{n["missing_ratio"]*100:.1f}%</td><td>{badge(_mi)}</td>'
                  f'<td>{badge(_sc)}</td></tr>')

    def fimg(name, title):
        return (f'<div class="fb"><h3>{title}</h3>'
                f'<img src="../figures/{name}" alt="{title}" style="max-width:100%"></div>')

    old_pct = min(old_comp, 100)
    new_pct = min(new_comp, 100)

    rec_bg  = "#e8f5e9" if primary_is_better else "#ffebee"
    rec_msg = ("✅ <strong>依主指標（P95 抖動），Replace old model with new model.</strong>"
               if primary_is_better else
               "❌ <strong>依主指標（P95 抖動），Keep old model.</strong>")
    if primary_is_better != is_better:
        rec_msg += ("<br><span style='color:#F44336;font-size:14px'>⚠ 注意：[參考] 複合分數建議相反的結論"
                    f"（{'新模型較優' if is_better else '舊模型較優'}），"
                    "但混入了信心值/缺失率等你不特別在意的指標，請以主指標為準。</span>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YOLO Pose Model Comparison</title>
<style>
body{{font-family:'Segoe UI',sans-serif;margin:0;padding:20px;background:#f0f2f5;color:#333}}
.wrap{{max-width:1200px;margin:0 auto;background:#fff;padding:30px;border-radius:10px;box-shadow:0 2px 12px rgba(0,0,0,.1)}}
h1{{color:#333;border-bottom:3px solid {COLOR_OLD};padding-bottom:10px}}
h2{{color:{COLOR_OLD};margin-top:30px}}h3{{color:#555}}
.models{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}}
.mbox{{background:#f0f4ff;border-radius:8px;padding:16px}}
.mbox h3{{margin:0 0 8px}}
.bar-bg{{background:#ddd;border-radius:4px;height:22px;margin:6px 0}}
.bar-fill{{height:100%;border-radius:4px;display:flex;align-items:center;padding-left:8px;font-size:12px;color:#fff;font-weight:bold}}
.verdict{{text-align:center;padding:22px;margin:20px 0;border-radius:8px;font-size:22px;font-weight:bold;color:#fff;background:{vc}}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:20px 0}}
.card{{background:#f8f9fa;border-radius:8px;padding:18px;text-align:center;border-left:4px solid #4C72B0}}
.clabel{{font-size:13px;color:#666}}.cval{{font-size:26px;font-weight:bold;margin:8px 0}}.csub{{font-size:12px;color:#888}}
table{{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px}}
th{{background:{COLOR_OLD};color:#fff;padding:10px 8px;text-align:center}}
td{{padding:8px;border:1px solid #ddd;text-align:center}}tr:nth-child(even){{background:#f9f9f9}}
.fgrid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.fb{{margin:20px 0;text-align:center}}
</style>
</head>
<body>
<div class="wrap">
<h1>YOLO Pose Model Comparison Report</h1>

<div class="models">
  <div class="mbox" style="border-left:4px solid {COLOR_OLD}">
    <h3>Old Model</h3><code>{_html.escape(Path(old_name).name)}</code>
    <div style="font-size:20px;font-weight:bold;margin:8px 0">{agg_old['p95_jitter_px']:.2f} px</div>
    <div style="font-size:12px;color:#888">P95 Jitter（主指標，越小越好）</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{old_pct:.0f}%;background:{COLOR_OLD}">{old_comp:.1f}</div></div>
    <div style="font-size:12px;color:#888">[參考] Composite Score</div>
  </div>
  <div class="mbox" style="border-left:4px solid {COLOR_NEW}">
    <h3>New Model</h3><code>{_html.escape(Path(new_name).name)}</code>
    <div style="font-size:20px;font-weight:bold;margin:8px 0">{agg_new['p95_jitter_px']:.2f} px</div>
    <div style="font-size:12px;color:#888">P95 Jitter（主指標，越小越好）</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{new_pct:.0f}%;background:{COLOR_NEW}">{new_comp:.1f}</div></div>
    <div style="font-size:12px;color:#888">[參考] Composite Score</div>
  </div>
</div>

<div class="verdict">{"NEW MODEL IS MORE STABLE" if primary_is_better else "OLD MODEL IS MORE STABLE"}
<br><span style="font-size:16px">主指標：P95 Jitter Improvement {p95i:+.2f}%</span></div>

<h2>Key Metrics</h2>
<div class="cards">
{card("★ P95 Jitter (主指標) ↓ better", p95i, f"{agg_old['p95_jitter_px']:.2f}px", f"{agg_new['p95_jitter_px']:.2f}px")}
{card("[參考] Confidence", ci, f"{agg_old['mean_confidence']:.3f}", f"{agg_new['mean_confidence']:.3f}")}
{card("[參考] Mean Jitter  ↓ better", ji, f"{agg_old['mean_jitter_px']:.2f}px", f"{agg_new['mean_jitter_px']:.2f}px")}
{card("[參考] Missing Ratio  ↓ better", mi, f"{agg_old['missing_ratio']*100:.1f}%", f"{agg_new['missing_ratio']*100:.1f}%")}
{card("[參考] Tail Tip Missing  ↓ better", tti, f"{agg_old['tail_tip_missing_ratio']*100:.1f}%", f"{agg_new['tail_tip_missing_ratio']*100:.1f}%")}
{card("[參考] Best Behavior", best_s, best_b.capitalize(), "improved most")}
{card("[參考] Worst Behavior", worst_s, worst_b.capitalize(), "needs attention")}
</div>

<h2>Behavior Breakdown</h2>
<table>
<thead><tr>
<th>Behavior</th>
<th>Old Conf</th><th>New Conf</th><th>Conf Δ</th>
<th>Old Jitter</th><th>New Jitter</th><th>Jitter Δ</th>
<th>Old Miss</th><th>New Miss</th><th>Miss Δ</th>
<th>Score</th>
</tr></thead>
<tbody>{brows}</tbody>
</table>

<h2>Visualizations</h2>
<div class="fgrid">
{fimg("confidence_comparison.png","Confidence Comparison")}
{fimg("jitter_comparison.png","Jitter Comparison")}
</div>
<div class="fgrid">
{fimg("missing_ratio_comparison.png","Missing Ratio")}
{fimg("improvement_percent.png","Improvement %")}
</div>
{fimg("overall_score_radar.png","Performance Radar")}
{fimg("17_keypoint_heatmap.png","17-Keypoint Analysis")}
{fimg("tail_paw_analysis.png","Tail & Paw Analysis")}

<h2>Recommendation</h2>
<div style="background:{rec_bg};border-radius:8px;padding:20px;font-size:17px">{rec_msg}</div>

<p style="margin-top:30px;color:#aaa;font-size:12px">
Benchmark: {total_videos} videos | Weights: Conf {W_CONFIDENCE*100:.0f}% / Jitter {W_JITTER*100:.0f}% / Missing {W_MISSING*100:.0f}%
</p>
</div></body></html>"""

    p = report_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


# ─── Main ─────────────────────────────────────────────────────────
def main():
    old_path = OLD_MODEL_PATH
    new_path = NEW_MODEL_PATH
    videos   = BENCHMARK_DIR
    device   = INFERENCE_DEVICE

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out    = Path(OUTPUT_DIR) / run_ts
    dirs   = {k: out / k for k in ("csv", "figures", "logs", "report")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(dirs["logs"], run_ts)
    logger.info("=" * 52)
    logger.info(f"YOLO Pose Model Comparison  [{run_ts}]")
    logger.info(f"  Old : {old_path}")
    logger.info(f"  New : {new_path}")
    logger.info(f"  Data: {videos}")
    logger.info(f"  Out : {out}  Device: {device}")
    logger.info("=" * 52)

    # 路徑存在性檢查
    skip_old = not Path(old_path).exists()
    skip_new = not Path(new_path).exists()
    if skip_old:
        logger.warning(f"Old model not found, skipping: {old_path}")
    if skip_new:
        logger.warning(f"New model not found, skipping: {new_path}")
    if skip_old and skip_new:
        logger.error("Both model paths are missing. Nothing to compare.")
        sys.exit(1)

    if not Path(videos).exists():
        logger.error(f"BENCHMARK_DIR not found: {videos}")
        sys.exit(1)

    # Load models
    old_model = new_model = None
    if not skip_old:
        logger.info("Loading old model...")
        try:
            old_model = load_yolo_model(old_path, device)
        except Exception as e:
            logger.warning(f"Failed to load old model ({e}), skipping.")
            skip_old = True

    if not skip_new:
        logger.info("Loading new model...")
        try:
            new_model = load_yolo_model(new_path, device)
        except Exception as e:
            logger.warning(f"Failed to load new model ({e}), skipping.")
            skip_new = True

    if skip_old and skip_new:
        logger.error("Both models failed to load. Exiting.")
        sys.exit(1)

    # Discover videos
    behavior_videos = resolve_videos(Path(videos))
    total_videos    = sum(len(v) for v in behavior_videos.values())
    logger.info(f"Benchmark videos: {total_videos}")
    for b, vids in behavior_videos.items():
        logger.info(f"  {b}: {len(vids)}")

    if total_videos == 0:
        logger.error("No videos found. Check BENCHMARK_DIR and subfolder structure.")
        sys.exit(1)

    # Run inference for both models
    store = {
        "old": {b: [] for b in BEHAVIOR_CLASSES},
        "new": {b: [] for b in BEHAVIOR_CLASSES},
        "videos": behavior_videos,
    }

    for mk, model in (("old", old_model), ("new", new_model)):
        mpath    = old_path if mk == "old" else new_path
        ema_alpha = EMA_ALPHA_OLD if mk == "old" else EMA_ALPHA_NEW
        if model is None:
            logger.warning(f"跳過 {mk} model（路徑不存在或載入失敗）: {mpath}")
            for b in BEHAVIOR_CLASSES:
                store[mk][b] = [None] * len(behavior_videos[b])
            continue
        mname = Path(mpath).name
        logger.info(f"\n--- {mk.upper()} MODEL: {mname}  (EMA α={ema_alpha}) ---")
        for b in BEHAVIOR_CLASSES:
            for vpath in behavior_videos[b]:
                logger.info(f"Processing: {b}/{vpath.name}")
                t0 = time.time()
                r  = process_video(model, vpath, logger, ema_alpha=ema_alpha)
                elapsed = time.time() - t0
                if r:
                    logger.info(
                        f"  Frames: {r['frames']} | Inference Time: {elapsed:.2f} sec | "
                        f"Confidence={r['mean_confidence']:.3f} Jitter={r['mean_jitter_px']:.2f}"
                    )
                else:
                    logger.warning(f"  FAILED: {vpath}")
                store[mk][b].append(r)
                logger.info("-" * 32)

    # Aggregate
    logger.info("\nAggregating metrics...")
    class_old = {b: aggregate([r for r in store["old"][b] if r]) for b in BEHAVIOR_CLASSES}
    class_new = {b: aggregate([r for r in store["new"][b] if r]) for b in BEHAVIOR_CLASSES}
    agg_old   = aggregate([r for b in BEHAVIOR_CLASSES for r in store["old"][b] if r])
    agg_new   = aggregate([r for b in BEHAVIOR_CLASSES for r in store["new"][b] if r])

    old_comp = composite_score(agg_old)
    new_comp = composite_score(agg_new)
    ci = improvement(agg_old["mean_confidence"], agg_new["mean_confidence"], True)
    ji = improvement(agg_old["mean_jitter_px"],  agg_new["mean_jitter_px"],  False)
    mi = improvement(agg_old["missing_ratio"],   agg_new["missing_ratio"],   False)
    overall = W_CONFIDENCE * ci + W_JITTER * ji + W_MISSING * mi

    # CSV
    logger.info("\nWriting CSV files...")
    write_video_metrics_csv(dirs["csv"], store)
    write_keypoint_metrics_csv(dirs["csv"], agg_old, agg_new)
    write_class_summary_csv(dirs["csv"], class_old, class_new)
    write_improvement_summary_csv(dirs["csv"], agg_old, agg_new)

    # Figures
    logger.info("Generating figures...")
    plot_confidence_comparison(dirs["figures"], class_old, class_new, agg_old, agg_new)
    plot_jitter_comparison(dirs["figures"], class_old, class_new, agg_old, agg_new)
    plot_missing_ratio_comparison(dirs["figures"], class_old, class_new, agg_old, agg_new)
    plot_17_keypoint_heatmap(dirs["figures"], agg_old, agg_new)
    plot_tail_paw_analysis(dirs["figures"], agg_old, agg_new)
    plot_overall_score_radar(dirs["figures"], agg_old, agg_new)
    plot_improvement_percent(dirs["figures"], class_old, class_new)

    # Reports
    logger.info("Writing reports...")
    write_model_report_txt(dirs["report"], old_path, new_path, total_videos,
                           agg_old, agg_new, class_old, class_new,
                           overall, old_comp, new_comp)
    write_final_summary_log(dirs["logs"], old_path, new_path, total_videos,
                            agg_old, agg_new, class_old, class_new,
                            overall, old_comp, new_comp)
    write_html_report(dirs["report"], old_path, new_path, total_videos,
                      agg_old, agg_new, class_old, class_new,
                      overall, old_comp, new_comp)

    # ── 逐影片對比表 ──────────────────────────────────────────────
    # 欄位視覺寬度定義
    W_BEH  = 10   # 行為
    W_VID  = 32   # 影片名稱
    W_CONF = 8    # 信心值
    W_CDEL = 10   # 信心 Δ
    W_JIT  = 8    # 抖動
    W_JDEL = 10   # 抖動 Δ
    SEP1 = "=" * (W_BEH+1 + W_VID+1 + W_CONF+1 + W_CONF+1 + W_CDEL+1
                  + W_JIT+1 + W_JIT+1 + W_JDEL)

    logger.info("")
    logger.info(SEP1)
    logger.info("逐影片比較結果")
    logger.info(SEP1)
    logger.info(
        _vlj("行為",   W_BEH)  + " " +
        _vlj("影片",   W_VID)  + " " +
        _vrj("舊信心", W_CONF) + " " +
        _vrj("新信心", W_CONF) + " " +
        _vrj("信心Δ",  W_CDEL) + " " +
        _vrj("舊抖動", W_JIT)  + " " +
        _vrj("新抖動", W_JIT)  + " " +
        _vrj("抖動Δ",  W_JDEL)
    )
    logger.info("-" * len(SEP1))
    for b in BEHAVIOR_CLASSES:
        for vp, ro, rn in zip(store["videos"][b], store["old"][b], store["new"][b]):
            if ro is None or rn is None:
                logger.info(_vlj(b, W_BEH) + " " + _vlj(vp.name, W_VID) + "  (資料缺失，略過)")
                continue
            d_conf   = rn["mean_confidence"] - ro["mean_confidence"]
            d_jitter = rn["mean_jitter_px"]  - ro["mean_jitter_px"]
            logger.info(
                _vlj(b,        W_BEH)  + " " +
                _vlj(vp.name,  W_VID)  + " " +
                _vrj(f"{ro['mean_confidence']:.3f}",  W_CONF) + " " +
                _vrj(f"{rn['mean_confidence']:.3f}",  W_CONF) + " " +
                _vrj(f"{d_conf:+.3f}{'↑' if d_conf >= 0 else '↓'}",   W_CDEL) + " " +
                _vrj(f"{ro['mean_jitter_px']:.2f}",   W_JIT)  + " " +
                _vrj(f"{rn['mean_jitter_px']:.2f}",   W_JIT)  + " " +
                _vrj(f"{d_jitter:+.2f}{'↓' if d_jitter <= 0 else '↑'}", W_JDEL)
            )

    # ── 行為類別摘要 ───────────────────────────────────────────────
    W_B2   = 10   # 行為
    W_CV   = 8    # 信心 / 抖動 數值
    W_IMP  = 10   # 改善 %
    W_MV   = 8    # 缺失率數值
    SEP2 = "=" * (W_B2+1 + W_CV+1 + W_CV+1 + W_IMP+1
                  + W_CV+1 + W_CV+1 + W_IMP+1
                  + W_MV+1 + W_MV+1 + W_IMP)

    logger.info("")
    logger.info(SEP2)
    logger.info("行為類別摘要")
    logger.info(SEP2)
    logger.info(
        _vlj("行為",   W_B2)  + " " +
        _vrj("舊信心", W_CV)  + " " +
        _vrj("新信心", W_CV)  + " " +
        _vrj("信心改善", W_IMP) + " " +
        _vrj("舊抖動", W_CV)  + " " +
        _vrj("新抖動", W_CV)  + " " +
        _vrj("抖動改善", W_IMP) + " " +
        _vrj("舊缺失", W_MV)  + " " +
        _vrj("新缺失", W_MV)  + " " +
        _vrj("缺失改善", W_IMP)
    )
    logger.info("-" * len(SEP2))
    for b in BEHAVIOR_CLASSES:
        o, n = class_old[b], class_new[b]
        _ci = improvement(o["mean_confidence"], n["mean_confidence"], True)
        _ji = improvement(o["mean_jitter_px"],  n["mean_jitter_px"],  False)
        _mi = improvement(o["missing_ratio"],   n["missing_ratio"],   False)
        logger.info(
            _vlj(b,  W_B2)  + " " +
            _vrj(f"{o['mean_confidence']:.3f}",    W_CV)  + " " +
            _vrj(f"{n['mean_confidence']:.3f}",    W_CV)  + " " +
            _vrj(f"{_ci:+.2f}%",                   W_IMP) + " " +
            _vrj(f"{o['mean_jitter_px']:.2f}",     W_CV)  + " " +
            _vrj(f"{n['mean_jitter_px']:.2f}",     W_CV)  + " " +
            _vrj(f"{_ji:+.2f}%",                   W_IMP) + " " +
            _vrj(f"{o['missing_ratio']*100:.1f}%", W_MV)  + " " +
            _vrj(f"{n['missing_ratio']*100:.1f}%", W_MV)  + " " +
            _vrj(f"{_mi:+.2f}%",                   W_IMP)
        )

    # ── 主指標：P95 Jitter（尾部抖動/偶發大跳動，你指定在意的重點）───────────
    best_b, best_s, worst_b, worst_s = best_worst_behavior(class_old, class_new)
    SEP3 = "=" * len(SEP2)

    p95_o, p95_n = agg_old["p95_jitter_px"], agg_new["p95_jitter_px"]
    p95_imp = improvement(p95_o, p95_n, higher_is_better=False)
    primary_diff = abs(p95_o - p95_n)
    if primary_diff < 0.05:
        primary_winner = "="
    else:
        primary_winner = Path(new_path).name if p95_n < p95_o else Path(old_path).name

    logger.info("")
    logger.info(SEP3)
    logger.info("★★★ 主指標：P95 Jitter（尾部抖動，數值越小越穩定） ★★★")
    logger.info(SEP3)
    logger.info(
        _vlj("行為", W_B2) + " " +
        _vrj("舊P95抖動", W_CV) + " " +
        _vrj("新P95抖動", W_CV) + " " +
        _vrj("改善", W_IMP)
    )
    logger.info("-" * len(SEP2))
    for b in BEHAVIOR_CLASSES:
        o, n = class_old[b], class_new[b]
        _p95i = improvement(o["p95_jitter_px"], n["p95_jitter_px"], False)
        logger.info(
            _vlj(b, W_B2) + " " +
            _vrj(f"{o['p95_jitter_px']:.2f}", W_CV) + " " +
            _vrj(f"{n['p95_jitter_px']:.2f}", W_CV) + " " +
            _vrj(f"{_p95i:+.2f}%", W_IMP)
        )
    logger.info("-" * len(SEP2))
    logger.info(
        _vlj("整體", W_B2) + " " +
        _vrj(f"{p95_o:.2f}", W_CV) + " " +
        _vrj(f"{p95_n:.2f}", W_CV) + " " +
        _vrj(f"{p95_imp:+.2f}%", W_IMP)
    )
    logger.info("")
    if primary_winner == "=":
        logger.info(f"  ★ 主指標結論：兩模型 P95 抖動相近（Δ{primary_diff:.2f}px）")
    else:
        logger.info(f"  ★ 主指標結論：{primary_winner} 尾部抖動較小（Δ{primary_diff:.2f}px）"
                    f"— 依你指定的主指標，優先選它")
    logger.info(SEP3)

    # ── [參考，非主指標] 整體比較結果（複合分數：信心/平均抖動/缺失率混合） ──
    logger.info("")
    logger.info(SEP3)
    logger.info("[參考，非主指標] 整體比較結果（複合分數）")
    logger.info(SEP3)

    W_LBL = 14  # 標籤視覺寬度
    logger.info(_vlj("舊模型", W_LBL) + f"{Path(old_path).name}   複合分數 {old_comp:.2f}")
    logger.info(_vlj("新模型", W_LBL) + f"{Path(new_path).name}   複合分數 {new_comp:.2f}")
    logger.info("")
    logger.info(
        _vlj("平均信心值", W_LBL) +
        f"{agg_old['mean_confidence']:.3f}  →  {agg_new['mean_confidence']:.3f}"
        f"    改善 {ci:+.2f}%"
    )
    logger.info(
        _vlj("平均抖動", W_LBL) +
        f"{agg_old['mean_jitter_px']:.2f} px  →  {agg_new['mean_jitter_px']:.2f} px"
        f"    改善 {ji:+.2f}%"
    )
    logger.info(
        _vlj("缺失比率", W_LBL) +
        f"{agg_old['missing_ratio']*100:.1f}%  →  {agg_new['missing_ratio']*100:.1f}%"
        f"    改善 {mi:+.2f}%"
    )
    logger.info(
        _vlj("尾尖缺失", W_LBL) +
        f"{agg_old['tail_tip_missing_ratio']*100:.1f}%  →  {agg_new['tail_tip_missing_ratio']*100:.1f}%"
    )
    logger.info("")
    logger.info(_vlj("綜合改善分數", W_LBL) + f"{overall:+.2f}%")
    logger.info(_vlj("進步最多行為", W_LBL) + f"{best_b}  （{best_s:+.1f}%）")
    logger.info(_vlj("退步最多行為", W_LBL) + f"{worst_b}  （{worst_s:+.1f}%）")
    logger.info("")
    verdict = "BETTER（較優）" if overall > 0 else "WORSE（較差）"
    composite_winner = Path(new_path).name if overall > 0 else (Path(old_path).name if overall < 0 else "=")
    logger.info(_vlj("結論", W_LBL) + f"{Path(new_path).name}  相較  {Path(old_path).name}  →  {verdict}  [參考用複合分數]")
    logger.info(SEP3)

    if primary_winner != "=" and composite_winner != "=" and primary_winner != composite_winner:
        logger.info("")
        logger.info(f"⚠ 注意：[參考] 複合分數建議 {composite_winner}，但主指標（P95抖動）建議 {primary_winner}"
                    f"——兩者不一致時請以主指標為準（複合分數混入了信心值/缺失率等你不特別在意的項目）。")


if __name__ == "__main__":
    main()
