import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None


# ==================== 基本設定 ====================
IMGSZ = 640
CONF_THRES = 0.50
KP_CONF_THRES = 0.0
DEVIATION_THRES = 0.60
TOTAL_KPTS = 17
TARGET_MODEL_FPS = 30.0

DEFAULT_MODEL_1 = r"C:\ai_project\cat_pose\v11s_90.pt"
DEFAULT_MODEL_2 = r"C:\ai_project\cat_pose\v11s_91.pt"
DEFAULT_VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk\walk1.mp4"

KEYPOINT_NAMES = [
    "nose", "left_ear", "right_ear",
    "chest", "mid_back", "hip",
    "left_front_upper", "left_front_lower",
    "right_front_upper", "right_front_lower",
    "left_hind_upper", "left_hind_lower",
    "right_hind_upper", "right_hind_lower",
    "tail_base", "tail_mid", "tail_tip",
]

# 複合評分權重（權重總和 = 1.0）
SCORE_WEIGHTS = {
    "global_jitter":   0.35,   # 主要穩定性指標
    "jitter_p95":      0.20,   # 極端值抖動
    "unstable_rate":   0.20,   # 不穩定幀比率
    "jitter_std":      0.10,   # 抖動一致性
    "detection_rate":  0.10,   # 偵測覆蓋率（higher=better）
    "mean_confidence": 0.05,   # 信心值（higher=better）
}


# ==================== 資料結構 ====================
@dataclass
class ModelCompareResult:
    model_name: str
    model_path: str
    video_name: str
    video_path: str
    device: str
    source_fps: float
    frame_step: int
    sampled_frames: int
    detected_frames: int
    compared_frames: int
    missed_frames: int
    detection_rate: float
    global_jitter: float
    jitter_std: float
    jitter_p95: float
    unstable_rate: float
    mean_active_kpts: float
    mean_confidence: float
    worst_keypoint_index: int
    worst_keypoint_name: str
    worst_keypoint_jitter: float
    per_keypoint_mean_jitter: List[float]
    frame_jitters: List[float]


class ModelJitterStats:
    def __init__(self, model_path: str, video_path: str, device: str, source_fps: float, frame_step: int):
        self.model_path = model_path
        self.video_path = video_path
        self.device = device
        self.source_fps = source_fps
        self.frame_step = frame_step

        self.sampled_frames = 0
        self.detected_frames = 0
        self.compared_frames = 0
        self.missed_frames = 0

        self.active_kpts_sum = 0.0
        self.conf_sum = 0.0
        self.conf_count = 0

        self.frame_jitters: List[float] = []
        self.unstable_frames = 0
        self.per_kpt_jitters: List[List[float]] = [[] for _ in range(TOTAL_KPTS)]

    def add_frame_metrics(
        self,
        kpts: np.ndarray,
        kpt_conf: np.ndarray,
        prev_kpts: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, bool]:
        self.sampled_frames += 1
        self.detected_frames += 1

        active_mask = kpt_conf > KP_CONF_THRES
        self.active_kpts_sum += float(np.sum(active_mask))
        self.conf_sum += float(np.sum(kpt_conf))
        self.conf_count += int(len(kpt_conf))

        if prev_kpts is None:
            return kpts.copy(), False

        body_scale = float(np.linalg.norm(kpts[3] - kpts[5]))
        if body_scale < 1e-6:
            return kpts.copy(), False

        valid = active_mask
        if not np.any(valid):
            return kpts.copy(), False

        diffs = np.linalg.norm(kpts - prev_kpts, axis=1)
        norm_diffs = diffs[valid] / body_scale

        frame_jitter = float(np.mean(norm_diffs))
        frame_peak = float(np.max(norm_diffs))
        unstable = frame_peak > DEVIATION_THRES

        self.compared_frames += 1
        self.frame_jitters.append(frame_jitter)
        if unstable:
            self.unstable_frames += 1

        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            self.per_kpt_jitters[idx].append(float(diffs[idx] / body_scale))

        return kpts.copy(), unstable

    def add_missed_frame(self) -> None:
        self.sampled_frames += 1
        self.missed_frames += 1

    def finalize(self) -> ModelCompareResult:
        per_kpt_means = [float(np.mean(values)) if values else 0.0 for values in self.per_kpt_jitters]
        worst_kpt_idx = int(np.argmax(per_kpt_means)) if any(per_kpt_means) else -1
        worst_kpt_name = KEYPOINT_NAMES[worst_kpt_idx] if worst_kpt_idx >= 0 else "N/A"
        worst_kpt_jitter = per_kpt_means[worst_kpt_idx] if worst_kpt_idx >= 0 else 0.0

        return ModelCompareResult(
            model_name=Path(self.model_path).stem,
            model_path=self.model_path,
            video_name=Path(self.video_path).name,
            video_path=self.video_path,
            device=self.device,
            source_fps=self.source_fps,
            frame_step=self.frame_step,
            sampled_frames=self.sampled_frames,
            detected_frames=self.detected_frames,
            compared_frames=self.compared_frames,
            missed_frames=self.missed_frames,
            detection_rate=(self.detected_frames / self.sampled_frames) if self.sampled_frames else 0.0,
            global_jitter=float(np.mean(self.frame_jitters)) if self.frame_jitters else 0.0,
            jitter_std=float(np.std(self.frame_jitters)) if self.frame_jitters else 0.0,
            jitter_p95=float(np.percentile(self.frame_jitters, 95)) if self.frame_jitters else 0.0,
            unstable_rate=(self.unstable_frames / self.compared_frames) if self.compared_frames else 0.0,
            mean_active_kpts=(self.active_kpts_sum / self.detected_frames) if self.detected_frames else 0.0,
            mean_confidence=(self.conf_sum / self.conf_count) if self.conf_count else 0.0,
            worst_keypoint_index=worst_kpt_idx,
            worst_keypoint_name=worst_kpt_name,
            worst_keypoint_jitter=worst_kpt_jitter,
            per_keypoint_mean_jitter=per_kpt_means,
            frame_jitters=list(self.frame_jitters),
        )


# ==================== 評分 ====================
def composite_score(result: ModelCompareResult, all_results: List[ModelCompareResult]) -> float:
    """
    加權複合評分：將各指標正規化到 [0,1] 後加權求和。
    分數越低越好（與 global_jitter 方向一致）。
    """
    def _norm(val: float, vals: List[float], lower_better: bool) -> float:
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-12:
            return 0.5
        n = (val - lo) / (hi - lo)   # 0=best for lower_better
        return n if lower_better else (1.0 - n)

    metrics_lower = ["global_jitter", "jitter_p95", "unstable_rate", "jitter_std"]
    metrics_higher = ["detection_rate", "mean_confidence"]

    score = 0.0
    for m in metrics_lower:
        vals = [getattr(r, m) for r in all_results]
        score += SCORE_WEIGHTS[m] * _norm(getattr(result, m), vals, lower_better=True)
    for m in metrics_higher:
        vals = [getattr(r, m) for r in all_results]
        score += SCORE_WEIGHTS[m] * _norm(getattr(result, m), vals, lower_better=False)
    return score


# ==================== 報告渲染工具 ====================
def _pad(s: str, width: int, align: str = "l") -> str:
    s = str(s)
    if align == "r":
        return s.rjust(width)
    if align == "c":
        return s.center(width)
    return s.ljust(width)


def _ascii_table(headers: List[str], rows: List[List[str]], aligns: Optional[List[str]] = None) -> str:
    """Generate a plain ASCII table with column alignment."""
    if aligns is None:
        aligns = ["l"] * len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    def _row_line(cells: List[str], alns: List[str]) -> str:
        parts = [" " + _pad(c, col_widths[i], alns[i]) + " " for i, c in enumerate(cells)]
        return "|" + "|".join(parts) + "|"

    lines = [sep, _row_line(headers, ["c"] * len(headers)), sep]
    for row in rows:
        lines.append(_row_line([str(c) for c in row], aligns))
    lines.append(sep)
    return "\n".join(lines)


def _bar(value: float, max_val: float, width: int = 20, char: str = "█") -> str:
    if max_val < 1e-12:
        return " " * width
    filled = int(round(value / max_val * width))
    filled = max(0, min(width, filled))
    return char * filled + "░" * (width - filled)


def _jitter_histogram(jitters: List[float], bins: int = 8, bar_width: int = 20) -> List[str]:
    """Return lines showing a horizontal histogram of jitter distribution."""
    if not jitters:
        return ["  (no data)"]
    arr = np.array(jitters)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if counts.max() > 0 else 1
    lines = []
    for i, count in enumerate(counts):
        label = f"[{edges[i]:.3f}-{edges[i+1]:.3f})"
        bar = _bar(count, max_count, bar_width)
        lines.append(f"  {label:>18}  {bar}  {count:>4} frames")
    return lines


def _verdict_symbol(verdict: str) -> str:
    return {"improved": "▲ better", "regressed": "▼ worse", "same": "= same"}.get(verdict, verdict)


def _delta_str(old: float, new: float, pct: Optional[float], lower_better: bool) -> str:
    diff = new - old
    pct_s = f"{pct:+.1f}%" if pct is not None else "N/A"
    if abs(diff) < 1e-12:
        icon = "="
    elif (diff < 0) == lower_better:
        icon = "▲"
    else:
        icon = "▼"
    return f"{icon} {pct_s}"


# ==================== 分析函數 ====================
def percent_change(old_value: float, new_value: float) -> Optional[float]:
    if abs(old_value) < 1e-12:
        return None
    return (new_value - old_value) / old_value * 100.0


def improvement_percent(old_value: float, new_value: float) -> Optional[float]:
    if abs(old_value) < 1e-12:
        return None
    return (old_value - new_value) / old_value * 100.0


def jitter_amplitude_percent(old_value: float, new_value: float) -> Optional[float]:
    return improvement_percent(old_value, new_value)


def build_old_new_analysis(old_result: ModelCompareResult, new_result: ModelCompareResult) -> Dict[str, object]:
    overall_metrics = [
        ("global_jitter",  old_result.global_jitter,  new_result.global_jitter,  "lower_better"),
        ("jitter_std",     old_result.jitter_std,     new_result.jitter_std,     "lower_better"),
        ("jitter_p95",     old_result.jitter_p95,     new_result.jitter_p95,     "lower_better"),
        ("unstable_rate",  old_result.unstable_rate,  new_result.unstable_rate,  "lower_better"),
        ("detection_rate", old_result.detection_rate, new_result.detection_rate, "higher_better"),
        ("mean_confidence",old_result.mean_confidence,new_result.mean_confidence,"higher_better"),
    ]

    metric_rows: List[Dict[str, object]] = []
    for metric_name, old_value, new_value, better_direction in overall_metrics:
        diff = new_value - old_value
        pct = percent_change(old_value, new_value)
        if better_direction == "lower_better":
            verdict = "improved" if diff < 0 else ("regressed" if diff > 0 else "same")
        else:
            verdict = "improved" if diff > 0 else ("regressed" if diff < 0 else "same")
        metric_rows.append({
            "metric": metric_name,
            "old": old_value,
            "new": new_value,
            "difference": diff,
            "percent_change": pct,
            "better_direction": better_direction,
            "verdict": verdict,
        })

    per_keypoint_rows: List[Dict[str, object]] = []
    for idx, name in enumerate(KEYPOINT_NAMES):
        old_value = float(old_result.per_keypoint_mean_jitter[idx]) if idx < len(old_result.per_keypoint_mean_jitter) else 0.0
        new_value = float(new_result.per_keypoint_mean_jitter[idx]) if idx < len(new_result.per_keypoint_mean_jitter) else 0.0
        diff = new_value - old_value
        pct = percent_change(old_value, new_value)
        verdict = "improved" if diff < 0 else ("regressed" if diff > 0 else "same")
        per_keypoint_rows.append({
            "index": idx, "name": name,
            "old_jitter": old_value, "new_jitter": new_value,
            "difference": diff, "percent_change": pct, "verdict": verdict,
        })

    regressed_kpts = [r for r in per_keypoint_rows if r["difference"] > 0]
    improved_kpts  = [r for r in per_keypoint_rows if r["difference"] < 0]
    same_kpts      = [r for r in per_keypoint_rows if abs(float(r["difference"])) <= 1e-12]
    max_regression = max(regressed_kpts, key=lambda r: float(r["difference"]), default=None)
    max_improvement= min(improved_kpts,  key=lambda r: float(r["difference"]), default=None)

    return {
        "old_model": old_result.model_name,
        "new_model": new_result.model_name,
        "overall_metrics": metric_rows,
        "per_keypoint": per_keypoint_rows,
        "summary": {
            "regressed_keypoints": len(regressed_kpts),
            "improved_keypoints": len(improved_kpts),
            "unchanged_keypoints": len(same_kpts),
            "max_regression_keypoint": max_regression,
            "max_improvement_keypoint": max_improvement,
        },
    }


# ==================== 工具函數 ====================
def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {path}")
    return cap


def infer_frame_step(cap: cv2.VideoCapture, target_fps: float, forced_step: Optional[int]) -> Tuple[float, int, float]:
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap is not None else 0.0
    if source_fps <= 0:
        source_fps = target_fps
    if forced_step is not None and forced_step > 0:
        step = forced_step
    elif source_fps > target_fps:
        step = max(1, int(round(source_fps / target_fps)))
    else:
        step = 1
    effective_fps = source_fps / step if step > 0 else source_fps
    return source_fps, step, effective_fps


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("pose_jitter_compare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "comparison_report.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def analyze_model_on_video(
    model_path: str,
    video_path: str,
    device: str,
    target_fps: float,
    frame_step: Optional[int],
    logger: logging.Logger,
    csv_writer: csv.writer,
    run_tag: str,
) -> ModelCompareResult:
    logger.info("載入模型: %s", model_path)
    model = YOLO(model_path)

    cap = load_video(video_path)
    try:
        source_fps, step, effective_fps = infer_frame_step(cap, target_fps, frame_step)
        logger.info(
            "影片: %s | src_fps=%.2f | step=%d | eff_fps=%.2f",
            Path(video_path).name, source_fps, step, effective_fps,
        )

        stats = ModelJitterStats(model_path, video_path, device, source_fps, step)
        prev_kpts: Optional[np.ndarray] = None
        use_half = device.startswith("cuda")
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            for _ in range(max(0, step - 1)):
                if not cap.grab():
                    break

            result = model.predict(
                frame, imgsz=IMGSZ, conf=CONF_THRES,
                half=use_half, verbose=False, device=device,
            )[0]

            source_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            detected = result.keypoints is not None and len(result.keypoints.xy) > 0

            if detected:
                kpts = result.keypoints.xy[0].cpu().numpy()
                kpt_conf = (
                    result.keypoints.conf[0].cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones(TOTAL_KPTS, dtype=np.float32)
                )
                prev_kpts, unstable = stats.add_frame_metrics(kpts, kpt_conf, prev_kpts)
                csv_writer.writerow([
                    run_tag, Path(model_path).name, Path(video_path).name,
                    frame_index, source_frame_id,
                    f"{float(np.mean(kpt_conf)):.6f}",
                    int(np.sum(kpt_conf > KP_CONF_THRES)),
                    f"{stats.frame_jitters[-1]:.6f}" if stats.frame_jitters else "0.000000",
                    int(unstable),
                ])
            else:
                stats.add_missed_frame()
                prev_kpts = None
                csv_writer.writerow([
                    run_tag, Path(model_path).name, Path(video_path).name,
                    frame_index, source_frame_id, 0.0, 0, 0.0, 0,
                ])

            frame_index += 1
            if frame_index % 50 == 0:
                det_rate = stats.detected_frames / stats.sampled_frames * 100
                jitter_now = float(np.mean(stats.frame_jitters)) if stats.frame_jitters else 0.0
                logger.info(
                    "  [%s] frame=%d  det=%.1f%%  jitter_mean=%.4f",
                    Path(model_path).stem, frame_index, det_rate, jitter_now,
                )
    finally:
        cap.release()

    return stats.finalize()


def compare_results(results: List[ModelCompareResult]) -> Dict[str, str]:
    ranked = sorted(
        results,
        key=lambda r: (r.global_jitter, r.unstable_rate, -r.detection_rate, -r.mean_confidence),
    )
    best = ranked[0]
    worst = ranked[-1]
    return {
        "best_model": best.model_name,
        "best_model_path": best.model_path,
        "worst_model": worst.model_name,
        "worst_model_path": worst.model_path,
        "rank_rule": "global_jitter → unstable_rate → detection_rate → mean_confidence",
    }


# ==================== 報告生成 ====================
def _build_report_lines(
    results: List[ModelCompareResult],
    decision: Dict[str, str],
    old_new_analysis: Dict[str, object],
    scores: List[float],
    generated_at: str,
) -> List[str]:
    lines: List[str] = []
    W = 80
    RULE = "=" * W
    rule2 = "-" * W

    def section(title: str) -> None:
        lines.append(RULE)
        lines.append(f"  {title}")
        lines.append(RULE)

    def sub(title: str) -> None:
        lines.append("")
        lines.append(f"  ── {title}")
        lines.append(rule2)

    # ── Header ──────────────────────────────────────────────
    lines.append(RULE)
    lines.append(f"  YOLO POSE MODEL COMPARISON REPORT")
    lines.append(f"  Generated : {generated_at}")
    lines.append(f"  Video     : {results[0].video_name if results else 'N/A'}")
    lines.append(f"  Device    : {results[0].device if results else 'N/A'}")
    lines.append(f"  Rank rule : {decision['rank_rule']}")
    lines.append(RULE)
    lines.append("")

    # ── Composite score summary ──────────────────────────────
    section("1. COMPOSITE SCORE  (lower = better)")
    lines.append("")
    ranked_pairs = sorted(zip(scores, results), key=lambda x: x[0])
    score_rows = []
    for rank, (sc, r) in enumerate(ranked_pairs, 1):
        tag = "  ★ WINNER" if rank == 1 else ""
        score_rows.append([str(rank), r.model_name, f"{sc:.4f}", tag])
    lines.append(_ascii_table(
        ["Rank", "Model", "Composite Score", ""],
        score_rows,
        ["c", "l", "r", "l"],
    ))
    lines.append("")
    lines.append(f"  Weight formula:  global_jitter×{SCORE_WEIGHTS['global_jitter']}"
                 f"  jitter_p95×{SCORE_WEIGHTS['jitter_p95']}"
                 f"  unstable_rate×{SCORE_WEIGHTS['unstable_rate']}")
    lines.append(f"                   jitter_std×{SCORE_WEIGHTS['jitter_std']}"
                 f"  detection_rate×{SCORE_WEIGHTS['detection_rate']}"
                 f"  mean_confidence×{SCORE_WEIGHTS['mean_confidence']}")
    lines.append("")

    # ── Per-model statistics ─────────────────────────────────
    section("2. PER-MODEL STATISTICS")
    sorted_results = sorted(results, key=lambda r: r.global_jitter)

    stat_headers = ["Metric", "Unit"] + [r.model_name for r in sorted_results]
    stat_aligns  = ["l", "l"] + ["r"] * len(sorted_results)

    def _stat_row(label: str, unit: str, getter, fmt: str = ".6f") -> List[str]:
        vals = [f"{getter(r):{fmt}}" for r in sorted_results]
        return [label, unit] + vals

    stat_rows = [
        _stat_row("global_jitter",    "norm",  lambda r: r.global_jitter),
        _stat_row("jitter_std",       "norm",  lambda r: r.jitter_std),
        _stat_row("jitter_p95",       "norm",  lambda r: r.jitter_p95),
        _stat_row("unstable_rate",    "%",     lambda r: r.unstable_rate * 100, ".2f"),
        _stat_row("detection_rate",   "%",     lambda r: r.detection_rate * 100, ".2f"),
        _stat_row("mean_confidence",  "0-1",   lambda r: r.mean_confidence, ".4f"),
        _stat_row("mean_active_kpts", f"/{TOTAL_KPTS}", lambda r: r.mean_active_kpts, ".2f"),
        _stat_row("sampled_frames",   "frames",lambda r: r.sampled_frames, "d"),
        _stat_row("missed_frames",    "frames",lambda r: r.missed_frames, "d"),
    ]
    lines.append("")
    lines.append(_ascii_table(stat_headers, stat_rows, stat_aligns))
    lines.append("")

    # ── Old vs New diff table ────────────────────────────────
    if old_new_analysis and len(results) >= 2:
        old_name = str(old_new_analysis["old_model"])
        new_name = str(old_new_analysis["new_model"])

        section(f"3. HEAD-TO-HEAD DIFF  [{old_name}]  vs  [{new_name}]")
        lines.append("")

        sub("Overall metrics")
        diff_headers = ["Metric", "Better", old_name, new_name, "Δ abs", "Δ %", "Verdict"]
        diff_aligns  = ["l", "c", "r", "r", "r", "r", "c"]
        diff_rows = []
        for row in old_new_analysis.get("overall_metrics", []):
            if not isinstance(row, dict):
                continue
            m    = str(row["metric"])
            old_ = float(row["old"])
            new_ = float(row["new"])
            diff = float(row["difference"])
            pct  = row["percent_change"]
            lower_better = row["better_direction"] == "lower_better"
            better = "▼ low" if lower_better else "▲ high"
            verdict = _verdict_symbol(str(row["verdict"]))
            delta_s = _delta_str(old_, new_, pct if pct is None else float(pct), lower_better)
            diff_rows.append([m, better, f"{old_:.6f}", f"{new_:.6f}", f"{diff:+.6f}", delta_s, verdict])
        lines.append(_ascii_table(diff_headers, diff_rows, diff_aligns))
        lines.append("")

        sub("Per-keypoint jitter (new − old)")
        kpt_headers = ["#", "Keypoint", old_name, new_name, "Δ %", "Verdict", f"Bar ({new_name})"]
        kpt_aligns  = ["r", "l", "r", "r", "r", "c", "l"]
        kpt_rows = []
        max_new_jitter = max(
            (float(r["new_jitter"]) for r in old_new_analysis.get("per_keypoint", []) if isinstance(r, dict)),
            default=1.0,
        )
        for row in old_new_analysis.get("per_keypoint", []):
            if not isinstance(row, dict):
                continue
            old_ = float(row["old_jitter"])
            new_ = float(row["new_jitter"])
            pct  = row["percent_change"]
            delta_s = _delta_str(old_, new_, pct if pct is None else float(pct), lower_better=True)
            verdict = _verdict_symbol(str(row["verdict"]))
            bar = _bar(new_, max_new_jitter, width=14)
            kpt_rows.append([
                str(row["index"]), str(row["name"]),
                f"{old_:.4f}", f"{new_:.4f}", delta_s, verdict, bar,
            ])
        lines.append(_ascii_table(kpt_headers, kpt_rows, kpt_aligns))

        summ = old_new_analysis.get("summary", {})
        if isinstance(summ, dict):
            lines.append("")
            lines.append(
                f"  Keypoint summary:  improved={summ.get('improved_keypoints',0)}"
                f"  regressed={summ.get('regressed_keypoints',0)}"
                f"  unchanged={summ.get('unchanged_keypoints',0)}"
            )
            max_reg = summ.get("max_regression_keypoint")
            max_imp = summ.get("max_improvement_keypoint")
            if isinstance(max_reg, dict):
                pct = max_reg.get("percent_change")
                pct_s = f"{float(pct):+.2f}%" if pct is not None else "N/A"
                lines.append(f"  Worst regression : {max_reg.get('name','?')}  Δ={float(max_reg.get('difference',0)):+.4f}  ({pct_s})")
            if isinstance(max_imp, dict):
                pct = max_imp.get("percent_change")
                pct_s = f"{float(pct):+.2f}%" if pct is not None else "N/A"
                lines.append(f"  Best improvement : {max_imp.get('name','?')}  Δ={float(max_imp.get('difference',0)):+.4f}  ({pct_s})")
        lines.append("")

    # ── Jitter distribution histograms ───────────────────────
    section(f"{'4' if old_new_analysis else '3'}. JITTER DISTRIBUTION HISTOGRAM  (per-frame, normalised)")
    for r in results:
        lines.append("")
        lines.append(f"  Model: {r.model_name}  (compared_frames={r.compared_frames})")
        lines.append(f"  mean={r.global_jitter:.4f}  std={r.jitter_std:.4f}  p95={r.jitter_p95:.4f}  unstable_rate={r.unstable_rate*100:.1f}%")
        lines.extend(_jitter_histogram(r.frame_jitters, bins=8, bar_width=22))
    lines.append("")

    # ── Worst keypoints per model ────────────────────────────
    sec_n = 5 if old_new_analysis else 4
    section(f"{sec_n}. WORST KEYPOINTS PER MODEL")
    lines.append("")
    wk_headers = ["Model", "Rank", "Keypoint", "Mean Jitter", "Bar"]
    wk_aligns  = ["l", "r", "l", "r", "l"]
    wk_rows = []
    for r in sorted_results:
        kpt_pairs = sorted(enumerate(r.per_keypoint_mean_jitter), key=lambda x: x[1], reverse=True)
        top5 = kpt_pairs[:5]
        max_j = top5[0][1] if top5 else 1.0
        for rank, (idx, jitter) in enumerate(top5, 1):
            bar = _bar(jitter, max_j, width=16)
            wk_rows.append([r.model_name if rank == 1 else "", str(rank), KEYPOINT_NAMES[idx], f"{jitter:.4f}", bar])
        wk_rows.append(["", "", "", "", ""])
    lines.append(_ascii_table(wk_headers, wk_rows, wk_aligns))
    lines.append("")

    lines.append(RULE)
    lines.append(f"  CONCLUSION:  Best model → {decision['best_model']}")
    lines.append(RULE)
    return lines


def write_summary_files(
    output_dir: Path,
    results: List[ModelCompareResult],
    decision: Dict[str, str],
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    old_new_analysis: Dict[str, object] = {}
    if len(results) >= 2:
        old_new_analysis = build_old_new_analysis(results[0], results[1])

    scores = [composite_score(r, results) for r in results]

    # ── JSON ──────────────────────────────────────────────
    improve_pct = jitter_amplitude_percent(results[0].global_jitter, results[1].global_jitter) if len(results) >= 2 else None
    report_json = {
        "generated_at": generated_at,
        "rank_rule": decision["rank_rule"],
        "old_model": results[0].model_name if results else "N/A",
        "new_model": results[1].model_name if len(results) > 1 else "N/A",
        "best_model": decision["best_model"],
        "worst_model": decision["worst_model"],
        "headline_jitter_improvement_pct": improve_pct,
        "composite_scores": {r.model_name: sc for r, sc in zip(results, scores)},
        "score_weights": SCORE_WEIGHTS,
        "results": [asdict(r) for r in results],
        "old_vs_new_analysis": old_new_analysis,
    }
    with open(output_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    # ── TXT ───────────────────────────────────────────────
    report_lines = _build_report_lines(results, decision, old_new_analysis, scores, generated_at)
    with open(output_dir / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def print_console_report(
    results: List[ModelCompareResult],
    decision: Dict[str, str],
    scores: List[float],
) -> None:
    """Print a compact summary to stdout after all models finish."""
    W = 70
    print("\n" + "=" * W)
    print("  POSE MODEL COMPARISON  —  QUICK SUMMARY")
    print("=" * W)

    # Score table
    ranked = sorted(zip(scores, results), key=lambda x: x[0])
    print(f"\n  {'Rank':<5} {'Model':<20} {'Score':>8}  {'global_jitter':>14}  {'unstable%':>9}  {'det%':>6}")
    print("  " + "-" * 65)
    for rank, (sc, r) in enumerate(ranked, 1):
        star = "★" if rank == 1 else " "
        print(f"  {star}{rank:<4} {r.model_name:<20} {sc:>8.4f}  {r.global_jitter:>14.6f}  {r.unstable_rate*100:>8.1f}%  {r.detection_rate*100:>5.1f}%")

    if len(results) >= 2:
        old, new = results[0], results[1]
        imp = jitter_amplitude_percent(old.global_jitter, new.global_jitter)
        print(f"\n  Head-to-head jitter:  {old.model_name} → {new.model_name}")
        if imp is None:
            print("  Jitter improvement: N/A (old model jitter = 0)")
        elif imp >= 0:
            print(f"  Jitter improvement  : +{imp:.2f}%  (new model is BETTER)")
        else:
            print(f"  Jitter regression   : -{abs(imp):.2f}%  (old model was BETTER)")

    print(f"\n  Winner: {decision['best_model']}")
    print("=" * W + "\n")


# ==================== CLI ====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two YOLO pose models on the same video; report jitter metrics.")
    parser.add_argument("--model-1",     type=str,   default=DEFAULT_MODEL_1)
    parser.add_argument("--model-2",     type=str,   default=DEFAULT_MODEL_2)
    parser.add_argument("--video",       type=str,   default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output-dir",  type=str,   default=str(Path(__file__).parent / "jitter_compare_logs"))
    parser.add_argument("--frame-step",  type=int,   default=0,              help="0 = auto from FPS")
    parser.add_argument("--target-fps",  type=float, default=TARGET_MODEL_FPS)
    parser.add_argument("--device",      type=str,   default="auto",         help="auto / cpu / cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model_paths = [args.model_1, args.model_2]
    if not all(model_paths):
        raise RuntimeError("請提供兩個模型的路徑")

    video_path = str(Path(args.video))
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"找不到影片: {video_path}")
    for mp in model_paths:
        if not Path(mp).is_file():
            raise FileNotFoundError(f"找不到模型: {mp}")

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("=== 比較任務開始 ===")
    logger.info("輸出目錄: %s", output_dir)
    logger.info("裝置: %s | 影片: %s", device, Path(video_path).name)

    frame_metrics_path = output_dir / "frame_metrics.csv"
    with open(frame_metrics_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "run_tag", "model_name", "video_name",
            "sample_index", "source_frame_id",
            "mean_confidence", "active_kpts", "frame_jitter", "unstable",
        ])

        results: List[ModelCompareResult] = []
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        for index, model_path in enumerate(model_paths, start=1):
            logger.info("[%d/%d] %s", index, len(model_paths), Path(model_path).name)
            result = analyze_model_on_video(
                model_path=model_path,
                video_path=video_path,
                device=device,
                target_fps=args.target_fps,
                frame_step=args.frame_step if args.frame_step > 0 else None,
                logger=logger,
                csv_writer=writer,
                run_tag=run_tag,
            )
            results.append(result)
            logger.info(
                "  done | jitter=%.6f  std=%.6f  p95=%.6f  unstable=%.1f%%  det=%.1f%%",
                result.global_jitter, result.jitter_std, result.jitter_p95,
                result.unstable_rate * 100, result.detection_rate * 100,
            )

    decision = compare_results(results)
    scores = [composite_score(r, results) for r in results]
    write_summary_files(output_dir, results, decision)

    print_console_report(results, decision, scores)

    logger.info("=== 比較完成 ===")
    logger.info("較佳模型: %s  (score=%.4f)", decision["best_model"], min(scores))

    abs_dir = output_dir.resolve()
    print("\n" + "=" * 60)
    print(f"  輸出資料夾:  {abs_dir}")
    print(f"  TXT 報告  :  {abs_dir / 'comparison_report.txt'}")
    print(f"  JSON 報告 :  {abs_dir / 'comparison_report.json'}")
    print(f"  逐幀 CSV  :  {abs_dir / 'frame_metrics.csv'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
