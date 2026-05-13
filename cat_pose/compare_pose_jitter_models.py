from __future__ import annotations

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

# 直接填入兩個絕對路徑模型與一個絕對路徑影片
DEFAULT_MODEL_1 = r"C:\ai_project\cat_pose\v11s_70_1.pt"
DEFAULT_MODEL_2 = r"C:\ai_project\cat_pose\v11s_70_2.pt"
DEFAULT_VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\貓咪姿勢影片分類\暫存\walk\2752855.mp4"

KEYPOINT_NAMES = [
    "nose", "left_ear", "right_ear",
    "chest", "mid_back", "hip",
    "left_front_upper", "left_front_lower",
    "right_front_upper", "right_front_lower",
    "left_hind_upper", "left_hind_lower",
    "right_hind_upper", "right_hind_lower",
    "tail_base", "tail_mid", "tail_tip",
]


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
        )


def percent_change(old_value: float, new_value: float) -> Optional[float]:
    if abs(old_value) < 1e-12:
        return None
    return (new_value - old_value) / old_value * 100.0


def improvement_percent(old_value: float, new_value: float) -> Optional[float]:
    if abs(old_value) < 1e-12:
        return None
    return (old_value - new_value) / old_value * 100.0


def jitter_amplitude_percent(old_value: float, new_value: float) -> Optional[float]:
    """Return how much smaller the new jitter amplitude is compared with the old one."""
    return improvement_percent(old_value, new_value)


def build_old_new_analysis(old_result: ModelCompareResult, new_result: ModelCompareResult) -> Dict[str, object]:
    overall_metrics = [
        ("global_jitter", old_result.global_jitter, new_result.global_jitter, "lower_better"),
        ("jitter_std", old_result.jitter_std, new_result.jitter_std, "lower_better"),
        ("jitter_p95", old_result.jitter_p95, new_result.jitter_p95, "lower_better"),
        ("unstable_rate", old_result.unstable_rate, new_result.unstable_rate, "lower_better"),
        ("detection_rate", old_result.detection_rate, new_result.detection_rate, "higher_better"),
        ("mean_confidence", old_result.mean_confidence, new_result.mean_confidence, "higher_better"),
    ]

    metric_rows: List[Dict[str, object]] = []
    for metric_name, old_value, new_value, better_direction in overall_metrics:
        diff = new_value - old_value
        pct = percent_change(old_value, new_value)
        if better_direction == "lower_better":
            verdict = "improved" if diff < 0 else ("regressed" if diff > 0 else "same")
        else:
            verdict = "improved" if diff > 0 else ("regressed" if diff < 0 else "same")

        metric_rows.append(
            {
                "metric": metric_name,
                "old": old_value,
                "new": new_value,
                "difference": diff,
                "percent_change": pct,
                "better_direction": better_direction,
                "verdict": verdict,
            }
        )

    per_keypoint_rows: List[Dict[str, object]] = []
    old_per_kpt = old_result.per_keypoint_mean_jitter
    new_per_kpt = new_result.per_keypoint_mean_jitter

    for idx, name in enumerate(KEYPOINT_NAMES):
        old_value = float(old_per_kpt[idx]) if idx < len(old_per_kpt) else 0.0
        new_value = float(new_per_kpt[idx]) if idx < len(new_per_kpt) else 0.0
        diff = new_value - old_value
        pct = percent_change(old_value, new_value)
        verdict = "improved" if diff < 0 else ("regressed" if diff > 0 else "same")

        per_keypoint_rows.append(
            {
                "index": idx,
                "name": name,
                "old_jitter": old_value,
                "new_jitter": new_value,
                "difference": diff,
                "percent_change": pct,
                "verdict": verdict,
            }
        )

    regressed_kpts = [row for row in per_keypoint_rows if row["difference"] > 0]
    improved_kpts = [row for row in per_keypoint_rows if row["difference"] < 0]
    same_kpts = [row for row in per_keypoint_rows if abs(float(row["difference"])) <= 1e-12]

    max_regression = max(regressed_kpts, key=lambda row: float(row["difference"]), default=None)
    max_improvement = min(improved_kpts, key=lambda row: float(row["difference"]), default=None)

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
        logger.info("來源影片: %s | FPS=%.2f | 取樣步長=%d | 有效輸入FPS=%.2f", video_path, source_fps, step, effective_fps)

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
                frame,
                imgsz=IMGSZ,
                conf=CONF_THRES,
                half=use_half,
                verbose=False,
                device=device,
            )[0]

            source_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            detected = result.keypoints is not None and len(result.keypoints.xy) > 0

            if detected:
                kpts = result.keypoints.xy[0].cpu().numpy()
                if result.keypoints.conf is not None:
                    kpt_conf = result.keypoints.conf[0].cpu().numpy()
                else:
                    kpt_conf = np.ones(TOTAL_KPTS, dtype=np.float32)

                prev_kpts, unstable = stats.add_frame_metrics(kpts, kpt_conf, prev_kpts)

                csv_writer.writerow([
                    run_tag,
                    Path(model_path).name,
                    Path(video_path).name,
                    frame_index,
                    source_frame_id,
                    f"{float(np.mean(kpt_conf)):.6f}",
                    int(np.sum(kpt_conf > KP_CONF_THRES)),
                    f"{stats.frame_jitters[-1]:.6f}" if stats.frame_jitters else "0.000000",
                    int(unstable),
                ])
            else:
                stats.add_missed_frame()
                prev_kpts = None
                csv_writer.writerow([
                    run_tag,
                    Path(model_path).name,
                    Path(video_path).name,
                    frame_index,
                    source_frame_id,
                    0.0,
                    0,
                    0.0,
                    0,
                ])

            frame_index += 1
            if frame_index % 50 == 0:
                logger.info("%s | 已處理 %d 個取樣幀", Path(model_path).name, frame_index)

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
        "rank_rule": "global_jitter -> unstable_rate -> detection_rate -> mean_confidence",
    }


def write_summary_files(output_dir: Path, results: List[ModelCompareResult], decision: Dict[str, str]) -> None:
    old_new_analysis: Dict[str, object] = {}
    headline_message = ""
    winner_message = ""
    formula_message = ""
    if len(results) >= 2:
        old_new_analysis = build_old_new_analysis(results[0], results[1])
        improve_pct = jitter_amplitude_percent(results[0].global_jitter, results[1].global_jitter)
        formula_message = "抖動幅度定義: 以逐幀關鍵點位移歸一化後的平均值 global_jitter 表示；數值越小代表越穩定。"
        if improve_pct is None:
            headline_message = "新模型比舊模型變好 N/A（舊模型關鍵點抖動幅度為0，無法計算百分比）"
        elif improve_pct >= 0:
            headline_message = f"新模型比舊模型關鍵點抖動幅度變好 {improve_pct:.2f}%"
            winner_message = f"結果判定: 新模型較好（{results[1].model_name}）"
        else:
            headline_message = f"新模型比舊模型關鍵點抖動幅度變差 {abs(improve_pct):.2f}%"
            winner_message = f"結果判定: 舊模型較好（{results[0].model_name}）"

    report_json = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rank_rule": decision["rank_rule"],
        "old_model": results[0].model_name if results else "N/A",
        "new_model": results[1].model_name if len(results) > 1 else "N/A",
        "best_model": decision["best_model"],
        "worst_model": decision["worst_model"],
        "headline_message": headline_message,
        "winner_message": winner_message,
        "formula_message": formula_message,
        "results": [asdict(result) for result in results],
        "old_vs_new_analysis": old_new_analysis,
    }

    with open(output_dir / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append(f"產生時間: {report_json['generated_at']}")
    lines.append(f"比較規則: {decision['rank_rule']}")
    if formula_message:
        lines.append(f"計算方式: {formula_message}")
    lines.append(f"影片名稱: {results[0].video_name if results else 'N/A'}")
    if headline_message:
        lines.append(f"重點訊息: {headline_message}")
    if winner_message:
        lines.append(f"{winner_message}")
    lines.append("")

    for result in sorted(results, key=lambda r: (r.global_jitter, r.unstable_rate, -r.detection_rate)):
        lines.append(f"模型: {result.model_name}")
        lines.append(f"  路徑: {result.model_path}")
        lines.append(f"  全局抖動: {result.global_jitter:.6f}")
        lines.append(f"  抖動標準差: {result.jitter_std:.6f}")
        lines.append(f"  95百分位抖動: {result.jitter_p95:.6f}")
        lines.append(f"  不穩定率: {result.unstable_rate * 100:.2f}%")
        lines.append(f"  偵測率: {result.detection_rate * 100:.2f}%")
        lines.append(f"  平均可見關鍵點: {result.mean_active_kpts:.2f}/{TOTAL_KPTS}")
        lines.append(f"  平均信心值: {result.mean_confidence:.4f}")
        lines.append(f"  最不穩定關鍵點: {result.worst_keypoint_name} ({result.worst_keypoint_jitter:.6f})")
        lines.append("")

    lines.append(f"較佳模型: {decision['best_model']}")
    lines.append(f"較差模型: {decision['worst_model']}")

    if old_new_analysis:
        lines.append("")
        lines.append("=== 舊模型 vs 新模型 嚴格比較（百分比） ===")
        lines.append(f"舊模型: {str(old_new_analysis['old_model'])}")
        lines.append(f"新模型: {str(old_new_analysis['new_model'])}")
        lines.append("")
        lines.append("[整體指標]")

        overall_metrics = old_new_analysis.get("overall_metrics", [])
        if isinstance(overall_metrics, list):
            for row in overall_metrics:
                if not isinstance(row, dict):
                    continue
                metric = str(row.get("metric", "N/A"))
                old_value = float(row.get("old", 0.0))
                new_value = float(row.get("new", 0.0))
                diff = float(row.get("difference", 0.0))
                pct = row.get("percent_change", None)
                verdict = str(row.get("verdict", "same"))

                pct_str = "N/A (舊值為0)" if pct is None else f"{float(pct):+.2f}%"
                lines.append(
                    f"- {metric}: 舊={old_value:.6f}, 新={new_value:.6f}, 差值={diff:+.6f}, 變化率={pct_str}, 判定={verdict}"
                )

        lines.append("")
        lines.append("[關鍵點抖動（新相對舊）]")
        per_keypoint = old_new_analysis.get("per_keypoint", [])
        if isinstance(per_keypoint, list):
            for row in per_keypoint:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "N/A"))
                old_value = float(row.get("old_jitter", 0.0))
                new_value = float(row.get("new_jitter", 0.0))
                diff = float(row.get("difference", 0.0))
                pct = row.get("percent_change", None)
                verdict = str(row.get("verdict", "same"))

                pct_str = "N/A (舊值為0)" if pct is None else f"{float(pct):+.2f}%"
                lines.append(
                    f"- {name}: 舊={old_value:.6f}, 新={new_value:.6f}, 差值={diff:+.6f}, 變化率={pct_str}, 判定={verdict}"
                )

        summary = old_new_analysis.get("summary", {})
        if isinstance(summary, dict):
            lines.append("")
            lines.append(
                f"關鍵點總結: 惡化={int(summary.get('regressed_keypoints', 0))}, 改善={int(summary.get('improved_keypoints', 0))}, 不變={int(summary.get('unchanged_keypoints', 0))}"
            )

            max_reg = summary.get("max_regression_keypoint", None)
            if isinstance(max_reg, dict):
                pct = max_reg.get("percent_change", None)
                pct_str = "N/A (舊值為0)" if pct is None else f"{float(pct):+.2f}%"
                lines.append(
                    f"最大惡化關鍵點: {max_reg.get('name', 'N/A')} | 差值={float(max_reg.get('difference', 0.0)):+.6f} | 變化率={pct_str}"
                )

            max_imp = summary.get("max_improvement_keypoint", None)
            if isinstance(max_imp, dict):
                pct = max_imp.get("percent_change", None)
                pct_str = "N/A (舊值為0)" if pct is None else f"{float(pct):+.2f}%"
                lines.append(
                    f"最大改善關鍵點: {max_imp.get('name', 'N/A')} | 差值={float(max_imp.get('difference', 0.0)):+.6f} | 變化率={pct_str}"
                )

    with open(output_dir / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two pose models on the same video and log jitter differences.")
    parser.add_argument("--model-1", type=str, default=DEFAULT_MODEL_1, help="第一個模型的絕對路徑")
    parser.add_argument("--model-2", type=str, default=DEFAULT_MODEL_2, help="第二個模型的絕對路徑")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="影片的絕對路徑")
    parser.add_argument("--output-dir", type=str, default="jitter_compare_logs", help="輸出資料夾")
    parser.add_argument("--frame-step", type=int, default=0, help="強制取樣步長；0 代表依影片 FPS 自動計算")
    parser.add_argument("--target-fps", type=float, default=TARGET_MODEL_FPS, help="目標模型輸入 FPS")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda:0 等")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model_paths = [args.model_1, args.model_2]
    if not all(model_paths):
        raise RuntimeError("請提供兩個模型的絕對路徑")

    video_path = str(Path(args.video))
    if not video_path:
        raise RuntimeError("請提供影片的絕對路徑")
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"無法找到影片: {video_path}")

    for model_path in model_paths:
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"無法找到模型: {model_path}")
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("開始比較任務")
    logger.info("輸出目錄: %s", output_dir)
    logger.info("使用裝置: %s", device)
    logger.info("影片: %s", video_path)
    logger.info("模型數量: %d", len(model_paths))

    frame_metrics_path = output_dir / "frame_metrics.csv"
    with open(frame_metrics_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "run_tag",
            "model_name",
            "video_name",
            "sample_index",
            "source_frame_id",
            "mean_confidence",
            "active_kpts",
            "frame_jitter",
            "unstable",
        ])

        results: List[ModelCompareResult] = []
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        for index, model_path in enumerate(model_paths, start=1):
            logger.info("[%d/%d] 開始分析模型: %s", index, len(model_paths), model_path)
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
                "完成 %s | global_jitter=%.6f | unstable_rate=%.2f%% | detection_rate=%.2f%%",
                result.model_name,
                result.global_jitter,
                result.unstable_rate * 100,
                result.detection_rate * 100,
            )

    decision = compare_results(results)
    write_summary_files(output_dir, results, decision)

    if len(results) >= 2:
        old_result = results[0]
        new_result = results[1]
        improve_pct = jitter_amplitude_percent(old_result.global_jitter, new_result.global_jitter)
        if improve_pct is None:
            logger.info("新模型比舊模型關鍵點抖動幅度變好 N/A（舊模型關鍵點抖動幅度為0，無法計算百分比）")
        elif improve_pct >= 0:
            logger.info("新模型比舊模型關鍵點抖動幅度變好 %.2f%%", improve_pct)
            logger.info("結果判定: 新模型較好（%s）", new_result.model_name)
        else:
            logger.info("新模型比舊模型關鍵點抖動幅度變差 %.2f%%", abs(improve_pct))
            logger.info("結果判定: 舊模型較好（%s）", old_result.model_name)

    logger.info("比較完成")
    logger.info("較佳模型: %s", decision["best_model"])
    logger.info("較差模型: %s", decision["worst_model"])
    logger.info("詳細日誌: %s", output_dir / "comparison_report.log")
    logger.info("JSON 報告: %s", output_dir / "comparison_report.json")
    logger.info("文字報告: %s", output_dir / "comparison_report.txt")
    logger.info("逐幀 CSV: %s", output_dir / "frame_metrics.csv")


if __name__ == "__main__":
    main()