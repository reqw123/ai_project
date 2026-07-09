"""
匯出整段影片的關鍵點時序資料。

用途：
1. 將每支影片逐幀的關鍵點座標與信心值保存成 CSV
2. 將較完整的原始/EMA 平滑後資料保存成 NPZ
3. 產生摘要 Markdown，方便快速檢視每支影片的輸出檔案與基本統計
"""
import sys
import csv
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector
from models.stgcn_model import (
    flip_normalize,
    interpolate_missing,
    normalize_skeleton_coords,
    orientation_normalize,
)


# 配置
VIDEO_PATHS = [
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試\7月2日 (4).mp4",
   
]
YOLO_MODEL_PATH = r"C:\ai_project\cat_pose\v11s_121.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5
MAX_VIDEOS_TO_PROCESS = 1  # None 或 <= 0 表示處理全部影片
INTERPOLATE_CONF_THRESHOLD = 0.1
MOTION_METRICS_CONF_THRESHOLD = 0.3
MOTION_METRICS_TOP_K = 5
ROBUST_AMP_LOWER_PCT = 5
ROBUST_AMP_UPPER_PCT = 95
PLOT_USE_FIXED_REL_AXIS = True
PLOT_REL_AXIS_LOWER_PCT = 1
PLOT_REL_AXIS_UPPER_PCT = 99

# EMA 設定：與 test_video_inference_ema.py 保持一致
EMA_ALPHA = 1.0

OUTPUT_DIR = Path(r"C:\paper\output\keypoint_timeseries")
SUMMARY_MD_PATH = OUTPUT_DIR / "keypoint_timeseries_summary.md"
MOTION_METRICS_CSV_PATH = OUTPUT_DIR / "keypoint_motion_metrics.csv"
RAW_PLOT_OUTPUT_DIR = OUTPUT_DIR / "raw_plots"
NORMALIZED_PLOT_OUTPUT_DIR = OUTPUT_DIR / "normalized_plots"
SAVE_CSV = True
SAVE_NPZ = True

KEYPOINT_NAMES = [
    "Nose",
    "Left_Ear",
    "Right_Ear",
    "Chest",
    "Mid_Back",
    "Hip",
    "LF_Elbow",
    "LF_Paw",
    "RF_Elbow",
    "RF_Paw",
    "LH_Knee",
    "LH_Paw",
    "RH_Knee",
    "RH_Paw",
    "Tail_Root",
    "Tail_Mid",
    "Tail_Tip",
]


def make_safe_stem(video_path, fallback_idx):
    stem = Path(video_path).stem.strip()
    if stem:
        return stem.replace(" ", "_")
    return f"video_{fallback_idx}"


def get_selected_video_paths():
    if MAX_VIDEOS_TO_PROCESS is None or MAX_VIDEOS_TO_PROCESS <= 0:
        return VIDEO_PATHS
    return VIDEO_PATHS[:MAX_VIDEOS_TO_PROCESS]


def build_video_label(video_idx, video_file):
    return f"[{video_idx}] {video_file.parent.name}/{video_file.name}"


def build_normalized_timeseries(ema_keypoints, keypoint_conf):
    ema_arr = np.asarray(ema_keypoints, dtype=np.float32)
    conf_arr = np.asarray(keypoint_conf, dtype=np.float32)

    if ema_arr.size == 0:
        return ema_arr.copy(), ema_arr.copy()

    interpolated = interpolate_missing(ema_arr, conf_arr, threshold=INTERPOLATE_CONF_THRESHOLD).astype(np.float32)
    normalized = flip_normalize(interpolated)
    normalized = orientation_normalize(normalized)
    normalized = normalize_skeleton_coords(normalized).astype(np.float32)
    return interpolated, normalized


def _safe_float(value):
    if value is None or np.isnan(value):
        return "N/A"
    return f"{float(value):.4f}"


def _compute_body_scale(keypoint_arr, conf_arr, conf_threshold):
    if keypoint_arr.size == 0 or conf_arr.size == 0:
        return 1.0

    chest_xy = keypoint_arr[:, 4, :]
    hip_xy = keypoint_arr[:, 5, :]
    mid_back_xy = keypoint_arr[:, 4, :]
    hip_xy = keypoint_arr[:, 5, :]
    mid_back_conf = conf_arr[:, 4]
    hip_conf = conf_arr[:, 5]
    trunk_valid = (mid_back_conf > conf_threshold) & (hip_conf > conf_threshold)

    if np.any(trunk_valid):
        trunk_dist = np.linalg.norm(mid_back_xy[trunk_valid] - hip_xy[trunk_valid], axis=1)
        if trunk_dist.size > 0:
            return max(float(np.mean(trunk_dist)), 1e-6)
    return 1.0


def compute_motion_metrics(keypoints, keypoint_conf, conf_threshold):
    keypoint_arr = np.asarray(keypoints, dtype=np.float32)
    conf_arr = np.asarray(keypoint_conf, dtype=np.float32)

    if keypoint_arr.size == 0 or conf_arr.size == 0:
        return []

    body_scale = _compute_body_scale(keypoint_arr, conf_arr, conf_threshold)

    metrics = []
    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        valid = conf_arr[:, kp_idx] > conf_threshold
        valid_count = int(np.sum(valid))

        if valid_count < 2:
            metrics.append(
                {
                    "kp_idx": kp_idx,
                    "kp_name": kp_name,
                    "valid_count": valid_count,
                    "ptp_xy": np.nan,
                    "amp_x": np.nan,
                    "amp_y": np.nan,
                    "amp_x_rel": np.nan,
                    "amp_y_rel": np.nan,
                    "amp_rel": np.nan,
                    "robust_amp_x": np.nan,
                    "robust_amp_y": np.nan,
                    "robust_amp_xy": np.nan,
                    "robust_amp_x_rel": np.nan,
                    "robust_amp_y_rel": np.nan,
                    "robust_amp_rel": np.nan,
                    "rms_speed_rel": np.nan,
                    "iqr_rel": np.nan,
                    "body_scale": body_scale,
                }
            )
            continue

        xy = keypoint_arr[:, kp_idx, :]
        valid_xy = xy[valid]

        amp_x = float(np.max(valid_xy[:, 0]) - np.min(valid_xy[:, 0]))
        amp_y = float(np.max(valid_xy[:, 1]) - np.min(valid_xy[:, 1]))
        ptp_xy = float(np.sqrt(amp_x * amp_x + amp_y * amp_y))
        amp_x_rel = amp_x / body_scale
        amp_y_rel = amp_y / body_scale
        amp_rel = ptp_xy / body_scale

        q_low = np.percentile(valid_xy, ROBUST_AMP_LOWER_PCT, axis=0)
        q_high = np.percentile(valid_xy, ROBUST_AMP_UPPER_PCT, axis=0)
        robust_amp_x = float(q_high[0] - q_low[0])
        robust_amp_y = float(q_high[1] - q_low[1])
        robust_amp_xy = float(np.sqrt(robust_amp_x * robust_amp_x + robust_amp_y * robust_amp_y))
        robust_amp_x_rel = robust_amp_x / body_scale
        robust_amp_y_rel = robust_amp_y / body_scale
        robust_amp_rel = robust_amp_xy / body_scale

        delta = xy[1:] - xy[:-1]
        pair_valid = valid[1:] & valid[:-1]
        if np.any(pair_valid):
            speed = np.linalg.norm(delta[pair_valid], axis=1)
            rms_speed = float(np.sqrt(np.mean(speed * speed)))
        else:
            rms_speed = np.nan
        rms_speed_rel = rms_speed / body_scale if not np.isnan(rms_speed) else np.nan

        radius = np.linalg.norm(valid_xy, axis=1)
        if radius.size > 0:
            iqr_rel = float(np.percentile(radius, 75) - np.percentile(radius, 25)) / body_scale
        else:
            iqr_rel = np.nan

        metrics.append(
            {
                "kp_idx": kp_idx,
                "kp_name": kp_name,
                "valid_count": valid_count,
                "ptp_xy": ptp_xy,
                "amp_x": amp_x,
                "amp_y": amp_y,
                "amp_x_rel": amp_x_rel,
                "amp_y_rel": amp_y_rel,
                "amp_rel": amp_rel,
                "robust_amp_x": robust_amp_x,
                "robust_amp_y": robust_amp_y,
                "robust_amp_xy": robust_amp_xy,
                "robust_amp_x_rel": robust_amp_x_rel,
                "robust_amp_y_rel": robust_amp_y_rel,
                "robust_amp_rel": robust_amp_rel,
                "rms_speed_rel": rms_speed_rel,
                "iqr_rel": iqr_rel,
                "body_scale": body_scale,
            }
        )

    return metrics


def _compute_relative_xy_for_plot(keypoint_arr, conf_arr, conf_threshold):
    rel_arr = np.full_like(keypoint_arr, np.nan, dtype=np.float32)
    body_scale = _compute_body_scale(keypoint_arr, conf_arr, conf_threshold)

    for kp_idx in range(keypoint_arr.shape[1]):
        valid = conf_arr[:, kp_idx] > conf_threshold
        if not np.any(valid):
            continue
        x = keypoint_arr[:, kp_idx, 0].copy()
        y = keypoint_arr[:, kp_idx, 1].copy()
        x_med = float(np.median(x[valid]))
        y_med = float(np.median(y[valid]))
        rel_arr[:, kp_idx, 0] = (x - x_med) / body_scale
        rel_arr[:, kp_idx, 1] = (y - y_med) / body_scale

    return rel_arr


def _get_shared_axis_limits(rel_arr):
    x_vals = rel_arr[:, :, 0].reshape(-1)
    y_vals = rel_arr[:, :, 1].reshape(-1)
    x_vals = x_vals[~np.isnan(x_vals)]
    y_vals = y_vals[~np.isnan(y_vals)]

    def _calc(vals):
        if vals.size == 0:
            return (-1.0, 1.0)
        low = float(np.percentile(vals, PLOT_REL_AXIS_LOWER_PCT))
        high = float(np.percentile(vals, PLOT_REL_AXIS_UPPER_PCT))
        if abs(high - low) < 1e-6:
            pad = max(abs(high), 1.0) * 0.1
            low -= pad
            high += pad
        return (low, high)

    return _calc(x_vals), _calc(y_vals)


def save_motion_metrics_csv(csv_path, video_summaries):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_idx",
        "video_name",
        "mode",
        "kp_idx",
        "kp_name",
        "valid_count",
        "body_scale",
        "amp_x",
        "amp_y",
        "amp_xy",
        "amp_x_rel",
        "amp_y_rel",
        "amp_xy_rel",
        "robust_amp_x",
        "robust_amp_y",
        "robust_amp_xy",
        "robust_amp_x_rel",
        "robust_amp_y_rel",
        "robust_amp_xy_rel",
        "rms_speed_rel",
        "iqr_rel",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in video_summaries:
            for mode_key, mode_name in (("raw_motion_metrics", "RAW"), ("normalized_motion_metrics", "NORMALIZED")):
                for m in item.get(mode_key, []):
                    writer.writerow(
                        {
                            "video_idx": item["video_idx"],
                            "video_name": item["video_name"],
                            "mode": mode_name,
                            "kp_idx": m["kp_idx"],
                            "kp_name": m["kp_name"],
                            "valid_count": m["valid_count"],
                            "body_scale": "" if np.isnan(m["body_scale"]) else round(float(m["body_scale"]), 6),
                            "amp_x": "" if np.isnan(m["amp_x"]) else round(float(m["amp_x"]), 6),
                            "amp_y": "" if np.isnan(m["amp_y"]) else round(float(m["amp_y"]), 6),
                            "amp_xy": "" if np.isnan(m["ptp_xy"]) else round(float(m["ptp_xy"]), 6),
                            "amp_x_rel": "" if np.isnan(m["amp_x_rel"]) else round(float(m["amp_x_rel"]), 6),
                            "amp_y_rel": "" if np.isnan(m["amp_y_rel"]) else round(float(m["amp_y_rel"]), 6),
                            "amp_xy_rel": "" if np.isnan(m["amp_rel"]) else round(float(m["amp_rel"]), 6),
                            "robust_amp_x": "" if np.isnan(m["robust_amp_x"]) else round(float(m["robust_amp_x"]), 6),
                            "robust_amp_y": "" if np.isnan(m["robust_amp_y"]) else round(float(m["robust_amp_y"]), 6),
                            "robust_amp_xy": "" if np.isnan(m["robust_amp_xy"]) else round(float(m["robust_amp_xy"]), 6),
                            "robust_amp_x_rel": "" if np.isnan(m["robust_amp_x_rel"]) else round(float(m["robust_amp_x_rel"]), 6),
                            "robust_amp_y_rel": "" if np.isnan(m["robust_amp_y_rel"]) else round(float(m["robust_amp_y_rel"]), 6),
                            "robust_amp_xy_rel": "" if np.isnan(m["robust_amp_rel"]) else round(float(m["robust_amp_rel"]), 6),
                            "rms_speed_rel": "" if np.isnan(m["rms_speed_rel"]) else round(float(m["rms_speed_rel"]), 6),
                            "iqr_rel": "" if np.isnan(m["iqr_rel"]) else round(float(m["iqr_rel"]), 6),
                        }
                    )


def build_csv_fieldnames():
    fieldnames = [
        "frame_idx",
        "time_sec",
        "cat_detected",
        "det_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]
    for idx, name in enumerate(KEYPOINT_NAMES):
        prefix = f"kp_{idx:02d}_{name.lower()}"
        fieldnames.extend([
            f"{prefix}_x",
            f"{prefix}_y",
            f"{prefix}_conf",
        ])
    return fieldnames


def save_timeseries_csv(csv_path, records):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = build_csv_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_timeseries_npz(npz_path, data):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        frame_idx=np.asarray(data["frame_idx"], dtype=np.int32),
        time_sec=np.asarray(data["time_sec"], dtype=np.float32),
        detected=np.asarray(data["detected"], dtype=np.uint8),
        det_conf=np.asarray(data["det_conf"], dtype=np.float32),
        bbox=np.asarray(data["bbox"], dtype=np.float32),
        raw_keypoints=np.asarray(data["raw_keypoints"], dtype=np.float32),
        ema_keypoints=np.asarray(data["ema_keypoints"], dtype=np.float32),
        interpolated_keypoints=np.asarray(data["interpolated_keypoints"], dtype=np.float32),
        normalized_keypoints=np.asarray(data["normalized_keypoints"], dtype=np.float32),
        keypoint_conf=np.asarray(data["keypoint_conf"], dtype=np.float32),
        keypoint_names=np.asarray(KEYPOINT_NAMES),
    )


def plot_keypoint_timeseries(plot_dir, video_stem, video_label, plot_mode, time_sec, keypoints, keypoint_conf):
    plot_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.asarray(time_sec, dtype=np.float32)
    keypoint_arr = np.asarray(keypoints, dtype=np.float32)
    conf_arr = np.asarray(keypoint_conf, dtype=np.float32)
    velocity_arr = np.zeros_like(keypoint_arr)
    velocity_arr[1:] = keypoint_arr[1:] - keypoint_arr[:-1]
    rel_arr = _compute_relative_xy_for_plot(keypoint_arr, conf_arr, MOTION_METRICS_CONF_THRESHOLD)
    rel_x_lim, rel_y_lim = _get_shared_axis_limits(rel_arr)

    for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
        fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)

        x_values = keypoint_arr[:, kp_idx, 0]
        y_values = keypoint_arr[:, kp_idx, 1]
        x_rel_values = rel_arr[:, kp_idx, 0]
        y_rel_values = rel_arr[:, kp_idx, 1]
        conf_values = conf_arr[:, kp_idx]
        vx_values = velocity_arr[:, kp_idx, 0]
        vy_values = velocity_arr[:, kp_idx, 1]

        axes[0].plot(time_axis, x_values, color="#1f77b4", linewidth=1.2)
        axes[0].set_ylabel("x")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_axis, y_values, color="#ff7f0e", linewidth=1.2)
        axes[1].set_ylabel("y")
        axes[1].grid(True, alpha=0.3)

        if PLOT_USE_FIXED_REL_AXIS:
            axes[0].set_ylabel("x_rel")
            axes[1].set_ylabel("y_rel")
            axes[0].lines[0].set_ydata(x_rel_values)
            axes[1].lines[0].set_ydata(y_rel_values)
            axes[0].set_ylim(rel_x_lim[0], rel_x_lim[1])
            axes[1].set_ylim(rel_y_lim[0], rel_y_lim[1])

        axes[2].plot(time_axis, conf_values, color="#2ca02c", linewidth=1.2)
        axes[2].set_ylabel("conf")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(time_axis, vx_values, color="#d62728", linewidth=1.2)
        axes[3].set_ylabel("vx")
        axes[3].grid(True, alpha=0.3)

        axes[4].plot(time_axis, vy_values, color="#9467bd", linewidth=1.2)
        axes[4].set_ylabel("vy")
        axes[4].set_xlabel("time (sec)")
        axes[4].grid(True, alpha=0.3)

        fig.suptitle(f"{plot_mode} | Source: {video_label} | KP {kp_idx:02d} {kp_name}")
        fig.tight_layout()

        plot_path = plot_dir / f"{video_stem}_kp_{kp_idx:02d}_{kp_name.lower()}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_summary(summary_path, video_summaries):
    lines = [
        "# Keypoint Timeseries Export Summary",
        "",
        f"Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"YOLO model: {YOLO_MODEL_PATH}",
        f"Device: {INFERENCE_DEVICE}",
        f"YOLO imgsz: {YOLO_IMGSZ}",
        f"YOLO conf threshold: {YOLO_CONF_THRESHOLD}",
        f"Max videos to process: {MAX_VIDEOS_TO_PROCESS if MAX_VIDEOS_TO_PROCESS is not None else 'all'}",
        f"Interpolation conf threshold: {INTERPOLATE_CONF_THRESHOLD}",
        f"Motion metrics conf threshold: {MOTION_METRICS_CONF_THRESHOLD}",
        f"Robust amplitude percentiles: P{ROBUST_AMP_LOWER_PCT} - P{ROBUST_AMP_UPPER_PCT}",
        f"Plot fixed relative axis: {PLOT_USE_FIXED_REL_AXIS} (P{PLOT_REL_AXIS_LOWER_PCT} - P{PLOT_REL_AXIS_UPPER_PCT})",
        f"EMA alpha: {EMA_ALPHA}",
        "",
        "## Output Overview",
        "",
        "說明：",
        "- CSV：保存逐幀 EMA 平滑後、未正規化的關鍵點座標與信心值，方便用 Excel / pandas 檢視。",
        "- NPZ：保存逐幀 raw / EMA / interpolate / normalized 關鍵點、bbox 與偵測旗標，適合後續 NumPy 分析。",
        "- Raw PNG：每個關鍵點各輸出一張未正規化時序圖，使用 EMA + interpolate 後的原始影像座標。",
        "- Normalized PNG：每個關鍵點各輸出一張模型同款正規化時序圖，流程為 interpolate -> flip -> orientation -> scale normalize。",
        "- Motion Metrics：輸出相對振幅與相對速度指標（以 mid_back-hip 平均距離做尺度基準），可跨圖比較。",
        "",
        "| # | Video | Frames | Detected Frames | Detection Rate | CSV | NPZ | Raw Plots Dir | Normalized Plots Dir |",
        "|---:|:---|---:|---:|---:|:---|:---|:---|:---|",
    ]

    for item in video_summaries:
        csv_name = item["csv_path"].name if item["csv_path"] is not None else "-"
        npz_name = item["npz_path"].name if item["npz_path"] is not None else "-"
        raw_plot_dir_name = item["raw_plot_dir"].name if item.get("raw_plot_dir") is not None else "-"
        normalized_plot_dir_name = item["normalized_plot_dir"].name if item.get("normalized_plot_dir") is not None else "-"
        lines.append(
            f"| {item['video_idx']} | {item['video_name']} | {item['total_frames']} | "
            f"{item['detected_frames']} | {item['detection_rate'] * 100:.2f}% | {csv_name} | {npz_name} | {raw_plot_dir_name} | {normalized_plot_dir_name} |"
        )

    for item in video_summaries:
        lines.append("")
        lines.append(f"## Motion Metrics - Video [{item['video_idx']}] {item['video_name']}")
        lines.append("")
        lines.append("指標說明：")
        lines.append("- amp_rel: 2D 振幅（max-min）/ 軀幹尺度（越大表示相對活動範圍越大）")
        lines.append("- robust_amp_rel: 2D robust 振幅（P95-P5）/ 軀幹尺度（較不受離群值影響）")
        lines.append("- rms_speed_rel: 2D 速度 RMS / 軀幹尺度（越大表示整體動得更快）")
        lines.append("- iqr_rel: 半徑分布 IQR / 軀幹尺度（越大表示典型波動範圍越廣）")
        lines.append("")

        for mode_key, mode_title in (("raw_motion_metrics", "RAW"), ("normalized_motion_metrics", "NORMALIZED")):
            lines.append(f"### {mode_title} Top-{MOTION_METRICS_TOP_K} by robust_amp_rel")
            lines.append("")
            lines.append("| Rank | KP idx | Name | Valid Frames | robust_amp_rel | amp_rel | rms_speed_rel | iqr_rel |")
            lines.append("|---:|---:|:---|---:|---:|---:|---:|---:|")

            metric_rows = [m for m in item.get(mode_key, []) if not np.isnan(m["robust_amp_rel"])]
            metric_rows.sort(key=lambda x: x["robust_amp_rel"], reverse=True)

            if not metric_rows:
                lines.append("| - | - | - | - | N/A | N/A | N/A | N/A |")
                lines.append("")
                continue

            for rank, m in enumerate(metric_rows[:MOTION_METRICS_TOP_K], start=1):
                lines.append(
                    f"| {rank} | {m['kp_idx']} | {m['kp_name']} | {m['valid_count']} | "
                    f"{_safe_float(m['robust_amp_rel'])} | {_safe_float(m['amp_rel'])} | {_safe_float(m['rms_speed_rel'])} | {_safe_float(m['iqr_rel'])} |"
                )
            lines.append("")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("匯出整段影片關鍵點時序資料")
    print("=" * 60)
    print(f"YOLO 模型: {YOLO_MODEL_PATH}")
    print(f"推論裝置: {INFERENCE_DEVICE}")
    print(f"YOLO imgsz: {YOLO_IMGSZ}")
    print(f"YOLO conf threshold: {YOLO_CONF_THRESHOLD}")
    print(f"處理影片數量: {MAX_VIDEOS_TO_PROCESS if MAX_VIDEOS_TO_PROCESS is not None else 'all'}")
    print(f"插值信心閾值: {INTERPOLATE_CONF_THRESHOLD}")
    print(f"運動指標信心閾值: {MOTION_METRICS_CONF_THRESHOLD}")
    print(f"Robust 幅度分位數: P{ROBUST_AMP_LOWER_PCT} ~ P{ROBUST_AMP_UPPER_PCT}")
    print(f"統一相對座標圖軸: {PLOT_USE_FIXED_REL_AXIS}")
    print(f"EMA alpha: {EMA_ALPHA}")
    print(f"輸出資料夾: {OUTPUT_DIR}")
    print("=" * 60)

    detector = KeypointDetector(
        YOLO_MODEL_PATH,
        device=INFERENCE_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf_thres=YOLO_CONF_THRESHOLD,
    )

    video_summaries = []

    for video_idx, video_path in enumerate(get_selected_video_paths()):
        video_file = Path(video_path)
        if not video_file.exists():
            print(f"❌ 找不到影片，跳過: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"❌ 無法開啟影片，跳過: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1:
            fps = 30.0

        video_stem = make_safe_stem(video_path, video_idx)
        video_label = build_video_label(video_idx, video_file)
        csv_path = OUTPUT_DIR / f"{video_stem}_timeseries.csv" if SAVE_CSV else None
        npz_path = OUTPUT_DIR / f"{video_stem}_timeseries.npz" if SAVE_NPZ else None
        raw_plot_dir = RAW_PLOT_OUTPUT_DIR / video_stem
        normalized_plot_dir = NORMALIZED_PLOT_OUTPUT_DIR / video_stem

        print("\n" + "-" * 60)
        print(f"處理影片 [{video_idx}] {video_path}")
        print(f"總幀數: {total_frames} | FPS: {fps:.2f}")

        ema_kpts = None
        detected_frames = 0
        frame_idx_list = []
        time_sec_list = []
        detected_list = []
        det_conf_list = []
        bbox_list = []
        raw_keypoints_list = []
        ema_keypoints_list = []
        keypoint_conf_list = []
        csv_records = []

        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame += 1
            kpts, kpt_conf, bbox, det_conf = detector.detect(frame)

            raw_kpts = np.zeros((17, 2), dtype=np.float32)
            ema_out = np.zeros((17, 2), dtype=np.float32)
            conf_out = np.zeros((17,), dtype=np.float32)
            bbox_out = np.full((4,), np.nan, dtype=np.float32)
            detected = 0
            det_conf_value = np.nan

            if kpts is not None:
                detected = 1
                detected_frames += 1
                raw_kpts = kpts.astype(np.float32)
                conf_out = kpt_conf.astype(np.float32)

                if ema_kpts is None:
                    ema_kpts = raw_kpts.copy()
                else:
                    ema_kpts = EMA_ALPHA * raw_kpts + (1.0 - EMA_ALPHA) * ema_kpts

                ema_out = ema_kpts.astype(np.float32)
                if bbox is not None:
                    bbox_out = np.asarray(bbox, dtype=np.float32)
                if det_conf is not None:
                    det_conf_value = float(det_conf)
            else:
                ema_kpts = None

            frame_idx_list.append(current_frame)
            time_sec_list.append(current_frame / fps)
            detected_list.append(detected)
            det_conf_list.append(det_conf_value)
            bbox_list.append(bbox_out)
            raw_keypoints_list.append(raw_kpts)
            ema_keypoints_list.append(ema_out)
            keypoint_conf_list.append(conf_out)

            record = {
                "frame_idx": current_frame,
                "time_sec": round(current_frame / fps, 6),
                "cat_detected": detected,
                "det_conf": "" if np.isnan(det_conf_value) else round(det_conf_value, 6),
                "bbox_x1": "" if np.isnan(bbox_out[0]) else round(float(bbox_out[0]), 4),
                "bbox_y1": "" if np.isnan(bbox_out[1]) else round(float(bbox_out[1]), 4),
                "bbox_x2": "" if np.isnan(bbox_out[2]) else round(float(bbox_out[2]), 4),
                "bbox_y2": "" if np.isnan(bbox_out[3]) else round(float(bbox_out[3]), 4),
            }

            for kp_idx, kp_name in enumerate(KEYPOINT_NAMES):
                prefix = f"kp_{kp_idx:02d}_{kp_name.lower()}"
                record[f"{prefix}_x"] = round(float(ema_out[kp_idx, 0]), 4)
                record[f"{prefix}_y"] = round(float(ema_out[kp_idx, 1]), 4)
                record[f"{prefix}_conf"] = round(float(conf_out[kp_idx]), 6)

            csv_records.append(record)

            if current_frame % 100 == 0:
                pct = current_frame / max(total_frames, 1) * 100
                print(f"  已處理: {current_frame}/{total_frames} ({pct:.1f}%)")

        cap.release()

        interpolated_keypoints, normalized_keypoints = build_normalized_timeseries(
            ema_keypoints_list,
            keypoint_conf_list,
        )
        raw_motion_metrics = compute_motion_metrics(
            interpolated_keypoints,
            keypoint_conf_list,
            conf_threshold=MOTION_METRICS_CONF_THRESHOLD,
        )
        normalized_motion_metrics = compute_motion_metrics(
            normalized_keypoints,
            keypoint_conf_list,
            conf_threshold=MOTION_METRICS_CONF_THRESHOLD,
        )

        if csv_path is not None:
            save_timeseries_csv(csv_path, csv_records)
            print(f"✓ CSV 已保存: {csv_path}")

        if npz_path is not None:
            save_timeseries_npz(
                npz_path,
                {
                    "frame_idx": frame_idx_list,
                    "time_sec": time_sec_list,
                    "detected": detected_list,
                    "det_conf": det_conf_list,
                    "bbox": bbox_list,
                    "raw_keypoints": raw_keypoints_list,
                    "ema_keypoints": ema_keypoints_list,
                    "interpolated_keypoints": interpolated_keypoints,
                    "normalized_keypoints": normalized_keypoints,
                    "keypoint_conf": keypoint_conf_list,
                },
            )
            print(f"✓ NPZ 已保存: {npz_path}")

        plot_keypoint_timeseries(
            raw_plot_dir,
            video_stem,
            video_label,
            "RAW",
            time_sec_list,
            interpolated_keypoints,
            keypoint_conf_list,
        )
        print(f"✓ Raw 關鍵點時序圖已保存: {raw_plot_dir}")

        plot_keypoint_timeseries(
            normalized_plot_dir,
            video_stem,
            video_label,
            "NORMALIZED",
            time_sec_list,
            normalized_keypoints,
            keypoint_conf_list,
        )
        print(f"✓ Normalized 關鍵點時序圖已保存: {normalized_plot_dir}")

        detection_rate = detected_frames / max(total_frames, 1)
        video_summaries.append(
            {
                "video_idx": video_idx,
                "video_name": video_file.name,
                "total_frames": total_frames,
                "detected_frames": detected_frames,
                "detection_rate": detection_rate,
                "csv_path": csv_path,
                "npz_path": npz_path,
                "raw_plot_dir": raw_plot_dir,
                "normalized_plot_dir": normalized_plot_dir,
                "raw_motion_metrics": raw_motion_metrics,
                "normalized_motion_metrics": normalized_motion_metrics,
            }
        )

    write_summary(SUMMARY_MD_PATH, video_summaries)
    save_motion_metrics_csv(MOTION_METRICS_CSV_PATH, video_summaries)
    print("\n" + "=" * 60)
    print(f"✓ 摘要報告已保存: {SUMMARY_MD_PATH}")
    print(f"✓ 波形幅度指標 CSV 已保存: {MOTION_METRICS_CSV_PATH}")
    print("匯出完成")
    print("=" * 60)


if __name__ == "__main__":
    main()