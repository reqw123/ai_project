"""
檢查各行為類別訓練影片的光照分布，記錄現階段亮度狀況，供之後追蹤資料集
光照多樣性有沒有隨新增影片而改善。

背景：docs/YOLO-Pose應用文獻與專案優化建議.md「中相關建議」提到 CBR-YOLO
論文強調的「多場景天氣穩健性」，猜測本專案訓練資料的光照/場景多樣性可能
不夠，但原本只是文獻上的推測，沒有實測數據。這支腳本直接抽樣每支訓練影
片幾幀、算 HSV V channel（亮度）均值當代理指標，統計各類別的亮度分布，
並把「最暗/最亮」幾支影片列出來供人工抽查（brightness 只是代理指標，真
正的畫質/是否為低光場景需要人工開影片確認）。

2026-08-11 首次執行結果：全體 501 支影片裡只有 6 支（1.2%）落在「很暗
(V<60)」區間，且這 6 支實際亮度也只是 27~59（勉強壓線），沒有一支是真正
的夜間/低光影片——資料集幾乎不含低光照條件，是一個有數據佐證的真實缺口
（貓晨昏活動較多，居家監控情境下低光很常見）。詳見 memory:
project_kalman_smoothing_eval 或搜尋本檔輸出歷史紀錄。

用法：
    python eval_lighting_distribution.py
    python eval_lighting_distribution.py --n_samples 8 --dark_threshold 60
"""
import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')  # Windows 主控台預設編碼常是 cp950，會把中文印成亂碼

BEHAVIOR_ORDER = ['walk', 'lick', 'scratch', 'shake', 'stop']

# 訓練影片來源資料夾（跟 tools/train_data/0_dataset_collect.py 的 VIDEO_FOLDERS 同一批）
DEFAULT_VIDEO_BASE = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用"
DEFAULT_OUTPUT_DIR = r"C:\ai_project\paper\cat_monitoring_system\eval_results\lighting_distribution"

# 亮度分級門檻（HSV V channel 0-255 均值），純粗略分級用來看分布形狀
_BINS = [
    (60,  "很暗(<60)"),
    (100, "偏暗(60-100)"),
    (160, "正常(100-160)"),
    (200, "偏亮(160-200)"),
    (256, "很亮(>200)"),
]


def brightness_bin(v: float) -> str:
    for upper, name in _BINS:
        if v < upper:
            return name
    return _BINS[-1][1]


def sample_video_brightness(video_path: Path, n_samples: int):
    """均勻抽 n_samples 幀，回傳 (mean_v, std_v) 或 None（開檔失敗/無幀）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(total - 1, 0), n_samples, dtype=int)
    v_means = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_means.append(float(hsv[:, :, 2].mean()))
    cap.release()
    if not v_means:
        return None
    return float(np.mean(v_means)), float(np.std(v_means))


def scan_lighting(video_base: str, n_samples: int):
    """回傳 {class_name: [(video_name, mean_v, std_v), ...]}。"""
    base = Path(video_base)
    results = defaultdict(list)
    for cls in BEHAVIOR_ORDER:
        folder = base / cls
        if not folder.is_dir():
            print(f"[Warning] 資料夾不存在，跳過: {folder}")
            continue
        videos = sorted(folder.glob("*.mp4"))
        for vp in videos:
            r = sample_video_brightness(vp, n_samples)
            if r is not None:
                results[cls].append((vp.name, r[0], r[1]))
    return results


def print_and_save_report(results: dict, dark_threshold: float, output_dir: str):
    """印出分布報告，同時把逐支影片數據存成 CSV、摘要存成 txt（皆帶時間戳）。"""
    sep = '─' * 90
    print(f"\n{'=' * 90}")
    print("各類別訓練影片亮度分布（HSV V channel 均值，0-255）")
    print(f"{'=' * 90}")

    summary_lines = []
    all_rows = []  # for CSV

    for cls in BEHAVIOR_ORDER:
        rows = results.get(cls, [])
        if not rows:
            continue
        means = [r[1] for r in rows]
        line0 = (f"\n[{cls}]  n={len(rows)}  平均亮度={np.mean(means):.1f}  "
                 f"標準差={np.std(means):.1f}  最暗={min(means):.1f}  最亮={max(means):.1f}")
        print(line0)
        summary_lines.append(line0)

        bins = defaultdict(int)
        for _, mv, _ in rows:
            bins[brightness_bin(mv)] += 1
        for _, bname in _BINS:
            cnt = bins.get(bname, 0)
            pct = cnt / len(rows) * 100
            bar = '█' * int(pct / 2)
            line = f"    {bname:<14} {cnt:>4} 支 ({pct:>5.1f}%)  {bar}"
            print(line)
            summary_lines.append(line)

        sorted_rows = sorted(rows, key=lambda r: r[1])
        darkest = [(n, round(m, 1)) for n, m, _ in sorted_rows[:3]]
        brightest = [(n, round(m, 1)) for n, m, _ in sorted_rows[-3:]]
        line1 = f"    最暗3支（建議人工抽查是否為真低光場景）: {darkest}"
        line2 = f"    最亮3支: {brightest}"
        print(line1); print(line2)
        summary_lines.append(line1); summary_lines.append(line2)

        for name, mv, sv in rows:
            all_rows.append({'class': cls, 'video': name, 'mean_v': round(mv, 2),
                             'std_v': round(sv, 2), 'brightness_bin': brightness_bin(mv)})

    all_means = [r[1] for rows in results.values() for r in rows]
    print(f"\n{'=' * 90}")
    print("全部類別總覽")
    print(f"{'=' * 90}")
    if all_means:
        n_dark = sum(1 for m in all_means if m < dark_threshold)
        overview = (f"全體 n={len(all_means)}  平均亮度={np.mean(all_means):.1f}  "
                    f"標準差={np.std(all_means):.1f}  範圍=[{min(all_means):.1f}, {max(all_means):.1f}]\n"
                    f"「很暗(<{dark_threshold:.0f})」影片數：{n_dark} / {len(all_means)} "
                    f"({n_dark / len(all_means) * 100:.1f}%)")
        print(overview)
        summary_lines.append(overview)
        if n_dark / len(all_means) < 0.05:
            warn = "⚠ 低光影片占比 <5%，資料集光照多樣性明顯不足，建議之後補拍夜間/昏暗場景影片。"
            print(warn)
            summary_lines.append(warn)

    # ── 持久化：CSV（逐支影片）+ txt（摘要，供之後追蹤比較）─────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"lighting_{ts}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'video', 'mean_v', 'std_v', 'brightness_bin'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    summary_path = out_dir / f"lighting_summary_{ts}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"執行時間: {datetime.now().isoformat()}\n")
        f.write('\n'.join(summary_lines))

    print(f"\n✓ 逐支影片數據已存: {csv_path}")
    print(f"✓ 摘要報告已存: {summary_path}")
    print("  （檔名帶時間戳，之後補拍新影片重跑本腳本，可以比對不同時間點的快照追蹤改善情況）")


def main():
    parser = argparse.ArgumentParser(description="檢查訓練影片光照分布，記錄現階段亮度狀況。")
    parser.add_argument('--video_base', default=DEFAULT_VIDEO_BASE,
                        help="訓練影片來源根目錄（底下要有 walk/lick/scratch/shake/stop 五個子資料夾）")
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n_samples', type=int, default=5, help="每支影片均勻抽樣幾幀計算亮度")
    parser.add_argument('--dark_threshold', type=float, default=60.0,
                        help="判定「很暗」的 V channel 均值門檻")
    args = parser.parse_args()

    print(f"[影片來源] {args.video_base}")
    results = scan_lighting(args.video_base, args.n_samples)
    print_and_save_report(results, args.dark_threshold, args.output)


if __name__ == '__main__':
    main()
