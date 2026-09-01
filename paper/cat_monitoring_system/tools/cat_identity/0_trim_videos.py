"""
0_trim_videos.py — 用 YOLO 找出影片裡「有貓」的時間區間，前後各留 2 秒
容錯，其餘片段剪掉，輸出一支只剩有貓片段的影片
================================================================================
獨立腳本。用 ultralytics(YOLO) 掃描 + ffmpeg 剪接，不 import 本專案任何模組。

流程：
  1. 以 SAMPLE_FPS 對整支影片抽樣跑 YOLO，記錄每個抽樣點「有沒有貓」
  2. 去抖動（連續 MIN_CONSECUTIVE_HITS 個抽樣點都有貓才算一段的開始）
  3. 合併間隔小於 MERGE_GAP_SECONDS 的相鄰區間（貓短暫離開/被遮擋不切斷）
  4. 丟掉短於 MIN_SEGMENT_SECONDS 的區間（雜訊）
  5. 每段前後各加 PAD_SECONDS 容錯，夾在 [0, 片長] 內，再合併重疊
  6. 用 ffmpeg 把保留區間逐段重新編碼、串接成一支輸出影片
     （OUTPUT_MODE="segments" 則每段各存一支）

典型情境：一支幾十秒的短影片，貓只在其中一小段（例如第 5~8 秒）出現，其餘都是空景。
本腳本把那一段抓出來、前後各留 PAD_SECONDS 秒容錯、其餘剪掉，輸出 <檔名>_catonly.mp4。
所有門檻（PAD / MERGE_GAP / MIN_SEGMENT）單位都是「秒」，直接對應影片時間軸。
（長影片也適用，只是那些秒數門檻要按比例調大。）

安全設計：
  - 絕不覆蓋、絕不硬刪原始檔。輸出寫到以執行時間命名的獨立資料夾。
  - DELETE_ORIGINAL 只是「搬到 _removed_originals/ 備份夾」，不是刪除。
  - DRY_RUN=True 不剪不刪，只印區間 + 寫 _timeline_<檔名>.csv（每個抽樣點 YOLO 偵測結果），
    用來確認「它抓的區間 = 你要的那一段」再正式剪。

執行環境：需要 ultralytics + ffmpeg。本機用專案 conda 環境：
    & "C:\\Users\\lynnc\\anaconda3\\envs\\yolo\\python.exe" 0_trim_videos.py
"""

# ═══════════════════════════════════════════════════════════════
#  設定區
# ═══════════════════════════════════════════════════════════════
YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_149.pt"
DEVICE = "cuda"                 # "cuda" 優先，抓不到自動 fallback "cpu"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5      # 「這幀有沒有貓」用的偵測門檻（比身分辨識寬鬆些，寧可多留）

# 影片來源：檔案或資料夾（遞迴掃描），可放多個
SOURCE = [
 r"D:\目標貓採樣-20260828T210421Z-1-001\目標貓採樣\他貓\2026-07-19 00_18_34_0.mp4",
]
# 若設定環境變數 TEST_VIDEO_PATH，優先只用該路徑（覆蓋上面的 SOURCE）
import os as _os
_env = _os.getenv("TEST_VIDEO_PATH", "").strip()
if _env:
    SOURCE = [_env]

# ── 掃描 ──
SAMPLE_FPS = 6.0               # 每秒對影片抽樣幾次跑 YOLO（短片抓精細一點；長片可降到 2~4 省時間）
SCAN_BATCH = 16               # 一次餵給 YOLO 幾張抽樣幀（批次推論較快）

# ── 區間邏輯（單位：全部是「秒」，直接對應影片時間軸）──
# 這組預設是給「幾十秒短片、有貓的部分只有幾秒」調的。
#   - 長影片（分鐘級）：PAD / MERGE_GAP / MIN_SEGMENT 按比例調大（例如 2 / 15 / 3）
#   - 空景很容易被 YOLO 誤判成貓：MIN_CONSECUTIVE_HITS 調到 3~4、MIN_SEGMENT_SECONDS 調大
PAD_SECONDS = 1.0             # 每段前後各留的容錯（3 秒的片段 + 2 秒 pad 就變 7 秒，短片別設太大）
MERGE_GAP_SECONDS = 1.5       # 兩段之間「沒貓」的空檔小於這個值就併成一段
MIN_SEGMENT_SECONDS = 1.0     # 加容錯前，短於這個長度的「有貓」區間視為雜訊丟棄
MIN_CONSECUTIVE_HITS = 2      # 連續幾個抽樣點都偵測到貓，才承認一段的開始（去抖動）

# ── 輸出 ──
import pathlib as _pathlib
OUTPUT_ROOT = _pathlib.Path(r"C:\ai_project\paper\cat_monitoring_system\tools\train_data\cat_identity\trimmed")
OUTPUT_MODE = "trimmed"        # "trimmed" = 串成一支；"segments" = 每段各一支
OUTPUT_SUFFIX = "_catonly"     # trimmed 模式輸出檔名 = <原檔名><OUTPUT_SUFFIX>.mp4
REENCODE_CRF = 20             # 重新編碼品質（18~23，越小越好越大）
REENCODE_PRESET = "veryfast"

DRY_RUN = False               # True：不剪不刪，只印區間 + 寫掃描時間軸 CSV 供檢查
DELETE_ORIGINAL = False       # True：剪完後把原始檔「搬到」OUTPUT_ROOT/_removed_originals/（非硬刪）
SKIP_IF_OUTPUT_EXISTS = True  # 輸出已存在就跳過該影片（分批續跑）
SAVE_TIMELINE_CSV = True      # 每支影片輸出 _timeline_<檔名>.csv：每個抽樣點的 time / has_cat / n_boxes / max_conf
                              #（DRY_RUN 抓錯區間時，用這個看 YOLO 到底在哪些時間點偵測到東西）

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
# ═══════════════════════════════════════════════════════════════

import sys
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

_MISSING = []
try:
    import cv2
except ImportError:
    _MISSING.append("opencv-python")
try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")
try:
    from ultralytics import YOLO
except ImportError:
    _MISSING.append("ultralytics")

if _MISSING:
    print("❌ 缺少套件：", ", ".join(_MISSING))
    print(r'   本機請用：& "C:\Users\lynnc\anaconda3\envs\yolo\python.exe" 0_trim_videos.py')
    sys.exit(1)


def find_ffmpeg():
    for name in ("ffmpeg", "ffmpeg.exe"):
        p = shutil.which(name)
        if p:
            return p
    for cand in (
        r"C:\Users\lynnc\anaconda3\Library\bin\ffmpeg.exe",
        r"C:\Users\lynnc\anaconda3\envs\yolo\Library\bin\ffmpeg.exe",
    ):
        if Path(cand).exists():
            return cand
    return None


FFMPEG = find_ffmpeg()


# ═══════════════════════════════════════════════════════════════
def get_device():
    try:
        import torch
        if DEVICE.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        if DEVICE.lower() == "cuda":
            print("⚠ 找不到可用的 CUDA，改用 CPU")
    except ImportError:
        pass
    return "cpu"


def resolve_videos(sources):
    out = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_VIDEO_EXTS))
        else:
            print(f"⚠ 略過無效路徑: {p}")
    return out


def scan_cat_presence(model, video_path, device):
    """以 SAMPLE_FPS 抽樣跑 YOLO，回傳 (sample_times, has_cat, n_boxes, max_conf, fps, duration)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total / fps if total else 0.0
    step = max(1, int(round(fps / SAMPLE_FPS)))

    sample_times, has_cat, n_boxes, max_conf = [], [], [], []
    batch_frames, batch_times = [], []

    def flush():
        if not batch_frames:
            return
        results = model.predict(batch_frames, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD,
                                device=device, verbose=False)
        for t, r in zip(batch_times, results):
            n = 0 if r.boxes is None else len(r.boxes)
            mc = float(r.boxes.conf.max().cpu()) if n else 0.0
            sample_times.append(t)
            has_cat.append(n > 0)
            n_boxes.append(n)
            max_conf.append(mc)
        batch_frames.clear()
        batch_times.clear()

    idx = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                batch_frames.append(frame)
                batch_times.append(idx / fps)
                if len(batch_frames) >= SCAN_BATCH:
                    flush()
        idx += 1
    flush()
    cap.release()
    if duration <= 0 and sample_times:
        duration = sample_times[-1] + step / fps
    return sample_times, has_cat, n_boxes, max_conf, fps, duration


def build_intervals(sample_times, has_cat, duration):
    """抽樣的 has_cat 序列 → 最終保留的 (start, end) 秒區間清單。"""
    sample_dt = (sample_times[1] - sample_times[0]) if len(sample_times) > 1 else 1.0 / SAMPLE_FPS

    # 1) 去抖動：連續 >= MIN_CONSECUTIVE_HITS 個 True 才算「有貓」
    raw = []
    run_start = None
    run_len = 0
    for i, flag in enumerate(has_cat):
        if flag:
            if run_start is None:
                run_start = sample_times[i]
            run_len += 1
        else:
            if run_start is not None and run_len >= MIN_CONSECUTIVE_HITS:
                raw.append((run_start, sample_times[i - 1] + sample_dt))
            run_start = None
            run_len = 0
    if run_start is not None and run_len >= MIN_CONSECUTIVE_HITS:
        raw.append((run_start, sample_times[-1] + sample_dt))

    if not raw:
        return []

    # 2) 合併間隔 < MERGE_GAP_SECONDS 的相鄰區間
    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] < MERGE_GAP_SECONDS:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 3) 丟掉太短的（加容錯前）
    merged = [iv for iv in merged if iv[1] - iv[0] >= MIN_SEGMENT_SECONDS]
    if not merged:
        return []

    # 4) 前後加容錯、夾範圍、再合併重疊
    end_cap = duration if duration > 0 else merged[-1][1] + PAD_SECONDS
    padded = [[max(0.0, s - PAD_SECONDS), min(end_cap, e + PAD_SECONDS)] for s, e in merged]
    final = [list(padded[0])]
    for s, e in padded[1:]:
        if s <= final[-1][1]:
            final[-1][1] = max(final[-1][1], e)
        else:
            final.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in final]


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  ✗ ffmpeg 失敗:\n" + (r.stderr[-1500:] if r.stderr else ""))
        return False
    return True


def cut_segments(video_path, intervals, run_dir):
    """用 ffmpeg 逐段重新編碼；OUTPUT_MODE 決定串成一支還是各存一支。回傳輸出檔清單。"""
    tmp = run_dir / f"_tmp_{video_path.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    seg_files = []
    for i, (s, e) in enumerate(intervals):
        seg = tmp / f"seg_{i:03d}.mp4"
        ok = _run([
            FFMPEG, "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video_path),
            "-c:v", "libx264", "-preset", REENCODE_PRESET, "-crf", str(REENCODE_CRF),
            "-c:a", "aac", "-movflags", "+faststart", str(seg),
        ])
        if ok:
            seg_files.append(seg)

    if not seg_files:
        shutil.rmtree(tmp, ignore_errors=True)
        return []

    if OUTPUT_MODE == "segments":
        outs = []
        for i, seg in enumerate(seg_files):
            dst = run_dir / f"{video_path.stem}{OUTPUT_SUFFIX}_{i:03d}.mp4"
            shutil.move(str(seg), str(dst))
            outs.append(dst)
        shutil.rmtree(tmp, ignore_errors=True)
        return outs

    # trimmed：concat demuxer 串接（各段編碼參數一致，可直接 -c copy）
    listfile = tmp / "list.txt"
    listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in seg_files), encoding="utf-8")
    dst = run_dir / f"{video_path.stem}{OUTPUT_SUFFIX}.mp4"
    ok = _run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(dst)])
    shutil.rmtree(tmp, ignore_errors=True)
    return [dst] if ok else []


def main():
    if FFMPEG is None:
        print("❌ 找不到 ffmpeg，請確認已安裝並在 PATH 上")
        sys.exit(1)

    videos = resolve_videos(SOURCE)
    if not videos:
        print("❌ 找不到任何影片")
        return

    device = get_device()
    print(f"載入 YOLO: {YOLO_MODEL_PATH}  device={device}  ffmpeg={FFMPEG}")
    model = YOLO(YOLO_MODEL_PATH)

    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not DRY_RUN or SAVE_TIMELINE_CSV:
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"影片數: {len(videos)}   輸出: {run_dir}   DRY_RUN={DRY_RUN}\n")

    report = {}
    for vp in videos:
        print(f"── {vp.name}")
        out_name = f"{vp.stem}{OUTPUT_SUFFIX}.mp4"
        if SKIP_IF_OUTPUT_EXISTS and (run_dir / out_name).exists():
            print("   ↷ 輸出已存在，跳過")
            continue

        scan = scan_cat_presence(model, vp, device)
        if scan is None:
            print("   ⚠ 無法開啟，跳過")
            continue
        sample_times, has_cat, n_boxes, max_conf, fps, duration = scan
        n_hit = sum(has_cat)
        intervals = build_intervals(sample_times, has_cat, duration)
        kept = sum(e - s for s, e in intervals)
        removed = max(0.0, duration - kept)

        if SAVE_TIMELINE_CSV:
            with (run_dir / f"_timeline_{vp.stem}.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["time_s", "has_cat", "n_boxes", "max_conf"])
                for t, hc, nb, mc in zip(sample_times, has_cat, n_boxes, max_conf):
                    w.writerow([f"{t:.2f}", int(hc), nb, f"{mc:.3f}"])

        print(f"   片長 {duration:.1f}s  抽樣 {len(has_cat)} 點（{n_hit} 有貓）"
              f" → 保留 {len(intervals)} 段 共 {kept:.1f}s"
              f"（{kept / max(duration, 1e-6) * 100:.0f}%，剪掉 {removed:.1f}s）")
        for s, e in intervals:
            print(f"      [{s:8.2f} → {e:8.2f}]  ({e - s:.1f}s)")

        report[vp.stem] = {
            "video": str(vp), "duration_s": round(duration, 2), "fps": round(fps, 2),
            "samples": len(has_cat), "cat_samples": n_hit,
            "kept_intervals": [[s, e] for s, e in intervals],
            "kept_seconds": round(kept, 2),
            "removed_seconds": round(removed, 2),
        }

        if DRY_RUN:
            continue
        if not intervals:
            print("   （整支影片都沒偵測到貓，不輸出）")
            continue

        outs = cut_segments(vp, intervals, run_dir)
        if not outs:
            print("   ✗ 剪接失敗，原始檔保留不動")
            continue
        print("   ✓ " + "  ".join(o.name for o in outs))

        if DELETE_ORIGINAL:
            backup = run_dir / "_removed_originals"
            backup.mkdir(exist_ok=True)
            shutil.move(str(vp), str(backup / vp.name))
            print(f"   ↪ 原始檔已搬到 {backup / vp.name}（非刪除）")

    if report:
        (run_dir / "_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if DRY_RUN:
        print(f"\n（DRY_RUN：未剪未刪。時間軸 CSV 在 {run_dir}，確認區間對了再把 DRY_RUN 設 False）")
    else:
        print(f"\n✓ 完成 → {run_dir}")


if __name__ == "__main__":
    main()
