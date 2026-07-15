"""
靜止偵測測試腳本
模式 1：背景 CSV  — 自動逐幀處理所有影片，輸出 CSV，無 GUI
模式 2：GUI 測試  — 互動視窗，骨架 / bbox / 靜止標語，1/2 鍵切換影片
操作（GUI）：q/ESC=結束  Space=暫停  r=重播  1=上一支  2=下一支  s=套用調整
終端（GUI）：輸入 'motion <值>' 或 'mean <整數>' 後按 s 套用，停止後自動還原
"""
import re
import sys
import csv
import glob
import os
import time
import threading
import ctypes
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detectors.keypoint_detector import KeypointDetector
from processors.anomaly_detector import AnomalyDetector
from utils.constants import (
    EAR_DISTANCE_SKELETON_EDGES,
    EAR_DISTANCE_EDGE_COLORS,
    EAR_DISTANCE_KP_COLORS,
    BLACK,
    COLOR_HEAD,
)
from config import ModelPaths

# ── 設定 ──────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH  = ModelPaths.YOLO_MODEL
VIDEO_FOLDER     = r"C:\Users\homec\Downloads\5562"  # 影片資料夾
CSV_OUTPUT_DIR   = r"C:\ai_project\paper\still_analysis"
DEVICE           = "cuda"
YOLO_IMGSZ       = 640
YOLO_CONF        = 0.5
WINDOW_NAME      = "Still Detection Test"
DISPLAY_W        = 1080
DISPLAY_H        = 720
MOTION_THRESHOLD = 0.30   # rolling_mean 低於此值（body_fraction×10）判為靜止（終端指令：motion）
ROLLING_WINDOW   = 16    # 滾動均值視窗大小（幀數）
STRIDE           = 1     # 每 N 幀才寫入一次視窗（1=每幀，2=隔幀）
KP_CONF_THRES    = 0.5   # 關鍵點信心門檻（骨架繪製 & 位移計算）
VIDEO_EXTS       = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI")
# ─────────────────────────────────────────────────────────────────────────────

BANNER_H = 56
FONT_D   = cv2.FONT_HERSHEY_DUPLEX
FONT_S   = cv2.FONT_HERSHEY_SIMPLEX

_BEHAVIOR_PATTERN = re.compile(r"(walk|lick|scratch|shake|stop)", re.IGNORECASE)

# ── 即時調整狀態（模式 2 專用）────────────────────────────────────────────────
_pending      = {}         # {'motion': float, 'mean': int}
_pending_lock = threading.Lock()
_gui_running  = [False]    # list 讓子執行緒可寫入，無需 global


def _terminal_reader():
    print("\n  [即時調整] 指令：motion <小數> / mean <整數> / stride <整數>，視窗按 s 套用")
    print("  停止後自動還原預設值\n")
    while _gui_running[0]:
        try:
            line = input().strip()
        except EOFError:
            break
        if not line or not _gui_running[0]:
            continue
        parts = line.split()
        if len(parts) != 2:
            print("  格式：motion <小數> / mean <整數> / stride <整數>")
            continue
        cmd, raw = parts[0].lower(), parts[1]
        try:
            with _pending_lock:
                if cmd == "motion":
                    _pending["motion"] = np.float64(raw)
                    print(f"  [待套用] MOTION_THRESHOLD → {_pending['motion']:.2f}  (視窗按 s)")
                elif cmd == "mean":
                    _pending["mean"] = max(2, int(raw))
                    print(f"  [待套用] ROLLING_WINDOW → {_pending['mean']}  (視窗按 s)")
                elif cmd == "stride":
                    _pending["stride"] = max(1, int(raw))
                    print(f"  [待套用] STRIDE → {_pending['stride']}  (視窗按 s)")
                else:
                    print("  未知指令，支援：motion / mean / stride")
        except ValueError:
            print(f"  無效數值：{raw}")


# ── 共用工具 ──────────────────────────────────────────────────────────────────

def parse_behavior_label(video_path):
    m = _BEHAVIOR_PATTERN.search(Path(video_path).stem)
    return m.group(1).lower() if m else "unknown"


def load_videos(folder):
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)


def open_csv(video_path):
    Path(CSV_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    stem  = Path(video_path).stem
    ts    = datetime.now().strftime("%H%M%S")
    out   = Path(CSV_OUTPUT_DIR) / f"{stem}_{ts}.csv"
    label = parse_behavior_label(video_path)
    f     = open(out, "w", newline="", encoding="utf-8")
    w     = csv.writer(f)
    w.writerow(["frame_idx", "motion_norm", "rolling_mean_norm", "is_still", "detected", "behavior_label"])
    return f, w, str(out), label


def analyze_and_print(csv_path):
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["detected"] == "1":
                    rows.append(r["rolling_mean_px"])
    except Exception as e:
        print(f"  [分析失敗] {e}")
        return
    if not rows:
        print("  [無偵測幀，略過分析]")
        return

    arr = np.array(rows, dtype="float64")
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n  rolling_mean 分布（共 {len(arr)} 有效幀，單位：body_fraction）")
    print(f"  min={arr.min():.4f}  p5={p5:.4f}  p25={p25:.4f}  "
          f"p50={p50:.4f}  p75={p75:.4f}  p95={p95:.4f}  max={arr.max():.4f}")
    print(f"  ▶ 建議靜止門檻區間：p25({p25:.4f}) ～ p50({p50:.4f})")
    print(f"  ▶ 保守門檻（少誤報）：p5({p5:.4f})\n")

    lo, hi = arr.min(), arr.max()
    bins = np.linspace(lo, hi, 11)
    counts, _ = np.histogram(arr, bins=bins)
    max_c = max(counts) or 1
    print("  ASCII 直方圖（rolling_mean px）：")
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_c * 30)
        print(f"  {bins[i]:6.2f}-{bins[i+1]:6.2f} | {bar} ({c})")
    print()


def _write_csv_row(csv_w, frame_idx, anomaly, detected, cur_label):
    win          = anomaly._motion_window
    rolling_mean = sum(win) / len(win) if len(win) >= 2 else 0.0
    motion       = anomaly.last_motion_score
    # is_still 跟隨 anomaly 當前的 _still_threshold（可能已被即時調整）
    is_still_csv = len(win) >= 2 and rolling_mean < anomaly._still_threshold
    csv_w.writerow([
        frame_idx,
        f"{motion:.4f}",
        f"{rolling_mean:.4f}",
        "1" if is_still_csv else "0",
        "1" if detected else "0",
        cur_label,
    ])
    return rolling_mean, motion


# ── GUI 繪圖輔助 ───────────────────────────────────────────────────────────────

def _draw_skeleton(frame, kpts, kpt_conf):
    thres = KP_CONF_THRES
    for idx, (i, j) in enumerate(EAR_DISTANCE_SKELETON_EDGES):
        if i >= len(kpts) or j >= len(kpts):
            continue
        if kpt_conf[i] > thres and kpt_conf[j] > thres:
            c = EAR_DISTANCE_EDGE_COLORS[idx] if idx < len(EAR_DISTANCE_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, tuple(map(int, kpts[i])), tuple(map(int, kpts[j])), c, 3, cv2.LINE_AA)
    for i, pt in enumerate(kpts):
        if kpt_conf[i] > thres:
            c = EAR_DISTANCE_KP_COLORS[i] if i < len(EAR_DISTANCE_KP_COLORS) else (200, 200, 200)
            p = tuple(map(int, pt))
            cv2.circle(frame, p, 5, c, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 5, BLACK, 1, cv2.LINE_AA)


def _draw_bbox(frame, bbox, conf):
    if bbox is None:
        return
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, 5, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, 3, cv2.LINE_AA)
    if conf is not None:
        lbl = f"cat {conf:.2f}"
        cv2.putText(frame, lbl, (x1, max(y1 - 10, BANNER_H + 18)),
                    FONT_S, 0.60, BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, lbl, (x1, max(y1 - 10, BANNER_H + 18)),
                    FONT_S, 0.60, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_top_banner(frame, is_still, motion, rolling_mean, frame_idx,
                     vid_idx, vid_total, vid_name, label,
                     cur_thresh, cur_win, cur_stride, has_pending):
    w = frame.shape[1]
    bar_color = (80, 80, 80) if is_still else (0, 150, 0)
    cv2.rectangle(frame, (0, 0), (w, BANNER_H), bar_color, -1)
    cv2.rectangle(frame, (0, 0), (w, BANNER_H), (20, 20, 20), 1)
    status = "STILL" if is_still else "ACTIVE"
    cv2.putText(frame, status, (14, BANNER_H - 13),
                FONT_D, 1.20, (255, 255, 255), 2, cv2.LINE_AA)
    pending_mark = "  [*]" if has_pending else ""
    info = (f"motion={motion:.4f}  mean={rolling_mean:.4f}  "
            f"thresh={cur_thresh:.4f}  win={cur_win}  stride={cur_stride}{pending_mark}  |  "
            f"frame={frame_idx}  [{vid_idx+1}/{vid_total}] {vid_name}  [{label}]")
    cv2.putText(frame, info, (160, BANNER_H - 17),
                FONT_S, 0.54, (210, 210, 210), 1, cv2.LINE_AA)


def _draw_frame_number(frame, frame_idx):
    txt = f"#{frame_idx}"
    cv2.putText(frame, txt, (12, DISPLAY_H - 12), FONT_D, 0.90, BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, txt, (12, DISPLAY_H - 12), FONT_D, 0.90, (255, 255, 0), 2, cv2.LINE_AA)


# ── 模式 1：背景 CSV ──────────────────────────────────────────────────────────

def run_background(videos, detector):
    csv_paths = []
    t0 = time.time()
    for i, path in enumerate(videos):
        label = parse_behavior_label(path)
        cap   = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[SKIP] 無法開啟：{path}")
            continue

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        anomaly = AnomalyDetector(still_threshold=MOTION_THRESHOLD,
                                  rolling_window=ROLLING_WINDOW,
                                  stride=STRIDE,
                                  kp_conf_thres=KP_CONF_THRES)
        csv_f, csv_w, csv_path, _ = open_csv(path)
        csv_paths.append(csv_path)
        print(f"\n[{i+1}/{len(videos)}] {Path(path).name}  label={label}")
        print(f"  → {csv_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LANCZOS4)

                kpts, kpt_conf, _, _ = detector.detect(frame)
                detected = kpts is not None
                anomaly.detect(kpts, kpt_conf) if detected else anomaly.detect(None, None)

                _write_csv_row(csv_w, frame_idx, anomaly, detected, label)

                if frame_idx % 30 == 0:
                    csv_f.flush()
                    pct = frame_idx / total * 100
                    print(f"  {frame_idx}/{total}  ({pct:.0f}%)", end="\r", flush=True)
        finally:
            cap.release()
            if not csv_f.closed:
                csv_f.flush()
                csv_f.close()

        print(f"  完成  {frame_idx} 幀{' ' * 20}")

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  總耗時：{mins}m {secs:02d}s（{elapsed:.1f}s）")
    print("\n" + "=" * 60)
    print("  rolling_mean 分布分析（協助設定 MOTION_THRESHOLD）")
    print("=" * 60)
    for cp in csv_paths:
        print(f"\n  檔案：{Path(cp).name}")
        analyze_and_print(cp)


# ── 模式 2：GUI 測試 ───────────────────────────────────────────────────────────

def run_gui(videos, detector):
    csv_paths = []

    def open_video(idx, thresh=MOTION_THRESHOLD, win=ROLLING_WINDOW, strd=STRIDE):
        """開啟影片，用傳入的參數建立偵測器（切換影片時保留即時調整值）。"""
        path = videos[idx]
        cap  = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[ERROR] 無法開啟：{path}")
            return None, None, None, None, None, "unknown"
        ano             = AnomalyDetector(still_threshold=thresh,
                                          rolling_window=win,
                                          stride=strd,
                                          kp_conf_thres=KP_CONF_THRES)
        cf, cw, cp, lbl = open_csv(path)
        csv_paths.append(cp)
        print(f"\n[VIDEO] {Path(path).name}  label={lbl}  →  CSV: {cp}")
        return cap, ano, cf, cw, cp, lbl

    # 啟動終端輸入執行緒
    _gui_running[0] = True
    t = threading.Thread(target=_terminal_reader, daemon=True)
    t.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_W, DISPLAY_H)

    vid_idx = 0
    cap, anomaly, csv_f, csv_w, _, cur_label = open_video(vid_idx)
    frame_idx       = 0
    paused          = False
    _last_switch_t  = 0.0   # 防抖：避免 key repeat 造成連跳兩支影片
    _VK_1, _VK_2   = 0x31, 0x32          # Windows virtual-key codes for '1' / '2'
    _sw1_prev = _sw2_prev = False         # 上緣偵測用的前幀狀態

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            # GetAsyncKeyState：全域偵測 1/2，不受視窗焦點影響
            _sw1 = bool(ctypes.windll.user32.GetAsyncKeyState(_VK_1) & 0x8000)
            _sw2 = bool(ctypes.windll.user32.GetAsyncKeyState(_VK_2) & 0x8000)
            sw1_edge = _sw1 and not _sw1_prev   # 上緣：剛按下
            sw2_edge = _sw2 and not _sw2_prev
            _sw1_prev, _sw2_prev = _sw1, _sw2

            if key in (ord('q'), 27):
                break

            if key == ord(' '):
                paused = not paused

            # r：回到影片開頭並清空所有偵測狀態（保留門檻設定）
            if key == ord('r'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                anomaly.reset()
                frame_idx = 0

            # s：套用終端待套用的調整
            if key == ord('s'):
                with _pending_lock:
                    applied = []
                    if "motion" in _pending:
                        anomaly._still_threshold = _pending.pop("motion")
                        applied.append(f"MOTION_THRESHOLD={anomaly._still_threshold:.2f}")
                    if "mean" in _pending:
                        new_win  = _pending.pop("mean")
                        old_data = list(anomaly._motion_window)[-new_win:]
                        anomaly._motion_window = deque(old_data, maxlen=new_win)
                        applied.append(f"ROLLING_WINDOW={new_win}")
                    if "stride" in _pending:
                        anomaly._stride = _pending.pop("stride")
                        anomaly._stride_count = 0
                        applied.append(f"STRIDE={anomaly._stride}")
                if applied:
                    print(f"  ✅ 已套用：{', '.join(applied)}")
                else:
                    print("  （無待套用的變更）")

            switch = None
            if time.time() - _last_switch_t > 0.3:   # 300 ms 防抖，避免 key repeat 連跳
                if sw1_edge or key == ord('1'):
                    switch = (vid_idx - 1) % len(videos)
                elif sw2_edge or key == ord('2'):
                    switch = (vid_idx + 1) % len(videos)

            if switch is not None:
                # 立即顯示切換畫面，讓使用者確認按鍵已生效（codec 初始化前）
                _sw_frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                _next_name = Path(videos[switch]).name
                cv2.putText(_sw_frame,
                            f"Loading [{vid_idx+1} -> {switch+1}]  {_next_name}",
                            (30, DISPLAY_H // 2), FONT_S, 0.80,
                            (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, _sw_frame)
                cv2.waitKey(1)   # 刷新顯示並清除按鍵緩衝

                # 切換影片：繼承門檻 / 視窗 / stride 設定，但清空所有偵測狀態
                t_now, w_now, s_now = anomaly._still_threshold, anomaly._motion_window.maxlen, anomaly._stride
                cap.release()
                csv_f.flush(); csv_f.close()
                vid_idx = switch
                cap, anomaly, csv_f, csv_w, _, cur_label = open_video(vid_idx,
                                                                        thresh=t_now,
                                                                        win=w_now,
                                                                        strd=s_now)
                # open_video 建立全新 AnomalyDetector，motion_window / prev_kpts 已清空
                _last_switch_t = time.time()
                frame_idx = 0
                paused    = False
                if cap is None:
                    break
                continue

            if paused:
                continue

            ret, frame = cap.read()
            if not ret:
                # 影片結束重播
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                anomaly.reset()
                frame_idx = 0
                continue

            frame_idx += 1
            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LANCZOS4)

            kpts, kpt_conf, bbox, conf = detector.detect(frame)
            detected = kpts is not None
            if detected:
                is_still, _ = anomaly.detect(kpts, kpt_conf)
            else:
                anomaly.detect(None, None)
                is_still = False

            rolling_mean, motion = _write_csv_row(csv_w, frame_idx, anomaly, detected, cur_label)
            if frame_idx % 30 == 0:
                csv_f.flush()

            if detected:
                _draw_skeleton(frame, kpts, kpt_conf)
                _draw_bbox(frame, bbox, conf)

            with _pending_lock:
                has_p = bool(_pending)
            _draw_top_banner(frame, is_still, motion, rolling_mean,
                             frame_idx, vid_idx, len(videos),
                             Path(videos[vid_idx]).name, cur_label,
                             anomaly._still_threshold, anomaly._motion_window.maxlen,
                             anomaly._stride, has_p)
            _draw_frame_number(frame, frame_idx)
            cv2.imshow(WINDOW_NAME, frame)

    finally:
        _gui_running[0] = False
        with _pending_lock:
            _pending.clear()
        cap.release()
        if csv_f and not csv_f.closed:
            csv_f.flush()
            csv_f.close()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  rolling_mean 分布分析（協助設定 MOTION_THRESHOLD）")
    print("=" * 60)
    for cp in csv_paths:
        print(f"\n  檔案：{Path(cp).name}")
        analyze_and_print(cp)


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    videos = load_videos(VIDEO_FOLDER)
    if not videos:
        print(f"[ERROR] 資料夾內找不到影片：{VIDEO_FOLDER}")
        return
    print(f"[INFO] 載入 {len(videos)} 支影片")
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {Path(v).name}  [{parse_behavior_label(v)}]")

    # 支援命令列參數：python test_anomaly_detection.py 1 / 2
    if len(sys.argv) > 1 and sys.argv[1] in ("1", "2"):
        mode = sys.argv[1]
    else:
        print("\n選擇模式：")
        print("  1  背景 CSV（自動處理所有影片，無 GUI）")
        print("  2  GUI 測試（互動視窗，骨架顯示，支援即時調整 motion/mean）")
        mode = input("輸入 1 或 2：").strip()

    if mode not in ("1", "2"):
        print("[ERROR] 無效選擇，結束。")
        return

    detector = KeypointDetector(YOLO_MODEL_PATH, device=DEVICE,
                                imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF)

    if mode == "1":
        run_background(videos, detector)
    else:
        run_gui(videos, detector)


if __name__ == "__main__":
    main()
