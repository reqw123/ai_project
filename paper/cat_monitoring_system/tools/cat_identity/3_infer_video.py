"""
3_infer_video.py — 用訓練好的身分 CNN 對影片做逐幀貓咪身分辨識 + 視覺化
================================================================================
獨立腳本，不 import 本專案任何自訂模組（不碰 Tracking / Pose / ST-GCN /
FrameProcessor）。只用 PyTorch + torchvision + OpenCV + ultralytics(YOLO)。

流程（每一幀）：
  1. YOLO 偵測畫面上所有貓的 bbox
  2. 每個 bbox 裁切區域 → MobileNetV3-Small 身分分類頭 → softmax
  3. 最高信心 < CONFIDENCE_THRESHOLD → 判為 "Unknown"（非訓練類別）
  4. 簡易 IoU 追蹤把同一隻貓在連續幀之間串起來，取最近 SMOOTH_WINDOW 幀的
     多數決當顯示結果，稀釋單幀雜訊
  5. 疊框 + 標籤畫回畫面：即時預覽視窗，可選存成標註影片，逐幀結果輸出 CSV

裁切/縮放/normalize 與 2_train.py 的 val/test transform 完全一致
（bbox 往外擴 CROP_PADDING_RATIO → Resize → CenterCrop → ImageNet normalize），
權重檔裡存的 image_size / norm 參數會被讀出來用，不寫死。

執行環境：需要 torch/torchvision/ultralytics（CUDA 版）。本機用專案 conda 環境：
    & "C:\\Users\\lynnc\\anaconda3\\envs\\yolo\\python.exe" 3_infer_video.py

用法：改下面「設定區」的路徑，然後 python 3_infer_video.py

控制鍵（SHOW_PREVIEW=True 時）：q=退出　space=暫停/繼續　1=上一部影片　2=下一部影片
"""

# ═══════════════════════════════════════════════════════════════
#  設定區
# ═══════════════════════════════════════════════════════════════

# ── 模型 ──
YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_149.pt"        # 抓貓 bbox 用（偵測/pose 皆可）
IDENTITY_MODEL_PATH = r"C:\ai_project\identity_models\run_20260829-2011\001.pt"  # 2_train.py 最近一次訓練的 best；要指定某次就填該 run 資料夾裡的 <流水號>.pt
DEVICE = "cuda"                # "cuda" 優先，抓不到自動 fallback "cpu"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5      # bbox 偵測信心門檻

# ── 輸入影片（檔案或資料夾，資料夾會遞迴掃描；可放多個）──
# 預設指向身分辨識驗證影片夾——放「沒進過訓練」的整支影片（見該資料夾 _README.txt）
SOURCE = [
    r"D:\貓咪顏色比對影\貓咪顏色比對影片",
]
# 若設定環境變數 TEST_VIDEO_PATH，優先只用該路徑（覆蓋上面的 SOURCE）
import os as _os
_env = _os.getenv("TEST_VIDEO_PATH", "").strip()
if _env:
    SOURCE = [_env]

# ── 身分判定 ──
CONFIDENCE_THRESHOLD = 0.80    # 平滑前的單幀 softmax 最高值低於此 → 該幀判 Unknown
CROP_PADDING_RATIO = 0.04      # bbox 往外擴的比例，需與 1_build_dataset.py 訓練時一致
MIN_BBOX_SIZE = 24             # bbox 邊長小於此（px）直接跳過，太小分類不可靠

# ── 跨幀平滑（緩解單幀雜訊）──
SMOOTH_WINDOW = 9              # 每個追蹤目標取最近幾幀的原始判定做多數決
TRACK_IOU_THRES = 0.3         # 這幀 bbox 跟上一幀哪個追蹤是同一隻貓的 IoU 門檻
TRACK_MAX_MISSED = 10         # 追蹤連續幾幀沒配對到偵測才捨棄

# ── 顯示名稱（OpenCV 畫面不支援中文，用英文短名；中文全名仍寫進 CSV/console）──
# key = 權重檔 class_names 裡的名字；沒對到的類別用 "cls{index}"
DISPLAY_NAMES = {
    "目標貓": "target",
    "他貓": "other",
}
CLASS_COLORS = {                # BGR，預覽框線 / 標籤底色
    "目標貓": (0, 200, 0),
    "他貓": (0, 140, 255),
}
UNKNOWN_COLOR = (120, 120, 120)

# ── 輸出 ──
import pathlib as _pathlib
OUTPUT_ROOT = _pathlib.Path(
    r"C:\ai_project\paper\cat_monitoring_system\tools\train_data\cat_identity\infer"
)
SAVE_ANNOTATED_VIDEO = True    # 每支影片輸出一份疊好框的 mp4
SAVE_CSV = True               # 每支影片輸出逐幀 CSV

SHOW_PREVIEW = True
PREVIEW_DISPLAY_SIZE = (1280, 720)
PREVIEW_WINDOW_NAME = "Cat Identity CNN Inference (q=quit  space=pause  1/2=prev/next)"

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
# ═══════════════════════════════════════════════════════════════

import sys
import csv
import json
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

_MISSING = []
try:
    import numpy as np
except ImportError:
    _MISSING.append("numpy")
try:
    import cv2
except ImportError:
    _MISSING.append("opencv-python")
try:
    import torch
    import torch.nn as nn
except ImportError:
    _MISSING.append("torch")
try:
    from torchvision import transforms
    from torchvision.models import mobilenet_v3_small
except ImportError:
    _MISSING.append("torchvision")
try:
    from PIL import Image
except ImportError:
    _MISSING.append("pillow")
try:
    from ultralytics import YOLO
except ImportError:
    _MISSING.append("ultralytics")

if _MISSING:
    print("❌ 缺少套件：", ", ".join(_MISSING))
    print(r'   本機請用：& "C:\Users\lynnc\anaconda3\envs\yolo\python.exe" 3_infer_video.py')
    sys.exit(1)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════
#  模型載入
# ═══════════════════════════════════════════════════════════════
def get_device():
    if DEVICE.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if DEVICE.lower() == "cuda":
        print("⚠ 找不到可用的 CUDA，改用 CPU")
    return torch.device("cpu")


def load_yolo(device):
    model = YOLO(YOLO_MODEL_PATH)
    try:
        model.to(str(device))
    except Exception as e:
        print(f"⚠ YOLO 無法移到 {device}（{e}）")
    return model


def load_identity_model(device):
    """回傳 (model, class_names, image_size, transform)。結構與
    2_train.py 的 build_model_for_inference 一致。"""
    ckpt = torch.load(IDENTITY_MODEL_PATH, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    image_size = int(ckpt.get("image_size", 128))
    mean = ckpt.get("norm_mean", IMAGENET_MEAN)
    std = ckpt.get("norm_std", IMAGENET_STD)

    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    resize_to = int(round(image_size * 1.25))
    transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    print(f"身分模型：{class_names}  image_size={image_size}  (val_acc={ckpt.get('val_acc')})")
    return model, class_names, image_size, transform


# ═══════════════════════════════════════════════════════════════
#  偵測 + 分類
# ═══════════════════════════════════════════════════════════════
def detect_boxes(yolo, frame):
    """回傳這一幀所有偵測到的貓：[(bbox(x1,y1,x2,y2) float ndarray, conf float), ...]"""
    results = yolo.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    out = []
    if results.boxes is None or len(results.boxes) == 0:
        return out
    for i in range(len(results.boxes)):
        bbox = results.boxes.xyxy[i].cpu().numpy()
        conf = float(results.boxes.conf[i].cpu().numpy())
        out.append((bbox, conf))
    return out


def crop_bbox(frame, bbox, padding_ratio):
    """依 bbox 裁切、往外擴 padding_ratio 後夾在畫面內。回傳 BGR ndarray 或 None。
    與 1_build_dataset.py 的 crop_bbox 邏輯一致。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_BBOX_SIZE or bh < MIN_BBOX_SIZE:
        return None
    px, py = bw * padding_ratio, bh * padding_ratio
    cx1 = int(np.clip(x1 - px, 0, w - 1))
    cy1 = int(np.clip(y1 - py, 0, h - 1))
    cx2 = int(np.clip(x2 + px, 0, w))
    cy2 = int(np.clip(y2 + py, 0, h))
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return None
    return frame[cy1:cy2, cx1:cx2]


@torch.no_grad()
def classify_crops(model, transform, crops_bgr, device):
    """一批 BGR 裁切圖 → [(pred_idx, probs ndarray), ...]，一次 forward。"""
    if not crops_bgr:
        return []
    tensors = []
    for c in crops_bgr:
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb)))
    batch = torch.stack(tensors).to(device)
    probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    return [(int(p.argmax()), p) for p in probs]


# ═══════════════════════════════════════════════════════════════
#  簡易 IoU 追蹤 + 多數決平滑
# ═══════════════════════════════════════════════════════════════
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, bboxes):
        assigned = [None] * len(bboxes)
        used = set()
        for i, bbox in enumerate(bboxes):
            if bbox is None:
                continue
            best_tid, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                v = _iou(t["bbox"], bbox)
                if v > best_iou:
                    best_tid, best_iou = tid, v
            if best_tid is not None and best_iou >= TRACK_IOU_THRES:
                assigned[i] = best_tid
                used.add(best_tid)
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["missed"] = 0
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": bbox, "history": deque(maxlen=SMOOTH_WINDOW), "missed": 0}
                assigned[i] = tid
                used.add(tid)
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > TRACK_MAX_MISSED:
                    del self.tracks[tid]
        return assigned

    def record(self, tid, label):
        if tid in self.tracks:
            self.tracks[tid]["history"].append(label)

    def smoothed(self, tid):
        if tid not in self.tracks or not self.tracks[tid]["history"]:
            return None
        return Counter(self.tracks[tid]["history"]).most_common(1)[0][0]


# ═══════════════════════════════════════════════════════════════
#  繪圖
# ═══════════════════════════════════════════════════════════════
def _display_name(class_name):
    return DISPLAY_NAMES.get(class_name, None)


def draw_label_block(img, x, y, text, border_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.8, 2
    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
    pad = 6
    y1 = max(0, y - tht - pad * 2)
    x2 = min(img.shape[1], x + tw + pad * 2)
    ov = img.copy()
    cv2.rectangle(ov, (x, y1), (x2, y1 + tht + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x, y1), (x2, y1 + tht + pad * 2), border_color, 2)
    cv2.putText(img, text, (x + pad, y1 + tht + pad), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def resize_letterbox(image, target):
    tw, th = target
    sh, sw = image.shape[:2]
    if sw <= 0 or sh <= 0:
        return image
    s = min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * s)), max(1, int(sh * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(image, (nw, nh), interpolation=interp)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas[py:py + nh, px:px + nw] = r
    return canvas


def render_frame(frame, decisions, class_names, frame_idx, video_name, paused, fps_txt):
    disp = frame.copy()
    for d in decisions:
        if d["bbox"] is None:
            continue
        x1, y1, x2, y2 = map(int, d["bbox"])
        label = d["label"]  # class name str 或 None(=Unknown)
        if label is not None:
            color = CLASS_COLORS.get(label, (0, 200, 0))
            short = _display_name(label) or f"cls{class_names.index(label)}"
        else:
            color, short = UNKNOWN_COLOR, "unknown"
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)
        conf_txt = f"{d['cnn_conf']:.2f}" if d["cnn_conf"] is not None else "--"
        draw_label_block(disp, x1, y1, f"{short} {conf_txt}", color)
    status = "PAUSED" if paused else "PLAY"
    cv2.putText(
        disp,
        f"[{status}] {video_name}  f{frame_idx}  cats={len(decisions)}  {fps_txt}",
        (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return disp


# ═══════════════════════════════════════════════════════════════
#  單幀處理
# ═══════════════════════════════════════════════════════════════
def process_frame(frame, yolo, model, transform, class_names, tracker, device):
    dets = detect_boxes(yolo, frame)

    crops, crop_map = [], []
    decisions = []
    for idx, (bbox, bconf) in enumerate(dets):
        crop = crop_bbox(frame, bbox, CROP_PADDING_RATIO)
        decisions.append({
            "inst_idx": idx, "bbox": bbox, "bbox_conf": bconf,
            "raw_label": None, "cnn_conf": None, "probs": None,
        })
        if crop is not None:
            crops.append(crop)
            crop_map.append(idx)

    for (pred_idx, probs), inst_idx in zip(classify_crops(model, transform, crops, device), crop_map):
        top_p = float(probs[pred_idx])
        d = decisions[inst_idx]
        d["probs"] = probs
        d["cnn_conf"] = top_p
        d["raw_label"] = class_names[pred_idx] if top_p >= CONFIDENCE_THRESHOLD else None

    # 追蹤 + 多數決平滑
    tids = tracker.update([d["bbox"] for d in decisions])
    for d, tid in zip(decisions, tids):
        d["track_id"] = tid
        if tid is not None:
            tracker.record(tid, d["raw_label"])
            d["label"] = tracker.smoothed(tid)
        else:
            d["label"] = None

    # 同幀兩個框被判成同一隻貓 → 只留 CNN 信心最高的，其餘降級 Unknown
    by_label = defaultdict(list)
    for d in decisions:
        if d["label"] is not None:
            by_label[d["label"]].append(d)
    for _lbl, ds in by_label.items():
        if len(ds) > 1:
            keep = max(ds, key=lambda x: x["cnn_conf"] if x["cnn_conf"] is not None else -1)
            for d in ds:
                if d is not keep:
                    d["label"] = None
    return decisions


# ═══════════════════════════════════════════════════════════════
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


def main():
    device = get_device()
    yolo = load_yolo(device)
    model, class_names, image_size, transform = load_identity_model(device)

    videos = resolve_videos(SOURCE)
    if not videos:
        print("❌ 找不到任何影片")
        return

    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"裝置: {device}   輸出資料夾: {run_dir}\n影片數: {len(videos)}")

    if SHOW_PREVIEW:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, *PREVIEW_DISPLAY_SIZE)
        print("控制鍵: q=退出　space=暫停/繼續　1=上一部　2=下一部")

    summary = {}
    vi = 0
    stop = False
    processed = set()
    while not stop:
        # headless（無預覽視窗）：每支影片跑一次就結束，不進入 1/2 循環導覽
        if not SHOW_PREVIEW and len(processed) >= len(videos):
            break
        vp = videos[vi]
        print(f"\n[{vi + 1}/{len(videos)}] {vp.name}")
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print("  ⚠ 無法開啟，跳過")
            processed.add(vi)
            vi = (vi + 1) % len(videos)
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if SAVE_ANNOTATED_VIDEO:
            writer = cv2.VideoWriter(
                str(run_dir / f"{vp.stem}_annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H),
            )

        tracker = SimpleTracker()
        csv_rows = []
        counts = {c: 0 for c in class_names}
        counts["unknown"] = 0
        frame_idx = 0
        paused = False
        switch = 1
        last_disp = None
        t_prev = datetime.now()
        fps_txt = ""

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    switch = 1
                    break
                frame_idx += 1
                decisions = process_frame(frame, yolo, model, transform, class_names, tracker, device)

                for d in decisions:
                    lbl = d["label"] if d["label"] is not None else "unknown"
                    if lbl in counts:
                        counts[lbl] += 1
                    row = {
                        "frame": frame_idx, "inst_idx": d["inst_idx"], "track_id": d["track_id"],
                        "bbox_conf": f"{d['bbox_conf']:.3f}" if d["bbox_conf"] is not None else "",
                        "label": lbl,
                        "cnn_conf": f"{d['cnn_conf']:.3f}" if d["cnn_conf"] is not None else "",
                        "bbox": ",".join(f"{v:.0f}" for v in d["bbox"]) if d["bbox"] is not None else "",
                    }
                    for ci, cname in enumerate(class_names):
                        row[f"p_{DISPLAY_NAMES.get(cname, 'cls'+str(ci))}"] = (
                            f"{d['probs'][ci]:.3f}" if d["probs"] is not None else ""
                        )
                    csv_rows.append(row)

                if frame_idx % 30 == 0:
                    now = datetime.now()
                    fps_txt = f"{30 / max((now - t_prev).total_seconds(), 1e-6):.1f} fps"
                    t_prev = now

                disp_full = render_frame(frame, decisions, class_names, frame_idx, vp.name, paused, fps_txt)
                if writer is not None:
                    writer.write(disp_full)
                if SHOW_PREVIEW:
                    last_disp = resize_letterbox(disp_full, PREVIEW_DISPLAY_SIZE)

            if SHOW_PREVIEW:
                if last_disp is not None:
                    cv2.imshow(PREVIEW_WINDOW_NAME, last_disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    stop = True
                    break
                if k == ord(" "):
                    paused = not paused
                elif k == ord("2"):
                    switch = 1
                    break
                elif k == ord("1"):
                    switch = -1
                    break

        cap.release()
        if writer is not None:
            writer.release()

        if SAVE_CSV and csv_rows:
            fields = ["frame", "inst_idx", "track_id", "bbox_conf", "label", "cnn_conf",
                      *[k for k in csv_rows[0] if k.startswith("p_")], "bbox"]
            with (run_dir / f"{vp.stem}_identity.csv").open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(csv_rows)

        denom = max(frame_idx, 1)
        summary[vp.stem] = {
            "video": vp.name, "frames": frame_idx,
            **{f"{c}_frames": counts[c] for c in counts},
            **{f"{c}_pct": round(counts[c] / denom * 100, 1) for c in counts},
        }
        print(f"  幀數 {frame_idx}  " + "  ".join(
            f"{c}:{counts[c]}({counts[c] / denom * 100:.0f}%)" for c in counts
        ))

        (run_dir / "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        processed.add(vi)
        if stop:
            break
        if not SHOW_PREVIEW:
            vi += 1
            if vi >= len(videos):
                break
        else:
            vi = (vi + switch) % len(videos)

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print(f"\n✓ 完成，輸出於: {run_dir}")


if __name__ == "__main__":
    main()
