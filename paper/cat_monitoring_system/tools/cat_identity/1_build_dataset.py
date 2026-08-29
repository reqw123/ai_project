"""
貓咪身分資料集收集工具（YOLO 定位 + 逐幀抽圖，同時保留 bbox 裁切圖與整張原圖）
==========================================================================
用途：拿現有的 YOLO(-Pose) 模型，把「目標貓 / 他貓」兩個資料夾底下的影片
逐幀跑一遍，通過品質關卡的幀就同時存兩份：
  - bbox 裁切圖 -> <OUTPUT_ROOT>/crops/<類別>/
  - 整張原始畫面 -> <OUTPUT_ROOT>/frames/<類別>/
兩者檔名對應（同一幀的 crop 與 frame 檔名前綴相同），可自由取用。
（SAVE_CROP / SAVE_FULL_FRAME 可各自關掉。）

產生的資料可拿去：
  - 訓練身分辨識 CNN：crops/<類別>/ 直接餵給 cat_identity/2_train.py
    （DATASET_PATH 指到 <OUTPUT_ROOT>/crops）
  - 整張原圖可拿去做偵測標註資料集、或之後自己重新裁切

品質關卡（一幀要全部通過才會取樣）：
  1) 這一幀 YOLO「剛好」偵測到一隻貓（0 隻或 >=2 隻整幀跳過，避免混入
     空景雜訊或另一隻貓）——REQUIRE_SINGLE_CAT=False 可放寬成多貓時每隻
     各存一張裁切圖（整張原圖仍只存一份）
  2) bbox 偵測信心 >= MIN_BBOX_CONF
  3) bbox 高度 >= 畫面高度 * MIN_BBOX_HEIGHT_RATIO（太遠/太小隻不要）
  4) 裁切區域 Laplacian 變異數 >= MIN_SHARPNESS（動態模糊/失焦不要）

用法：不用下指令列參數，改下面「使用者設定區」的路徑跟參數，然後
      python 1_build_dataset.py 執行。
      影片來源預設走 INPUT_ROOT：指到一個「底下剛好有 目標貓/ 和 他貓/ 兩個子
      資料夾」的父資料夾即可。開跑前會做防呆檢查（父資料夾存在、兩個子資料夾
      都在、且各自至少掃得到一支影片），缺任一項就直接報錯停止，不會產生只有
      半邊的資料集。要各類分開指定路徑時把 INPUT_ROOT 設成 "" 改用 CLASS_VIDEO_DIRS。
      預設 RESUME_SKIP_DONE_VIDEOS=True，同一支影片已經處理過就整支跳過，
      可以分批累積、隨時中斷再續跑。
"""
import os
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:  # tqdm 非必要，缺了就用不顯示進度條的替身
    def tqdm(x=None, **kwargs):
        return x if x is not None else iter(())

# ═══════════════════════════════════════════════════════
#  使用者設定區
# ═══════════════════════════════════════════════════════
YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_149.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5   # YOLO predict 的基礎信心門檻（品質關卡 ② 會再收更緊）

CLASS_DIR_NAMES = ("目標貓", "他貓")   # 兩類的子資料夾名稱（＝輸出子資料夾名稱），順序 = class index 0/1

# 【推薦用法】INPUT_ROOT 指到一個「底下剛好有 目標貓/ 和 他貓/ 兩個子資料夾」的父資料夾。
# 程式自動把 <INPUT_ROOT>/目標貓、<INPUT_ROOT>/他貓 當兩類來源，並在開跑前檢查
# 這兩個子資料夾都存在、且各自至少掃得到一支影片——缺一個就直接報錯停止，
# 不會產生只有半邊的資料集。
INPUT_ROOT = r"D:\目標貓採樣-20260828T210421Z-1-001\目標貓採樣"

# 【進階用法】把 INPUT_ROOT 設成 "" 時才會改用這個：手動指定每一類的影片檔/資料夾清單
# （key 必須剛好是 CLASS_DIR_NAMES 那兩個）。資料夾會被 resolve_video_paths() 遞迴掃描。
CLASS_VIDEO_DIRS = {
    "目標貓": [
        r"D:\目標貓採樣-20260828T210421Z-1-001\目標貓採樣\目標貓",
    ],
    "他貓": [
        r"D:\目標貓採樣-20260828T210421Z-1-001\目標貓採樣\他貓",
    ],
}

# 環境變數覆蓋（沒設定就用上面的預設）：
#   TEST_VIDEO_ROOT        → 覆蓋 INPUT_ROOT（一樣要求底下有 目標貓/他貓 兩夾）
#   TEST_VIDEO_PATH        → 同義於 TEST_VIDEO_ROOT。settings_window.py 的「獨立腳本工具」
#                            只會塞這個變數（值＝那邊選的影片資料夾），這裡把它當 INPUT_ROOT
#                            用：從設定視窗跑時，請選「底下有 目標貓/他貓 的那層父資料夾」。
#   TEST_VIDEO_PATH_TARGET → 進階模式下覆蓋「目標貓」來源清單
#   TEST_VIDEO_PATH_OTHER  → 進階模式下覆蓋「他貓」來源清單
_env_root = os.getenv("TEST_VIDEO_ROOT", "").strip() or os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_root:
    INPUT_ROOT = _env_root
_env_target = os.getenv("TEST_VIDEO_PATH_TARGET", "").strip()
_env_other = os.getenv("TEST_VIDEO_PATH_OTHER", "").strip()
if _env_target:
    CLASS_VIDEO_DIRS["目標貓"] = [_env_target]
if _env_other:
    CLASS_VIDEO_DIRS["他貓"] = [_env_other]

# 輸出根目錄。底下會是：
#   <OUTPUT_ROOT>/crops/目標貓/*.jpg     bbox 裁切圖
#   <OUTPUT_ROOT>/crops/他貓/*.jpg
#   <OUTPUT_ROOT>/frames/目標貓/*.jpg    同一幀的整張原始畫面（檔名前綴與 crop 對應）
#   <OUTPUT_ROOT>/frames/他貓/*.jpg
#   <OUTPUT_ROOT>/_manifest.csv          每個取樣一列（類別/來源影片/幀號/bbox/信心/清晰度/crop 檔/frame 檔）
#   <OUTPUT_ROOT>/_run_meta.json         這次執行的參數與統計
# 整個資料夾已加進 .gitignore，不進版控。
OUTPUT_ROOT = Path(r"C:\ai_project\paper\cat_monitoring_system\tools\train_data\cat_identity\dataset")

# ── 輸出內容 ──
SAVE_CROP = True         # 存 bbox 裁切圖 -> crops/<類別>/
SAVE_FULL_FRAME = True   # 存整張原始畫面 -> frames/<類別>/

# ── 取樣設定 ──
FRAME_STRIDE = 2            # 每隔幾個「通過品質關卡的有效幀」才實際取樣一次（相鄰幀太像，浪費空間）
MAX_CROPS_PER_CLASS = 3000  # 每一類最多取樣幾次（達到就停，不同影片平均分配靠先到先得）
CROP_PADDING_RATIO = 0.04   # 裁切時 bbox 往外擴的比例（保留一點邊緣輪廓，設 0 就是貼齊 bbox）
JPEG_QUALITY = 95

# ── 品質關卡（每張進資料集的裁切圖都要清楚、單獨入鏡、不模糊）──
REQUIRE_SINGLE_CAT = True    # True：一幀偵測到的貓 != 1 隻就整幀跳過
MIN_BBOX_CONF = 0.6          # 品質關卡 ②：bbox 偵測信心門檻
MIN_BBOX_HEIGHT_RATIO = 0.15 # 品質關卡 ③：bbox 高度至少佔畫面高度的比例
MIN_SHARPNESS = 60.0         # 品質關卡 ④：裁切區域 Laplacian 變異數下限

# ── 續跑 ──
RESUME_SKIP_DONE_VIDEOS = True   # True：輸出目錄已有某支影片的裁切圖，就整支跳過

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
# ═══════════════════════════════════════════════════════


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite 在 Windows 對非 ASCII 路徑（例如「目標貓」資料夾）會失敗，
    改用 cv2.imencode + ndarray.tofile。成功回傳 True。"""
    p = Path(path)
    try:
        ok, buf = cv2.imencode(p.suffix, img, params or [])
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False


def resolve_video_paths(sources):
    """來源可為影片檔或資料夾，展開成影片檔路徑清單（資料夾遞迴掃描）。"""
    resolved = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            resolved.append(p)
        elif p.is_dir():
            resolved.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_VIDEO_EXTS))
        else:
            print(f"⚠ 略過無效路徑: {p}")
    return resolved


def resolve_class_sources():
    """決定兩類的影片來源並做防呆檢查，回傳 {class_name: [已解析的影片路徑, ...]}。

    - INPUT_ROOT 有值：來源固定為 <INPUT_ROOT>/目標貓、<INPUT_ROOT>/他貓。
      INPUT_ROOT 不是資料夾、或底下缺少任一子資料夾、或某一類掃不到影片 → 直接中止。
    - INPUT_ROOT 為空：改用手動 CLASS_VIDEO_DIRS（key 必須剛好是 CLASS_DIR_NAMES）。
    """
    root = (INPUT_ROOT or "").strip()
    if root:
        root_p = Path(root).expanduser()
        if not root_p.is_dir():
            raise SystemExit(f"❌ INPUT_ROOT 不是有效資料夾：{root_p}")
        missing = [d for d in CLASS_DIR_NAMES if not (root_p / d).is_dir()]
        if missing:
            raise SystemExit(
                f"❌ 防呆檢查未通過：{root_p}\n"
                f"   底下必須要有 {' 和 '.join(CLASS_DIR_NAMES)} 這兩個子資料夾，目前缺少：{'、'.join(missing)}"
            )
        class_sources = {d: [str(root_p / d)] for d in CLASS_DIR_NAMES}
    else:
        if set(CLASS_VIDEO_DIRS) != set(CLASS_DIR_NAMES):
            raise SystemExit(
                f"❌ CLASS_VIDEO_DIRS 的 key 必須剛好是 {CLASS_DIR_NAMES}，目前是 {tuple(CLASS_VIDEO_DIRS)}"
            )
        class_sources = {d: CLASS_VIDEO_DIRS[d] for d in CLASS_DIR_NAMES}

    resolved = {c: resolve_video_paths(s) for c, s in class_sources.items()}
    empty = [c for c, v in resolved.items() if not v]
    if empty:
        details = "；".join(f"{c} ← {class_sources[c]}" for c in empty)
        raise SystemExit(f"❌ 這些類別找不到任何影片：{details}")
    return resolved


def load_model():
    model = YOLO(YOLO_MODEL_PATH)
    try:
        model.to(INFERENCE_DEVICE)
    except Exception as e:
        print(f"⚠ 無法使用 {INFERENCE_DEVICE}，改用 CPU（{e}）")
    return model


def detect_boxes(model, frame):
    """回傳這一幀所有偵測到的貓：[(bbox(x1,y1,x2,y2) float ndarray, conf float), ...]"""
    results = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    out = []
    if results.boxes is None or len(results.boxes) == 0:
        return out
    for i in range(len(results.boxes)):
        bbox = results.boxes.xyxy[i].cpu().numpy()
        conf = float(results.boxes.conf[i].cpu().numpy())
        out.append((bbox, conf))
    return out


def compute_sharpness(frame, bbox):
    """裁切區域灰階 Laplacian 變異數（越高越清晰），跟 enroll 用的同一套。"""
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, bbox[0])), int(max(0, bbox[1]))
    x2, y2 = int(min(w, bbox[2])), int(min(h, bbox[3]))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_bbox(frame, bbox, padding_ratio):
    """依 bbox 裁切，往外擴 padding_ratio 後夾在畫面範圍內；太小回傳 None。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None
    px, py = bw * padding_ratio, bh * padding_ratio
    cx1 = int(np.clip(x1 - px, 0, w - 1))
    cy1 = int(np.clip(y1 - py, 0, h - 1))
    cx2 = int(np.clip(x2 + px, 0, w))
    cy2 = int(np.clip(y2 + py, 0, h))
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return None
    return frame[cy1:cy2, cx1:cx2]


def _stem_already_done(crop_dir, frame_dir, video_stem):
    """這支影片是否已處理過（依啟用的輸出類型檢查其中一個資料夾有無檔案）。"""
    check_dir = crop_dir if SAVE_CROP else frame_dir
    return any(check_dir.glob(f"{video_stem}__f*.jpg"))


def process_class(model, class_name, video_paths, manifest_writer):
    crop_dir = OUTPUT_ROOT / "crops" / class_name
    frame_dir = OUTPUT_ROOT / "frames" / class_name
    if SAVE_CROP:
        crop_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_FULL_FRAME:
        frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\n類別「{class_name}」：{len(video_paths)} 支影片"
          f"（crop={SAVE_CROP} frame={SAVE_FULL_FRAME}）\n{'=' * 60}")
    if not video_paths:
        print("❌ 找不到任何影片，略過這一類")
        return {"class": class_name, "videos": 0, "crops_saved": 0, "frames_read": 0}

    saved = 0
    frames_read_total = 0
    skip_counts = {"no_cat": 0, "multi_cat": 0, "low_conf": 0, "too_small": 0, "blurry": 0, "crop_fail": 0}

    for vp in video_paths:
        if saved >= MAX_CROPS_PER_CLASS:
            print(f"已達每類上限 {MAX_CROPS_PER_CLASS} 張，剩餘影片不再處理")
            break

        if RESUME_SKIP_DONE_VIDEOS and _stem_already_done(crop_dir, frame_dir, vp.stem):
            print(f"↷ 已處理過，跳過: {vp.name}")
            continue

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"⚠ 無法開啟: {vp.name}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        frame_idx = 0
        valid_seen = 0
        saved_this_video = 0
        frames_saved_this_video = 0
        frame_h = None
        pbar = tqdm(total=total, desc=f"  {vp.name[:40]}", unit="f", leave=False)
        while True:
            if saved >= MAX_CROPS_PER_CLASS:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames_read_total += 1
            if hasattr(pbar, "update"):
                pbar.update(1)
            if frame_h is None:
                frame_h = frame.shape[0]

            dets = detect_boxes(model, frame)

            # ① 剛好一隻貓
            if len(dets) == 0:
                skip_counts["no_cat"] += 1
                continue
            if REQUIRE_SINGLE_CAT and len(dets) > 1:
                skip_counts["multi_cat"] += 1
                continue

            base = f"{vp.stem}__f{frame_idx:06d}"
            frame_written_path = None   # 這一幀的整張原圖只寫一次，多個 crop 共用同一個 frame_file

            for inst_idx, (bbox, conf) in enumerate(dets):
                # ② 信心
                if conf < MIN_BBOX_CONF:
                    skip_counts["low_conf"] += 1
                    continue
                # ③ 大小
                if (bbox[3] - bbox[1]) < frame_h * MIN_BBOX_HEIGHT_RATIO:
                    skip_counts["too_small"] += 1
                    continue
                # ④ 清晰度
                sharp = compute_sharpness(frame, bbox)
                if sharp < MIN_SHARPNESS:
                    skip_counts["blurry"] += 1
                    continue

                valid_seen += 1
                if valid_seen % FRAME_STRIDE != 0:
                    continue
                if saved >= MAX_CROPS_PER_CLASS:
                    break

                crop_rel = ""
                if SAVE_CROP:
                    crop = crop_bbox(frame, bbox, CROP_PADDING_RATIO)
                    if crop is None:
                        skip_counts["crop_fail"] += 1
                        continue
                    crop_name = base + (f"_i{inst_idx}" if not REQUIRE_SINGLE_CAT else "") + ".jpg"
                    imwrite_unicode(crop_dir / crop_name, crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    crop_rel = f"crops/{class_name}/{crop_name}"

                frame_rel = ""
                if SAVE_FULL_FRAME:
                    if frame_written_path is None:
                        frame_name = base + ".jpg"
                        imwrite_unicode(frame_dir / frame_name, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        frame_written_path = f"frames/{class_name}/{frame_name}"
                        frames_saved_this_video += 1
                    frame_rel = frame_written_path

                manifest_writer.writerow({
                    "class": class_name,
                    "crop_file": crop_rel,
                    "frame_file": frame_rel,
                    "source_video": str(vp),
                    "frame_idx": frame_idx,
                    "inst_idx": inst_idx,
                    "bbox": ",".join(f"{v:.0f}" for v in bbox),
                    "bbox_conf": f"{conf:.3f}",
                    "sharpness": f"{sharp:.1f}",
                })
                saved += 1
                saved_this_video += 1

        cap.release()
        if hasattr(pbar, "close"):
            pbar.close()
        print(f"  {vp.name}: 讀 {frame_idx} 幀 -> 取樣 {saved_this_video} 次"
              f"（crop 累計 {saved}，整張原圖 {frames_saved_this_video}）")

    print(f"\n類別「{class_name}」完成：取樣 {saved} 次")
    print(f"  跳過統計：沒偵測到貓 {skip_counts['no_cat']}　多貓同框 {skip_counts['multi_cat']}　"
          f"信心不足 {skip_counts['low_conf']}　太小/太遠 {skip_counts['too_small']}　"
          f"模糊 {skip_counts['blurry']}　裁切失敗 {skip_counts['crop_fail']}")
    return {
        "class": class_name,
        "videos": len(video_paths),
        "crops_saved": saved,
        "frames_read": frames_read_total,
        "skip_counts": skip_counts,
    }


def main():
    if not SAVE_CROP and not SAVE_FULL_FRAME:
        print("❌ SAVE_CROP 與 SAVE_FULL_FRAME 都是 False，沒有東西可輸出")
        return
    class_videos = resolve_class_sources()   # 防呆：資料夾/影片缺任一 → 這裡就 SystemExit
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "_manifest.csv"
    # append 模式：分批續跑時把新的裁切圖接在既有 manifest 後面
    manifest_exists = manifest_path.exists()
    model = load_model()

    fields = ["class", "crop_file", "frame_file", "source_video", "frame_idx", "inst_idx",
              "bbox", "bbox_conf", "sharpness"]
    with manifest_path.open("a", newline="", encoding="utf-8-sig") as mf:
        writer = csv.DictWriter(mf, fieldnames=fields)
        if not manifest_exists:
            writer.writeheader()

        stats = []
        for class_name, video_paths in class_videos.items():
            stats.append(process_class(model, class_name, video_paths, writer))

    run_meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model": YOLO_MODEL_PATH,
        "input_root": (INPUT_ROOT or "").strip() or None,
        "params": {
            "YOLO_IMGSZ": YOLO_IMGSZ,
            "YOLO_CONF_THRESHOLD": YOLO_CONF_THRESHOLD,
            "SAVE_CROP": SAVE_CROP,
            "SAVE_FULL_FRAME": SAVE_FULL_FRAME,
            "FRAME_STRIDE": FRAME_STRIDE,
            "MAX_CROPS_PER_CLASS": MAX_CROPS_PER_CLASS,
            "CROP_PADDING_RATIO": CROP_PADDING_RATIO,
            "REQUIRE_SINGLE_CAT": REQUIRE_SINGLE_CAT,
            "MIN_BBOX_CONF": MIN_BBOX_CONF,
            "MIN_BBOX_HEIGHT_RATIO": MIN_BBOX_HEIGHT_RATIO,
            "MIN_SHARPNESS": MIN_SHARPNESS,
        },
        "classes": stats,
    }
    (OUTPUT_ROOT / "_run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}\n全部完成（SAVE_CROP={SAVE_CROP} SAVE_FULL_FRAME={SAVE_FULL_FRAME}）")
    for s in stats:
        print(f"  {s['class']}: 取樣 {s['crops_saved']} 次（來自 {s['videos']} 支影片）")
    if SAVE_CROP:
        print(f"  裁切圖: {OUTPUT_ROOT / 'crops'}")
    if SAVE_FULL_FRAME:
        print(f"  整張原圖: {OUTPUT_ROOT / 'frames'}")
    print(f"  manifest: {manifest_path}")
    print(f"  參數/統計: {OUTPUT_ROOT / '_run_meta.json'}")


if __name__ == "__main__":
    main()
