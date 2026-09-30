"""用訓練好的四類姿勢模型（lick／scratch／walk／other）分類沒有標註的新圖片：先用 YOLO-Pose 偵測 17 個關鍵點，再分類。

流程：
  1. 偵測（要 ultralytics＋OpenCV，用 yolo_new 環境）：主系統同一個 YOLO 模型與參數（ModelPaths.YOLO_MODEL、
     YOLOConfig.IMAGE_SIZE／CONFIDENCE_THRESHOLD、CUDA 上 FP16）。取法跟產生骨架訓練資料的
     train_data/0_dataset_collect.py 一樣：取第一個（信心最高的）偵測、keypoints.xy 全部 17 點都用（含低信心點）。
  2. 分類：載入 MODEL_PATH（1_train_pose_classifier_from_tags.py 存的 model.joblib），原圖與鏡像機率取平均。
     model.joblib 是 base 環境的 sklearn 1.7 存的，yolo_new 的 sklearn 1.9 讀不了 → 目前環境載入失敗時，
     自動用 CLASSIFIER_PYTHON 重新執行本腳本的分類步驟（不用手動切環境）。

輸出（原圖不動）：
  <輸入資料夾>_pose4/<lick|scratch|walk|other>/ 與 _沒偵測到貓/：圖片複製過去（子資料夾的圖檔名前面加上相對路徑）
  paper/reports/eval/pose_classifier_apply/NNN_<時間>/predictions.csv：每張圖的類別、四類機率、最高機率、偵測到幾隻貓
    （多隻貓只分類信心最高的那隻；訓練資料只有單貓圖，多貓的結果較不可靠）、summary.txt、keypoints.npz

⚠ 模型限制（2026-09-29 報告 018）：人工考卷 macro-F1 0.872；使用者 90 張截圖（獨立測試集）accuracy 0.852，
  walk 55/55 全對，other 32 張對 19 張——正面坐著仍常被判成 walk、躺著有時判成 lick。訓練用的是人工標註的關鍵點＋YOLO 骨架幀，YOLO 偵測錯的圖分類也會錯。

用法：
  C:\\Users\\homec\\anaconda3\\envs\\yolo_new\\python.exe 1_classify_new_images_by_pose.py <圖片資料夾>
  （也可在設定視窗「🎬 影片路徑」填圖片資料夾；子資料夾的圖也會處理）
  --model <model.joblib> 換模型；--no-copy 不複製圖片；--truth-dir <人工分好類的資料夾> 順便算準確度
  不帶參數直接執行＝用頂部寫死的 INPUT_DIR、TRUTH_DIR、COPY_IMAGES、MODEL_PATH（命令列有給的優先）
  獨立測試集（使用者 90 張截圖，2026-09-29 人工分好）：
    python 1_classify_new_images_by_pose.py <cat_test_image> --no-copy --truth-dir <cat_test_image_pose4> --model <新模型>
"""
import argparse
import csv
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # config.py 在 paper/ 根目錄

from _report_paths import report_dir  # noqa: E402

# ==================== 使用者設定區 ====================
# ---- 輸入路徑（寫死；命令列有給就用命令列的） ----
# 要分類的圖片資料夾（子資料夾也會處理）；設成 "" ＝用設定視窗「🎬 影片路徑」
INPUT_DIR = r"C:\Users\homec\OneDrive\圖片\Screenshots\cat_test_image"
# 人工分好類的答案資料夾（<類別>/<同檔名>）：有設就順便算準確度；"" ＝不算
TRUTH_DIR = r""
COPY_IMAGES = True  # True＝依類別複製到 <輸入資料夾>_pose4/；False＝只輸出 predictions.csv
# 018：補了坐／趴／躺影片（EXTRA_SKELETON_VIDEOS）；獨立測試集 other recall 0.375→0.594（016→018）
MODEL_PATH = Path(r"C:\ai_project\paper\reports\eval\pose_classifier_from_tags\018_20260929_161832\model.joblib")
CLASSIFIER_PYTHON = Path(r"C:\Users\homec\anaconda3\python.exe")  # 能載入 model.joblib 的環境（sklearn 1.7）
# ---- 其他設定 ----
OUTPUT_SUFFIX = "_pose4"
NO_CAT_DIR = "_沒偵測到貓"
# ======================================================

_env_path = os.getenv("TEST_VIDEO_PATH", "").strip().strip('"')
if not INPUT_DIR and _env_path and os.path.isdir(_env_path):
    INPUT_DIR = _env_path

_spec = importlib.util.spec_from_file_location(
    "pose_tags", Path(__file__).with_name("1_train_pose_classifier_from_tags.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)  # 特徵、鏡像推論、長路徑處理都沿用訓練腳本（同一套才對得上模型）


def list_images(root):
    out = []
    for dirpath, _, files in os.walk(pt._lp(root)):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in pt.IMG_EXTS:
                full = Path(dirpath[len(pt._LONG):] if dirpath.startswith(pt._LONG) else dirpath) / f
                out.append(full)
    return sorted(out)


def detect(images, run_dir):
    """YOLO-Pose 偵測 → keypoints.npz（P[N,17,2]、B[N,4]，沒偵測到為 NaN）。"""
    import cv2
    from config import ModelPaths, YOLOConfig
    from ultralytics import YOLO
    import torch

    cuda = torch.cuda.is_available()
    model = YOLO(ModelPaths.YOLO_MODEL)
    model.to("cuda" if cuda else "cpu")
    print(f"  YOLO：{Path(ModelPaths.YOLO_MODEL).name}（imgsz {YOLOConfig.IMAGE_SIZE}、conf {YOLOConfig.CONFIDENCE_THRESHOLD}、"
          f"{'CUDA FP16' if cuda else 'CPU'}）")
    N = len(images)
    P = np.full((N, pt.N_KP, 2), np.nan)
    B = np.full((N, 4), np.nan)
    det_conf, n_cats = np.full(N, np.nan), np.zeros(N, int)
    unreadable = 0
    step = max(1, N // 10)
    for i, p in enumerate(images):
        img = cv2.imdecode(np.fromfile(pt._lp(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ❌ 讀不到圖片：{p}")
            unreadable += 1
            if unreadable == N:
                raise SystemExit("❌ 全部圖片都讀不到，結束")
            continue
        r = model.predict(img, imgsz=YOLOConfig.IMAGE_SIZE, conf=YOLOConfig.CONFIDENCE_THRESHOLD,
                          quantize=16 if cuda else None, verbose=False)[0]
        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.boxes is not None and len(r.boxes) > 0:
            n_cats[i] = len(r.keypoints.xy)
            P[i] = r.keypoints.xy[0].cpu().numpy()
            B[i] = r.boxes.xyxy[0].cpu().numpy()
            det_conf[i] = float(r.boxes.conf[0])
        if (i + 1) % step == 0 or i + 1 == N:
            print(f"  [{i + 1}/{N}] ▶ 目前圖片：{p.name}")
    np.savez(run_dir / "keypoints.npz", files=np.array([str(p) for p in images]), P=P, B=B,
             det_conf=det_conf, n_cats=n_cats)
    print(f"  偵測到貓 {int((n_cats > 0).sum())} 張、沒偵測到 {int((n_cats == 0).sum())} 張、多隻貓 {int((n_cats > 1).sum())} 張")


def _load_model():
    import warnings

    import joblib

    with warnings.catch_warnings():  # sklearn 版本不同的警告：讀得了就照用、讀不了會換環境，不必在畫面上嚇人
        warnings.simplefilter("ignore")
        return joblib.load(MODEL_PATH)


def score(files, pred, classes, truth_dir, lines):
    """跟人工分好的資料夾（<truth_dir>/<類別>/<同檔名>）比對，算各類 precision／recall。"""
    from sklearn.metrics import classification_report, confusion_matrix

    truth = {}
    for c in os.listdir(pt._lp(truth_dir)):
        if os.path.isdir(pt._lp(truth_dir / c)):
            for f in os.listdir(pt._lp(truth_dir / c)):
                truth[f] = c
    pairs = [(truth[f.name], p) for f, p in zip(files, pred) if truth.get(f.name) in classes]
    skipped = len(files) - len(pairs)
    y, q = zip(*pairs)
    lines += ["", f"【對人工答案：{truth_dir}】計分 {len(pairs)} 張（沒有答案或答案不是四類的 {skipped} 張不算）",
              classification_report(y, q, labels=classes, digits=3, zero_division=0),
              "混淆矩陣（列＝人工、欄＝模型，最後一欄＝沒偵測到貓）"]
    for c, row in zip(classes, confusion_matrix(y, q, labels=classes + [NO_CAT_DIR])[:len(classes)]):
        lines.append(f"  {c:<8}{row}")


def classify(run_dir, input_dir, copy=True, truth_dir=None):
    """keypoints.npz → predictions.csv＋依類別複製圖片（copy=False 只輸出 CSV）。"""
    d = np.load(run_dir / "keypoints.npz")
    files, P, B, det_conf, n_cats = [Path(f) for f in d["files"]], d["P"], d["B"], d["det_conf"], d["n_cats"]
    m = _load_model()
    classes = list(m["classes"])
    ok = n_cats > 0
    proba = np.full((len(files), len(classes)), np.nan)
    if ok.any():
        proba[ok] = pt.predict_proba_tta(m["model"], P[ok], B[ok], m["ratios"], np.array(m["flip_idx"]))
    pred = [classes[int(np.argmax(p))] if k else NO_CAT_DIR for p, k in zip(proba, ok)]

    out_root = pt._free_dir(input_dir.with_name(input_dir.name + OUTPUT_SUFFIX)) if copy else None
    if copy:
        for c in classes + [NO_CAT_DIR]:
            os.makedirs(pt._lp(out_root / c), exist_ok=True)
    rows = []
    for f, c, p, k, n, dc in zip(files, pred, proba, ok, n_cats, det_conf):
        rel = f.relative_to(input_dir)
        if copy:
            name = "__".join(rel.parts)  # 子資料夾的圖：檔名前面加相對路徑，避免同名互蓋
            shutil.copy2(pt._lp(f), pt._lp(out_root / c / name))
        rows.append([str(rel), c, *(f"{v:.4f}" if k else "" for v in p), f"{np.nanmax(p):.4f}" if k else "",
                     int(n), f"{dc:.3f}" if k else ""])
    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "類別", *(f"p_{c}" for c in classes), "最高機率", "偵測到幾隻貓", "偵測信心"])
        w.writerows(sorted(rows, key=lambda r: (r[1], r[-3] or "9")))

    cnt = Counter(pred)
    lines = [f"四類姿勢分類（新圖片）  {datetime.now():%Y-%m-%d %H:%M}",
             f"輸入：{input_dir}（{len(files)} 張）", f"模型：{MODEL_PATH}",
             f"複製到：{out_root}" if copy else "沒有複製圖片（--no-copy）", ""]
    for c in classes + [NO_CAT_DIR]:
        lines.append(f"  {c:<12}{cnt.get(c, 0):>6} 張")
    low = int(sum(1 for p, k in zip(proba, ok) if k and np.nanmax(p) < 0.6))
    lines += ["", f"最高機率 < 0.6（模型沒把握，建議人工看）：{low} 張；多隻貓：{int((n_cats > 1).sum())} 張",
              "⚠ other（躺／坐）是最弱的類別：正面坐著常被判成 walk、躺著有時判成 lick"]
    if truth_dir:
        score(files, pred, classes, Path(truth_dir), lines)
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"✓ 報告：{run_dir}")


def main():
    global MODEL_PATH
    ap = argparse.ArgumentParser(description="用四類姿勢模型分類沒有標註的新圖片")
    ap.add_argument("input_dir", nargs="?", default=INPUT_DIR, help="圖片資料夾")
    ap.add_argument("--model", help=f"改用別的 model.joblib（預設 {MODEL_PATH}）")
    ap.add_argument("--no-copy", action="store_true", help="不複製圖片，只輸出 predictions.csv")
    ap.add_argument("--truth-dir", default=TRUTH_DIR or None,
                    help="人工分好類的資料夾（<類別>/<同檔名>），有給就算準確度（預設 TRUTH_DIR）")
    ap.add_argument("--classify-only", metavar="RUN_DIR", help=argparse.SUPPRESS)  # 內部用：換環境跑分類步驟
    args = ap.parse_args()
    args.no_copy = args.no_copy or not COPY_IMAGES
    if args.truth_dir and not os.path.isdir(args.truth_dir):
        raise SystemExit(f"❌ 找不到答案資料夾：{args.truth_dir!r}")
    if args.model:
        MODEL_PATH = Path(args.model)

    if args.classify_only:
        run_dir = Path(args.classify_only)
        classify(run_dir, Path(args.input_dir), copy=not args.no_copy, truth_dir=args.truth_dir)
        return

    if not args.input_dir or not os.path.isdir(pt._lp(args.input_dir)):
        raise SystemExit(f"❌ 找不到圖片資料夾：{args.input_dir!r}\n   用法：python {Path(__file__).name} <圖片資料夾>")
    if not MODEL_PATH.exists():
        raise SystemExit(f"❌ 找不到模型：{MODEL_PATH}")
    input_dir = Path(os.path.abspath(args.input_dir))
    t0 = time.time()
    images = list_images(input_dir)
    if not images:
        raise SystemExit(f"❌ {input_dir} 底下沒有圖片")
    run_dir = pt._next_run_dir(report_dir("eval", "pose_classifier_apply"))

    print(f"▶ [1/2] 偵測關鍵點：{len(images)} 張（{input_dir}）")
    try:
        detect(images, run_dir)
    except ImportError as e:
        raise SystemExit(f"❌ 這個環境沒有 {e.name}，偵測步驟要用 yolo_new 環境執行")

    print("▶ [2/2] 分類、複製圖片")
    try:
        _load_model()
    except Exception as e:  # sklearn 版本不同時讀不了 → 換環境
        print(f"  目前環境載入不了模型（{type(e).__name__}: {e}），改用 {CLASSIFIER_PYTHON} 分類")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        r = subprocess.run([str(CLASSIFIER_PYTHON), __file__, str(input_dir), "--classify-only", str(run_dir),
                            "--model", str(MODEL_PATH)] + (["--no-copy"] if args.no_copy else [])
                           + (["--truth-dir", args.truth_dir] if args.truth_dir else []), env=env)
        if r.returncode:
            raise SystemExit(f"❌ 分類步驟失敗（exit {r.returncode}）")
    else:
        classify(run_dir, input_dir, copy=not args.no_copy, truth_dir=args.truth_dir)
    print(f"（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
