"""
單幀姿態分類器（四類，依姿勢）的訓練腳本：tag 圖片＋骨架抽幀訓練，用使用者逐張人工標記的 299 張評估。
分類新圖片用另一支 1_classify_new_images_by_pose.py（它從這支 import 特徵與模型函式）。

類別（2026-09-29 使用者定案，依「姿勢」而不是影片行為）：
  lick     舔毛
  scratch  抓癢
  walk     走路、站著、甩頭
  other    側躺、坐下等休息姿勢

訓練資料：
  1. Roboflow tag 圖片（照 tag 原樣）：Downloads\\11.v282i.yolov8_all_walk → walk（使用者確認只有站／走）、
     _train_lick → lick、_train_scratch → scratch、_stop → other。
     _stop 是 2026-09-29 從 lick／scratch tag 移出的圖：模型判成 other、使用者看圖確認都是躺／坐
     （移動紀錄在 _stop/moved_from_log.csv）。
  2. 骨架抽幀（skeletons/train/，使用者手動標記的影片）：每類抽一樣多（＝容量最小的類別），
     每支影片最多 MAX_FRAMES_PER_VIDEO 幀、在標記幀中等距取；只用 label＝資料夾名、有偵測到的幀。
     stop 影片依使用者肉眼檢視結果（STOP_REVIEW_CSV，2026-09-29 逐支看過）分兩邊：
     other＝躺／坐的 92 支、not_other＝站著的 18 支歸 walk。
  3. 額外骨架影片（EXTRA_SKELETON_VIDEOS）：使用者另外挑的影片（2026-09-29：14 支坐／趴／躺的 stop 影片，
     補 other 缺的坐姿），在 skeletons/ 的 train／val／test 找同名 JSON，每支最多 EXTRA_MAX_PER_VIDEO 幀、
     整支影片都算該類別（不看幀的 label）；不跟上面的「各類等量」搶名額，另外加。
  tag 可能是整支影片一起貼的（舔毛影片裡躺著休息的幀也叫 lick），所以不拿 tag 當評估標準。

評估資料：Downloads\\11.v282i.yolov8_labeling\\image_sort\\<walk|stop|shake|lick|scratch>\\
  使用者逐張看圖標記（stop 資料夾＝躺／坐 → other；walk、shake → walk），圖片旁邊同檔名 .txt 是 YOLO-pose 標註。

特徵與模型：17 點 → 跟位置／大小／朝向無關的幾何特徵（點間距離除以脊椎弧長、左右對稱彙整、關節夾角、相對高度）、
  HistGradientBoosting（class_weight=balanced）、訓練加左右鏡像（骨架幀另加遮點）、推論原圖與鏡像取平均。
  只讀標註好的關鍵點，不跑 YOLO；v=0（沒標）的點當缺失。只收單貓的圖。圖片檔名不參與判斷。

評估切法：tag 圖與評估圖合在一起，依「來源」（同一支影片抽出的幀、同一系列截圖）分組做 5 折；
  評估圖只在它那一折被考，同來源的 tag 圖不會出現在該折的訓練集。
  同一張圖同時在 tag 資料夾與評估集（圖片內容 MD5）時，訓練用 tag 標記、考試用人工標記。
  防外洩：很多 tag／考卷圖跟骨架影片是同一支影片（骨架影片改過檔名，看檔名對不出來）。訓練前先比對畫面
  產生同源清單 OVERLAP_CSV（資料有新增就自動重跑），第 k 折訓練時排除「跟第 k 折要考的圖同源」的骨架影片。
  超參數用 tag 資料本身的交叉驗證挑，不看評估集。

同源比對方法：圖片裁掉 letterbox 黑邊、影片每 FRAME_STEP 幀取 1 幀，都縮成 32×32 灰階、減平均後算相關係數，
  每張圖對每支影片取最像的那一幀；相似度 ≥ SAME_VIDEO_THRESHOLD 就判定同源（一張圖可以對到多支影片）。
  門檻刻意放低、寧可多判：多判只是某一折少用幾支骨架影片，漏判才會外洩。
  校正（2026-09-29 人工看並排圖）：考卷 ≥0.60 的 134 對全部看過，同一支影片 77 對（0.70 以下的 2 對寫進 MANUAL_LINKS）、
  不同影片最高 0.851；tag 只抽查（≥0.855 12/12、0.80～0.855 7/12、0.70～0.75 5/16 同一支 → tag 門檻以下還會漏一些，
  只影響選超參數與「對 tag 吻合度」，不影響考卷分數）。試過遮貓只比背景：純白背景會爆掉、分數重疊更多，沒採用。
  需要 OpenCV：目前環境沒有就自動用 OPENCV_PYTHON（yolo_new）跑這一步。

輸出：paper/reports/eval/pose_classifier_from_tags/NNN_<時間>/
  summary.txt、metrics.csv、eval_predictions.csv（考卷每張的判斷）、tag_predictions.csv（tag 每張，找 tag 錯誤用）、
  skeleton_frames.csv（抽到哪些幀）、confusion_matrices.png、model.joblib（用全部訓練資料訓練的最終模型）
  同源清單：paper/reports/review/image_video_overlap/overlap.csv、compared_videos.txt

用法（base 環境；model.joblib 要給 1_classify_new_images_by_pose.py 用，兩邊 sklearn 版本要一致）：
  python 1_train_pose_classifier_from_tags.py                          # 訓練＋評估（同源清單過期會自動重跑）
  python 1_train_pose_classifier_from_tags.py --max-per-video 20       # 每支影片多抽一點
  python 1_train_pose_classifier_from_tags.py --no-skeleton-sample     # 只用 tag 圖片
  python 1_train_pose_classifier_from_tags.py --update-overlap         # 只重跑同源比對
  python 1_train_pose_classifier_from_tags.py --extract-frames [影片資料夾]
      # 補資料用：依骨架 JSON 把影片抽成圖片＋YOLO-pose 標註（<影片資料夾>_frames/），附拼貼圖給人工確認姿勢；
      # 沒給資料夾就用 EXTRACT_FRAMES_DIR
  輸入路徑都寫死在「使用者設定區」開頭（TRAIN_DIRS、EVAL_DIR、FLIP_YAML、SKELETON_ROOT、STOP_REVIEW_CSV、
  EXTRA_SKELETON_VIDEOS、EXTRACT_FRAMES_DIR、OPENCV_PYTHON），換電腦或搬資料夾時改那裡。
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from _report_paths import report_dir  # 報告輸出位置統一定義在 tools/_report_paths.py

# ==================== 使用者設定區 ====================
# ---- 輸入路徑（寫死；換電腦或搬資料夾時改這裡） ----
TRAIN_DIRS = {  # 輸出類別 → tag 資料夾（底下 <split>/{images,labels}/）
    "walk": Path(r"C:\Users\homec\Downloads\11.v282i.yolov8_all_walk"),
    "lick": Path(r"C:\Users\homec\Downloads\11.v282i.yolov8_train_lick"),
    "scratch": Path(r"C:\Users\homec\Downloads\11.v282i.yolov8_train_scratch"),
    "other": Path(r"C:\Users\homec\Downloads\11.v282i.yolov8_stop"),
}
# 人工考卷：底下 <人工類別>/ 圖片＋同檔名 .txt
EVAL_DIR = Path(r"C:\Users\homec\Downloads\11.v282i.yolov8_labeling\image_sort")
# 左右成對點的定義（flip_idx）。Roboflow 新匯出的 data.yaml 是 [0..16] 不交換左右，不能用
FLIP_YAML = Path(r"C:\Users\homec\Downloads\data.yaml")
SKELETON_ROOT = Path(r"C:\ai_project\paper\skeletons")  # 底下 <train|val|test>/<類別>/<影片>.json
# stop 影片人工檢視結果（躺坐 other／站著 not_other）
STOP_REVIEW_CSV = Path(r"C:\ai_project\paper\reports\review\stop_video_review\stop_review.csv")
# 額外骨架影片：類別 → 影片資料夾（檔名對到 skeletons/<split>/<資料夾>/<同名>.json）；整支影片都算該類別
EXTRA_SKELETON_VIDEOS = {
    "other": [Path(r"C:\Users\homec\Downloads\CLASS_OTHER\OTHER")],  # 2026-09-29 使用者整理的坐／趴／躺影片（補坐姿）
}
# --extract-frames 沒給資料夾時用這個
EXTRACT_FRAMES_DIR = Path(r"C:\Users\homec\Downloads\CLASS_OTHER\OTHER")
OPENCV_PYTHON = Path(r"C:\Users\homec\anaconda3\envs\yolo_new\python.exe")  # 有 OpenCV 的環境
# ---- 其他設定 ----
EVAL_CLASS_MAP = {"walk": "walk", "shake": "walk", "stop": "other", "lick": "lick", "scratch": "scratch"}
# 骨架抽幀來源：輸出類別 → [(skeletons/train/ 底下的資料夾, 影片篩選)]；篩選 None＝全部影片，
# 其他值＝stop_review.csv 裡 choice 等於該值的影片
SKELETON_SOURCES = {
    "lick": [("lick", None)],
    "scratch": [("scratch", None)],
    "walk": [("walk", None), ("shake", None), ("stop", "not_other")],
    "other": [("stop", "other")],
}
MAX_FRAMES_PER_VIDEO = 10  # 同一支影片最多抽幾幀（在標記幀中等距取，避開幾乎一樣的相鄰幀）
EXTRA_MAX_PER_VIDEO = 10
N_FOLDS = 5
PARAM_GRID = [dict(learning_rate=0.05, max_iter=it, max_leaf_nodes=31, min_samples_leaf=msl, l2_regularization=1.0)
              for msl in (10, 30, 100) for it in (200, 500)]
MASK_SCALE = 1.5  # 骨架幀隨機遮點的機率 = tag 圖各點缺失比例 × 此倍數
RANDOM_SEED = 0

# 同源比對（防外洩）
OVERLAP_DIR = report_dir("review", "image_video_overlap")
OVERLAP_CSV = OVERLAP_DIR / "overlap.csv"
SAME_VIDEO_THRESHOLD = 0.70
FRAME_STEP = 3  # 影片每幾幀取 1 幀比對
SIG_SIZE = 32
# 門檻以下、人工看並排圖確認是同一支影片的考卷圖：考卷資料夾/檔名 → 骨架影片（<資料夾>/<json 檔名>）
MANUAL_LINKS = {
    "lick/The_video_duration_must_be_exa-12-_mp4-0007_jpg.rf.e8097da0f53e948028ac8b0b562a191e.jpg": "stop/stop_150",
    "walk/black-dog-road-1-_jpg.rf.83a4a83c70a534e468c96b3f83108c51.jpg": "shake/shake_75",  # 考卷圖是直式裁切
}

# --extract-frames
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
JPEG_QUALITY = 95
REVIEW_FRAMES = 12  # 每支影片拼貼幾幀
# ======================================================

# 17 點 YOLO-Pose 排列（同 plugins/lick_stage/ext_body_zones/config.py）
NOSE, L_EAR, R_EAR, CHEST, MID_BACK, HIP = 0, 1, 2, 3, 4, 5
FL_KNEE, FL_PAW, FR_KNEE, FR_PAW = 6, 7, 8, 9
HL_KNEE, HL_PAW, HR_KNEE, HR_PAW = 10, 11, 12, 13
TAIL_ROOT, TAIL_MID, TAIL_TIP = 14, 15, 16
N_KP = 17
KP_NAMES = ["nose", "l_ear", "r_ear", "chest", "mid_back", "hip", "fl_knee", "fl_paw", "fr_knee", "fr_paw",
            "hl_knee", "hl_paw", "hr_knee", "hr_paw", "tail_root", "tail_mid", "tail_tip"]
PAIRS = [(i, j) for i in range(N_KP) for j in range(i + 1, N_KP)]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASSES = sorted(SKELETON_SOURCES)  # sklearn classes_ 是字母序
_RF_SUFFIX_RE = re.compile(r"_(?:png|jpe?g|bmp|webp)\.rf\.[0-9a-f]+$", re.I)
_LONG = "\\\\?\\"  # 這台 LongPathsEnabled=0，部分 Roboflow 檔名放進資料夾後超過 260 字元


# ── 共用小工具 ─────────────────────────────────────────────────────────

def _lp(p):
    return _LONG + os.path.abspath(p)


def _md5(p):
    with open(_lp(p), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _next_run_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[:3]) for p in root.iterdir() if p.is_dir() and p.name[:3].isdigit()]
    d = root / f"{max(nums, default=0) + 1:03d}_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir()
    return d


def _free_dir(base):
    d, n = base, 2
    while d.exists():
        d = base.with_name(f"{base.name}_{n}")
        n += 1
    return d


def _source_group(stem):
    """交叉驗證分組用：同一支影片抽出的幀（…_mp4-0007）、同一系列截圖（lick7-3）歸同一組，避免相鄰幀
    同時落在訓練與驗證而灌水。只拿檔名判斷「來源是否相同」，不拿來判斷行為。"""
    base = _RF_SUFFIX_RE.sub("", stem)
    m = re.match(r"^(.*)_(?:mp4|mov|avi)-\d+$", base, re.I) or re.match(r"^(.*?)-\d+$", base)
    return m.group(1) if m else base


def _has_opencv():
    return importlib.util.find_spec("cv2") is not None


def _rerun_with_opencv(extra_args):
    """目前環境沒有 OpenCV 時，用 OPENCV_PYTHON 重新執行本腳本的某個步驟。"""
    if not OPENCV_PYTHON.exists():
        raise SystemExit(f"❌ 這個環境沒有 OpenCV，也找不到 OPENCV_PYTHON：{OPENCV_PYTHON}")
    print(f"  目前環境沒有 OpenCV，改用 {OPENCV_PYTHON}")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(OPENCV_PYTHON), __file__, *extra_args], env=env)
    if r.returncode:
        raise SystemExit(f"❌ OpenCV 步驟失敗（exit {r.returncode}）")


# ── 特徵與模型（1_classify_new_images_by_pose.py 也用這些，改了要重新訓練） ─────────

def load_flip_idx(path):
    """讀 data.yaml 的 flip_idx；不存在、長度不對、或根本沒交換左右（[0..16]）都直接停下，避免鏡像把左右教反。"""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"❌ 找不到 flip 定義檔：{p}")
    idx = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("flip_idx")
    if not idx or len(idx) != N_KP or sorted(idx) != list(range(N_KP)):
        raise SystemExit(f"❌ {p} 的 flip_idx 不是 0～{N_KP - 1} 的排列：{idx}")
    if list(idx) == list(range(N_KP)):
        raise SystemExit(f"❌ {p} 的 flip_idx 沒有交換任何左右點（[0..16]），請改用正確的定義檔")
    return np.array(idx)


def mirror(P, B, flip_idx):
    """水平鏡像：x 取負、左右成對點依 flip_idx 互換（特徵與平移無關，所以不用管畫面寬度）。"""
    Pm = P.copy()
    Pm[:, :, 0] = -Pm[:, :, 0]
    Bm = np.stack([-B[:, 2], B[:, 1], -B[:, 0], B[:, 3]], axis=1)
    return Pm[:, flip_idx], Bm


def _d(P, a, b):
    return np.linalg.norm(P[:, a] - P[:, b], axis=1)


def _angle(P, a, o, b):
    """o 點上 o→a 與 o→b 的夾角（度），任一點缺失為 NaN。"""
    v1, v2 = P[:, a] - P[:, o], P[:, b] - P[:, o]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _nmin(*arrs):
    return np.fmin.reduce(np.stack(arrs), axis=0)  # 忽略 NaN 的逐元素最小值（全缺才是 NaN）


def fit_scale_ratios(P, B):
    """脊椎弧長缺失時的備援尺度換算比例（弧長／弦長、弧長／bbox 對角線），取訓練資料中位數。"""
    arc = _d(P, CHEST, MID_BACK) + _d(P, MID_BACK, HIP)
    chord = _d(P, CHEST, HIP)
    diag = np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(arc_per_chord=float(np.nanmedian(arc / chord)), arc_per_diag=float(np.nanmedian(arc / diag)))


def make_features(P, B, ratios):
    """P[N,17,2]（缺失為 NaN）→ 特徵[N,F]。距離都除以身體尺度（脊椎弧長 胸→背中→髖），
    只用距離、無號夾角、垂直方向的相對高度，所以跟貓在畫面的位置、大小、朝左或朝右都無關。"""
    arc = _d(P, CHEST, MID_BACK) + _d(P, MID_BACK, HIP)
    chord = _d(P, CHEST, HIP)
    diag = np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1])
    S = np.where(np.isfinite(arc), arc, chord * ratios["arc_per_chord"])
    S = np.where(np.isfinite(S) & (S > 0), S, diag * ratios["arc_per_diag"])
    S = np.fmax(S, 0.1 * diag)  # 正對鏡頭時脊椎投影很短，避免距離被放大到失真

    cols, names = [], []

    def add(name, val):
        names.append(name)
        cols.append(val)

    for i, j in PAIRS:
        add(f"d_{KP_NAMES[i]}_{KP_NAMES[j]}", _d(P, i, j) / S)

    head = (NOSE, L_EAR, R_EAR)
    # 左右對稱彙整：取兩側較近者，不受「看得到哪一側」與 YOLO 左右腳對調影響
    add("nose_front_paw", _nmin(_d(P, NOSE, FL_PAW), _d(P, NOSE, FR_PAW)) / S)
    add("nose_hind_paw", _nmin(_d(P, NOSE, HL_PAW), _d(P, NOSE, HR_PAW)) / S)
    add("nose_knee", _nmin(*[_d(P, NOSE, k) for k in (FL_KNEE, FR_KNEE, HL_KNEE, HR_KNEE)]) / S)
    add("nose_rear_body", _nmin(_d(P, NOSE, MID_BACK), _d(P, NOSE, HIP), _d(P, NOSE, TAIL_ROOT)) / S)
    add("head_hind_paw", _nmin(*[_d(P, h, p) for h in head for p in (HL_PAW, HR_PAW)]) / S)
    add("head_front_paw", _nmin(*[_d(P, h, p) for h in head for p in (FL_PAW, FR_PAW)]) / S)
    add("head_hind_knee", _nmin(*[_d(P, h, k) for h in head for k in (HL_KNEE, HR_KNEE)]) / S)

    add("ang_neck", _angle(P, NOSE, CHEST, HIP))            # 180＝頭朝前一直線，越小＝頭越往身體轉
    add("ang_head_back", _angle(P, NOSE, CHEST, MID_BACK))
    add("ang_spine", _angle(P, CHEST, MID_BACK, HIP))       # 越小＝背越弓
    add("ang_front_knee", _nmin(_angle(P, CHEST, FL_KNEE, FL_PAW), _angle(P, CHEST, FR_KNEE, FR_PAW)))
    add("ang_hind_knee", _nmin(_angle(P, HIP, HL_KNEE, HL_PAW), _angle(P, HIP, HR_KNEE, HR_PAW)))
    add("ang_tail", _angle(P, TAIL_ROOT, TAIL_MID, TAIL_TIP))

    y = P[:, :, 1]  # 影像座標 y 向下；以下「above」為正代表在上方
    ground = np.fmax.reduce(y[:, [FL_PAW, FR_PAW, HL_PAW, HR_PAW]].T, axis=0)  # 最低的爪子≈地面
    hind_top = np.fmin.reduce(y[:, [HL_PAW, HR_PAW]].T, axis=0)
    front_top = np.fmin.reduce(y[:, [FL_PAW, FR_PAW]].T, axis=0)
    add("chest_above_hip", (y[:, HIP] - y[:, CHEST]) / S)
    add("nose_above_chest", (y[:, CHEST] - y[:, NOSE]) / S)
    add("nose_above_hip", (y[:, HIP] - y[:, NOSE]) / S)
    add("nose_above_ground", (ground - y[:, NOSE]) / S)
    add("hind_paw_lift", (ground - hind_top) / S)            # 抓癢時後爪離地抬高
    add("front_paw_lift", (ground - front_top) / S)
    add("hind_paw_above_hip", (y[:, HIP] - hind_top) / S)
    add("body_tilt", np.degrees(np.arctan2(np.abs(P[:, CHEST, 1] - P[:, HIP, 1]), np.abs(P[:, CHEST, 0] - P[:, HIP, 0]))))
    with np.errstate(invalid="ignore", divide="ignore"):
        add("bbox_aspect", (B[:, 2] - B[:, 0]) / (B[:, 3] - B[:, 1]))
        add("spine_curl", chord / arc)                       # 弦長／弧長：越小＝身體越捲
        add("body_vs_bbox", S / diag)

    X = np.column_stack(cols).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, names


def mask_keypoints(P, rates, rng):
    Pm = P.copy()
    Pm[rng.random(P.shape[:2]) < rates[None, :]] = np.nan
    return Pm


def fit_model(X, y, params, sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                         random_state=RANDOM_SEED, **params)
    return clf.fit(X, y, sample_weight=sample_weight)


def augmented(P, B, y, rates, ratios, rng, flip_idx):
    """原始幀＋左右鏡像＋一份遮點複本（遮點前隨機一半先鏡像），遮點比例依資料集各點 v=0 比例。"""
    Pm, Bm = mirror(P, B, flip_idx)
    sel = rng.random(len(P)) < 0.5
    Pmix, Bmix = np.where(sel[:, None, None], Pm, P), np.where(sel[:, None], Bm, B)
    X1, names = make_features(P, B, ratios)
    X2, _ = make_features(Pm, Bm, ratios)
    X3, _ = make_features(mask_keypoints(Pmix, rates, rng), Bmix, ratios)
    return np.vstack([X1, X2, X3]), np.concatenate([y, y, y]), names


def human_augmented(P, B, y, ratios, flip_idx):
    """人工標註樣本＋左右鏡像（標註的關鍵點本身就有真實缺點，不另外遮點）。"""
    X1, _ = make_features(P, B, ratios)
    X2, _ = make_features(*mirror(P, B, flip_idx), ratios)
    return np.vstack([X1, X2]), np.concatenate([y, y])


def predict_proba_tta(model, P, B, ratios, flip_idx):
    """原姿勢與左右鏡像各算一次機率再平均，朝左朝右的同一姿勢得到一致的判斷。"""
    X, _ = make_features(P, B, ratios)
    Xm, _ = make_features(*mirror(P, B, flip_idx), ratios)
    return (model.predict_proba(X) + model.predict_proba(Xm)) / 2


def summarize(y_true, y_pred, classes):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return dict(macro_f1=f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                bal_acc=balanced_accuracy_score(y_true, y_pred),
                per_class={c: dict(precision=p[i], recall=r[i], f1=f[i], support=int(s[i])) for i, c in enumerate(classes)},
                cm=confusion_matrix(y_true, y_pred, labels=classes))


def print_summary(title, res, classes, lines):
    lines.append(f"\n{title}\n  macro-F1 {res['macro_f1']:.3f}｜balanced acc {res['bal_acc']:.3f}")
    lines.append(f"  {'類別':<10}{'precision':>10}{'recall':>9}{'F1':>7}{'樣本':>8}")
    for c in classes:
        m = res["per_class"][c]
        lines.append(f"  {c:<10}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>7.3f}{m['support']:>8}")
    w = max(len(c) for c in classes) + 2
    lines.append("  混淆矩陣（列＝真實、欄＝預測）\n  " + " " * w + "".join(f"{c:>9}" for c in classes))
    for i, c in enumerate(classes):
        lines.append(f"  {c:<{w}}" + "".join(f"{n:>9}" for n in res["cm"][i]))
    print("\n".join(lines[-(len(classes) * 2 + 3):]))


def plot_confusions(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.8))
    for ax, (title, cm, labels) in zip(np.atleast_1d(axes), panels):
        rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{rown[i, j] * 100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if rown[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("預測")
        ax.set_ylabel("真實")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ── 讀資料 ─────────────────────────────────────────────────────────────

def _read_pose(img_path, lab_path):
    """讀一張圖的 YOLOv8-pose 標註 → [(P[17,2] 像素座標、v=0 為 NaN, bbox xyxy)]。"""
    from PIL import Image

    with Image.open(_lp(img_path)) as im:
        W, H = im.size
    insts = []
    with open(_lp(lab_path), encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            a = line.split()
            if len(a) != 5 + 3 * N_KP:
                continue
            v = np.array(a[1:], dtype=float)
            cx, cy, w, h = v[:4]
            kp = v[4:].reshape(N_KP, 3)
            P = kp[:, :2] * (W, H)
            P[kp[:, 2] == 0] = np.nan  # v=0 的座標是 Roboflow 預設位置，不能用
            insts.append((P, np.array([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])))
    return insts


def load_tag_images():
    """tag 資料夾 → 單貓圖片清單 dict(md5, y, P, B, group, file)。"""
    out, skipped = [], Counter()
    for y, root in TRAIN_DIRS.items():
        if not root.is_dir():
            raise SystemExit(f"❌ 找不到 tag 資料夾：{root}")
        for split in sorted(os.listdir(_lp(root))):
            img_dir, lab_dir = root / split / "images", root / split / "labels"
            if not os.path.isdir(_lp(img_dir)):
                continue
            for f in sorted(os.listdir(_lp(img_dir))):
                stem = os.path.splitext(f)[0]
                lab = lab_dir / (stem + ".txt")
                if not os.path.exists(_lp(lab)):
                    skipped["沒有標註檔"] += 1
                    continue
                insts = _read_pose(img_dir / f, lab)
                if len(insts) != 1:
                    skipped[f"{len(insts)} 隻貓"] += 1
                    continue
                out.append(dict(md5=_md5(img_dir / f), y=y, P=insts[0][0], B=insts[0][1],
                                group=_source_group(stem), file=f"{root.name}/{split}/{f}"))
    print(f"  tag 訓練圖：{len(out)} 張 {dict(Counter(r['y'] for r in out))}"
          + (f"；略過 {dict(skipped)}" if skipped else ""))
    return out


def load_eval_images():
    """人工標記資料夾 → 單貓圖片清單 dict(md5, y, y_raw, P, B, group, file)。"""
    if not EVAL_DIR.is_dir():
        raise SystemExit(f"❌ 找不到人工標記資料夾：{EVAL_DIR}")
    out, skipped = [], Counter()
    for d in sorted(os.listdir(_lp(EVAL_DIR))):
        if d not in EVAL_CLASS_MAP:
            continue
        for f in sorted(os.listdir(_lp(EVAL_DIR / d))):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXTS:
                continue
            lab = EVAL_DIR / d / (stem + ".txt")
            if not os.path.exists(_lp(lab)):
                skipped["沒有標註檔"] += 1
                continue
            insts = _read_pose(EVAL_DIR / d / f, lab)
            if len(insts) != 1:
                skipped[f"{len(insts)} 隻貓"] += 1
                continue
            out.append(dict(md5=_md5(EVAL_DIR / d / f), y=EVAL_CLASS_MAP[d], y_raw=d, P=insts[0][0], B=insts[0][1],
                            group=_source_group(stem), file=f"{d}/{f}"))
    print(f"  人工標記評估圖：{len(out)} 張 {dict(Counter(r['y'] for r in out))}"
          + (f"；略過 {dict(skipped)}" if skipped else ""))
    return out


def _load_stop_review():
    if not STOP_REVIEW_CSV.exists():
        raise SystemExit(f"❌ 找不到 stop 影片檢視結果：{STOP_REVIEW_CSV}（使用者逐支看 stop 影片標記的躺坐／站著）")
    with open(STOP_REVIEW_CSV, encoding="utf-8-sig") as fh:
        return {Path(r["video"]).stem: r["choice"] for r in csv.DictReader(fh)}


def _frame_pose(fr):
    kp = np.full((N_KP, 2), np.nan)
    for j in fr["keypoints"]:
        kp[j["joint_id"]] = (j["x"], j["y"])
    return kp, fr["bbox"]


def sample_skeleton_frames(sources, max_per_video):
    """從 skeletons/train/ 抽幀，各類抽到一樣多（＝容量最小的類別），每類的名額輪流分給各支影片
    （影片多的類別每支少抽、影片少的每支多抽），每支影片內在標記幀中等距取。
    回傳 (P, B, y, 抽樣清單[(類別, 影片, frame_id)])。"""
    review = _load_stop_review() if any(flt for src in sources.values() for _, flt in src) else {}
    per_class = {}
    for c, src in sources.items():
        vids = []
        for folder, flt in src:
            for p in sorted((SKELETON_ROOT / "train" / folder).glob("*.json")):
                if flt is not None and review.get(p.stem) != flt:
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                frames = [fr for fr in d["frames"] if fr.get("detected") and fr.get("label") == folder]
                if frames:
                    vids.append((f"{folder}/{p.stem}", frames))
        if not vids:
            raise SystemExit(f"❌ 類別 {c} 從 {src} 抽不到任何幀")
        per_class[c] = vids
    cap = {c: sum(min(len(f), max_per_video) for _, f in v) for c, v in per_class.items()}
    target = min(cap.values())
    P, B, Y, picked = [], [], [], []
    for c, vids in per_class.items():
        lim = [min(len(f), max_per_video) for _, f in vids]
        quota, left = [0] * len(vids), target
        while left:  # 輪流每支影片加 1，直到湊滿（容量用完的影片跳過）
            for i in range(len(vids)):
                if left and quota[i] < lim[i]:
                    quota[i] += 1
                    left -= 1
        for (vid, frames), q in zip(vids, quota):
            for k in ((np.arange(q) + 0.5) * len(frames) / q).astype(int) if q else []:
                fr = frames[k]
                kp, bb = _frame_pose(fr)
                P.append(kp)
                B.append(bb)
                Y.append(c)
                picked.append((c, vid, fr["frame_id"]))
        by_folder = Counter(v.split("/")[0] for (v, _), q in zip(vids, quota) for _ in range(q))
        src_txt = "+".join(f + (f"({flt})" if flt else "") for f, flt in sources[c])
        print(f"  骨架 {c:<8}← {src_txt}：{len(vids)} 支影片、容量 {cap[c]} → 抽 {target} 幀"
              f"（每支 {min(q for q in quota if q)}～{max(quota)} 幀；來源 {dict(by_folder)}）")
    return np.array(P), np.array(B, dtype=float), np.array(Y), picked


def extra_skeleton_jsons():
    """EXTRA_SKELETON_VIDEOS → [(類別, json 路徑)]。對不到骨架 JSON 就停下（避免少了資料卻沒發現）。"""
    out = []
    for c, dirs in EXTRA_SKELETON_VIDEOS.items():
        for d in dirs:
            vids = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in VIDEO_EXTS)
            if not vids:
                raise SystemExit(f"❌ EXTRA_SKELETON_VIDEOS 的資料夾沒有影片：{d}")
            for v in vids:
                hits = sorted(SKELETON_ROOT.glob(f"*/*/{v.stem}.json"))
                if len(hits) != 1:
                    raise SystemExit(f"❌ {v.name} 在 skeletons/ 對到 {len(hits)} 個同名 JSON：{[str(h) for h in hits]}")
                out.append((c, hits[0]))
    return out


def sample_extra_frames(max_per_video):
    """額外骨架影片：每支在有偵測到的幀中等距取最多 max_per_video 幀。回傳 (P, B, y, 抽樣清單)。"""
    P, B, Y, picked = [], [], [], []
    for c, p in extra_skeleton_jsons():
        frames = [fr for fr in json.loads(p.read_text(encoding="utf-8"))["frames"] if fr.get("detected")]
        q = min(len(frames), max_per_video)
        for k in ((np.arange(q) + 0.5) * len(frames) / q).astype(int) if q else []:
            kp, bb = _frame_pose(frames[k])
            P.append(kp)
            B.append(bb)
            Y.append(c)
            picked.append((c, f"{p.parent.name}/{p.stem}", frames[k]["frame_id"]))
    print(f"  額外骨架影片：{len({v for _, v, _ in picked})} 支、{len(picked)} 幀 {dict(Counter(Y))}")
    return np.array(P), np.array(B, dtype=float), np.array(Y), picked


def stack(rows):
    return np.array([r["P"] for r in rows]), np.array([r["B"] for r in rows]), np.array([r["y"] for r in rows])


# ── 同源比對（防外洩；需要 OpenCV） ──────────────────────────────────────

def overlap_videos():
    """要比對的骨架影片：skeletons/train 全部＋EXTRA_SKELETON_VIDEOS（可能在 val／test）。"""
    jsons = sorted((SKELETON_ROOT / "train").glob("*/*.json"))
    return jsons + [p for _, p in extra_skeleton_jsons() if p not in jsons]


def _video_id(p):
    return f"{p.parent.name}/{p.stem}"


def _overlap_images():
    """(集合, file, 路徑)：tag 資料夾全部 split 的圖＋考卷圖，跟訓練讀的是同一批。"""
    out = []
    for root in TRAIN_DIRS.values():
        for split in sorted(os.listdir(_lp(root))):
            d = root / split / "images"
            if os.path.isdir(_lp(d)):
                out += [("tag", f"{root.name}/{split}/{f}", d / f) for f in sorted(os.listdir(_lp(d)))]
    for c in sorted(os.listdir(_lp(EVAL_DIR))):
        d = EVAL_DIR / c
        if c in EVAL_CLASS_MAP:
            out += [("eval", f"{c}/{f}", d / f) for f in sorted(os.listdir(_lp(d)))
                    if os.path.splitext(f)[1].lower() in IMG_EXTS]
    return out


def update_overlap():
    """比對 tag／考卷圖跟骨架影片的畫面 → OVERLAP_CSV＋compared_videos.txt。"""
    import cv2

    def sig(gray):
        v = cv2.resize(gray, (SIG_SIZE, SIG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-6)

    def image_sig(path):  # 裁掉 letterbox 黑邊
        g = cv2.imdecode(np.fromfile(_lp(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        rows, cols = np.where(g.max(1) > 12)[0], np.where(g.max(0) > 12)[0]
        if len(rows) and len(cols):
            g = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        return sig(g), _md5(path)

    t0 = time.time()
    print("▶ 同源比對 [1/3] 讀圖片")
    imgs = _overlap_images()
    sigs, md5s = zip(*(image_sig(p) for _, _, p in imgs))
    E = np.stack(sigs)
    print(f"  tag {sum(s == 'tag' for s, *_ in imgs)} 張、考卷 {sum(s == 'eval' for s, *_ in imgs)} 張")

    jsons = overlap_videos()
    print(f"▶ 同源比對 [2/3] 比對 {len(jsons)} 支骨架影片（每 {FRAME_STEP} 幀取 1）")
    best = np.full(len(E), -2.0)
    best_vid, best_fr = [""] * len(E), [-1] * len(E)
    linked = [[] for _ in E]  # 每張圖：相似度 ≥ 門檻的 (影片, 相似度)
    failed = 0
    for i, p in enumerate(jsons, 1):
        vpath = json.loads(p.read_text(encoding="utf-8"))["video_metadata"]["video_path"]
        cap = cv2.VideoCapture(vpath)
        vs, idx, k = [], [], 0
        while cap.isOpened() and cap.grab():
            if k % FRAME_STEP == 0:
                ok, fr = cap.retrieve()
                if ok:
                    vs.append(sig(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
                    idx.append(k)
            k += 1
        cap.release()
        if not vs:
            print(f"  ❌ 讀不到影片：{vpath}")
            failed += 1
            if failed == len(jsons):
                raise SystemExit("❌ 全部影片都讀不到，結束")
            continue
        S = E @ np.stack(vs).T
        j = S.argmax(1)
        m = S[np.arange(len(E)), j]
        vid = _video_id(p)
        for e in np.flatnonzero(m >= SAME_VIDEO_THRESHOLD):
            linked[e].append((vid, m[e]))
        for e in np.flatnonzero(m > best):
            best[e], best_vid[e], best_fr[e] = m[e], vid, idx[j[e]]
        if i % 25 == 0 or i == len(jsons):
            print(f"  [{i}/{len(jsons)}] ▶ 目前影片：{vid}")

    print("▶ 同源比對 [3/3] 存檔")
    OVERLAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    found = {f for s, f, _ in imgs if s == "eval"} & set(MANUAL_LINKS)
    if len(found) != len(MANUAL_LINKS):
        print(f"  ⚠ MANUAL_LINKS 有 {len(MANUAL_LINKS) - len(found)} 張在考卷裡找不到（檔案被移動？）")
    for (s, f, _), md5, sim, vid, fr, lk in zip(imgs, md5s, best, best_vid, best_fr, linked):
        lk = [v for v, _ in sorted(lk, key=lambda t: -t[1])]
        if s == "eval" and MANUAL_LINKS.get(f) and MANUAL_LINKS[f] not in lk:
            lk.append(MANUAL_LINKS[f])
        rows.append([s, f, md5, vid, fr, f"{sim:.4f}", ";".join(lk)])
    rows.sort(key=lambda r: (r[0], -float(r[5])))
    with open(OVERLAP_CSV, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["集合", "file", "md5", "最像影片", "影片幀", "相似度", "同源影片"])
        w.writerows(rows)
    (OVERLAP_DIR / "compared_videos.txt").write_text("\n".join(_video_id(p) for p in jsons) + "\n", encoding="utf-8")
    for s in ("tag", "eval"):
        hit = [r for r in rows if r[0] == s and r[6]]
        print(f"  {s}：{len(hit)} 張跟 {len({v for r in hit for v in r[6].split(';')})} 支骨架影片同源"
              f"（門檻 {SAME_VIDEO_THRESHOLD}）")
    print(f"✓ {OVERLAP_CSV}（{time.time() - t0:.0f} 秒）")


def _overlap_stale(md5s):
    """同源清單不存在、少了現在的圖、或少了現在要用的骨架影片 → 回傳原因；沒過期回傳 None。"""
    listed = OVERLAP_DIR / "compared_videos.txt"
    if not OVERLAP_CSV.exists() or not listed.exists():
        return "還沒產生"
    with open(OVERLAP_CSV, encoding="utf-8-sig") as fh:
        have = {r["md5"] for r in csv.DictReader(fh)}
    if set(md5s) - have:
        return f"有 {len(set(md5s) - have)} 張新圖"
    new_vids = {_video_id(p) for p in overlap_videos()} - set(listed.read_text(encoding="utf-8").split())
    if new_vids:
        return f"有 {len(new_vids)} 支新骨架影片"
    return None


def load_overlap(md5s):
    """OVERLAP_CSV → {md5: set(同源骨架影片)}；清單過期就先重跑同源比對。"""
    why = _overlap_stale(md5s)
    if why:
        print(f"  同源清單需要更新（{why}），先跑同源比對（約 3 分鐘）")
        if _has_opencv():
            update_overlap()
        else:
            _rerun_with_opencv(["--update-overlap"])
        why = _overlap_stale(md5s)
        if why:
            raise SystemExit(f"❌ 同源比對跑完清單還是不完整（{why}）")
    with open(OVERLAP_CSV, encoding="utf-8-sig") as fh:
        return {r["md5"]: set(filter(None, r["同源影片"].split(";"))) for r in csv.DictReader(fh)}


# ── --extract-frames：依骨架 JSON 把影片抽成圖片＋YOLO-pose 標註（補資料用；需要 OpenCV） ──

def extract_frames(video_dir):
    """<影片資料夾>/*.mp4 → <影片資料夾>_frames/<影片名>/{images,labels}/（結構同 tag 資料夾）＋_review 拼貼＋frames.csv。
    影片名＝skeletons/<split>/<類別>/<同名>.json，並比對 JSON 記的影片檔大小確認是同一支。
    依 original_frame_id 取影片的同一幀；17 點都寫 v=2（跟訓練讀骨架幀的方式一致），各點信心存 frames.csv。"""
    import cv2

    def find_json(video):
        hits = sorted(SKELETON_ROOT.glob(f"*/*/{video.stem}.json"))
        if not hits:
            return None, "skeletons/ 裡找不到同名 JSON"
        for h in hits:
            d = json.loads(h.read_text(encoding="utf-8"))
            vp = d.get("video_metadata", {}).get("video_path", "")
            if vp and os.path.exists(vp) and os.path.getsize(vp) == video.stat().st_size:
                return h, d
        return None, f"同名 JSON {len(hits)} 個，但記錄的影片檔大小都對不上"

    def draw(img, kps, bbox):
        img = img.copy()
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 3)
        r = max(3, img.shape[1] // 200)
        for k in kps:
            cv2.circle(img, (int(k["x"]), int(k["y"])), r, (0, 0, 255) if k["conf"] >= 0.5 else (255, 0, 255), -1)
        return img

    def sheet(frames, title, path, T=300, H=30, C=6):
        tiles = []
        for fid, img in frames:
            h, w = img.shape[:2]
            s = T / max(h, w)
            im = cv2.resize(img, (int(w * s), int(h * s)))
            t = np.full((T + H, T, 3), 255, np.uint8)
            y0, x0 = H + (T - im.shape[0]) // 2, (T - im.shape[1]) // 2
            t[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.putText(t, f"{title} f{fid}", (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(t)
        nr = (len(tiles) + C - 1) // C
        out = np.full((nr * (T + H + 6) + 6, C * (T + 6) + 6, 3), 190, np.uint8)
        for k, t in enumerate(tiles):
            y, x = 6 + (k // C) * (T + H + 6), 6 + (k % C) * (T + 6)
            out[y:y + T + H, x:x + T] = t
        cv2.imencode(".png", out)[1].tofile(str(path))

    vdir = Path(os.path.abspath(video_dir))
    if not vdir.is_dir():
        raise SystemExit(f"❌ 找不到影片資料夾：{vdir}")
    videos = sorted(p for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"❌ {vdir} 裡沒有影片")
    out_root = _free_dir(vdir.with_name(vdir.name + "_frames"))
    (out_root / "_review").mkdir(parents=True)
    t0 = time.time()
    print(f"▶ 抽幀：{len(videos)} 支影片 → {out_root}")
    skipped, total = [], 0
    with open(out_root / "frames.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["影片", "骨架 JSON", "幀號", "JSON label", *(f"conf_{i}" for i in range(N_KP))])
        for i, v in enumerate(videos, 1):
            jpath, d = find_json(v)
            if jpath is None:
                print(f"  [{i}/{len(videos)}] ❌ 略過 {v.name}：{d}")
                skipped.append(v.name)
                if len(skipped) == len(videos):
                    raise SystemExit("❌ 全部影片都找不到對應的骨架，結束")
                continue
            want = {}
            for fr in d["frames"]:
                if fr.get("detected") and fr.get("keypoints") and fr.get("bbox"):
                    want.setdefault(int(fr.get("original_frame_id", fr["frame_id"])), fr)
            img_dir, lab_dir = out_root / v.stem / "images", out_root / v.stem / "labels"
            img_dir.mkdir(parents=True)
            lab_dir.mkdir(parents=True)
            ids = sorted(want)
            review_ids = {ids[j] for j in np.linspace(0, len(ids) - 1, min(REVIEW_FRAMES, len(ids))).astype(int)} if ids else set()
            review, cap, k, saved = [], cv2.VideoCapture(str(v)), 0, 0
            while saved < len(want) and cap.grab():
                if k in want:
                    ok, img = cap.retrieve()
                    if ok:
                        fr = want[k]
                        H, W = img.shape[:2]
                        name = f"{v.stem}_mp4-{k:04d}"
                        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tofile(str(img_dir / f"{name}.jpg"))
                        x1, y1, x2, y2 = fr["bbox"]
                        kps = sorted(fr["keypoints"], key=lambda j: j["joint_id"])
                        vals = [f"{(x1 + x2) / 2 / W:.6f}", f"{(y1 + y2) / 2 / H:.6f}", f"{(x2 - x1) / W:.6f}", f"{(y2 - y1) / H:.6f}"]
                        for j in kps:
                            vals += [f"{j['x'] / W:.6f}", f"{j['y'] / H:.6f}", "2"]
                        (lab_dir / f"{name}.txt").write_text("0 " + " ".join(vals) + "\n", encoding="utf-8")
                        w.writerow([v.stem, str(jpath.relative_to(SKELETON_ROOT)), k, fr.get("label", ""),
                                    *(f"{j['conf']:.3f}" for j in kps)])
                        if k in review_ids:
                            review.append((k, draw(img, kps, fr["bbox"])))
                        saved += 1
                k += 1
            cap.release()
            if review:
                sheet(review, v.stem, out_root / "_review" / f"{v.stem}.png")
            total += saved
            print(f"  [{i}/{len(videos)}] {jpath.relative_to(SKELETON_ROOT)}：存 {saved}/{len(want)} 幀")
            print(f"▶ 目前影片：{v.name}")
    print(f"✓ 共 {total} 幀，{len(videos) - len(skipped)} 支影片；略過 {len(skipped)} 支 {skipped or ''}")
    print(f"  人工確認拼貼：{out_root / '_review'}（紅點＝信心 ≥0.5、紫點＝信心 <0.5）（{time.time() - t0:.0f} 秒）")


# ── 訓練 ───────────────────────────────────────────────────────────────

def train_and_evaluate(args):
    global MAX_FRAMES_PER_VIDEO
    if args.max_per_video:
        MAX_FRAMES_PER_VIDEO = args.max_per_video

    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    t0 = time.time()
    out_report = _next_run_dir(report_dir("eval", "pose_classifier_from_tags"))
    lines = [f"單幀姿態分類（四類，tag＋骨架抽幀訓練、人工標記評估）  {datetime.now():%Y-%m-%d %H:%M}",
             f"類別：{CLASSES}（walk＝走／站／甩頭、other＝躺／坐）",
             "tag：" + "、".join(f"{c}={p.name}" for c, p in TRAIN_DIRS.items()), f"評估：{EVAL_DIR}"]

    print("▶ [1/4] 讀取 tag 圖片、人工標記、骨架抽幀")
    flip_idx = load_flip_idx(FLIP_YAML)
    tag, ev = load_tag_images(), load_eval_images()
    lines.append(f"tag 訓練圖 {len(tag)} 張 {dict(Counter(r['y'] for r in tag))}；"
                 f"人工評估圖 {len(ev)} 張 {dict(Counter(r['y'] for r in ev))}")

    # 同一張圖（MD5）在 tag 與評估集都有時，分組以 tag 那份為準（評估資料夾裡的檔名可能被截短過）
    tag_group = {r["md5"]: r["group"] for r in tag}
    for r in ev:
        r["group"] = tag_group.get(r["md5"], r["group"])
    tag_label = {r["md5"]: r["y"] for r in tag}
    n_both = sum(r["md5"] in tag_label for r in ev)
    n_diff = sum(r["md5"] in tag_label and tag_label[r["md5"]] != r["y"] for r in ev)
    lines.append(f"評估圖同時在 tag 資料夾：{n_both} 張，其中 tag 與人工不同類：{n_diff} 張")
    print(f"  {lines[-1]}")

    Pt, Bt, yt = stack(tag)
    Pe, Be, ye = stack(ev)
    gt = np.array([r["group"] for r in tag])
    ge = np.array([r["group"] for r in ev])
    ratios = fit_scale_ratios(Pt, Bt)

    # 骨架抽樣幀：鏡像＋遮點（遮點比例依 tag 圖各點缺失比例）
    Xsk, ysk = np.zeros((0, len(make_features(Pt[:1], Bt[:1], ratios)[1])), np.float32), np.array([], dtype=str)
    if not args.no_skeleton_sample:
        print(f"  骨架抽幀（每支影片最多 {MAX_FRAMES_PER_VIDEO} 幀，各類等量）")
        Psk, Bsk, ysk0, picked = sample_skeleton_frames(SKELETON_SOURCES, MAX_FRAMES_PER_VIDEO)
        if EXTRA_SKELETON_VIDEOS:
            Pex, Bex, yex, pex = sample_extra_frames(EXTRA_MAX_PER_VIDEO)
            Psk, Bsk, ysk0, picked = (np.concatenate([Psk, Pex]), np.concatenate([Bsk, Bex]),
                                      np.concatenate([ysk0, yex]), picked + pex)
        v0 = np.isnan(Pt[:, :, 0]).mean(0)
        Xsk, ysk, _ = augmented(Psk, Bsk, ysk0, np.clip(v0 * MASK_SCALE, 0, 0.5), ratios,
                                np.random.default_rng(RANDOM_SEED), flip_idx)
        sk_vid = np.tile(np.array([v for _, v, _ in picked]), 3)  # augmented() 疊成 原圖／鏡像／遮點 三份
        lines.append(f"骨架抽幀：{dict(Counter(ysk0))}（每支影片最多 {MAX_FRAMES_PER_VIDEO} 幀；"
                     f"來源 {SKELETON_SOURCES}；額外 {EXTRA_SKELETON_VIDEOS}，每支最多 {EXTRA_MAX_PER_VIDEO} 幀）；"
                     "每一折都加入訓練（排除跟該折考題同源的影片），鏡像＋遮點")
        with open(out_report / "skeleton_frames.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["類別", "影片", "frame_id"])
            w.writerows(picked)
    else:
        lines.append("骨架抽幀：不使用（--no-skeleton-sample）")
        sk_vid = np.array([], dtype=str)
    missing = [c for c in CLASSES if c not in set(yt) | set(ysk)]
    if missing:
        raise SystemExit(f"❌ 類別 {missing} 沒有任何訓練資料（other 只來自骨架抽幀，不能加 --no-skeleton-sample）")

    # 依來源分組切 5 折：以「來源」為單位分配，tag 圖與評估圖同來源就同一折
    groups = sorted(set(gt) | set(ge))
    gidx = {g: i for i, g in enumerate(groups)}
    g_label = {}  # 每個來源的主要類別（分層用）
    for g, y in list(zip(gt, yt)) + list(zip(ge, ye)):
        g_label.setdefault(g, Counter())[y] += 1
    g_arr = np.arange(len(groups))
    g_y = np.array([g_label[g].most_common(1)[0][0] for g in groups])
    fold_of = np.empty(len(groups), dtype=int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
                                 .split(g_arr, g_y, g_arr)):
        fold_of[te] = k
    ft = fold_of[[gidx[g] for g in gt]]
    fe = fold_of[[gidx[g] for g in ge]]
    lines.append(f"交叉驗證：依來源分組 {N_FOLDS} 折（{len(groups)} 個來源）")

    # 防外洩：第 k 折訓練時，排除跟第 k 折要考的 tag／考卷圖同源的骨架影片
    sk_keep = {k: np.ones(len(sk_vid), bool) for k in range(N_FOLDS)}
    if len(sk_vid):
        links = load_overlap([r["md5"] for r in tag] + [r["md5"] for r in ev])
        n_frames = len(sk_vid) // 3
        for k in range(N_FOLDS):
            bad = set().union(*(links[r["md5"]] for r, f in zip(tag, ft) if f == k),
                              *(links[r["md5"]] for r, f in zip(ev, fe) if f == k))
            sk_keep[k] = ~np.isin(sk_vid, list(bad))
            drop = Counter(ysk[:n_frames][~sk_keep[k][:n_frames]])
            lines.append(f"  第 {k} 折排除同源骨架影片 {len(bad & set(sk_vid))} 支、"
                         f"{n_frames - sk_keep[k][:n_frames].sum()} 幀 {dict(sorted(drop.items()))}")
            print(lines[-1])

    classes = np.array(CLASSES)

    def train(mask, params, fold=None):
        X, y = human_augmented(Pt[mask], Bt[mask], yt[mask], ratios, flip_idx)
        keep = np.ones(len(ysk), bool) if fold is None else sk_keep[fold]  # fold=None：最終模型，全部骨架幀都用
        return fit_model(np.vstack([X, Xsk[keep]]), np.concatenate([y, ysk[keep]]), params)

    def cv(params):
        """回傳 (tag OOF 機率, 評估圖 OOF 機率)。"""
        oof_t, oof_e = np.zeros((len(yt), len(CLASSES))), np.zeros((len(ye), len(CLASSES)))
        for k in range(N_FOLDS):
            mdl = train(ft != k, params, fold=k)
            assert list(mdl.classes_) == CLASSES
            if (ft == k).any():
                oof_t[ft == k] = predict_proba_tta(mdl, Pt[ft == k], Bt[ft == k], ratios, flip_idx)
            if (fe == k).any():
                oof_e[fe == k] = predict_proba_tta(mdl, Pe[fe == k], Be[fe == k], ratios, flip_idx)
        return oof_t, oof_e

    tag_classes = sorted(set(yt))  # 選超參數只看 tag 有的類別
    print(f"▶ [2/4] 選超參數（{len(PARAM_GRID)} 組，只看 tag 資料的交叉驗證 {tag_classes}，不看人工評估集）")
    best = None
    for i, params in enumerate(PARAM_GRID, 1):
        oof_t, oof_e = cv(params)
        f1 = f1_score(yt, classes[oof_t.argmax(1)], labels=tag_classes, average="macro")
        print(f"  [{i}/{len(PARAM_GRID)}] min_samples_leaf={params['min_samples_leaf']} "
              f"max_iter={params['max_iter']} → tag 交叉驗證 macro-F1 {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, params, oof_t, oof_e)
    _, params, oof_t, oof_e = best
    lines.append(f"\n最佳超參數（tag 交叉驗證 macro-F1 {best[0]:.3f}）：{params}")
    print(f"  ✓ 最佳：{params}")

    print("▶ [3/4] 評估")
    how = "只用 tag" if args.no_skeleton_sample else f"tag＋骨架抽幀（每支影片最多 {MAX_FRAMES_PER_VIDEO} 幀）"
    res_e = summarize(ye, classes[oof_e.argmax(1)], CLASSES)
    print_summary(f"【主要結果：人工標記 {len(ye)} 張（依來源分組交叉驗證），{how}訓練】", res_e, CLASSES, lines)
    res_t = summarize(yt, classes[oof_t.argmax(1)], CLASSES)
    res_t["macro_f1"] = f1_score(yt, classes[oof_t.argmax(1)], labels=tag_classes, average="macro")
    print_summary(f"【參考：對 tag 標記的吻合度（{len(yt)} 張，tag 本身可能有錯；macro-F1 只算 tag 有的類別）】",
                  res_t, CLASSES, lines)
    diff = np.array([r["md5"] in tag_label and tag_label[r["md5"]] != r["y"] for r in ev])
    if diff.any():
        pred = classes[oof_e.argmax(1)]
        lines.append(f"  tag 與人工不同類的 {diff.sum()} 張：判成人工類別 {(pred[diff] == ye[diff]).sum()} 張、"
                     f"判成 tag 類別 {sum(pred[i] == tag_label[ev[i]['md5']] for i in np.flatnonzero(diff))} 張")
        print(lines[-1])

    print("▶ [4/4] 用全部訓練資料訓練最終模型、存報告")
    final = train(np.ones(len(yt), bool), params)
    import joblib

    joblib.dump(dict(model=final, classes=CLASSES, feature_names=make_features(Pt[:1], Bt[:1], ratios)[1],
                     ratios=ratios, flip_idx=flip_idx.tolist(), params=params,
                     train_dirs={c: str(p) for c, p in TRAIN_DIRS.items()},
                     skeleton_sources=None if args.no_skeleton_sample else SKELETON_SOURCES,
                     extra_skeleton_videos=None if args.no_skeleton_sample else
                     {c: [str(d) for d in ds] for c, ds in EXTRA_SKELETON_VIDEOS.items()},
                     max_frames_per_video=MAX_FRAMES_PER_VIDEO), out_report / "model.joblib")
    plot_confusions([(f"人工標記（{how}）", res_e["cm"], CLASSES), ("對 tag 的吻合度", res_t["cm"], CLASSES)],
                    out_report / "confusion_matrices.png")
    with open(out_report / "metrics.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["評估", "類別", "precision", "recall", "f1", "support"])
        for name, res in (("eval_human", res_e), ("tag_agreement", res_t)):
            for c in CLASSES:
                m = res["per_class"][c]
                w.writerow([name, c, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]])
            w.writerow([name, "macro", "", "", f"{res['macro_f1']:.4f}", ""])
    with open(out_report / "eval_predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "人工", "人工類別", "tag類別", "來源", "fold", "pred", *(f"p_{c}" for c in CLASSES)])
        for i, r in enumerate(ev):
            w.writerow([r["file"], r["y_raw"], r["y"], tag_label.get(r["md5"], ""), r["group"], int(fe[i]),
                        classes[oof_e[i].argmax()], *(f"{v:.4f}" for v in oof_e[i])])
    # tag 圖片的交叉驗證判斷：模型跟 tag 不同類的，是 tag 可能標錯的候選（給人工複查）
    with open(out_report / "tag_predictions.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "tag類別", "來源", "fold", "pred", *(f"p_{c}" for c in CLASSES)])
        for i, r in enumerate(tag):
            w.writerow([r["file"], r["y"], r["group"], int(ft[i]), classes[oof_t[i].argmax()],
                        *(f"{v:.4f}" for v in oof_t[i])])
    lines.append(f"\n總花費時間：{time.time() - t0:.0f} 秒")
    (out_report / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 報告：{out_report}")


def main():
    ap = argparse.ArgumentParser(description="四類單幀姿態分類器：tag 圖片＋骨架抽幀訓練，人工標記評估")
    ap.add_argument("--no-skeleton-sample", action="store_true", help="不從骨架抽幀，只用 tag 圖片訓練")
    ap.add_argument("--max-per-video", type=int, metavar="N",
                    help=f"每支影片最多抽幾幀（預設 {MAX_FRAMES_PER_VIDEO}）")
    ap.add_argument("--update-overlap", action="store_true", help="只重跑圖片↔骨架影片同源比對（需要 OpenCV）")
    ap.add_argument("--extract-frames", metavar="影片資料夾", nargs="?", const=str(EXTRACT_FRAMES_DIR),
                    help=f"依骨架 JSON 把影片抽成圖片＋YOLO-pose 標註（補資料用，需要 OpenCV；沒給資料夾＝{EXTRACT_FRAMES_DIR}）")
    args = ap.parse_args()

    if args.update_overlap:
        update_overlap() if _has_opencv() else _rerun_with_opencv(["--update-overlap"])
    elif args.extract_frames:
        extract_frames(args.extract_frames) if _has_opencv() else _rerun_with_opencv(["--extract-frames", args.extract_frames])
    else:
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
