"""
Compare 2~5 ST-GCN models on up to five labeled videos (walk, lick, scratch, shake, stop).

Metrics (per model, per class):
  1. Discrete accuracy     — argmax(probs) == true_label       (硬指標)
  2. Avg true-class prob   — mean(probs[:, true_class])        (軟指標：平均信心)
  3. Max true-class prob   — max(probs[:, true_class])         (峰值信號：事件偵測能力)
  4. Event detected        — n_correct >= EVENT_MIN_WINDOWS    (影片層級偵測，對含靜止段影片最公平)
  5. Macro F1 & confusion matrix                                (分類輪廓)

主指標：Accuracy（argmax 正確率）——其餘指標（Event Detection Rate/Macro F1/複合分數）
皆為參考，不會單獨決定最終結論。詳見 print_final_summary()。

Output directory: <output>/comparison_NNN_<name1>_vs_<name2>_vs_.../
  NNN = next sequential number, consistent with 0_train_gcn.py convention

Files:
  <name>_<class>_preds.csv        （每個模型各一份）
  comparison_summary.csv
  accuracy_comparison.png    (grouped bar: discrete acc + avg true-class prob, N 個模型並排)
  confusion_matrices.png     (N 個 heatmap 並排)
"""
import argparse
import csv
import re
from pathlib import Path
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# Joint Prior Weights 標註用中文，預設 DejaVu Sans 沒有 CJK 字型會顯示成方框，
# 跟 eval_model_worst_videos.py 用同一組字型清單
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'MingLiU', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from detectors.keypoint_detector import KeypointDetector
from detectors.behavior_classifier import BehaviorClassifier
from models.stgcn_model import (
    interpolate_missing,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    build_feature_tensor,
)
from utils.constants import BEHAVIOR_CLASSES
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import chi2, binomtest

# ── Channel → feature mode ────────────────────────────────────────────────
CH_TO_FEATURE = {
    2: 'xy',
    3: 'xy_conf',
    5: 'xy_conf_v',
    7: 'xy_conf_v_bone',
    9: 'xy_conf_v_bone_bmotion',
}

# ── Default / hardcoded paths ─────────────────────────────────────────────
DEFAULT_YOLO    = r"C:\AI_Project\cat_pose\v11s_121.pt"
DEFAULT_IMGSZ   = 640
DEFAULT_CONF    = 0.5
DEFAULT_SEQ_LEN = 16
DEFAULT_STRIDE  = 3
EVENT_MIN_WINDOWS = 3   # 保留作備用下限；實際以比例門檻為主
EVENT_MIN_RATIO   = 0.30  # 事件偵測（比例門檻）：正確 window 數 / 總 window 數 ≥ 此值即視為偵測成功
PROB_EVENT_THRESHOLD = 0.40  # 機率門檻型事件偵測：true-class prob ≥ 此值的 window 達 EVENT_MIN_WINDOWS 即算偵測
DEFAULT_DEVICE  = 'cuda'

# 影片內並非整段都屬於該類別的行為（例如 scratch/shake 只在影片中出現一小段，
# 其餘畫面是 stop 或其他行為），這些類別的絕對準確率會被結構性拉低——不是模型
# 缺陷，是評測資料本身的特性。這裡列出的類別在報表中會加 † 標記，並且比較
# 多個模型時建議優先看「Δ vs baseline」的相對變化，而非本表的絕對數字。
PARTIAL_COVERAGE_CLASSES = {'scratch', 'shake'}

# 預設比較清單：2~5 筆皆可，每筆為 {path, name, ema_alpha, seq_len}。
# name=None 時自動從檔名推導；--models 等 CLI 參數會整個覆蓋這份清單。
HARD_MODELS = [
    {'path': r"C:\Users\homec\Downloads\stgcn_results\run_119_xy_conf_v_bone_att_on\119_best_model.pth",
     'name': None, 'ema_alpha': 1.0, 'seq_len': 16},
    {'path': r"C:\Users\homec\Downloads\stgcn_results\run_120_xy_conf_v_bone_att_on\120_best_model.pth",
     'name': None, 'ema_alpha': 1.0, 'seq_len': 16},
   # {'path': r"C:\Users\homec\Downloads\stgcn_results\run_116_seqlen_ablation_att_on\116_xy_conf_v_bone_T32_att_on.pth",
   #  'name': None, 'ema_alpha': 1.0, 'seq_len': 32},
  #   {'path': r"C:\Users\homec\Downloads\stgcn_results\run_095_reg_ablation_att_on\4.pth",
   #  'name': None, 'ema_alpha': 1.0, 'seq_len': 16},
 #    {'path': r"C:\Users\homec\Downloads\stgcn_results\run_095_reg_ablation_att_on\5.pth",
#     'name': None, 'ema_alpha': 1.0, 'seq_len': 16},
]

HARD_VIDEO_WALK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk"
HARD_VIDEO_LICK_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick"
HARD_VIDEO_SCRATCH_DIR = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch"
HARD_VIDEO_SHAKE_DIR   = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake"
HARD_VIDEO_STOP_DIR    = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\stop"
HARD_OUTPUT_DIR        = r"C:\ai_project\paper\cat_monitoring_system\eval_results"

# ── 視覺樣式：最多支援 5 個模型，一模型一色 ─────────────────────────────────
_PALETTE = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']  # 藍/橘/綠/紫/紅

# 跟 0_train_gcn.py / eval_pose_models.py 同一份骨架定義（joint_id 0~16 依序對應）
KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back",
    "Hip", "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]


def get_joint_prior_weights_str(clf) -> str:
    """讀取分類器內部 STGCN 模型的 JointAttention.prior_weights buffer（訓練時
    設定的固定關節先驗權重，隨 checkpoint 存讀，見 0_train_gcn.py 的
    USE_JOINT_PRIOR_WEIGHTS/JOINT_PRIOR_WEIGHTS），格式化成一行文字，只列出非
    1.0 的關節。模型沒有 attention 模組，或找不到這個 buffer（例如比較新舊版本
    checkpoint 時舊版沒有這個欄位）時回傳說明文字，不中斷繪圖。"""
    try:
        joint_attn = clf.model.model.joint_attention
        weights = joint_attn.prior_weights.detach().cpu().numpy().flatten()
    except AttributeError:
        return "無 attention 模組"
    names = KEYPOINT_NAMES[:len(weights)]
    non_default = [f'{names[i]}={weights[i]:.1f}' for i in range(len(weights))
                  if abs(weights[i] - 1.0) > 1e-6]
    return ', '.join(non_default) if non_default else "無自訂權重（全部關節=1.0）"

# ── Windows MAX_PATH 保護 ────────────────────────────────────────────────────
# 完整 label（例如 "095_xy_conf_v_bone_manual_tuning_scaled_lr_att_on"）在 N 個
# 模型時會被 "_vs_" 全部串在一起當資料夾名稱，同時每個模型的 preds.csv 檔名也
# 用同一份全名——5 個這種長 label 串起來，加上輸出根目錄，總路徑很容易超過
# Windows 預設 260 字元的 MAX_PATH，導致 mkdir()/開檔失敗
# （FileNotFoundError: [WinError 3]，錯誤本身跟 mkdir 邏輯無關，是路徑長度限制）。
# 這裡一律用截短過的安全短名（"A_xxx" 這種格式，字母字首保證彼此不會撞名）
# 取代 label 用在所有實際檔案系統路徑；完整名稱仍完整保留在
# comparison_summary.csv 的欄位標題跟主控台輸出裡，不會遺失資訊。
_MAX_LABEL_LEN_FOR_PATH = 18
_WINDOWS_MAX_PATH_SAFE  = 240  # 留一點餘裕，不用卡在剛好 259


def _fs_safe_labels(labels: list) -> list:
    """把顯示用的完整 label 轉成資料夾/檔名安全的短版本，只用在實際檔案系統
    路徑，不影響任何顯示用的完整名稱（console 輸出、CSV 欄位標題等）。"""
    return [f"{chr(65 + i)}_{l[:_MAX_LABEL_LEN_FOR_PATH]}" for i, l in enumerate(labels)]


def _build_output_dir(out_root: Path, run_tag: str, fs_labels: list) -> Path:
    """組出這次比較的輸出資料夾路徑，確保總長度不會撞到 Windows MAX_PATH。
    先試完整（已截短過的）label 串接；估算加上資料夾內最長檔名
    （label_behavior_preds.csv）後仍太長，就退化成只標模型數量的精簡命名。"""
    margin = 60  # 資料夾內最長可能檔名（含分隔符）的安全預留空間
    candidates = [
        f"comparison_{run_tag}_{'_vs_'.join(fs_labels)}",
        f"comparison_{run_tag}_{len(fs_labels)}models",
    ]
    for name in candidates:
        if len(str(out_root / name)) + margin <= _WINDOWS_MAX_PATH_SAFE:
            return out_root / name
    print(f"  ⚠ 輸出路徑仍接近 Windows 260 字元上限（--output 本身路徑太深），"
          f"建議改用更短的 --output")
    return out_root / candidates[-1]


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════
def _next_comparison_number(out_root: str) -> int:
    """掃描 out_root/comparison_NNN_* 目錄，回傳下一個可用編號。"""
    p = Path(out_root)
    if not p.exists():
        return 1
    pat = re.compile(r'^comparison_(\d+)')
    max_num = 0
    for d in p.iterdir():
        if d.is_dir():
            m = pat.match(d.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def _short_name(model_path: str) -> str:
    """從模型路徑提取簡短名稱，優先取 run number + mode 部分。"""
    stem = Path(model_path).stem          # e.g. stgcn_best_001_xy_v_att_on
    parts = stem.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) >= 3:
            return '_'.join(parts[i:])    # e.g. 001_xy_v_att_on
    return stem[-28:] if len(stem) > 28 else stem


def infer_bn_input_channels(model_path: str):
    """從 checkpoint 的 bn_input.weight 推斷輸入通道數。"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"模型檔案不存在: {model_path}")
    ck = torch.load(model_path, map_location='cpu')
    sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
    if isinstance(sd, dict):
        for k, v in sd.items():
            if k.endswith('bn_input.weight'):
                return int(v.shape[0])
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Inference（單一模型，跟模型數量無關，維持原樣）
# ═══════════════════════════════════════════════════════════════════════════
def _apply_ema(preds: list, alpha: float) -> list:
    """
    對 preds 序列的 probs 套用 EMA 平滑。
    alpha=1.0 → 不平滑（原始值）；alpha=0.5 → 半衰期平滑。
    pred/conf 會從平滑後的 probs 重新計算。
    """
    if alpha >= 1.0 or not preds:
        return preds
    smoothed = []
    ema = None
    for p in preds:
        raw = np.array(p['probs'], dtype=np.float32)
        ema = raw if ema is None else alpha * raw + (1.0 - alpha) * ema
        new_pred = int(np.argmax(ema))
        smoothed.append({
            **p,
            'probs': ema.tolist(),
            'pred':  new_pred,
            'conf':  float(ema[new_pred]),
        })
    return smoothed


def evaluate_video(video_path, kp_detector, classifier, feature_mode,
                   sequence_length=16, classify_stride=2, ema_alpha=1.0):
    """
    對單一影片執行逐幀推論。
    ema_alpha: EMA 平滑係數，1.0 = 不平滑，0.5 = 半衰期平滑。
    回傳 list of dicts: {frame, time, pred, conf, probs}
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    buf = deque(maxlen=sequence_length)
    preds = []
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        kpts, kpt_conf, _, _ = kp_detector.detect(frame)
        buf.append((kpts, kpt_conf) if kpts is not None else (None, None))

        if len(buf) < sequence_length or frame_idx % classify_stride != 0:
            continue

        kpts_arr = np.array([
            item[0] if item[0] is not None else np.zeros((17, 2), np.float32)
            for item in buf
        ])
        conf_arr = np.array([
            item[1] if item[1] is not None else np.zeros((17,), np.float32)
            for item in buf
        ])

        _model_joints = getattr(classifier.model, 'num_joints', 17)
        if _model_joints < 17:
            kpts_arr = kpts_arr[:, :_model_joints, :]
            conf_arr = conf_arr[:, :_model_joints]

        seq = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)
        seq = flip_normalize(seq)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        feats = build_feature_tensor(seq, conf_arr, feature_mode)

        pred_id, pred_conf, pred_probs = classifier.model.predict(feats, precomputed=True)
        if pred_id is None:
            pred_id, pred_conf, pred_probs = -1, 0.0, [0.0] * len(BEHAVIOR_CLASSES)

        preds.append({
            'frame': frame_idx,
            'time':  round(frame_idx / fps, 3),
            'pred':  int(pred_id),
            'conf':  float(pred_conf),
            'probs': [float(x) for x in pred_probs],
        })

    cap.release()
    return _apply_ema(preds, ema_alpha)


_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.mpg', '.mpeg', '.webm'}


def evaluate_folder(folder_path, kp_detector, classifier, feature_mode,
                    sequence_length=16, classify_stride=2, ema_alpha=1.0):
    """
    對資料夾內所有影片執行推論。
    回傳 list of (filename, preds_list)，每部影片一個元素。
    """
    folder = Path(folder_path)
    videos = sorted(f for f in folder.iterdir() if f.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        print(f"    ⚠ No videos found in: {folder}")
        return []
    results = []
    for vid in videos:
        print(f"    {vid.name} ...", end=' ', flush=True)
        preds = evaluate_video(vid, kp_detector, classifier, feature_mode,
                               sequence_length, classify_stride, ema_alpha)
        print(f"{len(preds)} windows")
        results.append((vid.name, preds))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Metrics（單一模型，跟模型數量無關，維持原樣）
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(preds_by_class: dict,
                    event_min_windows: int = EVENT_MIN_WINDOWS,
                    event_min_ratio: float = EVENT_MIN_RATIO,
                    prob_event_threshold: float = PROB_EVENT_THRESHOLD) -> dict:
    """
    preds_by_class: {class_idx: [[pred_dict, ...], ...]}
      外層 list = 每部影片；內層 list = 該影片的每個 window。

    Per-class metrics（window 層級指標跨所有影片合併計算）：
      accuracy, top2_accuracy, avg_true_prob, max_true_prob
    Per-class 事件偵測（逐影片判斷，回傳偵測率）：
      event_rate          — 比例門檻：各影片獨立判斷後取平均（偵測成功影片數 / 總影片數）
      prob_event_rate     — 機率門檻：同上
      n_videos, n_videos_detected, n_videos_prob_detected
    """
    n_cls = len(BEHAVIOR_CLASSES)
    all_true, all_pred = [], []
    per_class = {i: {'accuracy': 0.0, 'top2_accuracy': 0.0,
                     'avg_true_prob': 0.0, 'max_true_prob': 0.0,
                     'event_detected': False, 'prob_event_detected': False,
                     'event_rate': 0.0, 'prob_event_rate': 0.0,
                     'n_correct': 0, 'n_prob_hit': 0, 'n_windows': 0,
                     'n_videos': 0, 'n_videos_detected': 0, 'n_videos_prob_detected': 0}
                 for i in range(n_cls)}

    for cls_idx, vid_preds_list in preds_by_class.items():
        # 合併所有影片 windows 用於 window 層級指標
        preds = [p for vid_preds in vid_preds_list for p in vid_preds]
        if not preds:
            continue

        probs      = np.array([p['probs'] for p in preds])
        actual_cls = probs.shape[1]
        pred_ids   = np.clip(np.array([p['pred'] for p in preds], dtype=int),
                             0, actual_cls - 1)
        probs_cls  = probs[:, cls_idx] if cls_idx < actual_cls else np.zeros(len(preds))
        n_correct  = int((pred_ids == cls_idx).sum())
        # NOTE (shake / scratch): Discrete accuracy and avg true-class probability are
        # artificially deflated for impulsive behaviors such as "shake" (head shake),
        # which typically lasts only 0.5–1 s and accounts for roughly 1/10 of the
        # clip duration; the remaining ~90 % of windows contain "stop" and are counted
        # as misclassifications even though the model predicts them correctly.
        # Use max_true_prob and event_detected as the primary validity metrics for
        # these classes — they reflect whether the model fires during the event peak,
        # regardless of how many non-event windows surround it.

        top2_ids = np.argsort(probs, axis=1)[:, -2:]
        n_top2   = int(np.any(top2_ids == cls_idx, axis=1).sum())

        # 逐影片事件偵測
        n_videos = len(vid_preds_list)
        n_vid_detected = 0
        n_vid_prob_detected = 0
        for vid_preds in vid_preds_list:
            if not vid_preds:
                continue
            vp = np.clip(np.array([p['pred'] for p in vid_preds], dtype=int), 0, actual_cls - 1)
            vc = int((vp == cls_idx).sum())
            vn = len(vid_preds)
            if (vc / vn) >= event_min_ratio or vc >= event_min_windows:
                n_vid_detected += 1
            vpc = np.array([p['probs'][cls_idx] for p in vid_preds if cls_idx < len(p['probs'])])
            if len(vpc) > 0 and int((vpc >= prob_event_threshold).sum()) >= event_min_windows:
                n_vid_prob_detected += 1

        event_rate      = n_vid_detected / n_videos if n_videos > 0 else 0.0
        prob_event_rate = n_vid_prob_detected / n_videos if n_videos > 0 else 0.0
        n_prob_hit      = int((probs_cls >= prob_event_threshold).sum())

        per_class[cls_idx] = {
            'accuracy':               float(n_correct / len(preds)),
            'top2_accuracy':          float(n_top2 / len(preds)),
            'avg_true_prob':          float(probs_cls.mean()),
            'max_true_prob':          float(probs_cls.max()),
            'event_detected':         n_vid_detected > 0,
            'prob_event_detected':    n_vid_prob_detected > 0,
            'event_rate':             float(event_rate),
            'prob_event_rate':        float(prob_event_rate),
            'n_correct':              n_correct,
            'n_prob_hit':             n_prob_hit,
            'n_windows':              len(preds),
            'n_videos':               n_videos,
            'n_videos_detected':      n_vid_detected,
            'n_videos_prob_detected': n_vid_prob_detected,
        }
        all_true.extend([cls_idx] * len(preds))
        all_pred.extend(pred_ids.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)

    evaluated       = [i for i in range(n_cls) if per_class[i]['n_windows'] > 0]
    event_rate      = float(np.mean([per_class[i]['event_rate']      for i in evaluated])) if evaluated else 0.0
    prob_event_rate = float(np.mean([per_class[i]['prob_event_rate'] for i in evaluated])) if evaluated else 0.0
    top2_acc        = float(np.mean([per_class[i]['top2_accuracy']   for i in evaluated])) if evaluated else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'accuracy':                 float(accuracy_score(y_true, y_pred)),
            'top2_accuracy':            top2_acc,
            'macro_f1':                 float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'event_detection_rate':     event_rate,
            'prob_event_detection_rate': prob_event_rate,
        },
        'confusion_matrix':      confusion_matrix(y_true, y_pred, labels=list(range(n_cls))),
        'event_min_windows':     event_min_windows,
        'event_min_ratio':       event_min_ratio,
        'prob_event_threshold':  prob_event_threshold,
    }


# ═══════════════════════════════════════════════════════════════════════════
# McNemar's test（同一批 window 上的配對顯著性檢定，回答「這次的準確率差異
# 是不是真的顯著，還是雜訊」——見與使用者討論的驗證方法 1）
# ═══════════════════════════════════════════════════════════════════════════
def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    對兩個模型在同一批（配對）window 上的正確與否做 McNemar's test。
    b = A對B錯的 window 數，c = A錯B對的 window 數；只有 b、c 這兩格不一致的
    windows 才有資訊量，兩者都對/都錯的 windows 不影響檢定。

    n = b + c < 25 時用精確二項檢定（exact binomial test，p=0.5），
    n >= 25 時用連續性校正卡方近似（standard practice，small-sample chi2 校正不可靠）。
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(correct_a & ~correct_b))   # A 對、B 錯
    c = int(np.sum(~correct_a & correct_b))   # A 錯、B 對
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'n_discordant': n, 'statistic': 0.0,
                'p_value': 1.0, 'method': 'n/a（兩模型完全一致，無法檢定）'}
    if n < 25:
        res = binomtest(min(b, c), n, 0.5, alternative='two-sided')
        return {'b': b, 'c': c, 'n_discordant': n, 'statistic': float(min(b, c)),
                'p_value': float(res.pvalue), 'method': 'exact_binomial'}
    stat = (abs(b - c) - 1) ** 2 / n   # continuity-corrected chi-square, df=1
    p = float(1 - chi2.cdf(stat, df=1))
    return {'b': b, 'c': c, 'n_discordant': n, 'statistic': float(stat),
            'p_value': p, 'method': 'chi2_corrected'}


def _pairwise_align_by_frame(vid_preds_a: list, vid_preds_b: list) -> tuple:
    """把同一部影片、兩個模型的 window 預測依 'frame' 對齊（因不同模型可能用
    不同 seq_len/EMA，window 數量不一定完全一致），回傳交集後的兩份 pred 清單。"""
    by_frame_a = {p['frame']: p for p in vid_preds_a}
    by_frame_b = {p['frame']: p for p in vid_preds_b}
    common = sorted(set(by_frame_a) & set(by_frame_b))
    return [by_frame_a[f] for f in common], [by_frame_b[f] for f in common]


def compute_mcnemar_pairs(preds_list: list, evaluated_cls: list) -> dict:
    """
    對 N 個模型的每一對做 McNemar's test（overall + per-class）。
    preds_list[model_idx] = {cls_idx: [[window_dict, ...] per video]}（跟 main() 裡
    preds_list 同一份資料，未展平）。假設同一 class 下所有模型走的是同一批、
    同順序的影片（main() 對每個模型都用同一個 dir_path 資料夾排序過的影片清單）。

    Returns: {(i, j): {'overall': mcnemar_result, 'per_class': {cls_idx: mcnemar_result}}}
    """
    n_models = len(preds_list)
    results = {}
    for i in range(n_models):
        for j in range(i + 1, n_models):
            per_class_res = {}
            overall_a, overall_b = [], []
            for cls_idx in evaluated_cls:
                vids_a = preds_list[i].get(cls_idx, [])
                vids_b = preds_list[j].get(cls_idx, [])
                cls_a, cls_b = [], []
                for va, vb in zip(vids_a, vids_b):
                    pa, pb = _pairwise_align_by_frame(va, vb)
                    cls_a.extend(int(p['pred']) == cls_idx for p in pa)
                    cls_b.extend(int(p['pred']) == cls_idx for p in pb)
                cls_a_arr = np.array(cls_a, dtype=bool)
                cls_b_arr = np.array(cls_b, dtype=bool)
                per_class_res[cls_idx] = mcnemar_test(cls_a_arr, cls_b_arr)
                overall_a.extend(cls_a)
                overall_b.extend(cls_b)
            results[(i, j)] = {
                'overall': mcnemar_test(np.array(overall_a, dtype=bool),
                                        np.array(overall_b, dtype=bool)),
                'per_class': per_class_res,
            }
    return results


def print_mcnemar_summary(mcnemar_results: dict, names: list, classes: list,
                          evaluated_cls: list, alpha: float = 0.05) -> list:
    """把 compute_mcnemar_pairs() 的結果印成人類可讀的顯著性檢定報告。
    回傳 lines list，方便同時印到終端與寫進檔案。"""
    SEP = '─' * 70
    lines = [
        '',
        '● McNemar\'s Test  (配對顯著性檢定：這次的準確率差異是否顯著，還是雜訊)',
        f'  {SEP}',
        '  p < 0.05 視為顯著差異（★）；n_discordant 是兩模型「答案不同」的 window 數，',
        '  只有這些 window 有資訊量，n_discordant 太小（<10）時檢定力不足，結果僅供參考。',
        '',
    ]
    for (i, j), res in mcnemar_results.items():
        ov = res['overall']
        sig = '★ 顯著' if ov['p_value'] < alpha else '  不顯著'
        winner = names[i] if ov['b'] > ov['c'] else (names[j] if ov['c'] > ov['b'] else '=')
        lines.append(f"  {names[i]}  vs  {names[j]}")
        lines.append(f"    Overall: p={ov['p_value']:.4f}  {sig}  "
                     f"(n_discordant={ov['n_discordant']}, {names[i]}對B錯={ov['b']}, "
                     f"{names[j]}對A錯={ov['c']}, method={ov['method']}) → 較優: {winner}")
        for cls_idx in evaluated_cls:
            cr = res['per_class'][cls_idx]
            csig = '★' if cr['p_value'] < alpha else ' '
            lines.append(f"      [{classes[cls_idx]:<8}] p={cr['p_value']:.4f} {csig}  "
                         f"(n_discordant={cr['n_discordant']}, b={cr['b']}, c={cr['c']}, {cr['method']})")
        lines.append('')
    print('\n'.join(lines))
    return lines


def save_mcnemar_csv(mcnemar_results: dict, names: list, classes: list,
                     evaluated_cls: list, out_path: Path):
    """把 McNemar 檢定結果存成 CSV，方便論文附錄引用。"""
    rows = [['pair', 'scope', 'b', 'c', 'n_discordant', 'statistic', 'p_value', 'method', 'significant(p<0.05)']]
    for (i, j), res in mcnemar_results.items():
        pair = f'{names[i]}_vs_{names[j]}'
        ov = res['overall']
        rows.append([pair, 'overall', ov['b'], ov['c'], ov['n_discordant'],
                    f"{ov['statistic']:.4f}", f"{ov['p_value']:.4f}", ov['method'],
                    'YES' if ov['p_value'] < 0.05 else ''])
        for cls_idx in evaluated_cls:
            cr = res['per_class'][cls_idx]
            rows.append([pair, classes[cls_idx], cr['b'], cr['c'], cr['n_discordant'],
                        f"{cr['statistic']:.4f}", f"{cr['p_value']:.4f}", cr['method'],
                        'YES' if cr['p_value'] < 0.05 else ''])
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-video accuracy breakdown（診斷用：找出某類別是不是被少數幾支影片拖累）
# ═══════════════════════════════════════════════════════════════════════════
def compute_per_video_accuracy(preds_list: list, video_names: dict, evaluated_cls: list,
                               event_min_windows: int = EVENT_MIN_WINDOWS,
                               event_min_ratio: float = EVENT_MIN_RATIO) -> dict:
    """
    對每個類別、每支影片、每個模型算 discrete accuracy（pred == cls_idx 的比例），
    以及 max_true_prob（該影片所有 window 裡，true-class 機率的最高值）跟
    event_detected（影片層級偵測：正確 window 比例 ≥ event_min_ratio 或
    數量 ≥ event_min_windows，邏輯跟 compute_metrics 的事件偵測一致）。

    discrete accuracy 是用「整支影片＝該類別」當 ground truth 算出來的，如果該
    行為在影片裡只佔一小段（例如 scratch/shake 這種瞬發行為，其餘畫面是趨近
    /其他行為），accuracy 會被結構性拉低，即使模型在真正發生的那幾秒表現正常。
    max_true_prob／event_detected 是判斷「這支影片是不是真的評不出模型能力，
    而不是模型真的很差」的關鍵佐證——如果 accuracy 很低但 max_true_prob 很高、
    event_detected=True，代表模型有抓到事件峰值，只是被同一支影片裡大量非
    事件畫面的 window 稀釋了平均準確率。
    影片依 model 0（baseline）的準確率由低到高排序，方便一眼看出哪幾支影片特別差。

    Returns: {cls_idx: [{'video', 'accuracy': [...], 'n_windows': [...],
                         'max_true_prob': [...], 'event_detected': [...]}]}
    """
    n_models = len(preds_list)
    result = {}
    for cls_idx in evaluated_cls:
        names_for_cls = video_names.get(cls_idx, [])
        rows = []
        for vi, vid_name in enumerate(names_for_cls):
            accs, n_windows, max_probs, event_flags = [], [], [], []
            for mi in range(n_models):
                vids = preds_list[mi].get(cls_idx, [])
                if vi >= len(vids) or not vids[vi]:
                    accs.append(None)
                    n_windows.append(0)
                    max_probs.append(None)
                    event_flags.append(None)
                    continue
                preds = vids[vi]
                n_correct = sum(1 for p in preds if int(p['pred']) == cls_idx)
                n = len(preds)
                accs.append(n_correct / n)
                n_windows.append(n)
                probs_cls = [p['probs'][cls_idx] for p in preds if cls_idx < len(p['probs'])]
                max_probs.append(max(probs_cls) if probs_cls else 0.0)
                event_flags.append((n_correct / n) >= event_min_ratio or n_correct >= event_min_windows)
            rows.append({'video': vid_name, 'accuracy': accs, 'n_windows': n_windows,
                        'max_true_prob': max_probs, 'event_detected': event_flags})
        rows.sort(key=lambda r: r['accuracy'][0] if r['accuracy'][0] is not None else 1.0)
        result[cls_idx] = rows
    return result


def print_per_video_accuracy(per_video: dict, names: list, classes: list,
                             low_flag: float = 0.15) -> list:
    """印出逐影片準確率，標記明顯低於同類別平均（以 baseline/names[0] 為準）的影片——
    這些影片是拉低該類別整體準確率的主要來源，值得優先檢查（是否標註邊界模糊、
    畫面角度特殊、或跟訓練資料的場景差異特別大）。同時列出 max_true_prob／
    event_detected：如果一支影片 accuracy 很低但這兩項顯示模型仍抓到事件峰值，
    代表低準確率主要是「事件在影片裡佔比太短、被大量非事件 window 稀釋」的
    資料特性，不是模型真的辨識失敗。"""
    SEP = '─' * 70
    lines = ['', '● Per-Video Accuracy Breakdown  (診斷用：找出特定類別裡表現特別差的影片)',
            f'  {SEP}']
    for cls_idx, rows in per_video.items():
        if not rows:
            continue
        cls = classes[cls_idx]
        valid = [r['accuracy'][0] for r in rows if r['accuracy'][0] is not None]
        mean_acc = sum(valid) / len(valid) if valid else 0.0
        lines.append(f'  [{cls}]  baseline({names[0]}) 影片平均準確率 = {mean_acc:.1%}')
        for r in rows:
            cells = []
            for mi, acc in enumerate(r['accuracy']):
                if acc is None:
                    cells.append(f'{names[mi]}: n/a')
                else:
                    mtp = r['max_true_prob'][mi]
                    ev = r['event_detected'][mi]
                    ev_s = '✓' if ev else '✗'
                    cells.append(f'{names[mi]}: {acc:.1%}(n={r["n_windows"][mi]}, '
                                f'max_prob={mtp:.0%}, event={ev_s})')
            a0 = r['accuracy'][0]
            flag = '  ⚠ 遠低於該類別平均' if (a0 is not None and mean_acc - a0 >= low_flag) else ''
            if flag and a0 is not None and r['max_true_prob'][0] is not None and r['max_true_prob'][0] >= 0.7:
                flag += '（但 max_prob 高，可能只是事件片段太短被稀釋，非模型缺陷）'
            lines.append(f'    {r["video"]:<28}  ' + '   '.join(cells) + flag)
        lines.append('')
    print('\n'.join(lines))
    return lines


def save_per_video_accuracy_csv(per_video: dict, names: list, classes: list, out_path: Path):
    """把逐影片準確率存成 CSV。"""
    header = (['class', 'video'] + [f'{n}_accuracy' for n in names]
             + [f'{n}_n_windows' for n in names]
             + [f'{n}_max_true_prob' for n in names]
             + [f'{n}_event_detected' for n in names])
    rows = [header]
    for cls_idx, cls_rows in per_video.items():
        cls = classes[cls_idx]
        for r in cls_rows:
            acc_cells = [f"{a:.4f}" if a is not None else '' for a in r['accuracy']]
            prob_cells = [f"{p:.4f}" if p is not None else '' for p in r['max_true_prob']]
            event_cells = ['' if e is None else ('YES' if e else '') for e in r['event_detected']]
            rows.append([cls, r['video']] + acc_cells + list(r['n_windows']) + prob_cells + event_cells)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations（泛化成 N 個模型，N = 2~5）
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_comparison(metrics_list, names, classes, out_path):
    """
    三列 grouped bar chart，每列 N 根長條（一模型一色）：
      Row 0 — Discrete Accuracy         (硬指標：argmax 是否正確) + ✓/✗ 事件偵測標記
      Row 1 — Avg True-Class Prob       (軟指標：平均信心)
      Row 2 — Max True-Class Prob       (峰值信號：整部影片最高信心，反映事件偵測能力)
    每個 x 位置用 ★ 標出當前最高值的模型（可並列多個 ★）。
    """
    n_models = len(metrics_list)
    n_cls    = len(classes)
    x_labels = classes + ['Overall']
    x        = np.arange(len(x_labels))
    w        = min(0.8 / n_models, 0.28)
    ev_min   = metrics_list[0].get('event_min_windows', EVENT_MIN_WINDOWS)
    colors   = _PALETTE[:n_models]

    def _vals(metrics, key):
        pc  = metrics['per_class']
        out = [pc[i].get(key, 0.0) for i in range(n_cls)]
        if key == 'accuracy':
            out.append(metrics['overall']['accuracy'])
        else:
            out.append(float(np.mean([pc[i].get(key, 0.0) for i in range(n_cls)])))
        return out

    def _evt(metrics):
        pc  = metrics['per_class']
        out = [pc[i].get('event_detected', False) for i in range(n_cls)]
        out.append(all(pc[i].get('event_detected', False)
                       for i in range(n_cls) if pc[i].get('n_windows', 0) > 0))
        return out

    accs  = [_vals(m, 'accuracy')      for m in metrics_list]
    probs = [_vals(m, 'avg_true_prob') for m in metrics_list]
    maxes = [_vals(m, 'max_true_prob') for m in metrics_list]
    evts  = [_evt(m)                   for m in metrics_list]

    fig, axes = plt.subplots(3, 1, figsize=(max(11, 2.4 * n_models + 4), 13), constrained_layout=True)
    fig.suptitle(
        'ST-GCN Model Comparison\n' +
        '   |   '.join(f'{chr(65 + i)}: {n}' for i, n in enumerate(names)),
        fontsize=12, fontweight='bold'
    )

    def _draw(ax, vals_list, title, ylabel, evt_list=None):
        offsets = (np.arange(n_models) - (n_models - 1) / 2) * w
        bars_list = []
        for mi, vals in enumerate(vals_list):
            bars = ax.bar(x + offsets[mi], vals, w, label=f'{chr(65 + mi)}: {names[mi]}',
                          color=colors[mi], alpha=0.88)
            bars_list.append(bars)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylim(0, 1.30)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8, loc='upper right', ncol=min(n_models, 3))
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        for bars in bars_list:
            for bar in bars:
                h = bar.get_height()
                if h > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                            f'{h:.0%}', ha='center', va='bottom', fontsize=7)

        # 每個 x 位置標出最高值的模型（可並列）
        for i in range(len(x_labels)):
            vs = [vals_list[mi][i] for mi in range(n_models)]
            best_v = max(vs)
            for mi, v in enumerate(vs):
                if abs(v - best_v) < 0.005:
                    ax.text(x[i] + offsets[mi], v + 0.045, '★',
                            ha='center', va='bottom', fontsize=10, color=colors[mi])

        if evt_list is not None:
            for mi, evt in enumerate(evt_list):
                for i, det in enumerate(evt):
                    sym, col = ('✓', colors[mi]) if det else ('✗', '#bbb')
                    ax.text(x[i] + offsets[mi], 1.20, sym, ha='center', va='bottom',
                            fontsize=9, color=col, fontweight='bold')

    _draw(axes[0], accs,
          f'Discrete Accuracy  (argmax == true label)\n'
          f'✓/✗ = event detected (≥{ev_min} correct windows per clip)',
          'Accuracy', evt_list=evts)
    _draw(axes[1], probs,
          'Avg True-Class Probability  (model conviction)',
          'Avg Prob')
    _draw(axes[2], maxes,
          'Max True-Class Probability  (peak event signal)',
          'Max Prob')

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_confusion_matrices(cms, names, classes, out_path, joint_weight_strs=None):
    """N 個 confusion matrix 並排，支援 seaborn（可選）。

    joint_weight_strs: 若提供（每個模型一個字串，見 get_joint_prior_weights_str()），
    會在 suptitle 正下方、熱力圖正上方保留一塊完全空白的區域，由上到下依序列出
    每個模型的 Joint Prior Weights 設定，置中對齊——這塊區域是額外預留的圖高，
    不是疊在既有的熱力圖/座標軸/colorbar 上面，不會遮擋任何原本的內容。
    """
    n_models = len(cms)
    try:
        import seaborn as sns
        _sns = True
    except ImportError:
        _sns = False

    # 用絕對英吋精確算出每一段區域要多高，由上到下依序是：
    # 頂部留白 → suptitle → (權重文字區塊) → 跟熱力圖標題之間的間隔 → 熱力圖自己
    # 的雙行標題（model 名稱 + Avg Recall）→ 熱力圖本體 → 底部（x 軸刻度/標籤）。
    # 全部用固定英吋數堆疊，而非用比例反推，確保權重文字區塊有絕對足夠、不會
    # 被 ax.set_title() 的雙行標題擠壓到重疊的獨立空間。
    n_weight_lines       = n_models if joint_weight_strs else 0
    top_margin_in        = 0.10
    suptitle_h_in        = 0.42
    gap_after_suptitle_in = 0.08
    weight_line_h_in     = 0.24
    gap_before_title_in  = 0.14
    subplot_title_h_in   = 0.66   # 雙行標題（名稱 + Avg Recall）的預留高度
    plot_area_h_in       = 4.3
    bottom_margin_in     = 0.75

    above_axes_in = (top_margin_in + suptitle_h_in + gap_after_suptitle_in
                    + n_weight_lines * weight_line_h_in
                    + (gap_before_title_in if n_weight_lines else 0.05)
                    + subplot_title_h_in)
    fig_h = above_axes_in + plot_area_h_in + bottom_margin_in

    fig, axes = plt.subplots(1, n_models, figsize=(6.2 * n_models, fig_h))
    if n_models == 1:
        axes = [axes]

    suptitle_y = 1.0 - (top_margin_in + suptitle_h_in * 0.5) / fig_h
    fig.suptitle('Confusion Matrices', fontsize=13, fontweight='bold', y=suptitle_y)

    if joint_weight_strs:
        block_top_y = 1.0 - (top_margin_in + suptitle_h_in + gap_after_suptitle_in) / fig_h
        line_h_frac = weight_line_h_in / fig_h
        for i, (name, wstr) in enumerate(zip(names, joint_weight_strs)):
            y = block_top_y - i * line_h_frac
            fig.text(0.5, y, f'{name}: {wstr}', ha='center', va='top',
                     fontsize=9, color=_PALETTE[i % len(_PALETTE)])

    top_for_axes = 1.0 - above_axes_in / fig_h
    bottom_for_axes = bottom_margin_in / fig_h
    fig.subplots_adjust(top=top_for_axes, bottom=bottom_for_axes, left=0.055, right=0.98, wspace=0.4)

    for ax, cm, name, col in zip(axes, cms, names, _PALETTE[:n_models]):
        if _sns:
            # Normalize to show row-wise recall
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes,
                        ax=ax, cbar=True, linewidths=0.5, vmin=0, vmax=1)
        else:
            ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=30, ha='right')
            ax.set_yticklabels(classes)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black',
                            fontsize=12)

        # Per-class recall on diagonal annotation
        diag_recall = [cm[i, i] / max(cm[i].sum(), 1) for i in range(len(classes))]
        avg_recall = float(np.mean(diag_recall))

        ax.set_title(f'{name}\nAvg Recall = {avg_recall:.1%}', fontsize=11,
                     fontweight='bold', color=col)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_prob_histograms(preds_list, names, classes, out_path):
    """
    True-class probability histogram（每個類別一列，N 個模型並排）
    橫軸：true-class prob 0~1，縱軸：window 數量。
    可立即看出 Softmax collapse（機率全堆在同一處）或分布健康與否。
    """
    n_models = len(preds_list)
    n_cls = len(classes)
    fig, axes = plt.subplots(n_cls, n_models, figsize=(5.2 * n_models, 2.8 * n_cls), constrained_layout=True)
    fig.suptitle('True-Class Probability Histogram\n(each row = one behavior class)',
                 fontsize=12, fontweight='bold')

    bins = np.linspace(0, 1, 21)
    for i, cls in enumerate(classes):
        for j in range(n_models):
            preds, name, col = preds_list[j], names[j], _PALETTE[j]
            ax = axes[i][j]
            if i not in preds or not preds[i]:
                ax.set_visible(False)
                continue
            probs_cls = np.array([p['probs'][i] for p in preds[i]
                                  if i < len(p['probs'])])
            ax.hist(probs_cls, bins=bins, color=col, alpha=0.85, edgecolor='white')
            ax.axvline(float(probs_cls.mean()), color='k', linestyle='--',
                       linewidth=1.2, label=f'mean={probs_cls.mean():.2f}')
            ax.set_xlim(0, 1)
            ax.set_title(f'[{cls}]  {name}', fontsize=9, fontweight='bold')
            ax.set_xlabel('True-class probability', fontsize=8)
            ax.set_ylabel('Windows', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# CSV outputs（泛化成 N 個模型）
# ═══════════════════════════════════════════════════════════════════════════
def save_summary_csv(metrics_list, names, classes, out_path):
    """對比摘要 CSV：每行一個指標，N 個模型各一欄，含最佳欄位。primary 欄位標示指定的
    主指標（accuracy，argmax 正確率），其餘指標僅供參考。"""
    rows = [['metric'] + names + ['best', 'delta', 'primary']]

    def _row(label, vals, primary=False):
        best_idx = int(np.argmax(vals))
        delta = max(vals) - min(vals)
        best_name = '=' if delta < 0.001 else names[best_idx]
        rows.append([label] + [f'{v:.4f}' for v in vals] +
                    [best_name, f'{delta:.4f}', 'YES' if primary else ''])

    _row('overall_accuracy',             [m['overall']['accuracy']             for m in metrics_list], primary=True)
    _row('overall_top2_accuracy',        [m['overall']['top2_accuracy']        for m in metrics_list])
    _row('overall_macro_f1',             [m['overall']['macro_f1']             for m in metrics_list])
    _row('overall_event_det_rate_ratio', [m['overall']['event_detection_rate'] for m in metrics_list])
    _row('overall_event_det_rate_prob',  [m['overall']['prob_event_detection_rate'] for m in metrics_list])
    rows.append([])   # blank separator

    for i, cls in enumerate(classes):
        pcs = [m['per_class'][i] for m in metrics_list]
        _row(f'{cls}_accuracy',        [p['accuracy']        for p in pcs], primary=True)
        _row(f'{cls}_event_rate',      [p['event_rate']      for p in pcs])
        _row(f'{cls}_top2_accuracy',   [p['top2_accuracy']   for p in pcs])
        _row(f'{cls}_avg_true_prob',   [p['avg_true_prob']   for p in pcs])
        _row(f'{cls}_max_true_prob',   [p['max_true_prob']   for p in pcs])
        _row(f'{cls}_prob_event_rate', [p['prob_event_rate'] for p in pcs])
        rows.append([f'{cls}_n_videos_detected'] +
                    [f"{p['n_videos_detected']}/{p['n_videos']}" for p in pcs] + ['-', '-'])
        rows.append([f'{cls}_prob_videos_detected'] +
                    [f"{p['n_videos_prob_detected']}/{p['n_videos']}" for p in pcs] + ['-', '-'])
        rows.append([f'{cls}_n_correct'] + [p['n_correct'] for p in pcs] + ['-', '-'])
        rows.append([f'{cls}_n_windows'] + [p['n_windows'] for p in pcs] + ['-', '-'])
        rows.append([])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"  ✓ {out_path.name}")


def save_preds_csv(preds_by_class: dict, model_name: str, classes, out_dir: Path):
    """儲存每類別的逐窗口預測結果（單一模型，跟模型數量無關）。"""
    for cls_idx, preds in preds_by_class.items():
        cls_name = classes[cls_idx]
        out = out_dir / f"{model_name}_{cls_name}_preds.csv"
        with out.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'time', 'pred_class', 'conf'] +
                       [f'prob_{c}' for c in classes])
            for p in preds:
                w.writerow([p['frame'], p['time'],
                            classes[p['pred']] if 0 <= p['pred'] < len(classes) else 'none',
                            f"{p['conf']:.4f}"] +
                           [f"{x:.4f}" for x in p['probs']])


# ═══════════════════════════════════════════════════════════════════════════
# Composite scoring（泛化成 N 個模型：每個指標 winner-take-all，差距 < 門檻視為並列）
# ═══════════════════════════════════════════════════════════════════════════
def _score_models(metrics_list, names) -> tuple:
    """
    多指標加權評分，衡量 N 個模型的綜合強弱（僅供參考，不是主指標）。

    每個指標最高分者得 W 分，差距 < 0.2% 視為並列（平分該指標權重）。
    邏輯延續原本兩模型版本的 Borda 計分，泛化到 N 個模型時改成
    winner-take-all（而非完整排名給分），維持原本「每個指標只獎勵最強者」的精神。

    加權設計依據文獻慣例：
      Macro F1 / Min-class acc 各 W=2（最重要，行為辨識首選 + 木桶原則）
      其餘各 W=1（輔助判斷）
      Avg True-Class Prob 為 W=0（只顯示、不計分——見下方該指標定義處的說明：
      這個數值會被各模型訓練時的 LABEL_SMOOTHING 系統性壓低，label_smoothing
      不同的模型之間比較本身不公平，不該讓它影響排名）

    參考：
      Macro F1 作為行為辨識主指標 —— NTU RGB+D, Kinetics benchmark 慣例
      Min-class accuracy (木桶原則) —— 部署可靠性評估
      Avg true-class prob            —— 對應 ECE 概念, Guo et al. ICML 2017（僅供參考，見上）
      Borda count 多指標排名          —— Kittler et al. 1998

    Returns:
        (scores: list[float], breakdown: list[dict])
    """
    n_cls     = len(BEHAVIOR_CLASSES)
    n_models  = len(metrics_list)
    per_list  = [m['per_class'] for m in metrics_list]
    evaluated = [i for i in range(n_cls) if per_list[0][i]['n_windows'] > 0]
    ev_min    = metrics_list[0].get('event_min_windows', EVENT_MIN_WINDOWS)

    candidate_metrics = []

    # ── 1. Overall Accuracy  (W=1) ─────────────────────────────────────────
    candidate_metrics.append(dict(
        label='Overall Accuracy', weight=1, higher_better=True,
        note='argmax 正確率，直觀但不考慮類別平衡',
        values=[m['overall']['accuracy'] for m in metrics_list],
    ))

    # ── 2. Macro F1  (W=2) ─────────────────────────────────────────────────
    candidate_metrics.append(dict(
        label='Macro F1', weight=2, higher_better=True,
        note='類別平均 F1，行為辨識首選指標，兼顧 Precision + Recall',
        values=[m['overall']['macro_f1'] for m in metrics_list],
    ))

    # ── 3. Avg True-Class Probability  (W=0，僅供參考，不計入總分) ───────────
    # 不計分是刻意的：這個數值會被訓練時的 LABEL_SMOOTHING 直接壓低（smoothing
    # 越高，即使預測完全正確，softmax 也被訓練成不能輸出接近 1.0），跟模型真正
    # 的辨識能力/校準品質無關。這裡比較的 5 個模型 label_smoothing 彼此不同
    # （0 / 0.01 / 0.05 / 0.1），這項指標會系統性偏袒 label_smoothing 較低的
    # 模型，不是公平比較，所以仍顯示在明細表供參考，但權重設為 0、不計入排名。
    if evaluated:
        candidate_metrics.append(dict(
            label='Avg True-Class Prob', weight=0, higher_better=True,
            note='[參考，不計分] 受訓練時 label_smoothing 影響，label_smoothing 不同的模型之間比較不公平',
            values=[float(np.mean([per[i]['avg_true_prob'] for i in evaluated])) for per in per_list],
        ))

    # ── 4. Min Per-Class Accuracy  (W=2) ───────────────────────────────────
    if evaluated:
        candidate_metrics.append(dict(
            label='Min Class Accuracy', weight=2, higher_better=True,
            note='最差類別準確率，衡量部署穩健性（木桶原則，最脆弱的一環）',
            values=[float(min(per[i]['accuracy'] for i in evaluated)) for per in per_list],
        ))

    # ── 5. Std Per-Class Accuracy  (W=1, lower is better) ──────────────────
    if len(evaluated) > 1:
        candidate_metrics.append(dict(
            label='Std Class Accuracy', weight=1, higher_better=False,
            note='各類準確率標準差，越小代表各類表現越均衡（↓ lower is better）',
            values=[float(np.std([per[i]['accuracy'] for i in evaluated])) for per in per_list],
        ))

    # ── 6. Event Detection Rate  (W=2) ─────────────────────────────────────
    candidate_metrics.append(dict(
        label='Event Detection Rate', weight=2, higher_better=True,
        note=f'影片層級偵測率，≥{ev_min} windows 正確即算偵測成功（對含靜止段的影片最公平）',
        values=[m['overall']['event_detection_rate'] for m in metrics_list],
    ))

    TIE_THRESHOLD = 0.002   # 差距 < 0.2% 視為並列
    scores = [0.0] * n_models
    breakdown = []

    for m in candidate_metrics:
        vals, w = m['values'], m['weight']
        best_val = max(vals) if m['higher_better'] else min(vals)
        winners  = [i for i, v in enumerate(vals) if abs(v - best_val) < TIE_THRESHOLD]
        pts = [0.0] * n_models
        share = w / len(winners)
        for i in winners:
            pts[i] = share
            scores[i] += share
        result = '=' if len(winners) > 1 else names[winners[0]]
        breakdown.append({**m, 'pts': pts, 'result': result})

    return scores, breakdown


# ═══════════════════════════════════════════════════════════════════════════
# Final summary（泛化成 N 個模型）
# ═══════════════════════════════════════════════════════════════════════════
def print_final_summary(metrics_list, names):
    """執行結束後在終端列印人類可讀的對比分析，支援 2~5 個模型。"""
    classes  = BEHAVIOR_CLASSES
    n_cls    = len(classes)
    n_models = len(metrics_list)
    per_list = [m['per_class'] for m in metrics_list]

    col_w = max(max(len(n) for n in names), 10)
    BOX  = '═' * max(62, 14 + (col_w + 2) * n_models)
    SEP  = '─' * len(BOX)
    NL   = ''

    def _hdr_row(label_w=10):
        return f'  {"":<{label_w}}  ' + '  '.join(f'{n:>{col_w}}' for n in names)

    def _val_row(label, vals, fmt='{:.1%}', label_w=10, higher_better=True):
        best = max(vals) if higher_better else min(vals)
        cells = []
        for v in vals:
            s = fmt.format(v)
            cells.append(('*' + s) if abs(v - best) < 0.005 else (' ' + s))
        return f'  {label:<{label_w}}  ' + '  '.join(f'{c:>{col_w}}' for c in cells)

    def _cls_label(cls):
        """類別名稱加 † 標記（若屬於 PARTIAL_COVERAGE_CLASSES，見常數定義處說明）。"""
        return f'{cls} †' if cls in PARTIAL_COVERAGE_CLASSES else cls

    def _delta_row(label, base_val, rest_vals, label_w=10):
        """Δ vs baseline（names[0]）：只顯示 names[1:] 相對 baseline 的變化。"""
        cells = []
        for v in rest_vals:
            d = v - base_val
            s = f'{d:+.1%}'
            cells.append(('▲' + s) if d > 0.005 else (('▼' + s) if d < -0.005 else (' ' + s)))
        return f'  {label:<{label_w}}  ' + '  '.join(f'{c:>{col_w}}' for c in cells)

    # ── 主指標：Accuracy（argmax 正確率，獨立於下方複合分數）──────────────────
    accs = [m['overall']['accuracy'] for m in metrics_list]
    primary_best     = max(accs)
    primary_winners  = [i for i, v in enumerate(accs) if abs(v - primary_best) < 0.005]
    primary_winner   = '=' if len(primary_winners) > 1 else names[primary_winners[0]]

    lines = [
        NL,
        f'╔{BOX}╗',
        f'║{f"  FINAL COMPARISON SUMMARY  ({n_models} models)":^{len(BOX)}}║',
        f'╚{BOX}╝',
        NL,
        '★★★ 主指標：Accuracy  (argmax == true label) ★★★',
        _hdr_row(),
        f'  {SEP}',
    ]
    flagged_evaluated = []
    for i, cls in enumerate(classes):
        pcs = [per[i] for per in per_list]
        if all(p['n_windows'] == 0 for p in pcs):
            continue
        if cls in PARTIAL_COVERAGE_CLASSES:
            flagged_evaluated.append(cls)
        vals = [p['accuracy'] for p in pcs]
        lines.append(_val_row(_cls_label(cls), vals))
    lines += [
        f'  {SEP}',
        _val_row('Overall', accs),
        NL,
    ]
    if primary_winner == '=':
        lines.append(f'  ★ 主指標結論：{len(primary_winners)} 個模型準確率並列最高')
    else:
        lines.append(f'  ★ 主指標結論：{primary_winner} 準確率最高 — 依你指定的主指標，優先選它')

    if flagged_evaluated:
        lines += [
            NL,
            f"  † {'/'.join(flagged_evaluated)}：影片中該行為只出現一小段（其餘畫面多為 stop 或其他行為），"
            f"絕對準確率因此結構性偏低，這是資料特性、不是模型缺陷。",
            f"  比較多個模型時，這些類別請優先看下方「Δ vs baseline」的相對變化，而非本表的絕對數字。",
        ]

    # ── [參考，非主指標] Overall Performance  (Event Detection Rate / Macro F1) ──
    ev_min = metrics_list[0].get('event_min_windows', EVENT_MIN_WINDOWS)
    edrs   = [m['overall']['event_detection_rate'] for m in metrics_list]
    f1s    = [m['overall']['macro_f1'] for m in metrics_list]
    lines += [
        NL,
        f'● [參考，非主指標] Overall Performance  (Event Detection Rate ratio≥{EVENT_MIN_RATIO:.0%} '
        f'or ≥{ev_min} windows / Macro F1)',
        _hdr_row(14),
        f'  {SEP}',
        _val_row('Event Rate', edrs, label_w=14),
        _val_row('Macro F1',   f1s,  label_w=14),
        NL,
    ]

    # ── [參考] 加權複合分數（winner-take-all，混入了你不特別在意的指標）──
    scores, score_breakdown = _score_models(metrics_list, names)
    max_score = sum(m['weight'] for m in score_breakdown)
    best_score = max(scores)
    composite_winners = [i for i, s in enumerate(scores) if abs(s - best_score) < 1e-9]
    composite_winner  = '=' if len(composite_winners) > 1 else names[composite_winners[0]]
    score_str = '  '.join(f'{names[i]}={s:.1f}' for i, s in enumerate(scores))
    lines += [f'  [參考] Composite scores: {score_str}  (/ {max_score:.0f} pts)  → {composite_winner}', NL]

    # ── Per-class accuracy breakdown ──
    lines += ['● Per-Class Accuracy  (argmax == true label)', f'  {SEP}']
    class_wins = [0.0] * n_models
    class_ranges = []
    for i, cls in enumerate(classes):
        vals = [per[i]['accuracy'] for per in per_list]
        best = max(vals)
        winners = [j for j, v in enumerate(vals) if abs(v - best) < 0.005]
        for j in winners:
            class_wins[j] += 1.0 / len(winners)
        class_ranges.append((max(vals) - min(vals), cls, vals))
        lines.append(_val_row(_cls_label(cls), vals))
    lines.append(NL)
    n_evaluated = sum(1 for i in range(n_cls) if per_list[0][i]['n_windows'] > 0)
    wins_str = '   '.join(f'{names[j]}: {class_wins[j]:.1f}/{n_evaluated}' for j in range(n_models))
    lines.append(f'  Class wins → {wins_str}')
    lines.append(NL)

    # ── Per-Class Accuracy Δ vs baseline（names[0]）──────────────────────────
    # 判斷「有沒有改善」用這張表：對 PARTIAL_COVERAGE_CLASSES（見常數定義）而言，
    # 絕對準確率結構性偏低，Δ（相對 baseline 的變化）才是有意義的比較基準。
    if n_models >= 2:
        lines += [
            f'● Per-Class Accuracy Δ vs baseline ({names[0]})  — 判斷「有沒有改善」看這張表',
            f'  {"":<10}  ' + '  '.join(f'{n:>{col_w}}' for n in names[1:]),
            f'  {SEP}',
        ]
        for i, cls in enumerate(classes):
            pcs = [per[i] for per in per_list]
            if all(p['n_windows'] == 0 for p in pcs):
                continue
            vals = [p['accuracy'] for p in pcs]
            lines.append(_delta_row(_cls_label(cls), vals[0], vals[1:]))
        overall_vals = accs
        lines.append(f'  {SEP}')
        lines.append(_delta_row('Overall', overall_vals[0], overall_vals[1:]))
        lines.append(NL)

    # ── Per-class avg true-class probability ──
    lines += ['● Avg True-Class Probability  (model conviction)', f'  {SEP}']
    for i, cls in enumerate(classes):
        vals = [per[i]['avg_true_prob'] for per in per_list]
        lines.append(_val_row(cls, vals))
    lines.append(NL)

    # ── Biggest gap class ──
    class_ranges.sort(reverse=True)
    biggest_gap, biggest_cls, biggest_vals = class_ranges[0]
    gap_winner = names[int(np.argmax(biggest_vals))]
    lines.append(f'● Biggest gap: [{biggest_cls}]  Δ={biggest_gap:.1%}  ({gap_winner} leads)')

    # ── Per-model profile ──
    lines += [NL, '● Per-model profile']
    for j, name in enumerate(names):
        accs_j  = [(per_list[j][i]['accuracy'], classes[i]) for i in range(n_cls)]
        best_j  = max(accs_j)
        worst_j = min(accs_j)
        lines.append(f'  {name:<20}  best={best_j[1]}({best_j[0]:.1%})  worst={worst_j[1]}({worst_j[0]:.1%})')
    lines.append(NL)

    # ── 加權複合計分明細 ──
    lines += ['● [參考] Composite Score  (Weighted, winner-take-all — 非主指標)', f'  {SEP}']
    lines.append(f'  {"Metric":<26}  W  ' + '  '.join(f'{n:>{col_w}}' for n in names) + '   Result')
    for m in score_breakdown:
        dir_hint = '↑' if m['higher_better'] else '↓'
        val_cells = '  '.join(f'{v:>{col_w}.3f}' for v in m['values'])
        lines.append(f'  {m["label"]:<26}  {m["weight"]}  {val_cells}   {dir_hint} → {m["result"]}')
        lines.append(f'    └ {m["note"]}')
    total_cells = '  '.join(f'{s:>{col_w}.1f}' for s in scores)
    lines += [f'  {SEP}',
              f'  {"TOTAL":<26}     {total_cells}',
              NL]

    # ── Recommendation — 以主指標（Accuracy）為準，複合分數僅供參考 ──────────
    lines.append('● Recommendation')
    if primary_winner == '=':
        lines.append('  主指標（準確率）打平；請改看上方 [參考] 複合分數或逐類別表現決定。')
    else:
        lines.append(f'  依主指標（Accuracy），{primary_winner} 較適合部署。')
    if biggest_gap >= 0.10:
        lines.append(f'  Note: [{biggest_cls}] shows a large gap (Δ{biggest_gap:.1%}); '
                     f'prioritize {gap_winner} if this class is critical.')
    if primary_winner != '=' and composite_winner != '=' and primary_winner != composite_winner:
        lines.append(f'  ⚠ 注意：[參考] 複合分數建議 {composite_winner}，但主指標（準確率）建議 {primary_winner}'
                     f'——兩者不一致時請以主指標為準（複合分數混入了 F1/Event Detection Rate 等你不特別在意的項目）。')
    lines.append(NL)

    print('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Compare 2~5 ST-GCN models on behavior video folders.'
    )
    parser.add_argument('--models', nargs='+', default=None,
                        help='2~5 個模型權重路徑；不指定則使用程式內 HARD_MODELS 預設清單')
    parser.add_argument('--names', nargs='+', default=None,
                        help='對應 --models 的顯示名稱（需與 --models 數量一致）；省略則自動從檔名推導')
    parser.add_argument('--ema_alphas', nargs='+', type=float, default=None,
                        help='對應 --models 的 EMA alpha（需與 --models 數量一致）；省略則全部使用 1.0')
    parser.add_argument('--seq_lens', nargs='+', type=int, default=None,
                        help='對應 --models 的序列長度（需與 --models 數量一致）；省略則全部使用 DEFAULT_SEQ_LEN')
    parser.add_argument('--video_walk_dir',      default=HARD_VIDEO_WALK_DIR)
    parser.add_argument('--video_lick_dir',      default=HARD_VIDEO_LICK_DIR)
    parser.add_argument('--video_scratch_dir',   default=HARD_VIDEO_SCRATCH_DIR)
    parser.add_argument('--video_shake_dir',     default=HARD_VIDEO_SHAKE_DIR)
    parser.add_argument('--video_stop_dir',      default=HARD_VIDEO_STOP_DIR,
                        help='stop 類別資料夾（留空則跳過）')
    parser.add_argument('--yolo',                default=DEFAULT_YOLO)
    parser.add_argument('--imgsz',               type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument('--conf',                type=float, default=DEFAULT_CONF)
    parser.add_argument('--output',              default=HARD_OUTPUT_DIR)
    parser.add_argument('--classify_stride',     type=int,   default=DEFAULT_STRIDE)
    parser.add_argument('--device',              default=DEFAULT_DEVICE)
    args = parser.parse_args()

    # ── 組出 2~5 個模型的設定清單 ────────────────────────────────────────────
    if args.models:
        n = len(args.models)
        names_in   = args.names       if args.names       else [None] * n
        alphas_in  = args.ema_alphas  if args.ema_alphas  else [1.0] * n
        seqlens_in = args.seq_lens    if args.seq_lens    else [DEFAULT_SEQ_LEN] * n
        if len(names_in) != n or len(alphas_in) != n or len(seqlens_in) != n:
            parser.error('--names / --ema_alphas / --seq_lens 若指定，數量必須跟 --models 一致')
        models_cfg = [
            {'path': p, 'name': nm, 'ema_alpha': a, 'seq_len': s}
            for p, nm, a, s in zip(args.models, names_in, alphas_in, seqlens_in)
        ]
    else:
        models_cfg = HARD_MODELS

    if not (2 <= len(models_cfg) <= 5):
        parser.error(f'需要 2~5 個模型，目前是 {len(models_cfg)} 個')

    n_models = len(models_cfg)
    names = [cfg['name'] or _short_name(cfg['path']) for cfg in models_cfg]
    # EMA alpha 不是 1.0 或序列長度不一致時附加標記，讓圖表/CSV 一眼看出差異
    labels = [f"{n}[ema={c['ema_alpha']}]" if c['ema_alpha'] < 1.0 else n
              for n, c in zip(names, models_cfg)]
    if len(set(c['seq_len'] for c in models_cfg)) > 1:
        labels = [f"{l}[T{c['seq_len']}]" for l, c in zip(labels, models_cfg)]

    # 檔案系統安全短名（避免 Windows MAX_PATH），只用在資料夾/檔名，不影響顯示用的 labels
    fs_labels = _fs_safe_labels(labels)

    # Sequential output directory
    out_root = Path(args.output)
    run_num  = _next_comparison_number(str(out_root))
    run_tag  = f"{run_num:03d}"
    out_dir  = _build_output_dir(out_root, run_tag, fs_labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = '=' * 78
    print(f"\n{sep}")
    print(f"  Comparison #{run_tag}  ({n_models} models)")
    for i, (cfg, lbl) in enumerate(zip(models_cfg, labels)):
        print(f"  Model {chr(65 + i)} : {lbl}  (EMA α={cfg['ema_alpha']}  T={cfg['seq_len']})")
    print(f"  Output  : {out_dir}")
    print(f"{sep}")

    # Load YOLO once, shared across all models
    print('\n[Loading YOLO]')
    kp_det = KeypointDetector(args.yolo, device=args.device,
                               imgsz=args.imgsz, conf_thres=args.conf)

    def _load_classifier(model_path, device, seq_len):
        bn_ch = infer_bn_input_channels(model_path)
        fm    = CH_TO_FEATURE.get(bn_ch, 'xy')
        if bn_ch not in CH_TO_FEATURE:
            print(f"  ⚠ {Path(model_path).name}: cannot infer feature_mode "
                  f"(bn_ch={bn_ch}), defaulting to 'xy'")
        else:
            print(f"  ✓ {Path(model_path).name} → {fm} ({bn_ch} ch)  T={seq_len}")
        clf = BehaviorClassifier(
            model_path, device=device,
            sequence_length=seq_len,
            normalize=True, feature_mode=fm, in_channels=bn_ch,
        )
        return clf, fm

    print('\n[Loading ST-GCN models]')
    classifiers = [_load_classifier(cfg['path'], args.device, cfg['seq_len']) for cfg in models_cfg]

    _all_dirs = [
        (args.video_walk_dir,    0),
        (args.video_lick_dir,    1),
        (args.video_scratch_dir, 2),
        (args.video_shake_dir,   3),
        (args.video_stop_dir,    4),
    ]
    # 資料夾不存在或未設定時跳過
    dirs = [(p, idx) for p, idx in _all_dirs if p and Path(p).is_dir()]
    skipped = [(BEHAVIOR_CLASSES[idx], p) for p, idx in _all_dirs
               if not p or not Path(p).is_dir()]
    if skipped:
        for cls_name, p in skipped:
            reason = '（路徑未設定）' if not p else f'（找不到資料夾: {p}）'
            print(f"  ⚠ 跳過 [{cls_name}] {reason}")

    # ── Inference ──────────────────────────────────────────────────────────
    # preds_list[model_idx][cls_idx] = [[pred_dict, ...], ...]  （外層 = 每部影片）
    preds_list = [dict() for _ in range(n_models)]
    # video_names[cls_idx] = [filename, ...]，跟 preds_list[*][cls_idx] 外層順序一致
    # （同一個資料夾、同樣排序過的檔案清單，所有模型看到的影片順序相同，只取 model 0 的）
    video_names = {}

    for dir_path, cls_idx in dirs:
        cls_name = BEHAVIOR_CLASSES[cls_idx]
        print(f"\n[{cls_name.upper()}]  {Path(dir_path).name}/")

        for mi, (cfg, (clf, fm)) in enumerate(zip(models_cfg, classifiers)):
            print(f"  ▶ {chr(65 + mi)} ({labels[mi]})")
            folder_results = evaluate_folder(
                dir_path, kp_det, clf, fm,
                cfg['seq_len'], args.classify_stride, ema_alpha=cfg['ema_alpha']
            )
            preds_list[mi][cls_idx] = [p for _, p in folder_results]
            if mi == 0:
                video_names[cls_idx] = [name for name, _ in folder_results]

        nv = len(preds_list[0][cls_idx])
        counts_str = '  '.join(
            f"{chr(65 + mi)}={sum(len(v) for v in preds_list[mi][cls_idx])}"
            for mi in range(n_models)
        )
        print(f"  → {nv} videos  |  {counts_str} windows")

    # ── Compute metrics ─────────────────────────────────────────────────────
    metrics_list = [compute_metrics(preds) for preds in preds_list]

    # ── Console summary（逐模型簡表；完整分析見 print_final_summary） ──────────
    col_w = max(max(len(l) for l in labels), 10)
    hdr = f"  {'Metric':<30}  " + '  '.join(f'{l:>{col_w}}' for l in labels)
    print(f"\n{sep}\n{hdr}\n  {'-' * (len(hdr) - 2)}")

    def _row(label, vals):
        best = max(vals)
        cells = [f"{'*' if abs(v - best) < 0.001 else ' '}{v:.4f}" for v in vals]
        print(f"  {label:<30}  " + '  '.join(f'{c:>{col_w}}' for c in cells))

    _row('Overall Accuracy',           [m['overall']['accuracy']                  for m in metrics_list])
    _row('Overall Top-2 Accuracy',     [m['overall']['top2_accuracy']             for m in metrics_list])
    _row('Overall Macro-F1',           [m['overall']['macro_f1']                  for m in metrics_list])
    _row('Event Det. Rate (ratio)',    [m['overall']['event_detection_rate']      for m in metrics_list])
    _row('Event Det. Rate (prob thr)', [m['overall']['prob_event_detection_rate'] for m in metrics_list])
    print()
    evaluated_cls = sorted({idx for _, idx in dirs})
    for i, cls in enumerate(BEHAVIOR_CLASSES):
        if i not in evaluated_cls:
            continue
        pcs = [m['per_class'][i] for m in metrics_list]
        _row(f'{cls:<8} accuracy',      [p['accuracy']      for p in pcs])
        _row(f'{cls:<8} top2_accuracy', [p['top2_accuracy'] for p in pcs])
        _row(f'{cls:<8} avg_true_prob', [p['avg_true_prob'] for p in pcs])
        _row(f'{cls:<8} event_rate',    [p['event_rate']    for p in pcs])
        detected_str = '  '.join(f"{chr(65+mi)}={p['n_videos_detected']}/{p['n_videos']}" for mi, p in enumerate(pcs))
        prob_str     = '  '.join(f"{chr(65+mi)}={p['n_videos_prob_detected']}/{p['n_videos']}" for mi, p in enumerate(pcs))
        print(f"  {cls:<8} clips detected  {detected_str}  "
              f"(prob≥{PROB_EVENT_THRESHOLD:.2f}: {prob_str})")

    print(sep)

    # ── Save outputs ─────────────────────────────────────────────────────────
    # plot_prob_histograms / save_preds_csv 需要 flat list，先在此展開
    flat_list = [
        {cls_idx: [p for vid in vids for p in vid] for cls_idx, vids in preds.items()}
        for preds in preds_list
    ]

    print('\n[Saving]')
    for fs_label, flat in zip(fs_labels, flat_list):
        # 用短檔名（fs_label）而非完整 label，理由同上（MAX_PATH）；
        # 完整名稱仍在 comparison_summary.csv 的欄位標題裡查得到。
        save_preds_csv(flat, fs_label, BEHAVIOR_CLASSES, out_dir)
    save_summary_csv(
        metrics_list, labels, BEHAVIOR_CLASSES, out_dir / 'comparison_summary.csv'
    )
    mcnemar_results = compute_mcnemar_pairs(preds_list, evaluated_cls)
    save_mcnemar_csv(
        mcnemar_results, labels, BEHAVIOR_CLASSES, evaluated_cls,
        out_dir / 'mcnemar_test.csv'
    )
    per_video_acc = compute_per_video_accuracy(preds_list, video_names, evaluated_cls)
    save_per_video_accuracy_csv(
        per_video_acc, labels, BEHAVIOR_CLASSES, out_dir / 'per_video_accuracy.csv'
    )
    plot_accuracy_comparison(
        metrics_list, labels, BEHAVIOR_CLASSES, out_dir / 'accuracy_comparison.png'
    )
    joint_weight_strs = [get_joint_prior_weights_str(clf) for clf, fm in classifiers]
    plot_confusion_matrices(
        [m['confusion_matrix'] for m in metrics_list], labels, BEHAVIOR_CLASSES,
        out_dir / 'confusion_matrices.png', joint_weight_strs=joint_weight_strs
    )
    plot_prob_histograms(
        flat_list, labels, BEHAVIOR_CLASSES, out_dir / 'prob_histograms.png'
    )

    print(f'\n✓ All results saved to: {out_dir}')

    print_final_summary(metrics_list, labels)
    print_mcnemar_summary(mcnemar_results, labels, BEHAVIOR_CLASSES, evaluated_cls)
    print_per_video_accuracy(per_video_acc, labels, BEHAVIOR_CLASSES)


if __name__ == '__main__':
    main()
