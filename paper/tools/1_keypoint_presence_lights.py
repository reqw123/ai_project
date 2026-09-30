"""
關鍵點燈號／缺失統計工具（延伸自 1_run_video_inference.py：YOLO-Pose + ST-GCN）

每個關鍵點對應一個燈號：該幀 YOLO 有偵測到貓、且這個點的信心值 >= KP_CONF_THRESHOLD
就算「推測出來」（綠燈），否則紅燈。同時跑 ST-GCN，知道關鍵點缺失的當下貓咪在做什麼，
用來回答「哪個行為容易丟哪些點」，例如舔毛時後腳膝蓋被身體擋住、搔抓時掌部動作太快而糊掉。

ST-GCN 的餵法跟 1_run_video_inference.py 一致：統一到 30fps 時基（高 fps 影片跳幀）、
16 幀滑動窗口、每 CLASSIFY_STRIDE 幀分類一次、偵測短暫消失時沿用最後一次姿態
（CAT_MISSING_TOLERANCE_FRAMES）、信心低於 STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
算低信心。每一幀的「當下行為」＝最近一次分類結果。

模式 1 — 背景分析（不開視窗）：
    預設走訪 模型專用/{train,val,test}/{walk,lick,scratch,shake,stop}/ 的所有影片，逐幀記錄
    17 點信心值與 ST-GCN 判定，用兩種分組各統計一次每個點的缺失率：
      folder：影片所在的行為資料夾（整支影片同一個標籤）
      stgcn ：該幀 ST-GCN 判定的行為（比較貼近「缺失當下的姿態」）；ST-GCN 信心不足的幀
              另外歸成 uncertain 一組——缺點太多本身就會讓 ST-GCN 判不出來，丟掉會低估缺失率
    ANALYSIS_SOURCE（或設定視窗的影片路徑）指定一個跟行為無關的資料夾時，只做 stgcn 分組。
    輸出到 OUTPUT_DIR：
      per_class_summary.csv    分組方式 × 類別 × 點 × 門檻：缺失率（影片平均／逐幀）、平均信心、
                               跟其他四類相比多缺多少（delta_vs_others）
      per_video.csv            每支影片 × 每個點的缺失率（主門檻）＋ ST-GCN 各行為佔比
      missing_frames.csv       每一個有缺點的幀：影片、時間、資料夾類別、ST-GCN 判定、缺了哪些點
                               （想回去看某個缺失畫面就從這裡找時間點）
      missing_heatmap.png      上排依資料夾、下排依 ST-GCN；左：缺失率，右：delta（紅＝這類特別容易缺）
    每支影片的推論結果快取在 OUTPUT_DIR/cache/（影片、兩個模型任一改變就失效），
    之後只改門檻重跑不必再推論。中途 Ctrl+C 會用已完成的影片照樣出報告。
    「沒偵測到貓」的幀不算進關鍵點缺失率，另外以 no_cat_rate 欄位列出。

模式 2 — GUI 檢視：
    播放五個行為資料夾的影片，左上角是 17 列燈號面板（一個點一列），面板最上面顯示
    ST-GCN 目前判定的行為與信心；這幀沒偵測到貓時 17 顆燈號全部改成黃色驚嘆號。每個點名稱右邊是本片「從有到無」的次數（綠→紅切換一次
    算一次；沒偵測到貓的幀跳過，不算成 17 個點各缺一次），影片循環播完一輪會歸零重算。
    按鍵：
      空白 暫停/繼續    1/2 上/下一部    z/x/c/v/b 切換 walk/lick/scratch/shake/stop 資料夾
      [ / ] 門檻 -/+0.05  l 顯示/隱藏燈號面板  p 面板換角落  s 再多一欄本片累計缺失%
      k 骨架＋偵測框顯示/隱藏（低信心點畫成紅色空心圈，看模型猜在哪）  r 從頭播放  ESC 結束
      Ctrl+加號／減號 縮放視窗
"""
import csv
import hashlib
import os
import sys
import time
from collections import deque
from pathlib import Path
from _report_paths import report_dir  # 報告輸出位置統一定義在 tools/_report_paths.py

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cat_monitoring_system"))
from utils.video_name_overlay import draw_video_name_label
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import YOLOConfig as _YOLOConfig
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig
import _window_zoom  # Ctrl+加號／減號縮放視窗（tools/_window_zoom.py）
from utils.skeleton_splits import video_class_folders, split_of_video

# ═══════════════════════════════════════════════════════
#  使用者設定區
# ═══════════════════════════════════════════════════════
RUN_MODE = 0  # 0: 啟動時選擇, 1: 背景分析, 2: GUI 檢視

KEYPOINT_NAMES = [
    "Nose", "Left_Ear", "Right_Ear", "Chest", "Mid_Back",
    "Hip", "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]
NUM_KP = len(KEYPOINT_NAMES)
BEHAVIOR_CLASSES = ["walk", "lick", "scratch", "shake", "stop"]
PRED_NONE = -1    # ST-GCN 信心不足：這幀沒有可用的行為判定（跟 utils.constants.LOW_CONF_ID 同值）
PRED_WARMUP = -2  # 還沒湊滿 16 幀，ST-GCN 還沒跑過
# 依 ST-GCN 分組時多一組 uncertain（信心不足的幀）：關鍵點缺太多正是 ST-GCN 判不出來的原因之一，
# 丟掉這些幀會低估缺失率，所以獨立列出來。warmup 幀跟姿態無關，不列入。
UNCERTAIN = "uncertain"
STGCN_GROUPS = BEHAVIOR_CLASSES + [UNCERTAIN]

YOLO_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "yolo_models" / "v11s_152.pt")
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()  # 設定視窗「🧠 模型路徑」欄位
if _env_yolo_model:
    YOLO_MODEL_PATH = _env_yolo_model
STGCN_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "stgcn_models" / "run_153_xy_conf_v_bone_att_on" / "153_best_model.pth")
_env_stgcn_model = os.getenv("CAT_MONITORING_STGCN_MODEL", "").strip()  # 設定視窗「⚙ 額外設定」
if _env_stgcn_model:
    STGCN_MODEL_PATH = _env_stgcn_model
STGCN_FEATURE_MODE = "xy"  # checkpoint 讀不到通道數時的 fallback；正常會依 checkpoint 自動校正
STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
CLASSIFY_STRIDE = 2        # 每幾個取樣幀做一次分類（跟 1_run_video_inference.py 一致）
TARGET_MODEL_FPS = 30.0    # 高於 30fps 的影片跳幀拉回 30fps 時基
BEHAVIOR_MIN_CONFIDENCE = _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
CAT_MISSING_TOLERANCE_FRAMES = _BehaviorTrackingConfig.CAT_MISSING_TOLERANCE_FRAMES
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = _YOLOConfig.IMAGE_SIZE  # 跟主系統同步（預設 640）

YOLO_CONF_THRESHOLD = 0.5  # YOLO bbox 偵測信心門檻（不是關鍵點門檻）
_env_yolo_conf = os.getenv("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", "").strip()
if _env_yolo_conf:
    try:
        YOLO_CONF_THRESHOLD = float(_env_yolo_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD={_env_yolo_conf!r} 不是數字，沿用預設 {YOLO_CONF_THRESHOLD}")

# 關鍵點「有推測出來」的門檻：預設跟 1_run_video_inference.py 畫骨架的 DRAW_KP_CONF_THRESHOLD 一致
KP_CONF_THRESHOLD = 0.7
_env_kp_conf = os.getenv("CAT_MONITORING_KP_CONF_THRES", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫
if _env_kp_conf:
    try:
        KP_CONF_THRESHOLD = float(_env_kp_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_KP_CONF_THRES={_env_kp_conf!r} 不是數字，沿用預設 {KP_CONF_THRESHOLD}")

# ── 模式 1 ──
OUTPUT_DIR = report_dir("analysis", "keypoint_presence")
# 報告同時列出的門檻（主門檻 KP_CONF_THRESHOLD 會自動加入）；0.3 對應抖動統計的 JITTER_CONF_THRESHOLD
REPORT_THRESHOLDS = (0.3, 0.5, 0.7)
# 分析來源：留空＝走訪五個行為資料夾（依資料夾、依 ST-GCN 兩種分組都做）；
# 填影片檔或資料夾（跟行為無關的影片，例如居家攝影機錄影）＝只依 ST-GCN 當下判定分組。
# 設定視窗的 TEST_VIDEO_PATH（🎬 影片路徑）也會套用到這裡。
ANALYSIS_SOURCE = ""
if _env_test_video_m1 := os.getenv("TEST_VIDEO_PATH", "").strip():
    ANALYSIS_SOURCE = _env_test_video_m1
MIN_FRAMES_PER_VIDEO = 10  # 某支影片在某組裡的有貓幀數少於這個數，不列入該組「影片平均」（比例不穩）
USE_CACHE = True

# ── 模式 2 ──
FOLDER_MAP = {
    'z': ("walk", "WALK"),
    'x': ("lick", "LICK"),
    'c': ("scratch", "SCRATCH"),
    'v': ("shake", "SHAKE"),
    'b': ("stop", "STOP"),
}
DEFAULT_FOLDER_KEY = 'z'
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()  # 設定了就只播這支影片／這個資料夾（z/x/c/v/b 失效）
WINDOW_NAME = "Keypoint Presence Lights"
DISPLAY_RESOLUTION = "720p"
_DISPLAY_RESOLUTION_PRESETS = {"720p": (1280, 720), "1080p": (1920, 1080)}
_env_resolution = os.getenv("DISPLAY_RESOLUTION", "").strip()
if _env_resolution:
    if _env_resolution in _DISPLAY_RESOLUTION_PRESETS:
        DISPLAY_RESOLUTION = _env_resolution
    else:
        print(f"⚠ 環境變數 DISPLAY_RESOLUTION={_env_resolution!r} 無效（只接受 {list(_DISPLAY_RESOLUTION_PRESETS)}），沿用預設 {DISPLAY_RESOLUTION}")
DISPLAY_SIZE = _DISPLAY_RESOLUTION_PRESETS[DISPLAY_RESOLUTION]
WINDOW_SCALE_STEP = 0.10
WINDOW_SCALE_MIN = 0.50
WINDOW_SCALE_MAX = 2.00

PANEL_CORNERS = ("tl", "tr", "bl")  # p 鍵循環；右下角留給影片檔名
LIGHT_ON = (80, 220, 80)     # BGR 綠
LIGHT_OFF = (60, 60, 235)    # BGR 紅
LIGHT_NO_CAT = (0, 200, 255)  # BGR 黃：沒偵測到貓時每顆燈號改畫黃色三角形驚嘆號
BBOX_COLOR = (255, 190, 0)    # BGR 青藍：貓咪偵測框

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}


# ═══════════════════════════════════════════════════════
#  共用
# ═══════════════════════════════════════════════════════
def resolve_run_mode():
    if RUN_MODE in (1, 2):
        return RUN_MODE
    print("\n請選擇執行模式:")
    print("  1) 背景分析（依行為統計各關鍵點缺失率，不開視窗）")
    print("  2) GUI 檢視（燈號面板）")
    while True:
        try:
            choice = input("輸入模式 (1/2, 預設=2): ").strip()
        except EOFError:
            print("\n未輸入模式，預設使用模式 2")
            return 2
        if choice in ("", "2"):
            return 2
        if choice == "1":
            return 1
        print(f"⚠ 輸入無效「{choice}」，請輸入 1 或 2（直接按 Enter 預設為 2）")


def list_videos(sources):
    """影片檔或資料夾（遞迴）展開成排序好、去重的影片路徑清單。"""
    out, seen = [], set()
    for src in sources:
        p = Path(src).expanduser()
        if p.is_file():
            files = [p] if p.suffix.lower() in SUPPORTED_VIDEO_EXTS else []
        elif p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS)
        else:
            print(f"⚠ 路徑不存在，略過: {p}")
            files = []
        for f in files:
            key = str(f.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(str(f))
    return out


def presence_mask(kpt_conf, threshold):
    """(17,) 布林陣列：True＝這個點有推測出來。沒偵測到貓（kpt_conf 為 None 或全 NaN）時全 False。"""
    if kpt_conf is None:
        return np.zeros(NUM_KP, dtype=bool)
    return np.asarray(kpt_conf, dtype=np.float32)[:NUM_KP] >= threshold  # NaN >= thr 為 False


def count_dropouts(confs, threshold):
    """confs (N,17)，沒偵測到貓的幀整列 NaN。回傳 (17,) 每個點「從有到無」的次數。
    沒偵測到貓的幀直接跳過（整隻貓不見是偵測問題，不算成 17 個點各缺一次）：
    有 → [沒貓] → 無 算一次，有 → [沒貓] → 有 不算。"""
    confs = np.asarray(confs, dtype=np.float32)
    if len(confs) == 0:
        return np.zeros(NUM_KP, dtype=np.int64)
    cat = confs[~np.isnan(confs).all(axis=1)]
    present = cat[:, :NUM_KP] >= threshold
    return (present[:-1] & ~present[1:]).sum(axis=0).astype(np.int64)


class PresenceStats:
    """GUI 用的本片累計：有貓幀數、各點缺失幀數、各點「從有到無」次數（規則同 count_dropouts）。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_cat = 0
        self.miss_count = np.zeros(NUM_KP, dtype=np.int64)
        self.drop_count = np.zeros(NUM_KP, dtype=np.int64)
        self.prev_present = None

    def update(self, kpt_conf, threshold):
        if kpt_conf is None:
            return
        present = presence_mask(kpt_conf, threshold)
        self.n_cat += 1
        self.miss_count += ~present
        if self.prev_present is not None:
            self.drop_count += self.prev_present & ~present
        self.prev_present = present


def frame_step_for(cap):
    """跟 1_run_video_inference.py 一樣：高於 30fps 的影片每 frame_step 幀取一幀。回傳 (frame_step, 原始 fps)。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = TARGET_MODEL_FPS
    step = max(1, int(round(fps / TARGET_MODEL_FPS))) if fps > TARGET_MODEL_FPS + 1e-6 else 1
    return step, fps


def _stgcn_feature_mode_and_channels():
    """依 checkpoint 的 bn_input 通道數校正 feature mode（同 1_run_video_inference.py main() 的做法）。"""
    from models.stgcn_model import get_in_channels_for_mode
    feature_mode, in_channels = STGCN_FEATURE_MODE, None
    ck_channel_map = {2: 'xy', 3: 'xy_conf', 5: 'xy_conf_v', 7: 'xy_conf_v_bone', 9: 'xy_conf_v_bone_bmotion'}
    try:
        import torch
        try:
            ck = torch.load(STGCN_MODEL_PATH, map_location='cpu', weights_only=True)
        except Exception:
            ck = torch.load(STGCN_MODEL_PATH, map_location='cpu')
        state_dict = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        if isinstance(state_dict, dict) and 'bn_input.weight' in state_dict:
            in_channels = int(state_dict['bn_input.weight'].shape[0])
            feature_mode = ck_channel_map.get(in_channels, feature_mode)
    except Exception as e:
        print(f"⚠ 無法從 checkpoint 推斷通道數（{e}），沿用 feature_mode={feature_mode}")
    if in_channels is None:
        in_channels = get_in_channels_for_mode(feature_mode)
    return feature_mode, in_channels


def load_models():
    from detectors.keypoint_detector import KeypointDetector
    from detectors.behavior_classifier import BehaviorClassifier
    detector = KeypointDetector(YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD)
    feature_mode, in_channels = _stgcn_feature_mode_and_channels()
    print(f"ST-GCN: {STGCN_MODEL_PATH}（feature_mode={feature_mode}）")
    classifier = BehaviorClassifier(
        STGCN_MODEL_PATH, device=INFERENCE_DEVICE, sequence_length=SEQUENCE_LENGTH,
        normalize=STGCN_NORMALIZE, feature_mode=feature_mode, in_channels=in_channels,
    )
    return detector, BehaviorTracker(classifier, feature_mode)


class BehaviorTracker:
    """逐幀餵 YOLO 結果、回傳這幀的「當下行為」。跟 1_run_video_inference.py 的分類段落同一套流程
    （EMA_ALPHA=1.0 等於不平滑，這裡省略；SQA 幾何檢查原腳本預設關閉，這裡也不做）。"""

    def __init__(self, classifier, feature_mode):
        self.classifier = classifier
        self.feature_mode = feature_mode
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.n_sampled = 0
        self.missing_streak = 0
        self.last_known = None
        self.pred_id, self.pred_conf = PRED_WARMUP, 0.0
        self.probs = np.zeros(len(BEHAVIOR_CLASSES), dtype=np.float32)

    def push(self, kpts, kpt_conf):
        """回傳 (pred_id, pred_conf)；pred_id 為 PRED_NONE＝信心不足、PRED_WARMUP＝還沒湊滿 16 幀。"""
        from models.stgcn_model import (interpolate_missing, flip_normalize, orientation_normalize,
                                         normalize_skeleton_coords, build_feature_tensor)
        self.n_sampled += 1
        if kpts is not None:
            self.missing_streak = 0
            self.last_known = (kpts.copy(), kpt_conf.copy())
        elif self.last_known is not None and self.missing_streak < CAT_MISSING_TOLERANCE_FRAMES:
            self.missing_streak += 1
            kpts, kpt_conf = self.last_known
        if kpts is None:
            return self.pred_id, self.pred_conf

        self.buffer.append((kpts, kpt_conf))
        if len(self.buffer) >= SEQUENCE_LENGTH and self.n_sampled % CLASSIFY_STRIDE == 0:
            kpts_arr = np.array([b[0] for b in self.buffer])
            conf_arr = np.array([b[1] for b in self.buffer])
            n_joints = getattr(self.classifier.model, 'num_joints', 17)
            kpts_arr, conf_arr = kpts_arr[:, :n_joints, :], conf_arr[:, :n_joints]
            seq = interpolate_missing(kpts_arr, conf_arr, threshold=0.0)
            if STGCN_NORMALIZE:
                seq = normalize_skeleton_coords(orientation_normalize(flip_normalize(seq)))
            feats = build_feature_tensor(seq, conf_arr, self.feature_mode)
            pid, pconf, probs = self.classifier.model.predict(feats, precomputed=True)
            if pid is None or float(pconf) < BEHAVIOR_MIN_CONFIDENCE:
                self.pred_id, self.pred_conf = PRED_NONE, float(pconf or 0.0)
            else:
                self.pred_id, self.pred_conf = int(pid), float(pconf)
            if probs is not None:
                self.probs = np.array((list(probs) + [0.0] * 5)[:5], dtype=np.float32)
        return self.pred_id, self.pred_conf


# ═══════════════════════════════════════════════════════
#  模式 1：背景分析
# ═══════════════════════════════════════════════════════
def _file_sig(p):
    p = Path(p)
    return f"{p.resolve()}|{p.stat().st_mtime}" if p.exists() else str(p)


def _cache_path(video_path):
    vp = Path(video_path)
    key = "|".join([_file_sig(vp), str(vp.stat().st_size), _file_sig(YOLO_MODEL_PATH), _file_sig(STGCN_MODEL_PATH),
                    str(YOLO_IMGSZ), str(YOLO_CONF_THRESHOLD), str(BEHAVIOR_MIN_CONFIDENCE),
                    str(CAT_MISSING_TOLERANCE_FRAMES), str(CLASSIFY_STRIDE), "v3"])
    return OUTPUT_DIR / "cache" / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".npz")


def extract_video(detector, tracker, video_path):
    """逐幀（30fps 時基）推論。回傳 dict：
      confs (N,17)：沒偵測到貓的幀整列 NaN
      pred  (N,)  ：ST-GCN 當下行為 id，PRED_NONE＝信心不足、PRED_WARMUP＝還沒湊滿 16 幀
      pred_conf (N,)、frame_idx (N,)：原影片幀號、fps"""
    cache = _cache_path(video_path) if USE_CACHE else None
    if cache is not None and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files} | {"fps": float(d["fps"])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("無法開啟影片")
    step, fps = frame_step_for(cap)
    detector.reset_track()
    tracker.reset()
    confs, preds, pconfs, fidx = [], [], [], []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                kpts, kpt_conf, _, _ = detector.detect(frame)
                pid, pconf = tracker.push(kpts, kpt_conf)
                confs.append(np.full(NUM_KP, np.nan, np.float32) if kpt_conf is None
                             else np.asarray(kpt_conf, np.float32)[:NUM_KP])
                preds.append(pid)
                pconfs.append(pconf)
                fidx.append(idx)
            idx += 1
    finally:
        cap.release()
    out = {
        "confs": np.stack(confs) if confs else np.zeros((0, NUM_KP), np.float32),
        "pred": np.array(preds, np.int16),
        "pred_conf": np.array(pconfs, np.float32),
        "frame_idx": np.array(fidx, np.int32),
        "fps": float(fps),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


def group_frames(records, by):
    """把每支影片的幀依分組拆成 [{"video", "cls", "confs"(有貓的幀), "n_no_cat"}]。
    by="folder"：整支影片歸到它的資料夾類別（資料夾跟行為無關的影片 cls=None，不列入）；
    by="stgcn"：依每幀 ST-GCN 判定分組，信心不足（PRED_NONE）歸 uncertain，warmup 幀不列入。"""
    out = []
    groups = BEHAVIOR_CLASSES if by == "folder" else STGCN_GROUPS
    for r in records:
        confs = r["confs"]
        if by == "folder":
            if r["cls"] is None:
                continue
            labels = np.full(len(confs), BEHAVIOR_CLASSES.index(r["cls"]))
        else:
            labels = r["pred"].astype(int)
            labels = np.where(labels == PRED_NONE, STGCN_GROUPS.index(UNCERTAIN), labels)
        has_cat = ~np.isnan(confs).all(axis=1) if len(confs) else np.zeros(0, bool)
        for cid, cls in enumerate(groups):
            sel = labels == cid
            if not sel.any():
                continue
            out.append({"video": r["video"], "cls": cls,
                        "confs": confs[sel & has_cat], "n_no_cat": int((sel & ~has_cat).sum())})
    return out


def summarize(video_records, thresholds, min_frames=MIN_FRAMES_PER_VIDEO, classes=BEHAVIOR_CLASSES):
    """video_records: [{"cls", "confs"(N,17，只含有貓的幀), "n_no_cat"}]（group_frames 的輸出）。
    回傳 {(cls, thr): {"n_videos", "n_cat_frames", "no_cat_rate", "miss_video"(17,),
    "miss_frame"(17,), "mean_conf"(17,), "delta"(17,)}}。
      miss_video：每支影片先算缺失率再平均（長片不會蓋過短片）
      miss_frame：所有幀混在一起算
      delta：這類的 miss_video 減掉「其他行為類別」miss_video 的平均（>0 代表這類特別容易缺這個點）；
             比較基準只用五個行為，不含 uncertain，uncertain 自己的 delta 則是跟五個行為的平均比
    某類沒有夠長的影片時 miss_video 為 NaN。"""
    out = {}
    for thr in thresholds:
        for cls in classes:
            recs = [r for r in video_records if r["cls"] == cls]
            all_confs = [r["confs"] for r in recs if len(r["confs"])]
            stacked = np.concatenate(all_confs) if all_confs else np.zeros((0, NUM_KP), np.float32)
            n_cat = len(stacked)
            n_no_cat = sum(r["n_no_cat"] for r in recs)
            per_video = [(r["confs"] < thr).mean(axis=0) for r in recs if len(r["confs"]) >= min_frames]
            out[(cls, thr)] = {
                "n_videos": len(per_video),
                "n_cat_frames": n_cat,
                "no_cat_rate": n_no_cat / (n_cat + n_no_cat) if (n_cat + n_no_cat) else float("nan"),
                "miss_video": np.mean(per_video, axis=0) if per_video else np.full(NUM_KP, np.nan),
                "miss_frame": (stacked < thr).mean(axis=0) if n_cat else np.full(NUM_KP, np.nan),
                "mean_conf": stacked.mean(axis=0) if n_cat else np.full(NUM_KP, np.nan),
            }
        for cls in classes:
            others = [out[(c, thr)]["miss_video"] for c in classes if c != cls and c in BEHAVIOR_CLASSES]
            others = [o for o in others if not np.all(np.isnan(o))]
            base = np.mean(others, axis=0) if others else np.full(NUM_KP, np.nan)
            out[(cls, thr)]["delta"] = out[(cls, thr)]["miss_video"] - base
    return out


def group_classes(by):
    return BEHAVIOR_CLASSES if by == "folder" else STGCN_GROUPS


def _pred_name(pid):
    if 0 <= pid < len(BEHAVIOR_CLASSES):
        return BEHAVIOR_CLASSES[pid]
    return "warmup" if pid == PRED_WARMUP else UNCERTAIN


def write_reports(records, summaries, thresholds, main_thr):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "per_video.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "folder_class", "split", "cat_frames", "no_cat_frames", "threshold"]
                   + [f"miss_{n}" for n in KEYPOINT_NAMES] + [f"drops_{n}" for n in KEYPOINT_NAMES]
                   + [f"stgcn_{c}_ratio" for c in STGCN_GROUPS] + ["stgcn_warmup_ratio"])
        for r in records:
            confs = r["confs"]
            cat = confs[~np.isnan(confs).all(axis=1)] if len(confs) else confs
            miss = (cat < main_thr).mean(axis=0) if len(cat) else np.full(NUM_KP, np.nan)
            n = max(1, len(r["pred"]))
            ratios = [(r["pred"] == i).sum() / n for i in range(len(BEHAVIOR_CLASSES))] + [(r["pred"] == PRED_NONE).sum() / n, (r["pred"] == PRED_WARMUP).sum() / n]
            w.writerow([Path(r["video"]).name, r["cls"] or "", r["split"] or "", len(cat), len(confs) - len(cat), main_thr]
                       + [f"{v:.4f}" for v in miss] + list(count_dropouts(confs, main_thr))
                       + [f"{v:.4f}" for v in ratios])

    with open(OUTPUT_DIR / "per_class_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group_by", "class", "threshold", "keypoint", "miss_rate_video_avg", "miss_rate_frame",
                    "mean_conf", "delta_vs_others", "n_videos", "cat_frames", "no_cat_rate"])
        for by, summary in summaries.items():
            for thr in thresholds:
                for cls in group_classes(by):
                    s = summary[(cls, thr)]
                    for k, name in enumerate(KEYPOINT_NAMES):
                        w.writerow([by, cls, thr, name, f"{s['miss_video'][k]:.4f}", f"{s['miss_frame'][k]:.4f}",
                                    f"{s['mean_conf'][k]:.4f}", f"{s['delta'][k]:+.4f}",
                                    s["n_videos"], s["n_cat_frames"], f"{s['no_cat_rate']:.4f}"])

    with open(OUTPUT_DIR / "missing_frames.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video", "folder_class", "frame", "time_sec", "stgcn_pred", "stgcn_conf",
                    "threshold", "n_missing", "missing_keypoints"])
        for r in records:
            for i, c in enumerate(r["confs"]):
                if np.isnan(c).all():
                    continue
                miss = np.where(c < main_thr)[0]
                if len(miss) == 0:
                    continue
                w.writerow([Path(r["video"]).name, r["cls"] or "", int(r["frame_idx"][i]),
                            f"{r['frame_idx'][i] / r['fps']:.2f}", _pred_name(int(r["pred"][i])),
                            f"{r['pred_conf'][i]:.3f}", main_thr, len(miss),
                            ";".join(KEYPOINT_NAMES[k] for k in miss)])

    _plot_heatmap(summaries, main_thr, OUTPUT_DIR / "missing_heatmap.png")


def _plot_heatmap(summaries, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(summaries.items())
    fig, axes = plt.subplots(len(groups), 2, figsize=(20, 4.6 * len(groups)), squeeze=False)
    titles = {"folder": "grouped by video folder", "stgcn": "grouped by ST-GCN prediction (per frame)"}
    for row, (by, summary) in enumerate(groups):
        classes = group_classes(by)
        miss = np.array([summary[(c, thr)]["miss_video"] for c in classes]) * 100
        delta = np.array([summary[(c, thr)]["delta"] for c in classes]) * 100
        ylabels = [f"{c} (n={summary[(c, thr)]['n_videos']})" for c in classes]
        lim = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() and np.nanmax(np.abs(delta)) > 0 else 1
        panels = [
            (axes[row][0], miss, "Reds", 0, 100, f"Missing rate % (conf < {thr:.2f}) — {titles.get(by, by)}"),
            (axes[row][1], delta, "RdBu_r", -lim, lim, f"Delta vs other behaviors (pt) — {titles.get(by, by)}"),
        ]
        for ax, data, cmap, vmin, vmax, title in panels:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(NUM_KP))
            ax.set_xticklabels(KEYPOINT_NAMES, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(ylabels, fontsize=9)
            ax.set_title(title, fontsize=10)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isfinite(data[i, j]):
                        ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_console_summary(summaries, main_thr, top_n=3):
    names = {"folder": "依影片資料夾", "stgcn": "依 ST-GCN 當下判定"}
    for by, summary in summaries.items():
        print("\n" + "=" * 70)
        print(f"各行為最容易缺失的關鍵點（{names.get(by, by)}，門檻 {main_thr:.2f}，影片平均缺失率）")
        print("=" * 70)
        for cls in group_classes(by):
            s = summary[(cls, main_thr)]
            if s["n_videos"] == 0:
                print(f"\n[{cls}] 沒有足夠的幀")
                continue
            if cls == UNCERTAIN:
                print("\n（uncertain＝ST-GCN 信心不足的幀；缺失率明顯高於各行為＝缺點是 ST-GCN 判不出來的原因之一）")
            print(f"\n[{cls}] {s['n_videos']} 部影片，{s['n_cat_frames']} 幀有貓，沒偵測到貓 {s['no_cat_rate']*100:.1f}%")
            top = np.argsort(-np.nan_to_num(s["miss_video"], nan=-1))[:top_n]
            print("  缺失率最高: " + "、".join(f"{KEYPOINT_NAMES[k]} {s['miss_video'][k]*100:.1f}%" for k in top))
            special = [k for k in np.argsort(-np.nan_to_num(s["delta"], nan=-1))[:top_n] if s["delta"][k] > 0.05]
            if special:
                print("  比其他類別特別容易缺: " + "、".join(f"{KEYPOINT_NAMES[k]} +{s['delta'][k]*100:.1f}pt" for k in special))
            else:
                print("  沒有比其他類別明顯多缺（>5pt）的點")


def run_analysis():
    thresholds = tuple(sorted(set(REPORT_THRESHOLDS) | {KP_CONF_THRESHOLD}))
    jobs = []
    if ANALYSIS_SOURCE:
        # 資料夾跟行為無關：沒有資料夾標籤，只能依 ST-GCN 當下判定分組
        jobs = [(None, v) for v in list_videos([ANALYSIS_SOURCE])]
        groupings = ("stgcn",)
        print(f"分析來源: {ANALYSIS_SOURCE}（{len(jobs)} 部影片，不看資料夾，只依 ST-GCN 判定分組）")
    else:
        for cls in BEHAVIOR_CLASSES:
            vids = list_videos(video_class_folders(classes=cls))
            print(f"  {cls}: {len(vids)} 部影片")
            jobs += [(cls, v) for v in vids]
        groupings = ("folder", "stgcn")
    if not jobs:
        print("❌ 找不到影片，請確認 ANALYSIS_SOURCE 或 utils/skeleton_splits.py 的 VIDEO_ROOT")
        return

    print(f"\nYOLO 模型: {YOLO_MODEL_PATH}")
    print(f"報告門檻 {thresholds}（主門檻 {KP_CONF_THRESHOLD}），ST-GCN 行為信心門檻 {BEHAVIOR_MIN_CONFIDENCE}")
    print(f"輸出資料夾: {OUTPUT_DIR}\n")
    detector, tracker = load_models()

    records = []
    t0 = time.time()
    try:
        for i, (cls, video) in enumerate(jobs, 1):
            try:
                r = extract_video(detector, tracker, video)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] ⚠ {Path(video).name} 略過：{e}")
                continue
            r.update(cls=cls, video=video, split=split_of_video(video))
            records.append(r)
            cat = r["confs"][~np.isnan(r["confs"]).all(axis=1)] if len(r["confs"]) else r["confs"]
            miss = (cat < KP_CONF_THRESHOLD).mean() * 100 if len(cat) else float("nan")
            valid = r["pred"][r["pred"] >= 0]
            top = (f"ST-GCN 最常判 {BEHAVIOR_CLASSES[np.bincount(valid).argmax()]}" if len(valid)
                   else "ST-GCN 沒有有效判定")
            print(f"[{i}/{len(jobs)}] {cls or '':<8s}{Path(video).name}  有貓 {len(cat)}/{len(r['confs'])} 幀、"
                  f"平均缺 {miss:.1f}%、{top}  ({time.time() - t0:.0f}s)")
    except KeyboardInterrupt:
        print(f"\n⚠ 使用者中斷，用已完成的 {len(records)} 部影片出報告")

    if not records:
        return
    summaries = {by: summarize(group_frames(records, by), thresholds, classes=group_classes(by)) for by in groupings}
    write_reports(records, summaries, thresholds, KP_CONF_THRESHOLD)
    print_console_summary(summaries, KP_CONF_THRESHOLD)
    print(f"\n✓ 報告已輸出: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════
#  模式 2：GUI 檢視
# ═══════════════════════════════════════════════════════
def letterbox(image, target_size):
    """等比縮放置中補黑邊；回傳 (畫面, 縮放倍率, x 偏移, y 偏移)。"""
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, scale, ox, oy


def draw_skeleton(frame, kpts, present, s):
    """有推測出來的點畫實心點與連線；低信心點畫紅色空心圈（看模型猜在哪）。"""
    pts = [tuple(int(v) for v in p) for p in kpts]
    for a, b in SKELETON_EDGES:
        if present[a] and present[b]:
            cv2.line(frame, pts[a], pts[b], (230, 230, 230), max(1, int(2 * s)), cv2.LINE_AA)
    for k, p in enumerate(pts):
        if present[k]:
            cv2.circle(frame, p, max(2, int(4 * s)), LIGHT_ON, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, p, max(2, int(4 * s)), LIGHT_OFF, 1, cv2.LINE_AA)


def draw_bbox(frame, bbox, bbox_conf, s):
    """貓咪偵測框（已換算到顯示座標）＋左上角信心值標籤。"""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    th = max(1, int(2 * s))
    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, th, cv2.LINE_AA)
    if bbox_conf is None:
        return
    label = f"cat {bbox_conf:.2f}"
    fs = 0.45 * s
    (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    ty = y1 - int(4 * s) if y1 - tht - base - int(4 * s) > 0 else y1 + tht + int(4 * s)  # 框太靠上就畫進框內
    cv2.rectangle(frame, (x1, ty - tht - int(3 * s)), (x1 + tw + int(6 * s), ty + base), BBOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + int(3 * s), ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)


def draw_warning_icon(frame, center, r):
    """黃色三角形＋黑色驚嘆號，大小跟燈號圓點差不多（r＝燈號半徑）。"""
    cx, cy = center
    h = int(r * 2.3)
    pts = np.array([[cx, cy - h // 2 - 1], [cx - h // 2 - 1, cy + h // 2], [cx + h // 2 + 1, cy + h // 2]], np.int32)
    cv2.fillPoly(frame, [pts], LIGHT_NO_CAT, cv2.LINE_AA)
    lw = max(1, r // 3)
    cv2.line(frame, (cx, cy - h // 4), (cx, cy + h // 8), (0, 0, 0), lw, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy + h // 3), max(1, lw // 2), (0, 0, 0), -1, cv2.LINE_AA)


def draw_light_panel(frame, present, header_lines, s, corner="tl", drop_count=None, miss_pct=None, no_cat=False):
    """17 列燈號面板（一個點一列）。header_lines：面板最上面的幾行文字（行為判定、門檻）。
    no_cat=True（這幀沒偵測到貓）時，每顆燈號改畫黃色三角形驚嘆號，跟「有貓但這個點沒抓到」的紅燈區分。
    drop_count 不為 None 時，名稱右邊顯示本片「從有到無」次數（>0 黃字）；
    miss_pct 不為 None 時，再多一欄本片累計缺失%。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.36 * s
    row_h = int(13 * s)
    pad = int(5 * s)
    r = max(3, int(4 * s))
    gap = int(6 * s)
    name_w = max(cv2.getTextSize(n, font, fs, 1)[0][0] for n in KEYPOINT_NAMES)
    drop_w = 0
    if drop_count is not None:
        drop_w = max(cv2.getTextSize(str(int(v)), font, fs, 1)[0][0] for v in drop_count)
        drop_w = max(drop_w, cv2.getTextSize("00", font, fs, 1)[0][0]) + gap
    pct_w = cv2.getTextSize("100%", font, fs, 1)[0][0] + gap if miss_pct is not None else 0
    header_w = max(cv2.getTextSize(t, font, fs, 1)[0][0] for t in header_lines) if header_lines else 0
    pw = pad * 2 + max(r * 2 + gap + name_w + drop_w + pct_w, header_w)
    ph = pad * 2 + row_h * (len(header_lines) + NUM_KP) + (int(3 * s) if header_lines else 0)
    H, W = frame.shape[:2]
    margin = int(8 * s)
    x0 = margin if corner in ("tl", "bl") else W - pw - margin
    y0 = margin if corner in ("tl", "tr") else H - ph - margin

    roi = frame[y0:y0 + ph, x0:x0 + pw]
    frame[y0:y0 + ph, x0:x0 + pw] = (roi * 0.35).astype(np.uint8)  # 半透明深色底

    y = y0 + pad
    for text in header_lines:
        cv2.putText(frame, text, (x0 + pad, y + int(row_h * 0.8)), font, fs, (235, 235, 235), 1, cv2.LINE_AA)
        y += row_h
    if header_lines:
        y += int(3 * s)
    for k, name in enumerate(KEYPOINT_NAMES):
        cy = y + row_h // 2
        cx = x0 + pad + r
        if no_cat:
            draw_warning_icon(frame, (cx, cy), r)
        else:
            cv2.circle(frame, (cx, cy), r, LIGHT_ON if present[k] else LIGHT_OFF, -1, cv2.LINE_AA)
        tx = cx + r + gap
        cv2.putText(frame, name, (tx, cy + int(4 * s)), font, fs, (220, 220, 220), 1, cv2.LINE_AA)
        if drop_count is not None:
            n = int(drop_count[k])
            txt = str(n)
            right = tx + name_w + drop_w  # 次數欄靠右對齊
            color = (60, 210, 255) if n > 0 else (140, 140, 140)
            cv2.putText(frame, txt, (right - cv2.getTextSize(txt, font, fs, 1)[0][0], cy + int(4 * s)),
                        font, fs, color, 1, cv2.LINE_AA)
        if miss_pct is not None:
            pct = f"{miss_pct[k]:.0f}%"
            pw_txt = cv2.getTextSize(pct, font, fs, 1)[0][0]
            color = (120, 120, 255) if miss_pct[k] >= 20 else (200, 200, 200)
            cv2.putText(frame, pct, (x0 + pw - pad - pw_txt, cy + int(4 * s)), font, fs, color, 1, cv2.LINE_AA)
        y += row_h


def print_video_stats(stats):
    if not stats.n_cat:
        return
    worst = np.argsort(-stats.miss_count)[:3]
    print("  本片最常缺: " + "、".join(f"{KEYPOINT_NAMES[k]} {stats.miss_count[k] / stats.n_cat * 100:.0f}%" for k in worst))
    drops = [(KEYPOINT_NAMES[k], int(stats.drop_count[k])) for k in np.argsort(-stats.drop_count) if stats.drop_count[k] > 0]
    print("  從有到無次數: " + ("、".join(f"{n} {c}次" for n, c in drops) if drops else "無"))


def run_gui():
    if _env_test_video:
        playlists = {k: list_videos([_env_test_video]) for k in FOLDER_MAP}
        print(f"[TEST_VIDEO_PATH] {_env_test_video}：{len(playlists[DEFAULT_FOLDER_KEY])} 部影片（z/x/c/v/b 不切換）")
    else:
        playlists = {}
        for k, (cls, label) in FOLDER_MAP.items():
            playlists[k] = list_videos(video_class_folders(classes=cls))
            print(f"  [{k}] {label}: {len(playlists[k])} 部影片")
    folder_key = DEFAULT_FOLDER_KEY
    positions = {k: 0 for k in FOLDER_MAP}
    if not any(playlists.values()):
        print("❌ 找不到影片")
        return

    detector, tracker = load_models()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
    window_scale = 1.0
    s = DISPLAY_SIZE[1] / 720.0

    thr = KP_CONF_THRESHOLD
    show_panel, show_stats, show_skeleton = True, False, True
    corner_idx = 0
    paused = False

    while True:
        videos = playlists[folder_key]
        if not videos:
            print(f"⚠ {FOLDER_MAP[folder_key][1]} 沒有影片，切回有影片的資料夾")
            folder_key = next(k for k, v in playlists.items() if v)
            continue
        idx = positions[folder_key] % len(videos)
        video_path = videos[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠ 無法開啟，跳過: {video_path}")
            positions[folder_key] = idx + 1
            continue
        step, _ = frame_step_for(cap)
        print(f"\n▶ [{FOLDER_MAP[folder_key][1]} {idx + 1}/{len(videos)}] {Path(video_path).name}")
        detector.reset_track()
        tracker.reset()
        # 本片累計（播完一輪循環、門檻改了、按 r 都從頭算）
        stats = PresenceStats()
        delta, next_folder = 0, None
        last = None  # (kpts, kpt_conf, bbox, bbox_conf, frame, pred_id, pred_conf)

        while True:
            if not paused or last is None:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循環播放
                    print_video_stats(stats)
                    stats.reset()
                    tracker.reset()
                    detector.reset_track()
                    ok, frame = cap.read()
                    if not ok:
                        delta = 1
                        break
                for _ in range(step - 1):  # 高 fps 影片跳幀，跟 ST-GCN 的 30fps 時基一致
                    cap.grab()
                kpts, kpt_conf, bbox, bbox_conf = detector.detect(frame)
                pid, pconf = tracker.push(kpts, kpt_conf)
                last = (kpts, kpt_conf, bbox, bbox_conf, frame, pid, pconf)
                stats.update(kpt_conf, thr)
            kpts, kpt_conf, bbox, bbox_conf, frame, pid, pconf = last
            present = presence_mask(kpt_conf, thr)

            show, scale, ox, oy = letterbox(frame, DISPLAY_SIZE)
            if show_skeleton and bbox is not None:
                draw_bbox(show, np.asarray(bbox, dtype=np.float32) * scale + (ox, oy, ox, oy), bbox_conf, s)
            if show_skeleton and kpts is not None:
                draw_skeleton(show, np.asarray(kpts)[:NUM_KP] * scale + (ox, oy), present, s)
            if show_panel:
                if pid >= 0:
                    behavior = f"{BEHAVIOR_CLASSES[pid].upper()} {pconf * 100:.0f}%"
                elif pid == PRED_WARMUP:
                    behavior = "GCN warming up"
                else:
                    behavior = f"GCN low conf {pconf * 100:.0f}%"
                status = "NO CAT" if kpt_conf is None else f"{int(present.sum())}/{NUM_KP} found"
                header = [behavior, f"{status}  thr {thr:.2f}"]
                miss_pct = stats.miss_count / stats.n_cat * 100 if (show_stats and stats.n_cat) else None
                draw_light_panel(show, present, header, s, PANEL_CORNERS[corner_idx], stats.drop_count, miss_pct,
                                 no_cat=kpt_conf is None)
            folder_label = Path(video_path).parent.name if _env_test_video else FOLDER_MAP[folder_key][1]
            nav = f"folder: {folder_label}" + ("  PAUSED" if paused else "")
            (tw, _), _ = cv2.getTextSize(nav, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 1)
            cv2.putText(show, nav, (DISPLAY_SIZE[0] - tw - int(10 * s), int(22 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (160, 210, 255), 1, cv2.LINE_AA)
            draw_video_name_label(show, video_path, idx, len(videos), ui_scale=s)
            cv2.imshow(WINDOW_NAME, show)

            key = cv2.waitKey(30 if paused else 1) & 0xFF
            zoom = _window_zoom.poll(WINDOW_NAME)
            if zoom:
                window_scale = _window_zoom.clamp_scale(window_scale, zoom, WINDOW_SCALE_STEP, WINDOW_SCALE_MIN, WINDOW_SCALE_MAX)
                cv2.resizeWindow(WINDOW_NAME, int(DISPLAY_SIZE[0] * window_scale), int(DISPLAY_SIZE[1] * window_scale))
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                key = 27
            if key == 255:
                continue
            ch = chr(key)
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if ch == ' ':
                paused = not paused
            elif ch == '2':
                delta = 1
                break
            elif ch == '1':
                delta = -1
                break
            elif ch in FOLDER_MAP and ch != folder_key and not _env_test_video:
                next_folder = ch
                break
            elif ch in ('[', ']'):
                thr = round(min(0.95, max(0.05, thr + (0.05 if ch == ']' else -0.05))), 2)
                stats.reset()
                print(f"門檻 → {thr:.2f}（本片缺失次數／缺失% 重新計算）")
            elif ch == 'l':
                show_panel = not show_panel
            elif ch == 'p':
                corner_idx = (corner_idx + 1) % len(PANEL_CORNERS)
            elif ch == 's':
                show_stats = not show_stats
            elif ch == 'k':
                show_skeleton = not show_skeleton
            elif ch == 'r':
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                stats.reset()
                detector.reset_track()
                tracker.reset()
                last = None

        cap.release()
        print_video_stats(stats)
        if next_folder:
            positions[folder_key] = idx
            folder_key = next_folder
        else:
            positions[folder_key] = idx + delta


def main():
    if resolve_run_mode() == 1:
        run_analysis()
    else:
        run_gui()


if __name__ == "__main__":
    main()
