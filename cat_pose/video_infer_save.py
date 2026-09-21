 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cat Pose Video Inference & Save Tool
- 單純推論影片
- 按 S 鍵儲存當前影像到指定資料夾
- 空白鍵切換 Step/Auto 模式
- 影像命名: walk{i}.png

按鍵（每次按下都會在終端印出訊息）：
- S 儲存  Space 切換 Step/Auto  ESC 離開
- Step：D/A 前後一幀（步長 Z/X）
- 通用：[ ] 往回/往前跳 JUMP_FRAMES 幀   t 終端輸入跳轉（秒數加 s 或幀號）
- 視窗頂端進度條可拖曳即時 seek
- 1 / 2（數字列或 numpad，需開 NumLock）上一部 / 下一部影片
- = 或 + 加大快轉倍率、- 減小（階梯 0.5/1/2/4/8x，單次切換）
- . / , 在倍率之外額外微調抽幀（快轉）
- 解析度上限：改開頭 DISPLAY_RESOLUTION（"720p" / "1080p"）
"""

import os
import re
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# 讓 print 一律逐行 flush：從 settings_window 啟動時 stdout 是 pipe（非 TTY），
# Python 預設會區塊緩衝，按鍵訊息會卡住等緩衝滿或程式結束才一次噴出。逐行
# 緩衝後不論在真終端機或被 pipe 都即時顯示。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(line_buffering=True)
    except Exception:
        pass

from constants import (
    EAR_DISTANCE_SKELETON_EDGES as _SKELETON_EDGES,
    EAR_DISTANCE_KP_COLORS as _KP_COLORS,
    EAR_DISTANCE_EDGE_COLORS as _EDGE_COLORS,
)

# ==================== 設定 ====================
MODEL_PATH = r"C:\ai_project\yolo_models\v11s_150.pt"

# 若設定 YOLO_MODEL_PATH 環境變數，優先使用該模型路徑（覆蓋上面寫死的 MODEL_PATH，
# 對應 settings_window.py 的「🧠 模型路徑」欄位）
_env_yolo_model = os.getenv("YOLO_MODEL_PATH", "").strip()
if _env_yolo_model:
    MODEL_PATH = _env_yolo_model

VIDEO_DIR = r"C:\Users\homec\Downloads\walk_標記圖片"  # 讀取資料夾下所有影片

# 若設定 TEST_VIDEO_PATH 環境變數，優先使用該路徑（覆蓋上面寫死的 VIDEO_DIR，對應
# settings_window.py 的「🎬 影片路徑」欄位）。填單一檔案 → 只處理那一支影片；
# 填資料夾 → 整個換成掃描這個資料夾（VIDEO_DIR = 該資料夾）。
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video and os.path.isfile(_env_test_video):
    VIDEO_DIR = None
elif _env_test_video and os.path.isdir(_env_test_video):
    VIDEO_DIR = _env_test_video

OUTPUT_DIR = r"C:/cat_pose/cat53"
IMG_NAME_FORMAT = "lick_real3-{}.png"
TARGET_MODEL_FPS = 30.0

# ==================== 播放 / 跳幀 / 跳轉設定 ====================
# 你要的「t」：開啟每一部影片時，直接跳到這一幀開始（0 = 從頭）。
# 想略過片頭、或每次都從某個時間點檢視就設這裡（例：30fps 影片想從第 10 秒開始 → 300）。
SEEK_FRAME = 0

# Auto（自動播放）模式的快轉：每顯示一幀後「額外」跳過幾幀。
#   0 = 正常逐幀；1 ≈ 2倍速；4 ≈ 5倍速 …
# 用 cap.grab() 前進，不做解碼/色彩轉換，所以真的比較快。
# 執行時可用 . （加速）/ , （減速）即時調整。
AUTO_FRAME_SKIP = 0

# 執行時按 [ / ] 一次往回 / 往前跳的幀數（快速定位用，兩種模式都可用）。
JUMP_FRAMES = 300

# 執行時按 = / +（加大）、-（減小）單次切換快轉倍率，於下列階梯間移動。
#   0.5x 會用節流放慢每個顯示幀的停留時間；>=1x 用 cap.grab() 抽幀加速。
#   . / , 仍可在此基礎上「額外」微調抽幀數。
SPEED_STEPS = [0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_PLAYBACK_SPEED = 1.0

# 支援影片副檔名
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.mpeg', '.mpg'}

# ==================== 資料夾自動接續命名 ====================
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# 根據 IMG_NAME_FORMAT 前綴搜尋現有檔案
prefix = IMG_NAME_FORMAT.split('{')[0]
pattern = re.compile(rf"{re.escape(prefix)}(\d+)\.(jpg|png)$")
max_idx = 0
for file in output_dir.iterdir():
    if file.is_file():
        m = pattern.match(file.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
save_idx = max_idx + 1

# 取得所有影片清單（自然排序：screen_1, screen_2, ..., screen_10 順序正確）
def _natural_key(path):
    parts = re.split(r'(\d+)', os.path.basename(path).lower())
    return [int(p) if p.isdigit() else p for p in parts]

def get_all_videos(folder):
    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(os.path.join(root, file))
    videos.sort(key=_natural_key)
    return videos

if _env_test_video and os.path.isfile(_env_test_video):
    video_list = [_env_test_video]
else:
    video_list = get_all_videos(VIDEO_DIR)
if not video_list:
    print(f"[Error] No videos found in {VIDEO_DIR}")
    exit(1)
video_idx = 0
VIDEO_PATH = video_list[video_idx]

# ==================== 載入模型 ====================
print("[Loading] Model...")
try:
    model = YOLO(MODEL_PATH)
    model.to("cuda")
    print("[OK] Model loaded (GPU)!")
    USE_HALF = True
except Exception as e:
    print(f"[Failed] GPU loading failed: {e}")
    try:
        model = YOLO(MODEL_PATH)
        print("[OK] Model loaded (CPU)!")
        USE_HALF = False
    except Exception as e2:
        print(f"[Failed] Model loading failed: {e2}")
        exit(1)

# ---- 推論精度參數 ----
# ultralytics 8.4+ 把 predict 的 half=True 改名成 quantize=16（FP16）；
# 舊版沒有 quantize，仍用 half。依安裝版本自動選對的參數名，避免 deprecated 警告。
def _ver_tuple(s):
    out = []
    for part in str(s).split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)

try:
    from ultralytics import __version__ as _ULTRA_VER
except Exception:
    _ULTRA_VER = "0.0.0"

if not USE_HALF:
    PRECISION_KW = {}
elif _ver_tuple(_ULTRA_VER) >= (8, 4, 0):
    PRECISION_KW = {"quantize": 16}
else:
    PRECISION_KW = {"half": True}
print(f"[Info] ultralytics {_ULTRA_VER}  precision={PRECISION_KW or 'fp32'}")

import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[1] / "paper"))  # config.py 在 paper/ 根目錄
from config import YOLOConfig as _YOLOConfig

import os as _os
YOLO_CONF_THRESHOLD = 0.5  # YOLO 偵測框（bbox）信心門檻（predict 的 conf）；不是關鍵點（kp）的門檻，kp 見下面的 KP_CONF_THRES
_env_yolo_conf = _os.getenv("CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫；變數名同 config.py 的 YOLOConfig.CONFIDENCE_THRESHOLD
if _env_yolo_conf:
    try:
        YOLO_CONF_THRESHOLD = float(_env_yolo_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_YOLO_CONFIDENCE_THRESHOLD={_env_yolo_conf!r} 不是數字，沿用預設 {YOLO_CONF_THRESHOLD}")


def infer(frame):
    """單張影像推論，回傳第一個 Results。"""
    return model.predict(frame, imgsz=_YOLOConfig.IMAGE_SIZE, conf=YOLO_CONF_THRESHOLD, verbose=False, **PRECISION_KW)[0]

# ========== 儲存日誌 ==========
saved_log = {}  # 影片路徑: [儲存過的圖片檔名]

# ==================== 開啟影片 ====================
cap = None
fps = 0.0
total_frames = 0
width = 0
height = 0
auto_infer_interval = 1
skeleton_ov = 1.0



def open_video(idx):
    global cap, frame_idx, fps, total_frames, width, height, VIDEO_PATH
    global auto_infer_interval, skeleton_ov, seek_request
    VIDEO_PATH = video_list[idx]
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {VIDEO_PATH}")
        exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = fps if fps > 1 else TARGET_MODEL_FPS
    auto_infer_interval = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
    model_input_fps = source_fps / auto_infer_interval
    skeleton_ov = max(0.6, height / 720.0)
    # 依 SEEK_FRAME 決定起始幀（夾在合法範圍內），並把影片位置移過去。
    if total_frames > 0:
        frame_idx = max(0, min(int(SEEK_FRAME), total_frames - 1))
    else:
        frame_idx = max(0, int(SEEK_FRAME))
    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    # 進度條上限對齊新影片，並把游標移到起始幀；清掉可能殘留的跳轉請求。
    try:
        cv2.setTrackbarMax(TRACKBAR_NAME, WIN_NAME, max(1, total_frames - 1))
    except cv2.error:
        pass
    sync_trackbar(frame_idx)
    seek_request = None
    print(
        f"[Info] Video: {VIDEO_PATH}  {width}x{height}  {fps:.2f} FPS  {total_frames} frames"
        f"  | model_fps≈{model_input_fps:.2f} (target={TARGET_MODEL_FPS:.0f}, step={auto_infer_interval})"
        f"  | start_frame={frame_idx}  auto_skip={AUTO_FRAME_SKIP}"
    )

frame_idx = 0
# save_idx 已於上方自動取得
step_mode = True  # 預設逐幀
last_result = None  # 保存最後一次推論結果，避免閃爍
last_infer_frame_idx = -1  # 記錄上次推論的 frame_idx，避免重複推論
cached_frame = None  # 快取當前 frame（step 模式用）
last_frame_auto = None  # auto 模式保存最後一幀原始畫質（供 S 鍵儲存）
last_shown_frame = None  # 最近一次顯示的原始畫面（Auto／Step 都記），影片播完時用來停在最後一幀
last_shown_idx = 0       # 上面那一幀的 frame_idx
last_shown_video = -1    # 上面那一幀屬於哪一部影片（video_idx），換片後就不會誤用舊影片的畫面
frame_step = 1  # 每次移動的幀數（Z 增加，X 減少）
playback_speed = DEFAULT_PLAYBACK_SPEED  # Auto 模式倍速（1~5 鍵切換）
pending_key = 255       # 節流等待期間收到的按鍵，暫存到下一輪處理（255 = 無）
seek_request = None     # 進度條拖曳 / t 鍵請求跳轉的目標幀（None = 無）
_trackbar_syncing = False  # 程式碼主動移動進度條時擋掉回呼，避免自我觸發
_last_display_ts = 0.0  # 0.5x 節流用的上次顯示時間戳




# ==================== 顯示窗口限制 ====================
# 改 DISPLAY_RESOLUTION 切換顯示上限："720p" → 不超過 1280x720，"1080p" → 不超過 1920x1080。
# 畫面只依「原始長寬比」等比縮小到這個上限內（只縮不放、不補黑邊、不拉伸），
# 視窗用 WINDOW_AUTOSIZE 自動貼合縮小後的畫面。
DISPLAY_RESOLUTION = "1080p"
_DISPLAY_RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
import os as _os
_env_resolution = _os.getenv("DISPLAY_RESOLUTION", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫上面的預設；空白＝沿用預設
if _env_resolution:
    if _env_resolution in _DISPLAY_RESOLUTION_PRESETS:
        DISPLAY_RESOLUTION = _env_resolution
    else:
        print(f"⚠ 環境變數 DISPLAY_RESOLUTION={_env_resolution!r} 無效（只接受 {list(_DISPLAY_RESOLUTION_PRESETS)}），沿用預設 {DISPLAY_RESOLUTION}")
DISPLAY_SIZE = _DISPLAY_RESOLUTION_PRESETS[DISPLAY_RESOLUTION]  # 視窗顯示解析度上限（寬, 高）

def cur_src_fps():
    """目前影片的有效 FPS（讀不到時退回 TARGET_MODEL_FPS）。"""
    return fps if fps > 1 else TARGET_MODEL_FPS

def fmt_time(sec):
    """秒數 → mm:ss.s 字串。"""
    sec = max(0.0, float(sec))
    m, s = divmod(sec, 60)
    return f"{int(m):d}:{s:04.1f}"

def cycle_speed(direction):
    """direction: +1 加大 / -1 減小快轉倍率（在 SPEED_STEPS 階梯間單次切換）。"""
    global playback_speed
    try:
        i = SPEED_STEPS.index(playback_speed)
    except ValueError:
        i = SPEED_STEPS.index(DEFAULT_PLAYBACK_SPEED)
    new_i = max(0, min(len(SPEED_STEPS) - 1, i + direction))
    playback_speed = SPEED_STEPS[new_i]
    if new_i == i:
        edge = "最高" if direction > 0 else "最低"
        print(f"[Speed] 已達{edge}倍率，維持 {playback_speed:g}x")
    else:
        print(f"[Speed] 快轉倍率 = {playback_speed:g}x")

# 實際使用的顯示上限（寬, 高）＝DISPLAY_SIZE，第一次顯示前會依螢幕工作區校正（見 _fit_cap_to_screen）：
# 視窗的標題列、邊框、進度條都算在視窗外框裡，畫面直接開到 DISPLAY_SIZE 大會讓視窗底部／右側超出螢幕
# （1080p 時最明顯：畫面 1080 高 + 外框 100 多像素 > 螢幕高度，下方被裁掉）。
_display_cap = list(DISPLAY_SIZE)
_cap_fitted = False
_hud_extra_h = 0  # compose_with_hud() 在影片畫面上下加的資訊色塊總高度；它是縮小之後才加上去的，要從上限裡預留
_screen_limited = False  # 螢幕放不下 DISPLAY_SIZE（上限被校正縮小）時才為 True；只有這時才預留 HUD 高度／整張再縮


def _fit_cap_to_screen(w, h):
    """第一次顯示前執行一次：量視窗外框（標題列＋邊框＋進度條）實際佔多少像素，把顯示上限縮到
    「螢幕工作區（不含工作列）扣掉外框」以內，並把視窗移到工作區左上角。w, h＝這支影片要顯示的畫面原始大小。
    做法：用「跟實際顯示同寬、但高度很小」的黑圖去量——外框大小跟圖高無關、只跟寬度有關，而且這種小視窗不會被
    作業系統因為超出螢幕而強制縮小，量得準。Windows 限定；量不到（非 Windows、找不到視窗）就維持原上限。"""
    global _cap_fitted
    _cap_fitted = True
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        work = wintypes.RECT()
        user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work), 0)  # SPI_GETWORKAREA
        work_w, work_h = work.right - work.left, work.bottom - work.top
        probe_h = 300
        chrome = (16, 100)  # 量不到時的保守起始估計（邊框 16、標題列＋邊框＋進度條約 100）
        for _ in range(3):  # 外框高度會隨寬度略有變動，最多修正幾次就收斂
            sc = min(_display_cap[0] / w, _display_cap[1] / h, 1.0)
            probe_w = max(320, int(round(w * sc)))
            cv2.imshow(WIN_NAME, np.zeros((probe_h, probe_w, 3), dtype=np.uint8))
            hwnd = user32.FindWindowW(None, WIN_NAME)
            if not hwnd:
                return
            rect, client = wintypes.RECT(), wintypes.RECT()
            for _ in range(60):  # 等視窗貼合這張圖（客戶區寬度＝圖寬）才量，量太早會量到預設視窗大小
                cv2.waitKey(10)
                user32.GetClientRect(hwnd, ctypes.byref(client))
                if client.right == probe_w:
                    break
            else:
                break
            time.sleep(0.25)  # 貼合寬度後進度條列還會再調整一次高度
            cv2.waitKey(1)
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            new_chrome = ((rect.right - rect.left) - probe_w, (rect.bottom - rect.top) - probe_h)
            new_cap = (min(DISPLAY_SIZE[0], work_w - new_chrome[0]), min(DISPLAY_SIZE[1], work_h - new_chrome[1]))
            converged = new_chrome == chrome
            chrome = new_chrome
            _display_cap[0], _display_cap[1] = max(320, new_cap[0]), max(240, new_cap[1])
            if converged:
                break
        if _display_cap[0] < DISPLAY_SIZE[0] or _display_cap[1] < DISPLAY_SIZE[1]:
            global _screen_limited
            _screen_limited = True
            cv2.moveWindow(WIN_NAME, work.left, work.top)
            print(f"[顯示] 視窗外框佔 {chrome[0]}x{chrome[1]}px，為了不超出螢幕，畫面上限自動縮到 {_display_cap[0]}x{_display_cap[1]}")
    except Exception as e:  # noqa: BLE001 — 校正失敗不影響播放
        print(f"[顯示] 無法依螢幕校正視窗大小（{e}），維持 {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]} 上限")


def shrink_to_cap(img):
    """imshow 前的最後一道保險：整張畫面（含 HUD 色塊）超過顯示上限就等比縮小，避免視窗超出螢幕。"""
    if not _screen_limited:
        return img
    h, w = img.shape[:2]
    s = min(_display_cap[0] / w, _display_cap[1] / h, 1.0)
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def fit_for_display(img):
    """依原始長寬比等比縮小到不超過 _display_cap（只縮不放、不補黑邊）。
    回傳 (縮小後畫面, 縮小後高度)。第一次呼叫時會先依螢幕校正 _display_cap。"""
    h, w = img.shape[:2]
    if not _cap_fitted:
        _fit_cap_to_screen(w, h)
    reserve = _hud_extra_h if _screen_limited else 0  # 螢幕夠大（例如 720p）時維持原本行為，不預留
    s = min(_display_cap[0] / w, max(240, _display_cap[1] - reserve) / h, 1.0)
    if s >= 1.0:
        return img, h
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), nh

# ==================== 建立顯示視窗 ====================
WIN_NAME = "Cat Pose Inference"
# AUTOSIZE：視窗自動貼合 imshow 畫面（我們已自行縮到 720/1080 內），
# 後端不會再縮放/拉伸畫面，比例一定正確。
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

# ---- 視窗頂端進度條：拖曳即時 seek（兩種模式都可用）----
TRACKBAR_NAME = "Frame"

def _on_trackbar(pos):
    global seek_request
    if _trackbar_syncing:
        return
    seek_request = pos

cv2.createTrackbar(TRACKBAR_NAME, WIN_NAME, 0, 1, _on_trackbar)

def sync_trackbar(idx):
    """把進度條移到 idx（擋掉回呼，避免自我觸發跳轉）。"""
    global _trackbar_syncing
    hi = max(1, total_frames - 1)
    pos = max(0, min(int(idx), hi))
    _trackbar_syncing = True
    try:
        cv2.setTrackbarPos(TRACKBAR_NAME, WIN_NAME, pos)
    except cv2.error:
        pass
    finally:
        _trackbar_syncing = False

def prompt_seek():
    """按 t：於終端輸入目標位置，回傳夾好範圍的幀號（None = 取消）。"""
    src_fps = fps if fps > 1 else TARGET_MODEL_FPS
    try:
        raw = input(
            f"[Seek] 跳到（秒數加 s，例 12.5s；或直接輸入幀號 0~{max(0, total_frames - 1)}）> "
        ).strip()
    except EOFError:
        return None
    if not raw:
        return None
    try:
        if raw.lower().endswith("s"):
            tgt = int(round(float(raw[:-1]) * src_fps))
        else:
            tgt = int(float(raw))
    except ValueError:
        print("[Seek] 輸入無效，取消")
        return None
    if total_frames > 0:
        tgt = max(0, min(tgt, total_frames - 1))
    else:
        tgt = max(0, tgt)
    print(f"[Seek] → 第 {tgt} 幀（約 {tgt / src_fps:.2f}s）")
    return tgt

END_PROBE_MAX_BACK = 60  # 讀不到幀且離標示的結尾在這個幀數內，才當成「總幀數偏多」處理

def find_last_readable_frame(from_idx, max_back=END_PROBE_MAX_BACK):
    """從 from_idx 往前一幀一幀找最近讀得到的幀，回傳 (幀號, 畫面)；找不到回傳 None。"""
    for back in range(1, max_back + 1):
        i = from_idx - back
        if i < 0:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            return i, fr
    return None

def resize_window_to_video():
    """WINDOW_AUTOSIZE：視窗自動貼合縮小後的畫面，無需手動調整。"""
    pass

def apply_video_switch(initial_key):
    """
    連續換片：先以 waitKey(30) 收集所有排隊中的 1 / 2 按鍵，
    再一次完成切換，避免 imshow/open_video/resizeWindow 的
    Windows 訊息幫浦中途吃掉後續按鍵。
    回傳 True 代表確實切換了影片。
    """
    global video_idx, frame_idx, last_result, last_frame_auto, last_infer_frame_idx
    delta = 0
    key = initial_key
    while True:
        if key == ord('1'):
            delta -= 1
        elif key == ord('2'):
            delta += 1
        else:
            break
        key = cv2.waitKey(30) & 0xFF  # 非阻塞，30ms 內無新鍵則停止收集
    if delta == 0:
        return False
    # 循環播放清單：最後一部按「下一部」跳回第一部，第一部按「上一部」跳到最後一部。
    new_idx = (video_idx + delta) % len(video_list)
    if new_idx == video_idx:
        print(f"[Video] {'+' if delta > 0 else ''}{delta}：只有一部影片 ({video_idx+1}/{len(video_list)})")
        return False
    print(f"[Video] {'+' if delta > 0 else ''}{delta} → 影片 {new_idx+1}/{len(video_list)}")
    video_idx = new_idx
    open_video(video_idx)          # 內部已依 SEEK_FRAME 設好 frame_idx 與影片位置
    resize_window_to_video()
    print_mode()
    last_result = None
    last_infer_frame_idx = -1
    last_frame_auto = None
    return True

def print_mode():
    mode = "Step" if step_mode else "Auto"
    print(
        f"\n[操作說明] S=儲存影像  Z=增加步長  X=減少步長 (當前步長={frame_step})  "
        f"1=上一部影片  2=下一部影片  Space=切換模式({mode})  ESC=離開\n"
        f"[通用] [ =往回跳 {JUMP_FRAMES} 幀   ] =往前跳 {JUMP_FRAMES} 幀   "
        f"t=輸入跳轉(秒數/幀號)   視窗頂端進度條可拖曳即時 seek\n"
        f"[Step模式] D=下一幀  A=上一幀\n"
        f"[Auto模式] 目標推論FPS={TARGET_MODEL_FPS:.0f}，每 {auto_infer_interval} 幀推論一次   "
        f"= 或 + =加大倍率   - =減小倍率  階梯 {'/'.join(f'{s:g}' for s in SPEED_STEPS)}x (目前 {playback_speed:g}x)   "
        f". =快轉加速   , =快轉減速 (當前額外跳 {AUTO_FRAME_SKIP} 幀/顯示幀)\n"
        f"[起始幀] SEEK_FRAME={SEEK_FRAME}   [顯示上限] {DISPLAY_SIZE[0]}x{DISPLAY_SIZE[1]} (DISPLAY_RESOLUTION={DISPLAY_RESOLUTION}, 依原比例等比縮小)\n"
    )

print_mode()
open_video(video_idx)
resize_window_to_video()

# ==================== 骨架顏色與連結（統一從 constants.py import，
# 該檔案直接複製自 paper/cat_monitoring_system/utils/constants.py 目前使用中
# 的版本；原本這裡 _KP_COLORS 索引 3/4/5 是舊的黃綠色系版本，換成共用模組後
# 會變成色相分離的黃/洋紅/綠，畫面上關鍵點顏色會有感知得到的變化） ====================
KP_CONF_THRES = 0.5  # 關鍵點（kp）信心門檻：低於此值的關鍵點不畫（不是 bbox 偵測框門檻，bbox 見 YOLO_CONF_THRESHOLD）
_env_kp_conf = _os.getenv("CAT_MONITORING_KP_CONF_THRES", "").strip()  # 設定視窗「⚙ 額外設定」可覆寫；變數名同 config.py 的 AnomalyDetectionConfig.KP_CONF_THRES
if _env_kp_conf:
    try:
        KP_CONF_THRES = float(_env_kp_conf)
    except ValueError:
        print(f"⚠ 環境變數 CAT_MONITORING_KP_CONF_THRES={_env_kp_conf!r} 不是數字，沿用預設 {KP_CONF_THRES}")
# 啟動時印出實際採用的門檻，從設定視窗覆寫後可以直接在終端面板確認有沒有套用
print(f"[Info] 信心門檻：bbox（偵測框）={YOLO_CONF_THRESHOLD}  kp（關鍵點）={KP_CONF_THRES}")
BLUE = (255, 0, 0)


def draw_styled_skeleton(frame, kpts, kpt_conf, ov, conf_thresh=KP_CONF_THRES):
    """套用跟貓咪辨識系統一致的骨架視覺風格（見
    paper/cat_monitoring_system/processors/visualizer.py Visualizer.draw()）：
    固定 2px 線寬、無反鋸齒，關鍵點為單一 3px 實心圓（無黑色外框），不隨解析度縮放。
    ov 保留參數位置僅為呼叫端相容，不再影響繪製尺寸。
    """
    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        if a >= len(kpts) or b >= len(kpts):
            continue
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, 2)

    for i in range(min(17, len(kpts))):
        if float(kpt_conf[i]) <= conf_thresh:
            continue
        cx, cy = int(kpts[i][0]), int(kpts[i][1])
        col = _KP_COLORS[i] if i < len(_KP_COLORS) else (200, 200, 200)
        cv2.circle(frame, (cx, cy), 3, col, -1)

# ==================== 绘制函数 ====================
def draw_result(frame, result):
    disp_frame = frame.copy()

    # BBox：所有偵測到的貓
    if result and result.boxes is not None and len(result.boxes.xyxy) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names.get(cls_id, f"id_{cls_id}")
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), BLUE, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(disp_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BLUE, -1)
            cv2.putText(disp_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 骨架：第一隻貓
    if result and result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts     = result.keypoints.xy[0].cpu().numpy()
        kpt_conf = result.keypoints.conf[0].cpu().numpy()
        draw_styled_skeleton(disp_frame, kpts, kpt_conf, ov=skeleton_ov)

    return disp_frame

def _draw_hud_line(img, text, x, y, font_scale, thickness):
    """畫一行黑色描邊＋白字（跟原本 draw_text 的可讀性處理相同），只是現在畫在
    HUD 色塊上而不是直接疊在影片畫面上——見 compose_with_hud() 的說明。"""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# HUD（影片路徑/幀數/儲存數/操作提示）色塊底色：接近純黑但不是純黑，跟視窗其餘
# 深色 UI 一致；刻意不用純黑（0,0,0）——純黑背景跟黑貓的毛色太接近，色塊邊緣
# 反而不容易一眼認出「這裡是資訊列、不是影片畫面」。
HUD_BG = (28, 28, 28)


def compose_with_hud(disp_frame, scale, top_lines, bottom_lines):
    """原本 4 行影片資訊＋2 行操作提示是直接疊在影片畫面左上/左下角，貓咪剛好
    站在那個角落時，文字會蓋住牠的骨架/輪廓，影響用肉眼判斷姿態/推論品質準不準
    ——這是這支工具本來的用途（人工核對推論結果、按 S 存標註用的圖）。

    改成上下各加一條「HUD 色塊」，跟影片畫面上下拼接成一張更高的新畫面：文字
    永遠畫在色塊裡，不管貓咪站在畫面哪個角落，影片內容本身完全不會被文字蓋到。
    `cv2.namedWindow(..., cv2.WINDOW_AUTOSIZE)` 會自動貼合這張變高的畫面，不用
    另外處理視窗縮放；進度條是 `cv2.createTrackbar` 建立的原生 HighGUI 元件，
    跟畫面像素座標無關，也不受影響。

    `top_lines`/`bottom_lines`：[(文字, font_scale, thickness), ...]，由上到下
    依序排列；`font_scale`/`thickness` 是「未乘上 scale」的原始值（跟原本
    `draw_text()` 的參數語意相同），這裡會統一乘上 `scale` 才實際拿去畫——
    呼叫端不用自己重複算一次。每一行的色塊高度用 `cv2.getTextSize()` 實際
    量測文字本身的高度（含 baseline）決定，不是猜一個固定數字，避免文字被
    列高裁切。"""
    h, w = disp_frame.shape[:2]
    # 版面資訊是輔助用的、次要於影片畫面本身——行距/內距盡量收窄，把畫面盡量
    # 讓給實際的影片內容，不是喧賓奪主佔掉一大塊。
    line_gap = int(4 * scale)
    pad = int(6 * scale)
    x = int(14 * scale)

    def _scaled(lines):
        return [(text, fscale * scale, max(1, int(thick * scale))) for text, fscale, thick in lines]

    def _band_height(lines):
        if not lines:
            return 0
        total = pad * 2 + line_gap * (len(lines) - 1)
        for text, font_scale, thickness in lines:
            (_tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            total += th + baseline
        return total

    def _draw_band(canvas, lines, y0):
        y = y0 + pad
        for text, font_scale, thickness in lines:
            (_tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            y += th
            _draw_hud_line(canvas, text, x, y, font_scale, thickness)
            y += baseline + line_gap
        return y

    top_lines = _scaled(top_lines)
    bottom_lines = _scaled(bottom_lines)

    top_h = _band_height(top_lines)
    bottom_h = _band_height(bottom_lines)

    canvas = np.full((top_h + h + bottom_h, w, 3), HUD_BG, dtype=disp_frame.dtype)
    canvas[top_h:top_h + h, :] = disp_frame
    _draw_band(canvas, top_lines, 0)
    _draw_band(canvas, bottom_lines, top_h + h)
    global _hud_extra_h
    _hud_extra_h = top_h + bottom_h  # 之後 fit_for_display() 縮影片畫面時要預留這段高度
    return canvas

while True:
    if not cap.isOpened():
        break

    if step_mode:
        # 逐幀模式下，僅在按鍵時才換幀，畫面每次都即時更新
        step_break_outer = False
        while True:
            # 進度條拖曳 / t 鍵請求的跳轉：先套用再渲染
            if seek_request is not None:
                frame_idx = max(0, min(int(seek_request), max(0, total_frames - 1)))
                seek_request = None
            # 只有當 frame_idx 改變時才需要重新讀取和推論
            if frame_idx != last_infer_frame_idx:
                # 注意：cap.set() random access 在 h264/h265 影片很慢（需重新解碼 GOP）
                # 如果需要大量標註，建議先用 ffmpeg 將影片轉為 image sequence
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret and total_frames - frame_idx <= END_PROBE_MAX_BACK:
                    # 讀不到、又接近影片標示的結尾：多半是檔案的總幀數 metadata 比實際多（VFR／錄影軟體常見），
                    # 往回找真正的最後一幀並停在那，不要繞回第 0 幀
                    real_last = find_last_readable_frame(frame_idx)
                    if real_last is not None:
                        frame_idx, frame = real_last
                        ret = True
                        total_frames = frame_idx + 1  # 以實際讀到的為準，之後 A/D、進度條範圍都跟著正確
                        try:
                            cv2.setTrackbarMax(TRACKBAR_NAME, WIN_NAME, max(1, total_frames - 1))
                        except cv2.error:
                            pass
                        print(f"[End] 已到影片最後一幀（檔案標示的總幀數偏多，已修正為 {total_frames} 幀）")
                if not ret:
                    frame_idx = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                cached_frame = frame.copy()
                last_result = infer(frame)
                last_infer_frame_idx = frame_idx
            else:
                # frame_idx 未改變，直接使用快取的 frame 和 result
                frame = cached_frame
            
            last_shown_frame, last_shown_idx, last_shown_video = frame, frame_idx, video_idx  # 自動模式播完時要停在這一幀
            # 畫標註 → 依原比例等比縮小到 720/1080 內（保存時仍用 cached_frame 原始畫質）
            disp_frame, _disp_h = fit_for_display(draw_result(frame, last_result))
            scale = min(_disp_h / 720.0, 1.0)
            video_info = f"Video: [{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}"
            _sfps = cur_src_fps()
            mode_str = "Step" if step_mode else "Auto"
            infer_text = f"Infer: target {TARGET_MODEL_FPS:.0f} FPS / every {auto_infer_interval} frame(s)   Speed {playback_speed:g}x"
            help_text = f"Mode: {mode_str}  S=Save  D/A=Frame  Z/X=Step {frame_step}  [ ]=Jump{JUMP_FRAMES}  t=Seek  1/2=Video  -/+=Speed  Space=Switch  ESC=Quit"
            # 資訊列改畫在影片畫面外的 HUD 色塊，不再疊在畫面上蓋住貓咪——見
            # compose_with_hud() 的說明。
            disp_frame = compose_with_hud(
                disp_frame, scale,
                top_lines=[
                    (video_info, 0.6, 1),
                    (f"Frame: {frame_idx+1}/{total_frames}   {fmt_time(frame_idx/_sfps)} / {fmt_time((total_frames-1)/_sfps)}", 0.7, 2),
                    (f"Saved: {save_idx-1}   Step: {frame_step}   Speed: {playback_speed:g}x", 0.55, 1),
                ],
                bottom_lines=[
                    (infer_text, 0.45, 1),
                    (help_text, 0.45, 1),
                ],
            )
            cv2.imshow(WIN_NAME, shrink_to_cap(disp_frame))
            sync_trackbar(frame_idx)
            # 阻塞等待按鍵，但同時讓進度條拖曳能中斷（拖曳會設定 seek_request）
            key = 255
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key != 255 or seek_request is not None:
                    break
            if seek_request is not None:
                continue

            # 合併 event queue 裡排隊的移動鍵：長按 A/D（或 [ ]）時，OS 的
            # 按鍵重複會塞滿緩衝區，逐一處理會導致「放開後畫面還在過幀」而且每幀
            # 都要 seek+推論而卡頓。這裡把排隊的移動鍵一次算成總位移，只渲染一次。
            nav_delta = 0
            while key != 255:
                if key in (ord('d'), ord('D')):
                    nav_delta += frame_step
                elif key in (ord('a'), ord('A')):
                    nav_delta -= frame_step
                elif key == ord(']'):
                    nav_delta += JUMP_FRAMES
                elif key == ord('['):
                    nav_delta -= JUMP_FRAMES
                else:
                    break  # 非移動鍵：保留給下方 elif 鏈處理
                key = cv2.waitKey(1) & 0xFF
            if nav_delta != 0:
                new_idx = max(0, min(frame_idx + nav_delta, max(0, total_frames - 1)))
                if new_idx != frame_idx:
                    frame_idx = new_idx
                    print(f"[Frame] {'+' if nav_delta > 0 else ''}{nav_delta} → {frame_idx+1}/{total_frames}  ({fmt_time(frame_idx/cur_src_fps())})")
                else:
                    edge = "最後一幀" if nav_delta > 0 else "第一幀"
                    print(f"[Frame] {'+' if nav_delta > 0 else ''}{nav_delta}：已在{edge} ({frame_idx+1}/{total_frames})")
                if key == 255:
                    continue  # 純導覽，重新渲染

            if key == ord('z') or key == ord('Z'):
                frame_step += 1
                print(f"[Step] 每次移動步長: {frame_step}")
            elif key == ord('x') or key == ord('X'):
                frame_step = max(1, frame_step - 1)
                print(f"[Step] 每次移動步長: {frame_step}")
            elif key == ord('1') or key == ord('2'):
                apply_video_switch(key)
                step_break_outer = True
                break
            elif key == ord('=') or key == ord('+'):
                cycle_speed(+1)
            elif key == ord('-'):
                cycle_speed(-1)
            elif key == ord('.'):
                AUTO_FRAME_SKIP += 1
                print(f"[Auto] 快轉：每幀多跳 {AUTO_FRAME_SKIP} 幀")
            elif key == ord(','):
                AUTO_FRAME_SKIP = max(0, AUTO_FRAME_SKIP - 1)
                print(f"[Auto] 快轉：每幀多跳 {AUTO_FRAME_SKIP} 幀")
            elif key == ord('t') or key == ord('T'):
                print("[Seek] t：等待終端輸入…")
                tgt = prompt_seek()
                if tgt is not None:
                    seek_request = tgt
                else:
                    print("[Seek] 取消跳轉")
            elif key == 27:
                print("[Exit] Quit.")
                cap.release()
                cv2.destroyAllWindows()
                log_path = output_dir / "saved_log.txt"
                with open(log_path, "w", encoding="utf-8") as f:
                    for vpath, imgs in saved_log.items():
                        f.write(f"{vpath}\n")
                        for img in imgs:
                            f.write(f"    {img}\n")
                        f.write("\n")
                print(f"\n[Complete] 共儲存 {save_idx-1} 張影像於: {OUTPUT_DIR}")
                print(f"[Log] 已產生日誌: {log_path}")
                exit(0)
            elif key == ord('s') or key == ord('S'):
                img_name = IMG_NAME_FORMAT.format(save_idx)
                img_name_png = Path(img_name).with_suffix('.png')
                save_path = output_dir / img_name_png
                # 保存原始分辨率的 frame，不受显示缩放影响
                cv2.imwrite(str(save_path), cached_frame)
                print(f"[Saved] {img_name_png} ({width}x{height})")
                vpath = VIDEO_PATH
                if vpath not in saved_log:
                    saved_log[vpath] = []
                saved_log[vpath].append(str(img_name_png))
                save_idx += 1
            elif key == 32:  # Space
                step_mode = not step_mode
                if not step_mode:
                    # 逐幀模式讀完 frame_idx 後，影片讀取位置已在 frame_idx+1；不校正的話自動模式讀到的是
                    # frame_idx+1 這一幀，卻標成 frame_idx（畫面與幀號差 1，播到最後時停住的幀號也會差 1）
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                print(f"[Mode] Space → {'Step' if step_mode else 'Auto'}")
                print_mode()
                break
            elif key != 255:
                print(f"[Key] 未綁定按鍵: {chr(key) if 32 <= key < 127 else key}")
            # 其他按鍵不動，繼續顯示當前 frame
        if step_break_outer:
            continue

    else:
        # 自動模式：先偵測按鍵（捕捉上一幀推論期間累積的按鍵事件），再處理畫面
        # cv2.imshow 在 Windows 上有時會消耗按鍵事件，因此在 imshow 之前先 poll 一次
        if pending_key != 255:      # 0.5x 節流等待期間收到的按鍵，優先處理
            key = pending_key
            pending_key = 255
        else:
            key = cv2.waitKey(1) & 0xFF

        # ---- 按鍵處理 ----
        if key == ord('1') or key == ord('2'):
            apply_video_switch(key)
            continue
        elif key == ord('=') or key == ord('+'):
            cycle_speed(+1)
            continue
        elif key == ord('-'):
            cycle_speed(-1)
            continue
        elif key == ord('t') or key == ord('T'):
            print("[Seek] t：等待終端輸入…")
            tgt = prompt_seek()
            if tgt is not None:
                frame_idx = tgt
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                last_result = None
                sync_trackbar(frame_idx)
                print(f"[Seek] → 第 {frame_idx} 幀  ({fmt_time(frame_idx/cur_src_fps())})")
            else:
                print("[Seek] 取消跳轉")
            continue
        elif key == 27:
            print("[Exit] Quit.")
            break
        elif key == ord('s') or key == ord('S'):
            if last_frame_auto is None:
                print("[Saved] 尚無可儲存的畫面")
            if last_frame_auto is not None:
                img_name = IMG_NAME_FORMAT.format(save_idx)
                img_name_png = Path(img_name).with_suffix('.png')
                save_path = output_dir / img_name_png
                cv2.imwrite(str(save_path), last_frame_auto)
                print(f"[Saved] {img_name_png} ({width}x{height})")
                vpath = VIDEO_PATH
                if vpath not in saved_log:
                    saved_log[vpath] = []
                saved_log[vpath].append(str(img_name_png))
                save_idx += 1
        elif key == ord('z') or key == ord('Z'):
            frame_step += 1
            print(f"[Step] 每次移動步長: {frame_step}")
        elif key == ord('x') or key == ord('X'):
            frame_step = max(1, frame_step - 1)
            print(f"[Step] 每次移動步長: {frame_step}")
        elif key == ord('.'):  # 快轉加速
            AUTO_FRAME_SKIP += 1
            print(f"[Auto] 快轉：每幀多跳 {AUTO_FRAME_SKIP} 幀 (≈{AUTO_FRAME_SKIP + 1}x)")
        elif key == ord(','):  # 快轉減速
            AUTO_FRAME_SKIP = max(0, AUTO_FRAME_SKIP - 1)
            print(f"[Auto] 快轉：每幀多跳 {AUTO_FRAME_SKIP} 幀 (≈{AUTO_FRAME_SKIP + 1}x)")
        elif key == ord('['):  # 往回跳 JUMP_FRAMES
            frame_idx = max(0, frame_idx - JUMP_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_result = None
            print(f"[Jump] [ -{JUMP_FRAMES} → 第 {frame_idx} 幀  ({fmt_time(frame_idx/cur_src_fps())})")
        elif key == ord(']'):  # 往前跳 JUMP_FRAMES
            frame_idx = min(max(0, total_frames - 1), frame_idx + JUMP_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_result = None
            print(f"[Jump] ] +{JUMP_FRAMES} → 第 {frame_idx} 幀  ({fmt_time(frame_idx/cur_src_fps())})")
        elif key == 32:  # Space
            step_mode = not step_mode
            print(f"[Mode] Space → {'Step' if step_mode else 'Auto'}")
            print_mode()
            continue
        elif key != 255:
            print(f"[Key] 未綁定按鍵: {chr(key) if 32 <= key < 127 else key}")

        # ---- 進度條拖曳跳轉 ----
        if seek_request is not None:
            frame_idx = max(0, min(int(seek_request), max(0, total_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_result = None
            seek_request = None
            sync_trackbar(frame_idx)
            print(f"[Seek] 進度條 → 第 {frame_idx} 幀  ({fmt_time(frame_idx/cur_src_fps())})")

        # ---- 讀取與推論 ----
        ret, frame = cap.read()
        if not ret:
            # 影片播完：不自動切下一部、也不結束程式，停在這部影片的最後一幀（改成 Step 模式）。
            # 之後可以 A／[ 往回看、S 存圖、1／2 換片、Space 回 Auto（會馬上又停回最後一幀）、ESC 離開。
            if last_shown_frame is not None and last_shown_video == video_idx:
                frame_idx = last_shown_idx
                cached_frame = last_shown_frame.copy()
                last_result = infer(cached_frame)  # 用最後一幀重新推論一次，骨架與畫面一致
                last_infer_frame_idx = frame_idx   # 逐幀模式看到相同 frame_idx 就直接用快取，不再去讀影片
            else:
                # 這部影片一幀都沒顯示過（例如一開始就跳到影片尾端之後）：交給逐幀模式讀最後一幀
                frame_idx = max(0, total_frames - 1)
                last_infer_frame_idx = -1
            step_mode = True
            print(f"[End] 影片播放完畢，停在最後一幀（{frame_idx+1}/{total_frames}）："
                  f"A／[ 往回看  S 存圖  1／2 換片  ESC 離開")
            continue
        last_frame_auto = frame  # 保留原始畫質供 S 鍵儲存
        last_shown_frame, last_shown_idx, last_shown_video = frame, frame_idx, video_idx  # 播完時要停在這一幀
        # 抽幀（快轉 / >=2x 倍速）時每一顯示幀都推論；否則照 auto_infer_interval
        do_infer = (AUTO_FRAME_SKIP > 0) or (playback_speed >= 2.0) or (frame_idx % auto_infer_interval == 0)
        if do_infer:
            # 使用 640 與 step mode 一致，避免漏檢小目標
            last_result = infer(frame)
        # 使用最後一次推論結果繪制，避免閃爍 → 依原比例等比縮小到 720/1080 內
        # （保存時仍用 last_frame_auto 原始畫質）
        disp_frame, _disp_h = fit_for_display(draw_result(frame, last_result))
        scale = min(_disp_h / 720.0, 1.0)
        video_info = f"Video: [{video_idx+1}/{len(video_list)}] {os.path.basename(VIDEO_PATH)}"
        _sfps = cur_src_fps()
        mode_str = "Step" if step_mode else "Auto"
        ff_str = f" +FF{AUTO_FRAME_SKIP}" if AUTO_FRAME_SKIP > 0 else ""
        infer_text = f"Infer: target {TARGET_MODEL_FPS:.0f} FPS / every {auto_infer_interval} frame(s)   Speed {playback_speed:g}x{ff_str}"
        help_text = f"Mode: {mode_str}  S=Save  -/+=Speed  .,=FF  [ ]=Jump{JUMP_FRAMES}  t=Seek  1/2=Video  Space=Switch  ESC=Quit"
        # 資訊列改畫在影片畫面外的 HUD 色塊，不再疊在畫面上蓋住貓咪——見
        # compose_with_hud() 的說明。
        disp_frame = compose_with_hud(
            disp_frame, scale,
            top_lines=[
                (video_info, 0.6, 1),
                (f"Frame: {frame_idx+1}/{total_frames}   {fmt_time(frame_idx/_sfps)} / {fmt_time((total_frames-1)/_sfps)}", 0.7, 2),
                (f"Saved: {save_idx-1}   Step: {frame_step}   Speed: {playback_speed:g}x{ff_str}", 0.55, 1),
            ],
            bottom_lines=[
                (infer_text, 0.45, 1),
                (help_text, 0.45, 1),
            ],
        )
        cv2.imshow(WIN_NAME, shrink_to_cap(disp_frame))
        sync_trackbar(frame_idx)
        frame_idx += 1

        # ---- 倍速 / 快轉推進 ----
        # >=1x：用 cap.grab() 抽幀（不解碼，比 read 快）；0.5x：不抽幀，改用節流放慢。
        speed_skip = int(round(playback_speed)) - 1 if playback_speed >= 1.0 else 0
        for _ in range(speed_skip + AUTO_FRAME_SKIP):
            if not cap.grab():
                break
            frame_idx += 1

        # ---- 0.5x 節流：讓每個顯示幀停留更久（約達到 0.5x 實時速度）----
        if playback_speed < 1.0:
            _src_fps = fps if fps > 1 else TARGET_MODEL_FPS
            _target_period = (1.0 / _src_fps) / playback_speed
            _wait_s = _target_period - (time.perf_counter() - _last_display_ts)
            if _wait_s > 0:
                _k = cv2.waitKey(int(_wait_s * 1000)) & 0xFF
                if _k != 255:
                    pending_key = _k
            _last_display_ts = time.perf_counter()


cap.release()
cv2.destroyAllWindows()

# 輸出日誌
log_path = output_dir / "saved_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    for vpath, imgs in saved_log.items():
        f.write(f"{vpath}\n")
        for img in imgs:
            f.write(f"    {img}\n")
        f.write("\n")
print(f"\n[Complete] 共儲存 {save_idx-1} 張影像於: {OUTPUT_DIR}")
print(f"[Log] 已產生日誌: {log_path}")
