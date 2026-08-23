"""
BBox 佔畫面比例工具（雙模式）。

模式 1 — 即時影片/攝影機量測：
    對指定影片逐幀跑 YOLO-Pose 偵測，即時在畫面上顯示 YOLO 偵測到的貓咪
    bbox 面積佔整個畫面（原始影格）的百分比；同框有多隻貓時每隻都個別
    標示（追蹤鎖定的主要目標另外標註）。滑鼠左鍵在視窗上按住拖曳可畫出
    一個矩形，放開左鍵時定形，這個矩形也會即時算出佔畫面的百分比，當作
    額外資訊獨立顯示（同一時間只保留一個，按 x 清除）。

模式 2 — 資料夾批次分類（完整移植自 cat_pose/cat_pose_size_tier_report.py）：
    掃描一個資料夾內的所有靜態圖片，依 bbox 占比自動分類搬進
    small/tier_1~5/no 對應資料夾，每張圖上標註骨架、bbox 與占比（同框
    多隻貓也會個別標示，但分類判定固定用信心值最高的那一隻），輸出長條
    圖統計摘要與最小的前 5 張清單。改用主專案的 KeypointDetector（跟模式
    1 共用同一套偵測介面），不再用獨立的 ultralytics.YOLO 物件。

兩種模式的 bbox 占比計算方式完全一致（共用 `_bbox_area_pct()`）：保留
浮點精度（不截斷成 int 再相減）、結果夾在 [0, 100]% 範圍內，多貓同框時
個別計算並顯示。

模式 1 純即時顯示，不輸出檔案；模式 2 會搬移/複製圖片並輸出分類結果，
執行前請務必確認 MODE2_FOLDER / MODE2_OUTPUT_DIR 設定正確。
"""
import os
import shutil
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detectors.keypoint_detector import KeypointDetector

# ═══════════════════════════════════════════════════════
#  使用者設定區
# ═══════════════════════════════════════════════════════
RUN_MODE = 0  # 0: 啟動時互動選擇, 1: 模式1（即時影片/攝影機量測）, 2: 模式2（資料夾批次分類）

# ── 共用 YOLO 設定（模式 1 與模式 2 的偵測器行為預設值一致，僅模型路徑各自獨立）──
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5
INFERENCE_DEVICE = "cuda"

# ── 模式 1：即時影片/攝影機量測 ──────────────────────────────────────────
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\暫存\two_cat\1 (229).mp4"  # 0 = 預設攝影機（webcam）；改成影片檔路徑字串即可測試指定影片

# 若設定 TEST_VIDEO_PATH 環境變數，優先使用該影片路徑（覆蓋上面寫死的 VIDEO_PATH，僅影響模式 1）
_env_test_video = os.getenv("TEST_VIDEO_PATH", "").strip()
if _env_test_video:
    VIDEO_PATH = _env_test_video

YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_134.pt"
LOOP_PLAYBACK = True  # 影片播完是否自動從頭重播（webcam 模式下無影響）

DISPLAY_RESOLUTION = "1080p"  # "720p" 或 "1080p"，控制 GUI 視窗顯示解析度
_DISPLAY_RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
DISPLAY_SIZE = _DISPLAY_RESOLUTION_PRESETS[DISPLAY_RESOLUTION]  # 視窗顯示解析度（寬, 高）

WINDOW_NAME = "BBox Area Ratio"
MIN_DRAG_PX = 4  # 拖曳矩形至少要有這麼多像素寬/高才算數，避免手滑誤觸發

BBOX_COLOR = (0, 165, 255)             # 橘色：YOLO 鎖定追蹤中的主要目標 bbox
EXTRA_BBOX_COLOR = (60, 220, 220)      # 青色：同框內其餘偵測到的貓（非追蹤目標）
USER_RECT_COLOR = (60, 220, 60)        # 綠色：使用者自畫矩形（已定形）
USER_RECT_DRAG_COLOR = (60, 220, 220)  # 黃綠色：拖曳中、尚未放開左鍵

# ── 模式 2：資料夾批次分類（沿用 cat_pose/cat_pose_size_tier_report.py 原始設定）──
MODE2_YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_114.pt"  # 換成你要用的模型
MODE2_FOLDER = r"C:\Users\homec\OneDrive\圖片\Screenshots\screen_cat"           # 換成你的資料夾
MODE2_OUTPUT_DIR = r"C:\Users\homec\OneDrive\圖片\Screenshots\screen_cat\class"  # 預設輸出資料夾，請自行修改
MODE2_MIN_RATIO_PCT = 50.0  # 門檻（%）：低於此值視為 small，高於則分層 5 級 (50% ~ 100%)
MODE2_DIR_NO = os.path.join(MODE2_OUTPUT_DIR, "no")
MODE2_DIR_SMALL = os.path.join(MODE2_OUTPUT_DIR, "small")
# 5 個等級資料夾，每級占 (100 - MODE2_MIN_RATIO_PCT) / 5 的範圍
MODE2_TIER_DIRS = [os.path.join(MODE2_OUTPUT_DIR, f"tier_{i+1}") for i in range(5)]
MODE2_CHART_PATH = os.path.join(MODE2_OUTPUT_DIR, "summary.png")

# 模式 2 專用骨架視覺樣式（沿用原腳本 cat_pose_size_tier_report.py 的定義，
# 跟模式 1 的畫面顯示完全獨立、互不影響）
_M2_COLOR_HEAD = (255, 255, 0)
_M2_COLOR_BODY = (0, 255, 0)
_M2_COLOR_TAIL = (255, 0, 255)
_M2_COLOR_KPT = (0, 0, 255)
_M2_COLOR_LEFT_FRONT = (255, 0, 255)
_M2_COLOR_RIGHT_FRONT = (0, 255, 255)
_M2_COLOR_LEFT_HIND = (255, 165, 0)
_M2_COLOR_RIGHT_HIND = (0, 255, 0)
_M2_HEAD_LINKS = [(0, 1), (0, 2), (1, 2)]
_M2_BODY_LINKS = [(0, 3), (3, 4), (4, 5)]
_M2_TAIL_LINKS = [(5, 14), (14, 15), (15, 16)]
_M2_LEFT_FRONT_LINKS = [(3, 6), (6, 7)]
_M2_RIGHT_FRONT_LINKS = [(3, 8), (8, 9)]
_M2_LEFT_HIND_LINKS = [(5, 10), (10, 11)]
_M2_RIGHT_HIND_LINKS = [(5, 12), (12, 13)]
_M2_EXTRA_BOX_COLOR = (255, 200, 0)  # 藍色系，跟主要判定框（cat#1）的顏色區分
# ═══════════════════════════════════════════════════════


def resize_with_letterbox(image, target_size):
    """等比例縮放至目標尺寸（保留完整畫面，四周補黑邊）。
    回傳 (canvas, scale, pad_x, pad_y, content_w, content_h)：
    content_w/content_h 是縮放後「實際影片內容」在畫布內的寬高（不含黑邊），
    用來當使用者自畫矩形的百分比分母。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0, target_w, target_h

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, new_w, new_h


# ── 滑鼠拖曳狀態（模組層級，供 mouse callback 與主迴圈共用；僅模式 1 使用）──
_drag_start = None          # (x, y) 顯示座標，剛按下左鍵時的起點
_drag_current = None        # (x, y) 顯示座標，拖曳中的目前位置
_is_dragging = False
_user_rect_display = None   # 定形後的矩形，顯示座標 (x1, y1, x2, y2)；None = 目前沒有


def _on_mouse(event, x, y, flags, param):
    global _drag_start, _drag_current, _is_dragging, _user_rect_display
    if event == cv2.EVENT_LBUTTONDOWN:
        _user_rect_display = None  # 開始畫新的，舊的（若有）直接取代
        _drag_start = (x, y)
        _drag_current = (x, y)
        _is_dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and _is_dragging:
        _drag_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _is_dragging:
        _is_dragging = False
        _drag_current = (x, y)
        x1, x2 = sorted((_drag_start[0], _drag_current[0]))
        y1, y2 = sorted((_drag_start[1], _drag_current[1]))
        if (x2 - x1) >= MIN_DRAG_PX and (y2 - y1) >= MIN_DRAG_PX:
            _user_rect_display = (x1, y1, x2, y2)
        _drag_start = None
        _drag_current = None


def _clip_to_content(x1, y1, x2, y2, pad_x, pad_y, content_w, content_h):
    """把矩形裁切到「實際影片內容區域」範圍內（排除 letterbox 黑邊），
    只影響百分比計算，不影響畫面上實際畫出來的矩形位置。"""
    cx1 = max(pad_x, min(x1, pad_x + content_w))
    cx2 = max(pad_x, min(x2, pad_x + content_w))
    cy1 = max(pad_y, min(y1, pad_y + content_h))
    cy2 = max(pad_y, min(y2, pad_y + content_h))
    return cx1, cy1, cx2, cy2


def _rect_area_pct(x1, y1, x2, y2, pad_x, pad_y, content_w, content_h):
    cx1, cy1, cx2, cy2 = _clip_to_content(x1, y1, x2, y2, pad_x, pad_y, content_w, content_h)
    w = max(0, cx2 - cx1)
    h = max(0, cy2 - cy1)
    if content_w <= 0 or content_h <= 0:
        return 0.0
    return float(w * h) / float(content_w * content_h) * 100.0


def _bbox_area_pct(bbox, frame_w, frame_h):
    """bbox 面積佔整個原始影格（或圖片）的百分比：保留完整浮點精度（不截斷
    成 int 再相減），並夾在 [0, 100] 範圍內，避免 YOLO 回傳的座標萬一略微
    超出畫面邊界時顯示超過 100%。模式 1 與模式 2 共用同一套計算方式。"""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    frame_area = float(frame_w) * float(frame_h)
    if frame_area <= 0:
        return 0.0
    return max(0.0, min(100.0, area / frame_area * 100.0))


def _draw_hud(frame, lines):
    for i, text in enumerate(lines):
        ty = 32 + i * 32
        cv2.putText(frame, text, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════
#  模式 1：即時影片/攝影機量測
# ═══════════════════════════════════════════════════════
def run_mode1_video():
    global _user_rect_display

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片/攝影機來源: {VIDEO_PATH}")
        return

    print("初始化 YOLO-Pose...")
    keypoint_detector = KeypointDetector(
        YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    print("=" * 60)
    print("控制: q=退出  space=暫停/播放  x=清除自畫矩形")
    print("滑鼠左鍵在畫面上按住拖曳可畫一個矩形，放開左鍵定形；同一時間只保留一個。")
    print("=" * 60)

    paused = False
    last_frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if LOOP_PLAYBACK and isinstance(VIDEO_PATH, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            last_frame = frame

        frame = last_frame
        if frame is None:
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            continue

        orig_h, orig_w = frame.shape[:2]
        _kpts, _kpt_conf, bbox, _bbox_conf, all_instances = keypoint_detector.detect(
            frame, return_all_instances=True
        )

        show_frame, scale, pad_x, pad_y, content_w, content_h = resize_with_letterbox(frame, DISPLAY_SIZE)

        # ── YOLO bbox：多貓同框時，每一隻都個別算出佔「原始影格」面積的百分比並
        # 分別標示；目前追蹤鎖定的主要目標用橘色＋"tracked"，其餘用青色區分 ──
        bbox_pct_lines = []
        for idx, (_kpts_i, _kconf_i, bbox_i, bconf_i) in enumerate(all_instances or [], start=1):
            if bbox_i is None:
                continue
            pct_i = _bbox_area_pct(bbox_i, orig_w, orig_h)
            is_tracked = bbox is not None and np.array_equal(bbox_i, bbox)
            color = BBOX_COLOR if is_tracked else EXTRA_BBOX_COLOR

            bx1, by1, bx2, by2 = bbox_i
            dx1 = int(bx1 * scale) + pad_x
            dy1 = int(by1 * scale) + pad_y
            dx2 = int(bx2 * scale) + pad_x
            dy2 = int(by2 * scale) + pad_y
            cv2.rectangle(show_frame, (dx1, dy1), (dx2, dy2), color, 2, cv2.LINE_AA)
            conf_str = f" {float(bconf_i):.2f}" if bconf_i is not None else ""
            cv2.putText(show_frame, f"#{idx}{conf_str}", (dx1, max(0, dy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            tag = " (tracked)" if is_tracked else ""
            bbox_pct_lines.append(f"Cat #{idx}{tag}: {pct_i:5.2f}% of frame")

        # ── 使用者自畫矩形（拖曳中預覽 or 已定形）：百分比以「影片內容區域」為基準 ──
        user_pct = None
        if _is_dragging and _drag_start is not None and _drag_current is not None:
            x1, x2 = sorted((_drag_start[0], _drag_current[0]))
            y1, y2 = sorted((_drag_start[1], _drag_current[1]))
            cv2.rectangle(show_frame, (x1, y1), (x2, y2), USER_RECT_DRAG_COLOR, 2, cv2.LINE_AA)
            user_pct = _rect_area_pct(x1, y1, x2, y2, pad_x, pad_y, content_w, content_h)
        elif _user_rect_display is not None:
            x1, y1, x2, y2 = _user_rect_display
            cv2.rectangle(show_frame, (x1, y1), (x2, y2), USER_RECT_COLOR, 2, cv2.LINE_AA)
            user_pct = _rect_area_pct(x1, y1, x2, y2, pad_x, pad_y, content_w, content_h)

        # ── HUD 文字 ──
        lines = bbox_pct_lines if bbox_pct_lines else ["BBox: no cat detected"]
        if user_pct is not None:
            lines = lines + [f"Custom region: {user_pct:5.2f}% of frame"]
        _draw_hud(show_frame, lines)

        cv2.imshow(WINDOW_NAME, show_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸ 已暫停" if paused else "▶ 繼續播放")
        elif key == ord('x'):
            _user_rect_display = None
            print("✓ 已清除自畫矩形")

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
#  模式 2：資料夾批次分類（移植自 cat_pose/cat_pose_size_tier_report.py）
# ═══════════════════════════════════════════════════════
def _mode2_draw_skeleton(vis, kpts_xy, kpts_conf, box_thickness):
    """畫出骨架連線與關鍵點，只過濾絕對無效點（原點 (0,0)）。kpts_conf 目前
    未用於過濾（沿用原腳本行為：只看座標是否為原點），保留參數是為了跟
    KeypointDetector.detect() 的回傳介面一致。"""
    def is_valid(idx):
        if idx >= len(kpts_xy):
            return False
        x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
        return not (x == 0.0 and y == 0.0)

    def draw_links(links, color):
        for i, j in links:
            if is_valid(i) and is_valid(j):
                pt1 = (int(kpts_xy[i][0]), int(kpts_xy[i][1]))
                pt2 = (int(kpts_xy[j][0]), int(kpts_xy[j][1]))
                cv2.line(vis, pt1, pt2, color, box_thickness)

    draw_links(_M2_HEAD_LINKS, _M2_COLOR_HEAD)
    draw_links(_M2_BODY_LINKS, _M2_COLOR_BODY)
    draw_links(_M2_TAIL_LINKS, _M2_COLOR_TAIL)
    draw_links(_M2_LEFT_FRONT_LINKS, _M2_COLOR_LEFT_FRONT)
    draw_links(_M2_RIGHT_FRONT_LINKS, _M2_COLOR_RIGHT_FRONT)
    draw_links(_M2_LEFT_HIND_LINKS, _M2_COLOR_LEFT_HIND)
    draw_links(_M2_RIGHT_HIND_LINKS, _M2_COLOR_RIGHT_HIND)

    for idx in range(len(kpts_xy)):
        if is_valid(idx):
            x, y = int(kpts_xy[idx][0]), int(kpts_xy[idx][1])
            cv2.circle(vis, (x, y), max(3, box_thickness + 1), _M2_COLOR_KPT, -1)


def _mode2_ensure_output_dir():
    if os.path.exists(MODE2_DIR_NO):
        shutil.rmtree(MODE2_DIR_NO)
    if os.path.exists(MODE2_DIR_SMALL):
        shutil.rmtree(MODE2_DIR_SMALL)
    for tier_dir in MODE2_TIER_DIRS:
        if os.path.exists(tier_dir):
            shutil.rmtree(tier_dir)

    os.makedirs(MODE2_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODE2_DIR_NO, exist_ok=True)
    os.makedirs(MODE2_DIR_SMALL, exist_ok=True)
    for tier_dir in MODE2_TIER_DIRS:
        os.makedirs(tier_dir, exist_ok=True)


def _mode2_get_tier_by_ratio(pct):
    """根據百分比（0-100）分配等級 (0-4) 或 small (-1)。"""
    if pct < MODE2_MIN_RATIO_PCT:
        return -1
    normalized = (pct - MODE2_MIN_RATIO_PCT) / (100.0 - MODE2_MIN_RATIO_PCT)
    tier = int(normalized * 5)
    return min(tier, 4)


def _mode2_annotate_and_save(frame, file, status_text, color, pct, dest_dir):
    """在圖上標註並輸出視覺化結果。"""
    vis = frame.copy()
    _h, w = vis.shape[:2]

    base_width = 640
    font_scale = max(1.0, (w / base_width) * 1.2)
    thickness = max(2, int((w / base_width) * 3))
    box_height = int(50 * (w / base_width))

    text = f"{status_text} | ratio: {pct:.1f}%"
    (text_w, _text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(vis, (5, 5), (text_w + 20, box_height), (0, 0, 0), -1)
    cv2.putText(vis, text, (10, int(box_height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    out_path = os.path.join(dest_dir, file)
    cv2.imwrite(out_path, vis)


def run_mode2_batch_folder():
    """資料夾批次分類模式：掃描 MODE2_FOLDER 內所有圖片，依 bbox 占比分類
    搬進 small/tier_1~5/no 資料夾。改用主專案的 KeypointDetector（跟模式 1
    共用同一套偵測介面）取代原本獨立的 ultralytics.YOLO 物件；bbox 占比
    計算沿用共用的 _bbox_area_pct()（浮點精度＋100% 上限夾取）。

    每張圖片彼此獨立，處理前都會呼叫 reset_track()，避免 KeypointDetector
    內建的跨幀 IoU 追蹤把上一張不相干圖片的 bbox 錯誤延續過來。

    多貓同框時，每一隻都會個別算出占比並標示在圖上（cat#2、cat#3...），
    但分類判定（決定搬進哪個 tier 資料夾）固定用信心值最高的那一隻
    （cat#1），不受同框其餘貓咪影響。
    """
    _mode2_ensure_output_dir()

    print("初始化 YOLO-Pose（模式 2 專用模型）...")
    detector = KeypointDetector(
        MODE2_YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD,
    )

    files = [f for f in os.listdir(MODE2_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    summary = []  # (file, ratio_0to1, tier)
    tier_counts = [0] * 5
    small_count = 0
    miss_count = 0

    for file in files:
        path = os.path.join(MODE2_FOLDER, file)
        img = cv2.imread(path)

        if img is None:
            print(f"{file}: ❌ 無法讀取圖片")
            continue

        h, w = img.shape[:2]

        detector.reset_track()
        kpts, kpt_conf, bbox, _bbox_conf, all_instances = detector.detect(img, return_all_instances=True)

        if bbox is None:
            print(f"{file}: ⚠ 未偵測到貓")
            miss_count += 1
            _mode2_annotate_and_save(img, file, "NO CAT", (0, 0, 255), 0.0, MODE2_DIR_NO)
            summary.append((file, 0.0, "no_cat"))
            continue

        pct = _bbox_area_pct(bbox, w, h)
        tier = _mode2_get_tier_by_ratio(pct)
        tier = max(-1, min(tier, 4))  # 安全 clamp

        # 設定顏色：red for small, 漸層綠色 for tier
        color_map = [(0, 0, 255), (0, 100, 200), (0, 150, 150), (0, 200, 100), (100, 200, 0), (0, 200, 0)]
        color = color_map[tier + 1]  # small=-1 映射到 index 0

        if tier == -1:
            status = "SMALL"
            small_count += 1
            dest_dir = MODE2_DIR_SMALL
        else:
            status = f"TIER_{tier+1}"
            tier_counts[tier] += 1
            dest_dir = MODE2_TIER_DIRS[tier]

        vis = img.copy()

        base_width = 640
        font_scale = max(1.0, (w / base_width) * 1.2)
        thickness = max(2, int((w / base_width) * 3))
        box_thickness = max(2, int((w / base_width) * 3))

        if kpts is not None:
            _mode2_draw_skeleton(vis, kpts, kpt_conf, box_thickness)

        x1, y1, x2, y2 = np.asarray(bbox).astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thickness)
        cv2.putText(vis, f"cat#1 {pct:.1f}%", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # 多貓同框：其餘每一隻貓也個別算出佔比並畫出來標示（額外資訊，不影響
        # 這張圖被分類到哪個 tier/small/no 資料夾——分類依據固定用 cat#1）
        extra_pcts = []
        for _kpts_i, _kconf_i, bbox_i, _bconf_i in (all_instances or []):
            if bbox_i is None or np.array_equal(bbox_i, bbox):
                continue
            extra_pct = _bbox_area_pct(bbox_i, w, h)
            extra_pcts.append(extra_pct)

            exi1, eyi1, exi2, eyi2 = np.asarray(bbox_i).astype(int)
            cv2.rectangle(vis, (exi1, eyi1), (exi2, eyi2), _M2_EXTRA_BOX_COLOR, box_thickness)
            cv2.putText(vis, f"cat#{len(extra_pcts)+1} {extra_pct:.1f}%", (exi1, max(20, eyi1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, _M2_EXTRA_BOX_COLOR, thickness, cv2.LINE_AA)

        _mode2_annotate_and_save(vis, file, status, color, pct, dest_dir)

        if extra_pcts:
            extra_str = ", ".join(f"cat#{i+2}={p:.1f}%" for i, p in enumerate(extra_pcts))
            print(f"{file}: {w}x{h}, 貓佔比 = {pct/100:.3f} -> {status}  [同框其餘 {len(extra_pcts)} 隻: {extra_str}]")
        else:
            print(f"{file}: {w}x{h}, 貓佔比 = {pct/100:.3f} -> {status}")
        summary.append((file, pct / 100.0, status.lower()))

    # 整體摘要
    total = len(files)
    passed_count = sum(tier_counts)
    print("\n===== Summary =====")
    print(f"Total images: {total}")
    print(f"No cat:  {miss_count}")
    print(f"Small:   {small_count}")
    print(f"Passed (tier_1~5): {passed_count}")
    for i, c in enumerate(tier_counts):
        low = MODE2_MIN_RATIO_PCT + i / 5 * (100 - MODE2_MIN_RATIO_PCT)
        high = MODE2_MIN_RATIO_PCT + (i + 1) / 5 * (100 - MODE2_MIN_RATIO_PCT)
        print(f"  Tier {i+1} ({low:.0f}% - {high:.0f}%): {c}")

    print("\n===== Output Folders =====")
    print(f"All results saved in: {os.path.abspath(MODE2_OUTPUT_DIR)}")
    print(f"  No cat:   {os.path.abspath(MODE2_DIR_NO)}")
    print(f"  Small:    {os.path.abspath(MODE2_DIR_SMALL)}")
    for i, d in enumerate(MODE2_TIER_DIRS):
        print(f"  Tier {i+1}:  {os.path.abspath(d)}")
    print(f"  Chart:    {os.path.abspath(MODE2_CHART_PATH)}")

    # 畫長條圖
    all_labels = ["no"] + [f"tier_{i+1}" for i in range(5)] + ["small"]
    all_counts = [miss_count] + tier_counts + [small_count]
    colors_chart = ["red"] + ["#FF6B35", "#F7931E", "#FDB833", "#90EE90", "#228B22"] + ["orange"]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(all_labels, all_counts, color=colors_chart)
    for bar, c in zip(bars, all_counts):
        if c > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(c), ha="center", va="bottom")
    plt.title("Cat size distribution")
    plt.ylabel("count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(MODE2_CHART_PATH)
    plt.close()
    print(f"Chart saved: {MODE2_CHART_PATH}")

    # 列出最小的前 5 張
    summary_sorted = sorted([s for s in summary if s[2] != "no_cat"], key=lambda x: x[1])
    top_n = min(5, len(summary_sorted))
    if top_n > 0:
        print("\nSmallest cats:")
        for i in range(top_n):
            f, r, st = summary_sorted[i]
            print(f"  {f}: {r*100:.2f}% ({st})")


# ═══════════════════════════════════════════════════════
#  模式選擇與進入點
# ═══════════════════════════════════════════════════════
def resolve_run_mode():
    if RUN_MODE in (1, 2):
        return RUN_MODE

    # 原本這裡有 `if not sys.stdin.isatty(): return 1` 的提前判斷。拿掉理由跟
    # 1_run_video_inference.py 的 resolve_run_mode() 一樣：stdin 被導向 pipe（例如
    # settings_window.py 的「獨立腳本工具」面板）時 isatty() 也是 False，但那個 pipe
    # 其實可以讀，使用者能透過該面板的輸入框送文字進來，不該直接跳過詢問。真的完全
    # 沒有 stdin 可讀時，下面 input() 的 except 還是會接住、給預設值，不會卡住。
    print("\n請選擇執行模式:")
    print("  1) 模式1：即時影片/攝影機量測（bbox 占比 + 滑鼠自畫矩形）")
    print("  2) 模式2：資料夾批次分類（依 bbox 占比分類圖片到 tier 資料夾）")
    try:
        choice = input("輸入模式 (1/2, 預設=1): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n未輸入模式，預設使用模式 1")
        return 1

    if choice == "2":
        return 2
    return 1


def main():
    mode = resolve_run_mode()
    if mode == 2:
        run_mode2_batch_folder()
    else:
        run_mode1_video()


if __name__ == "__main__":
    main()
