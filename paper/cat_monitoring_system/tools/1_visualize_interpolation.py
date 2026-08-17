"""
interpolate_missing() 前後視覺化測試腳本——左右分割畫面比對「YOLO 原始偵測」
vs「時間軸插值補全後」的骨架差異，用來實際驗證插值在不同遮擋長度下的行為
邊界（見 models/stgcn_model.py 的 interpolate_missing()：SEQUENCE_LENGTH=16
幀、TARGET_MODEL_FPS=30 代表插值真正能用的「記憶範圍」只有約 0.53 秒，
超過這個範圍的持續遮擋，不管之前偵測多穩定，都會被歸零，不是靠平滑就能
撐過去的）。

不用指令列參數，路徑/設定都寫在下面「使用者設定區」。

遮蔽區塊用滑鼠互動即時畫，不用在程式碼裡先設固定座標：暫停影片後按 o
進入畫遮蔽區模式，在左側 RAW 面板上連續點 4 下滑鼠左鍵，這 4 個點會自動
組成一個黑色遮蔽多邊形（依你點擊的順序連成四邊形，建議照著想遮的區域邊緣
順時鐘或逆時鐘點，不要跳著點以免自我交叉變成蝴蝶結形狀）；可以連續點好幾組
4 點疊加出多個遮蔽區塊。按 s 儲存目前畫的所有區塊、正式套用到偵測；按 x
清除所有已存的遮蔽區塊（未存滿 4 點的那組點擊不算數，s/x 都會先清掉）。

儲存後的遮蔽區塊會真的畫黑色多邊形到 detect_frame 上再餵給 YOLO 重新推論，
信心值和座標都是 YOLO 看不到那塊區域時的真實反應，不是人為假設；同時另外
用一個獨立的第二份 KeypointDetector 對「沒被遮住的原始畫面」多跑一次推論
當 ground truth（不會拿去餵插值，純粹用來畫比對線／算誤差像素數），藉此在
任意影片上重現、驗證「短暫遮擋 vs 持續遮擋超過視窗長度」這兩種情境的插值
行為差異，不需要真的找一段剛好被遮到的影片。遮蔽區塊存了之後會持續套用到
後續每一幀（模擬「有東西一直擋在鏡頭前」），直到你按 x 清除，不是照時間
自動開關。

按鍵：
    space = 暫停/繼續（恢復播放時會自動離開畫遮蔽區模式）
    a/d   = 往回/往前逐幀瀏覽（只重播畫面快取，不重新推論；跟
            1_run_video_inference.py 的 a/d 是同一套設計）
    o     = 開關「畫遮蔽區模式」（僅暫停時可進入，進入後在左側面板點 4 下
            滑鼠左鍵形成一個區塊，可連續點多組）
    s     = 儲存目前畫的遮蔽區塊，套用到後續偵測（暫停時按下畫面會立刻更新）
    x     = 清除所有已存的遮蔽區塊（暫停時按下畫面會立刻更新，不用等恢復播放）
    t     = 開關「是否套用已存的遮蔽區塊」——區塊本身不會被刪除，只是暫時
            不塗黑/不影響偵測，播放中隨時可以按，方便快速對照「有遮擋」跟
            「沒遮擋」兩種狀態，不用重畫或用 x 真的刪掉再重畫
    r     = 重新從頭播放（遮蔽區塊不會被重置，只有 x 會清除）
    q     = 離開

畫面上的文字全部使用英文——OpenCV 內建的 Hershey 字型不支援中文字元，
中文會被畫成一堆問號/亂碼方框，這裡刻意避開。
"""
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Ensure both the package folder and repository root are on sys.path so
# top-level modules like config.py can be imported when running this script
# from within the cat_monitoring_system folder.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from detectors.keypoint_detector import KeypointDetector
from models.stgcn_model import interpolate_missing

# ═══════════════════════════════ 使用者設定區 ═══════════════════════════════
VIDEO_PATH = r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\1_貓咪姿勢影片分類\模型專用\walk\walk_12.mp4"

YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_133.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5

# 跟 production（config.py STGCNConfig）保持一致，才能反映真實的記憶邊界
SEQUENCE_LENGTH = 16
TARGET_MODEL_FPS = 30.0
INTERP_THRESHOLD = 0.1  # 跟 stgcn_model.py interpolate_missing() 預設一致

# 左右影像同一個關節的原始像素距離超過這個門檻，就視為「偏移」，在右側面板
# 用橘色標出來——這是直接拿左側（遮擋畫面偵測結果）跟右側（完全未遮擋的
# ground truth）逐點比對位置算出來的，不看信心值，所以能抓到「YOLO 信心值
# 沒掉到 INTERP_THRESHOLD 以下、但實際座標其實已經因為遮擋而偏掉」這種
# confidence 抓不到的情況。
OFFSET_THRESHOLD_PX = 15

DISPLAY_SIZE = (860, 620)  # 左右各半的顯示大小，兩半合併後視窗寬度是兩倍
WINDOW_NAME = "Interpolation Before/After (space=pause  a/d=step  o=draw  s=save  x=clear  t=toggle-occl  r=restart  q=quit)"
DRAW_KP_CONF_THRESHOLD = 0.25  # 高於此信心值才視為「有效」正常畫（僅影響左側原始畫面的顯示判斷）

FRAME_HISTORY_SIZE = 150  # a/d 逐幀瀏覽用的畫面快取上限（雙拼畫面較寬，數量抓 1_run_video_inference.py 的一半）

KEYPOINT_NAMES = [
    "Nose", "L_Ear", "R_Ear", "Chest", "MidBack", "Hip",
    "LF_Elbow", "LF_Paw", "RF_Elbow", "RF_Paw",
    "LH_Knee", "LH_Paw", "RH_Knee", "RH_Paw",
    "Tail_Root", "Tail_Mid", "Tail_Tip",
]

_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]
# ═══════════════════════════════════════════════════════════════════════════


def resize_with_letterbox(image, target_size):
    """等比例縮放至目標尺寸（保留完整畫面，四周補黑邊）。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def draw_panel(frame, kpts, kpt_conf, title, hollow_joints=None, highlight_joints=None):
    """畫一個面板：骨架線段＋關鍵點。
    hollow_joints   ─ 這些關節畫成空心紅圈（代表「原始信心不足/被模擬遮擋」）。
    highlight_joints─ 這些關節畫成黃色實心＋外框（代表「這幀數值是插值補回來的」）。
    兩者互斥使用（左側原始畫面用 hollow，右側插值畫面用 highlight）。
    """
    hollow_joints = hollow_joints or set()
    highlight_joints = highlight_joints or set()

    for a, b in _SKELETON_EDGES:
        if a in hollow_joints or b in hollow_joints:
            continue  # 任一端被遮擋就不畫這段骨架線，視覺上直接看出斷開
        pa = (int(kpts[a][0]), int(kpts[a][1]))
        pb = (int(kpts[b][0]), int(kpts[b][1]))
        cv2.line(frame, pa, pb, (180, 180, 180), 2, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        cx, cy = int(kpts[i][0]), int(kpts[i][1])
        if i in hollow_joints:
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        if i in highlight_joints:
            cv2.circle(frame, (cx, cy), 8, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 6, (0, 230, 255), -1)
            continue
        if float(kpt_conf[i]) > DRAW_KP_CONF_THRESHOLD:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 4, (100, 255, 100), -1)

    cv2.putText(frame, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_status_panel(canvas, x0, y0, lines):
    """左上角半透明黑底文字狀態面板（多行）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th, line_h, pad = 0.5, 1, 20, 6
    sizes = [cv2.getTextSize(text, font, fs, th)[0] for text, _color in lines]
    box_w = max((w for w, _ in sizes), default=0) + pad * 2
    box_h = line_h * len(lines) + pad
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (line, color) in enumerate(lines):
        ty = y0 + pad + line_h * (i + 1) - 5
        cv2.putText(canvas, line, (x0 + pad, ty), font, fs, color, th, cv2.LINE_AA)
    return canvas


def main():
    if not Path(VIDEO_PATH).exists():
        print(f"❌ 找不到影片: {VIDEO_PATH}")
        return

    try:
        keypoint_detector = KeypointDetector(
            YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD,
        )
    except Exception as e:
        print(f"❌ 無法載入 YOLO 模型（{YOLO_MODEL_PATH}）：{e}")
        return

    # 一旦存了遮蔽區塊，就需要對「沒被擋住的原始畫面」另外跑一次推論當
    # ground truth——必須用獨立的第二個 KeypointDetector 實例，如果共用同一個，
    # 兩次 detect() 呼叫會互相覆蓋掉彼此的 IoU 追蹤狀態（_prev_bbox），互相干擾。
    try:
        truth_detector = KeypointDetector(
            YOLO_MODEL_PATH, device=INFERENCE_DEVICE, imgsz=YOLO_IMGSZ, conf_thres=YOLO_CONF_THRESHOLD,
        )
    except Exception as e:
        print(f"⚠ 無法載入第二份 YOLO 模型做 ground truth 比對，遮蔽區塊的預測誤差顯示將停用：{e}")
        truth_detector = None

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_MODEL_FPS
    window_seconds = SEQUENCE_LENGTH / TARGET_MODEL_FPS
    print("=" * 60)
    print("插值 (interpolate_missing) 前後視覺化測試")
    print("=" * 60)
    print(f"影片: {VIDEO_PATH}")
    print(f"視窗長度: {SEQUENCE_LENGTH} 幀 @ {TARGET_MODEL_FPS} fps ≈ {window_seconds:.2f} 秒（插值真正的記憶邊界）")
    print("遮蔽區塊: 用滑鼠互動畫（暫停後按 o 進入畫遮蔽區模式，左側面板點 4 下形成一個區塊，s 儲存 / x 清除）")
    print("=" * 60)

    kpts_buffer = deque(maxlen=SEQUENCE_LENGTH)
    conf_buffer = deque(maxlen=SEQUENCE_LENGTH)

    paused = False
    frame_idx = 0

    # a/d 逐幀瀏覽用的畫面快取（跟 1_run_video_inference.py 同一套設計）：
    # 只重播已經畫好的 canvas，不重新跑偵測/插值，避免逆放時弄亂狀態。
    frame_history = deque(maxlen=FRAME_HISTORY_SIZE)
    history_back_steps = 0

    # 滑鼠互動畫遮蔽區塊用的狀態：occlusion_regions 是已存的區塊清單
    # （每個是 4 點的 np.int32 陣列），pending_points 是目前正在收集、
    # 還沒湊滿 4 點的暫存點。last_letterbox 記錄左側面板最近一次縮放的
    # scale/pad，讓滑鼠回呼能把顯示座標換算回原始影片解析度座標。
    occlusion_regions = []
    pending_points = []
    drawing_mode = False
    last_letterbox = {"scale": 1.0, "pad_x": 0, "pad_y": 0}
    # occlusion_enabled：t 鍵切換，暫時開關「已存的區塊要不要真的套用到偵測」，
    # 跟 x（永久刪除）分開——關掉時 occlusion_regions 內容完全不變，只是這段
    # 時間偵測時當作沒有遮蔽區塊，方便隨時 A/B 對照有無遮擋、不用重畫。
    occlusion_enabled = True

    def rebuild_left_preview(base_frame):
        """畫遮蔽區模式下的即時預覽：目前已存的區塊＋正在收集的點。"""
        preview = base_frame.copy()
        for region in occlusion_regions:
            cv2.fillPoly(preview, [region], (0, 0, 0))
            cv2.polylines(preview, [region], True, (0, 0, 255), 2, cv2.LINE_AA)
        for i, (px, py) in enumerate(pending_points):
            cv2.circle(preview, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(preview, str(i + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        if len(pending_points) >= 2:
            cv2.polylines(preview, [np.array(pending_points, dtype=np.int32)], False, (0, 255, 255), 1, cv2.LINE_AA)
        label = f"DRAW MODE: click 4 pts/region ({len(pending_points)}/4)  regions={len(occlusion_regions)}  s=save  x=clear"
        cv2.putText(preview, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return preview

    def on_mouse(event, x, y, flags, param):
        if not drawing_mode or event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= DISPLAY_SIZE[0]:
            print("\n⚠ 請在左側 RAW 面板上點擊（右側是比對用畫面，不用來定義遮蔽區）")
            return
        scale = last_letterbox["scale"]
        pad_x, pad_y = last_letterbox["pad_x"], last_letterbox["pad_y"]
        if scale <= 0:
            return
        orig_x = int(round((x - pad_x) / scale))
        orig_y = int(round((y - pad_y) / scale))
        pending_points.append((orig_x, orig_y))
        print(f"\n📍 已記錄第 {len(pending_points)}/4 個點: ({orig_x}, {orig_y})")
        if len(pending_points) == 4:
            occlusion_regions.append(np.array(pending_points, dtype=np.int32))
            pending_points.clear()
            print(f"✓ 已形成第 {len(occlusion_regions)} 個遮蔽區塊，可再點 4 下加下一個，或按 s 儲存 / x 清除")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n播放完畢")
                break
            frame_idx += 1
            frame_time_sec = frame_idx / source_fps

            # detect_frame 才是「系統實際看到的畫面」：只要有存好的遮蔽區塊、
            # 且 occlusion_enabled（t 鍵）目前是開的，就真的把它們塗黑到畫面上
            # 再餵給 YOLO 重新推論，信心值/座標都是 YOLO 看不到那些區域時的
            # 真實反應，不是人為假設。t 關掉時區塊還在（不會被刪），只是暫時
            # 不套用，方便快速切換對照；x 才是真的永久刪除。
            occlusion_active_now = bool(occlusion_regions) and occlusion_enabled
            detect_frame = frame
            if occlusion_active_now:
                detect_frame = frame.copy()
                cv2.fillPoly(detect_frame, occlusion_regions, (0, 0, 0))

            kpts, kpt_conf, bbox, bbox_conf = keypoint_detector.detect(detect_frame)

            if kpts is None:
                left_full = detect_frame.copy()
                right_full = frame.copy()
                cv2.putText(left_full, "No cat detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(right_full, "No cat detected", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                # 一律縮放到 DISPLAY_SIZE，跟正常偵測到貓的分支保持一致大小，
                # 避免跟畫遮蔽區模式的預覽（固定 DISPLAY_SIZE）hstack 時尺寸對不上而噴錯。
                left, scale, pad_x, pad_y = resize_with_letterbox(left_full, DISPLAY_SIZE)
                last_letterbox["scale"], last_letterbox["pad_x"], last_letterbox["pad_y"] = scale, pad_x, pad_y
                right, _, _, _ = resize_with_letterbox(right_full, DISPLAY_SIZE)
            else:
                kpt_conf = kpt_conf.copy()

                # ground truth：只有這一幀真的有套用遮蔽（occlusion_active_now）
                # 時，kpts 才是「看不到那些區域」的真實反應，需要另外對乾淨畫面
                # frame 用獨立的 truth_detector 多跑一次才能拿到真正的位置當比對
                # 基準；沒套用遮蔽時（沒存區塊，或 t 鍵暫時關掉）kpts 本來就等於
                # 乾淨畫面的結果，不需要多跑一次，省一份推論。truth_conf 一起
                # 留著，右側面板要拿它畫「完全未遮擋」的那副骨架本身，不能只當
                # 比對用的隱藏數據。
                truth_kpts = kpts
                truth_conf = kpt_conf
                if occlusion_active_now and truth_detector is not None:
                    t_kpts, t_conf, _t_bbox, _t_bconf = truth_detector.detect(frame)
                    if t_kpts is not None:
                        truth_kpts = t_kpts
                        truth_conf = t_conf

                kpts_buffer.append(kpts.copy())
                conf_buffer.append(kpt_conf.copy())

                kpts_arr = np.array(kpts_buffer)   # (t<=T, 17, 2)
                conf_arr = np.array(conf_buffer)   # (t<=T, 17)
                interpolated_arr = interpolate_missing(kpts_arr, conf_arr, threshold=INTERP_THRESHOLD)

                raw_last_conf = conf_arr[-1]
                interp_last_kpts = interpolated_arr[-1]

                hollow = {i for i in range(17) if raw_last_conf[i] <= INTERP_THRESHOLD}
                # 這一幀原始信心不足、但插值後仍有非零座標的關節，才算「靠插值補回來」；
                # 如果窗口內這個關節從頭到尾都無效，interpolate_missing 會整個歸零，
                # 用 highlight 標黃只會誤導成「補成功」，改用另一種狀態呈現。
                recovered = {
                    i for i in hollow
                    if np.any(conf_arr[:, i] > INTERP_THRESHOLD)
                }
                # exhausted（窗口內完全沒有有效幀 → 已歸零，插值救不回來）不用另外
                # 存成變數，下面判斷式都用 "j in recovered" 的否定即可涵蓋。
                # 只用來決定「左側原始畫面」要不要畫某條骨架線的門檻：比插值用的
                # INTERP_THRESHOLD(0.1) 嚴格，避免信心低但非零的雜訊座標連出一堆
                # 亂飄的線段，畫面看起來像是壞掉（實際上只是 YOLO 本身信心低、
                # 座標不準，不是這支腳本的 bug，用比較嚴格的門檻過濾掉方便肉眼看）。
                weak_for_display = {i for i in range(17) if raw_last_conf[i] <= DRAW_KP_CONF_THRESHOLD}

                window_fill = len(kpts_buffer)
                # 左側用 detect_frame（系統實際看到的畫面）當底圖：有存遮蔽區塊時
                # 會直接看到那些黑色區域，一眼看出 YOLO 是在什麼輸入下做出這個
                # 判斷的，不是憑空想像的遮擋。
                left_full = draw_panel(detect_frame.copy(), kpts, raw_last_conf, "RAW (occluded, before interpolate_missing)", hollow_joints=weak_for_display)
                # 右側骨架本身必須是「完全沒被遮擋」時偵測到的真實結果
                # （truth_kpts/truth_conf，來自對乾淨畫面單獨推論的 truth_detector），
                # 不是插值猜測結果——插值猜的位置只當疊加標記用，不能拿去當右側
                # 這副骨架的主體，否則沒觸發插值時（信心值剛好還沒低到門檻）右側
                # 會誤導成「系統看到的就是這樣」，實際上只是原始遮擋輸出照抄。
                right_full = draw_panel(frame.copy(), truth_kpts, truth_conf, "GROUND TRUTH (fully unoccluded detection)")

                # 核心：逐點直接比對左側（kpts，keypoint_detector 對遮擋畫面獨立
                # 推論出來的結果）跟右側（truth_kpts，truth_detector 對完全未遮擋
                # 畫面獨立推論出來的結果）的原始像素距離——不看信心值，因為信心值
                # 沒掉到門檻以下、但座標其實已經偏掉的情況，hollow 那組判斷抓不到，
                # 逐點量距離才抓得到。超過 OFFSET_THRESHOLD_PX 的關節，只把右側
                # 這個點本身的顏色改成橘色，不畫連線／文字——遮擋範圍大時，被遮住
                # 的關節在左側可能被擠壓/collapse 到同一小塊區域，如果拉線連回
                # 右側對應的真實位置，會變成一堆長線交叉整張圖，反而看不清楚，
                # 單純變色最乾淨。
                OFFSET_COLOR = (0, 140, 255)
                for j in range(17):
                    px, py = kpts[j][0], kpts[j][1]
                    tx, ty = truth_kpts[j][0], truth_kpts[j][1]
                    offset_px = float(np.hypot(px - tx, py - ty))
                    if offset_px <= OFFSET_THRESHOLD_PX:
                        continue
                    cv2.circle(right_full, (int(tx), int(ty)), 8, (0, 0, 0), -1)
                    cv2.circle(right_full, (int(tx), int(ty)), 6, OFFSET_COLOR, -1)

                # 只在這一幀真的有套用遮蔽時才疊紅色外框，t 關掉時 detect_frame
                # 沒被塗黑，畫外框反而會誤導成「有套用」——外框要跟實際套用狀態
                # 一致，不能只看區塊清單存不存在。
                if occlusion_active_now:
                    for region in occlusion_regions:
                        cv2.polylines(left_full, [region], True, (0, 0, 255), 2, cv2.LINE_AA)

                left, scale, pad_x, pad_y = resize_with_letterbox(left_full, DISPLAY_SIZE)
                last_letterbox["scale"], last_letterbox["pad_x"], last_letterbox["pad_y"] = scale, pad_x, pad_y
                right, _, _, _ = resize_with_letterbox(right_full, DISPLAY_SIZE)

                # OpenCV 內建字型不支援中文，狀態文字一律用英文，避免畫成問號亂碼。
                status_lines = [
                    (f"t={frame_time_sec:5.2f}s  window={window_fill}/{SEQUENCE_LENGTH} "
                     f"(~{window_fill / TARGET_MODEL_FPS:.2f}s)  regions={len(occlusion_regions)}"
                     f"  occl={'ON' if occlusion_enabled else 'OFF'}", (255, 255, 255)),
                ]
                for j in sorted(hollow):
                    px, py = interp_last_kpts[j][0], interp_last_kpts[j][1]
                    tx, ty = truth_kpts[j][0], truth_kpts[j][1]
                    err_px = float(np.hypot(px - tx, py - ty))
                    if j in recovered:
                        status_lines.append((f"  {KEYPOINT_NAMES[j]}: recovered, error vs ground truth = {err_px:.0f}px", (0, 230, 255)))
                    else:
                        status_lines.append((f"  {KEYPOINT_NAMES[j]}: past {window_seconds:.2f}s memory limit -> zeroed, error = {err_px:.0f}px", (0, 0, 255)))
                if not hollow:
                    status_lines.append(("  (all joints valid this frame, no interpolation needed)", (150, 255, 150)))
                # y0=60：title 用 0.8 字級+4px 外框粗細畫在 y=30，下緣大概到 y~40，
                # 這裡留夠間距避免狀態面板疊字看起來像亂碼（之前 y0=44 太近會疊到）。
                right = draw_status_panel(right, 8, 60, status_lines)

            canvas = np.hstack([left, right])
            # 進來的都是「新推進」的一幀（a/d 瀏覽只重播快取，不會走到這裡），
            # 所以每次都重置回捲位置到最新。
            frame_history.append(canvas)
            history_back_steps = 0
        elif drawing_mode:
            # 暫停且在畫遮蔽區模式：每一輪都用目前這幀的乾淨畫面重畫一次左側
            # 預覽（已存區塊＋正在收集的點），這樣每次滑鼠點擊後畫面能立刻
            # 更新，不用等到真的推進下一幀。右側沿用最後一次算好的比對畫面。
            preview = rebuild_left_preview(frame)
            preview_letterboxed, scale, pad_x, pad_y = resize_with_letterbox(preview, DISPLAY_SIZE)
            last_letterbox["scale"], last_letterbox["pad_x"], last_letterbox["pad_y"] = scale, pad_x, pad_y
            canvas = np.hstack([preview_letterboxed, right])

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key == ord('q'):
            print("\n使用者中斷：q")
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                drawing_mode = False  # 恢復播放時自動離開畫遮蔽區模式，避免背景繼續收滑鼠點擊
        elif key == ord('a'):
            if history_back_steps < len(frame_history) - 1:
                history_back_steps += 1
                paused = True
                canvas = frame_history[len(frame_history) - 1 - history_back_steps]
            else:
                print(f"\n⏮ 已到快取最舊的一幀（上限 {FRAME_HISTORY_SIZE} 幀）")
        elif key == ord('d'):
            if history_back_steps > 0:
                history_back_steps -= 1
                canvas = frame_history[len(frame_history) - 1 - history_back_steps]
            else:
                print("\n⏭ 已是目前最新一幀，按空白鍵可恢復即時播放")
        elif key == ord('o'):
            if not paused:
                print("\n⚠ 請先暫停影片（space）再進入畫遮蔽區模式")
            else:
                drawing_mode = not drawing_mode
                pending_points = []
                print(f"\n✏ 畫遮蔽區模式: {'開啟（在左側面板點 4 下形成一個區塊）' if drawing_mode else '關閉'}")
        elif key == ord('s'):
            if pending_points:
                print(f"\n⚠ 還有 {len(pending_points)} 個點未湊滿 4 個，這組不算數")
            pending_points = []
            drawing_mode = False
            print(f"\n💾 已儲存，目前共 {len(occlusion_regions)} 個遮蔽區塊，套用到後續偵測")
            if paused:
                preview = rebuild_left_preview(frame)
                preview_letterboxed, scale, pad_x, pad_y = resize_with_letterbox(preview, DISPLAY_SIZE)
                last_letterbox["scale"], last_letterbox["pad_x"], last_letterbox["pad_y"] = scale, pad_x, pad_y
                canvas = np.hstack([preview_letterboxed, right])
        elif key == ord('x'):
            occlusion_regions = []
            pending_points = []
            drawing_mode = False
            print("\n🗑 已清除所有遮蔽區塊")
            # 暫停時按 x 要立刻反映在畫面上，不能等恢復播放才看到黑色區塊消失
            # ——不管當下有沒有在畫遮蔽區模式，都重畫一次左側預覽（這時候
            # occlusion_regions 已經是空的，rebuild_left_preview 自然不會再畫
            # 黑色區塊/紅色外框）。
            if paused:
                preview = rebuild_left_preview(frame)
                preview_letterboxed, scale, pad_x, pad_y = resize_with_letterbox(preview, DISPLAY_SIZE)
                last_letterbox["scale"], last_letterbox["pad_x"], last_letterbox["pad_y"] = scale, pad_x, pad_y
                canvas = np.hstack([preview_letterboxed, right])
        elif key == ord('t'):
            occlusion_enabled = not occlusion_enabled
            print(f"\n🔀 遮蔽套用: {'開啟' if occlusion_enabled else '暫時停用（區塊還在，只是這段時間不套用到偵測，按 t 可再開回來）'}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            kpts_buffer.clear()
            conf_buffer.clear()
            frame_history.clear()
            history_back_steps = 0
            keypoint_detector.reset_track()
            if truth_detector is not None:
                truth_detector.reset_track()
            print("\n↺ 已重置：回到影片開頭（遮蔽區塊不受影響，按 x 才會清除）")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
