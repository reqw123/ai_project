"""
貓咪身分驗證測試腳本（YOLO-Pose 定位 + 顏色特徵比對，不含行為辨識）
=================================================================
背景問題：個體化基線（cat_health_v3_flow.json 的「個體化基線計算器」）假設
全程只有目標貓一隻入鏡。但 detectors/keypoint_detector.py 的 KeypointDetector
目前處理多貓同框的方式，只有「IoU 追蹤延續上一幀鎖定的 bbox」+「信心值最高
者退回」兩層 —— 這只保證「空間位置連續」，不保證「這是同一隻貓」。目標貓
暫時離開、另一隻貓剛好在附近位置出現時，追蹤可能誤鎖到錯的貓，行為資料就
會被另一隻貓污染，個體化基線的正當性就跨掉了。

第一版方法（骨架比例特徵）已捨棄：軀幹長當正規化基準本身會隨姿勢（弓背/
伏低）系統性伸縮，四肢/尾巴的 2D 投影長度又受透視縮短跟捲曲角度影響很大，
單幀雜訊太高。改用顏色特徵：YOLO-Pose 仍然用來抓 bbox（定位貓在哪），但
身分判定只看 bbox 區域裁切出來的 HSV 色彩直方圖，不看關鍵點幾何。

第三版（本版）：既然畫面上最多只會同時出現兩隻貓，與其只建立「目標貓」的
基準、拿一個絕對距離門檻去判斷「像不像目標貓」，不如兩隻貓的顏色特徵都建
基準，每個偵測到的個體同時跟兩份基準比距離，取比較近的那個——這是「兩類
最近鄰分類」，比單一類別的開放集合門檻判斷更穩，因為兩隻貓的特徵本來就會
互相把彼此推開。

兩階段：
    1) enroll —— 對 CAT_PROFILES 裡設定的每一隻貓，各自用一段「確定只有那
                隻貓」的參考影片，取多幀 bbox 裁切區域的顏色直方圖，存成
                一組樣本（保留多筆而非單一平均值，比對時取最近鄰）
    2) verify —— 在測試影片上，對每個偵測到的個體，同時算出跟每一隻已知貓
                基準的最小 Bhattacharyya 距離，取最近的當判定結果；兩邊都
                太遠則判定為「未知」。用簡易 IoU 追蹤 + 多幀多數決平滑，
                避免單幀雜訊造成偶發誤判。逐幀結果輸出成 CSV，並可用
                SHOW_PREVIEW 開即時預覽視窗肉眼核對判定對不對。

控制鍵（SHOW_PREVIEW=True 時）：q=退出　space=暫停/繼續　1=上一部影片　2=下一部影片

用法：不用下指令列參數，直接改下面「使用者設定區」的路徑跟 RUN_MODE，
      然後 python 3_cat_identity_verification_test.py 執行即可。

這是驗證「這個方法可不可行」用的測試腳本，還沒接回正式的 KeypointDetector。
"""
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════
#  使用者設定區
# ═══════════════════════════════════════════════════════
YOLO_MODEL_PATH = r"C:\ai_project\yolo_models\v11s_133.pt"
INFERENCE_DEVICE = "cuda"
YOLO_IMGSZ = 640
YOLO_CONF_THRESHOLD = 0.5   # bbox 偵測信心門檻

# "enroll"   = 對 CAT_PROFILES 裡的每一隻貓建立顏色特徵基準
# "verify"   = 在測試影片上做身分驗證（要先跑過一次 enroll，才會有基準檔）
# "diagnose" = 不跑任何影片，只用已經 enroll 好的基準檔做 leave-one-out 交叉
#              驗證，量化「這兩隻貓的顏色特徵本身分不分得開」，判斷精度不夠時
#              該往哪個方向加強（見 cmd_diagnose() 開頭的說明）
RUN_MODE = "verify"

# 貓咪身分設定：每一隻要辨識的貓，各自的名稱、預覽畫面顏色、enroll 用參考
# 影片/資料夾。key（"cat_a"/"cat_b"）只是內部識別用，label 是給你自己看/
# console 訊息用的中文名字，short_label 是 GUI 預覽畫面上顯示用的英文短名
# （GUI 不顯示中文，避免中文字型在 OpenCV 畫面上顯示成方框亂碼）。
# 畫面上最多只會同時出現兩隻貓，所以這裡放兩筆；如果你的環境確定不會有第二
# 隻貓固定出現，把 "cat_b" 那筆刪掉即可，會退回「只認一隻、其餘都算未知」
# 的模式。每隻貓的 enroll_source 建議放好幾支不同時段/光線的影片或一個資料夾。
CAT_PROFILES = {
    "cat_a": {
        "label": "橘貓（目標貓）",
        "short_label": "cat1",     # GUI 上顯示的英文短名
        "color": (0, 200, 0),      # BGR，預覽畫面框線/標籤顏色
        "enroll_source": [
            r"C:\Users\homec\Downloads\目標貓採樣\目標貓",
        ],
    },
    "cat_b": {
        "label": "黑白貓",
        "short_label": "cat2",
        "color": (0, 140, 255),
        "enroll_source": [
            r"C:\Users\homec\Downloads\目標貓採樣\他貓",   # TODO: 換成黑白貓的單貓參考影片/資料夾
        ],
    },
}

# verify 用的測試影片/資料夾（可能有雙貓同框的那種），可放多個路徑
VERIFY_SOURCE = [
    r"C:\Users\homec\Downloads\2026-07-17 15_15_52_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_18_54_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-15 05_49_20_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-16 15_27_45_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-16 15_28_05_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_18_28_0.mp4",
    r"C:\Users\homec\Downloads\目標貓採樣\他貓\2026-07-17 04_51_46_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_18_54_0(1).mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_30_55_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_32_00_0.mp4",
    r"C:\Users\homec\Downloads\2026-07-18 10_32_31_0.mp4"
]

SHOW_PREVIEW = True   # verify 時是否開即時預覽視窗
PREVIEW_DISPLAY_SIZE = (1280, 720)   # 預覽視窗顯示大小（寬, 高），等比縮放＋黑邊，不裁切畫面
PREVIEW_WINDOW_NAME = "Cat Identity Verification Test (q=quit  space=pause  1/2=prev/next video)"
UNKNOWN_COLOR = (120, 120, 120)   # 兩份基準都比不上（判定為未知）時的框線顏色

# ── 顏色直方圖設定 ──
H_BINS = 30                 # Hue（色相）bin 數
S_BINS = 32                 # Saturation（飽和度）bin 數
HSV_S_MIN = 60               # 飽和度低於此值視為接近灰階/反光，不列入直方圖統計
HSV_V_MIN = 32                # 亮度太暗（陰影）不列入
HSV_V_MAX = 255
BBOX_INSET_RATIO = 0.12      # bbox 往內縮的比例，避開邊緣背景，盡量只取貓身體中心區域
UNKNOWN_DISTANCE_CEILING = 0.5   # 跟最近的那隻貓基準距離還是超過這個值，判定為「未知」（可能是第三隻貓/雜訊）
MAX_ENROLL_SAMPLES = 1000     # enroll 每隻貓最多保留幾筆直方圖樣本（比對時取最近鄰，太多筆會拖慢 verify）
ENROLL_SAMPLE_STRIDE = 2     # enroll 每隔幾個有效幀才取一次樣本，避免相鄰幀幾乎重複、浪費樣本額度

# ── enroll 專用的資料品質門檻（比一般偵測用的 YOLO_CONF_THRESHOLD 更嚴格，
# 確保寫進特徵基準的每一筆樣本，貓咪都是清楚、確實被捕捉到的，不是模糊/太遠/
# 邊緣判定的案例——基準本身品質不好，後面 verify 判得再準也沒用）──
ENROLL_MIN_BBOX_CONF = 0.6          # bbox 偵測信心門檻，比一般的 0.5 更嚴格
ENROLL_MIN_BBOX_HEIGHT_RATIO = 0.15  # bbox 高度至少要佔畫面高度的比例，太小（太遠/太小隻）就跳過
ENROLL_MIN_SHARPNESS = 60.0          # 裁切區域的 Laplacian 變異數，低於此值視為模糊（動態模糊/失焦），跳過

# ── 跨幀平滑設定（緩解單幀雜訊造成的偶發誤判）──
# verify 每一幀都是獨立判定，單一幀因為模糊/瞬間角度/bbox 帶到背景等雜訊，偶爾會誤判。
# 這裡用簡易 IoU 追蹤把同一隻貓在連續幀之間串起來，取最近 SMOOTH_WINDOW 幀的原始判定
# 做多數決（哪個標籤出現次數最多就採用），單一幀的偶發誤判會被稀釋掉。
SMOOTH_WINDOW = 9            # 每個追蹤目標取最近幾幀的原始判定做多數決
TRACK_IOU_THRES = 0.3        # 這一幀的 bbox 跟上一幀哪個追蹤是同一隻貓的 IoU 門檻
TRACK_MAX_MISSED = 10        # 追蹤連續幾幀沒配對到偵測，才捨棄這個追蹤

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"}
# ═══════════════════════════════════════════════════════


def profile_path_for(key):
    return Path(__file__).parent / f"cat_profile_{key}.json"


def resize_with_letterbox(image, target_size):
    """等比例縮放至目標尺寸（保留完整畫面，四周補黑邊），跟 1_run_video_inference.py
    的同名函式一致，只是這裡用途單純是縮放已經畫好框線/文字的預覽畫面。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return image
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def draw_label_block(img, x, y, lines, border_color):
    """在 (x,y)（通常是 bbox 左上角）上方畫一個半透明黑底的多行文字標籤，
    不管背景畫面顏色/亮度如何都看得清楚。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.85, 2
    line_h = 32
    pad = 8
    sizes = [cv2.getTextSize(line, font, fs, th)[0] for line in lines]
    box_w = max(w for w, _ in sizes) + pad * 2
    box_h = line_h * len(lines) + pad
    y1 = max(0, y - box_h)
    y2 = y1 + box_h
    x2 = min(img.shape[1], x + box_w)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x, y1), (x2, y2), border_color, 2)
    for i, line in enumerate(lines):
        ty = y1 + pad + line_h * (i + 1) - 6
        cv2.putText(img, line, (x + pad, ty), font, fs, (255, 255, 255), th, cv2.LINE_AA)


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


class SimpleTracker:
    """簡易 IoU 追蹤器，只用來把同一隻貓在連續幀之間的原始判定（貓的 key 或
    None=未知）串起來做多數決平滑，不是給生產環境用的完整多目標追蹤器（沒有
    處理身分交換/長時間遮擋等情況，那些超出這支測試腳本要驗證的範圍）。"""

    def __init__(self):
        self.tracks = {}  # track_id -> {"bbox":..., "history": deque, "missed": int}
        self.next_id = 0

    def update(self, bboxes):
        """bboxes: 這一幀每個偵測的 bbox（可能是 None）。回傳等長的 track_id 清單。"""
        assigned = [None] * len(bboxes)
        used_tracks = set()
        for i, bbox in enumerate(bboxes):
            if bbox is None:
                continue
            best_tid, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(t["bbox"], bbox)
                if iou > best_iou:
                    best_tid, best_iou = tid, iou
            if best_tid is not None and best_iou >= TRACK_IOU_THRES:
                assigned[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid]["bbox"] = bbox
                self.tracks[best_tid]["missed"] = 0
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": bbox, "history": deque(maxlen=SMOOTH_WINDOW), "missed": 0}
                assigned[i] = tid
                used_tracks.add(tid)

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > TRACK_MAX_MISSED:
                    del self.tracks[tid]
        return assigned

    def record(self, track_id, label):
        if track_id in self.tracks:
            self.tracks[track_id]["history"].append(label)

    def smoothed_label(self, track_id):
        """最近 SMOOTH_WINDOW 幀裡，出現次數最多的標籤（含 None=未知）才算數。"""
        if track_id not in self.tracks:
            return None
        hist = self.tracks[track_id]["history"]
        if not hist:
            return None
        return Counter(hist).most_common(1)[0][0]


def resolve_video_paths(sources):
    """來源可為影片檔或資料夾，展開成影片檔路徑清單。"""
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


def detect_all_instances(model, frame):
    """回傳這一幀所有偵測到的個體：[(kpts(17,2), kpt_conf(17,), bbox(4,), bbox_conf), ...]
    跟 KeypointDetector.detect() 不同——這裡刻意不挑「最佳一隻」，把所有偵測到的
    個體都回傳，因為身分驗證的前提就是要先看到雙貓同框時的『所有』候選者。
    kpts/kpt_conf 這裡其實只用來定位/除錯，真正的身分特徵改看 bbox 顏色。"""
    results = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    out = []
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return out
    has_boxes = results.boxes is not None and len(results.boxes) > 0
    for i in range(len(results.keypoints.xy)):
        kpts = results.keypoints.xy[i].cpu().numpy()
        if results.keypoints.conf is not None:
            kpt_conf = results.keypoints.conf[i].cpu().numpy()
        else:
            kpt_conf = np.ones(kpts.shape[0], dtype=np.float32)
        bbox = results.boxes.xyxy[i].cpu().numpy() if has_boxes and i < len(results.boxes) else None
        bconf = float(results.boxes.conf[i].cpu().numpy()) if has_boxes and i < len(results.boxes) else None
        out.append((kpts, kpt_conf, bbox, bconf))
    return out


def extract_color_histogram(frame, bbox):
    """從 bbox 區域裁切出貓的 HSV 顏色直方圖特徵（H-S 2D，L1 正規化，攤平成 1D）。
    bbox 先往內縮一個比例，避開邊緣背景雜訊；縮完太小則退回用原始 bbox。
    低飽和度（近灰階/反光）與過暗（陰影）像素不列入統計，避免被光線雜訊污染。
    回傳 None 代表這個 bbox 太小或裁切失敗，無法算出可用的直方圖。"""
    if bbox is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None

    ix1 = int(np.clip(x1 + bw * BBOX_INSET_RATIO, 0, w - 1))
    iy1 = int(np.clip(y1 + bh * BBOX_INSET_RATIO, 0, h - 1))
    ix2 = int(np.clip(x2 - bw * BBOX_INSET_RATIO, 0, w))
    iy2 = int(np.clip(y2 - bh * BBOX_INSET_RATIO, 0, h))
    if ix2 - ix1 < 5 or iy2 - iy1 < 5:
        # 縮太多幾乎沒剩（貓在畫面裡很小），退回用原始 bbox
        ix1, iy1 = int(max(0, x1)), int(max(0, y1))
        ix2, iy2 = int(min(w, x2)), int(min(h, y2))
        if ix2 - ix1 < 5 or iy2 - iy1 < 5:
            return None

    crop = frame[iy1:iy2, ix1:ix2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, HSV_S_MIN, HSV_V_MIN), (180, 255, HSV_V_MAX))
    if cv2.countNonZero(mask) < 20:
        # 幾乎整塊都被遮罩掉（例如毛色本身就低飽和度，或整塊過曝/陰影），退回不遮罩，至少留下訊號
        mask = None

    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def compute_sharpness(frame, bbox):
    """裁切區域的清晰度分數（灰階 Laplacian 變異數，數值越高越清晰）。
    只用在 enroll 階段過濾動態模糊/失焦的幀，避免這類雜訊幀混進顏色特徵基準——
    模糊會讓毛色邊界暈開，直方圖的色相/飽和度分布會被拖向不真實的中間值。"""
    if bbox is None:
        return 0.0
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, bbox[0])), int(max(0, bbox[1]))
    x2, y2 = int(min(w, bbox[2])), int(min(h, bbox[3]))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hist_distance(a, b):
    """Bhattacharyya 距離：0=完全相同，1=完全不同。"""
    ha = a.reshape(H_BINS, S_BINS)
    hb = b.reshape(H_BINS, S_BINS)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA))


def best_match_distance(hist, profile_hists):
    """回傳 hist 跟整批基準樣本裡「最近的一筆」的距離（最近鄰比對，比單一平均
    直方圖更抗光線變化——參考影片裡不同時段/光線的樣本各自保留，不會被平均模糊掉）。"""
    return min(hist_distance(hist, p) for p in profile_hists)


def compute_within_scale(hists):
    """這隻貓自己的樣本彼此之間的平均最近鄰距離，反映這份基準『天生鬆緊
    程度』（例如黑白貓不同角度露出的黑/白比例差很多，組內距離通常比毛色
    單一的橘貓大）。目前只用在 cmd_diagnose() 的分離度報告裡當參考數字，
    不影響 classify_instance() 的判定——曾經試過拿它把距離正規化再判定，
    結果矯枉過正：組內距離較大的貓（黑白貓）的正規化門檻被放得太寬，反而
    幾乎所有東西都被判成黑白貓，所以改回原始距離比較，這個函式純粹保留
    給診斷報告用。"""
    n = len(hists)
    if n < 2:
        return 0.02
    within = [min(hist_distance(hists[i], hists[j]) for j in range(n) if j != i) for i in range(n)]
    return max(float(np.mean(within)), 0.02)


def classify_instance(hist, profiles):
    """拿一個偵測到的個體的直方圖，跟每一隻已知貓的基準都比一次距離，取最近的
    當判定結果；如果連最近的都超過 UNKNOWN_DISTANCE_CEILING，判定為未知（key=None）。
    回傳 (best_key, best_dist, all_dists)。"""
    dists = {key: best_match_distance(hist, p["hists"]) for key, p in profiles.items()}
    best_key = min(dists, key=dists.get)
    best_dist = dists[best_key]
    if best_dist > UNKNOWN_DISTANCE_CEILING:
        return None, best_dist, dists
    return best_key, best_dist, dists


def load_model():
    model = YOLO(YOLO_MODEL_PATH)
    try:
        model.to(INFERENCE_DEVICE)
    except Exception as e:
        print(f"⚠ 無法使用 {INFERENCE_DEVICE}，改用 CPU（{e}）")
    return model


# ─── enroll：對每一隻設定好的貓建立顏色特徵基準 ─────────────────────────
def _enroll_one_cat(model, key, cfg):
    print(f"\n{'=' * 60}\n建立「{cfg['label']}」的顏色特徵基準\n{'=' * 60}")
    video_paths = resolve_video_paths(cfg["enroll_source"])
    if not video_paths:
        print(f"❌ 找不到「{cfg['label']}」的任何參考影片，略過")
        return

    samples = []
    valid_seen = 0
    total_frames_seen = 0
    skip_counts = {"multi_cat": 0, "no_cat": 0, "low_conf": 0, "too_small": 0, "blurry": 0}

    for vp in video_paths:
        print(f"處理參考影片: {vp.name}")
        cap = cv2.VideoCapture(str(vp))
        frame_h = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames_seen += 1
            if frame_h is None:
                frame_h = frame.shape[0]
            instances = detect_all_instances(model, frame)

            # ① 這幀應該剛好偵測到這隻貓一隻——0 隻或 >=2 隻都跳過，避免把
            #   「不是這隻貓」或空幀的雜訊混進特徵基準
            if len(instances) == 0:
                skip_counts["no_cat"] += 1
                continue
            if len(instances) > 1:
                skip_counts["multi_cat"] += 1
                continue

            _kpts, _kconf, bbox, bconf = instances[0]

            # ② 偵測信心不夠高：可能是邊緣/部分遮擋/曖昧案例，門檻比一般偵測嚴格
            if bconf is None or bconf < ENROLL_MIN_BBOX_CONF:
                skip_counts["low_conf"] += 1
                continue

            # ③ 貓在畫面裡太小（太遠/太小隻）：色塊解析度不足，顏色統計不可靠
            if bbox is not None and (bbox[3] - bbox[1]) < frame_h * ENROLL_MIN_BBOX_HEIGHT_RATIO:
                skip_counts["too_small"] += 1
                continue

            # ④ 動態模糊/失焦：毛色邊界暈開，直方圖會失真
            if compute_sharpness(frame, bbox) < ENROLL_MIN_SHARPNESS:
                skip_counts["blurry"] += 1
                continue

            hist = extract_color_histogram(frame, bbox)
            if hist is None:
                continue
            valid_seen += 1
            if valid_seen % ENROLL_SAMPLE_STRIDE == 0 and len(samples) < MAX_ENROLL_SAMPLES:
                samples.append(hist)
        cap.release()

    print(f"總共讀取 {total_frames_seen} 幀，通過所有品質關卡的有效幀: {valid_seen}")
    print(f"  跳過：沒偵測到貓 {skip_counts['no_cat']}　多貓同框 {skip_counts['multi_cat']}　"
          f"信心不足 {skip_counts['low_conf']}　太小/太遠 {skip_counts['too_small']}　模糊 {skip_counts['blurry']}")

    if not samples:
        print(f"❌「{cfg['label']}」沒有任何有效樣本，無法建立特徵基準（可能是參考影片品質太差，或門檻設太嚴）")
        return
    if len(samples) < 15:
        print(f"⚠ 有效樣本數偏少（{len(samples)}），基準可能不夠穩定，建議提供更長/更多支/光線更多變的參考影片")

    profile = {
        "label": cfg["label"],
        "hist_shape": [H_BINS, S_BINS],
        "samples": [s.tolist() for s in samples],
        "sample_count": len(samples),
        "source": [str(p) for p in video_paths],
    }
    out_path = profile_path_for(key)
    out_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    print(f"✓ 已建立「{cfg['label']}」顏色特徵基準（{len(samples)} 筆樣本，來自 {valid_seen} 個有效幀）-> {out_path}")


def cmd_enroll():
    model = load_model()
    for key, cfg in CAT_PROFILES.items():
        _enroll_one_cat(model, key, cfg)


# ─── verify：在測試影片上做多貓身分驗證 ─────────────────────────────────
def _load_profiles():
    profiles = {}
    for key, cfg in CAT_PROFILES.items():
        p = profile_path_for(key)
        if not p.exists():
            print(f"⚠ 找不到「{cfg['label']}」的基準檔（{p.name}），verify 時不會辨識這隻貓——請先把 RUN_MODE 改成 \"enroll\" 跑一次")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        hists = [np.array(s, dtype=np.float32) for s in data["samples"]]
        within_scale = compute_within_scale(hists)
        profiles[key] = {
            "label": cfg["label"],
            "short_label": cfg.get("short_label", key),
            "color": cfg.get("color", (0, 200, 0)),
            "hists": hists,
            "within_scale": within_scale,
        }
        print(f"已載入「{cfg['label']}」顏色特徵基準（{data['sample_count']} 筆樣本，within_scale={within_scale:.3f}）")
    return profiles


# ─── diagnose：只用已 enroll 好的基準檔量化「兩隻貓分不分得開」──────────
def cmd_diagnose():
    """不跑任何影片，純粹用 enroll 產生的基準檔做兩件事：

    1) 組內 vs 組間平均最近距離——組內距離該遠小於組間距離，兩者差距越大，
       代表這兩隻貓的顏色特徵天生越好分。

    2) Leave-one-out 交叉驗證：把每一筆樣本自己拿掉，用「剩下的基準」重新
       判定這一筆該歸類到哪，看能不能正確判回自己。這完全不需要真正的雙貓
       測試影片，直接告訴你「就算是 enroll 當下最理想的畫面，這個方法本身
       能達到的準確率上限大概是多少」——如果連 leave-one-out 準確率都不高，
       代表問題出在特徵本身（顏色不夠有區分度、背景污染直方圖等），加大
       SMOOTH_WINDOW 或調 UNKNOWN_DISTANCE_CEILING 不會有實質幫助，要先解決
       特徵品質；如果 leave-one-out 準確率很高，但實際 verify 測試表現差，
       代表問題是「enroll 資料的情境跟真實雙貓同框時的情境差太多」（例如
       enroll 影片背景/光線跟測試影片不同），該加強的是 enroll 素材的多樣性。
    """
    profiles = _load_profiles()
    if len(profiles) < 2:
        print("❌ 需要至少兩隻貓的基準檔才能做分離度診斷（目前只載入到 {} 份）".format(len(profiles)))
        return

    keys = list(profiles.keys())
    print(f"\n{'=' * 60}\n基準分離度診斷\n{'=' * 60}")

    print("\n[1] 組內 vs 組間平均最近距離（Bhattacharyya，0=完全相同 1=完全不同）")
    for key in keys:
        hists = profiles[key]["hists"]
        w_mean = profiles[key]["within_scale"]  # 跟 _load_profiles() 算的是同一份，不重算
        across = []
        for other_key in keys:
            if other_key == key:
                continue
            for h in profiles[other_key]["hists"]:
                across.append(best_match_distance(h, hists))
        a_mean = float(np.mean(across)) if across else float("nan")
        gap = a_mean - w_mean
        print(f"  「{profiles[key]['label']}」({len(hists)}筆)：組內={w_mean:.3f}　組間={a_mean:.3f}　"
              f"差距={gap:+.3f}{'  ⚠ 差距太小，兩隻貓顏色特徵重疊嚴重' if gap < 0.15 else ''}")

    print(f"\n[2] Leave-one-out 交叉驗證分類正確率（UNKNOWN_DISTANCE_CEILING={UNKNOWN_DISTANCE_CEILING}）")
    overall_correct, overall_total = 0, 0
    for key in keys:
        hists = profiles[key]["hists"]
        correct = wrong = unknown = 0
        for i, h in enumerate(hists):
            leave_out_profiles = {
                k: {"hists": (p["hists"][:i] + p["hists"][i + 1:]) if k == key else p["hists"]}
                for k, p in profiles.items()
            }
            pred_key, _dist, _dists = classify_instance(h, leave_out_profiles)
            if pred_key == key:
                correct += 1
            elif pred_key is None:
                unknown += 1
            else:
                wrong += 1
        n = len(hists)
        overall_correct += correct
        overall_total += n
        print(f"  「{profiles[key]['label']}」({n}筆)：正確 {correct} ({correct/n*100:.1f}%)　"
              f"誤判成其他貓 {wrong} ({wrong/n*100:.1f}%)　判為未知 {unknown} ({unknown/n*100:.1f}%)")

    overall_acc = overall_correct / max(overall_total, 1) * 100
    print(f"\n整體 leave-one-out 準確率: {overall_acc:.1f}%")
    if overall_acc >= 95:
        print("→ 特徵本身分得很開，實際 verify 表現不好的話，問題大概率出在 enroll 素材\n"
              "  的情境跟真實雙貓同框時差太多（背景/光線/角度），該加強 enroll 素材多樣性。")
    elif overall_acc >= 80:
        print("→ 特徵有一定區分度但不夠乾淨，可以先試著調整 UNKNOWN_DISTANCE_CEILING，\n"
              "  或改善裁切遮罩（現在只是 bbox 內縮，背景污染沒有精確排除）。")
    else:
        print("→ 特徵本身區分度不足，兩隻貓的顏色在目前的直方圖特徵下重疊嚴重。\n"
              "  建議：① 用更精確的遮罩（keypoint 凸包排除背景）取代目前簡單的 bbox 內縮，\n"
              "  ② 額外把 V（亮度）也納入直方圖維度，不要只用 H-S，\n"
              "  ③ 如果兩隻貓真的顏色太接近，顏色特徵這個方法可能先天就不適合，\n"
              "     需要考慮加回幾何特徵或其他訊號做輔助判斷。")


def _process_frame(frame, model, profiles, tracker):
    """對一幀畫面跑偵測 + 分類 + 追蹤平滑，回傳 decisions 清單。"""
    instances = detect_all_instances(model, frame)

    decisions = []
    for inst_idx, (_kpts, _kconf, bbox, bconf) in enumerate(instances):
        hist = extract_color_histogram(frame, bbox)
        if hist is None:
            decisions.append({
                "inst_idx": inst_idx, "bbox": bbox, "bconf": bconf,
                "raw_label": None, "best_dist": None, "dists": {}, "reason": "bbox太小/裁切失敗",
            })
            continue
        raw_key, best_dist, dists = classify_instance(hist, profiles)
        decisions.append({
            "inst_idx": inst_idx, "bbox": bbox, "bconf": bconf,
            "raw_label": raw_key, "best_dist": best_dist, "dists": dists, "reason": "",
        })

    # 用 IoU 把這一幀的偵測跟前幾幀的追蹤串起來，紀錄本幀的原始判定，
    # 再取最近 SMOOTH_WINDOW 幀的多數決當成真正顯示/輸出用的判定，
    # 避免單一幀的雜訊（模糊、bbox 帶到背景等）直接造成誤判。
    track_ids = tracker.update([d["bbox"] for d in decisions])
    for d, tid in zip(decisions, track_ids):
        d["track_id"] = tid
        if tid is not None:
            tracker.record(tid, d["raw_label"])
            d["label"] = tracker.smoothed_label(tid)
        else:
            d["label"] = None

    # 同一幀有兩個以上的個體被判成同一隻貓時（例如短暫誤判），只留距離最近
    # 的那個維持這個標籤，其餘降級為「未知」，避免同一隻貓同時出現兩個框。
    ambiguous = False
    by_label = defaultdict(list)
    for d in decisions:
        if d["label"] is not None:
            by_label[d["label"]].append(d)
    for _label, ds in by_label.items():
        if len(ds) > 1:
            ambiguous = True
            best = min(ds, key=lambda d: d["best_dist"] if d["best_dist"] is not None else float("inf"))
            for d in ds:
                if d is not best:
                    d["label"] = None

    return decisions, ambiguous


def _render_preview(frame, decisions, profiles, frame_idx, video_label, paused):
    disp = frame.copy()
    for d in decisions:
        if d["bbox"] is None:
            continue
        x1, y1, x2, y2 = map(int, d["bbox"])
        if d["label"] is not None:
            cat = profiles[d["label"]]
            color, short_name = cat["color"], cat["short_label"]
        else:
            color, short_name = UNKNOWN_COLOR, "unknown"
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3)

        conf_text = f"{d['bconf']:.2f}" if d["bconf"] is not None else "--"
        lines = [f"conf:{conf_text}  {short_name}"]
        draw_label_block(disp, x1, y1, lines, color)

    status = "PAUSED" if paused else "PLAY"
    cv2.putText(disp, f"[{status}] {video_label}  frame {frame_idx}  cats_detected={len(decisions)}",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return resize_with_letterbox(disp, PREVIEW_DISPLAY_SIZE)


def cmd_verify(sources, show):
    profiles = _load_profiles()
    if not profiles:
        print("❌ 沒有任何已建立的特徵基準，請先把 RUN_MODE 改成 \"enroll\" 跑一次")
        return

    video_paths = resolve_video_paths(sources)
    if not video_paths:
        print("❌ 找不到測試影片")
        return

    model = load_model()

    if show:
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, *PREVIEW_DISPLAY_SIZE)
        print("控制鍵: q=退出　space=暫停/繼續　1=上一部影片　2=下一部影片")

    current_video_idx = 0
    stop_requested = False

    while not stop_requested:
        video_path = video_paths[current_video_idx]
        print(f"\n驗證影片 [{current_video_idx + 1}/{len(video_paths)}]: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        tracker = SimpleTracker()  # 每支影片重新開始追蹤，不跨影片延續
        csv_rows = []
        frame_idx = 0
        multi_cat_frames = 0
        identified_frames = {key: 0 for key in profiles}
        ambiguous_frames = 0

        dist_cols = [f"dist_{key}" for key in profiles]
        paused = False
        switch_delta = 0
        last_disp = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    switch_delta = 1  # 自然播完，自動前進下一部
                    break
                frame_idx += 1
                decisions, ambiguous = _process_frame(frame, model, profiles, tracker)
                if ambiguous:
                    ambiguous_frames += 1
                if len(decisions) >= 2:
                    multi_cat_frames += 1
                for d in decisions:
                    if d["label"] is not None:
                        identified_frames[d["label"]] += 1

                for d in decisions:
                    row = {
                        "frame": frame_idx, "inst_idx": d["inst_idx"], "track_id": d["track_id"],
                        "bbox_conf": f"{d['bconf']:.3f}" if d["bconf"] is not None else "",
                        "label": profiles[d["label"]]["label"] if d["label"] is not None else "unknown",
                        "reason": d["reason"],
                        "bbox": ",".join(f"{v:.0f}" for v in d["bbox"]) if d["bbox"] is not None else "",
                    }
                    for key in profiles:
                        row[f"dist_{key}"] = f"{d['dists'][key]:.3f}" if key in d["dists"] else ""
                    csv_rows.append(row)

                if show:
                    last_disp = _render_preview(frame, decisions, profiles, frame_idx, video_path.name, paused)

            if show:
                if last_disp is not None:
                    cv2.imshow(PREVIEW_WINDOW_NAME, last_disp)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    stop_requested = True
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('2'):
                    switch_delta = 1
                    break
                elif key == ord('1'):
                    switch_delta = -1
                    break

        cap.release()

        out_path = Path(__file__).parent / f"{video_path.stem}_identity_verify.csv"
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "inst_idx", "track_id", "bbox_conf", "label", *dist_cols, "reason", "bbox"])
            w.writeheader()
            w.writerows(csv_rows)

        print(f"  總幀數: {frame_idx}")
        print(f"  雙貓（含以上）同框幀數: {multi_cat_frames} ({multi_cat_frames / max(frame_idx, 1) * 100:.1f}%)")
        for key, cnt in identified_frames.items():
            print(f"  判定為「{profiles[key]['label']}」的幀數: {cnt} ({cnt / max(frame_idx, 1) * 100:.1f}%)")
        print(f"  同幀重複判定、需裁決的幀數: {ambiguous_frames}")
        print(f"  ✓ 逐幀驗證結果已輸出: {out_path}")

        if stop_requested:
            break
        current_video_idx = (current_video_idx + switch_delta) % len(video_paths)

    if show:
        cv2.destroyAllWindows()


def main():
    if RUN_MODE == "enroll":
        cmd_enroll()
    elif RUN_MODE == "verify":
        cmd_verify(VERIFY_SOURCE, show=SHOW_PREVIEW)
    elif RUN_MODE == "diagnose":
        cmd_diagnose()
    else:
        print(f"❌ 未知的 RUN_MODE: {RUN_MODE!r}（應為 \"enroll\"、\"verify\" 或 \"diagnose\"）")


if __name__ == "__main__":
    main()
