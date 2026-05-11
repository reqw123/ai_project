"""
測試影片推論腳本（EMA 平滑版）- 使用指數移動平均對關鍵點座標平滑，提升穩定性
其餘功能與 test_video_inference.py 完全相同
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from collections import defaultdict
from datetime import datetime
from typing import Iterable

# 加入系統路徑
sys.path.insert(0, str(Path(__file__).parent))

from detectors.keypoint_detector import KeypointDetector
from processors.visualizer import Visualizer
from models.stgcn_model import (
    STGCN,
    flip_normalize,
    orientation_normalize,
    normalize_skeleton_coords,
    interpolate_missing,
    compute_bone_feature,
    build_feature_tensor,
)
from utils.constants import (
    BEHAVIOR_CLASSES,
    BEHAVIOR_TEXT_MAP,
    BEHAVIOR_COLORS,
    LOW_CONF_ID,
)

# 配置
# VIDEO_PATHS 每個元素可為：
# 1) 單一影片檔案路徑
# 2) 資料夾路徑（會遞迴搜尋常見影片副檔名）
VIDEO_PATHS = [
   r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\walk", 
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\lick",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\scratch",
    r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\主要測試\shake",
    #r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\5925936_Cute_Stray_1920x1080.mp4",#不要刪
    #r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\5913249_Cat_Feline_1920x1080.mp4",#不要刪
   #r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\摳圖影片集\5923455_Black_Cat_Runs_1920x1080.mp4",#不要刪
  # r"C:\Users\homec\OneDrive\圖片\貓咪圖像資料集\泛化測試"
]
YOLO_MODEL_PATH = r"C:\cat_pose\v11s_59.pt"
STGCN_MODEL_PATH = r"C:\cat_pose\gcn_pose\models\stgcn_best_xyv_conf_bone.pth"
INFERENCE_DEVICE = 'cuda'
YOLO_IMGSZ = 640  # 與 YOLO 訓練尺寸一致
YOLO_CONF_THRESHOLD = 0.5
STGCN_NORMALIZE = True
SEQUENCE_LENGTH = 16
CONFIDENCE_THRESHOLD = 0.5
TARGET_MODEL_FPS = 30.0  # 模型訓練/推論設計時基
ENABLE_FPS_DOWNSAMPLE = True  # 僅對 source_fps > TARGET_MODEL_FPS 做降採樣
CLASSIFY_STRIDE = 1  # 每幾個處理幀做一次分類（1=每幀）
FAST_PREVIEW_OVERLAY = True  # 預覽視窗直接縮放已繪製 overlay，減少重複繪圖
DISPLAY_WINDOW = True
WINDOW_NAME = "Cat Behavior Inference (EMA)"
DISPLAY_SIZE = (1080, 720)  # 視窗顯示解析度（寬, 高），設為 None 維持原始解析度
SAVE_OUTPUT_VIDEO = True  # 是否保存帶 overlay 的影片
OUTPUT_VIDEO_PATH = r"C:\paper\output\test_inference_overlay_ema.mp4"  # 輸出影片路徑
LOOP_PLAYBACK = True  # 是否循環播放
JITTER_CONF_THRESHOLD = 0.3  # 抖動統計只使用高於此信心值的關鍵點
REPORT_OUTPUT_PATH = r"C:\paper\output\inference_analysis_report_ema.md"  # 最終分析報告
RUN_MODE = 0  # 0: 啟動時選擇, 1: 只生成統計, 2: 只做視窗測試
JITTER_WARNING_THRESHOLD = 30.0  # 像素抖動警告閾值

# ===== 關鍵點顯示/統計門檻 =====
DRAW_KP_CONF_THRESHOLD = 0.25  # 畫骨架線段與關鍵點圓點用門檻（>此值才畫）

# ===== 幾何可視提示開關 =====
SHOW_HEAD_VECTOR_HINT = True   # 是否顯示頭部方向向量箭頭
SHOW_BODY_REGION_HINT = True   # 是否顯示 chest/hip 橢圓中心區域

# ===== 頭部朝向身體區域判斷與可視化參數 =====
GEOM_KP_CONF_THRESHOLD = 0.25  # 幾何提示所需關鍵點門檻（耳/胸/臀）
GEOM_NOSE_CONF_THRESHOLD = 0.5  # 鼻尖門檻獨立設高，避免頭向量抖動
BODY_REGION_RADIUS_RATIO = 0.30
BODY_REGION_RADIUS_MIN_PX = 12.0
BODY_REGION_AXIS_RATIO = 0.65
HEAD_RAY_LENGTH_RATIO = 1.60
HEAD_RAY_MIN_PX = 60.0
HEAD_TARGET_LABEL = "HEAD TARGET"

# ===== EMA 平滑設定 =====
# alpha 越大 → 越貼近原始偵測值（響應快、平滑少）
# alpha 越小 → 越平滑（延遲多、噪音少）
# 建議範圍：0.2（非常平滑）~ 0.6（輕微平滑）
EMA_ALPHA = 1.0  # 須與 train_gcn.py 的 KP_EMA_ALPHA 保持一致

# 7 通道特徵：x, y, conf, vx, vy, bone_x, bone_y
FEATURE_DIM = 7

# 17 關鍵點名稱映射（根據 YOLO-Pose v11 cat skeleton）
KEYPOINT_NAMES = [
    "Nose",           # 0: 鼻尖
    "Left_Ear",       # 1: 左耳
    "Right_Ear",      # 2: 右耳
    "Chest",          # 3: 前胸
    "Mid_Back",       # 4: 中背
    "Hip",            # 5: 髖部
    "LF_Elbow",       # 6: 左前肢肘
    "LF_Paw",         # 7: 左前肢掌
    "RF_Elbow",       # 8: 右前肢肘
    "RF_Paw",         # 9: 右前肢掌
    "LH_Knee",        # 10: 左後肢膝
    "LH_Paw",         # 11: 左後肢掌
    "RH_Knee",        # 12: 右後肢膝
    "RH_Paw",         # 13: 右後肢掌
    "Tail_Root",      # 14: 尾根
    "Tail_Mid",       # 15: 尾中
    "Tail_Tip",       # 16: 尾尖
]

# 17 關鍵點中文名稱
KEYPOINT_NAMES_ZH = [
    "鼻子",              # 0: nose
    "左耳尖",           # 1: left_ear_tip
    "右耳尖",           # 2: right_ear_tip
    "胸口",             # 3: 前胸（前肢附著點）
    "中背",             # 4: 身體中背
    "臀部",             # 5: hip
    "左前腿肘部",       # 6: left_front_elbow
    "左前爪",           # 7: left_front_paw
    "右前腿肘部",       # 8: right_front_elbow
    "右前爪",           # 9: right_front_paw
    "左後腿膝部",       # 10: left_hind_knee
    "左後爪",           # 11: left_hind_paw
    "右後腿膝部",       # 12: right_hind_knee
    "右後爪",           # 13: right_hind_paw
    "尾巴根部",         # 14: tail_base
    "尾巴中段",         # 15: tail_mid
    "尾巴尖端",         # 16: tail_tip
]

# ===== test2.py 骨架視覺樣式 =====
_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (3, 8), (8, 9),
    (5, 10), (10, 11), (5, 12), (12, 13),
    (5, 14), (14, 15), (15, 16),
]

_KP_COLORS = [
    (255, 80, 80), (255, 160, 40), (255, 160, 40),
    (255, 255, 60), (200, 255, 60), (100, 255, 100),
    (60, 200, 255), (60, 120, 255), (60, 200, 255), (60, 120, 255),
    (180, 80, 255), (120, 40, 255), (180, 80, 255), (120, 40, 255),
    (80, 220, 180), (60, 180, 140), (40, 140, 100),
]

_EDGE_COLORS = [
    (255, 120, 60), (255, 120, 60), (255, 120, 60),
    (220, 220, 60), (200, 220, 60), (160, 220, 60),
    (102, 85, 255), (102, 85, 255), (255, 68, 204), (255, 68, 204),
    (255, 170, 34), (255, 170, 34), (0, 153, 255), (0, 153, 255),
    (80, 200, 160), (60, 170, 130), (40, 140, 100),
]

KP_NOSE = 0
KP_LEFT_EAR = 1
KP_RIGHT_EAR = 2
KP_CHEST = 3
KP_HIP = 5

SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg", ".webm"
}

# COCO 17 點骨架父節點索引，需與 train_gcn.py 保持一致
PARENTS = np.array([0, 0, 0, 0, 3, 4, 3, 6, 3, 8, 5, 10, 5, 12, 5, 14, 15], dtype=np.int64)


def resolve_video_paths(video_sources: Iterable[str]):
    """將來源清單展開成影片檔路徑；來源可為影片檔或資料夾。"""
    resolved = []
    seen = set()

    for src in video_sources:
        p = Path(src).expanduser()

        if p.is_file():
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTS:
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    resolved.append(str(p))
            else:
                print(f"⚠ 非支援影片副檔名，略過: {p}")
            continue

        if p.is_dir():
            matched = sorted(
                [
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTS
                ]
            )
            if not matched:
                print(f"⚠ 資料夾內未找到影片，略過: {p}")
            for f in matched:
                key = str(f.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    resolved.append(str(f))
            continue

        print(f"⚠ 路徑不存在，略過: {p}")

    return resolved


def compute_ui_scale(width, height, base_width=1920.0, base_height=1080.0):
    """依影像對角線估算 UI 縮放，讓不同解析度下 overlay 視覺一致。"""
    diag = float(np.hypot(max(1.0, float(width)), max(1.0, float(height))))
    base_diag = float(np.hypot(base_width, base_height))
    scale = diag / max(base_diag, 1.0)
    return float(np.clip(scale, 0.65, 2.4))


def scale_px(value, ui_scale, min_px=1):
    """將像素值依 UI 縮放後取整，並限制最小值。"""
    return max(int(min_px), int(round(float(value) * float(ui_scale))))


def resize_with_letterbox(image, target_size, pad_color=(20, 20, 20)):
    """等比例縮放到 target_size，剩餘區域以 pad_color 補齊。"""
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]

    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return cv2.resize(image, target_size), 1.0, 0, 0

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((target_h, target_w, 3), pad_color, dtype=image.dtype)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas, scale, offset_x, offset_y


def draw_no_cat_overlay(frame, text="No cat detected"):
    """依畫面解析度自適應繪製無偵測提示文字。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    x = scale_px(12, ui_scale, min_px=8)
    y = scale_px(34, ui_scale, min_px=20)
    font_scale = 0.62 * ui_scale
    outline = scale_px(3, ui_scale, min_px=2)
    thickness = scale_px(1, ui_scale, min_px=1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return frame


def build_7ch_features(sequence_xy, conf_seq):
    """
    構建 7 通道特徵（x, y, conf, vx, vy, bone_x, bone_y）
    已改為使用共享函數以確保跨腳本一致性
    """
    return build_feature_tensor(sequence_xy, conf_seq, "xyv_conf_bone")


class STGCN7chClassifier:
    def __init__(self, model_path, device='cuda', sequence_length=32, num_classes=4, normalize=True):
        self.device = torch.device(device if device != 'cuda' or torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.normalize = normalize
        self.model = STGCN(
            num_classes=num_classes,
            in_channels=FEATURE_DIM,
            num_joints=17,
            spatial_kernel_size=3,
            temporal_kernel_size=9,
            num_layers=3,
        ).to(self.device)

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 7 通道模型權重: {model_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # 允許與舊 checkpoint 相容；若模型新增 shake head 分支，缺少該權重時不阻斷載入
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print(f"✓ 7通道 ST-GCN 模型已載入: {model_path}")

    def normalize_keypoints(self, keypoints_sequence):
        seq = flip_normalize(keypoints_sequence)
        seq = orientation_normalize(seq)
        seq = normalize_skeleton_coords(seq)
        return seq

    def classify(self, keypoints_sequence, conf_sequence):
        if keypoints_sequence.shape[0] < self.sequence_length:
            return None, 0.0, np.zeros(self.num_classes, dtype=np.float32)

        seq_xy = keypoints_sequence[-self.sequence_length:, :, :2].copy()
        seq_conf = conf_sequence[-self.sequence_length:, :].copy()

        if self.normalize:
            seq_xy = self.normalize_keypoints(seq_xy)

        seq_7ch = build_7ch_features(seq_xy, seq_conf)
        seq_tensor = torch.from_numpy(seq_7ch).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_conf = float(probs[pred_id])
        return pred_id, pred_conf, probs


def _ray_hits_ellipse(ear_center, head_dir, body_center, body_normal, body_axis_unit, rx, ry):
    """檢查從 ear_center 沿 head_dir 的射線是否與局部身體橢圓區域相交。"""
    rel = ear_center - body_center
    u0 = float(np.dot(rel, body_normal))
    v0 = float(np.dot(rel, body_axis_unit))
    du = float(np.dot(head_dir, body_normal))
    dv = float(np.dot(head_dir, body_axis_unit))

    rx2 = max(rx * rx, 1e-9)
    ry2 = max(ry * ry, 1e-9)
    a = (du * du) / rx2 + (dv * dv) / ry2
    b = 2.0 * (u0 * du / rx2 + v0 * dv / ry2)
    c = (u0 * u0) / rx2 + (v0 * v0) / ry2 - 1.0

    if c <= 0.0:
        return True
    if abs(a) < 1e-12:
        return False

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = float(np.sqrt(max(0.0, disc)))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    return (t1 >= 0.0) or (t2 >= 0.0)


def compute_head_body_target_geometry(kpts, kpt_conf):
    """建立頭部方向向量、身體中心與局部區域，並判斷是否指向該區域。"""
    if kpts is None or kpt_conf is None:
        return None

    left_ok = float(kpt_conf[KP_LEFT_EAR]) > GEOM_KP_CONF_THRESHOLD
    right_ok = float(kpt_conf[KP_RIGHT_EAR]) > GEOM_KP_CONF_THRESHOLD
    nose_ok = float(kpt_conf[KP_NOSE]) >= GEOM_NOSE_CONF_THRESHOLD
    chest_ok = float(kpt_conf[KP_CHEST]) > GEOM_KP_CONF_THRESHOLD
    hip_ok = float(kpt_conf[KP_HIP]) > GEOM_KP_CONF_THRESHOLD
    if not (left_ok and right_ok and nose_ok and chest_ok and hip_ok):
        return None

    left_ear = np.asarray(kpts[KP_LEFT_EAR], dtype=np.float64)
    right_ear = np.asarray(kpts[KP_RIGHT_EAR], dtype=np.float64)
    nose = np.asarray(kpts[KP_NOSE], dtype=np.float64)
    chest = np.asarray(kpts[KP_CHEST], dtype=np.float64)
    hip = np.asarray(kpts[KP_HIP], dtype=np.float64)

    ear_center = 0.5 * (left_ear + right_ear)
    head_vec = nose - ear_center
    head_norm = float(np.linalg.norm(head_vec))
    if head_norm < 1e-6:
        return None
    head_dir = head_vec / head_norm

    body_axis = hip - chest
    body_len = float(np.linalg.norm(body_axis))
    if body_len < 1e-6:
        return None
    body_axis_unit = body_axis / body_len
    body_normal = np.array([-body_axis_unit[1], body_axis_unit[0]], dtype=np.float64)

    body_center = 0.5 * (chest + hip)
    rx = max(BODY_REGION_RADIUS_MIN_PX, BODY_REGION_RADIUS_RATIO * body_len)
    ry = max(4.0, BODY_REGION_AXIS_RATIO * rx)

    region_left = body_center + body_normal * rx
    region_right = body_center - body_normal * rx

    ray_len = max(HEAD_RAY_MIN_PX, HEAD_RAY_LENGTH_RATIO * body_len)
    ray_end = ear_center + head_dir * ray_len

    hit = _ray_hits_ellipse(ear_center, head_dir, body_center, body_normal, body_axis_unit, rx, ry)

    return {
        "ear_center": ear_center,
        "nose": nose,
        "ray_end": ray_end,
        "body_center": body_center,
        "body_axis_unit": body_axis_unit,
        "region_left": region_left,
        "region_right": region_right,
        "region_rx": rx,
        "region_ry": ry,
        "hit": hit,
    }


def draw_test2_style_overlay(
    frame,
    kpts,
    kpt_conf,
    bbox,
    behavior_id,
    confidence,
    probs,
    visualizer,
    show_info=True,
    conf_thresh=DRAW_KP_CONF_THRESHOLD,
):
    """使用 test2.py 的骨架外觀，並沿用既有行為資訊 HUD。"""
    h, w = frame.shape[:2]
    ui_scale = compute_ui_scale(w, h)
    bbox_thickness = scale_px(1, ui_scale, min_px=1)
    edge_thickness = scale_px(2, ui_scale, min_px=1)
    kp_outer_radius = scale_px(4, ui_scale, min_px=2)
    kp_inner_radius = max(1, kp_outer_radius - 1)
    center_outer_radius = scale_px(4, ui_scale, min_px=2)
    center_inner_radius = max(1, center_outer_radius - 2)
    text_font_scale = 0.52 * ui_scale
    text_outline = scale_px(3, ui_scale, min_px=2)
    text_thickness = scale_px(1, ui_scale, min_px=1)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), bbox_thickness)

    for ei, (a, b) in enumerate(_SKELETON_EDGES):
        # 骨架線段：兩端關鍵點都要高於顯示門檻才畫
        if float(kpt_conf[a]) > conf_thresh and float(kpt_conf[b]) > conf_thresh:
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            col = _EDGE_COLORS[ei] if ei < len(_EDGE_COLORS) else (180, 180, 180)
            cv2.line(frame, pa, pb, col, edge_thickness, cv2.LINE_AA)

    for i in range(min(17, len(kpts))):
        # 關鍵點圓點：該點信心高於顯示門檻才畫
        if float(kpt_conf[i]) > conf_thresh:
            cx, cy = int(kpts[i][0]), int(kpts[i][1])
            col = _KP_COLORS[i] if i < len(_KP_COLORS) else (200, 200, 200)
            cv2.circle(frame, (cx, cy), kp_outer_radius, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), kp_inner_radius, col, -1)

    if not show_info:
        return frame

    target_geom = None
    if SHOW_HEAD_VECTOR_HINT or SHOW_BODY_REGION_HINT:
        target_geom = compute_head_body_target_geometry(kpts, kpt_conf)

    if target_geom is not None:
        ear_c = target_geom["ear_center"]
        nose_p = target_geom["nose"]
        ray_end = target_geom["ray_end"]
        body_c = target_geom["body_center"]
        region_left = target_geom["region_left"]
        region_right = target_geom["region_right"]
        rx = target_geom["region_rx"]
        ry = target_geom["region_ry"]
        hit_region = bool(target_geom["hit"])

        if SHOW_HEAD_VECTOR_HINT:
            ear_pt = (int(ear_c[0]), int(ear_c[1]))
            nose_pt = (int(nose_p[0]), int(nose_p[1]))
            ray_pt = (int(ray_end[0]), int(ray_end[1]))
            cv2.arrowedLine(frame, ear_pt, nose_pt, (255, 230, 0), edge_thickness, cv2.LINE_AA, tipLength=0.22)
            cv2.arrowedLine(frame, ear_pt, ray_pt, (255, 170, 0), bbox_thickness, cv2.LINE_AA, tipLength=0.08)

        if SHOW_BODY_REGION_HINT:
            body_pt = (int(body_c[0]), int(body_c[1]))
            cv2.circle(frame, body_pt, center_outer_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, body_pt, center_inner_radius, (0, 180, 255), -1, cv2.LINE_AA)

            axes = (max(2, int(rx)), max(2, int(ry)))
            axis = target_geom["body_axis_unit"]
            angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

            overlay_region = frame.copy()
            fill_color = (30, 210, 80) if hit_region else (80, 120, 240)
            cv2.ellipse(overlay_region, body_pt, axes, angle_deg, 0, 360, fill_color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay_region, 0.28, frame, 0.72, 0, frame)
            cv2.ellipse(frame, body_pt, axes, angle_deg, 0, 360, (230, 230, 230), bbox_thickness, cv2.LINE_AA)

            rl_pt = (int(region_left[0]), int(region_left[1]))
            rr_pt = (int(region_right[0]), int(region_right[1]))
            cv2.circle(frame, rl_pt, max(1, kp_inner_radius - 1), (220, 220, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, rr_pt, max(1, kp_inner_radius - 1), (220, 220, 220), -1, cv2.LINE_AA)

            target_text = f"{HEAD_TARGET_LABEL}: {'BODY REGION' if hit_region else 'NOT TARGETING'}"
            text_x = scale_px(12, ui_scale, min_px=8)
            text_y = max(scale_px(18, ui_scale, min_px=12), frame.shape[0] - scale_px(14, ui_scale, min_px=10))
            cv2.putText(frame, target_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, (0, 0, 0), text_outline, cv2.LINE_AA)
            cv2.putText(frame, target_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, (40, 230, 40) if hit_region else (210, 210, 210), text_thickness, cv2.LINE_AA)

    yolo_model_path = getattr(visualizer, 'yolo_model_path', None)
    frame_number = getattr(visualizer, 'frame_idx', None)
    fps_val = getattr(visualizer, 'fps', None)
    if behavior_id == LOW_CONF_ID:
        visualizer.draw_prediction_on_frame(
            frame,
            'Normal',
            0.0,
            (200, 200, 200),
            frame_number=frame_number,
            fps=fps_val,
            yolo_model_path=yolo_model_path,
        )
        if probs is not None and any(float(p) > 0 for p in probs):
            visualizer.draw_probability_bars(frame, probs, BEHAVIOR_CLASSES)
    elif behavior_id is not None and confidence > 0:
        behavior_name = BEHAVIOR_CLASSES[behavior_id] if 0 <= behavior_id < len(BEHAVIOR_CLASSES) else str(behavior_id)
        visualizer.draw_prediction_on_frame(
            frame,
            behavior_name,
            confidence,
            BEHAVIOR_COLORS.get(behavior_id, (255, 255, 255)),
            frame_number=frame_number,
            fps=fps_val,
            yolo_model_path=yolo_model_path,
        )
        visualizer.draw_probability_bars(frame, probs if probs is not None else np.zeros(4, dtype=np.float32), BEHAVIOR_CLASSES)

    return frame


def print_jitter_report(title, jitter_px, jitter_norm, valid_counts, pair_counts):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    all_px = [v for arr in jitter_px for v in arr]
    all_norm = [v for arr in jitter_norm for v in arr]
    total_valid = int(np.sum(valid_counts))
    total_pairs = int(np.sum(pair_counts))

    if not all_px:
        print("無足夠資料計算抖動（可能關鍵點信心不足或連續幀不足）")
        return

    print("[全域抖動指標]")
    print(f"  樣本數(像素): {len(all_px)}")
    print(f"  平均: {np.mean(all_px):.3f} px")
    print(f"  標準差: {np.std(all_px):.3f} px")
    print(f"  P95: {np.percentile(all_px, 95):.3f} px")
    print(f"  最大值: {np.max(all_px):.3f} px")
    print(f"  有效關鍵點數: {total_valid}")
    print(f"  連續可比較配對數: {total_pairs}")

    if all_norm:
        print(f"  正規化平均(除以bbox對角線): {np.mean(all_norm):.5f}")
        print(f"  正規化P95: {np.percentile(all_norm, 95):.5f}")

    print("\n[17關鍵點逐點統計]")
    print("  idx | valid | pairs | mean_px | std_px | p95_px | max_px | mean_norm")
    for i in range(17):
        if jitter_px[i]:
            mean_px = np.mean(jitter_px[i])
            std_px = np.std(jitter_px[i])
            p95_px = np.percentile(jitter_px[i], 95)
            max_px = np.max(jitter_px[i])
        else:
            mean_px = std_px = p95_px = max_px = 0.0

        mean_norm = np.mean(jitter_norm[i]) if jitter_norm[i] else 0.0

        print(
            f"  {i:>3d} | {int(valid_counts[i]):>5d} | {int(pair_counts[i]):>5d} | "
            f"{mean_px:>7.3f} | {std_px:>6.3f} | {p95_px:>6.3f} | {max_px:>6.3f} | {mean_norm:>9.5f}"
        )


def generate_report_file(report_path, recorded_video_stats):
    def infer_expected_behavior(video_path):
        video_name = Path(video_path).stem.lower()
        if "run" in video_name or "walk" in video_name:
            return 0
        if "scratch" in video_name:
            return 1
        if "lick" in video_name:
            return 2
        if "shake" in video_name:
            return 3
        return None

    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Cat Behavior Inference Analysis Report (EMA Smoothed)")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"EMA Alpha: {EMA_ALPHA} (keypoint smoothing applied before model input and jitter measurement)")
    lines.append("")

    if not recorded_video_stats:
        lines.append("No completed video pass was recorded.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    # ==================== 模式1主行為信心摘要（置頂） ====================
    lines.append("---")
    lines.append("## 模式1重點摘要（以信心值為準）")
    lines.append("")
    lines.append("### A. 各影片主導行為平均準確率（Confidence）")
    lines.append("")
    lines.append("說明：每支影片先找出主導行為（預測次數最多），再計算該主導行為所有預測的平均信心值。")
    lines.append("最終平均準確率 = 所有影片主導行為平均信心值的算術平均（每支影片權重相同）。")
    lines.append("")
    lines.append("| 影片編號 | 影片檔名 | 預期行為 | 主導行為 | 主導行為平均信心 | 主導行為預測數 |")
    lines.append("|---:|:---|:---|:---|---:|---:|")

    dominant_conf_values = []
    dominant_conf_by_behavior = defaultdict(list)
    for vid_idx in sorted(recorded_video_stats.keys()):
        s = recorded_video_stats[vid_idx]
        behavior_counts = s["behavior_counts"]
        behavior_confidences_by_class = s.get("behavior_confidences_by_class", [[] for _ in range(4)])
        total_behavior = int(np.sum(behavior_counts))

        expected_behavior = infer_expected_behavior(s["video_path"])
        expected_text = BEHAVIOR_CLASSES[expected_behavior] if expected_behavior is not None else "unknown"

        if total_behavior > 0:
            dominant_bid = int(np.argmax(behavior_counts))
            dominant_behavior = BEHAVIOR_CLASSES[dominant_bid]
            dominant_conf_list = behavior_confidences_by_class[dominant_bid]
            if dominant_conf_list:
                dominant_conf = float(np.mean(dominant_conf_list))
                dominant_conf_str = f"{dominant_conf * 100:.2f}%"
                dominant_conf_values.append(dominant_conf)
                dominant_conf_by_behavior[dominant_bid].append(dominant_conf)
            else:
                dominant_conf_str = "N/A"
            dominant_count = int(behavior_counts[dominant_bid])
        else:
            dominant_behavior = "N/A"
            dominant_conf_str = "N/A"
            dominant_count = 0

        lines.append(
            f"| {vid_idx} | {Path(s['video_path']).name} | {expected_text} | {dominant_behavior} | "
            f"{dominant_conf_str} | {dominant_count} |"
        )

    lines.append("")
    if dominant_conf_values:
        lines.append(f"- **最終平均準確率（Confidence）**：{np.mean(dominant_conf_values) * 100:.2f}%")
    else:
        lines.append("- **最終平均準確率（Confidence）**：N/A")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### B. 各主導行為的平均信心值")
    lines.append("")
    lines.append("| 行為 | 主導平均信心 | 影片數 |")
    lines.append("|:---|---:|---:|")
    for bid in range(4):
        if dominant_conf_by_behavior[bid]:
            mean_conf = float(np.mean(dominant_conf_by_behavior[bid]))
            lines.append(f"| {BEHAVIOR_CLASSES[bid]} | {mean_conf * 100:.2f}% | {len(dominant_conf_by_behavior[bid])} |")
        else:
            lines.append(f"| {BEHAVIOR_CLASSES[bid]} | N/A | 0 |")
    lines.append("")

    # ==================== 關鍵點名稱映射 ====================
    lines.append("---")
    lines.append("")
    lines.append("## Keypoint Index Reference")
    lines.append("")
    lines.append("| Index | Name | Body Part |")
    lines.append("|---:|:---|:---|")
    for i, name in enumerate(KEYPOINT_NAMES):
        lines.append(f"| {i} | {name} | {KEYPOINT_NAMES_ZH[i]} |")
    lines.append("")

    # ==================== 全域統計 ====================
    lines.append("---")
    lines.append("")
    lines.append("## Global Summary (All Completed Videos)")
    lines.append("")

    # 計算全域行為統計
    global_behavior_counts = np.zeros(4, dtype=np.int64)
    global_confidences = []
    global_total_frames = 0
    global_cat_frames = 0
    global_no_cat_frames = 0

    for s in recorded_video_stats.values():
        global_behavior_counts += s["behavior_counts"]
        global_confidences.extend(s["behavior_confidences"])
        global_total_frames += s["processed_frames"]
        global_cat_frames += s["frames_with_cat"]
        global_no_cat_frames += s["frames_without_cat"]

    total_predictions = int(np.sum(global_behavior_counts))

    lines.append(f"- **Completed videos**: {len(recorded_video_stats)}")
    lines.append(f"- **Total processed frames**: {global_total_frames}")
    lines.append(f"- **Cat detected frames**: {global_cat_frames} ({global_cat_frames/global_total_frames*100:.2f}%)")
    lines.append(f"- **Total behavior predictions**: {total_predictions}")
    lines.append("")
    lines.append("**Behavior Distribution (Global):**")
    lines.append("")
    for bid in range(4):
        cnt = int(global_behavior_counts[bid])
        pct = (cnt / total_predictions * 100.0) if total_predictions > 0 else 0.0
        lines.append(f"- {BEHAVIOR_TEXT_MAP[bid]} ({BEHAVIOR_CLASSES[bid]}): {cnt} ({pct:.2f}%)")
    lines.append("")

    if global_confidences:
        lines.append("**Confidence Statistics (Global):**")
        lines.append("")
        lines.append(f"- Mean: {np.mean(global_confidences) * 100:.2f}%")
        lines.append(f"- Median: {np.median(global_confidences) * 100:.2f}%")
        lines.append(f"- Min: {np.min(global_confidences) * 100:.2f}%")
        lines.append(f"- Max: {np.max(global_confidences) * 100:.2f}%")
        lines.append("")

    # ==================== Top-5 高抖動關鍵點 ====================
    global_kp_jitter = []
    for i in range(17):
        all_jitter = []
        for s in recorded_video_stats.values():
            all_jitter.extend(s["jitter_px"][i])
        if all_jitter:
            mean_jitter = np.mean(all_jitter)
            max_jitter = np.max(all_jitter)
            global_kp_jitter.append((i, KEYPOINT_NAMES[i], mean_jitter, max_jitter))

    global_kp_jitter.sort(key=lambda x: x[2], reverse=True)

    lines.append("**Top-5 Highest Jitter Keypoints (Global, EMA smoothed):**")
    lines.append("")
    lines.append("| Rank | Index | Name | 中文名稱 | Mean Jitter (px) | Max Jitter (px) |")
    lines.append("|---:|---:|:---|:---|---:|---:|")
    for rank, (idx, name, mean_j, max_j) in enumerate(global_kp_jitter[:5], start=1):
        lines.append(f"| {rank} | {idx} | {name} | {KEYPOINT_NAMES_ZH[idx]} | {mean_j:.4f} | {max_j:.4f} |")
    lines.append("")

    # ==================== 異常警告 ====================
    lines.append("---")
    lines.append("")
    lines.append("## Warnings & Anomalies")
    lines.append("")

    warnings = []

    for vid_idx, s in recorded_video_stats.items():
        behavior_counts = s["behavior_counts"]
        total_behavior = int(np.sum(behavior_counts))

        if total_behavior == 0:
            continue

        expected_behavior = infer_expected_behavior(s["video_path"])

        if expected_behavior is not None:
            actual_dominant = int(np.argmax(behavior_counts))
            dominant_pct = behavior_counts[actual_dominant] / total_behavior * 100.0

            if actual_dominant != expected_behavior:
                warnings.append(
                    f"⚠️ **Video [{vid_idx}]** `{Path(s['video_path']).name}`: "
                    f"Expected `{BEHAVIOR_CLASSES[expected_behavior]}` but got "
                    f"`{BEHAVIOR_CLASSES[actual_dominant]}` ({dominant_pct:.1f}%)"
                )
            elif dominant_pct < 80.0:
                other_behaviors = []
                for bid in range(4):
                    if bid != actual_dominant and behavior_counts[bid] > 0:
                        other_pct = behavior_counts[bid] / total_behavior * 100.0
                        if other_pct > 15.0:
                            other_behaviors.append(f"{BEHAVIOR_CLASSES[bid]} {other_pct:.1f}%")
                if other_behaviors:
                    warnings.append(
                        f"⚠️ **Video [{vid_idx}]** `{Path(s['video_path']).name}`: "
                        f"Low dominant behavior purity ({dominant_pct:.1f}%), "
                        f"mixed with {', '.join(other_behaviors)}"
                    )

        confidences = s["behavior_confidences"]
        if confidences:
            mean_conf = np.mean(confidences) * 100.0
            if mean_conf < 70.0:
                warnings.append(
                    f"⚠️ **Video [{vid_idx}]** `{Path(s['video_path']).name}`: "
                    f"Low mean confidence ({mean_conf:.1f}%)"
                )

        for i in range(17):
            if s["jitter_px"][i]:
                max_jitter = np.max(s["jitter_px"][i])
                if max_jitter > JITTER_WARNING_THRESHOLD:
                    warnings.append(
                        f"⚠️ **Video [{vid_idx}]** Keypoint `{i} ({KEYPOINT_NAMES[i]} / {KEYPOINT_NAMES_ZH[i]})`: "
                        f"High jitter max = {max_jitter:.2f} px (threshold: {JITTER_WARNING_THRESHOLD} px)"
                    )

    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("✅ No anomalies detected.")
    lines.append("")

    # ==================== Overview ====================
    lines.append("---")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Completed videos: {len(recorded_video_stats)}")
    lines.append("")

    # ==================== 每影片詳細統計 ====================
    for vid_idx in sorted(recorded_video_stats.keys()):
        s = recorded_video_stats[vid_idx]
        behavior_counts = s["behavior_counts"]
        total_behavior = int(np.sum(behavior_counts))
        confidences = s["behavior_confidences"]

        lines.append("---")
        lines.append("")
        lines.append(f"## Video [{vid_idx}] {s['video_path']}")
        lines.append("")
        lines.append("### Playback Summary")
        lines.append("")
        lines.append(f"- Resolution: {s['width']}x{s['height']}")
        lines.append(f"- FPS used: {s['fps']:.2f}")
        lines.append(f"- Total frames (source): {s['total_frames']}")
        lines.append(f"- Processed frames (first full pass): {s['processed_frames']}")
        lines.append(f"- Cat detected frames: {s['frames_with_cat']}")
        lines.append(f"- No cat frames: {s['frames_without_cat']}")
        lines.append("")

        lines.append("### Behavior Statistics")
        lines.append("")
        if total_behavior == 0:
            lines.append("- No high-confidence behavior prediction in this video.")
        else:
            for bid in range(4):
                cnt = int(behavior_counts[bid])
                pct = (cnt / total_behavior * 100.0) if total_behavior > 0 else 0.0
                lines.append(
                    f"- {BEHAVIOR_TEXT_MAP[bid]} ({BEHAVIOR_CLASSES[bid]}): {cnt} ({pct:.2f}%)"
                )
            if confidences:
                lines.append(f"- Mean confidence: {np.mean(confidences) * 100:.2f}%")
                lines.append(f"- Min confidence: {np.min(confidences) * 100:.2f}%")
                lines.append(f"- Max confidence: {np.max(confidences) * 100:.2f}%")
        lines.append("")

        jp = s["jitter_px"]
        jn = s["jitter_norm"]
        vc = s["valid_counts"]
        pc = s["pair_counts"]

        all_px = [v for arr in jp for v in arr]
        all_norm = [v for arr in jn for v in arr]

        lines.append("### Keypoint Jitter Summary (EMA smoothed)")
        lines.append("")
        if not all_px:
            lines.append("- No valid jitter samples.")
        else:
            lines.append(f"- Pixel mean: {np.mean(all_px):.4f}")
            lines.append(f"- Pixel std: {np.std(all_px):.4f}")
            lines.append(f"- Pixel p95: {np.percentile(all_px, 95):.4f}")
            lines.append(f"- Pixel max: {np.max(all_px):.4f}")
            if all_norm:
                lines.append(f"- Normalized mean: {np.mean(all_norm):.6f}")
                lines.append(f"- Normalized p95: {np.percentile(all_norm, 95):.6f}")
        lines.append("")

        lines.append("### 17-Keypoint Details")
        lines.append("")
        lines.append("| kp_idx | Name | 中文名稱 | valid | pairs | mean_px | std_px | p95_px | max_px | mean_norm |")
        lines.append("|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
        for i in range(17):
            if jp[i]:
                mean_px = np.mean(jp[i])
                std_px = np.std(jp[i])
                p95_px = np.percentile(jp[i], 95)
                max_px = np.max(jp[i])
            else:
                mean_px = std_px = p95_px = max_px = 0.0
            mean_norm = np.mean(jn[i]) if jn[i] else 0.0

            max_px_str = f"**{max_px:.4f}**" if max_px > JITTER_WARNING_THRESHOLD else f"{max_px:.4f}"

            lines.append(
                f"| {i} | {KEYPOINT_NAMES[i]} | {KEYPOINT_NAMES_ZH[i]} | {int(vc[i])} | {int(pc[i])} | "
                f"{mean_px:.4f} | {std_px:.4f} | {p95_px:.4f} | {max_px_str} | {mean_norm:.6f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def resolve_run_mode():
    if RUN_MODE in (1, 2):
        return RUN_MODE

    print("\n請選擇執行模式:")
    print("  1) 只生成統計結果（不開視窗）")
    print("  2) 只測試模型效果（開視窗）")
    choice = input("輸入模式 (1/2, 預設=2): ").strip()
    if choice == "1":
        return 1
    return 2

def main():
    run_mode = resolve_run_mode()
    is_stats_mode = (run_mode == 1)
    is_test_mode = (run_mode == 2)

    video_paths = resolve_video_paths(VIDEO_PATHS)
    if not video_paths:
        print("❌ 找不到可用影片，請確認 VIDEO_PATHS 內的檔案/資料夾路徑")
        return

    display_window = DISPLAY_WINDOW and is_test_mode
    save_output_video = SAVE_OUTPUT_VIDEO and is_test_mode
    loop_playback = LOOP_PLAYBACK and is_test_mode

    print("="*60)
    print("影片推論測試（EMA 平滑版）")
    print("="*60)
    print(f"執行模式: {'模式1-統計分析' if is_stats_mode else '模式2-視窗測試'}")
    print(f"EMA Alpha: {EMA_ALPHA}")
    print(f"影片路徑 (展開後共 {len(video_paths)} 部):")
    for i, p in enumerate(video_paths):
        print(f"  [{i}] {p}")
    print(f"YOLO 模型: {YOLO_MODEL_PATH}")
    print(f"ST-GCN 模型: {STGCN_MODEL_PATH}")
    print(f"推論裝置: {INFERENCE_DEVICE}")
    print(f"YOLO imgsz: {YOLO_IMGSZ}")
    print(f"YOLO conf threshold: {YOLO_CONF_THRESHOLD}")
    print(f"ST-GCN normalize: {STGCN_NORMALIZE}")
    print(f"模型目標 FPS: {TARGET_MODEL_FPS}")
    print(f"FPS 對齊降採樣: {'開啟' if ENABLE_FPS_DOWNSAMPLE else '關閉'}")
    print(f"分類步長 CLASSIFY_STRIDE: {CLASSIFY_STRIDE}")
    print(f"FAST_PREVIEW_OVERLAY: {'開啟' if FAST_PREVIEW_OVERLAY else '關閉'}")
    print(f"序列長度: {SEQUENCE_LENGTH}")
    print("="*60)

    # 初始化偵測器
    print("\n初始化模型...")
    keypoint_detector = KeypointDetector(
        YOLO_MODEL_PATH,
        device=INFERENCE_DEVICE,
        imgsz=YOLO_IMGSZ,
        conf_thres=YOLO_CONF_THRESHOLD,
    )
    behavior_classifier = STGCN7chClassifier(
        STGCN_MODEL_PATH,
        device=INFERENCE_DEVICE,
        sequence_length=SEQUENCE_LENGTH,
        normalize=STGCN_NORMALIZE
    )
    visualizer = Visualizer()
    visualizer.yolo_model_path = YOLO_MODEL_PATH

    # 統計累計（僅計入完整播放完成的影片）
    frame_count = 0
    predictions = []
    behavior_change_count = 0
    frames_with_cat = 0
    frames_without_cat = 0

    # 17點抖動統計（跨影片）
    global_jitter_px = [[] for _ in range(17)]
    global_jitter_norm = [[] for _ in range(17)]
    global_valid_counts = np.zeros(17, dtype=np.int64)
    global_pair_counts = np.zeros(17, dtype=np.int64)

    # 每影片抖動統計
    per_video_stats = defaultdict(
        lambda: {
            "jitter_px": [[] for _ in range(17)],
            "jitter_norm": [[] for _ in range(17)],
            "valid_counts": np.zeros(17, dtype=np.int64),
            "pair_counts": np.zeros(17, dtype=np.int64),
        }
    )

    # 完整播完才會寫入的每影片最終統計
    recorded_video_stats = {}

    # 狀態控制
    paused = False
    stop_requested = False
    current_video_idx = 0
    show_overlay_info = True

    # 即時顯示狀態
    behavior_id = LOW_CONF_ID
    confidence = 0.0
    probs = np.zeros(4, dtype=np.float32)
    if display_window:
        if DISPLAY_SIZE is not None:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    while not stop_requested:
        video_path = video_paths[current_video_idx]
        if not Path(video_path).exists():
            print(f"❌ 影片不存在，跳過: {video_path}")
            if is_stats_mode:
                current_video_idx += 1
                if current_video_idx >= len(video_paths):
                    break
            else:
                current_video_idx = (current_video_idx + 1) % len(video_paths)
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 無法開啟影片，跳過: {video_path}")
            if is_stats_mode:
                current_video_idx += 1
                if current_video_idx >= len(video_paths):
                    break
            else:
                current_video_idx = (current_video_idx + 1) % len(video_paths)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 1:
            source_fps = TARGET_MODEL_FPS
        frame_step = 1
        if ENABLE_FPS_DOWNSAMPLE and source_fps > TARGET_MODEL_FPS + 1e-6:
            frame_step = max(1, int(round(source_fps / TARGET_MODEL_FPS)))
        model_input_fps = source_fps / frame_step

        if source_fps < TARGET_MODEL_FPS - 0.5:
            print(
                f"⚠ 來源影片 FPS={source_fps:.2f} 低於模型目標 {TARGET_MODEL_FPS:.2f}，"
                "目前不做補幀，時序可能偏慢"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = (total_frames / source_fps) if source_fps > 0 else 0.0

        # 初始化 VideoWriter（如果需要保存輸出影片）
        video_writer = None
        if save_output_video:
            base_path = Path(OUTPUT_VIDEO_PATH)
            output_path = base_path.with_name(f"{base_path.stem}_{current_video_idx}{base_path.suffix}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer_fps = model_input_fps
            video_writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))
            if video_writer.isOpened():
                print(f"✓ 輸出影片將保存至: {output_path} (writer_fps={writer_fps:.2f})")
            else:
                print("❌ 無法創建輸出影片檔案")
                video_writer = None

        print("\n" + "=" * 60)
        print(f"目前影片 [{current_video_idx}] {video_path}")
        print(f"影片資訊: {width}x{height}, source_fps={source_fps:.1f}, total={total_frames} 幀")
        print(f"模型輸入時基: {model_input_fps:.2f} fps (frame_step={frame_step})")
        print(f"時長: {duration:.1f} 秒")
        if is_test_mode:
            print("控制: q=退出, space=暫停, r=重置本片, 2=下一部, 1=上一部, i=顯示/隱藏資訊")
        if loop_playback:
            print("🔁 循環播放模式（當前影片播完會重播）")
        print("-" * 60)

        keypoints_buffer = deque(maxlen=SEQUENCE_LENGTH)
        local_loop_count = 0
        switch_delta = 0
        prev_kpts = None
        prev_kpt_conf = None
        first_pass_completed = False
        switched_before_first_pass_complete = False

        # EMA 狀態：跨幀累積，切影片或貓消失時重置
        ema_kpts = None  # shape (17, 2)，儲存上一幀的 EMA 平滑座標

        # 本次影片臨時統計（只有完整第一輪才會被提交）
        local_predictions = []
        local_behavior_change_count = 0
        local_last_behavior = None
        raw_frames_read = 0
        local_frames_processed = 0
        local_sampled_frames = 0
        local_frames_with_cat = 0
        local_frames_without_cat = 0
        local_jitter_px = [[] for _ in range(17)]
        local_jitter_norm = [[] for _ in range(17)]
        local_valid_counts = np.zeros(17, dtype=np.int64)
        local_pair_counts = np.zeros(17, dtype=np.int64)

        while True:
            ret, frame = cap.read()
            if not ret:
                # 影片播放完畢
                if loop_playback and not stop_requested:
                    if local_loop_count == 0:
                        first_pass_completed = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    keypoints_buffer.clear()
                    prev_kpts = None
                    prev_kpt_conf = None
                    ema_kpts = None  # 循環重播時重置 EMA
                    raw_frames_read = 0
                    local_sampled_frames = 0
                    local_loop_count += 1
                    print(f"\n🔁 影片 [{current_video_idx}] 循環播放第 {local_loop_count} 次...\n")
                    continue
                if local_loop_count == 0:
                    first_pass_completed = True
                break

            # 只記錄第一輪統計；後續循環僅供展示
            is_first_pass = (local_loop_count == 0)
            raw_frames_read += 1
            local_sampled_frames += 1

            if is_first_pass:
                local_frames_processed += 1

            frame_time_sec = raw_frames_read / source_fps if source_fps > 0 else 0.0

            # YOLO-Pose 偵測
            kpts, kpt_conf, bbox, _ = keypoint_detector.detect(frame)

            if kpts is not None:
                # ===== EMA 平滑：對 YOLO 偵測的原始座標做指數移動平均 =====
                # 初始化：第一幀直接使用原始值作為起始 EMA
                if ema_kpts is None:
                    ema_kpts = kpts.copy()
                else:
                    ema_kpts = EMA_ALPHA * kpts + (1.0 - EMA_ALPHA) * ema_kpts
                # 以下所有處理均使用平滑後的座標
                kpts = ema_kpts.copy()
                # ============================================================

                # kpts: (17, 2), kpt_conf: (17,)
                if is_first_pass:
                    local_frames_with_cat += 1

                # 統計有效關鍵點幀數
                valid_mask = (kpt_conf > JITTER_CONF_THRESHOLD)
                if is_first_pass:
                    local_valid_counts += valid_mask.astype(np.int64)

                # 計算 bbox 對角線供正規化抖動使用
                bbox_diag = None
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    w_box = max(1.0, float(x2 - x1))
                    h_box = max(1.0, float(y2 - y1))
                    bbox_diag = float(np.sqrt(w_box * w_box + h_box * h_box))

                # 計算逐點抖動（EMA 平滑後的座標，反映模型實際接收到的穩定度）
                if prev_kpts is not None and prev_kpt_conf is not None:
                    pair_mask = (kpt_conf > JITTER_CONF_THRESHOLD) & (prev_kpt_conf > JITTER_CONF_THRESHOLD)
                    for kp_idx in range(17):
                        if not pair_mask[kp_idx]:
                            continue

                        jitter_px = float(np.linalg.norm(kpts[kp_idx] - prev_kpts[kp_idx]))
                        if is_first_pass:
                            local_jitter_px[kp_idx].append(jitter_px)
                            local_pair_counts[kp_idx] += 1

                        if bbox_diag is not None and bbox_diag > 0:
                            jitter_norm = jitter_px / bbox_diag
                            if is_first_pass:
                                local_jitter_norm[kp_idx].append(jitter_norm)

                prev_kpts = kpts.copy()
                prev_kpt_conf = kpt_conf.copy()

                # 加入緩衝區
                keypoints_buffer.append((kpts, kpt_conf))

                # 有足夠序列時做行為分類
                if len(keypoints_buffer) >= SEQUENCE_LENGTH and (local_sampled_frames % CLASSIFY_STRIDE == 0):
                    # 解包緩衝區
                    kpts_arr = np.array([item[0] for item in keypoints_buffer])  # (32, 17, 2)
                    conf_arr = np.array([item[1] for item in keypoints_buffer])  # (32, 17)

                    # 插值補全
                    seq_array = interpolate_missing(kpts_arr, conf_arr, threshold=0.1)

                    # ST-GCN 推論
                    pred_id, pred_conf, pred_probs = behavior_classifier.classify(seq_array, conf_arr)
                    if pred_id is None:
                        behavior_id = LOW_CONF_ID
                        confidence = 0.0
                        probs = np.zeros(4, dtype=np.float32)
                    else:
                        behavior_id = int(pred_id)
                        confidence = float(pred_conf)
                        probs = pred_probs.copy()

                    # 與主系統一致：低信心顯示「目前正常」
                    if confidence < CONFIDENCE_THRESHOLD:
                        behavior_id_for_display = LOW_CONF_ID
                    else:
                        behavior_id_for_display = behavior_id

                    # 只統計高信心預測
                    if behavior_id_for_display != LOW_CONF_ID:
                        behavior_text = BEHAVIOR_TEXT_MAP.get(behavior_id, BEHAVIOR_CLASSES[behavior_id])
                        if is_first_pass:
                            local_predictions.append({
                                'video_idx': current_video_idx,
                                'video_path': video_path,
                                'frame': local_frames_processed,
                                'time': frame_time_sec,
                                'behavior_id': behavior_id,
                                'behavior_name': BEHAVIOR_CLASSES[behavior_id],
                                'confidence': confidence,
                                'probs': probs.copy()
                            })
                        if local_last_behavior != behavior_id:
                            if local_last_behavior is not None and is_first_pass:
                                local_behavior_change_count += 1
                                print(f"影片[{current_video_idx}] 幀 {local_frames_processed:6d}: {behavior_text:6s} {confidence*100:5.1f}% " +
                                    f"[walk:{probs[0]*100:4.1f}% lick:{probs[1]*100:4.1f}% scratch:{probs[2]*100:4.1f}% shake:{probs[3]*100:4.1f}%]")
                            local_last_behavior = behavior_id
                    else:
                        behavior_id = LOW_CONF_ID
            else:
                if is_first_pass:
                    local_frames_without_cat += 1
                prev_kpts = None
                prev_kpt_conf = None
                ema_kpts = None  # 貓消失時重置 EMA，避免下次出現時使用過時的平均值

            # 繪製 overlay（與主系統 Visualizer 一致）——畫在原始解析度，供輸出影片使用
            display = frame.copy()
            if kpts is not None:
                visualizer.frame_idx = frame_count
                visualizer.fps = source_fps
                display = draw_test2_style_overlay(
                    display,
                    kpts,
                    kpt_conf,
                    bbox,
                    behavior_id,
                    confidence,
                    probs if len(probs) == 4 else np.zeros(4, dtype=np.float32),
                    visualizer,
                    show_info=show_overlay_info,
                )
            else:
                draw_no_cat_overlay(display)

            # 額外狀態欄（寫入 display，供輸出影片使用）
            current_frame_in_video = raw_frames_read
            status_text = f"Video[{current_video_idx}] Frame {current_frame_in_video}/{total_frames}"
            if LOOP_PLAYBACK:
                status_text += f" | Loop {local_loop_count}"
            status_text += f" | srcFPS={source_fps:.1f} modelFPS={model_input_fps:.1f}"
            status_text += f" | EMA={EMA_ALPHA} | r:reset 2:next 1:prev i:info q:quit space:pause"
            if show_overlay_info:
                status_scale = compute_ui_scale(width, height)
                status_x = scale_px(10, status_scale, min_px=8)
                status_y = max(scale_px(20, status_scale, min_px=14), height - scale_px(12, status_scale, min_px=8))
                status_font = 0.55 * status_scale
                cv2.putText(display, status_text, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, status_font, (0, 0, 0), scale_px(3, status_scale, min_px=2), cv2.LINE_AA)
                cv2.putText(display, status_text, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, status_font, (255, 255, 255), scale_px(1, status_scale, min_px=1), cv2.LINE_AA)

            # 保存到輸出影片（僅第一次循環）
            if video_writer is not None and video_writer.isOpened() and local_loop_count == 0:
                video_writer.write(display)

            # 顯示視窗 — 在目標解析度上重繪 overlay，確保字體與線條清晰
            if display_window:
                if DISPLAY_SIZE is not None:
                    disp_w, disp_h = DISPLAY_SIZE
                    if FAST_PREVIEW_OVERLAY:
                        show_frame, _, _, _ = resize_with_letterbox(display, DISPLAY_SIZE)
                    else:
                        show_frame, preview_scale, preview_pad_x, preview_pad_y = resize_with_letterbox(frame, DISPLAY_SIZE)
                        if kpts is not None:
                            scaled_kpts = kpts * preview_scale + np.array([preview_pad_x, preview_pad_y], dtype=np.float32)
                            scaled_bbox = None
                            if bbox is not None:
                                x1, y1, x2, y2 = bbox
                                scaled_bbox = np.array([
                                    x1 * preview_scale + preview_pad_x,
                                    y1 * preview_scale + preview_pad_y,
                                    x2 * preview_scale + preview_pad_x,
                                    y2 * preview_scale + preview_pad_y,
                                ], dtype=np.float32)
                            visualizer.frame_idx = frame_count
                            visualizer.fps = source_fps
                            show_frame = draw_test2_style_overlay(
                                show_frame, scaled_kpts, kpt_conf, scaled_bbox,
                                behavior_id, confidence,
                                probs if len(probs) == 4 else np.zeros(4, dtype=np.float32),
                                visualizer,
                                show_info=show_overlay_info,
                            )
                        else:
                            draw_no_cat_overlay(show_frame)
                        if show_overlay_info:
                            # 底部半透明狀態列
                            preview_scale_ui = compute_ui_scale(disp_w, disp_h)
                            bar_h = scale_px(26, preview_scale_ui, min_px=18)
                            bar_margin = scale_px(8, preview_scale_ui, min_px=6)
                            _bar = show_frame.copy()
                            cv2.rectangle(_bar, (0, disp_h - bar_h), (disp_w, disp_h), (20, 20, 20), -1)
                            cv2.addWeighted(_bar, 0.65, show_frame, 0.35, 0, show_frame)
                            cv2.putText(show_frame, status_text, (bar_margin, disp_h - bar_margin),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52 * preview_scale_ui, (230, 230, 230), scale_px(1, preview_scale_ui, min_px=1), cv2.LINE_AA)
                else:
                    show_frame = display
                cv2.imshow(WINDOW_NAME, show_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n使用者中斷：q")
                    stop_requested = True
                    break
                if key == ord('i'):
                    show_overlay_info = not show_overlay_info
                    print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
                    continue
                if key == ord('2'):
                    switch_delta = 1
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    print("\n切換到下一部影片")
                    break
                if key == ord('1'):
                    switch_delta = -1
                    if not first_pass_completed:
                        switched_before_first_pass_complete = True
                    print("\n切換到上一部影片")
                    break
                if key == ord('r'):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    keypoints_buffer.clear()
                    local_loop_count = 0
                    first_pass_completed = False
                    switched_before_first_pass_complete = False
                    prev_kpts = None
                    prev_kpt_conf = None
                    ema_kpts = None
                    behavior_id = LOW_CONF_ID
                    confidence = 0.0
                    probs = np.zeros(4, dtype=np.float32)
                    local_predictions = []
                    local_behavior_change_count = 0
                    local_last_behavior = None
                    raw_frames_read = 0
                    local_frames_processed = 0
                    local_sampled_frames = 0
                    local_frames_with_cat = 0
                    local_frames_without_cat = 0
                    local_jitter_px = [[] for _ in range(17)]
                    local_jitter_norm = [[] for _ in range(17)]
                    local_valid_counts = np.zeros(17, dtype=np.int64)
                    local_pair_counts = np.zeros(17, dtype=np.int64)
                    print("\n↺ 已重置：回到影片開頭並清空偵測狀態")
                    continue
                if key == ord(' '):
                    paused = not paused
                    while paused:
                        k2 = cv2.waitKey(50) & 0xFF
                        if k2 == ord(' '):
                            paused = False
                        elif k2 == ord('q'):
                            paused = False
                            print("\n使用者中斷：q")
                            stop_requested = True
                            break
                        elif k2 == ord('i'):
                            show_overlay_info = not show_overlay_info
                            print(f"\n資訊面板: {'顯示' if show_overlay_info else '隱藏'}")
                        elif k2 == ord('2'):
                            paused = False
                            switch_delta = 1
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            print("\n切換到下一部影片")
                            break
                        elif k2 == ord('1'):
                            paused = False
                            switch_delta = -1
                            if not first_pass_completed:
                                switched_before_first_pass_complete = True
                            print("\n切換到上一部影片")
                            break
                        elif k2 == ord('r'):
                            paused = False
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            keypoints_buffer.clear()
                            local_loop_count = 0
                            first_pass_completed = False
                            switched_before_first_pass_complete = False
                            prev_kpts = None
                            prev_kpt_conf = None
                            ema_kpts = None
                            behavior_id = LOW_CONF_ID
                            confidence = 0.0
                            probs = np.zeros(4, dtype=np.float32)
                            local_predictions = []
                            local_behavior_change_count = 0
                            local_last_behavior = None
                            raw_frames_read = 0
                            local_frames_processed = 0
                            local_sampled_frames = 0
                            local_frames_with_cat = 0
                            local_frames_without_cat = 0
                            local_jitter_px = [[] for _ in range(17)]
                            local_jitter_norm = [[] for _ in range(17)]
                            local_valid_counts = np.zeros(17, dtype=np.int64)
                            local_pair_counts = np.zeros(17, dtype=np.int64)
                            print("\n↺ 已重置：回到影片開頭並清空偵測狀態")
                            break

            if stop_requested or switch_delta != 0:
                break

            # 降採樣時跳過後續 frame_step-1 幀，避免不必要的完整解碼
            if frame_step > 1:
                for _ in range(frame_step - 1):
                    if not cap.grab():
                        break
                    raw_frames_read += 1

            # 每 100 幀顯示進度（僅第一次循環顯示）
            if local_loop_count == 0 and local_frames_processed % 100 == 0:
                pct = (raw_frames_read / total_frames * 100) if total_frames > 0 else 0.0
                print(f"  影片[{current_video_idx}] 處理進度: {raw_frames_read}/{total_frames} ({pct:.1f}%)")

        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"✓ 輸出影片已保存: {output_path}")

        # 只有完整播放第一輪且非中途切換，才提交本影片統計
        if first_pass_completed and not switched_before_first_pass_complete:
            behavior_counts = np.zeros(4, dtype=np.int64)
            behavior_confidences = []
            behavior_confidences_by_class = [[] for _ in range(4)]
            for p in local_predictions:
                behavior_counts[p['behavior_id']] += 1
                behavior_confidences.append(p['confidence'])
                behavior_confidences_by_class[p['behavior_id']].append(p['confidence'])

            recorded_video_stats[current_video_idx] = {
                "video_idx": current_video_idx,
                "video_path": video_path,
                "width": width,
                "height": height,
                "fps": float(source_fps),
                "model_input_fps": float(model_input_fps),
                "frame_step": int(frame_step),
                "total_frames": int(total_frames),
                "processed_frames": int(local_frames_processed),
                "frames_with_cat": int(local_frames_with_cat),
                "frames_without_cat": int(local_frames_without_cat),
                "behavior_counts": behavior_counts,
                "behavior_confidences": behavior_confidences,
                "behavior_confidences_by_class": behavior_confidences_by_class,
                "jitter_px": local_jitter_px,
                "jitter_norm": local_jitter_norm,
                "valid_counts": local_valid_counts,
                "pair_counts": local_pair_counts,
            }

            # 合併到全域統計
            frame_count += local_frames_processed
            frames_with_cat += local_frames_with_cat
            frames_without_cat += local_frames_without_cat
            predictions.extend(local_predictions)
            behavior_change_count += local_behavior_change_count

            for i in range(17):
                global_jitter_px[i].extend(local_jitter_px[i])
                global_jitter_norm[i].extend(local_jitter_norm[i])
            global_valid_counts += local_valid_counts
            global_pair_counts += local_pair_counts

            per_video_stats[current_video_idx] = {
                "jitter_px": local_jitter_px,
                "jitter_norm": local_jitter_norm,
                "valid_counts": local_valid_counts,
                "pair_counts": local_pair_counts,
            }
            print(f"✓ 影片[{current_video_idx}] 已完整播放，統計已記錄")
        else:
            if switched_before_first_pass_complete:
                print(f"⚠ 影片[{current_video_idx}] 中途切換，該影片統計不記錄")
            else:
                print(f"⚠ 影片[{current_video_idx}] 未完成第一輪播放，該影片統計不記錄")

        if stop_requested:
            break

        if is_stats_mode:
            current_video_idx += 1
            if current_video_idx >= len(video_paths):
                break
        else:
            if switch_delta != 0:
                current_video_idx = (current_video_idx + switch_delta) % len(video_paths)
            elif not loop_playback:
                break

    if display_window:
        cv2.destroyAllWindows()

    if is_test_mode:
        print("\n模式2完成：視窗測試結束（未產生統計報告）")
        print("=" * 60)
        return

    print("-"*60)
    print(f"\n推論完成！共納入 {frame_count} 幀（僅完整播放影片）")
    print(f"\nYOLO 偵測統計:")
    if frame_count > 0:
        print(f"  偵測到貓咪: {frames_with_cat} 幀 ({frames_with_cat/frame_count*100:.1f}%)")
        print(f"  未偵測到: {frames_without_cat} 幀 ({frames_without_cat/frame_count*100:.1f}%)")
    else:
        print("  偵測到貓咪: 0 幀 (0.0%)")
        print("  未偵測到: 0 幀 (0.0%)")
    print(f"\n有效預測: {len(predictions)} 次")
    print(f"行為變化: {behavior_change_count} 次")

    # 抖動統計（全域）
    print_jitter_report(
        title=f"17關鍵點抖動統計（全域，EMA={EMA_ALPHA}，conf>{JITTER_CONF_THRESHOLD}）",
        jitter_px=global_jitter_px,
        jitter_norm=global_jitter_norm,
        valid_counts=global_valid_counts,
        pair_counts=global_pair_counts,
    )

    # 抖動統計（每影片）
    for vid_idx, stats in sorted(per_video_stats.items(), key=lambda x: x[0]):
        print_jitter_report(
            title=f"17關鍵點抖動統計（影片[{vid_idx}]，EMA={EMA_ALPHA}，conf>{JITTER_CONF_THRESHOLD}）",
            jitter_px=stats["jitter_px"],
            jitter_norm=stats["jitter_norm"],
            valid_counts=stats["valid_counts"],
            pair_counts=stats["pair_counts"],
        )

    # 產出文檔報告
    report_path = generate_report_file(REPORT_OUTPUT_PATH, recorded_video_stats)
    print(f"\n✓ 分析報告已輸出: {report_path}")

    # 統計分析
    if predictions:
        print("\n" + "="*60)
        print("統計分析")
        print("="*60)

        from collections import Counter
        behavior_counts = Counter([p['behavior_id'] for p in predictions])
        print("\n各行為出現次數:")
        for bid in range(4):
            count = behavior_counts.get(bid, 0)
            pct = count / len(predictions) * 100 if predictions else 0
            print(f"  {BEHAVIOR_TEXT_MAP[bid]:6s} ({BEHAVIOR_CLASSES[bid]:8s}): {count:4d} 次 ({pct:5.1f}%)")

        avg_probs = np.mean([p['probs'] for p in predictions], axis=0)
        print("\n平均機率分布:")
        for i, (cls, prob) in enumerate(zip(BEHAVIOR_CLASSES, avg_probs)):
            print(f"  {BEHAVIOR_TEXT_MAP[i]:6s} ({cls:8s}): {prob*100:5.1f}%")

        confidences = [p['confidence'] for p in predictions]
        print(f"\n信心值統計:")
        print(f"  平均: {np.mean(confidences)*100:.1f}%")
        print(f"  最小: {np.min(confidences)*100:.1f}%")
        print(f"  最大: {np.max(confidences)*100:.1f}%")

        most_common_id = behavior_counts.most_common(1)[0][0]
        print(f"\n✓ 主要行為: {BEHAVIOR_TEXT_MAP[most_common_id]} ({BEHAVIOR_CLASSES[most_common_id]})")
        print(f"  出現比例: {behavior_counts[most_common_id]/len(predictions)*100:.1f}%")

        print("\n" + "="*60)
        print("結果分析")
        print("="*60)

        if most_common_id == 1:
            print("⚠ 主要預測為 scratch（搔抓），建議檢查:")
            print("  1. 影片內容是否包含抓癢/停頓等與 scratch 相似片段")
            print("  2. 是否有大量低信心窗被濾除，造成剩餘樣本偏向 scratch")
            print("  3. 重新檢視混淆矩陣與該影片逐幀機率曲線")
        elif most_common_id == 0:
            print("✓ 主要預測為 walk（走動），符合預期")
            print("  → 模型正確辨識出行走行為")
        else:
            print(f"預測為 {BEHAVIOR_TEXT_MAP[most_common_id]}，需檢查:")
            print("  1. 影片內容是否確實為此行為")
            print("  2. 模型訓練數據品質")
            print("  3. 正規化是否正確 (normalize=True)")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
