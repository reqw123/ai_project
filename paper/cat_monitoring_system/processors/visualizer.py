"""
繪圖函數集中管理
"""
import cv2
import numpy as np
import time
from pathlib import Path
from utils.constants import *
from config import AnomalyDetectionConfig as _AnomalyDetectionConfig
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig
from utils.helpers import get_behavior_name
import bisect

try:
    from PIL import Image, ImageSequence, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is available in the configured env, but keep a fallback.
    Image = None
    ImageSequence = None
    ImageDraw = None
    ImageFont = None


HIP_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / r"C:\ai_project\paper\cat_monitoring_system\tools\h6bxw-tkcsv.gif"
HIP_IMAGE_ALPHA_BOOST = 1.6 #透明度
HIP_IMAGE_SCALE = 0  # float | None — None=跟隨 bbox 動態計算；浮點數=原圖等比例倍率（1.0=原尺寸、2.0=放大兩倍）


def _load_overlay_frames(image_path):
    """載入靜態圖或動畫圖，回傳 (frames, durations_ms)。"""
    if not image_path.exists():
        return [], []

    if Image is not None and ImageSequence is not None:
        try:
            with Image.open(image_path) as im:
                frames = []
                durations = []
                default_duration = int(im.info.get("duration", 100) or 100)

                for frame in ImageSequence.Iterator(im):
                    rgba = frame.convert("RGBA")
                    arr = np.array(rgba, dtype=np.uint8)
                    # Pillow gives RGBA; OpenCV uses BGR(A). Convert to BGRA so colors match.
                    try:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                    except Exception:
                        # fallback: leave as-is
                        pass
                    frames.append(arr)
                    duration = int(frame.info.get("duration", default_duration) or default_duration)
                    durations.append(max(1, duration))

                if frames:
                    return frames, durations
        except Exception:
            pass

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return [], []
    return [image], [100]


HIP_IMAGE_FRAMES, HIP_IMAGE_DURATIONS = _load_overlay_frames(HIP_IMAGE_PATH)

# 常用中文字型候選（Windows 優先）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttf",
    r"C:\Windows\Fonts\msjhbd.ttf",
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\SimHei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
]

# 字型快取：_find_font 第一次成功後就不再重複開檔
_font_cache: dict = {}

def _find_font(size):
    if size in _font_cache:
        return _font_cache[size]
    if ImageFont is None:
        _font_cache[size] = None
        return None
    for p in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(p, size)
            _font_cache[size] = font
            return font
        except Exception:
            continue
    try:
        font = ImageFont.load_default()
        _font_cache[size] = font
        return font
    except Exception:
        _font_cache[size] = None
        return None

# PIL 文字量測用單例：避免每幀重複建立 Image.new + ImageDraw.Draw
_tmp_pil_img = None
_tmp_pil_draw = None

def _get_tmp_draw():
    global _tmp_pil_img, _tmp_pil_draw
    if _tmp_pil_draw is None and Image is not None and ImageDraw is not None:
        _tmp_pil_img  = Image.new('RGB', (1, 1))
        _tmp_pil_draw = ImageDraw.Draw(_tmp_pil_img)
    return _tmp_pil_draw

# overlay resize 快取：對齊最近 4px 後快取，減少相同尺寸的重複 resize
_overlay_resize_cache: dict = {}


def _draw_text_with_pil(frame, text, pos, font_size=28, color=(255, 255, 255), outline=2, bg_box=None):
    """使用 Pillow 在 BGR numpy frame 上繪製文字（支援中文）。
    pos: (x,y) 左上角位置
    bg_box: (x1,y1,x2,y2) or None，若提供則先畫背景方塊
    返回修改過的 frame（BGR numpy）
    """
    if Image is None or ImageDraw is None:
        return frame

    # Convert BGR -> RGB
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = frame.copy()

    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _find_font(font_size) or None

    x, y = int(pos[0]), int(pos[1])
    # 背景方塊
    if bg_box is not None:
        x1, y1, x2, y2 = map(int, bg_box)
        draw.rectangle([x1, y1, x2, y2], fill=(20, 20, 20, 255))

    # Outline: draw multiple offsets in black
    rgb_color = tuple(int(c) for c in color)
    outline_color = (0, 0, 0)
    if font is None:
        # fallback to default draw
        draw.text((x, y), text, fill=rgb_color)
    else:
        for dx in range(-outline, outline + 1):
            for dy in range(-outline, outline + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        draw.text((x, y), text, font=font, fill=rgb_color)

    out = np.array(pil)
    try:
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = out
    return bgr


def _overlay_image_centered(frame, image, center_xy, target_width, target_height=None):
    """將圖像以中心點方式疊到 frame 上，支援透明通道與邊界裁切。
    target_height=None 表示依原始比例自動計算高度。
    """
    if frame is None or image is None:
        return

    try:
        target_width = int(target_width)
    except (TypeError, ValueError):
        return

    if target_width <= 0:
        return

    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return

    # 對齊最近 4px 讓相同大小的貓共用快取，減少重複 resize
    tw = (int(target_width) // 4) * 4
    if tw <= 0:
        return
    th_fixed = (int(target_height) // 4) * 4 if target_height else None
    cache_key = (id(image), tw, th_fixed)
    if cache_key in _overlay_resize_cache:
        resized = _overlay_resize_cache[cache_key]
    else:
        th = th_fixed if th_fixed else max(1, int(round(tw * src_h / max(src_w, 1))))
        # 縮小用 INTER_AREA（品質最佳）；放大用 INTER_LANCZOS4（避免鋸齒）
        interp = cv2.INTER_AREA if (tw <= src_w and th <= src_h) else cv2.INTER_LANCZOS4
        resized = cv2.resize(image, (tw, th), interpolation=interp)
        if len(_overlay_resize_cache) < 64:  # 防止記憶體無限增長
            _overlay_resize_cache[cache_key] = resized
    # 不論快取命中或新計算，都從 resized 取得最終尺寸
    target_width  = tw
    target_height = resized.shape[0]

    x_c = int(round(float(center_xy[0])))
    y_c = int(round(float(center_xy[1])))
    x1 = x_c - target_width // 2
    y1 = y_c - target_height // 2
    x2 = x1 + target_width
    y2 = y1 + target_height

    frame_h, frame_w = frame.shape[:2]
    clip_x1 = max(0, x1)
    clip_y1 = max(0, y1)
    clip_x2 = min(frame_w, x2)
    clip_y2 = min(frame_h, y2)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    src_x1 = clip_x1 - x1
    src_y1 = clip_y1 - y1
    src_x2 = src_x1 + (clip_x2 - clip_x1)
    src_y2 = src_y1 + (clip_y2 - clip_y1)

    roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
    patch = resized[src_y1:src_y2, src_x1:src_x2]

    if patch.ndim != 3:
        return

    if patch.shape[2] == 4:
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        alpha = np.clip(alpha * HIP_IMAGE_ALPHA_BOOST, 0.0, 1.0)
        color = patch[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)
        blended = roi_float * (1.0 - alpha) + color * alpha
        frame[clip_y1:clip_y2, clip_x1:clip_x2] = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        frame[clip_y1:clip_y2, clip_x1:clip_x2] = patch[:, :, :3]

class Visualizer:
    def __init__(self):
        self._overlay_anim_start = time.monotonic()
        self._overlay_frames = HIP_IMAGE_FRAMES
        self._overlay_frame_durations = HIP_IMAGE_DURATIONS
        self._overlay_total_duration = sum(self._overlay_frame_durations) if self._overlay_frame_durations else 0
        # 預計算累積時間戳，供 bisect O(log N) 查找當前動畫幀（取代線性掃描）
        cumsum = 0
        self._overlay_cumulative: list = []
        for d in self._overlay_frame_durations:
            cumsum += d
            self._overlay_cumulative.append(cumsum)

    def _get_overlay_frame(self):
        if not self._overlay_frames:
            return None
        if len(self._overlay_frames) == 1 or self._overlay_total_duration <= 0:
            return self._overlay_frames[0]
        elapsed_ms = int((time.monotonic() - self._overlay_anim_start) * 1000)
        tick = elapsed_ms % self._overlay_total_duration
        idx = bisect.bisect_right(self._overlay_cumulative, tick)
        return self._overlay_frames[min(idx, len(self._overlay_frames) - 1)]

    def draw_prediction_on_frame(
        self,
        frame,
        prediction_text,
        confidence,
        color,
        show_confidence=True,
        emphasize_label=False,
        label_background=True,
        font_scale_override=None,
        **kwargs,
    ):
        h, w = frame.shape[:2]
        # 左上角：行為預測
        text = f"{prediction_text}: {confidence:.2%}" if show_confidence else str(prediction_text)
        x, y = 14, 30
        font_scale = 1.08 if font_scale_override is None else float(font_scale_override)
        outline_thickness = 4
        text_thickness = 2

        if emphasize_label:
            font_scale = 1.36
            outline_thickness = 6
            text_thickness = 3

        # 若文字包含非 ASCII（例如中文），使用 Pillow 繪製以支援 CJK；否則使用 OpenCV（效能較好）
        use_pil = any(ord(ch) > 127 for ch in text) and Image is not None and ImageDraw is not None

        if use_pil:
            # 計算字型大小與背景框（以 font_scale 為基準）
            font_px = max(14, int(24 * font_scale))
            # 嘗試用 PIL 取得文字尺寸（使用模組層級單例，不每幀重新建立）
            try:
                font = _find_font(font_px)
                if font is not None:
                    draw_tmp = _get_tmp_draw()
                    if draw_tmp is not None:
                        bbox_tmp = draw_tmp.textbbox((0, 0), text, font=font)
                        text_w, text_h = bbox_tmp[2] - bbox_tmp[0], bbox_tmp[3] - bbox_tmp[1]
                    else:
                        text_w, text_h = (int(len(text) * font_px * 0.6), font_px)
                else:
                    text_w, text_h = (int(len(text) * font_px * 0.6), font_px)
            except Exception:
                text_w, text_h = (int(len(text) * font_px * 0.6), font_px)

            pad_x = 14
            pad_y = 10
            x1 = max(0, x - pad_x)
            y1 = max(0, y - text_h - pad_y)
            x2 = min(w - 1, x + text_w + pad_x)
            y2 = min(h - 1, y + text_h + pad_y)
            # 繪製背景與文字
            frame = _draw_text_with_pil(frame, text, (x, y - text_h), font_size=font_px, color=(int(color[2]), int(color[1]), int(color[0])), outline=outline_thickness if outline_thickness>0 else 2, bg_box=(x1, y1, x2, y2) if label_background else None)
        else:
            if emphasize_label and label_background:
                (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                pad_x = 14
                pad_y = 10
                x1 = max(0, x - pad_x)
                y1 = max(0, y - text_h - pad_y)
                x2 = min(w - 1, x + text_w + pad_x)
                y2 = min(h - 1, y + baseline + pad_y)
                # ROI 局部混合：避免全幀 frame.copy() + addWeighted 只為畫一個小背景框
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = (roi.astype(np.float32) * 0.42 + 11.6).astype(np.uint8)

            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK, outline_thickness, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA)

    def draw_probability_bars(self, frame, class_probs, behavior_names):
        h, w = frame.shape[:2]
        bar_width = 245
        bar_height = 30
        start_x = w - bar_width - 30
        start_y = 18
        spacing = 20
        bar_colors = [
            BEHAVIOR_COLORS.get(0, (0, 255, 0)),     # walk
            BEHAVIOR_COLORS.get(1, (0, 255, 255)),   # lick
            BEHAVIOR_COLORS.get(2, (255, 165, 0)),   # scratch
            BEHAVIOR_COLORS.get(3, (0, 0, 255)),     # shake
            BEHAVIOR_COLORS.get(4, (0, 165, 255)),   # stop
        ]
        for i, (prob, class_name) in enumerate(zip(class_probs, behavior_names)):
            y = start_y + i * (bar_height + spacing)
            # 背景條
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), (40, 40, 40), -1)
            # 機率條
            color = bar_colors[i % len(bar_colors)]
            filled_width = int(bar_width * prob)
            cv2.rectangle(frame, (start_x, y), (start_x + filled_width, y + bar_height), color, -1)
            # 外框
            cv2.rectangle(frame, (start_x, y), (start_x + bar_width, y + bar_height), (200, 200, 200), 1)
            # 類別名稱與機率
            label = f"{class_name}: {prob*100:.1f}%"
            cv2.putText(frame, label, (start_x + 12, y + bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.86, BLACK, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (start_x + 12, y + bar_height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.86, WHITE, 1, cv2.LINE_AA)

    def draw(self, frame, kpts, kpt_conf, bbox, conf, behavior_id, confidence, class_probs,
             show_info=True, show_skeleton=True, show_bbox=True):
        if show_skeleton:
            for edge_idx, (i, j) in enumerate(EAR_DISTANCE_SKELETON_EDGES):
                if i >= len(kpts) or j >= len(kpts):
                    continue
                if kpt_conf[i] > _AnomalyDetectionConfig.KP_CONF_THRES and kpt_conf[j] > _AnomalyDetectionConfig.KP_CONF_THRES:
                    pt1 = tuple(map(int, kpts[i]))
                    pt2 = tuple(map(int, kpts[j]))
                    color = EAR_DISTANCE_EDGE_COLORS[edge_idx] if edge_idx < len(EAR_DISTANCE_EDGE_COLORS) else (180, 180, 180)
                    cv2.line(frame, pt1, pt2, color, 2)

            for i in range(len(kpts)):
                if kpt_conf[i] > _AnomalyDetectionConfig.KP_CONF_THRES:
                    pt = tuple(map(int, kpts[i]))
                    color = EAR_DISTANCE_KP_COLORS[i] if i < len(EAR_DISTANCE_KP_COLORS) else COLOR_KPT
                    cv2.circle(frame, pt, 3, color, -1)

        # 將貓臉疊層圖像貼在 nose 關鍵點(0)上，跟隨鼻子移動
        overlay_frame = self._get_overlay_frame()
        if overlay_frame is not None and len(kpts) > 0 and kpt_conf[0] > _AnomalyDetectionConfig.KP_CONF_THRES:
            nose_pt = tuple(map(int, kpts[0]))
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)
                base_width = int(np.clip(min(bbox_w, bbox_h) * 0.32, 52, 128))
            else:
                base_width = 72

            if HIP_IMAGE_SCALE is not None:
                src_h, src_w = overlay_frame.shape[:2]
                w = max(1, int(round(src_w * HIP_IMAGE_SCALE)))
                h = max(1, int(round(src_h * HIP_IMAGE_SCALE)))
            else:
                w, h = base_width, None
            _overlay_image_centered(frame, overlay_frame, nose_pt, w, h)

        # 畫YOLO bbox/conf
        if show_bbox and bbox is not None and conf is not None:
            x1, y1, x2, y2 = map(int, bbox)
            outer_w = 4
            inner_w = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLACK, outer_w, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HEAD, inner_w, cv2.LINE_AA)
            label = f"{conf:.2f}"
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        # 顯示行為預測與信心值、四行為機率條
        if not show_info:
            return frame

        if confidence is None:
            confidence = 0.0
        is_display_normal = (behavior_id == LOW_CONF_ID) or (float(confidence) < _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD)
        if is_display_normal:
            # 顯示層使用中文標籤（若可用字型）
            low_text = get_behavior_name(LOW_CONF_ID, use_text=False, fallback=LOW_CONF_TEXT, confidence=confidence)
            self.draw_prediction_on_frame(frame, low_text, confidence, (200, 200, 200))
            if class_probs is not None and any(p > 0 for p in class_probs):
                self.draw_probability_bars(frame, class_probs, BEHAVIOR_CLASSES)
        elif behavior_id is not None and confidence > 0:
            # overlay 使用中文名稱
            behavior_name = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
            self.draw_prediction_on_frame(frame, behavior_name, confidence, BEHAVIOR_COLORS.get(behavior_id, (255,255,255)))
            self.draw_probability_bars(frame, class_probs, BEHAVIOR_CLASSES)
        return frame
