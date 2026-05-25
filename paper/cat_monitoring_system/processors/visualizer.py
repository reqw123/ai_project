"""
繪圖函數集中管理
"""
import cv2
import numpy as np
import time
from pathlib import Path
from utils.constants import *
from utils.helpers import get_behavior_name

try:
    from PIL import Image, ImageSequence, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is available in the configured env, but keep a fallback.
    Image = None
    ImageSequence = None
    ImageDraw = None
    ImageFont = None


HIP_IMAGE_PATH = Path(__file__).resolve().parent.parent.parent / "cat_monitoring_system/maolex-blogs-cat-face-rabbit.webp"
HIP_IMAGE_ALPHA_BOOST = 1.6


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


def _find_font(size):
    if ImageFont is None:
        return None
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


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


def _overlay_image_centered(frame, image, center_xy, target_width):
    """將圖像以中心點方式疊到 frame 上，支援透明通道與邊界裁切。"""
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

    target_height = max(1, int(round(target_width * src_h / max(src_w, 1))))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

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

    def _get_overlay_frame(self):
        if not self._overlay_frames:
            return None

        if len(self._overlay_frames) == 1 or self._overlay_total_duration <= 0:
            return self._overlay_frames[0]

        elapsed_ms = int((time.monotonic() - self._overlay_anim_start) * 1000)
        tick = elapsed_ms % self._overlay_total_duration

        cumulative = 0
        for frame, duration in zip(self._overlay_frames, self._overlay_frame_durations):
            cumulative += duration
            if tick < cumulative:
                return frame

        return self._overlay_frames[-1]

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
            # 嘗試用 PIL 取得文字尺寸
            try:
                font = _find_font(font_px)
                if font is not None:
                    text_w, text_h = ImageDraw.Draw(Image.new('RGB', (1,1))).textsize(text, font=font)
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
                y2 = min(h - 1, x + baseline + pad_y)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

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
            (0, 255, 0),      # walk - Green
            (255, 0, 0),      # scratch - Blue
            (0, 255, 255),    # lick - Yellow
            (0, 0, 255)       # shake - Red
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
    def draw(self, frame, kpts, kpt_conf, bbox, conf, behavior_id, confidence, class_probs, show_info=True):
        # 頭部
        for i, j in HEAD_LINKS:
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                cv2.line(frame, pt1, pt2, COLOR_HEAD, 2)
        # 身體
        for i, j in BODY_LINKS:
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                cv2.line(frame, pt1, pt2, COLOR_BODY, 2)
        # 前肢
        for idx, (i, j) in enumerate(FRONT_LIMBS):
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                color = COLOR_LEFT_FRONT if idx < 2 else COLOR_RIGHT_FRONT
                cv2.line(frame, pt1, pt2, color, 2)
        # 後肢
        for idx, (i, j) in enumerate(HIND_LIMBS):
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                color = COLOR_LEFT_HIND if idx < 2 else COLOR_RIGHT_HIND
                cv2.line(frame, pt1, pt2, color, 2)
        # Hip → Tail_Root（橘黃色，與尾巴段區隔）
        for i, j in HIP_TAIL_LINK:
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                cv2.line(frame, pt1, pt2, COLOR_HIP_TAIL, 2)
        # 尾巴（Tail_Root → Tail_Mid → Tail_Tip，洋紅色）
        for i, j in TAIL_LINKS:
            if kpt_conf[i] > KP_CONF_THRES and kpt_conf[j] > KP_CONF_THRES:
                pt1 = tuple(map(int, kpts[i]))
                pt2 = tuple(map(int, kpts[j]))
                cv2.line(frame, pt1, pt2, COLOR_TAIL, 2)
        # 關鍵點
        for i in range(len(kpts)):
            if kpt_conf[i] > KP_CONF_THRES:
                pt = tuple(map(int, kpts[i]))
                cv2.circle(frame, pt, 3, COLOR_KPT, -1)

        # 將 hip.png 貼在 nose 關鍵點上，跟隨鼻子移動
        overlay_frame = self._get_overlay_frame()
        if overlay_frame is not None and len(kpts) > 0 and kpt_conf[0] > KP_CONF_THRES:
            nose_pt = tuple(map(int, kpts[0]))
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_w = max(1, x2 - x1)
                bbox_h = max(1, y2 - y1)
                base_width = int(np.clip(min(bbox_w, bbox_h) * 0.32, 52, 128))
            else:
                base_width = 72

            _overlay_image_centered(frame, overlay_frame, nose_pt, base_width)

        # 畫YOLO bbox/conf
        if bbox is not None and conf is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # 先畫白色粗框，再畫黑色細框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            label = f"conf: {conf:.2f}"
            # 只畫黑字標籤
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        # 顯示行為預測與信心值、四行為機率條
        if not show_info:
            return frame

        is_display_normal = (behavior_id == LOW_CONF_ID) or (float(confidence) < BEHAVIOR_MIN_CONFIDENCE)
        if is_display_normal:
            # 顯示層使用中文標籤（若可用字型）
            low_text = get_behavior_name(LOW_CONF_ID, use_text=False, fallback=LOW_CONF_TEXT, confidence=confidence)
            self.draw_prediction_on_frame(frame, low_text, 0.0, (200, 200, 200))
            if class_probs is not None and any(p > 0 for p in class_probs):
                self.draw_probability_bars(frame, class_probs, BEHAVIOR_CLASSES)
        elif behavior_id is not None and confidence > 0:
            # overlay 使用中文名稱
            behavior_name = get_behavior_name(behavior_id, use_text=False, fallback=str(behavior_id), confidence=confidence)
            self.draw_prediction_on_frame(frame, behavior_name, confidence, BEHAVIOR_COLORS.get(behavior_id, (255,255,255)))
            self.draw_probability_bars(frame, class_probs, BEHAVIOR_CLASSES)
        return frame
