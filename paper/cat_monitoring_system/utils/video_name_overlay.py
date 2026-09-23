"""
在播放畫面右下角畫出「目前播放的影片」標籤，例如：

    ▶ [3/12] lick_32.mp4

給 paper/tools/ 底下會播放影片的 GUI 腳本共用（1_run_video_inference.py、
1_measure_ear_distance_single_video.py、2_run_dual_model_compare.py 等），
統一放右下角：這些腳本的左上（時間/預測標籤）、右上（狀態框/導覽列）、
左下（行為統計面板）都已經有東西，右下角是共同的空位。

影片檔名常有中文（例如「7月3日 (2).mp4」），cv2.putText 畫不出中文，所以用 PIL +
系統中文字型把標籤畫成一小塊影像並快取（同一個標籤只畫一次），每幀只做貼圖，
不拖慢播放；找不到 PIL 或字型時退回 cv2.putText（非 ASCII 字元顯示成 ?）。
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",      # 微軟正黑體
    r"C:\Windows\Fonts\msyh.ttc",      # 微軟雅黑
    r"C:\Windows\Fonts\mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

_BG = (0, 0, 0)
_BG_ALPHA = 0.62
_BORDER = (90, 90, 90)
_TEXT = (240, 240, 240)
_ICON = (80, 220, 120)          # BGR 綠色播放三角形


@lru_cache(maxsize=8)
def _pil_font(px):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for fp in _FONT_CANDIDATES:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, px)
            except OSError:
                continue
    return None


@lru_cache(maxsize=64)
def _render_label(text, px):
    """把 ▶ + 文字畫成 (patch_bgr, alpha_mask)；同一組 (text, px) 只畫一次。"""
    font = _pil_font(px)
    pad = max(4, px // 3)
    icon = int(px * 0.75)
    gap = max(4, px // 3)
    if font is not None:
        from PIL import Image, ImageDraw
        l, t, r, b = font.getbbox(text)
        tw, th = r - l, b - t
        W, H = pad * 2 + icon + gap + tw, pad * 2 + max(th, icon)
        img = Image.new("RGB", (W, H), _BG)
        d = ImageDraw.Draw(img)
        d.text((pad + icon + gap - l, pad + (H - pad * 2 - th) // 2 - t), text, font=font,
               fill=_TEXT[::-1])
        patch = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        text = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text)
        fs = px / 30.0
        thk = max(1, px // 14)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thk)
        W, H = pad * 2 + icon + gap + tw, pad * 2 + max(th + base, icon)
        patch = np.zeros((H, W, 3), np.uint8)
        cv2.putText(patch, text, (pad + icon + gap, pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, _TEXT, thk, cv2.LINE_AA)
    cy = H // 2
    tri = np.array([[pad, cy - icon // 2], [pad, cy + icon // 2], [pad + icon, cy]], np.int32)
    cv2.fillPoly(patch, [tri], _ICON, cv2.LINE_AA)
    cv2.rectangle(patch, (0, 0), (W - 1, H - 1), _BORDER, 1)
    # 文字/圖示/邊框不透明，底色半透明
    mask = (patch.max(axis=2, keepdims=True) > 0).astype(np.float32)
    alpha = np.where(mask > 0, 1.0, _BG_ALPHA).astype(np.float32)
    return patch, alpha


def draw_video_name_label(frame, video, index=None, total=None, ui_scale=None,
                          margin=None, bottom_offset=0):
    """在 frame 右下角畫目前播放的影片名稱（原地修改並回傳 frame）。

    video:  影片路徑（str/Path，只取檔名）或已經是要顯示的名稱；None 時不畫
    index/total: 目前是第幾支（0-based）／共幾支；兩者都有才顯示 [i/N]
    ui_scale: 字級倍率；None 時依畫面寬度自動估（以 960px 寬為 1.0）
    bottom_offset: 往上抬高的像素（右下角另有東西時用）
    """
    if frame is None or video is None:
        return frame
    h, w = frame.shape[:2]
    s = ui_scale if ui_scale else max(0.6, w / 960.0)
    name = Path(str(video)).name or str(video)
    prefix = f"[{index + 1}/{total}] " if index is not None and total else ""
    px = max(12, int(round(17 * s)))
    m = margin if margin is not None else max(6, int(10 * s))

    text = prefix + name
    patch, alpha = _render_label(text, px)
    # 太長時從檔名左側截斷（保留編號與副檔名），避免超出畫面
    while patch.shape[1] > w - m * 2 and len(name) > 6:
        name = "…" + name[2:]
        patch, alpha = _render_label(prefix + name, px)

    ph, pw = patch.shape[:2]
    x2, y2 = w - m, h - m - int(bottom_offset)
    x1, y1 = max(0, x2 - pw), max(0, y2 - ph)
    patch, alpha = patch[ph - (y2 - y1):, pw - (x2 - x1):], alpha[ph - (y2 - y1):, pw - (x2 - x1):]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (patch * alpha + roi * (1 - alpha)).astype(np.uint8)
    return frame
