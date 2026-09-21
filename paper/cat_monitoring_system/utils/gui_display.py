"""本地 GUI 視窗（cv2.imshow）的畫面貼合：等比例縮放＋補黑邊，避免畫面被拉伸。

cv2.WINDOW_NORMAL 視窗會把影像直接拉伸填滿整個視窗；視窗長寬比跟畫面不同
（例如 16:9 的 1280×720 視窗顯示 ESP32-CAM 的 4:3 640×480，橫向放大 2.0 倍、
縱向只放大 1.5 倍）畫面就會變形。Windows 預設的 Win32 後端又不支援
cv2.WINDOW_KEEPRATIO（只有 Qt 後端支援），所以改成每幀自己補黑邊，使用者任意
拖曳改變視窗大小也不會變形。
"""

import cv2
import numpy as np


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """把 frame 等比例縮放後置中，放進 width×height 的黑底畫布。

    寬高任一 <= 0（例如視窗被最小化）時原樣回傳，不做任何處理。
    """
    if width <= 0 or height <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    if (new_w, new_h) == (width, height):
        return resized
    canvas = np.zeros((height, width) + frame.shape[2:], dtype=frame.dtype)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def fit_to_window(frame: np.ndarray, window_name: str) -> np.ndarray:
    """依視窗目前的實際大小做 letterbox；取不到視窗尺寸（被關閉/最小化）就原樣回傳。"""
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame
    return letterbox(frame, win_w, win_h)
