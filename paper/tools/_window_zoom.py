"""OpenCV 視窗的 Ctrl+加號／Ctrl+減號縮放偵測（Windows）。內部共用模組，底線開頭所以不會出現在設定視窗的腳本下拉選單。

為什麼不直接用 cv2.waitKey()：waitKey 收不到 Ctrl 組合鍵（Ctrl+「=」「-」不會產生字元，回傳 -1），
所以改成輪詢鍵盤狀態（GetAsyncKeyState）。只在「這個 OpenCV 視窗是目前有焦點的視窗」時才回應——
在別的程式（例如設定視窗終端機的輸入框、文字編輯器）按 Ctrl+加減，不會誤觸發這邊的縮放。

用法（每幀呼叫一次，例如緊接在 cv2.waitKey() 之後）：

    import _window_zoom
    zoom = _window_zoom.poll(WINDOW_NAME)   # 這段時間累積的縮放次數：正＝放大、負＝縮小、0＝沒有動作
    if zoom:
        window_scale = _window_zoom.clamp_scale(window_scale, zoom, step=0.10, lo=0.5, hi=2.0)
        cv2.resizeWindow(WINDOW_NAME, int(w * window_scale), int(h * window_scale))

按住不放會自動連續縮放（先等 0.4 秒，之後約每 0.12 秒一次）。非 Windows 一律回傳 0（不做事）。

實作：背景執行緒每 10 毫秒偵測一次並把事件累積起來，poll() 只是把累積的次數取走——不在主迴圈裡直接偵測，
是因為推論很慢時（每幀 100 毫秒以上）輕點一下的按鍵可能整個落在兩次偵測之間而漏掉。
"""

import os
import threading
import time
from collections import deque

_VK_CONTROL = 0x11
_PLUS_KEYS = (0xBB, 0x6B)    # 主鍵盤「=／+」鍵、數字鍵盤「+」
_MINUS_KEYS = (0xBD, 0x6D)   # 主鍵盤「-／_」鍵、數字鍵盤「-」
_REPEAT_FIRST_SEC = 0.4
_REPEAT_NEXT_SEC = 0.12
_SAMPLE_SEC = 0.01

_events = deque()             # 尚未被 poll() 取走的縮放事件（+1／-1）
_window_name = None           # 目前要偵測的視窗（以最近一次 poll() 傳入的為準）
_thread = None
_lock = threading.Lock()


def _user32():
    import ctypes
    user32 = ctypes.windll.user32
    user32.GetAsyncKeyState.restype = ctypes.c_short  # SHORT：預設 c_int 的高位元不可靠
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    return user32


def _is_down(user32, vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def _current_direction(user32, window_name):
    """視窗有焦點、按著 Ctrl，且按著加／減號時回傳 +1／-1，否則 0。"""
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd or user32.GetForegroundWindow() != hwnd or not _is_down(user32, _VK_CONTROL):
        return 0
    if any(_is_down(user32, k) for k in _PLUS_KEYS):
        return 1
    if any(_is_down(user32, k) for k in _MINUS_KEYS):
        return -1
    return 0


def _watch():
    held = 0
    next_fire = 0.0
    try:
        user32 = _user32()
    except Exception:  # noqa: BLE001
        return
    while True:
        time.sleep(_SAMPLE_SEC)
        name = _window_name
        try:
            direction = _current_direction(user32, name) if name else 0
        except Exception:  # noqa: BLE001 — 偵測失敗就當作沒有按鍵
            direction = 0
        now = time.monotonic()
        if direction == 0:
            held = 0
        elif direction != held:               # 剛按下（或換方向）：立刻觸發一次
            held = direction
            next_fire = now + _REPEAT_FIRST_SEC
            _events.append(direction)
        elif now >= next_fire:                # 按住不放：連續觸發
            next_fire = now + _REPEAT_NEXT_SEC
            _events.append(direction)


def poll(window_name):
    """回傳自上次呼叫以來累積的縮放次數（正＝Ctrl+加號放大、負＝Ctrl+減號縮小、0＝沒動作）。
    視窗沒有焦點、沒按 Ctrl、或非 Windows 一律 0。"""
    global _window_name, _thread
    if os.name != "nt":
        return 0
    _window_name = window_name
    if _thread is None:
        with _lock:
            if _thread is None:
                _thread = threading.Thread(target=_watch, name="window-zoom-watch", daemon=True)
                _thread.start()
    total = 0
    while _events:
        try:
            total += _events.popleft()
        except IndexError:
            break
    return total


def clamp_scale(scale, direction, step=0.10, lo=0.50, hi=2.00):
    """依方向（可為累積次數，例如 +2）調整縮放倍率並夾在 [lo, hi]，四捨五入到兩位小數避免浮點誤差累積。"""
    return round(min(hi, max(lo, scale + direction * step)), 2)
