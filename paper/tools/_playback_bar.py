"""影片視窗底部的播放控制列（時間拉桿＋時間／幀數／倍速資訊）與倍速節流時鐘。內部共用模組，底線開頭所以不會出現在設定視窗的腳本下拉選單。

參考 cat_pose/video_infer_save.py 的自繪拉桿：不用 cv2.createTrackbar（原生元件窄、樣式不能改、拖曳中沒有回饋），
改成畫進畫面最下方的一條色塊（band），自己接滑鼠事件：
    - 點擊／拖曳拉桿跳轉；拖曳中只顯示預覽（時間＋幀號），放開才真的跳——跳轉要重置推論狀態而且長影片 seek 很慢，
      拖曳中每動一下就 seek 會卡。
    - 滑鼠在拉桿上滾滾輪：一格前進／後退 WHEEL_STEP_SEC 秒。
    - 畫面上的文字一律英文（cv2.putText 畫不出中文），時間一律 HH:MM:SS（長達數小時的影片也清楚）。

用法（見 1_run_video_inference.py）：
    bar = SeekBar()
    cv2.setMouseCallback(WINDOW_NAME, bar.on_mouse)
    band = bar.render(width, y_offset=body_h, frame_idx=..., total_frames=..., fps=..., paused=..., speed=...)
    canvas = np.vstack([body, band])
    target = bar.take_pending()      # 使用者放開拉桿／滾滾輪要跳去的幀號（None＝沒有）

PlaybackClock 依「影片時間」節流播放速度（1x＝跟影片實際時間一樣快）。推論跟不上時：
    - 呼叫端允許跳幀（快轉，見 1_run_video_inference.py 的 FRAME_SKIP_ABOVE_SPEED）→ 用 due_frame() 算出「現在
      應該播到第幾幀」，直接跳過去追上倍速；跳幀期間 ST-GCN 序列不連續，呼叫端要暫停行為分類，控制列顯示 FAST-FWD。
    - 不允許跳幀（1x 以下，要完整推論）→ 不跳，只是實際倍速達不到，畫面上會顯示 actual 倍速。
"""

import math
import time

import cv2
import numpy as np

SPEED_STEPS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)  # 6 小時影片 256x ≈ 1.4 分鐘
WHEEL_STEP_SEC = 1.0  # 拉桿上滾一格滾輪跳幾秒

BAND_BG = (28, 28, 28)          # 色塊底色：接近黑但不是純黑（純黑跟黑貓、letterbox 黑邊分不出來）
TRACK_BG = (75, 75, 75)         # 拉桿未播放部分
ACCENT = (60, 200, 255)         # 已播放部分／把手（BGR 暖黃橘）
PREVIEW_COLOR = (255, 255, 255)
PLAYING_COLOR = (120, 230, 120)
PAUSED_COLOR = (80, 170, 255)
FAST_FWD_COLOR = (255, 120, 255)
HINT_COLOR = (170, 170, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fmt_hms(sec):
    """秒數 → HH:MM:SS（無條件捨去到秒；超過 99 小時照樣顯示完整時數）。"""
    total = int(max(0.0, float(sec)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_input(text):
    """把使用者輸入的時間轉成秒數：「12.5」（秒）、「3:25」（分:秒）、「1:02:03」（時:分:秒）。
    格式不對回傳 None。"""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) > 3:
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return sec


def speed_label(speed):
    return f"{speed:g}x"


def step_speed(speed, direction):
    """在 SPEED_STEPS 上往上（direction>0）或往下（<0）走一格；不在階梯上的值先對齊到最接近的一格。"""
    if speed in SPEED_STEPS:
        i = SPEED_STEPS.index(speed)
    else:
        i = min(range(len(SPEED_STEPS)), key=lambda k: abs(math.log(SPEED_STEPS[k] / max(speed, 1e-6))))
    i = max(0, min(len(SPEED_STEPS) - 1, i + (1 if direction > 0 else -1)))
    return SPEED_STEPS[i]


class PlaybackClock:
    """依影片時間節流：每顯示一幀呼叫 delay_ms(該幀的影片秒數, 倍速)，回傳 cv2.waitKey 該等幾毫秒。

    以「錨點」（某一刻的牆鐘時間＋當時的影片秒數）推算每一幀該出現的時間。不跳幀時（catch_up=False）落後太多
    （推論太慢、剛從暫停恢復）就把錨點重設到現在，不會為了追進度連續狂播；跳幀快轉時（catch_up=True）保留錨點，
    呼叫端用 due_frame() 跳到該播的位置追上（連 seek 本身花的時間也算進去，實際倍速才會接近設定值）。
    actual_speed 是最近實際播放倍速（指數平均），給畫面顯示用。

    暫停、跳轉、重播之後要呼叫 reset()，否則 due_frame() 會把暫停的那段時間也當成「該快轉過去」。"""

    MAX_LAG_SEC = 0.25    # 不跳幀時落後超過這麼多就重設錨點
    MAX_JUMP_SEC = 1.0    # 影片時間比預期多跳這麼多（seek）就重設錨點
    STALE_SEC = 2.0       # 超過這麼久沒有顯示新幀（暫停、等輸入）就不再推算 due_frame（保險，正常靠 reset()）

    def __init__(self):
        self.reset()

    def reset(self):
        self._anchor = None       # (牆鐘, 影片秒數, 倍速)
        self._last = None         # 上一次呼叫的 (牆鐘, 影片秒數)
        self.actual_speed = None

    def delay_ms(self, video_sec, speed, now=None, catch_up=False):
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            dw, dv = now - self._last[0], video_sec - self._last[1]
            max_dv = self.MAX_JUMP_SEC + 2.0 * speed * max(dw, 0.0)   # 快轉時一幀可能跳好幾秒，不算 seek
            if dw > 1e-4 and 0 < dv <= max_dv:
                rate = dv / dw
                self.actual_speed = rate if self.actual_speed is None else 0.8 * self.actual_speed + 0.2 * rate
            elif not 0 < dv <= max_dv:
                self.actual_speed = None
        self._last = (now, video_sec)

        if speed <= 0:
            self._anchor = None
            return 1
        a = self._anchor
        # 比預期多跳一大段＝使用者跳轉了（保險，呼叫端跳轉時本來就會 reset()）；快轉中（catch_up）停在目標之後的
        # 關鍵幀本來就會多跳一點，不能當成跳轉而重設錨點，否則多跳的那段永遠不會被扣回來、實際倍速會偏高
        jumped = not catch_up and video_sec - a[1] > self.MAX_JUMP_SEC + (now - a[0]) * speed if a else False
        if a is None or a[2] != speed or video_sec < a[1] or jumped:
            self._anchor = (now, video_sec, speed)
            return 1
        wait = a[0] + (video_sec - a[1]) / speed - now
        if wait < -self.MAX_LAG_SEC and not catch_up:
            self._anchor = (now, video_sec, speed)
            return 1
        return max(1, int(round(wait * 1000)))

    def due_frame(self, fps, now=None):
        """依錨點推算「現在應該播到第幾幀」（0 起算、浮點）；沒有錨點或太久沒顯示新幀回傳 None。"""
        now = time.perf_counter() if now is None else now
        a = self._anchor
        if a is None or self._last is None or now - self._last[0] > self.STALE_SEC:
            return None
        return (a[1] + (now - a[0]) * a[2]) * fps


class FastForwardReader:
    """快轉期間的讀幀器（PyAV）：快轉一開始跳幀就改由它供應畫面，直到回到不跳幀的倍速。

    為什麼不用 OpenCV：cap.set(CAP_PROP_POS_FRAMES) 一定要落在精確的那一幀（先跳關鍵幀再一路解碼＋轉色到目標），
    1080p 長影片實測每次 0.4～2.9 秒；cap.grab() 略過一幀也要約 14ms。PyAV 可以只跳關鍵幀（約 15ms），
    略過的幀只解碼不轉色（約 2ms／幀），高倍速快轉才不會變幻燈片。

    jump()：往前跳到 target 附近，絕不超過 target——
        - 距離比關鍵幀間隔短（或在 KEYFRAME_MIN_JUMP 以內）：從目前位置照順序往下解碼到 target（跳關鍵幀也只會落在後面，沒用）；
        - 距離夠遠：跳到 target「之前」最近的關鍵幀，它在目前位置前面就停在那（不精確，但快）；
        - 那個關鍵幀在目前位置之前（第一次進快轉、或還不知道關鍵幀間隔）：從它照順序解到 target。
      以前會改跳 target「之後」的關鍵幀，關鍵幀間隔長的影片（例如每 10 秒一個）一下就衝過頭好幾秒、
      然後畫面停住等時間追上，已經拿掉。關鍵幀間隔在解碼途中自動量（看 frame.key_frame）。
    next()：照順序讀下一幀。都回傳 (幀號, BGR 畫面)，讀到結尾回傳 None。PyAV 不能用、或畫面尺寸跟 OpenCV 不一致（例如帶旋轉資訊的手機影片）就 disabled，
    呼叫端退回 OpenCV。"""

    KEYFRAME_MIN_JUMP = 40  # 要跳的幀數超過這個才跳關鍵幀；更近的直接往下解碼（只解碼不轉色，1080p 約 2ms/幀）比較順

    def __init__(self, path, fps, expected_hw=None):
        self._path, self._fps, self._expected_hw = path, float(fps), expected_hw
        self._container = self._stream = self._frames = None
        self._start = 0
        self._last_idx = -1
        self._last_kf = None   # 照順序解碼時最近看到的關鍵幀幀號（seek 後重新起算）
        self.gop = None        # 量到的關鍵幀間隔（幀數，取最大值）；還沒量到是 None
        self.disabled = False
        self.active = False   # 解碼位置是否接在上一次回傳的那一幀後面（跳轉／重播後要設 False）

    def _fail(self, why):
        print(f"⚠ 快轉改用 OpenCV（{why}）")
        self.close()
        self.disabled = True

    def _open(self):
        try:
            import av
            self._container = av.open(self._path)
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._start = self._stream.start_time or 0
            return True
        except Exception as e:  # noqa: BLE001 — 沒有 PyAV／開不了檔就退回 OpenCV
            self._fail(f"PyAV 無法開啟影片：{e}")
            return False

    def _idx(self, frame, fallback):
        if frame.pts is None:
            return fallback
        return int(round(float((frame.pts - self._start) * self._stream.time_base) * self._fps))

    def _decode_next(self):
        """解下一幀（不轉色）；結尾回傳 None。"""
        try:
            frame = next(self._frames)
        except StopIteration:
            self.active = False
            return None
        if frame.key_frame:
            idx = self._idx(frame, None)
            if idx is not None:
                if self._last_kf is not None and idx > self._last_kf:
                    self.gop = max(self.gop or 0, idx - self._last_kf)
                self._last_kf = idx
        return frame

    def _seek_keyframe(self, target):
        """跳到 target 之前最近的關鍵幀，回傳解出的第一幀。"""
        ts = self._start + int(max(0.0, target / self._fps) / self._stream.time_base)
        self._container.seek(ts, stream=self._stream, backward=True, any_frame=False)
        self._last_kf = None   # 跳過之後前後兩個關鍵幀不相鄰，間隔重新量
        self._frames = self._container.decode(self._stream)
        return self._decode_next()

    def _to_bgr(self, frame, idx):
        img = frame.to_ndarray(format="bgr24")
        if self._expected_hw is not None and img.shape[:2] != tuple(self._expected_hw):
            self._fail(f"PyAV 畫面 {img.shape[1]}x{img.shape[0]} 跟 OpenCV 不一致，可能有旋轉資訊")
            return None
        self._last_idx = idx
        return idx, img

    def jump(self, target, current):
        """從 current（下一個要讀的幀號）往前跳到 target 附近。"""
        if self.disabled or (self._container is None and not self._open()):
            return None
        try:
            if self.active and target - current <= max(self.KEYFRAME_MIN_JUMP, self.gop or 0):
                frame = self._decode_next()                        # 近的：照順序往下解（只解碼不轉色）
            else:
                frame = self._seek_keyframe(target)                # 目標之前最近的關鍵幀
                if frame is not None and self._idx(frame, target) >= current:
                    self.active = True                             # 在目前位置前面：停在這個關鍵幀
                    return self._to_bgr(frame, self._idx(frame, target))
                # 關鍵幀在目前位置之前：從它照順序解到目標（不會衝過頭）。目標之前最近的關鍵幀就是它，
                # 所以關鍵幀間隔至少有這麼長；先記下來，之後這麼近的跳轉直接從目前位置往下解，不用每次都退回關鍵幀重解
                if frame is not None:
                    self.gop = max(self.gop or 0, target - self._idx(frame, target) + 1)
            while frame is not None and self._idx(frame, target) < target:
                frame = self._decode_next()
            if frame is None:
                return None
            self.active = True
            return self._to_bgr(frame, self._idx(frame, target))
        except Exception as e:  # noqa: BLE001
            self._fail(f"PyAV 解碼失敗：{e}")
            return None

    def next(self):
        """照順序讀下一幀；結尾或沒在快轉狀態回傳 None。"""
        if self.disabled or not self.active:
            return None
        try:
            frame = self._decode_next()
            if frame is None:
                return None
            return self._to_bgr(frame, self._idx(frame, self._last_idx + 1))
        except Exception as e:  # noqa: BLE001
            self._fail(f"PyAV 解碼失敗：{e}")
            return None

    def close(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:  # noqa: BLE001
                pass
        self._container = self._stream = self._frames = None
        self.active = False


def band_height(width):
    s = max(0.6, width / 1280.0)
    return int(round(62 * s))


class SeekBar:
    """畫面最下方的播放控制列：上半是可點擊／拖曳的時間拉桿，下半一行狀態文字。座標一律是 imshow 那張
    合成畫面的像素座標（WINDOW_NORMAL 視窗縮放後，OpenCV 回呼給的也是影像座標，不用自己換算）。"""

    def __init__(self, wheel_step_sec=WHEEL_STEP_SEC):
        self.wheel_step_sec = wheel_step_sec
        self.enabled = False      # 沒有總長度的來源（串流）不能拖
        self.dragging = False
        self.hover_frame = None   # 滑鼠懸停在拉桿上對應的幀號
        self.drag_frame = None    # 拖曳中的目標幀號（放開才送出）
        self.dirty = False        # 懸停／拖曳狀態變了，暫停中要重畫
        self._pending = None
        self._hit = None          # (x0, y0, x1, y1) 可點擊範圍
        self._track = None        # (x0, w) 軌道
        self._total = 0
        self._fps = 30.0
        self._cur = 0

    # ── 滑鼠 ──
    def _frame_at(self, x):
        x0, w = self._track
        ratio = min(1.0, max(0.0, (x - x0) / float(max(1, w))))
        return int(round(ratio * max(0, self._total - 1)))

    def on_mouse(self, event, x, y, flags, _param=None):
        if not self.enabled or self._hit is None:
            return
        hx0, hy0, hx1, hy1 = self._hit
        inside = hx0 <= x <= hx1 and hy0 <= y <= hy1
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            self.dragging = True
            self.drag_frame = self._frame_at(x)
            self.dirty = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.drag_frame = self._frame_at(x)
                self.dirty = True
            new_hover = self._frame_at(x) if inside else None
            if new_hover != self.hover_frame:
                self.hover_frame = new_hover
                self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self._pending = self.drag_frame
            self.drag_frame = None
            self.dirty = True
        elif event == cv2.EVENT_MOUSEWHEEL and inside:
            delta = (flags >> 16) & 0xFFFF   # 高 16 位是有號的滾動量
            if delta >= 0x8000:
                delta -= 0x10000
            base = self._pending if self._pending is not None else self._cur
            step = int(round(self.wheel_step_sec * self._fps)) * (1 if delta > 0 else -1)
            self._pending = max(0, min(self._total - 1, base + step))
            self.dirty = True

    def take_pending(self):
        """取走待處理的跳轉目標幀號（沒有回傳 None）。"""
        target, self._pending = self._pending, None
        return target

    def cancel(self):
        """換影片時清掉殘留的拖曳／跳轉請求。"""
        self.dragging = False
        self.drag_frame = None
        self.hover_frame = None
        self._pending = None

    # ── 繪製 ──
    def render(self, width, *, y_offset, frame_idx, total_frames, fps, paused, speed, actual_speed=None,
               fast_forward=False):
        """畫出控制列（寬 width、高 band_height(width)）。frame_idx 是目前畫面的 0 起算幀號；
        y_offset 是這條色塊在合成畫面裡的起始 y（滑鼠命中判斷用）。fast_forward＝正在跳幀快轉
        （狀態顯示 FAST-FWD，拉桿上方提示行為分類暫停）。"""
        s = max(0.6, width / 1280.0)
        h = band_height(width)
        img = np.full((h, max(1, width), 3), BAND_BG, dtype=np.uint8)
        fps = fps if fps and fps > 0 else 30.0
        self._total, self._fps, self._cur = int(total_frames), float(fps), int(frame_idx)
        self.enabled = total_frames > 1

        margin = int(round(14 * s))
        label_h = int(round(18 * s))
        track_row_h = int(round(18 * s))
        track_h = max(4, int(round(6 * s)))
        x0, tw = margin, max(1, width - 2 * margin)
        ty = int(round(4 * s)) + label_h + (track_row_h - track_h) // 2
        self._track = (x0, tw)
        self._hit = (0, y_offset, width - 1, y_offset + int(round(4 * s)) + label_h + track_row_h)
        th = max(1, int(round(s)))

        shown = self.drag_frame if self.dragging and self.drag_frame is not None else frame_idx
        cv2.rectangle(img, (x0, ty), (x0 + tw, ty + track_h), TRACK_BG, -1)
        if self.enabled:
            ratio = min(1.0, max(0.0, shown / float(total_frames - 1)))
            fx = x0 + int(round(tw * ratio))
            cv2.rectangle(img, (x0, ty), (fx, ty + track_h), ACCENT, -1)
            r = max(5, int(round(8 * s)))
            cy = ty + track_h // 2
            cv2.circle(img, (fx, cy), r, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(img, (fx, cy), max(1, r - 2), ACCENT, -1, cv2.LINE_AA)

            preview = self.drag_frame if self.dragging else self.hover_frame
            if preview is None and fast_forward and not paused:
                note = "Fast-forward: frames skipped, ST-GCN behavior paused"
                fs = 0.42 * s
                cv2.putText(img, note, (x0, int(round(4 * s)) + label_h - int(round(4 * s))), FONT, fs,
                            FAST_FWD_COLOR, th, cv2.LINE_AA)
            if preview is not None:
                label = f"{fmt_hms(preview / fps)}  frame {preview + 1}"
                fs = 0.45 * s
                (lw, _), _ = cv2.getTextSize(label, FONT, fs, th)
                px = x0 + int(round(tw * preview / float(total_frames - 1)))
                lx = max(2, min(width - lw - 2, px - lw // 2))
                ly = int(round(4 * s)) + label_h - int(round(4 * s))
                cv2.putText(img, label, (lx, ly), FONT, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.putText(img, label, (lx, ly), FONT, fs, PREVIEW_COLOR, th, cv2.LINE_AA)

        # 狀態列：PLAYING/PAUSED  目前時間 / 總長  幀號  倍速（實際倍速）  ……右邊是按鍵提示
        cur_sec = frame_idx / fps
        if total_frames > 0:
            time_txt = f"{fmt_hms(cur_sec)} / {fmt_hms(total_frames / fps)}"
            frame_txt = f"Frame {frame_idx + 1}/{total_frames}"
        else:
            time_txt = f"{fmt_hms(cur_sec)}  LIVE"
            frame_txt = f"Frame {frame_idx + 1}"
        spd = f"Speed {speed_label(speed)}"
        if actual_speed is not None and not paused and abs(actual_speed - speed) > 0.05 * speed:
            spd += f" (actual {actual_speed:.2f}x)" if actual_speed < 10 else f" (actual {actual_speed:.0f}x)"
        if paused:
            status, status_color = "PAUSED", PAUSED_COLOR
        elif fast_forward:
            status, status_color = "FAST-FWD", FAST_FWD_COLOR
        else:
            status, status_color = "PLAYING", PLAYING_COLOR
        fs = 0.5 * s
        y = h - int(round(9 * s))
        x = x0
        for text, color in ((status, status_color), (time_txt, (255, 255, 255)),
                            (frame_txt, (220, 220, 220)), (spd, ACCENT)):
            cv2.putText(img, text, (x, y), FONT, fs, color, th, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, fs, th)[0][0] + int(round(22 * s))
        hint = "[ ] 10s   { } 1min   -/= speed   0 1x   t go to"
        hfs = 0.42 * s
        hw = cv2.getTextSize(hint, FONT, hfs, th)[0][0]
        if x + hw < width - margin:   # 畫面太窄就不畫提示，不要跟狀態文字疊在一起
            cv2.putText(img, hint, (width - margin - hw, y), FONT, hfs, HINT_COLOR, th, cv2.LINE_AA)
        self.dirty = False
        return img
