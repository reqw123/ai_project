"""影片播放器（2026-10-01）：讀取影片並播放（畫面＋聲音），可以被 LINE「電腦指令」／cmd 分頁的動作控制。

用法：
    python -X utf8 1_play_video.py                       # 播放預設影片（DEFAULT_VIDEO）
    python -X utf8 1_play_video.py --video D:\\a.mp4      # 播放指定影片
    python -X utf8 1_play_video.py --no-audio --no-loop  # 不播聲音、播完一次就結束
也可以用環境變數 PLAY_VIDEO_PATH 指定影片。

按鍵（在影片視窗按；LINE／cmd 分頁的「影片播放器」動作也是送這些鍵）：
    p = 暫停（已經暫停就不動）   g = 繼續（已經在播就不動）   空白鍵 = 暫停／繼續切換
    r = 從頭重播                i = 顯示／隱藏資訊列          ESC 或 q = 結束（也可以直接關視窗）
    Ctrl+加號／減號 = 縮放視窗

畫面用 OpenCV 播，聲音用 PyAV 解出來存成暫存 WAV（%TEMP%\\play_video_audio\\，同一支影片只解一次）再交給 pygame 播，
不需要 ffmpeg 在 PATH 裡。影片沒有聲音、或 PyAV／pygame 不能用時，只播畫面。
畫面照「經過的時間」決定該顯示第幾幀（慢了就跳幀），跟聲音用同一個起點，暫停時兩邊一起停。
"""

import argparse
import hashlib
import os
import sys
import tempfile
import time
import wave

import cv2
import numpy as np

import _window_zoom  # Ctrl+加號／減號縮放視窗（tools/_window_zoom.py）

DEFAULT_VIDEO = r"C:\Users\homec\Downloads\gemini_generated_video_6be2fbd9.mp4"
WINDOW_NAME = "Video Player"   # cmd 動作表（影片播放器）的 escWindow／keyWindow 靠這個名稱找視窗，要改兩邊一起改
DISPLAY_MAX = (960, 540)       # 視窗一開始最大的大小（影片比這個小就用原尺寸）
WINDOW_SCALE_STEP, WINDOW_SCALE_MIN, WINDOW_SCALE_MAX = 0.10, 0.4, 2.5


def log(text):
    print(text, flush=True)


def extract_audio(video_path):
    """用 PyAV 把影片的聲音解成 44.1 kHz 立體聲 16-bit WAV（暫存資料夾，依檔案內容快取）；沒有聲音或失敗回傳 None。"""
    try:
        import av
    except Exception as e:
        log(f"⚠ 沒有 PyAV（{e}），只播畫面")
        return None
    st = os.stat(video_path)
    key = hashlib.sha1(f"{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    out_dir = os.path.join(tempfile.gettempdir(), "play_video_audio")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, key + ".wav")
    if os.path.exists(out) and os.path.getsize(out) > 44:
        return out
    try:
        with av.open(video_path) as container:
            if not container.streams.audio:
                log("（這支影片沒有聲音，只播畫面）")
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="stereo", rate=44100)
            tmp = out + ".part"
            with wave.open(tmp, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(44100)
                for frame in container.decode(stream):
                    for f in resampler.resample(frame):
                        w.writeframes(f.to_ndarray().tobytes())
                for f in resampler.resample(None):   # 把重新取樣器裡剩下的吐出來
                    w.writeframes(f.to_ndarray().tobytes())
        os.replace(tmp, out)
        return out
    except Exception as e:
        log(f"⚠ 解聲音失敗（{e}），只播畫面")
        return None


class Audio:
    """pygame 播聲音；不能用時所有方法都不做事。"""

    def __init__(self, wav_path):
        self.ok = False
        if not wav_path:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            pygame.mixer.music.load(wav_path)
            self.pg = pygame
            self.ok = True
        except Exception as e:
            log(f"⚠ 聲音播放初始化失敗（{e}），只播畫面")

    def _do(self, name):
        if self.ok:
            try:
                getattr(self.pg.mixer.music, name)()
            except Exception:
                pass

    def play(self):
        self._do("play")

    def pause(self):
        self._do("pause")

    def resume(self):
        self._do("unpause")

    def stop(self):
        self._do("stop")

    def close(self):
        if self.ok:
            try:
                self.pg.mixer.music.stop()
                self.pg.mixer.quit()
            except Exception:
                pass


def fmt(sec):
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def draw_info(frame, name, pos_sec, total_sec, paused, loop_no):
    """資訊列（OpenCV 的字型不能顯示中文，只用英數）"""
    h, w = frame.shape[:2]
    bar_h = max(28, h // 18)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    state = "PAUSED" if paused else "PLAYING"
    text = f"{state}  {fmt(pos_sec)} / {fmt(total_sec)}  loop {loop_no}  |  {name}"
    scale = bar_h / 42
    cv2.putText(frame, text, (12, h - int(bar_h * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)
    # 進度條
    if total_sec > 0:
        x = int(w * min(1.0, pos_sec / total_sec))
        cv2.rectangle(frame, (0, h - bar_h - 4), (x, h - bar_h), (80, 200, 255), -1)


def main():
    ap = argparse.ArgumentParser(description="影片播放器（可被 LINE 電腦指令控制）")
    ap.add_argument("--video", default=os.environ.get("PLAY_VIDEO_PATH") or DEFAULT_VIDEO)
    ap.add_argument("--no-audio", action="store_true", help="不播聲音")
    ap.add_argument("--no-loop", action="store_true", help="播完一次就結束（預設循環播放）")
    args = ap.parse_args()

    path = args.video
    if not os.path.isfile(path):
        log(f"✘ 找不到影片：{path}")
        return 2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log(f"✘ 打不開影片：{path}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1 <= fps <= 240):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_sec = total_frames / fps if total_frames > 0 else 0
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    name = os.path.basename(path)
    name_ascii = name.encode("ascii", "replace").decode()
    log(f"▶ 影片播放器：{name}（{vw}x{vh}、{fps:.1f} fps、{total_sec:.1f} 秒）")
    log("按鍵：p 暫停｜g 繼續｜空白 暫停／繼續｜r 重播｜i 資訊列｜ESC／q 結束｜Ctrl+加減 縮放")

    audio = Audio(None if args.no_audio else extract_audio(path))
    if audio.ok:
        log("🔊 聲音已載入")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fit = min(1.0, DISPLAY_MAX[0] / max(vw, 1), DISPLAY_MAX[1] / max(vh, 1))
    base_w, base_h = max(1, int(vw * fit)), max(1, int(vh * fit))
    cv2.resizeWindow(WINDOW_NAME, base_w, base_h)
    window_scale = 1.0

    paused = False
    show_info = True
    loop_no = 1
    pos = -1            # 目前顯示的是第幾幀
    frame = None
    start = time.perf_counter()
    paused_at = None
    audio_restart_pending = False   # 暫停中按重播：聲音已停止，繼續時要從頭 play，不能只 unpause
    exit_reason = "結束"

    def restart():
        nonlocal pos, start, paused_at, loop_no, frame, audio_restart_pending
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pos, frame = -1, None
        start = time.perf_counter()
        paused_at = time.perf_counter() if paused else None
        audio.stop()
        if paused:
            audio_restart_pending = True
        else:
            audio.play()

    audio.play()
    try:
        while True:
            now = time.perf_counter()
            elapsed = (paused_at if paused else now) - start
            want = int(elapsed * fps)
            if frame is None or (not paused and want > pos):   # 暫停中重播也要讀出第一幀
                if total_frames and want >= total_frames:
                    if args.no_loop:
                        exit_reason = "播完"
                        break
                    loop_no += 1
                    restart()
                    continue
                # 慢了就跳幀：grab 比 read 快（不解碼成影像）
                while pos < want - 1:
                    if not cap.grab():
                        break
                    pos += 1
                ok, img = cap.read()
                if not ok:
                    if args.no_loop:
                        exit_reason = "播完"
                        break
                    loop_no += 1
                    restart()
                    continue
                pos += 1
                frame = img
            if frame is not None:
                show = frame.copy()
                if show_info:
                    draw_info(show, name_ascii, pos / fps, total_sec, paused, loop_no)
                cv2.imshow(WINDOW_NAME, show)
            # 等到下一幀該出來的時間（最多 30 毫秒，按鍵才靈敏）
            next_due = start + (pos + 1) / fps
            delay = 30 if paused else max(1, min(30, int((next_due - time.perf_counter()) * 1000)))
            key = cv2.waitKey(delay)
            zoom = _window_zoom.poll(WINDOW_NAME)
            if zoom:
                window_scale = _window_zoom.clamp_scale(window_scale, zoom, WINDOW_SCALE_STEP, WINDOW_SCALE_MIN, WINDOW_SCALE_MAX)
                cv2.resizeWindow(WINDOW_NAME, int(base_w * window_scale), int(base_h * window_scale))
            # 使用者按右上角 X 關掉視窗
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                exit_reason = "視窗關閉"
                break
            if key == -1:
                continue
            key &= 0xFF
            if key in (27, ord("q")):
                exit_reason = "ESC" if key == 27 else "q"
                break
            if key in (ord("p"), ord(" ")) and not paused:
                paused, paused_at = True, time.perf_counter()
                audio.pause()
                log(f"⏸ 暫停（{fmt(pos / fps)}）")
            elif key in (ord("g"), ord(" ")) and paused:
                start += time.perf_counter() - paused_at   # 暫停的時間不算
                paused, paused_at = False, None
                if audio_restart_pending:
                    audio_restart_pending = False
                    audio.play()
                else:
                    audio.resume()
                log(f"▶ 繼續（{fmt(pos / fps)}）")
            elif key == ord("r"):
                loop_no = 1
                restart()
                log("🔁 從頭重播")
            elif key == ord("i"):
                show_info = not show_info
    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
    finally:
        audio.close()
        cap.release()
        cv2.destroyAllWindows()
    log(f"■ 影片播放器已關閉（{exit_reason}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
