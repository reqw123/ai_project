"""
攝影機伺服器（合併版）
支援：
  - 本機 Webcam    → SOURCE = 0  (或其他 index)
  - ESP32-CAM      → SOURCE = "http://192.168.4.1:81/stream"

功能：
  1. 將畫面推送到 Node-RED（可透過 ENABLE_NODE_RED 開關）
  2. 點擊視窗左上角按鈕 或按 R 鍵切換本地錄影
  3. 按 Q 鍵結束程式
"""
import cv2
import time
import datetime
import requests
import socket
import threading
import queue
from collections import deque
from datetime import datetime as dt

# ==============================
# 基本設定
# ==============================
SOURCE          = 0                                     # int=webcam, str=ESP32-CAM URL
NODE_RED_URL    = "http://localhost:1880/camera/frame"
ENABLE_NODE_RED = True     # False → 不推送到 Node-RED
TARGET_FPS      = 15
JPEG_QUALITY    = 75
RESIZE_WIDTH    = 640      # 0 = 不縮放
SHOW_PREVIEW    = True
REQUEST_TIMEOUT = 0.5
PRINT_EVERY_SEC = 2.0
FRAME_QUEUE_SIZE = 3


# ==============================
# 工具函數
# ==============================
def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def log_msg(msg: str, level: str = "INFO"):
    timestamp = dt.now().strftime("%H:%M:%S")
    icons = {"INFO": "ℹ️", "OK": "✅", "ERROR": "❌", "WARN": "⚠️", "STAT": "📊"}
    print(f"[{timestamp}] {icons.get(level, '•')} {msg}")


# ==============================
# 攝影機擷取線程
# 支援 webcam (int) 與 ESP32-CAM URL (str)
# ==============================
class CameraCapture(threading.Thread):
    def __init__(self, source=SOURCE, frame_queue=None):
        super().__init__(daemon=True)
        self.source      = source
        self.frame_queue = frame_queue or queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running     = True
        self.cap         = None
        self.frame_count = 0

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            log_msg(f"無法開啟影像來源: {self.source}", "ERROR")
            self.running = False
            return

        # Webcam 才設定硬體參數（URL stream 設了沒用）
        if isinstance(self.source, int):
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                if RESIZE_WIDTH:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
            except Exception as e:
                log_msg(f"設定攝影機參數失敗: {e}", "WARN")

        src_label = (f"Webcam (index={self.source})" if isinstance(self.source, int)
                     else f"ESP32-CAM ({self.source})")
        log_msg(f"{src_label} 已啟動", "OK")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                log_msg("影像讀取失敗，重試中...", "WARN")
                time.sleep(0.5)
                continue

            # 統一縮放（URL source 無法靠硬體設定，一律 resize）
            if RESIZE_WIDTH:
                h, w = frame.shape[:2]
                if w != RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / float(w)
                    frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)

            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass

            self.frame_count += 1

        self.cap.release()
        log_msg("攝影機線程已停止", "OK")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================
# 幀發送線程（Node-RED）
# ==============================
class FrameSender(threading.Thread):
    def __init__(self, frame_queue, node_red_url=NODE_RED_URL):
        super().__init__(daemon=True)
        self.frame_queue   = frame_queue
        self.node_red_url  = node_red_url
        self.running       = True
        self.ok_count      = 0
        self.fail_count    = 0
        self.latency_hist  = deque(maxlen=30)
        self.last_print    = 0.0
        self.connection_ok = False

    def run(self):
        log_msg("Node-RED 推送線程已啟動", "OK")
        interval  = 1.0 / max(1, TARGET_FPS)
        last_send = 0.0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            now = time.time()
            if now - last_send < interval:
                time.sleep(0.001)
                continue
            last_send = now

            t0 = time.time()
            ok, jpg = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                self.fail_count += 1
                continue

            try:
                r = requests.post(
                    self.node_red_url,
                    data=jpg.tobytes(),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=REQUEST_TIMEOUT,
                )
                self.latency_hist.append((time.time() - t0) * 1000)
                if r.status_code == 200:
                    self.ok_count += 1
                    self.connection_ok = True
                else:
                    self.fail_count += 1
            except requests.exceptions.ConnectionError:
                self.fail_count += 1
                self.connection_ok = False
                log_msg(f"無法連接 Node-RED ({self.node_red_url})", "WARN")
            except Exception:
                self.fail_count += 1
                self.connection_ok = False

            now = time.time()
            if now - self.last_print >= PRINT_EVERY_SEC:
                self.last_print = now
                avg_lat = (sum(self.latency_hist) / len(self.latency_hist)
                           if self.latency_hist else 0.0)
                fps    = self.ok_count / PRINT_EVERY_SEC
                status = "🟢 連接" if self.connection_ok else "🔴 斷開"
                log_msg(
                    f"{status} | OK={self.ok_count} Fail={self.fail_count} "
                    f"FPS={fps:.1f} Latency={avg_lat:.1f}ms",
                    "STAT",
                )
                self.ok_count = self.fail_count = 0

        log_msg("Node-RED 推送線程已停止", "OK")

    def stop(self):
        self.running = False


# ==============================
# 錄影控制（執行緒安全）
# ==============================
class Recorder:
    def __init__(self, fps: int, size: tuple):
        self.fps       = fps
        self.size      = size   # (width, height)
        self.recording = False
        self.writer    = None
        self.start_t   = 0.0
        self._lock     = threading.Lock()

    def toggle(self):
        with self._lock:
            if not self.recording:
                ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"record_{ts}.mp4"
                fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer    = cv2.VideoWriter(filename, fourcc, self.fps, self.size)
                self.recording = True
                self.start_t   = time.time()
                log_msg(f"開始錄影：{filename}", "OK")
            else:
                self.recording = False
                self.writer.release()
                self.writer = None
                log_msg("錄影已停止並保存", "OK")

    def write(self, frame):
        with self._lock:
            if self.recording and self.writer is not None:
                self.writer.write(frame)

    def elapsed(self) -> float:
        return time.time() - self.start_t if self.recording else 0.0

    def release(self):
        with self._lock:
            if self.writer:
                self.writer.release()
                self.writer = None


# ==============================
# 主程式
# ==============================
def main():
    local_ip  = get_local_ip()
    src_label = (f"Webcam index={SOURCE}" if isinstance(SOURCE, int)
                 else f"ESP32-CAM {SOURCE}")

    print("\n" + "=" * 60)
    print(f"🎥 攝影機伺服器  [{src_label}]")
    print("=" * 60)
    log_msg(f"本機 IP: {local_ip}", "INFO")
    log_msg(f"Node-RED: {'啟用' if ENABLE_NODE_RED else '停用'} → {NODE_RED_URL}", "INFO")
    log_msg(f"目標 FPS: {TARGET_FPS}  JPEG 品質: {JPEG_QUALITY}%", "INFO")
    log_msg("操作：[左上角按鈕 / R] 切換錄影  |  [Q] 結束", "INFO")
    print("=" * 60 + "\n")

    # ── 啟動攝影機線程 ──────────────────────────────────────────
    cap_queue     = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    camera_thread = CameraCapture(source=SOURCE, frame_queue=cap_queue)
    camera_thread.start()
    time.sleep(1.0)

    # ── 啟動 Node-RED 推送線程 ──────────────────────────────────
    nr_queue      = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    sender_thread = None
    if ENABLE_NODE_RED:
        sender_thread = FrameSender(frame_queue=nr_queue, node_red_url=NODE_RED_URL)
        sender_thread.start()

    # ── 取得第一幀以確定畫面尺寸 ────────────────────────────────
    pending = None
    for _ in range(30):
        try:
            pending = cap_queue.get(timeout=1.0)
            break
        except queue.Empty:
            pass

    h, w     = (pending.shape[:2] if pending is not None else (480, 640))
    recorder = Recorder(fps=TARGET_FPS, size=(w, h))

    # ── 預覽視窗與滑鼠回呼 ──────────────────────────────────────
    if SHOW_PREVIEW:
        cv2.namedWindow("Camera")

        def _mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 110 and 10 <= y <= 50:
                recorder.toggle()

        cv2.setMouseCallback("Camera", _mouse)

    frame_interval = 1.0 / max(1, TARGET_FPS)
    last_display   = 0.0

    # ── 主迴圈 ──────────────────────────────────────────────────
    try:
        while camera_thread.is_alive():
            # 取幀（優先消費第一幀，之後從 queue 取）
            if pending is not None:
                frame, pending = pending, None
            else:
                try:
                    frame = cap_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

            # 錄影
            recorder.write(frame)

            # 推送到 Node-RED
            if ENABLE_NODE_RED and sender_thread is not None:
                try:
                    nr_queue.put(frame, block=False)
                except queue.Full:
                    try:
                        nr_queue.get_nowait()
                        nr_queue.put(frame, block=False)
                    except queue.Empty:
                        pass

            # 預覽顯示（依 TARGET_FPS 限速）
            if SHOW_PREVIEW:
                now = time.time()
                if now - last_display >= frame_interval:
                    last_display = now
                    disp = frame.copy()

                    if recorder.recording:
                        cv2.rectangle(disp, (10, 10), (110, 50), (0, 0, 255), -1)
                        cv2.putText(disp, "Recording", (15, 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        t  = recorder.elapsed()
                        ts = f"{int(t // 60):02d}:{int(t % 60):02d}"
                        cv2.putText(disp, ts, (w - 100, 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(disp, (10, 10), (110, 50), (0, 200, 0), -1)
                        cv2.putText(disp, "Record", (22, 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    cv2.imshow("Camera", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('r'):
                        recorder.toggle()

    except KeyboardInterrupt:
        log_msg("收到中斷信號，正在關閉...", "WARN")
    finally:
        recorder.release()
        camera_thread.stop()
        if sender_thread:
            sender_thread.stop()
        camera_thread.join(timeout=2)
        if sender_thread:
            sender_thread.join(timeout=2)
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        log_msg("程式已終止", "OK")
        print("=" * 60)


if __name__ == "__main__":
    main()
