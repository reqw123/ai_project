import cv2
import time
import requests
import socket
import threading
import queue
from collections import deque
from datetime import datetime

# ==============================
# 基本設定
# ==============================
NODE_RED_URL = "http://localhost:1880/camera/frame"
CAMERA_INDEX = 0
TARGET_FPS = 15
JPEG_QUALITY = 60
RESIZE_WIDTH = 640
SHOW_PREVIEW = False
REQUEST_TIMEOUT = 0.5
RETRY_SLEEP = 0.1
PRINT_EVERY_SEC = 2.0
FRAME_QUEUE_SIZE = 3

# ==============================
# 工具函數
# ==============================
def get_local_ip() -> str:
    """取得本機IP"""
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
    """帶時間戳的日誌"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    level_icons = {
        "INFO": "ℹ️",
        "OK": "✅",
        "ERROR": "❌",
        "WARN": "⚠️",
        "STAT": "📊"
    }
    icon = level_icons.get(level, "•")
    print(f"[{timestamp}] {icon} {msg}")


# ==============================
# 攝影機擷取線程
# ==============================
class CameraCapture(threading.Thread):
    """獨立線程負責攝影機擷取"""
    
    def __init__(self, camera_index=0, frame_queue=None):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.frame_queue = frame_queue or queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.running = True
        self.cap = None
        self.frame_count = 0
        
    def run(self):
        """執行攝影機擷取"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            log_msg(f"無法開啟攝影機 (index={self.camera_index})", "ERROR")
            self.running = False
            return
        
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESIZE_WIDTH)
        except Exception as e:
            log_msg(f"設定攝影機參數失敗: {e}", "WARN")
        
        log_msg("攝影機已啟動", "OK")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                log_msg("攝影機讀取失敗，重試中...", "WARN")
                time.sleep(0.5)
                continue
            
            # 調整大小
            if RESIZE_WIDTH and RESIZE_WIDTH > 0:
                h, w = frame.shape[:2]
                if w != RESIZE_WIDTH:
                    scale = RESIZE_WIDTH / float(w)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (RESIZE_WIDTH, new_h), 
                                      interpolation=cv2.INTER_AREA)
            
            if SHOW_PREVIEW:
                cv2.imshow("Camera Preview (Press Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            
            # 放入隊列
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
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        log_msg("攝影機線程已停止", "OK")
    
    def stop(self):
        """停止擷取"""
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================
# 幀發送線程
# ==============================
class FrameSender(threading.Thread):
    """獨立線程負責發送幀到Node-RED"""
    
    def __init__(self, frame_queue=None, node_red_url=NODE_RED_URL):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.node_red_url = node_red_url
        self.running = True
        self.ok_count = 0
        self.fail_count = 0
        self.latency_hist = deque(maxlen=30)
        self.last_print_time = 0.0
        self.connection_ok = False
        
    def run(self):
        """執行發送"""
        log_msg("幀發送線程已啟動", "OK")
        frame_interval = 1.0 / max(1, TARGET_FPS)
        last_send_time = 0.0
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            
            now = time.time()
            if now - last_send_time < frame_interval:
                time.sleep(0.001)
                continue
            
            last_send_time = now
            t_send = time.time()
            
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
            ok, jpg_bytes = cv2.imencode(".jpg", frame, encode_params)
            
            if not ok:
                self.fail_count += 1
                continue
            
            try:
                response = requests.post(
                    self.node_red_url,
                    data=jpg_bytes.tobytes(),
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=REQUEST_TIMEOUT
                )
                
                latency = (time.time() - t_send) * 1000
                self.latency_hist.append(latency)
                
                if response.status_code == 200:
                    self.ok_count += 1
                    self.connection_ok = True
                else:
                    self.fail_count += 1
                    
            except requests.exceptions.Timeout:
                self.fail_count += 1
                self.connection_ok = False
            except requests.exceptions.ConnectionError:
                self.fail_count += 1
                self.connection_ok = False
                log_msg(f"無法連接 Node-RED ({self.node_red_url})", "WARN")
            except Exception as e:
                self.fail_count += 1
                self.connection_ok = False
            
            now = time.time()
            if now - self.last_print_time >= PRINT_EVERY_SEC:
                self.last_print_time = now
                avg_latency = (sum(self.latency_hist) / len(self.latency_hist) 
                              if self.latency_hist else 0.0)
                fps = self.ok_count / PRINT_EVERY_SEC if self.ok_count > 0 else 0
                status = "🟢 連接" if self.connection_ok else "🔴 斷開"
                
                log_msg(
                    f"{status} | OK={self.ok_count} Fail={self.fail_count} "
                    f"FPS={fps:.1f} Latency={avg_latency:.1f}ms",
                    "STAT"
                )
                self.ok_count = 0
                self.fail_count = 0
        
        log_msg("幀發送線程已停止", "OK")
    
    def stop(self):
        """停止發送"""
        self.running = False


# ==============================
# 主程式
# ==============================
def main():
    """主程式"""
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🎥 攝影機推播系統（Python端推送攝影機畫面）")
    print("="*60)
    log_msg(f"本機 IP: {local_ip}", "INFO")
    log_msg(f"Node-RED URL: {NODE_RED_URL}", "INFO")
    log_msg(f"攝影機索引: {CAMERA_INDEX}", "INFO")
    log_msg(f"目標 FPS: {TARGET_FPS}", "INFO")
    log_msg(f"JPEG 品質: {JPEG_QUALITY}%", "INFO")
    log_msg(f"縮小寬度: {RESIZE_WIDTH if RESIZE_WIDTH else '不縮小'}", "INFO")
    print("="*60 + "\n")
    
    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    
    camera_thread = CameraCapture(
        camera_index=CAMERA_INDEX,
        frame_queue=frame_queue
    )
    camera_thread.start()
    
    time.sleep(1)
    
    sender_thread = FrameSender(
        frame_queue=frame_queue,
        node_red_url=NODE_RED_URL
    )
    sender_thread.start()
    
    try:
        while camera_thread.is_alive() and sender_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        log_msg("收到中斷信號，正在關閉...", "WARN")
    
    camera_thread.stop()
    sender_thread.stop()
    
    camera_thread.join(timeout=2)
    sender_thread.join(timeout=2)
    
    log_msg("程式已終止", "OK")
    print("="*60)


if __name__ == "__main__":
    main()