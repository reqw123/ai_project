"""
主入口點

以 config.py 的 RunModeConfig.MODE（環境變數 CAT_MONITORING_RUN_MODE）切換：
  - "server"（預設）：Flask HTTP 伺服器 + Node-RED 上線通知（原本行為，不變）
  - "gui"           ：不啟動 Flask/Node-RED，直接用同一套 FrameProcessor 開本地視窗顯示
兩種模式共用 server/routes.py 的 _build_frame_processor() 等既有處理管線，
不重新設計架構，只是換一種「前端」呈現方式。
"""
import os
import threading
import time
import requests
import cv2

# 開發環境 workaround：避免 Windows 下 OpenMP runtime 重複載入導致程序中止
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from server.flask_app import create_app
from utils.helpers import get_ip
from config import FlaskConfig, NodeRedConfig, RunModeConfig

def send_ip_to_nodered(ip, node_red_url):
    """定期發送 Python IP 給 Node-RED，直到成功為止"""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(
                node_red_url,
                json={"ip": ip},
                timeout=NodeRedConfig.TIMEOUT
            )
            if response.status_code == 200:
                
                print(f"✅ 成功通知 Node-RED，Python IP: {ip}")
                break
            else:
                print(f"⚠ Node-RED 回應異常: {response.status_code}")
        except Exception as e:
            print(f"⚠ 無法連接 Node-RED (嘗試 {retry_count + 1}/{max_retries}): {e}")
        
        retry_count += 1
        time.sleep(3)  # 每 3 秒重試一次
    
    if retry_count >= max_retries:
        print("❌ 無法連接到 Node-RED，請檢查 Node-RED 是否啟動")

def run_server_mode():
    """HTTP 伺服器模式（原本 main.py 的行為，未更動）：Flask + Node-RED 上線通知。"""
    if FlaskConfig.DEBUG:
        import warnings
        warnings.warn(
            "Flask DEBUG=True（Werkzeug interactive debugger 開啟，LAN 環境下任何人都能執行任意程式碼）。"
            "生產環境請確認環境變數 CAT_MONITORING_FLASK_DEBUG 未設為 true。",
            RuntimeWarning, stacklevel=1,
        )

    app = create_app()
    ip = get_ip()
    if not ip:
        ip = "127.0.0.1"
    print(f"\n📺 Web 服務器啟動於 http://{ip}:{FlaskConfig.PORT}")
    print(f"📊 串流網址: http://{ip}:{FlaskConfig.PORT}/stream")

    node_red_url = NodeRedConfig.ENDPOINT_NOTIFY
    if ip and ip != "127.0.0.1":
        threading.Thread(
            target=send_ip_to_nodered,
            args=(ip, node_red_url),
            daemon=True
        ).start()
    else:
        print("⚠ 無法取得有效 IP，跳過 Node-RED 上線通知")

    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        threaded=FlaskConfig.THREADED,
        debug=FlaskConfig.DEBUG,
    )


GUI_MAX_WIDTH = 1280
GUI_MAX_HEIGHT = 720


def _resize_for_gui(frame):
    """把 GUI 視窗畫面等比例縮小到最大 720p，避免高解析度來源把視窗撐爆螢幕；
    畫面本身已小於 720p 時不放大，維持原尺寸。"""
    h, w = frame.shape[:2]
    scale = min(GUI_MAX_WIDTH / w, GUI_MAX_HEIGHT / h, 1.0)
    if scale >= 1.0:
        return frame
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_gui_mode():
    """本地 GUI 模式：不啟動 Flask/Node-RED，直接用同一套 FrameProcessor 開本地視窗顯示。

    重用 server/routes.py 的 _build_frame_processor() 與 plugin 註冊邏輯，
    確保與 HTTP 伺服器模式吃到完全相同的模型路徑/參數設定，不另外維護一份。
    """
    from server.routes import (
        _build_frame_processor,
        _try_register_lick_stage,
        _try_register_ext_body_zone,
    )

    processor = _build_frame_processor(enable_nodered=False)
    _try_register_lick_stage(processor)
    _try_register_ext_body_zone(processor)

    # OpenCV 在 Windows 上的視窗標題（cv2.namedWindow）不支援中文，非 ASCII
    # 字元會顯示成亂碼視窗標題，因此這裡固定用英文。
    window_name = "Cat Monitoring (Local GUI)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # WINDOW_NORMAL 預設視窗尺寸很小，需明確指定大小；畫面內容本身維持
    # 來源原解析度（_resize_for_gui 只在超過 GUI_MAX_WIDTH/HEIGHT 時才縮小）。
    cv2.resizeWindow(window_name, GUI_MAX_WIDTH, GUI_MAX_HEIGHT)
    print("\n🖥️ 本地 GUI 模式啟動（未啟動 HTTP 伺服器，也不會推送 Node-RED）")
    print("按鍵：q 離開　|　space 播放/暫停　|　暫停時 a/d 前一幀/後一幀　|　z/x 調整跳幀步長")
    print("     s 骨架顯示　|　l 標籤顯示　|　b bbox 顯示\n")

    paused = False
    frame_step_size = 1   # a/d 單次跳幀幀數，z/x 調整
    last_frame = None     # 暫停時重複顯示用；一開始尚未讀過畫面時為 None

    try:
        while True:
            if not paused or last_frame is None:
                ret, frame = processor.read_raw_frame()
                if not ret:
                    time.sleep(0.01)
                    continue
                last_frame, *_ = processor.process(frame)

            cv2.imshow(window_name, _resize_for_gui(last_frame))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("使用者中斷：q")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸ 暫停' if paused else '▶ 播放'}")
            elif key == ord('z'):
                frame_step_size = max(1, frame_step_size - 1)
                print(f"跳幀步長: {frame_step_size}")
            elif key == ord('x'):
                frame_step_size += 1
                print(f"跳幀步長: {frame_step_size}")
            elif paused and key in (ord('a'), ord('d')):
                # 暫停時直接操作底層 VideoCapture 位置做逐幀/跳幀瀏覽；
                # 注意：這仍會呼叫 processor.process()，非循序讀取會讓 ST-GCN
                # 時序 buffer／CSV 記錄／異常偵測滾動視窗吃到不連續的幀，
                # 屬於本除錯功能的預期取捨，不影響一般播放模式下的正確性。
                total = int(processor.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_pos = int(processor.cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_displayed = max(0, current_pos - 1)
                delta = -frame_step_size if key == ord('a') else frame_step_size
                target = current_displayed + delta
                target = max(0, target)
                if total > 0:
                    target = min(target, total - 1)
                processor.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = processor.cap.read()
                if ret:
                    last_frame, *_ = processor.process(frame)
            elif key == ord('s'):
                processor.show_skeleton = not processor.show_skeleton
            elif key == ord('l'):
                processor.show_label = not processor.show_label
            elif key == ord('b'):
                processor.show_bbox = not processor.show_bbox
    finally:
        processor.cleanup()


if __name__ == "__main__":
    if RunModeConfig.MODE == "gui":
        run_gui_mode()
    else:
        run_server_mode()
