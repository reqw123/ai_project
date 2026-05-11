import cv2
import datetime
import time

# === 1. 設定 ESP32-CAM 影像串流 ===
url = "http://192.168.4.1:81/stream"
cap = cv2.VideoCapture(url)

# === 2. 預設影像參數 ===
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
fps = 20

# === 3. 初始化變數 ===
recording = False
out = None
start_time = 0
elapsed_time = 0

# === 4. 錄影檔名產生函式 ===
def new_filename():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"record_{now}.mp4"

# === 5. 滑鼠事件：按下按鈕切換錄影狀態 ===
def mouse_click(event, x, y, flags, param):
    global recording, out, start_time, elapsed_time
    if event == cv2.EVENT_LBUTTONDOWN:
        # 按下「錄影按鈕」的區域
        if 10 <= x <= 110 and 10 <= y <= 50:
            if not recording:
                filename = new_filename()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                recording = True
                start_time = time.time()
                print(f"開始錄影：{filename}")
            else:
                recording = False
                out.release()
                out = None
                print("錄影已停止並保存。")

cv2.namedWindow("ESP32-CAM")
cv2.setMouseCallback("ESP32-CAM", mouse_click)

print("滑鼠點擊左上角按鈕可切換錄影狀態。按下 Q 可結束程式。")

# === 6. 主迴圈 ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像。")
        break

    # 若正在錄影，寫入影片
    if recording and out is not None:
        out.write(frame)
        elapsed_time = time.time() - start_time

    # === 畫出錄影按鈕 ===
    if recording:
        cv2.rectangle(frame, (10, 10), (110, 50), (0, 0, 255), -1)  # 紅色
        cv2.putText(frame, "Recording", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 顯示錄影時間
        timer_text = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        cv2.putText(frame, timer_text, (width - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.rectangle(frame, (10, 10), (110, 50), (0, 255, 0), -1)  # 綠色
        cv2.putText(frame, "Record", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 顯示影像
    cv2.imshow("ESP32-CAM", frame)

    # 按 Q 離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 7. 結束清理 ===
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print("程式結束。")
