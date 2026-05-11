from flask import Flask, Response
from djitellopy import Tello
import cv2
import time

app = Flask(__name__)

# ========================
# 連接 Tello
# ========================

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

tello.streamon()

frame_read = tello.get_frame_read()

# ========================
# 影像產生器
# ========================

def generate():

    while True:

        frame = frame_read.frame

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # JPEG壓縮
        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ========================
# Flask Route
# ========================

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ========================
# 啟動伺服器
# ========================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',   # 允許區網存取
        port=5000,
        debug=False,
        threaded=True
    )