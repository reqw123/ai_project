from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

import cv2
import base64
import threading
import time

# =====================================================
# Flask + SocketIO
# =====================================================

app = Flask(__name__)

socketio = SocketIO(

    app,

    cors_allowed_origins="*",

    async_mode="gevent",

    ping_timeout=10,
    ping_interval=5
)

# =====================================================
# 低延遲參數
# =====================================================

CAMERA_SOURCE = 0

WIDTH = 640
HEIGHT = 480

JPEG_QUALITY = 45

TARGET_FPS = 30

# =====================================================
# 全域最新 frame
# =====================================================

latest_frame = None

frame_lock = threading.Lock()
cap_lock = threading.Lock()  # 保護 cap 的 grab/retrieve 存取

# =====================================================
# OpenCV Camera
# =====================================================

cap = cv2.VideoCapture(

    CAMERA_SOURCE,

    cv2.CAP_DSHOW
)

# 超重要：降低 buffer

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# =====================================================
# 背景攝影機讀取
# 永遠只保留最新 frame
# =====================================================

def camera_reader():

    global latest_frame

    while True:

        with cap_lock:
            cap.grab()
            success, frame = cap.retrieve()

        if not success:

            time.sleep(0.005)

            continue

        frame = cv2.resize(

            frame,

            (WIDTH, HEIGHT)
        )

        # =================================================
        # Overlay
        # =================================================

        cv2.putText(

            frame,

            "AI Cat Monitor",

            (15, 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,255,0),

            2
        )

        cv2.putText(

            frame,

            time.strftime("%Y-%m-%d %H:%M:%S"),

            (15, 70),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255,255,255),

            2
        )

        # =================================================
        # 存最新 frame
        # =================================================

        with frame_lock:

            latest_frame = frame.copy()

        time.sleep(0.001)

# =====================================================
# HTML
# =====================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>AI Ultra Low Latency Stream</title>

<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>

<style>

body{

    background:#111;

    text-align:center;

    color:white;

    font-family:Arial;
}

canvas{

    width:90%;

    max-width:900px;

    border-radius:14px;

    background:black;

    margin-top:20px;
}

.info{

    color:#aaa;

    margin-top:10px;
}

</style>

</head>

<body>

<h2>AI Cat Monitor - Ultra Low Latency</h2>

<canvas id="canvas"></canvas>

<div class="info">

WebSocket Sync Mode

</div>

<script>

const socket = io({

    transports:["websocket"],

    upgrade:false
});

const canvas = document.getElementById("canvas");

const ctx = canvas.getContext("2d");

let busy = false;

function requestFrame(){

    if(busy) return;

    busy = true;

    socket.emit("request_frame");
}

socket.on("frame", function(data){

    const img = new Image();

    img.onload = function(){

        canvas.width = img.width;

        canvas.height = img.height;

        ctx.drawImage(img,0,0);

        busy = false;

        requestAnimationFrame(requestFrame);
    };

    img.onerror = function(){

        busy = false;

        setTimeout(requestFrame,30);
    };

    img.src = "data:image/jpeg;base64," + data;
});

socket.on("connect", function(){

    console.log("Connected");

    requestFrame();
});

socket.on("disconnect", function(){

    console.log("Disconnected");

    busy = false;
});

</script>

</body>

</html>

"""

# =====================================================
# 首頁
# =====================================================

@app.route("/")

def index():

    return render_template_string(HTML)

# =====================================================
# Client 請求 frame
# =====================================================

@socketio.on("request_frame")

def handle_request_frame():

    global latest_frame

    with frame_lock:

        if latest_frame is None:

            return

        frame = latest_frame.copy()

    # =================================================
    # JPEG encode
    # =================================================

    ret, buffer = cv2.imencode(

        ".jpg",

        frame,

        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    if not ret:

        return

    jpg_base64 = base64.b64encode(

        buffer

    ).decode("utf-8")

    emit("frame", jpg_base64)

# =====================================================
# 啟動
# =====================================================

if __name__ == "__main__":

    camera_thread = threading.Thread(

        target=camera_reader,

        daemon=True
    )

    camera_thread.start()

    socketio.run(

        app,

        host="0.0.0.0",

        port=5000,

        debug=False
    )