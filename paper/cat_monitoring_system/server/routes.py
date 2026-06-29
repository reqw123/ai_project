"""
Flask 路由
"""
from flask import Response, jsonify, request
import ipaddress
import time
import sys
import os
import datetime
import cv2
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import ModelPaths as _ModelPaths
from config import YOLOConfig as _YOLOConfig
from config import STGCNConfig as _STGCNConfig
from config import NodeRedConfig as _NodeRedConfig
from config import FlaskConfig as _FlaskConfig
from config import SystemInfo as _SystemInfo

_KP_EMA_ALPHA = _STGCNConfig.KP_EMA_ALPHA
_YOLO_MODEL_PATH = _ModelPaths.YOLO_MODEL
_STGCN_MODEL_PATH = _ModelPaths.STGCN_MODEL
_VIDEO_PATH = _ModelPaths.VIDEO_INPUT
_NODERED_RESULT_URL = _NodeRedConfig.ENDPOINT_RESULT
_IMAGE_SIZE = _YOLOConfig.IMAGE_SIZE
_CONF_THRES = _YOLOConfig.CONFIDENCE_THRESHOLD
_SEQUENCE_LENGTH = _STGCNConfig.SEQUENCE_LENGTH
_PORT = _FlaskConfig.PORT
from server.streaming import SharedFrameStreamer
from processors.frame_processor import FrameProcessor
from trackers.behavior_tracker import ImprovedBehaviorTracker
from utils.constants import *
from utils.helpers import get_ip


frame_streamer = None
frame_processor = None
_init_lock = __import__('threading').Lock()
LOCAL_IP = get_ip() or "127.0.0.1"


def _resolve_runtime_device(preferred='cuda'):
    if preferred != 'cuda':
        return preferred
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def _build_frame_processor():
    runtime_device = _resolve_runtime_device('cuda')
    return FrameProcessor(
        yolo_model_path=_YOLO_MODEL_PATH,
        stgcn_model_path=_STGCN_MODEL_PATH,
        video_path=_VIDEO_PATH,
        nodered_url=_NODERED_RESULT_URL,
        device=runtime_device,
        imgsz=_IMAGE_SIZE,
        conf_thres=_CONF_THRES,
        sequence_length=_SEQUENCE_LENGTH,
        overlay=True,
        width=_SystemInfo.OUTPUT_WIDTH,
        height=_SystemInfo.OUTPUT_HEIGHT,
        normalize=True,
        kp_ema_alpha=_KP_EMA_ALPHA,
    )




def _try_register_lick_stage(processor) -> None:
    """Optionally attach the Lick Stage plugin. Silently skipped if plugin is absent."""
    try:
        from plugins.lick_stage import LickStagePlugin as _LickStagePlugin
        processor.register_plugin(_LickStagePlugin())
    except ImportError:
        pass


def _ensure_processor_started():
    """在首次請求時啟動處理管線（double-checked locking，避免多執行緒重複建立）。"""
    global frame_streamer, frame_processor
    if frame_processor is not None and frame_streamer is not None:
        return
    with _init_lock:
        if frame_processor is None:
            frame_processor = _build_frame_processor()
            _try_register_lick_stage(frame_processor)
        if frame_streamer is None:
            frame_streamer = SharedFrameStreamer(frame_processor)


def register_routes(app):
    @app.route('/python_online', methods=['POST'])
    def python_online():
        data = request.get_json(force=True) or {}
        ip = data.get('ip', '')
        try:
            ipaddress.ip_address(str(ip))
        except ValueError:
            ip = ''
        print(f"[Node-RED] Python 上線通知，收到 IP: {ip}")
        return jsonify({'ip': ip})

    @app.route('/stream')
    def stream():
        _ensure_processor_started()
        def mjpeg_stream():
            frame_streamer.acquire_client()
            try:
                while True:
                    jpeg = frame_streamer.get_jpeg()
                    if jpeg is not None:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                    else:
                        time.sleep(0.01)
            finally:
                # GeneratorExit（客戶端斷線）或任何例外都能正確釋放計數
                frame_streamer.release_client()
        return Response(mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/snapshot')
    def snapshot():
        _ensure_processor_started()
        if frame_streamer is None:
            return Response(b'', status=503, mimetype='image/jpeg')
        # 暫時佔用一個 client slot，確保 JPEG 編碼執行緒會產生最新幀
        frame_streamer.acquire_client()
        try:
            deadline = time.time() + 0.5
            while time.time() < deadline:
                jpeg = frame_streamer.get_jpeg()
                if jpeg:
                    return Response(jpeg, mimetype='image/jpeg')
                time.sleep(0.02)
        finally:
            frame_streamer.release_client()
        return Response(b'', status=503, mimetype='image/jpeg')

    @app.route('/video_clip')
    def video_clip():
        _ensure_processor_started()
        frames = frame_streamer.get_clip_frames() if frame_streamer else []
        if not frames:
            return jsonify({'error': 'no frames available'}), 503

        ts_obj = datetime.datetime.now()
        ts_file = ts_obj.strftime('%Y%m%d_%H%M%S')
        ts_display = ts_obj.strftime('%Y/%m/%d %H:%M:%S')

        save_dir = Path(_ModelPaths.OUTPUT_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        fps = float(_STGCNConfig.TARGET_MODEL_FPS)

        # mp4v 在 Windows 無額外 codec 時 isOpened() 會為 False，fallback 到 MJPG+avi
        codecs = [
            (str(save_dir / f'clip_{ts_file}.mp4'), cv2.VideoWriter_fourcc(*'mp4v')),
            (str(save_dir / f'clip_{ts_file}.avi'), cv2.VideoWriter_fourcc(*'MJPG')),
        ]
        writer = None
        save_path = ''
        for path, fourcc in codecs:
            w_ = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if w_.isOpened():
                writer = w_
                save_path = path
                break
            w_.release()

        if writer is None:
            return jsonify({'error': 'no usable video codec on this machine'}), 500

        for f in frames:
            writer.write(f)
        writer.release()

        # 最後一幀轉 base64 供 Dashboard 顯示縮圖
        import base64
        last_frame = frames[-1]
        jpeg_quality = max(1, min(int(_FlaskConfig.JPEG_QUALITY), 100))
        _, buf = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        thumbnail = 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()

        duration = round(len(frames) / fps, 1)
        return jsonify({
            'path': save_path,
            'frames': len(frames),
            'duration': duration,
            'ts': ts_display,
            'thumbnail': thumbnail,
        })

    @app.route('/api/behavior_history')
    def api_behavior_history():
        """回傳各行為區段與持續時間，供行為趨勢分析使用。支援 ?limit=200。"""
        _ensure_processor_started()
        try:
            limit = max(1, min(int(request.args.get('limit', 200)), 1000))
        except (TypeError, ValueError):
            limit = 200
        records = frame_processor.get_behavior_history_records(limit)
        segments = [
            {
                "behavior_id": int(rec["gcn_behavior_id"]),
                "behavior": BEHAVIOR_TEXT_MAP.get(rec["gcn_behavior_id"], rec["behavior"]),
                "timestamp": rec["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": round(float(rec["duration"]), 1),
                "activity": int(rec.get("activity", 0)),
            }
            for rec in reversed(records)
        ]
        return jsonify({
            "count": len(segments),
            "segments": segments,
        })

    @app.route('/')
    def index():
        _ensure_processor_started()
        return Response(
            f"<html><body><p>{_SystemInfo.SYSTEM_NAME} {_SystemInfo.VERSION} &mdash; {LOCAL_IP}:{_PORT}</p>"
            f"<ul><li><a href='/stream'>stream</a></li>"
            f"<li><a href='/api/behavior_history?limit=500'>behavior history</a></li></ul>"
            f"</body></html>",
            mimetype='text/html'
        )
    