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
import numpy as np
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import ModelPaths as _ModelPaths
from config import YOLOConfig as _YOLOConfig
from config import STGCNConfig as _STGCNConfig
from config import NodeRedConfig as _NodeRedConfig
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig
from config import FlaskConfig as _FlaskConfig

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
tracker = ImprovedBehaviorTracker()
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
        width=640,
        height=640,
        normalize=True,
        kp_ema_alpha=_KP_EMA_ALPHA,
    )


def _get_latest_behavior():
    """從 frame_processor 取得最新行為推論，供首頁與 status API 使用。回傳皆為 JSON 可序列化的原生型別。"""
    latest_behavior, latest_confidence = 0, 0.0
    latest_probs = [0.0, 0.0, 0.0, 0.0, 0.0]
    if frame_processor and hasattr(frame_processor, 'behavior_classifier'):
        try:
            if hasattr(frame_processor, 'keypoints_buffer') and len(frame_processor.keypoints_buffer) >= frame_processor.sequence_length:
                from models.stgcn_model import interpolate_missing
                kpts_arr = np.array([item[0] for item in frame_processor.keypoints_buffer])  # (T, 17, 2)
                conf_arr = np.array([item[1] for item in frame_processor.keypoints_buffer])  # (T, 17)
                seq_array = interpolate_missing(kpts_arr, conf_arr)
                # Build feature tensor when model expects >4 channels
                model_obj = getattr(frame_processor.behavior_classifier, 'model', None)
                if model_obj is not None and getattr(model_obj, 'in_channels', 4) != 4:
                    try:
                        from models.stgcn_model import flip_normalize, orientation_normalize, normalize_skeleton_coords, build_feature_tensor
                        if getattr(model_obj, 'normalize', True):
                            seq_array = flip_normalize(seq_array)
                            seq_array = orientation_normalize(seq_array)
                            seq_array = normalize_skeleton_coords(seq_array)
                        seq_features = build_feature_tensor(seq_array, conf_arr, model_obj.feature_mode)
                        b, c, probs = frame_processor.behavior_classifier.classify(seq_features, precomputed=True)
                    except Exception:
                        b, c, probs = frame_processor.behavior_classifier.classify(seq_array)
                else:
                    b, c, probs = frame_processor.behavior_classifier.classify(seq_array)
                if b is not None:
                    latest_confidence = float(c)
                    latest_probs = [float(p) for p in (probs if probs is not None else latest_probs)]
                    latest_behavior = LOW_CONF_ID if latest_confidence < _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD else int(b)
        except Exception:
            pass
    return latest_behavior, latest_confidence, latest_probs


def _ensure_processor_started():
    """在首次請求時啟動處理管線（double-checked locking，避免多執行緒重複建立）。"""
    global frame_streamer, frame_processor
    if frame_processor is not None and frame_streamer is not None:
        return
    with _init_lock:
        if frame_processor is None:
            frame_processor = _build_frame_processor()
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
            try:
                while True:
                    jpeg = frame_streamer.get_jpeg()
                    if jpeg is not None:
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                    else:
                        time.sleep(0.01)
            except GeneratorExit:
                pass  # 客戶端正常斷線，允許 generator 終止
        return Response(mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/snapshot')
    def snapshot():
        _ensure_processor_started()
        jpeg = frame_streamer.get_jpeg() if frame_streamer else None
        if jpeg is None:
            return Response(b'', status=503, mimetype='image/jpeg')
        return Response(jpeg, mimetype='image/jpeg')

    @app.route('/video_clip')
    def video_clip():
        _ensure_processor_started()
        frames = frame_streamer.get_clip_frames() if frame_streamer else []
        if not frames:
            return jsonify({'error': 'no frames available'}), 503

        ts_obj = datetime.datetime.now()
        ts_file = ts_obj.strftime('%Y%m%d_%H%M%S')
        ts_display = ts_obj.strftime('%Y/%m/%d %H:%M:%S')

        save_dir = Path('C:/a')
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

    @app.route('/status')
    def status():
        _ensure_processor_started()
        # 使用與 Node-RED 相同的 tracker（frame_processor.tracker），行為資料才會一致
        t = frame_processor.tracker if frame_processor else tracker
        stats = t.get_today_stats()
        lb, lc, lprobs = _get_latest_behavior()
        return jsonify({
            "status": "running",
            "port": _PORT,
            "ip": LOCAL_IP,
            "activity_score": t.get_activity_score(),
            "today_stats": stats,
            "alerts": t.get_alerts(),
            "alerts_count": len(t.get_alerts()),
            "version": "v4.0-stgcn",
            "latest_behavior": lb,
            "latest_confidence": lc,
            "latest_probs": lprobs,
        })

    @app.route('/api/behavior_history')
    def api_behavior_history():
        """回傳各行為區段與持續時間，供行為趨勢分析使用。支援 ?limit=200。"""
        _ensure_processor_started()
        try:
            limit = max(1, min(int(request.args.get('limit', 200)), 1000))
        except (TypeError, ValueError):
            limit = 200
        t = frame_processor.tracker if frame_processor else tracker
        history = list(t.behavior_history)[-limit:]
        segments = []
        for rec in reversed(history):
            segments.append({
                "behavior_id": int(rec["gcn_behavior_id"]),
                "behavior": BEHAVIOR_TEXT_MAP.get(rec["gcn_behavior_id"], rec["behavior"]),
                "timestamp": rec["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": round(float(rec["duration"]), 1),
                "activity": int(rec.get("activity", 0)),
            })
        return jsonify({
            "count": len(segments),
            "segments": segments,
        })

    @app.route('/')
    def index():
        _ensure_processor_started()
        return Response(
            f"<html><body><p>Cat Monitoring System v4.0-stgcn &mdash; {LOCAL_IP}:5000</p>"
            f"<ul><li><a href='/stream'>stream</a></li>"
            f"<li><a href='/status'>status API</a></li>"
            f"<li><a href='/api/behavior_history?limit=500'>behavior history</a></li></ul>"
            f"</body></html>",
            mimetype='text/html'
        )
    