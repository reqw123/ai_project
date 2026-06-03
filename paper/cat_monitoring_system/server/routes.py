"""
Flask 路由
"""
from flask import Response, jsonify, request
import ipaddress
import json
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
from utils.helpers import get_ip, get_behavior_name


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
        _, buf = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
        lb, lc, lprobs = _get_latest_behavior()
        conf_pct = int(round(lc * 100))
        # 使用 json.dumps 確保資料安全嵌入 JS，避免 XSS
        behavior_names_js = json.dumps(BEHAVIOR_CLASSES)
        probs_json = json.dumps([round(float(p), 4) for p in lprobs])
        min_conf_js = json.dumps(float(_BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD))
        html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>貓咪健康監測 · ST-GCN</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #050508;
            --surface: #0d0d14;
            --surface2: #12121a;
            --cyan: #00f5ff;
            --cyan-dim: #00b8c4;
            --magenta: #ff00aa;
            --magenta-dim: #cc0088;
            --yellow: #ffed00;
            --yellow-dim: #c9b800;
            --text: #e8e8f0;
            --text-muted: #6b6b80;
            --border: rgba(0, 245, 255, 0.25);
            --glow-cyan: 0 0 20px rgba(0, 245, 255, 0.4);
            --glow-magenta: 0 0 20px rgba(255, 0, 170, 0.35);
            --font-title: 'Orbitron', sans-serif;
            --font-mono: 'Share Tech Mono', monospace;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: var(--font-mono);
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.5;
            position: relative;
            overflow-x: hidden;
        }}
        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background: 
                linear-gradient(180deg, transparent 0%, rgba(0,245,255,0.02) 50%, transparent 100%),
                repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px);
            pointer-events: none;
            z-index: 1;
        }}
        .layout {{ position: relative; z-index: 2; max-width: 1600px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 1fr 380px; gap: 24px; align-items: start; }}
        @media (max-width: 1200px) {{ .layout {{ grid-template-columns: 1fr; }} }}
        header {{
            grid-column: 1 / -1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            padding-bottom: 20px;
            border-bottom: 2px solid var(--border);
            box-shadow: 0 1px 0 rgba(0,245,255,0.1);
        }}
        .logo {{ display: flex; align-items: center; gap: 16px; }}
        .logo-icon {{
            width: 52px;
            height: 52px;
            background: linear-gradient(135deg, var(--cyan) 0%, var(--magenta) 100%);
            clip-path: polygon(10% 0%, 100% 0%, 100% 90%, 90% 100%, 0% 100%, 0% 10%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: var(--glow-cyan);
        }}
        .logo h1 {{
            font-family: var(--font-title);
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text);
            text-shadow: 0 0 30px rgba(0,245,255,0.3);
        }}
        .logo span {{ font-size: 0.75rem; color: var(--text-muted); letter-spacing: 0.15em; text-transform: uppercase; }}
        .badge {{
            font-family: var(--font-mono);
            padding: 6px 14px;
            border: 1px solid var(--cyan);
            color: var(--cyan);
            font-size: 0.7rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            box-shadow: var(--glow-cyan);
            background: rgba(0,245,255,0.08);
        }}
        .stream-wrap {{
            background: var(--surface);
            border: 2px solid var(--border);
            position: relative;
            overflow: hidden;
            box-shadow: var(--glow-cyan), inset 0 0 60px rgba(0,245,255,0.03);
        }}
        .stream-wrap::before, .stream-wrap::after {{
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid var(--cyan);
            z-index: 2;
            box-shadow: 0 0 10px var(--cyan);
        }}
        .stream-wrap::before {{ top: 0; left: 0; border-right: none; border-bottom: none; }}
        .stream-wrap::after {{ bottom: 0; right: 0; border-left: none; border-top: none; }}
        .stream-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 16px;
            background: linear-gradient(90deg, rgba(0,245,255,0.12) 0%, transparent 100%);
            border-bottom: 1px solid var(--border);
        }}
        .live-pill {{
            font-family: var(--font-title);
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            border: 1px solid var(--cyan);
            color: var(--cyan);
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.25em;
            text-transform: uppercase;
            box-shadow: var(--glow-cyan);
            background: rgba(0,245,255,0.1);
        }}
        .live-pill::before {{
            content: '';
            width: 6px;
            height: 6px;
            background: var(--cyan);
            animation: blink 0.8s ease-in-out infinite;
            box-shadow: 0 0 8px var(--cyan);
        }}
        @keyframes blink {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.2 }} }}
        .stream-header span {{ color: var(--text-muted); font-size: 0.8rem; letter-spacing: 0.1em; }}
        .stream-wrap img {{ display: block; width: 100%; max-width: 960px; height: auto; }}
        .panel {{
            background: var(--surface);
            border: 2px solid var(--border);
            padding: 20px;
            position: relative;
            box-shadow: 0 0 30px rgba(0,245,255,0.08), inset 0 0 40px rgba(0,0,0,0.3);
        }}
        .panel::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--cyan), var(--magenta), transparent);
            opacity: 0.6;
        }}
        .panel h2 {{
            font-family: var(--font-title);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--cyan);
            margin-bottom: 16px;
            text-shadow: 0 0 15px rgba(0,245,255,0.5);
        }}
        .behavior-now {{
            text-align: center;
            padding: 24px 16px;
            background: rgba(0,245,255,0.04);
            border: 1px solid rgba(0,245,255,0.2);
            margin-bottom: 20px;
            position: relative;
        }}
        .behavior-now::before {{
            content: '[ CURRENT STATE ]';
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.6rem;
            letter-spacing: 0.2em;
            color: var(--text-muted);
        }}
        .behavior-emoji {{ font-size: 3rem; margin-bottom: 8px; margin-top: 8px; }}
        .behavior-label {{
            font-family: var(--font-title);
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--cyan);
            letter-spacing: 0.1em;
            text-shadow: var(--glow-cyan);
        }}
        .behavior-conf {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; letter-spacing: 0.1em; }}
        .confidence-ring {{
            width: 80px;
            height: 80px;
            margin: 12px auto 0;
            border-radius: 50%;
            background: conic-gradient(var(--cyan) calc(var(--p)*1%), var(--surface2) 0);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 25px rgba(0,245,255,0.3), inset 0 0 15px rgba(0,0,0,0.5);
        }}
        .confidence-ring-inner {{
            width: 64px;
            height: 64px;
            background: var(--surface);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 1rem;
            color: var(--cyan);
            text-shadow: 0 0 10px var(--cyan);
        }}
        .probs {{ display: flex; flex-direction: column; gap: 10px; }}
        .prob-row {{ display: flex; align-items: center; justify-content: space-between; font-size: 0.8rem; color: var(--text-muted); letter-spacing: 0.05em; }}
        .prob-row span:last-child {{ color: var(--cyan); font-weight: 600; }}
        .prob-bar {{ height: 4px; background: var(--surface2); margin-top: 2px; overflow: hidden; border: 1px solid rgba(0,245,255,0.2); }}
        .prob-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--cyan-dim), var(--cyan));
            transition: width 0.4s ease;
            box-shadow: 0 0 10px var(--cyan);
        }}
        .activity-card {{
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 16px;
            background: rgba(0,245,255,0.04);
            border: 1px solid rgba(0,245,255,0.2);
            margin-bottom: 16px;
        }}
        .activity-gauge {{
            width: 72px;
            height: 72px;
            border-radius: 50%;
            background: conic-gradient(var(--magenta) calc(var(--a)*1%), var(--surface2) 0);
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 20px rgba(255,0,170,0.3), inset 0 0 15px rgba(0,0,0,0.5);
        }}
        .activity-gauge-inner {{
            width: 56px;
            height: 56px;
            background: var(--surface);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-mono);
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--magenta);
            text-shadow: 0 0 10px var(--magenta);
        }}
        .activity-label {{ font-weight: 600; color: var(--text); }}
        .activity-desc {{ font-size: 0.75rem; color: var(--text-muted); letter-spacing: 0.05em; }}
        .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .stat-item {{
            background: rgba(0,245,255,0.04);
            padding: 12px;
            border: 1px solid rgba(0,245,255,0.15);
            font-size: 0.8rem;
        }}
        .stat-item strong {{ color: var(--cyan); font-family: var(--font-title); letter-spacing: 0.05em; }}
        .stat-item span {{ color: var(--yellow); font-weight: 600; }}
        .alerts {{ margin-top: 16px; max-height: 140px; overflow-y: auto; }}
        .alert {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            background: rgba(255,0,170,0.08);
            border-left: 3px solid var(--magenta);
            font-size: 0.75rem;
            margin-bottom: 6px;
            color: var(--text);
            box-shadow: 0 0 15px rgba(255,0,170,0.1);
        }}
        .alerts-empty {{ font-size: 0.8rem; color: var(--text-muted); padding: 12px 0; letter-spacing: 0.05em; }}
        .footer-links {{
            grid-column: 1 / -1;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: 16px;
            border-top: 2px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
            letter-spacing: 0.1em;
        }}
        .footer-links a {{
            color: var(--cyan);
            text-decoration: none;
            font-weight: 500;
        }}
        .footer-links a:hover {{ text-shadow: var(--glow-cyan); color: var(--cyan); }}
    </style>
</head>
<body>
    <div class="layout">
        <header>
            <div class="logo">
                <div class="logo-icon">🐱</div>
                <div>
                    <h1>貓咪健康監測系統</h1>
                    <span>YOLO Pose + ST-GCN · 即時行為辨識</span>
                </div>
            </div>
            <span class="badge">v4.0-stgcn</span>
        </header>

        <section class="stream-wrap">
            <div class="stream-header">
                <span class="live-pill">Live</span>
                <span>FEED · 即時影像串流</span>
            </div>
            <img src="/stream" alt="即時串流" id="streamImg">
        </section>

        <aside class="side-panel">
            <div class="panel">
                <h2>目前行為</h2>
                <div class="behavior-now">
                    <div class="behavior-emoji" id="behaviorEmoji">{LOW_CONF_EMOJI if lc < _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD else BEHAVIOR_EMOJI_MAP.get(lb, '')}</div>
                    <div class="behavior-label" id="behaviorLabel">{get_behavior_name(lb, use_text=False, fallback='unknown', confidence=lc)}</div>
                    <div class="confidence-ring" id="confidenceRing" style="--p: {conf_pct};">
                        <div class="confidence-ring-inner"><span id="confidenceVal">{conf_pct}</span>%</div>
                    </div>
                    <div class="behavior-conf">辨識信心</div>
                </div>
                <div class="probs" id="probsList">
                    {''.join([f'<div class="prob-row"><span>{BEHAVIOR_CLASSES[i]}</span><span>{int(round(lprobs[i]*100))}%</span></div><div class="prob-bar"><div class="prob-fill" style="width:{int(round(lprobs[i]*100))}%"></div></div>' for i in range(5)])}
                </div>
            </div>

            <div class="panel" style="margin-top: 20px;">
                <h2>活動指數</h2>
                <div class="activity-card">
                    <div class="activity-gauge" id="activityGauge" style="--a: 50;">
                        <div class="activity-gauge-inner"><span id="activityVal">50</span></div>
                    </div>
                    <div>
                        <div class="activity-label">即時活動分數</div>
                        <div class="activity-desc">近 60 秒加權活動量</div>
                    </div>
                </div>
                <h2>今日統計</h2>
                <div class="stats-grid" id="todayStats">
                    <div class="stat-item"><strong>走動</strong><br><span id="statWalk">0</span> 次</div>
                    <div class="stat-item"><strong>搔抓</strong><br><span id="statScratch">0</span> 次</div>
                    <div class="stat-item"><strong>舔舐</strong><br><span id="statLick">0</span> 次</div>
                    <div class="stat-item"><strong>甩頭</strong><br><span id="statShake">0</span> 次</div>
                    <div class="stat-item"><strong>靜止</strong><br><span id="statStop">0</span> 次</div>
                    <div class="stat-item"><strong>未偵測到</strong><br><span id="statNotDetected">0</span> 秒</div>
                    <div class="stat-item"><strong>總時長</strong><br><span id="statTime">0</span> 分</div>
                </div>
                <div class="alerts" id="alertsBox">
                    <div class="alerts-empty" id="alertsEmpty">尚無警報</div>
                    <div id="alertsList"></div>
                </div>
            </div>
        </aside>

        <footer class="footer-links">
            <span>系統位址：{LOCAL_IP}:5000</span>
            <span>
                <a href="/stream">僅串流</a> · <a href="/status" target="_blank">狀態 API</a> · <a href="/api/behavior_history?limit=500" target="_blank">行為區段 (趨勢分析)</a>
            </span>
        </footer>
    </div>

    <script>
        const BEHAVIOR_NAMES = {behavior_names_js};
        const BEHAVIOR_EMOJI = {{ "-1": "😴", 0: "🐾", 1: "🐈", 2: "🧼", 3: "🐈↺", 4: "⏹" }};
        const BEHAVIOR_MIN_CONFIDENCE = {min_conf_js};

        function updateDashboard(data) {{
            const lb = data.latest_behavior ?? 0;
            const lc = data.latest_confidence ?? 0;
            const probs = data.latest_probs || [0,0,0,0,0];
            const confPct = Math.round(lc * 100);

            document.getElementById('behaviorEmoji').textContent = (lc < BEHAVIOR_MIN_CONFIDENCE ? '😴' : (BEHAVIOR_EMOJI[lb] || ''));
            document.getElementById('behaviorLabel').textContent = lc < BEHAVIOR_MIN_CONFIDENCE ? '目前正常' : (BEHAVIOR_NAMES[lb] || BEHAVIOR_NAMES[0]);
            document.getElementById('confidenceVal').textContent = confPct;
            document.getElementById('confidenceRing').style.setProperty('--p', confPct);

            const probsList = document.getElementById('probsList');
            probsList.innerHTML = BEHAVIOR_NAMES.map((name, i) => {{
                const p = Math.round((probs[i] || 0) * 100);
                return `<div class="prob-row"><span>${{name}}</span><span>${{p}}%</span></div><div class="prob-bar"><div class="prob-fill" style="width:${{p}}%"></div></div>`;
            }}).join('');

            const score = data.activity_score ?? 50;
            document.getElementById('activityVal').textContent = score;
            document.getElementById('activityGauge').style.setProperty('--a', score);

            const stats = data.today_stats || {{}};
            document.getElementById('statWalk').textContent = stats.walk ?? 0;
            document.getElementById('statScratch').textContent = stats.scratch ?? 0;
            document.getElementById('statLick').textContent = stats.lick ?? 0;
            document.getElementById('statShake').textContent = stats.shake ?? 0;
            document.getElementById('statStop').textContent = stats.stop ?? 0;
            document.getElementById('statNotDetected').textContent = stats.not_detected_time ?? 0;
            const totalMin = (stats.active_time ?? 0) / 60;
            document.getElementById('statTime').textContent = totalMin.toFixed(1);

            const alerts = data.alerts || [];
            const alertsList = document.getElementById('alertsList');
            const alertsEmpty = document.getElementById('alertsEmpty');
            if (alerts.length === 0) {{
                alertsEmpty.style.display = 'block';
                alertsList.innerHTML = '';
            }} else {{
                alertsEmpty.style.display = 'none';
                alertsList.innerHTML = alerts.map(a => {{
                    const msg = a.message || a.title || (typeof a === 'string' ? a : '');
                    return `<div class="alert">⚠ ${{msg}}</div>`;
                }}).join('');
            }}
        }}

        function fetchStatus() {{
            fetch('/status').then(r => r.json()).then(updateDashboard).catch(() => {{}});
        }}
        fetchStatus();
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>"""
        return html
    