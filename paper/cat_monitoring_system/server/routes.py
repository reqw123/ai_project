"""
Flask 路由
"""
from flask import Response, jsonify, request
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
from config import RunModeConfig as _RunModeConfig

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
from analytics.baseline import DailyRecord, compute_baseline, InsufficientDataError
from analytics.deviation import compute_deviation
from analytics.fusion import compute_fusion


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


def _daily_record_from_dict(d):
    """解析 /api/deviation 請求 body 裡的一筆每日紀錄。

    ``date`` 欄位僅接受 ISO 格式（YYYY-MM-DD）。呼叫端（目前是
    cat_health_v3_flow.json 的 v2_daily_history）若使用其他日期格式
    （例如 toLocaleDateString('zh-TW') 產生的 2026/7/2），需自行正規化
    後再送進來——這個端點刻意不嘗試猜測/相容多種日期格式，因為
    behavior_segments_log.csv 已知有 ISO 與本地格式混用的 bug，此處
    寧可讓格式錯誤在這裡就明確報錯，而不是靜默解析錯誤造成基線算錯。
    """
    raw_date = d.get('date')
    try:
        day = datetime.date.fromisoformat(str(raw_date)[:10])
    except (TypeError, ValueError):
        raise ValueError(f"date 必須是 ISO 格式 (YYYY-MM-DD)，收到: {raw_date!r}")

    kwargs = {'day': day}
    for field_name in (
        'monitoring_seconds', 'walk_time', 'walk_count', 'stop_time', 'stop_count',
        'lick_time', 'lick_count', 'scratch_time', 'scratch_count', 'shake_count',
        'active_time', 'rest_time',
    ):
        if field_name in d:
            kwargs[field_name] = d[field_name]
    return DailyRecord(**kwargs)


def _dataclass_to_jsonable(obj):
    import dataclasses
    return dataclasses.asdict(obj)


def _build_frame_processor(enable_nodered=True):
    """建立 FrameProcessor。enable_nodered=False 供本地 GUI 模式使用，
    避免在沒有 Node-RED/Flask 伺服器的情況下仍嘗試推送資料。"""
    runtime_device = _resolve_runtime_device('cuda')
    return FrameProcessor(
        yolo_model_path=_YOLO_MODEL_PATH,
        stgcn_model_path=_STGCN_MODEL_PATH,
        video_path=_VIDEO_PATH,
        nodered_url=_NODERED_RESULT_URL if enable_nodered else None,
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


def _try_register_ext_body_zone(processor) -> None:
    """Optionally attach the extended 7-zone body plugin. Silently skipped if plugin is absent."""
    try:
        from plugins.lick_stage.ext_body_zones import ExtBodyZonePlugin as _ExtBodyZonePlugin
        processor.register_plugin(_ExtBodyZonePlugin())
    except ImportError:
        pass


def _ensure_processor_started():
    """在首次請求時啟動處理管線（double-checked locking，避免多執行緒重複建立）。

    若設定了排程時間（RunModeConfig.SCHEDULED_START_TIME/SCHEDULED_END_TIME）且目前不在
    允許的時間內，即使有請求打進來（例如使用者提早打開 Dashboard 點播放）也不會啟動，
    直接原地不動；真正的啟動/暫停/恢復由 main.py 的排程迴圈依時間持續驅動。
    """
    global frame_streamer, frame_processor
    if frame_processor is not None and frame_streamer is not None:
        if frame_streamer.paused and _RunModeConfig.is_within_active_window():
            frame_streamer.paused = False
        return
    if not _RunModeConfig.is_within_active_window():
        return
    with _init_lock:
        if frame_processor is None:
            frame_processor = _build_frame_processor()
            _try_register_lick_stage(frame_processor)
            _try_register_ext_body_zone(frame_processor)
        if frame_streamer is None:
            frame_streamer = SharedFrameStreamer(frame_processor)


def _pause_processing():
    """排程區段執行用：離開允許時間時呼叫，暫停讀取/推論，但不釋放模型與 VideoCapture。"""
    if frame_streamer is not None:
        frame_streamer.paused = True


def register_routes(app):

    @app.route('/stream')
    def stream():
        _ensure_processor_started()
        # _ensure_processor_started() 在排程時段外（或初始化競爭條件下）會
        # 直接不初始化 frame_streamer 就返回，這裡要跟 /snapshot 一致地擋掉，
        # 否則 mjpeg_stream() 內對 None 呼叫 acquire_client() 會直接拋
        # AttributeError（曾在排程開始時間的邊界撞到過）。
        if frame_streamer is None:
            return Response(b'', status=503, mimetype='text/plain')
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

    @app.route('/api/deviation', methods=['POST'])
    def api_deviation():
        """個體化基線 + 行為偏差評分橋接端點。

        取代 cat_health_v3_flow.json 內「偏差分析引擎」與「行為偏差融合
        引擎」兩個 function node 的統計邏輯（見 analytics/README.md）。
        與攝影機/YOLO pipeline 無關，不會觸發 _ensure_processor_started()。

        請求 body：
            {
              "daily_history": [{"date": "2026-06-01", "monitoring_seconds": 7200,
                                  "walk_time": ..., "lick_count": ..., ...}, ...],
              "today": {"walk_time": ..., "lick_count": ..., ...},
              "excluded_dates": ["2026-06-05", ...],   // 可省略
              "min_baseline_days": 7,                   // 可省略，預設 7
              "class_c_score": 0                        // 可省略；節律/轉移分數暫由
                                                          // Node-RED 自行計算後傳入
            }

        回應：
            成功 → {"status":"ok", "baseline":{...}, "deviation":{...}, "fusion":{...}}
            基線資料不足 → {"status":"insufficient_data", "current_days":N, "required_days":M}
            請求格式錯誤 → 400 {"error": "..."}
        """
        body = request.get_json(silent=True) or {}
        raw_history = body.get('daily_history')
        today = body.get('today')
        if not isinstance(raw_history, list) or not isinstance(today, dict):
            return jsonify({"error": "需要 daily_history(list) 與 today(dict)"}), 400

        try:
            daily_records = [_daily_record_from_dict(d) for d in raw_history]
        except (ValueError, TypeError, AttributeError) as e:
            return jsonify({"error": str(e)}), 400

        excluded_dates = body.get('excluded_dates') or []
        try:
            min_baseline_days = int(body.get('min_baseline_days', 7))
        except (TypeError, ValueError):
            return jsonify({"error": "min_baseline_days 必須是整數"}), 400
        try:
            class_c_score = float(body.get('class_c_score', 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "class_c_score 必須是數字"}), 400

        try:
            baseline = compute_baseline(
                daily_records, min_days=min_baseline_days, excluded_dates=excluded_dates,
            )
        except InsufficientDataError as e:
            return jsonify({
                "status": "insufficient_data",
                "current_days": e.current_days,
                "required_days": e.required_days,
            })

        deviation = compute_deviation(today=today, baseline=baseline)
        fusion = compute_fusion(deviation, class_c_score=class_c_score)

        return jsonify({
            "status": "ok",
            "baseline": _dataclass_to_jsonable(baseline),
            "deviation": _dataclass_to_jsonable(deviation),
            "fusion": _dataclass_to_jsonable(fusion),
        })

    def _cors(resp, status=200):
        """在回應上加 CORS header，讓 ui_template 的 fetch() 可跨 port 呼叫。"""
        r = Response(resp.get_data(), status=status, mimetype='application/json')
        r.headers['Access-Control-Allow-Origin']  = '*'
        r.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return r

    @app.route('/api/overlay', methods=['GET', 'POST', 'OPTIONS'])
    def api_overlay():
        """讀取或更新畫面 overlay 顯示旗標。
        GET     → 回傳目前所有旗標狀態
        OPTIONS → CORS preflight
        POST    → {"key": "skeleton"|"label"|"bbox"|"master", "value": true|false}
                  或 {"key": "...", "action": "toggle"} → server 自行翻轉，不需 client 追蹤狀態
                  master=true 時同時重置所有子旗標為 true。
        """
        if request.method == 'OPTIONS':
            return _cors(jsonify({}))

        _ensure_processor_started()
        if request.method == 'GET':
            return _cors(jsonify({
                "master":   frame_processor.overlay,
                "skeleton": frame_processor.show_skeleton,
                "label":    frame_processor.show_label,
                "bbox":     frame_processor.show_bbox,
            }))

        body   = request.get_json(silent=True) or {}
        key    = body.get('key')
        value  = body.get('value')
        action = body.get('action')
        if key is None:
            return _cors(jsonify({"error": "需要 key"}), 400)

        if action == 'toggle':
            current = {
                'master':   frame_processor.overlay,
                'skeleton': frame_processor.show_skeleton,
                'label':    frame_processor.show_label,
                'bbox':     frame_processor.show_bbox,
            }
            if key not in current:
                return _cors(jsonify({"error": f"未知 key: {key!r}"}), 400)
            value = not current[key]

        if not isinstance(value, bool):
            return _cors(jsonify({"error": "需要 value(bool) 或 action='toggle'"}), 400)

        if key == 'master':
            frame_processor.overlay = value
            if value:
                frame_processor.show_skeleton = True
                frame_processor.show_label    = True
                frame_processor.show_bbox     = True
        elif key == 'skeleton':
            frame_processor.show_skeleton = value
        elif key == 'label':
            frame_processor.show_label = value
        elif key == 'bbox':
            frame_processor.show_bbox = value
        else:
            return _cors(jsonify({"error": f"未知 key: {key!r}"}), 400)

        return _cors(jsonify({
            "ok": True, "key": key, "value": value,
            "state": {
                "master":   frame_processor.overlay,
                "skeleton": frame_processor.show_skeleton,
                "label":    frame_processor.show_label,
                "bbox":     frame_processor.show_bbox,
            }
        }))

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
    