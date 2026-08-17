"""
Flask 路由
"""

import datetime
import os
import sys
import time
from pathlib import Path

import cv2
from flask import Response, jsonify, request

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config import BaselineDashboardConfig as _BaselineDashboardConfig
from config import FlaskConfig as _FlaskConfig
from config import ModelPaths as _ModelPaths
from config import NodeRedConfig as _NodeRedConfig
from config import RunModeConfig as _RunModeConfig
from config import STGCNConfig as _STGCNConfig
from config import SystemInfo as _SystemInfo
from config import YOLOConfig as _YOLOConfig

_KP_EMA_ALPHA = _STGCNConfig.KP_EMA_ALPHA
_YOLO_MODEL_PATH = _ModelPaths.YOLO_MODEL
_STGCN_MODEL_PATH = _ModelPaths.STGCN_MODEL
_VIDEO_PATH = _ModelPaths.VIDEO_INPUT
_NODERED_RESULT_URL = _NodeRedConfig.ENDPOINT_RESULT
_IMAGE_SIZE = _YOLOConfig.IMAGE_SIZE
_CONF_THRES = _YOLOConfig.CONFIDENCE_THRESHOLD
_SEQUENCE_LENGTH = _STGCNConfig.SEQUENCE_LENGTH
_PORT = _FlaskConfig.PORT
from analytics import daily_store
from analytics.baseline import (
    HISTORY_LOAD_LIMIT_DAYS,
    MIN_BASELINE_DAYS_DEFAULT,
    DailyRecord,
    InsufficientDataError,
    compute_baseline,
)
from analytics.deviation import compute_deviation
from analytics.fusion import compute_fusion
from processors.frame_processor import FrameProcessor
from server.streaming import SharedFrameStreamer
from utils.constants import *
from utils.helpers import get_ip

frame_streamer = None
frame_processor = None
_init_lock = __import__("threading").Lock()
LOCAL_IP = get_ip() or "127.0.0.1"


def _resolve_runtime_device(preferred="cuda"):
    if preferred != "cuda":
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _daily_record_from_dict(d):
    """解析 /api/deviation 請求 body 裡的一筆每日紀錄。

    ``date`` 欄位僅接受 ISO 格式（YYYY-MM-DD）。呼叫端（目前是
    cat_health_v3_flow.json 的 v2_daily_history）若使用其他日期格式
    （例如 toLocaleDateString('zh-TW') 產生的 2026/7/2），需自行正規化
    後再送進來——這個端點刻意不嘗試猜測/相容多種日期格式，因為
    behavior_segments_log.csv 已知有 ISO 與本地格式混用的 bug，此處
    寧可讓格式錯誤在這裡就明確報錯，而不是靜默解析錯誤造成基線算錯。
    """
    raw_date = d.get("date")
    try:
        day = datetime.date.fromisoformat(str(raw_date)[:10])
    except (TypeError, ValueError):
        raise ValueError(f"date 必須是 ISO 格式 (YYYY-MM-DD)，收到: {raw_date!r}")

    kwargs = {"day": day}
    for field_name in (
        "monitoring_seconds",
        "walk_time",
        "walk_count",
        "stop_time",
        "stop_count",
        "lick_time",
        "lick_count",
        "scratch_time",
        "scratch_count",
        "shake_count",
        "active_time",
        "rest_time",
    ):
        if field_name in d:
            kwargs[field_name] = d[field_name]
    return DailyRecord(**kwargs)


def _dataclass_to_jsonable(obj):
    import dataclasses

    return dataclasses.asdict(obj)


def _today_from_live_tracker():
    """把 frame_processor.tracker.get_today_stats() 轉成 compute_deviation()
    要的欄位形狀（walk_time/walk_count/... 扁平鍵名），供 /api/deviation 在
    請求沒帶 ``today`` 時當預設值。frame_processor 尚未啟動時回傳 None——
    刻意不呼叫 _ensure_processor_started()，維持這個端點原本「跟攝影機/
    YOLO pipeline 無關」的特性，不因為呼叫這個比對端點就把攝影機管線啟動。
    """
    tracker = getattr(frame_processor, "tracker", None)
    if tracker is None:
        return None
    stats = tracker.get_today_stats()
    return {
        "walk_time": stats.get("walk_time", 0),
        "walk_count": stats.get("walk", 0),
        "stop_time": stats.get("stop_time", 0),
        "stop_count": stats.get("stop", 0),
        "lick_time": stats.get("lick_time", 0),
        "lick_count": stats.get("lick", 0),
        "scratch_time": stats.get("scratch_time", 0),
        "scratch_count": stats.get("scratch", 0),
        "shake_count": stats.get("shake", 0),
    }


def _resolve_optional_field(
    raw,
    *,
    expected_type,
    type_name,
    field_name,
    default_factory=None,
    missing_default_error=None,
    transform=None,
):
    """``/api/deviation`` 請求裡「可省略、省略時走獨立預設值」的欄位共用驗證。

    2026-08-11 從三段結構相同的 if/elif/else 抽出來（daily_history/today/
    excluded_dates），避免之後每加一個同類型欄位就手動複製貼上一份、容易
    改一份漏掉其他份（見 daily_history 目前分支已經比另外兩個多一層巢狀
    transform 邏輯，正是這種各自演化的早期徵兆）。

    行為：
      - ``raw is None``：呼叫 ``default_factory()``（若有提供）取得預設值；
        若預設值本身也是 ``None`` 且提供了 ``missing_default_error``，代表
        「連預設資料源都沒有」，回傳對應的 400（例如 today 沒有即時 tracker
        可用時）。
      - ``raw`` 是 ``expected_type``：若提供 ``transform`` 就套用（允許拋出
        ``ValueError``/``TypeError``/``AttributeError``，會被轉成 400 附上
        例外訊息），否則原樣使用。
      - 其他型別：回傳「``{field_name}`` 必須是 ``{type_name}``（或省略）」400。

    回傳 ``(value, None)`` 或 ``(None, (flask_response, status_code))``——
    呼叫端用 ``value, err = ...; if err: return err`` 的模式串接。
    """
    if raw is None:
        value = default_factory() if default_factory is not None else None
        if value is None and missing_default_error is not None:
            return None, (jsonify({"error": missing_default_error}), 400)
        return value, None
    if isinstance(raw, expected_type):
        if transform is not None:
            try:
                return transform(raw), None
            except (ValueError, TypeError, AttributeError) as e:
                return None, (jsonify({"error": str(e)}), 400)
        return raw, None
    return None, (
        jsonify({"error": f"{field_name} 必須是 {type_name}（或省略）"}),
        400,
    )


def _build_frame_processor(enable_nodered=True):
    """建立 FrameProcessor。enable_nodered=False 供本地 GUI 模式使用，
    避免在沒有 Node-RED/Flask 伺服器的情況下仍嘗試推送資料。"""
    runtime_device = _resolve_runtime_device("cuda")
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


def _try_register_lick_stage(processor, enable_nodered: bool = True) -> None:
    """Optionally attach the Lick Stage plugin. Silently skipped if plugin is absent.

    enable_nodered=False 供本地 GUI 模式使用：這個外掛有自己獨立的 Node-RED 推送
    （plugins/lick_stage/publisher.py），不受 _build_frame_processor(enable_nodered=False)
    影響，過去 GUI 模式下仍會嘗試推送並在連不到 Node-RED 時洗出警告，這裡補上同一個開關。
    """
    try:
        from plugins.lick_stage import LickStagePlugin as _LickStagePlugin

        kwargs = {} if enable_nodered else {"nodered_url": None}
        processor.register_plugin(_LickStagePlugin(**kwargs))
    except ImportError:
        pass


def _try_register_ext_body_zone(processor, enable_nodered: bool = True) -> None:
    """Optionally attach the extended 7-zone body plugin. Silently skipped if plugin is absent.

    enable_nodered=False 供本地 GUI 模式使用，理由同 _try_register_lick_stage：
    這個外掛也有自己獨立的 Node-RED 推送（ext_body_zones/output.py 的 ZoneHttpPublisher）。
    """
    try:
        from plugins.lick_stage.ext_body_zones import (
            ExtBodyZonePlugin as _ExtBodyZonePlugin,
        )

        kwargs = {} if enable_nodered else {"nodered_enabled": False}
        processor.register_plugin(_ExtBodyZonePlugin(**kwargs))
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
    """向 Flask app 註冊所有 HTTP 路由（串流、快照、影片片段、狀態查詢等）。"""

    @app.route("/stream")
    def stream():
        """回傳 MJPEG 多部分串流（即時疊圖後畫面）。"""
        _ensure_processor_started()
        # _ensure_processor_started() 在排程時段外（或初始化競爭條件下）會
        # 直接不初始化 frame_streamer 就返回，這裡要跟 /snapshot 一致地擋掉，
        # 否則 mjpeg_stream() 內對 None 呼叫 acquire_client() 會直接拋
        # AttributeError（曾在排程開始時間的邊界撞到過）。
        if frame_streamer is None:
            return Response(b"", status=503, mimetype="text/plain")

        def mjpeg_stream():
            """逐幀產生 multipart/x-mixed-replace 格式的 JPEG 資料流。"""
            frame_streamer.acquire_client()
            try:
                while True:
                    jpeg = frame_streamer.get_jpeg()
                    if jpeg is not None:
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                    else:
                        time.sleep(0.01)
            finally:
                # GeneratorExit（客戶端斷線）或任何例外都能正確釋放計數
                frame_streamer.release_client()

        return Response(
            mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/snapshot")
    def snapshot():
        """回傳目前最新一幀的單張 JPEG 快照。"""
        _ensure_processor_started()
        if frame_streamer is None:
            return Response(b"", status=503, mimetype="image/jpeg")
        # 暫時佔用一個 client slot，確保 JPEG 編碼執行緒會產生最新幀
        frame_streamer.acquire_client()
        try:
            deadline = time.time() + 0.5
            while time.time() < deadline:
                jpeg = frame_streamer.get_jpeg()
                if jpeg:
                    return Response(jpeg, mimetype="image/jpeg")
                time.sleep(0.02)
        finally:
            frame_streamer.release_client()
        return Response(b"", status=503, mimetype="image/jpeg")

    @app.route("/video_clip")
    def video_clip():
        """將 ring buffer 中保存的最近幾秒畫面編碼成短片並回傳（含縮圖）。"""
        _ensure_processor_started()
        frames = frame_streamer.get_clip_frames() if frame_streamer else []
        if not frames:
            return jsonify({"error": "no frames available"}), 503

        ts_obj = datetime.datetime.now()
        ts_file = ts_obj.strftime("%Y%m%d_%H%M%S")
        ts_display = ts_obj.strftime("%Y/%m/%d %H:%M:%S")

        save_dir = Path(_ModelPaths.OUTPUT_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        fps = float(_STGCNConfig.TARGET_MODEL_FPS)

        # mp4v 在 Windows 無額外 codec 時 isOpened() 會為 False，fallback 到 MJPG+avi
        codecs = [
            (str(save_dir / f"clip_{ts_file}.mp4"), cv2.VideoWriter_fourcc(*"mp4v")),
            (str(save_dir / f"clip_{ts_file}.avi"), cv2.VideoWriter_fourcc(*"MJPG")),
        ]
        writer = None
        save_path = ""
        for path, fourcc in codecs:
            w_ = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if w_.isOpened():
                writer = w_
                save_path = path
                break
            w_.release()

        if writer is None:
            return jsonify({"error": "no usable video codec on this machine"}), 500

        for f in frames:
            writer.write(f)
        writer.release()

        # 最後一幀轉 base64 供 Dashboard 顯示縮圖
        import base64

        last_frame = frames[-1]
        jpeg_quality = max(1, min(int(_FlaskConfig.JPEG_QUALITY), 100))
        _, jpeg_buffer = cv2.imencode(
            ".jpg", last_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        thumbnail = (
            "data:image/jpeg;base64," + base64.b64encode(jpeg_buffer.tobytes()).decode()
        )

        duration = round(len(frames) / fps, 1)
        return jsonify(
            {
                "path": save_path,
                "frames": len(frames),
                "duration": duration,
                "ts": ts_display,
                "thumbnail": thumbnail,
            }
        )

    @app.route("/api/behavior_history")
    def api_behavior_history():
        """回傳各行為區段與持續時間，供行為趨勢分析使用。支援 ?limit=200。"""
        _ensure_processor_started()
        # 排程時段外（或初始化競爭條件下）frame_processor 可能仍是 None，
        # 比照 /stream、/snapshot 的保護，避免對 None 呼叫方法直接 500
        # （曾在排程開始時間的邊界撞到過）。
        if frame_processor is None:
            return jsonify({"count": 0, "segments": []}), 503
        try:
            limit = max(1, min(int(request.args.get("limit", 200)), 1000))
        except (TypeError, ValueError):
            limit = 200
        records = frame_processor.get_behavior_history_records(limit)
        segments = [
            {
                "behavior_id": int(rec["gcn_behavior_id"]),
                "behavior": BEHAVIOR_TEXT_MAP.get(
                    rec["gcn_behavior_id"], rec["behavior"]
                ),
                "timestamp": rec["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": round(float(rec["duration"]), 1),
                "activity": int(rec.get("activity", 0)),
            }
            for rec in reversed(records)
        ]
        return jsonify(
            {
                "count": len(segments),
                "segments": segments,
            }
        )

    @app.route("/api/deviation", methods=["POST"])
    def api_deviation():
        """個體化基線 + 行為偏差評分橋接端點。

        取代 cat_health_v3_flow.json 內「偏差分析引擎」與「行為偏差融合
        引擎」兩個 function node 的統計邏輯（見 analytics/README.md）。
        與攝影機/YOLO pipeline 無關，不會觸發 _ensure_processor_started()。

        2026-08-10 起：``daily_history``／``today``／``excluded_dates`` 都改成
        可省略。省略時分別預設讀取 Python 端**自己獨立收集、獨立持久化**的
        多天歷史（``analytics/daily_store.py``，資料來源是 ``behavior_tracker``
        每天跨日時自己彙整的一筆記錄）、即時 tracker 資料、與 Python 自己的
        排除清單（``daily_store.load_excluded_dates()``，透過
        ``analytics/manage_baseline_history.py`` GUI 工具編輯），不再需要呼叫端
        （目前是 Node-RED）把它自己的 ``v2_daily_history``/``v2_today``/
        ``v2_excluded_dates`` forward 過來——兩邊資料源從此互不相干，任一邊
        被寫壞都不會波及另一邊（見 docs/資料層架構現況與統一管理評估.md
        第九節）。仍然接受顯式帶入這三個欄位當一次性覆寫（例如測試、手動
        比對用），帶了就以請求內容為準，不去讀 Python 自己的資料源。

        請求 body（全部欄位皆可省略，預設走 Python 自己的資料源）：
            {
              "daily_history": [{"date": "2026-06-01", "monitoring_seconds": 7200,
                                  "walk_time": ..., "lick_count": ..., ...}, ...],  // 可省略
              "today": {"walk_time": ..., "lick_count": ..., ...},                 // 可省略
              "excluded_dates": ["2026-06-05", ...],   // 可省略；省略時預設用
                                                          // daily_store 自己的排除清單
              "min_baseline_days": 7,                   // 可省略，預設 7
              "class_c_score": 0                        // 可省略；節律/轉移分數暫由
                                                          // Node-RED 自行計算後傳入
            }

        回應：
            成功 → {"status":"ok", "baseline":{...}, "deviation":{...}, "fusion":{...}}
            基線資料不足 → {"status":"insufficient_data", "current_days":N, "required_days":M}
            請求格式錯誤／沒有可用的 today 資料來源 → 400 {"error": "..."}
        """
        body = request.get_json(silent=True) or {}

        daily_records, err = _resolve_optional_field(
            body.get("daily_history"),
            expected_type=list,
            type_name="list",
            field_name="daily_history",
            # limit_days：daily_history 這張表只會累積、從不刪除，
            # compute_baseline() 最終只用得到最近 MAX_BASELINE_DAYS_DEFAULT
            # 天，這個端點又是 Node-RED 每 ~2 秒觸發一次的高頻路徑，不限制
            # 讀取筆數的話系統跑越久這裡的 SQL 掃描/物件建構成本就越高，
            # 見 analytics/config.py 的 HISTORY_LOAD_LIMIT_DAYS 說明。
            default_factory=lambda: daily_store.load_history(
                limit_days=HISTORY_LOAD_LIMIT_DAYS
            ),
            transform=lambda items: [_daily_record_from_dict(d) for d in items],
        )
        if err:
            return err

        today, err = _resolve_optional_field(
            body.get("today"),
            expected_type=dict,
            type_name="dict",
            field_name="today",
            default_factory=_today_from_live_tracker,
            missing_default_error=(
                "沒有帶 today，且攝影機管線尚未啟動、"
                "無法取得即時資料；請帶入 today 或先啟動監測。"
            ),
        )
        if err:
            return err

        # excluded_dates 省略時預設用 Python 自己管理的排除清單（見
        # analytics/manage_baseline_history.py、analytics/daily_store.py 的
        # set_excluded()/load_excluded_dates()），跟 daily_history/today
        # 的預設邏輯一致；Node-RED 若明確帶了（即使是空陣列）就以請求
        # 內容為準，代表呼叫端刻意要用它自己的 v2_excluded_dates。
        excluded_dates, err = _resolve_optional_field(
            body.get("excluded_dates"),
            expected_type=list,
            type_name="list",
            field_name="excluded_dates",
            default_factory=daily_store.load_excluded_dates,
        )
        if err:
            return err
        try:
            # 2026-08-11 修正：預設值原本寫死 7，跟 BaselineConfig 引入後
            # dashboard/refresher.py、analytics/manage_baseline_history.py
            # 改用的 compute_baseline() 自身預設值（可被
            # CAT_MONITORING_ANALYTICS_MIN_BASELINE_DAYS 覆蓋）不一致，
            # 導致省略 min_baseline_days 的請求跟另外兩個引擎對同一份資料
            # 的「夠不夠天數」判斷會兜不起來。改成同一個來源。
            min_baseline_days = int(
                body.get("min_baseline_days", MIN_BASELINE_DAYS_DEFAULT)
            )
        except (TypeError, ValueError):
            return jsonify({"error": "min_baseline_days 必須是整數"}), 400
        try:
            class_c_score = float(body.get("class_c_score", 0.0))
        except (TypeError, ValueError):
            return jsonify({"error": "class_c_score 必須是數字"}), 400

        try:
            baseline = compute_baseline(
                daily_records,
                min_days=min_baseline_days,
                excluded_dates=excluded_dates,
            )
        except InsufficientDataError as e:
            return jsonify(
                {
                    "status": "insufficient_data",
                    "current_days": e.current_days,
                    "required_days": e.required_days,
                }
            )

        deviation = compute_deviation(today=today, baseline=baseline)
        fusion = compute_fusion(deviation, class_c_score=class_c_score)

        result = {
            "status": "ok",
            "baseline": _dataclass_to_jsonable(baseline),
            "deviation": _dataclass_to_jsonable(deviation),
            "fusion": _dataclass_to_jsonable(fusion),
        }
        if _BaselineDashboardConfig.ENABLED:
            # 延遲匯入：旗標關閉時完全不載入 dashboard 套件，見 dashboard/__init__.py。
            from dashboard.cache import set_latest as _set_baseline_dashboard_cache

            _set_baseline_dashboard_cache(result)
        return jsonify(result)

    def _cors(resp, status=200):
        """在回應上加 CORS header，讓 ui_template 的 fetch() 可跨 port 呼叫。"""
        r = Response(resp.get_data(), status=status, mimetype="application/json")
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r

    @app.route("/api/overlay", methods=["GET", "POST", "OPTIONS"])
    def api_overlay():
        """讀取或更新畫面 overlay 顯示旗標。
        GET     → 回傳目前所有旗標狀態
        OPTIONS → CORS preflight
        POST    → {"key": "skeleton"|"label"|"bbox"|"master", "value": true|false}
                  或 {"key": "...", "action": "toggle"} → server 自行翻轉，不需 client 追蹤狀態
                  master=true 時同時重置所有子旗標為 true。
        """
        if request.method == "OPTIONS":
            return _cors(jsonify({}))

        _ensure_processor_started()
        # 排程時段外（或初始化競爭條件下）frame_processor 可能仍是 None，
        # 比照 /stream、/snapshot 的保護，避免對 None 呼叫屬性直接 500
        # （曾在排程開始時間的邊界撞到過）。
        if frame_processor is None:
            return _cors(jsonify({"error": "processor not started"}), 503)

        if request.method == "GET":
            return _cors(
                jsonify(
                    {
                        "master": frame_processor.overlay,
                        "skeleton": frame_processor.show_skeleton,
                        "label": frame_processor.show_label,
                        "bbox": frame_processor.show_bbox,
                    }
                )
            )

        body = request.get_json(silent=True) or {}
        key = body.get("key")
        value = body.get("value")
        action = body.get("action")
        if key is None:
            return _cors(jsonify({"error": "需要 key"}), 400)

        if action == "toggle":
            current = {
                "master": frame_processor.overlay,
                "skeleton": frame_processor.show_skeleton,
                "label": frame_processor.show_label,
                "bbox": frame_processor.show_bbox,
            }
            if key not in current:
                return _cors(jsonify({"error": f"未知 key: {key!r}"}), 400)
            value = not current[key]

        if not isinstance(value, bool):
            return _cors(jsonify({"error": "需要 value(bool) 或 action='toggle'"}), 400)

        if key == "master":
            frame_processor.overlay = value
            frame_processor.show_skeleton = value
            frame_processor.show_label = value
            frame_processor.show_bbox = value
        elif key == "skeleton":
            frame_processor.show_skeleton = value
        elif key == "label":
            frame_processor.show_label = value
        elif key == "bbox":
            frame_processor.show_bbox = value
        else:
            return _cors(jsonify({"error": f"未知 key: {key!r}"}), 400)

        return _cors(
            jsonify(
                {
                    "ok": True,
                    "key": key,
                    "value": value,
                    "state": {
                        "master": frame_processor.overlay,
                        "skeleton": frame_processor.show_skeleton,
                        "label": frame_processor.show_label,
                        "bbox": frame_processor.show_bbox,
                    },
                }
            )
        )

    @app.route("/")
    def index():
        """回傳簡易首頁（含各端點連結），並確保處理管線已啟動。"""
        _ensure_processor_started()
        _dashboard_link = (
            "<li><a href='/dashboard/baseline'>individualized baseline dashboard</a></li>"
            if _BaselineDashboardConfig.ENABLED
            else ""
        )
        return Response(
            f"<html><body><p>{_SystemInfo.SYSTEM_NAME} {_SystemInfo.VERSION} &mdash; {LOCAL_IP}:{_PORT}</p>"
            f"<ul><li><a href='/stream'>stream</a></li>"
            f"<li><a href='/api/behavior_history?limit=500'>behavior history</a></li>"
            f"{_dashboard_link}</ul>"
            f"</body></html>",
            mimetype="text/html",
        )
