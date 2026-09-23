"""
幀處理管道（整合 Node-RED、CSV、異常檢測、overlay 控制）
"""

import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from config import (
    BehaviorTrackingConfig,
    CatIdentityConfig,
    ESP32CamConfig,
    NodeRedConfig,
    RunModeConfig,
    SQAConfig,
    STGCNConfig,
    SystemInfo,
    VisualizationConfig,
)

from communication.nodered_client import NodeRedClient
from detectors.behavior_classifier import BehaviorClassifier
from detectors.keypoint_detector import KeypointDetector
from logutils.csv_logger import BehaviorSegmentLogger, CSVLogger
from models.stgcn_model import interpolate_missing
from processors.anomaly_detector import AnomalyDetector
from processors.visualizer import Visualizer
from trackers.behavior_tracker import ImprovedBehaviorTracker
from utils.constants import *
from utils.esp32cam import configure_stream as _configure_esp32cam_stream
from utils.helpers import get_behavior_name, get_ip, is_stream_url, resolve_video_source

# Skeleton Quality Assessment（GCN 分類為主、幾何判斷為輔雙重判定）：獨立
# 模組，import 失敗時整套機制自動停用（_sqa_evaluate_window 保持 None），
# 不會讓主系統啟動失敗——這個模組也可以整個被刪除，不影響其餘功能。
try:
    from processors.skeleton_quality_assessment import (
        evaluate_window as _sqa_evaluate_window,
    )
except Exception:
    _sqa_evaluate_window = None

# 身分驗證（多貓辨識）：獨立模組，import 失敗時整套機制自動停用
# （_IdentityVerifier 保持 None），不會讓主系統啟動失敗——這個模組也可以
# 整個被刪除，不影響其餘偵測/分類流程（見 detectors/identity_verifier.py）。
try:
    from detectors.identity_verifier import IdentityVerifier as _IdentityVerifier
except Exception:
    _IdentityVerifier = None

# M4：舔毛/身體分區外掛共用的姿態平滑＋骨長品質模組（plugins/lick_stage/
# pose_filter.py）。獨立模組，import 失敗時整套機制自動停用（_PoseFilter
# 保持 None，_notify_plugins() 退回舊行為：不做平滑、直接傳 raw kpts、
# pose_quality 恆為 None），跟 lick_stage 外掛本身「可以整個被刪除」的既有
# 原則一致——這裡多一層獨立 import guard，即使只刪除 plugins/lick_stage/
# 也不會讓 FrameProcessor 啟動失敗。
try:
    from plugins.lick_stage.config import LickConfig as _LickConfig
    from plugins.lick_stage.pose_filter import PoseFilter as _PoseFilter
except Exception:
    _LickConfig = None
    _PoseFilter = None

# M6（2026-09-15，ext_body_zones 統一 ontology 融合）：_notify_plugins() 要
# 把 ext_body_zones 的逐幀分類結果讀出來、轉餵給 lick_stage 的事件聚合器當
# 補充欄位（見該方法內的說明）——這一步需要認得這兩個具體的外掛類別，跟
# M4 的 PoseFilter（對所有外掛一視同仁）不同，是刻意的、有目的性的耦合，
# 僅限於 frame_processor.py 這個組裝層，兩個外掛彼此的程式碼本身仍然零
# 依賴（ExtBodyZonePlugin 不 import LickStagePlugin，反之亦然）。同樣獨立
# import guard，任一個/兩個都不存在時這段融合邏輯自動停用，不影響其餘功能。
try:
    from plugins.lick_stage.manager import LickStagePlugin as _LickStagePlugin
    from plugins.lick_stage.ext_body_zones.plugin import (
        ExtBodyZonePlugin as _ExtBodyZonePlugin,
    )
    from plugins.lick_stage.ext_body_zones.config import ExtZoneConfig as _ExtZoneConfig
except Exception:
    _LickStagePlugin = None
    _ExtBodyZonePlugin = None
    _ExtZoneConfig = None


class _LatestFrameGrabber:
    """背景執行緒持續讀取 cv2.VideoCapture，只保留最新一幀。

    即時網路串流（RTSP/HLS 等）若推論速度跟不上來源幀率，ffmpeg/OS 內部
    緩衝區會持續堆積未消化的舊幀，導致畫面隨執行時間拉長越來越落後
    real-time（延遲會一直累積，不是固定值）。這裡用一個獨立執行緒盡快
    把緩衝區「抽乾」，永遠只保留最新一幀給主處理迴圈使用，讓延遲鎖定在
    串流協定本身的固定延遲，不會再疊加我們自己的處理耗時。

    本機影片檔案沒有這個問題（檔案沒有「即時」概念，讀取本身不會累積
    延遲），因此只在偵測到網路串流來源時才由 FrameProcessor 啟用。
    """

    # 連續讀取失敗次數門檻：超過此值才嘗試重新開啟連線。單次或偶發幾次
    # read() 失敗多半是暫時性的封包延遲，靠 FFmpeg 內部重試就會恢復；
    # 若持續失敗代表底層連線已經斷開，純粹重試 read() 不會自己好，
    # 需要 release() + open() 重新建立連線才能接回。門檻抓約 3 秒
    # （失敗時每次 sleep 0.05s）避免對單幀失敗過度敏感而誤觸重連。
    RECONNECT_FAILURE_THRESHOLD = 60

    def __init__(self, cap, video_url=None):
        self._cap = cap
        self._video_url = video_url
        self._lock = threading.Lock()
        self._latest_ret = False
        self._latest_frame = None
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        # 讀取成功時完全不節流會讓這個執行緒盡可能快地連續解碼（HLS 緩衝到
        # 一段之後常常可以遠快於即時速度解碼），持續佔用一整個 CPU 核心跟
        # GIL，跟主執行緒（YOLO/ST-GCN/畫面繪製）搶執行時間；偵測到貓時主
        # 執行緒單幀工作量變重（骨架/機率條/plugin overlay），搶輸的機率
        # 更高，感覺就像「畫框瞬間卡住」。這裡把讀取步調限制在來源幀率
        # 附近，讓這個執行緒平常有空檔可以釋出 GIL，不會這麼容易搶到。
        frame_interval = 1.0 / 30.0
        try:
            src_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if src_fps and src_fps > 1:
                frame_interval = 1.0 / src_fps
        except Exception:
            pass

        consecutive_failures = 0
        while self._running:
            loop_start = time.time()
            ret, frame = self._cap.read()
            consecutive_failures = 0 if ret else consecutive_failures + 1
            with self._lock:
                self._latest_ret = ret
                self._latest_frame = frame
            if not ret:
                if (
                    self._video_url is not None
                    and consecutive_failures >= self.RECONNECT_FAILURE_THRESHOLD
                ):
                    try:
                        self._cap.release()
                        self._cap.open(self._video_url)
                    except Exception:
                        pass
                    consecutive_failures = 0
                time.sleep(0.05)  # 讀取失敗（斷線/暫時中斷）時避免忙迴圈佔滿 CPU
                continue
            remaining = frame_interval - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    def read(self):
        """回傳背景執行緒目前保有的最新一幀 (ret, frame)。"""
        with self._lock:
            return self._latest_ret, self._latest_frame

    def stop(self):
        """停止背景讀取執行緒並等待其結束。"""
        self._running = False
        self._thread.join(timeout=2.0)


# 舔舐二階段處理（plugins/lick_stage 系列）僅在 ST-GCN 已「成功」判定為
# lick（即信心值 >= STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD，見下方判定式）
# 時才啟動，避免在 walk/scratch/shake/stop 或低信心期間空跑鼻部接觸幾何。
_LICK_BEHAVIOR_ID = BEHAVIOR_CLASSES.index("lick")


class FrameProcessor:
    """整合單一影片/串流來源的完整處理管線：YOLO 關鍵點偵測 → ST-GCN 行為
    分類 → 異常偵測 → 骨架品質雙重判定（SQA）→ 行為追蹤 → 視覺化疊圖 →
    CSV 記錄 → Node-RED 推送，並管理選用插件（plugins）的呼叫時機。"""

    def __init__(
        self,
        yolo_model_path,
        stgcn_model_path,
        video_path,
        nodered_url=None,
        device="cuda",
        imgsz=640,
        conf_thres=0.5,
        sequence_length=STGCNConfig.SEQUENCE_LENGTH,
        overlay=True,
        width=None,
        height=None,
        normalize=True,
        kp_ema_alpha=STGCNConfig.KP_EMA_ALPHA,
        feature_mode=None,
        window_stride=None,
        plugins=None,
    ):
        self.local_ip = get_ip()
        # YouTube 網頁網址無法直接餵給 cv2.VideoCapture，先用 yt_dlp 解析出
        # 實際的串流網址；非 YouTube 來源（檔案/攝影機 index/RTSP）原樣不變。
        resolved_video_path = resolve_video_source(video_path)
        if is_stream_url(resolved_video_path):
            # 網路串流（YouTube HLS/DASH 等）偶爾會遇到 CDN 切換或短暫封包延遲，
            # 讓 FFmpeg 底層的 read() 卡住甚至回傳失敗；預設不會自動重試連線。
            # 這裡透過 FFmpeg 的 AVOption 開啟自動重連，讓多數短暫斷線在
            # libavformat 內部就恢復，不會表現成畫面卡頓。setdefault 避免
            # 覆蓋使用者已自行設定的值。
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5",
            )
            # ESP32-CAM：MJPEG HTTP 串流的畫面尺寸由韌體端 framesize 決定，
            # 下面 cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT) 對它完全無效。framesize
            # 設太大時 ESP32-CAM 透過 WiFi 只能推個位數 fps、每張 JPEG 又大，
            # 會嚴重卡頓且畫面過大。開串流「之前」先送一次 framesize 控制請求
            # 把來源壓到目標解析度（預設 640x480），同時解掉「解析度過大」與
            # 「卡頓」。來源不是 ESP32-CAM（控制端點連不上）時請求失敗會被安靜
            # 略過，不影響後續開串流。
            if (
                ESP32CamConfig.AUTO_FRAMESIZE
                and isinstance(resolved_video_path, str)
                and resolved_video_path.lower().startswith(("http://", "https://"))
            ):
                _ok, _detail = _configure_esp32cam_stream(
                    resolved_video_path,
                    ESP32CamConfig.TARGET_WIDTH,
                    ESP32CamConfig.TARGET_HEIGHT,
                    control_port=ESP32CamConfig.CONTROL_PORT,
                    quality=ESP32CamConfig.QUALITY,
                    timeout=ESP32CamConfig.CONTROL_TIMEOUT,
                )
                if _ok:
                    print(f"✓ 已請求 ESP32-CAM 調整串流輸出：{_detail}")
                    # 感光元件切換 framesize 後需要一小段時間重新初始化，
                    # 太快開串流會先讀到幾張舊尺寸/破損的幀。
                    time.sleep(0.6)
                else:
                    print(
                        f"⚠ 未套用 ESP32-CAM 串流設定（來源可能不是 ESP32-CAM，"
                        f"不影響後續運作）：{_detail}"
                    )
        self.cap = cv2.VideoCapture(resolved_video_path)
        # 攝影機（尤其 USB webcam）驅動列舉/協商常比檔案或串流來源慢，
        # 短暫重試幾次再放棄，避免第一個 /stream 請求就直接 500。
        _open_retries = 5
        _open_retry_delay = 0.5
        for _ in range(_open_retries):
            if self.cap.isOpened():
                break
            time.sleep(_open_retry_delay)
            self.cap = cv2.VideoCapture(resolved_video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_path}")
        if width and height and not is_stream_url(resolved_video_path):
            # 串流來源（含 ESP32-CAM）的畫面尺寸不是這裡能控制的：MJPEG/RTSP 的
            # cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT) 是無效操作，讀回的值恆等於來源
            # 原始尺寸，比對必然「不同」，只會印出誤導的警告。ESP32-CAM 的解析度
            # 已在上面開串流前透過韌體控制端點調整，這裡直接跳過。
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # cv2.VideoCapture.set() 對攝影機來源是「請求」不是「保證」——驅動收到
            # 不支援的解析度時，可能悄悄退回它自己的預設值而不報錯，.set() 呼叫本身
            # 一樣回傳成功。讀回實際協商到的值印出來，才能在畫面看起來不對勁時
            # （像素感明顯／畫面過大）直接從 log 判斷是不是解析度沒被驅動採納，
            # 不用憑空猜測。對影片檔/串流這個 .set() 是無效操作，讀回的值會等於
            # 檔案原始解析度，不代表有問題。
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_w, actual_h) != (width, height):
                print(
                    f"⚠ 影像來源實際協商到的解析度（{actual_w}x{actual_h}）"
                    f"跟請求的（{width}x{height}）不同——多半是攝影機驅動不支援"
                    f"該解析度、自動退回它自己的預設值，屬正常現象，非程式錯誤。"
                )
        # 即時網路串流才啟用「只保留最新幀」的背景讀取，避免推論跟不上時
        # 延遲隨執行時間持續累積；本機檔案維持原本同步讀取行為不變。放在
        # width/height 設定之後才啟動背景執行緒，避免和 cap.set() 競態。
        self._grabber = (
            _LatestFrameGrabber(self.cap, video_url=resolved_video_path)
            if is_stream_url(resolved_video_path)
            else None
        )
        try:
            self.keypoint_detector = KeypointDetector(
                yolo_model_path, device=device, imgsz=imgsz, conf_thres=conf_thres
            )
            self.behavior_classifier = BehaviorClassifier(
                stgcn_model_path,
                device=device,
                sequence_length=sequence_length,
                normalize=normalize,
                feature_mode=feature_mode,
            )
        except Exception:
            if self._grabber is not None:
                self._grabber.stop()
            self.cap.release()
            raise
        # 身分驗證（多貓辨識）：獨立於上面 YOLO/ST-GCN 的 try/except 之外，
        # 失敗（模組被刪除/基準檔不存在或損毀）只會停用這一層、不影響
        # FrameProcessor 其餘功能——跟 SQA 同一套 fail-safe 慣例。
        self.identity_verifier = None
        # 多貓同框時只有「目標貓」進入行為分類 / tracker / CSV / Node-RED / 基線，
        # 非目標貓由 process() 用灰框灰骨架畫出來（VisualizationConfig.
        # SHOW_NON_TARGET_CATS）。_identity_filtering_active＝這一幀畫面裡找不到
        # 目標貓（都判為他貓 / 都低於信心門檻），只在此狀態轉換的瞬間印 console。
        self._identity_filtering_active = False
        # 追蹤中的貓被 CNN「明確」判為非目標類別的連續幀數；達到
        # CatIdentityConfig.IDENTITY_FILTER_HYSTERESIS_FRAMES 才真的放掉目標貓。
        # 目標貓 / 分不清（未知）都會把它歸零。
        self._identity_nontarget_streak = 0
        # 單貓 / 多貓系統模式（config.py RunModeConfig.SYSTEM_MODE）。single 模式下
        # 完全跳過多貓相關邏輯：只取信心最高的偵測、不畫非目標貓、不載入身分驗證 CNN。
        self._system_mode = RunModeConfig.SYSTEM_MODE if RunModeConfig.SYSTEM_MODE in ("single", "multi") else "single"
        self._multi_cat = self._system_mode == "multi"
        print(
            f"● 系統模式：{self._system_mode}"
            + ("（多貓：挑目標貓 / 畫其他貓）" if self._multi_cat else "（單貓：只取信心最高的偵測）")
        )
        # CatIdentityConfig.is_active()＝ENABLE_IDENTITY_VERIFICATION 且 SYSTEM_MODE=="multi"。
        # single 模式下一律 False，這一段整個跳過、不載入 CNN。
        if CatIdentityConfig.is_active() and _IdentityVerifier is not None:
            try:
                self.identity_verifier = _IdentityVerifier(
                    model_path=CatIdentityConfig.IDENTITY_MODEL_PATH,
                    target_class=CatIdentityConfig.TARGET_CAT_CLASS,
                    device=device,
                    identity_conf_threshold=CatIdentityConfig.IDENTITY_CONF_THRESHOLD,
                )
                print(
                    f"✓ 身分驗證已啟用（CNN）：只有判定為「{CatIdentityConfig.TARGET_CAT_CLASS}」"
                    f"的偵測結果會計入統計"
                )
            except Exception as e:
                print(f"⚠ 身分驗證初始化失敗，已停用（偵測到的貓將一律視為目標貓）：{e}")
                self.identity_verifier = None
        elif not self._multi_cat and CatIdentityConfig.ENABLE_IDENTITY_VERIFICATION:
            print("ℹ 系統模式為 single，已忽略「啟用身份驗證」設定（身分驗證只在 multi 模式生效）")

        self.tracker = ImprovedBehaviorTracker()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = Visualizer()
        self.keypoints_buffer = deque(maxlen=sequence_length)
        self.sequence_length = sequence_length
        self.window_stride = (
            window_stride if window_stride is not None else STGCNConfig.WINDOW_STRIDE
        )
        self._infer_frame_count = 0  # 累積有效幀計數器，以 window_stride 取模決定推論時機（貓咪消失時重置，確保重新出現後推論時機從 0 對齊）
        self.overlay = overlay
        self.show_skeleton = VisualizationConfig.SHOW_SKELETON
        self.show_label = VisualizationConfig.SHOW_GCN_RESULT
        self.show_bbox = VisualizationConfig.SHOW_BBOX
        self.prev_time = time.time()
        self.last_send_time = time.time()
        self.nodered = None
        if nodered_url:
            self.nodered = NodeRedClient(nodered_url)
        self.csv_logger = CSVLogger()
        self.segment_logger = BehaviorSegmentLogger()
        self.frame_idx = 0
        # ── 舔毛外掛的來源時間契約（說明書第一階段「來源時間規格」）──────
        # source_fps 每次讀取都會呼叫 cap.get()，這裡在建構時快取一份，供
        # process() 每幀換算 source_timestamp = frame_idx / source_fps 使用。
        try:
            self._plugin_source_fps = float(self.source_fps)
        except Exception:
            self._plugin_source_fps = float(STGCNConfig.TARGET_MODEL_FPS)
        if not self._plugin_source_fps or self._plugin_source_fps <= 1.0:
            self._plugin_source_fps = float(STGCNConfig.TARGET_MODEL_FPS)
        self._plugin_video_id = os.path.basename(str(video_path)) or "video"
        self._plugin_session_id = "S{}_{}".format(
            time.strftime("%Y%m%d_%H%M%S"), os.getpid()
        )
        self._plugin_sessions_started = False
        self._plugin_sessions_finished = False
        # 關鍵點 EMA：用於 overlay 顯示與異常偵測（不進入 ST-GCN buffer）
        self.kp_ema_alpha = kp_ema_alpha
        self._ema_kpts = None
        self._plugins: list = list(plugins) if plugins else []
        # M4：舔毛/身體分區外掛共用的姿態濾波器——取代 lick_stage/analyzer.py
        # 自己的樸素 EMA，讓兩個外掛吃同一份平滑後的關鍵點與骨長品質分數
        # （見 _notify_plugins()）。跟上面 self._ema_kpts（overlay 顯示/異常
        # 偵測用）完全獨立，互不影響。hold_decay_frames 用「秒數 ×
        # 本次來源實際 fps」換算，維持跟 GAP_TOLERANCE_SEC 等既有設定一樣
        # 以秒為單位思考、但套用在以幀為單位運作的濾波器上。
        self._pose_filter = None
        if _PoseFilter is not None and _LickConfig is not None:
            try:
                _hold_decay_frames = max(
                    0,
                    round(
                        _LickConfig.POSE_FILTER_HOLD_DECAY_SEC
                        * self._plugin_source_fps
                    ),
                )
                self._pose_filter = _PoseFilter(
                    alpha=_LickConfig.POSE_FILTER_ALPHA,
                    min_conf=_LickConfig.POSE_FILTER_MIN_CONF,
                    hold_decay_frames=_hold_decay_frames,
                    bone_pairs=EAR_DISTANCE_SKELETON_EDGES,
                )
            except Exception:
                self._pose_filter = None
        # 保存上次推論結果，非推論幀沿用，避免標籤閃爍
        self._last_behavior_id = LOW_CONF_ID
        self._last_confidence = 0.0
        self._last_class_probs = [0.0] * STGCNConfig.NUM_CLASSES

        # 貓咪偵測消失容忍：YOLO 連續漏偵測沒超過 CAT_MISSING_TOLERANCE_FRAMES
        # 前，沿用最後一次偵測到的姿態，避免單幀漏偵測就整個中斷分類/顯示
        # （與 1_run_video_inference.py 共用同一份 config.BehaviorTrackingConfig）
        self._cat_missing_streak = 0
        self._last_known_kpts = None
        self._last_known_kpt_conf = None
        self._last_known_bbox = None
        self._last_known_bbox_conf = None
        # 多貓同框 + 身分驗證時，上一幀鎖定的「目標貓」bbox；用來在多個實例都
        # 像目標貓時挑最接近的那個，避免在長相相近的貓之間跳（見
        # _select_target_instance()）。
        self._last_target_bbox = None

        # 顯示層 hysteresis：overlay/Node-RED「目前行為」需連續多個分類視窗判
        # 同一類才切換，過濾單一視窗瞬間誤判造成的畫面閃爍；tracker/CSV/
        # segment_logger 一律使用未經此處理的 self._last_behavior_id 等即時
        # 結果，統計/歷史資料不受影響（與測試腳本相同設計）
        self._display_behavior_id = LOW_CONF_ID
        self._display_confidence = 0.0
        self._display_class_probs = [0.0] * STGCNConfig.NUM_CLASSES
        self._hysteresis_candidate_id = LOW_CONF_ID
        self._hysteresis_candidate_streak = 0

    @property
    def source_fps(self) -> float:
        """影片來源 FPS；無效時回傳 TARGET_MODEL_FPS。"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 1 else STGCNConfig.TARGET_MODEL_FPS

    def is_stream_source(self) -> bool:
        """是否為即時串流來源（RTSP/HLS 等，沒有「結尾」概念）。
        本機檔案來源回傳 False——播完就是播完，不循環（見 read_raw_frame()）。"""
        return self._grabber is not None

    def read_raw_frame(self):
        """讀取下一幀。
        即時串流來源：由背景執行緒（_LatestFrameGrabber）持續抽乾緩衝區，
        這裡只取最新一幀，不做 loop/seek（串流沒有「結尾」的概念）。

        本機檔案來源：影片播完（cap.read() 回傳 False）就結束，不管有沒有設定
        排程都不會自動循環回開頭——排程只決定「這段時間該不該處理」，不代表
        影片本身要被撐長；影片多長，統計就收多長，播完之後 ret=False 會一路
        往上傳給 SharedFrameStreamer，由它的 finished 旗標正確地停下來
        （見 server/streaming.py），不會無限重播、讓同一段素材的統計無限疊加。
        Returns: (ret: bool, frame | None)
        """
        if self._grabber is not None:
            return self._grabber.read()
        return self.cap.read()

    def get_behavior_history_records(self, limit: int = 200) -> list:
        """回傳最近 limit 筆行為紀錄（原始 dict，由呼叫方決定格式化）。"""
        return list(self.tracker.behavior_history)[-limit:]

    def process(self, frame):
        """處理單一影格：偵測、分類、追蹤、疊圖、記錄、推送，回傳疊圖後的畫面。"""
        self.frame_idx += 1
        current_time = time.time()
        self.prev_time = current_time

        if self._multi_cat:
            kpts, kpt_conf, bbox, conf, all_instances = self.keypoint_detector.detect(
                frame, return_all_instances=True
            )
        else:
            # 單貓模式：刻意沿用合併多貓/身分驗證功能「之前」就有的呼叫方式
            # （不傳 single_cat，YOLO 用預設 max_det、照舊做跨幀 IoU 追蹤延續），
            # 不碰多貓挑選 / 畫非目標貓 / 身分驗證。這是有意的選擇：SYSTEM_MODE
            # 預設就是 single，如果連這裡的偵測呼叫方式都跟著換成 single_cat=True
            # （max_det=1、不追蹤延續），會讓完全沒用到多貓/身分驗證功能的既有
            # 單貓部署預設行為也跟著悄悄改變——這點被
            # test_frame_processor_characterization.py 的凍結快照測試抓到過
            # （同一支影片同一組預設設定，activity_value 第 1 幀從 24.0 變 36.0），
            # 使用者確認要保留原本行為，所以刻意不用 single_cat 參數。
            kpts, kpt_conf, bbox, conf = self.keypoint_detector.detect(frame)
            all_instances = None

        # ── 多貓同框：決定「目標貓」是哪個實例，其餘放 other_instances 畫灰框 ──
        # 目標貓＝唯一進入行為分類 / tracker / CSV / Node-RED / 個體化基線的那隻。
        #   - 身分驗證開啟：見 _select_target_instance()——追蹤中用 bbox IoU 延續
        #     同一隻貓（不管 CNN 這一幀信心），只有「該位置附近沒有貓」或「CNN
        #     連續 hyst 幀明確判為他貓」才放掉；分不清一律當目標貓。
        #   - 身分驗證關閉：沿用 detect() 用信心 / IoU 追蹤選出的 primary。
        # identity_filtered_now：身分驗證開啟、但這一幀找不到目標貓 → 走
        #   NOT_VISIBLE 統計路徑，且下面的消失容忍不沿用舊姿態。
        other_instances = []
        identity_filtered_now = False
        if all_instances and self.identity_verifier is not None:
            try:
                (
                    kpts,
                    kpt_conf,
                    bbox,
                    conf,
                    other_instances,
                ) = self._select_target_instance(frame, all_instances)
                identity_filtered_now = kpts is None
            except Exception:
                # fail-safe：挑選出錯不擋掉這一幀，回退成 detect() 的 primary
                other_instances = []
        elif all_instances:
            # 身分驗證關閉：primary（detect 已用信心/IoU 選好）進統計，其餘畫灰框
            other_instances = [
                inst
                for inst in all_instances
                if inst[2] is None
                or bbox is None
                or not np.array_equal(inst[2], bbox)
            ]

        # 非目標貓：灰框 + 灰骨架畫出來（只畫、不進任何統計）。畫在目標貓 overlay
        # 之前，讓目標貓的青框骨架蓋在最上層。
        if (
            self.overlay
            and VisualizationConfig.SHOW_NON_TARGET_CATS
            and other_instances
        ):
            for _oki, _okci, _obi, _obci in other_instances:
                if _oki is None:
                    continue
                frame = self.visualizer.draw(
                    frame, _oki, _okci, _obi, _obci,
                    NOT_VISIBLE_ID, 0.0, [0.0] * STGCNConfig.NUM_CLASSES,
                    show_skeleton=self.show_skeleton,
                    show_info=False,
                    show_bbox=self.show_bbox,
                    bbox_color=COLOR_BBOX_NONTARGET,
                    skeleton_color=COLOR_BBOX_NONTARGET,
                    draw_face_overlay=False,
                )

        # 貓咪偵測消失容忍：連續漏偵測沒超過門檻前，沿用最後一次偵測到的姿態，
        # 避免單幀 YOLO 漏偵測就整個中斷分類/顯示（見 config.py 說明）。
        # identity_filtered_now 時不沿用：畫面裡確定是另一隻貓，不該拿目標貓的
        # 舊姿態來橋接。
        if kpts is not None:
            self._cat_missing_streak = 0
            self._last_known_kpts = kpts.copy()
            self._last_known_kpt_conf = kpt_conf.copy()
            self._last_known_bbox = bbox
            self._last_known_bbox_conf = conf
        elif (
            not identity_filtered_now
            and self._cat_missing_streak
            < BehaviorTrackingConfig.CAT_MISSING_TOLERANCE_FRAMES
            and self._last_known_kpts is not None
        ):
            self._cat_missing_streak += 1
            kpts, kpt_conf = self._last_known_kpts, self._last_known_kpt_conf
            bbox, conf = self._last_known_bbox, self._last_known_bbox_conf

        # 沿用上次推論結果；僅在本幀推論成功時更新
        behavior_id = self._last_behavior_id
        confidence = self._last_confidence
        class_probs = self._last_class_probs
        is_still, activity_value = False, 0.0

        if kpts is not None:
            raw_kpts = kpts.copy()

            # === Plugin notification (raw keypoints, before any smoothing) ===
            # 舔舐二階段（plugins/lick_stage）只在 ST-GCN 目前已確認判定為 lick
            # 時才餵入真實關鍵點；否則只傳契約欄位（kpts=None），讓 plugin 走既有
            # 的 NOT_LICK / 重置路徑——不把 walk/scratch 等非舔舐期間的時間誤計進
            # 某個部位的理毛時長，同時重置梯形方向平滑等跨幀狀態。
            #
            # 第一階段（說明書「來源時間規格」）：改以來源媒體時間
            # source_timestamp = frame_idx / source_fps 驅動 plugin 內部計時，
            # 不再讓 plugin 用 time.monotonic()——確保同一支影片以不同處理速度
            # 得到一致的累積舔毛時長；並帶上 cat_present / is_lick，讓 plugin 能
            # 區分「沒有貓」與「有貓但非舔毛」。
            is_lick_behavior = (
                behavior_id == _LICK_BEHAVIOR_ID
                and confidence
                >= BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
            )
            self._notify_plugins(
                raw_kpts,
                kpt_conf,
                cat_present=True,
                is_lick=is_lick_behavior,
                lick_confidence=float(confidence),
            )

            # === Frame-level EMA：僅用於 overlay 顯示與異常偵測，原始 raw_kpts 進 ST-GCN buffer ===
            # 注意：此 EMA 不影響 STGCN 推論路徑；ST-GCN 輸入的唯一平滑來源是下方 window-level EMA
            if self._ema_kpts is None:
                self._ema_kpts = raw_kpts.copy()
            else:
                self._ema_kpts = (
                    self.kp_ema_alpha * raw_kpts
                    + (1.0 - self.kp_ema_alpha) * self._ema_kpts
                )
            display_kpts = self._ema_kpts.copy()

            # === 靜止偵測（使用 EMA 平滑後的關鍵點） ===
            is_still, activity_value = self.anomaly_detector.detect(
                display_kpts, kpt_conf
            )

            # === ST-GCN 行為推論（buffer 儲存 raw kpts，與訓練前處理順序一致） ===
            self.keypoints_buffer.append((raw_kpts, kpt_conf))
            self._infer_frame_count += 1
            should_infer = len(self.keypoints_buffer) >= self.sequence_length and (
                self._infer_frame_count % max(1, self.window_stride) == 0
            )
            if should_infer:
                kpts_arr = np.array(
                    [item[0] for item in self.keypoints_buffer]
                )  # (T, 17, 2)
                conf_arr = np.array(
                    [item[1] for item in self.keypoints_buffer]
                )  # (T, 17)
                seq_array = interpolate_missing(kpts_arr, conf_arr)
                # Window-level EMA：STGCN 輸入的唯一平滑步驟，須與訓練時使用的 KP_EMA_ALPHA 一致
                # alpha=1.0（預設）表示不平滑；調低時須確認訓練也用相同數值，切勿在此之外另加平滑
                if self.kp_ema_alpha < 1.0:
                    for t in range(1, seq_array.shape[0]):
                        seq_array[t] = (
                            self.kp_ema_alpha * seq_array[t]
                            + (1.0 - self.kp_ema_alpha) * seq_array[t - 1]
                        )
                new_bid, new_conf, new_probs = self.behavior_classifier.classify(
                    seq_array, conf_arr
                )
                if new_bid is None:
                    new_bid = LOW_CONF_ID
                    new_conf = 0.0
                elif (
                    new_conf
                    < BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
                ):
                    new_bid = LOW_CONF_ID

                # === Skeleton Quality Assessment（GCN 分類為主、幾何判斷為輔）===
                # 用跟 ST-GCN 同一個窗口的「原始（未插值）」關鍵點座標
                # kpts_arr/conf_arr（不是上面已經插值過、給 ST-GCN 用的
                # seq_array——evaluate_window 內部會自己做一次插值，避免
                # 跟診斷腳本 test_bone_length_stability.py 的前處理路徑不一致）。
                # 獨立模組、獨立總開關（SQAConfig.ENABLE_SQA_DUAL_JUDGMENT），
                # evaluate_window() 本身承諾不拋例外，這裡再包一層 try/except
                # 是防禦性寫法（跟現有 plugin 呼叫慣例一致）——任何錯誤都只會
                # 讓這次覆蓋不生效，不會中斷 process()。
                if (
                    SQAConfig.ENABLE_SQA_DUAL_JUDGMENT
                    and _sqa_evaluate_window is not None
                ):
                    try:
                        _sqa_reliable, _sqa_details = _sqa_evaluate_window(
                            kpts_arr, conf_arr
                        )
                        if not _sqa_reliable:
                            new_bid = LOW_CONF_ID
                            new_conf = 0.0
                    except Exception:
                        pass

                # 更新持久化結果，本幀也立即採用
                self._last_behavior_id = new_bid
                self._last_confidence = new_conf
                self._last_class_probs = (
                    new_probs
                    if new_probs is not None
                    else [0.0] * STGCNConfig.NUM_CLASSES
                )
                behavior_id = self._last_behavior_id
                confidence = self._last_confidence
                class_probs = self._last_class_probs
                self._update_display_hysteresis(behavior_id, confidence, class_probs)
            # 以 display_kpts 替換後續用到 kpts 的位置
            kpts = display_kpts

            # === 行為追蹤 ===
            len_before = len(self.tracker.behavior_history)
            self.tracker.update(behavior_id, activity_value)
            if self.segment_logger and len(self.tracker.behavior_history) > len_before:
                rec = list(self.tracker.behavior_history)[-1]
                if rec["gcn_behavior_id"] != LOW_CONF_ID:
                    self.segment_logger.log_segment(
                        rec["gcn_behavior_id"],
                        BEHAVIOR_TEXT_MAP.get(rec["gcn_behavior_id"], rec["behavior"]),
                        rec["duration"],
                        rec.get("activity", 0),
                    )

            # === Node-RED 資料推送（顯示用「目前行為」走 hysteresis 後的結果，
            # today_stats/behavior_log 等統計仍在 tracker 內部用未經處理的即時結果累積）===
            now = time.time()
            if self.nodered and (
                now - self.last_send_time >= NodeRedConfig.PUSH_INTERVAL
            ):
                self.tracker.add_monitoring_seconds(now - self.last_send_time)
                self.nodered.send_data(
                    self._build_nodered_payload(
                        self._display_behavior_id, self._display_confidence
                    )
                )
                self.last_send_time = now

            # === CSV 日誌 ===
            # CSV 日誌只在貓咪活動中（非靜止）且行為信心足夠時寫入
            if (
                self.csv_logger
                and not is_still
                and float(confidence)
                >= BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
                and behavior_id != LOW_CONF_ID
            ):
                behavior_name = get_behavior_name(
                    behavior_id, use_text=False, fallback="未知", confidence=confidence
                )
                self.csv_logger.log(
                    self.frame_idx,
                    behavior_name,
                    confidence,
                    is_still,
                    self.anomaly_detector.last_motion_score,
                )

            # === Overlay 畫圖（走 hysteresis 後的顯示結果，避免單一視窗誤判閃爍）===
            if self.overlay:
                frame = self.visualizer.draw(
                    frame,
                    kpts,
                    kpt_conf,
                    bbox,
                    conf,
                    self._display_behavior_id,
                    self._display_confidence,
                    self._display_class_probs,
                    show_skeleton=self.show_skeleton,
                    show_info=self.show_label,
                    show_bbox=self.show_bbox,
                )
                # Plugin overlays（e.g. lick-stage nose trapezoid）
                _show_trap = VisualizationConfig.SHOW_NOSE_TRAPEZOID
                for _plugin in self._plugins:
                    if hasattr(_plugin, "draw_overlay"):
                        _plugin.draw_overlay(frame, self.frame_idx, show=_show_trap)

        else:
            # === Plugin notification (no cat detected) ===
            self._notify_plugins(
                None,
                None,
                cat_present=False,
                is_lick=False,
                lick_confidence=0.0,
            )

            # 超過消失容忍門檻，才真的視為貓消失：重置 EMA、推論計數器、keypoints
            # buffer 與上次推論結果。_infer_frame_count 重置確保貓重新出現後推論
            # 時機從 0 對齊，不受之前計數影響；keypoints_buffer 清除確保舊幀不污染
            # 下次推論窗口
            self._ema_kpts = None
            self._infer_frame_count = 0
            self.keypoints_buffer.clear()
            self._last_behavior_id = LOW_CONF_ID
            self._last_confidence = 0.0
            self._last_class_probs = [0.0] * STGCNConfig.NUM_CLASSES
            # 同步更新本幀的區域變數，否則本幀回傳的 behavior_id/confidence 仍
            # 沿用上一幀（cat1 還在畫面時）的結果，慢一幀才變成「未偵測到」
            behavior_id = self._last_behavior_id
            confidence = self._last_confidence
            class_probs = self._last_class_probs
            # 顯示層立即切換為「不在畫面」，不套用 hysteresis 延遲
            self._update_display_hysteresis(
                NOT_VISIBLE_ID, 0.0, [0.0] * STGCNConfig.NUM_CLASSES
            )
            # 靜止偵測也走同一支介面：AnomalyDetector.detect(None, None) 內部有
            # 自己的短暫遺失容忍（_MAX_MISS_FRAMES），讓它接手判斷是否仍視為靜止，
            # 而不是在這裡硬寫死 False/0（此門檻與上面的貓消失容忍各自獨立管理）
            is_still, activity_value = self.anomaly_detector.detect(None, None)
            self.tracker.update(NOT_VISIBLE_ID, 0.0)
            # Node-RED 推送：通知貓咪不在畫面
            now = time.time()
            if self.nodered and (
                now - self.last_send_time >= NodeRedConfig.PUSH_INTERVAL
            ):
                self.nodered.send_data(self._build_nodered_payload(NOT_VISIBLE_ID, 0.0))
                self.last_send_time = now

        return (
            frame,
            self._display_behavior_id,
            self._display_confidence,
            self._display_class_probs,
            is_still,
            activity_value,
        )

    def register_plugin(self, plugin) -> None:
        """Register an optional plugin. Called before the first frame."""
        self._plugins.append(plugin)

    def _ensure_plugin_sessions(self) -> None:
        """首次處理幀時，替支援 Session 生命週期的外掛開場（說明書第一階段
        「Session 生命週期」）。不支援 start_session 的舊外掛自動略過。"""
        if self._plugin_sessions_started:
            return
        self._plugin_sessions_started = True
        for _plugin in self._plugins:
            start = getattr(_plugin, "start_session", None)
            if start is None:
                continue
            try:
                start(
                    self._plugin_session_id,
                    video_id=self._plugin_video_id,
                    model_version=getattr(STGCNConfig, "MODEL_TAG", "stgcn"),
                    source_fps=self._plugin_source_fps,
                )
            except Exception:
                pass

    def finish_plugin_sessions(self, end_source_timestamp=None) -> None:
        """來源播畢 / 管線關閉時呼叫：讓外掛結算最後一段未結束的 bout。"""
        if self._plugin_sessions_finished:
            return
        self._plugin_sessions_finished = True
        for _plugin in self._plugins:
            finish = getattr(_plugin, "finish_session", None)
            if finish is None:
                continue
            try:
                finish(end_source_timestamp)
            except Exception:
                pass

    def _current_source_timestamp(self) -> float:
        """本幀的來源媒體時間（秒），跨處理速度不變（說明書「來源時間規格」）。

        優先用 VideoCapture 的 CAP_PROP_POS_MSEC —— 這是解碼器回報的實際 PTS，
        即使串流層做了抽幀（frame_step > 1）也正確。取不到（多數即時串流回
        0 或不支援）才退回 frame_idx / source_fps。
        """
        try:
            pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            if pos_ms > 0.0:
                return pos_ms / 1000.0
        except Exception:
            pass
        return self.frame_idx / self._plugin_source_fps

    def _notify_plugins(self, kpts, kpt_conf, *, cat_present, is_lick, lick_confidence):
        """以第一階段契約參數通知所有外掛。kpts 只在 is_lick 時才餵真實值，
        其餘情況傳 None（維持既有「不污染統計」的行為），但 cat_present /
        is_lick / source_timestamp 一律帶上，讓外掛能區分 NO_CAT 與 NOT_LICK。"""
        self._ensure_plugin_sessions()
        source_ts = self._current_source_timestamp()
        feed_kpts = kpts if is_lick else None
        feed_conf = kpt_conf if is_lick else None

        # M4：共用 PoseFilter 一次算出平滑後的關鍵點＋骨長品質，兩個外掛
        # 吃同一份結果。feed_kpts 為 None（NO_CAT/NOT_LICK/lick 但無姿態）時
        # 呼叫 reset()，跟 lick_stage/analyzer.py 既有的
        # _reset_transient_state() 在同一組條件下清空跨幀平滑狀態，維持
        # 兩邊政策一致（見 analyzer.py 該函式的說明）。
        pose_quality = None
        if self._pose_filter is not None:
            if feed_kpts is None:
                self._pose_filter.reset()
            else:
                try:
                    pose_frame = self._pose_filter.update(
                        feed_kpts,
                        feed_conf,
                        frame_idx=self.frame_idx,
                        source_timestamp=source_ts,
                    )
                    feed_kpts = pose_frame.smoothed_kpts
                    pose_quality = pose_frame.quality
                except Exception:
                    pass

        base_kwargs = dict(
            source_timestamp=source_ts,
            frame_idx=self.frame_idx,
            cat_present=cat_present,
            is_lick=is_lick,
            lick_confidence=lick_confidence,
            session_id=self._plugin_session_id,
            pose_quality=pose_quality,
        )

        # M6（統一 ontology 融合）：ext_body_zones 的分類結果要在 lick_stage
        # 的 update() 之前先算好，才能當補充欄位一起餵過去（見下方）。用
        # isinstance 找出（若有）已註冊的 ext_body_zones 實例，不依賴
        # self._plugins 的註冊順序——不同呼叫端的註冊順序不保證一致（例如
        # tools/verify_lick_stage_m2.py 先註冊 lick_stage 再註冊
        # ext_body_zones）。ext_body_zones 本身完全不知道、也不依賴
        # lick_stage 的存在，這裡的耦合僅限於 frame_processor.py 這個組裝層
        # （見檔案開頭 import guard 的說明）。
        ext_zone_name = None
        ext_zone_confidence = None
        ext_plugin = None
        if _ExtBodyZonePlugin is not None:
            for _plugin in self._plugins:
                if isinstance(_plugin, _ExtBodyZonePlugin):
                    ext_plugin = _plugin
                    break
        if ext_plugin is not None:
            self._call_plugin_update(ext_plugin, feed_kpts, feed_conf, base_kwargs)
            _name = getattr(ext_plugin, "last_zone_name", None)
            _no_target = (
                _ExtZoneConfig.ZONE_NAMES[_ExtZoneConfig.ZONE_NO_TARGET]
                if _ExtZoneConfig is not None
                else "NO_TARGET"
            )
            if _name is not None and _name != _no_target:
                # 只有 ext_body_zones 真的給出具體 zone 時才當成補充欄位轉
                # 餵給 lick_stage；NO_TARGET（沒有分類結果）跟「完全沒有
                # ext_body_zones 這個外掛」在這裡是同一種語意——都是 None，
                # 呼叫端（bout_aggregator.py）已經是「None 就不計入」的既有
                # 慣例（跟 pose_quality/action_score 一致）。confidence 在
                # NO_TARGET 時是哨兵值 0.0（見 regions.py::classify_zone()），
                # 不是真正的信心值，一併捨棄，避免拉低 ext_zone_confidence_mean。
                ext_zone_name = _name
                ext_zone_confidence = getattr(ext_plugin, "last_confidence", None)

        for _plugin in self._plugins:
            if _plugin is ext_plugin:
                continue  # 上面已經呼叫過
            kwargs = base_kwargs
            if _LickStagePlugin is not None and isinstance(_plugin, _LickStagePlugin):
                kwargs = dict(
                    base_kwargs,
                    ext_zone_name=ext_zone_name,
                    ext_zone_confidence=ext_zone_confidence,
                )
            self._call_plugin_update(_plugin, feed_kpts, feed_conf, kwargs)

    @staticmethod
    def _call_plugin_update(plugin, feed_kpts, feed_conf, kwargs) -> None:
        """單一外掛的 fail-safe update() 呼叫，含舊版兩參數呼叫的退回路徑
        （從 _notify_plugins() 抽出，M6 融合後要對兩個具體外掛分別組不同
        的 kwargs，原本的單一迴圈不夠用，見呼叫端）。"""
        try:
            plugin.update(feed_kpts, feed_conf, **kwargs)
        except TypeError:
            # 舊版外掛只接受 update(kpts, kpt_conf) —— 退回舊呼叫方式
            try:
                plugin.update(feed_kpts, feed_conf)
            except Exception:
                pass
        except Exception:
            pass

    # 追蹤中：這一幀某個實例的 bbox 與「上一幀鎖定的目標貓 bbox」IoU 需 ≥ 此值，
    # 才算是同一隻貓的空間延續。低於此＝目標貓已離開原本位置（見 _select_target_instance
    # 的「案 A」）。
    _TARGET_TRACK_IOU_MIN = 0.1

    def _select_target_instance(self, frame, all_instances):
        """身分驗證開啟時，從這一幀所有偵測到的貓裡挑出「目標貓」的實例。

        回傳 (kpts, kpt_conf, bbox, bbox_conf, other_instances)：
          - 確立目標貓：前 4 個是該實例，other_instances 是其餘所有實例
          - 沒有目標貓：前 4 個為 None，other_instances 是全部實例——呼叫端走
            NOT_VISIBLE 統計路徑，但仍把每隻畫成灰框。

        核心規則（身分驗證存在的意義）：CNN 平滑後沒有「明確」判定為目標貓
        （verify() 回傳 is_target_cat=False——不論是明確判為別隻貓、還是
        信心不足的「分不清/未知」）一律視為這一幀沒有目標貓，不計入統計。
        不像舊版把「分不清」也當目標貓接受，這裡刻意不留模糊地帶：身分
        驗證的價值就在於「沒把握就不算」，寧可少算幾幀，不要讓誤判或
        另一隻貓的資料污染個體化基線。

        位置追蹤鎖定（_last_target_bbox）是獨立於上面那條規則的另一層
        機制，只管「接下來要盯著畫面哪個位置看」：候選貓一旦連續 hyst
        （IDENTITY_FILTER_HYSTERESIS_FRAMES）幀身分都沒過，才真正放掉
        鎖定、下次改用信心排序重新挑；期間即使某幀因為身分不明確沒被
        計入統計，只要附近還找得到候選貓，位置鎖定仍會跟著更新，短暫
        的判斷不確定不會馬上丟失追蹤。若目標貓的位置附近直接找不到任何
        候選貓（IoU 延續不上），則視為牠已經離開畫面，不等遲滯立即放掉。
        """
        verifier = self.identity_verifier
        hyst = max(1, CatIdentityConfig.IDENTITY_FILTER_HYSTERESIS_FRAMES)
        none_target = (None, None, None, None, list(all_instances))

        cold_start = self._last_target_bbox is None
        if not cold_start:
            ious = [
                KeypointDetector._iou(self._last_target_bbox, inst[2])
                if inst[2] is not None
                else 0.0
                for inst in all_instances
            ]
            if ious and max(ious) >= self._TARGET_TRACK_IOU_MIN:
                guess_i = int(np.argmax(ious))
            else:
                # 上一幀目標貓位置附近已經沒有貓 → 目標貓離開畫面，立即
                # 停止計入（不等遲滯）。舊 bbox 作廢，之後要重新確立。
                self._identity_nontarget_streak = 0
                self._last_target_bbox = None
                if not self._identity_filtering_active:
                    self._identity_filtering_active = True
                    print("🚫 身分驗證：目標貓已離開畫面，本幀起從統計中過濾")
                return none_target
        else:
            # 冷啟動：用未平滑的單幀機率挑「最像目標貓」的候選（1 隻貓時
            # 就是它自己），再交給下面的 verify() 做跨幀平滑做真正的判定；
            # 這裡選中不代表接受，純粹決定要對哪個 bbox 做身分判斷。
            scores = [
                (
                    verifier.target_probability(frame, inst[2])
                    if inst[2] is not None
                    else -1.0
                )
                for inst in all_instances
            ]
            scores = [s if s is not None else -1.0 for s in scores]
            guess_i = int(np.argmax(scores))

        # 挑到的實例做跨幀平滑（維持 verify() 一幀一次呼叫的契約）
        try:
            is_target_cat, match_key, _s = verifier.verify(
                frame, all_instances[guess_i][2]
            )
        except Exception:
            # 判斷本身出錯＝沒把握，比照「分不清」處理，不貿然接受為目標貓
            is_target_cat, match_key = False, None

        if not is_target_cat:
            self._identity_nontarget_streak = min(
                self._identity_nontarget_streak + 1, hyst
            )
            if self._identity_nontarget_streak >= hyst:
                self._last_target_bbox = None
            else:
                # 身分還沒過，但位置鎖定先跟著更新，避免貓移動時單純因為
                # bbox 沒跟上而誤判成「已離開畫面」（見上方 docstring）。
                self._last_target_bbox = all_instances[guess_i][2]
            if not self._identity_filtering_active:
                self._identity_filtering_active = True
                print(
                    f"🚫 身分驗證：CNN 未明確判定為「{CatIdentityConfig.TARGET_CAT_CLASS}」"
                    f"（本幀判定：{match_key or '分不清/未知'}），本幀起從統計中過濾"
                )
            return none_target

        self._identity_nontarget_streak = 0
        if self._identity_filtering_active:
            self._identity_filtering_active = False
            print("✓ 身分驗證：CNN 明確判定為目標貓，恢復計入統計")

        tgt = all_instances[guess_i]
        self._last_target_bbox = tgt[2]
        others = [inst for j, inst in enumerate(all_instances) if j != guess_i]
        return tgt[0], tgt[1], tgt[2], tgt[3], others

    def _update_display_hysteresis(
        self, candidate_id, candidate_confidence, candidate_probs
    ):
        """依候選類別各自的門檻（BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS[class_id]），
        連續達到該次數的分類視窗判同一類，才真的切換 overlay/Node-RED 顯示用的行為標籤，
        用來過濾單一視窗瞬間誤判（例如動作轉換瞬間）造成的畫面閃爍。tracker/CSV/
        segment_logger 走 self._last_behavior_id 等未經處理的即時結果，不受影響。
        candidate_id 為 LOW_CONF_ID/NOT_VISIBLE_ID 時立即顯示，不套用延遲。"""
        if candidate_id in (LOW_CONF_ID, NOT_VISIBLE_ID):
            threshold = 1
        else:
            threshold = BehaviorTrackingConfig.DISPLAY_HYSTERESIS_WINDOWS.get(
                candidate_id, 1
            )

        if threshold <= 1 or candidate_id in (LOW_CONF_ID, NOT_VISIBLE_ID):
            self._display_behavior_id = candidate_id
            self._display_confidence = candidate_confidence
            self._display_class_probs = candidate_probs
            self._hysteresis_candidate_id = LOW_CONF_ID
            self._hysteresis_candidate_streak = 0
            return

        if candidate_id == self._hysteresis_candidate_id:
            self._hysteresis_candidate_streak += 1
        else:
            self._hysteresis_candidate_id = candidate_id
            self._hysteresis_candidate_streak = 1

        if self._hysteresis_candidate_streak >= threshold:
            self._display_behavior_id = candidate_id
            self._display_confidence = candidate_confidence
            self._display_class_probs = candidate_probs
        # 未達門檻前維持前一次已確定顯示的類別（self._display_behavior_id 不變）

    def _build_nodered_payload(self, behavior_id, confidence) -> dict:
        """組裝 Node-RED 推送資料，貓咪在畫面與不在畫面共用此方法。"""
        if behavior_id == NOT_VISIBLE_ID:
            current = {
                "behavior_id": NOT_VISIBLE_ID,
                "text": NOT_VISIBLE_DISPLAY_TEXT,
                "behavior": NOT_VISIBLE_TEXT,
                "emoji": NOT_VISIBLE_EMOJI,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            gcn_confidence = 0.0
        else:
            is_low_conf = (behavior_id == LOW_CONF_ID) or (
                float(confidence)
                < BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
            )
            if is_low_conf:
                current = {
                    "behavior_id": int(behavior_id),
                    "text": LOW_CONF_TEXT,
                    "behavior": LOW_CONF_TEXT,
                    "emoji": LOW_CONF_EMOJI,
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            else:
                current = {
                    "behavior_id": int(behavior_id),
                    "text": BEHAVIOR_TEXT_MAP.get(behavior_id, "未知"),
                    "behavior": (
                        BEHAVIOR_CLASSES[int(behavior_id)]
                        if 0 <= int(behavior_id) < len(BEHAVIOR_CLASSES)
                        else "unknown"
                    ),
                    "emoji": BEHAVIOR_EMOJI_MAP.get(behavior_id, "❓"),
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            gcn_confidence = round(float(confidence), 3)

        return {
            "current": current,
            "activity_score": int(self.tracker.get_activity_score()),
            "today_stats": self.tracker.get_today_stats(),
            "behavior_log": [
                {
                    "behavior": rec["behavior"],
                    "gcn_id": rec["gcn_behavior_id"],
                    "time": (
                        rec["timestamp"].strftime("%H:%M:%S")
                        if hasattr(rec["timestamp"], "strftime")
                        else str(rec["timestamp"])
                    ),
                    "duration": rec["duration"],
                }
                for rec in list(self.tracker.behavior_history)[-10:]
            ],
            "alerts": self.tracker.get_alerts(),
            "system": {
                "ip": self.local_ip,
                "model": "YOLO-Pose + ST-GCN",
                "version": SystemInfo.VERSION,
                "gcn_confidence": gcn_confidence,
            },
        }

    def cleanup(self):
        """釋放攝影機/串流資源，關閉 CSV 記錄器與 Node-RED 連線、通知所有插件關閉。"""
        if self._grabber is not None:
            self._grabber.stop()
        self.cap.release()
        if self.csv_logger:
            self.csv_logger.close()
        if self.segment_logger:
            self.segment_logger.close()
        if self.nodered:
            self.nodered.close()
        for _plugin in self._plugins:
            try:
                _plugin.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
