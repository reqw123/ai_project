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
    NodeRedConfig,
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
        if width and height:
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
        # 只在「開始過濾非目標貓」／「目標貓恢復」這兩個狀態轉換的瞬間印
        # console 訊息，不逐幀印——身分驗證是純統計過濾閘門，不會在 Node-RED
        # 串流畫面上多畫框，這是唯一能即時確認它有沒有在運作的方式。
        self._identity_filtering_active = False
        if CatIdentityConfig.ENABLE_IDENTITY_VERIFICATION and _IdentityVerifier is not None:
            try:
                self.identity_verifier = _IdentityVerifier(
                    model_path=CatIdentityConfig.IDENTITY_MODEL_PATH,
                    target_class=CatIdentityConfig.TARGET_CAT_CLASS,
                    device=device,
                )
                print(
                    f"✓ 身分驗證已啟用（CNN）：只有判定為「{CatIdentityConfig.TARGET_CAT_CLASS}」"
                    f"的偵測結果會計入統計"
                )
            except Exception as e:
                print(f"⚠ 身分驗證初始化失敗，已停用（偵測到的貓將一律視為目標貓）：{e}")
                self.identity_verifier = None

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
        # 關鍵點 EMA：用於 overlay 顯示與異常偵測（不進入 ST-GCN buffer）
        self.kp_ema_alpha = kp_ema_alpha
        self._ema_kpts = None
        self._plugins: list = list(plugins) if plugins else []
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

        kpts, kpt_conf, bbox, conf = self.keypoint_detector.detect(frame)

        # 身分驗證（多貓辨識）：只驗證這一幀 YOLO 剛偵測到的貓，不驗證下面
        # 消失容忍期間沿用的舊姿態（那是同一隻已經驗證過的貓在之前幀留下的
        # 姿態，沒有新 bbox 可以重新驗證，也不需要）。判定不是目標貓時，把
        # kpts 視為 None——後續完全比照「這一幀 YOLO 沒偵測到貓」處理，會
        # 走既有的貓咪消失容忍/NOT_VISIBLE 路徑，不產生行為紀錄、不計入
        # Node-RED 統計，等同把目標貓以外的貓完全忽略。fail-safe：驗證本身
        # 出錯不擋掉這一幀，回退成原本「偵測到的貓一律視為目標貓」的行為。
        if kpts is not None and self.identity_verifier is not None:
            try:
                is_target_cat, match_key, match_score = self.identity_verifier.verify(
                    frame, bbox
                )
            except Exception:
                is_target_cat, match_key, match_score = True, None, None

            # 純視覺提示，跟下面的統計判斷（kpts=None）互相獨立：非目標貓的
            # 情況下 Visualizer.draw() 完全不會被呼叫（kpts=None 會跳過整個
            # 疊圖區塊），沒有這個小徽章的話畫面上什麼都不會畫，看起來就像
            # 「什麼都沒偵測到」，沒辦法跟真的沒貓做視覺區分。
            if self.overlay:
                self._draw_identity_badge(frame, bbox, is_target_cat, match_key, match_score)

            if not is_target_cat:
                if not self._identity_filtering_active:
                    self._identity_filtering_active = True
                    _score_str = f"{match_score:.3f}" if match_score is not None else "N/A"
                    print(
                        f"🚫 身分驗證：畫面中的貓判定為「{match_key or '未知'}」"
                        f"（信心={_score_str}），非目標貓，本幀起已從統計中過濾"
                    )
                kpts = None
            elif self._identity_filtering_active:
                self._identity_filtering_active = False
                print("✓ 身分驗證：目標貓重新出現，恢復計入統計")

        # 貓咪偵測消失容忍：連續漏偵測沒超過門檻前，沿用最後一次偵測到的姿態，
        # 避免單幀 YOLO 漏偵測就整個中斷分類/顯示（見 config.py 說明）
        if kpts is not None:
            self._cat_missing_streak = 0
            self._last_known_kpts = kpts.copy()
            self._last_known_kpt_conf = kpt_conf.copy()
            self._last_known_bbox = bbox
            self._last_known_bbox_conf = conf
        elif (
            self._cat_missing_streak
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
            # 時才餵入真實關鍵點；否則比照「貓咪不在畫面」的方式傳 (None, None)，
            # 讓 plugin 內部走既有的 NO_TARGET 重置路徑——這樣 dt_sec 會正確算進
            # NO_TARGET，而不是把 walk/scratch 等非舔舐期間的時間誤計進某個部位
            # 的理毛時長，同時也會重置梯形方向平滑等跨幀狀態，避免用陳舊姿態接續。
            is_lick_behavior = (
                behavior_id == _LICK_BEHAVIOR_ID
                and confidence
                >= BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD
            )
            for _plugin in self._plugins:
                try:
                    if is_lick_behavior:
                        _plugin.update(raw_kpts, kpt_conf)
                    else:
                        _plugin.update(None, None)
                except Exception:
                    pass

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
            for _plugin in self._plugins:
                try:
                    _plugin.update(None, None)
                except Exception:
                    pass

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

    def mark_resumed(self) -> None:
        """在處理管線「從暫停恢復」的當下呼叫，重設所有以 wall-clock 差值累積
        時間的元件的時間錨點。

        背景：排程「區段執行」暫停期間（server/streaming.py 的 paused 迴圈、
        或 gui 模式空白鍵暫停）完全不呼叫 process()，因此 BehaviorTracker
        與各 plugin 內部以 `now - last_*_time` 計算的 dt 不會推進。恢復後的
        第一次 update() 若不重設錨點，會把「整段暫停時間」當成一個巨大 dt
        灌進當日統計（hourly_distribution / not_detected_time / 行為時長），
        產生時間尖峰污染基線。這支方法把錨點對齊到「現在」，讓恢復後第一幀
        的 dt 回到正常的單幀量級。

        fail-safe：plugin 沒有對應 hook 就跳過，不影響其餘恢復流程。
        """
        try:
            self.tracker.reset_time_anchor()
        except Exception:
            pass
        mono = time.monotonic()
        for _plugin in self._plugins:
            for _attr in ("_last_wall_t",):
                if hasattr(_plugin, _attr):
                    try:
                        setattr(_plugin, _attr, mono)
                    except Exception:
                        pass

    def _draw_identity_badge(self, frame, bbox, is_target, match_key, match_score):
        """身分驗證結果的獨立視覺提示，跟 Visualizer.draw() 完全分開畫。

        非目標貓的情況下 kpts 會被設成 None，process() 後段整個疊圖區塊
        （包含 Visualizer.draw()）都不會執行，畫面上等於「什麼都沒畫」，
        跟真的沒偵測到貓沒有視覺差異。這裡固定在 bbox 位置畫一個小徽章
        （目標貓=綠色/其他=橘色），兩種狀態都畫，只讀不改 kpts/bbox，
        不影響任何統計或分類邏輯。標籤用 ASCII（OpenCV putText 不支援中文，
        中文類別名寫死顯示成方框），中文全名走 console 訊息。"""
        if bbox is None:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return

        if is_target:
            color = (0, 200, 0)
            label = "ID: target"
        else:
            color = (0, 128, 255)
            who = "unknown" if match_key is None else "other-cat"
            label = f"ID: {who} (filtered)"
        if match_score is not None:
            label += f" conf={match_score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
        cv2.putText(
            frame, label, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

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
