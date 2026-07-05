"""
工具函數
"""
import re
import socket

from utils.constants import BEHAVIOR_CLASSES, BEHAVIOR_TEXT_MAP, LOW_CONF_ID, LOW_CONF_TEXT
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

_YOUTUBE_URL_RE = re.compile(
    r'^https?://(www\.)?(youtube\.com/(watch\?|live/)|youtu\.be/)', re.IGNORECASE
)
_STREAM_URL_RE = re.compile(r'^(https?|rtsp|rtsps|rtmp)://', re.IGNORECASE)


def is_stream_url(video_path) -> bool:
    """是否為即時網路串流來源（HTTP(S)/RTSP/RTMP），而非本機檔案或攝影機 index。
    用來決定 FrameProcessor 要不要啟用「只保留最新幀」的背景讀取機制。
    """
    return isinstance(video_path, str) and bool(_STREAM_URL_RE.match(video_path))


def resolve_video_source(video_path):
    """cv2.VideoCapture 無法直接開啟 YouTube 網頁網址（watch?v=... / youtu.be/...），
    需要先用 yt_dlp 解析出實際可讀取的串流網址。非 YouTube 網址（本機檔案、
    攝影機 index、RTSP 等）原樣傳回，不受影響。

    Raises:
        RuntimeError: 解析失敗時（例如影片下架、網路不通），給出比
                      「Cannot open video source」更明確的錯誤原因。
    """
    if not isinstance(video_path, str) or not _YOUTUBE_URL_RE.match(video_path):
        return video_path

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "偵測到 YouTube 網址，但未安裝 yt_dlp（pip install yt-dlp）"
        ) from e

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_path, download=False)
            return info["url"]
    except Exception as e:
        raise RuntimeError(f"無法解析 YouTube 串流網址 {video_path}：{e}") from e

def get_ip():
    # 優先取 WiFi 介面的 IP（當乙太網路與 WiFi 同時連線時，OS 路由預設走 Ethernet）
    _WIFI_KEYWORDS = ('wi-fi', 'wifi', 'wlan', 'wireless')
    try:
        import psutil
        for iface, addrs in psutil.net_if_addrs().items():
            if any(kw in iface.lower() for kw in _WIFI_KEYWORDS):
                for addr in addrs:
                    if addr.family == socket.AF_INET and not addr.address.startswith('169.254'):
                        return addr.address
    except Exception:
        pass

    # fallback：讓 OS 路由決定出口 IP（有線 + 無線同時連線時可能取到 Ethernet）
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def get_behavior_name(behavior_id, use_text=False, fallback="未知", confidence=None):
    """安全地把行為 ID 轉成顯示名稱。

    規則：
    - 若提供了 `confidence`，且 confidence < BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD → 回傳 LOW_CONF_TEXT。
    - 若 behavior_id == LOW_CONF_ID → 回傳 LOW_CONF_TEXT。
    - 否則在有效索引範圍回傳對應文字或名稱。
    """
    try:
        idx = int(behavior_id)
    except (TypeError, ValueError):
        return fallback

    # 以顯示層的最低信心作為優先判斷（若有提供 confidence）
    if confidence is not None:
        try:
            if float(confidence) < _BehaviorTrackingConfig.STGCN_BEHAVIOR_LABEL_CONFIDENCE_THRESHOLD:
                return LOW_CONF_TEXT
        except Exception:
            pass

    if idx == LOW_CONF_ID:
        return LOW_CONF_TEXT

    if 0 <= idx < len(BEHAVIOR_CLASSES):
        return BEHAVIOR_TEXT_MAP.get(idx, BEHAVIOR_CLASSES[idx]) if use_text else BEHAVIOR_CLASSES[idx]

    return fallback

