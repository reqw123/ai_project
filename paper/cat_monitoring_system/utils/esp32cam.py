"""ESP32-CAM（CameraWebServer 韌體）串流來源的解析度控制。

ESP32-CAM 的 MJPEG HTTP 串流畫面尺寸由韌體端的 framesize 決定，
``cv2.VideoCapture.set(CAP_PROP_FRAME_WIDTH/HEIGHT)`` 對它「完全無效」
（那個 .set() 只對真的接 USB/內建攝影機時才生效）。要真的改變 ESP32-CAM
的輸出解析度，只能對韌體的 HTTP 控制端點送 ``?var=framesize&val=N`` 請求。

CameraWebServer 預設把控制端點（``/control``、``/status``）放在 80 埠，
把 MJPEG 串流（``/stream``）放在 81 埠，所以從串流網址推導控制端點時要換掉埠。

framesize 調小同時解掉「解析度過大」與「WiFi 頻寬吃滿 / fps 太低造成的卡頓」。
"""

import ipaddress
import urllib.request
from urllib.parse import urlsplit, urlunsplit

# ESP32-CAM (esp_camera) framesize_t 列舉 -> 對應像素尺寸。
# 只列 CameraWebServer 網頁下拉選單會出現、且 OV2640 支援的標準尺寸。
# 參考 esp32-camera/driver/include/sensor.h。
_FRAMESIZE_TABLE = [
    (96, 96, 0),      # FRAMESIZE_96X96
    (160, 120, 1),    # FRAMESIZE_QQVGA
    (176, 144, 2),    # FRAMESIZE_QCIF
    (240, 176, 3),    # FRAMESIZE_HQVGA
    (240, 240, 4),    # FRAMESIZE_240X240
    (320, 240, 5),    # FRAMESIZE_QVGA
    (400, 296, 6),    # FRAMESIZE_CIF
    (480, 320, 7),    # FRAMESIZE_HVGA
    (640, 480, 8),    # FRAMESIZE_VGA
    (800, 600, 9),    # FRAMESIZE_SVGA
    (1024, 768, 10),  # FRAMESIZE_XGA
    (1280, 720, 11),  # FRAMESIZE_HD
    (1280, 1024, 12), # FRAMESIZE_SXGA
    (1600, 1200, 13), # FRAMESIZE_UXGA
]


def framesize_value_for(width: int, height: int) -> tuple[int, int, int]:
    """把目標 (width, height) 換算成最接近的 ESP32-CAM framesize 列舉值。

    規則：
    1. 完全相符 -> 直接用。
    2. 否則取「寬高都不超過目標」的最大標準尺寸（畫面不會比要求的還大）。
    3. 全部都比目標大（目標非常小）-> 取最小的標準尺寸。

    Returns:
        (framesize_width, framesize_height, framesize_enum_value)
    """
    for w, h, val in _FRAMESIZE_TABLE:
        if w == width and h == height:
            return w, h, val

    not_larger = [row for row in _FRAMESIZE_TABLE if row[0] <= width and row[1] <= height]
    if not_larger:
        w, h, val = max(not_larger, key=lambda row: row[0] * row[1])
        return w, h, val

    w, h, val = min(_FRAMESIZE_TABLE, key=lambda row: row[0] * row[1])
    return w, h, val


def _is_lan_host(hostname: str) -> bool:
    """判斷主機名稱是否指向區網裝置（私有／loopback／link-local IP，或 .local mDNS）。

    ESP32-CAM 一定掛在區網；限制只對區網主機送控制請求，避免 YouTube／其他
    公開 HTTP 串流來源被送出無意義的 /control 請求到第三方伺服器。
    """
    host = hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def _control_base_url(stream_url: str, control_port: int) -> str | None:
    """從串流網址推導 ``/control`` 端點；非 http(s)／非區網主機／無法解析時回傳 None。

    control_port <= 0 代表沿用串流網址本身的埠。
    """
    parts = urlsplit(stream_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if not _is_lan_host(parts.hostname):
        return None
    if control_port and control_port > 0:
        netloc = f"{parts.hostname}:{control_port}"
    elif parts.port:
        netloc = f"{parts.hostname}:{parts.port}"
    else:
        netloc = parts.hostname
    return urlunsplit((parts.scheme, netloc, "/control", "", ""))


def _http_get(url: str, timeout: float) -> int:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (區網固定 http URL)
        return getattr(resp, "status", None) or resp.getcode()


def configure_stream(
    stream_url: str,
    target_width: int,
    target_height: int,
    *,
    control_port: int = 80,
    quality: int | None = None,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """對 ESP32-CAM 送 framesize（必要時再送 quality）控制請求。

    來源若不是 ESP32-CAM（控制端點連不上、回非預期），會回傳 ``(False, 原因)``，
    呼叫端應把它當成「這條串流不是 ESP32-CAM 或無法調整」，安靜略過、繼續開串流。

    Returns:
        (成功與否, 說明字串)
    """
    base = _control_base_url(stream_url, control_port)
    if base is None:
        return False, "串流網址非 http(s) 或無法解析主機，略過 ESP32-CAM 控制"

    fs_w, fs_h, fs_val = framesize_value_for(target_width, target_height)
    try:
        _http_get(f"{base}?var=framesize&val={fs_val}", timeout)
        detail = f"framesize={fs_val}（{fs_w}x{fs_h}）"
        if quality is not None and quality >= 0:
            _http_get(f"{base}?var=quality&val={int(quality)}", timeout)
            detail += f"、quality={int(quality)}"
    except Exception as e:  # urllib.error.URLError / socket.timeout / OSError 等
        return False, f"{type(e).__name__}: {e}"
    return True, detail
