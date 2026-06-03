"""
工具函數
"""
import socket

from utils.constants import BEHAVIOR_CLASSES, BEHAVIOR_TEXT_MAP, LOW_CONF_ID, LOW_CONF_TEXT
from config import BehaviorTrackingConfig as _BehaviorTrackingConfig

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

