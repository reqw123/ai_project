"""
Node-RED 通訊
"""
import requests
from datetime import datetime
from config import NodeRedConfig

class NodeRedClient:
    def __init__(self, url_result, url_notify=None, local_ip=None):
        self.url_result = url_result
        self.url_notify = url_notify
        self.local_ip = local_ip

    def send_data(self, data):
        try:
            # 先發送 python_online 狀態（如有設置）
            if self.url_notify and self.local_ip:
                now = datetime.now()
                timestamp = now.strftime("%H:%M:%S")
                notify_data = {"status": "online", "ip": self.local_ip, "timestamp": timestamp}
                requests.post(self.url_notify, json=notify_data, timeout=NodeRedConfig.TIMEOUT)
            # 再發送 yolo_result 行為資料
            requests.post(self.url_result, json=data, timeout=NodeRedConfig.TIMEOUT)
        except Exception:
            pass
