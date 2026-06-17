"""
Node-RED 通訊
"""
import logging
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
            if self.url_notify and self.local_ip:
                now = datetime.now()
                notify_data = {"status": "online", "ip": self.local_ip, "timestamp": now.strftime("%H:%M:%S")}
                requests.post(self.url_notify, json=notify_data, timeout=NodeRedConfig.TIMEOUT)
            requests.post(self.url_result, json=data, timeout=NodeRedConfig.TIMEOUT)
            v2_url = getattr(NodeRedConfig, 'ENDPOINT_RESULT_V2', None)
            if v2_url and v2_url != self.url_result:
                try:
                    requests.post(v2_url, json=data, timeout=NodeRedConfig.TIMEOUT)
                except Exception as e:
                    logging.warning("NodeRedClient.send_data v2 failed: %s", e)
            return True
        except Exception as e:
            logging.warning("NodeRedClient.send_data failed: %s", e)
            return False
