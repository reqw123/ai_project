"""
主入口點
"""
import os
import threading
import time
import requests

# 開發環境 workaround：避免 Windows 下 OpenMP runtime 重複載入導致程序中止
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from server.flask_app import create_app
from utils.helpers import get_ip
from config import FlaskConfig, NodeRedConfig

def send_ip_to_nodered(ip, node_red_url):
    """定期發送 Python IP 給 Node-RED，直到成功為止"""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(
                node_red_url,
                json={"ip": ip},
                timeout=NodeRedConfig.TIMEOUT
            )
            if response.status_code == 200:
                
                print(f"✅ 成功通知 Node-RED，Python IP: {ip}")
                break
            else:
                print(f"⚠ Node-RED 回應異常: {response.status_code}")
        except Exception as e:
            print(f"⚠ 無法連接 Node-RED (嘗試 {retry_count + 1}/{max_retries}): {e}")
        
        retry_count += 1
        time.sleep(3)  # 每 3 秒重試一次
    
    if retry_count >= max_retries:
        print("❌ 無法連接到 Node-RED，請檢查 Node-RED 是否啟動")

if __name__ == "__main__":
    if FlaskConfig.DEBUG:
        import warnings
        warnings.warn(
            "Flask DEBUG=True（Werkzeug interactive debugger 開啟，LAN 環境下任何人都能執行任意程式碼）。"
            "生產環境請確認環境變數 CAT_MONITORING_FLASK_DEBUG 未設為 true。",
            RuntimeWarning, stacklevel=1,
        )

    app = create_app()
    ip = get_ip()
    if not ip:
        ip = "127.0.0.1"
    print(f"\n📺 Web 服務器啟動於 http://{ip}:{FlaskConfig.PORT}")
    print(f"📊 串流網址: http://{ip}:{FlaskConfig.PORT}/stream")

    node_red_url = NodeRedConfig.ENDPOINT_NOTIFY
    if ip and ip != "127.0.0.1":
        threading.Thread(
            target=send_ip_to_nodered,
            args=(ip, node_red_url),
            daemon=True
        ).start()
    else:
        print("⚠ 無法取得有效 IP，跳過 Node-RED 上線通知")

    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        threaded=FlaskConfig.THREADED,
        debug=FlaskConfig.DEBUG,
    )
