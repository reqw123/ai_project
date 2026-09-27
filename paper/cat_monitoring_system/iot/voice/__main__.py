"""`python -m iot.voice` 進入點（從 ``paper/cat_monitoring_system/`` 執行）。

跟 ``main.py``、``python -m iot``（感測器 hub）都是各自獨立的行程，互不啟動對方。
"""

from iot.voice.runner import run

if __name__ == "__main__":
    run()
