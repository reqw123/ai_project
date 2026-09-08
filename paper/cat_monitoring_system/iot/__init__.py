"""物聯網感測子系統（IoT Sensor Hub）——完全獨立、可整包移除的加值模組。

定位
────
這個套件是整個貓咪監測系統的「錦上添花」部分：透過 MQTT 收 ESP32 環境感測、
PIR 移動偵測、HX711 重量感測的資料，做正規化 / 持久化 / 門檻告警，再把
結果發回 MQTT 給 Node-RED Dashboard 使用。

低耦合原則（硬性規定）
──────────────────────
* 本套件**不 import** 主系統任何模組（``config`` / ``analytics`` / ``processors``
  / ``detectors`` / ``trackers`` / ``server`` / ``utils`` …）。內部一律
  ``from iot.xxx import ...``。
* 主系統也**不 import** 本套件——``main.py`` / ``server/`` 完全不知道它存在。
* 唯一對外接觸點是 MQTT broker（沿用 Node-RED 的 mosquitto）。不持有任何
  Discord / 外部服務憑證。
* fail-safe：MQTT 斷線自動重連、parser 例外丟棄該筆、SQLite 失敗記警告後
  繼續。整個子系統掛掉對 ``python main.py`` 零影響。
* 獨立行程：``cd paper/cat_monitoring_system && python -m iot`` 啟動，跟
  ``main.py`` 分開跑、互不啟動對方。

刪除方式：直接刪掉 ``paper/cat_monitoring_system/iot/`` 整個資料夾，並把
``pytest.ini`` 裡對應的 testpath 那一行移除即可，主系統不受任何影響。
"""

from iot.config import IotHubConfig

__all__ = ["IotHubConfig"]
