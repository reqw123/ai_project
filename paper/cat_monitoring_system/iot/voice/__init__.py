"""飼主語音紀錄服務（Owner Voice Reports）——語音辨識系統與貓咪行為辨識系統之間的情境回饋通道。

定位
────
飼主對 ESP32 語音終端說「今天帶貓咪去看醫生」「貓咪剛剛吐了」，語音辨識（Node-RED 語音分頁）
找不到控制指令時，把文字經 MQTT 交給本服務：分類成 **情境事件**（整天，對應基線的混淆因子）
或 **飼主觀察**（時間點，當作行為偏離預警的弱標註），存進自己的 SQLite，再把結果發回
Node-RED（儀表板呈現、ESP32 唸確認句）。論文定位見 ``README.md``。

高內聚
──────
「飼主回報」這件事的全部知識都在這個資料夾：類別本體（``ontology``）、分類規則（``classifier``）、
儲存（``store``）、協調（``service``）、MQTT 介面（``transport``／``runner``）、論文用匯出（``export``）、
Node-RED 橋接與儀表板流程（``node_red/``）、測試（``tests/``）。

低耦合（硬性規定，比照 ``iot/`` 感測器 hub）
──────────────────────────────────────────
* 本套件**不 import** 主系統任何模組（``config``／``analytics``／``server``…），也**不 import**
  ``iot`` 感測器 hub 的模組（``iot.config``／``iot.ingest``／``iot.storage``…）；只 import ``iot.voice.*``。
* 主系統、感測器 hub 也都不 import 本套件。
* 唯一對外接觸點＝MQTT broker（``cat/voice/*``）。不知道 Node-RED 流程長什麼樣，也不知道 ESP32 存在。
* 獨立行程：``cd paper/cat_monitoring_system && python -m iot.voice``。掛掉時語音分頁照常控制燈／電腦，
  只是不記錄（Node-RED 等不到回應會當成「不是紀錄」）。

刪除方式：刪掉 ``iot/voice/`` 整個資料夾，並移除 ``pytest.ini`` 裡 ``iot/voice/tests`` 那一行。
"""

from iot.voice.config import VoiceReportConfig

__all__ = ["VoiceReportConfig"]
