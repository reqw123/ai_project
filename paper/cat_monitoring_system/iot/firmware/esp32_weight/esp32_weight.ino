/*
 * esp32_weight —— 重量感測節點（HX711 + load cell）
 *
 * 發布：cat/iot/weight/<SCALE_ID>
 * 週期：每 PUBLISH_INTERVAL_MS 送一次目前克數 {"grams": 123.4}
 *
 * hub 端（sensors/weight.py 的 WeightEventDetector）會自己做「去皮 + 穩定後
 * 才承認變化」的進食/加料事件偵測，所以這裡送原始克數即可，不必在韌體端
 * 做事件判斷。
 *
 * ── 硬體 / 感測器型號 ──────────────────────────────────────────────
 *   MCU     : Espressif ESP32-WROOM-32 dev board
 *   秤重 ADC : Avia Semiconductor HX711 24-bit ADC load-cell amplifier module
 *   荷重元   : straight-bar strain-gauge load cell（如 SparkFun TAL220 5 kg，
 *             或 5 kg / 10 kg / 20 kg 半橋、全橋皆可，配合 CALIBRATION_FACTOR）
 *
 * 接線：
 *   HX711 DOUT -> GPIO 16
 *   HX711 SCK  -> GPIO 4
 *   load cell 紅/黑/白/綠 -> HX711 E+/E-/A-/A+
 *
 * CALIBRATION_FACTOR 要用已知砝碼校正：先燒錄、開序列埠，放已知重量，
 * 調整這個數字直到讀數正確。
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "HX711.h"

// ─── 要改的設定 ──────────────────────────────────────────────
const char* WIFI_SSID     = "CBN-B4640-2.4G";
const char* WIFI_PASSWORD = "110106291208";
const char* MQTT_HOST     = "192.168.0.171";   // mosquitto / Node-RED 那台
const uint16_t MQTT_PORT  = 1883;
const char* MQTT_USER     = "";
const char* MQTT_PASS     = "";
const char* SCALE_ID      = "food_bowl";   // ← 要跟 config.py 的 FOOD_SCALE_IDS 一致
const float CALIBRATION_FACTOR = 420.0f;   // ← 用砝碼校正
// ────────────────────────────────────────────────────────────

// ═══ 這支節點的 MQTT 身分與 topic（都集中在這，一眼看完）═══════════
//   發布 (publish)：
const String MQTT_CLIENT_ID = String("esp32-weight-") + SCALE_ID;     // 連線用戶端 ID
const String T_PUB_WEIGHT   = String("cat/iot/weight/") + SCALE_ID;   // {"grams":float}
//   訂閱 (subscribe)：無
// ════════════════════════════════════════════════════════════════════

#define HX711_DOUT 16   // Avia Semiconductor HX711 load-cell amp — DOUT/DT
#define HX711_SCK  4    // Avia Semiconductor HX711 load-cell amp — SCK/PD_SCK
const unsigned long PUBLISH_INTERVAL_MS = 5000;

// 連線維護（非阻塞）：WiFi 或 broker 掛掉時 loop() 不會被卡住。
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long MQTT_RETRY_MS = 3000;
const uint16_t MQTT_SOCKET_TIMEOUT_SEC = 2;
const uint16_t MQTT_KEEPALIVE_SEC      = 15;

HX711 scale;
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

unsigned long g_lastPublish = 0;
unsigned long g_lastWifiRetry = 0;
unsigned long g_lastMqttRetry = 0;

// 非阻塞連線管理：連得上就 mqtt.loop() 維護，連不上就返回，絕不 while/delay 卡住。
void handleConnections() {
  unsigned long now = millis();
  static bool wifiBegun = false;

  if (WiFi.status() != WL_CONNECTED) {
    if (!wifiBegun || now - g_lastWifiRetry > WIFI_RETRY_MS) {
      g_lastWifiRetry = now;
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      wifiBegun = true;
    }
    return;
  }

  if (!mqtt.connected()) {
    if (now - g_lastMqttRetry > MQTT_RETRY_MS) {
      g_lastMqttRetry = now;
      mqtt.connect(MQTT_CLIENT_ID.c_str(),
                   strlen(MQTT_USER) ? MQTT_USER : nullptr,
                   strlen(MQTT_PASS) ? MQTT_PASS : nullptr);
    }
    return;
  }

  mqtt.loop();
}

void publishWeight() {
  if (!scale.is_ready()) return;
  float grams = scale.get_units(5);          // 5 次取平均

  StaticJsonDocument<64> doc;
  doc["grams"] = grams;
  char buf[64];
  size_t n = serializeJson(doc, buf);
  mqtt.publish(T_PUB_WEIGHT.c_str(), buf, n);
}

void setup() {
  Serial.begin(115200);

  scale.begin(HX711_DOUT, HX711_SCK);
  scale.set_scale(CALIBRATION_FACTOR);
  scale.tare();                              // 開機當下的重量視為 0

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setSocketTimeout(MQTT_SOCKET_TIMEOUT_SEC);
  mqtt.setKeepAlive(MQTT_KEEPALIVE_SEC);
}

void loop() {
  handleConnections();

  unsigned long now = millis();
  if (now - g_lastPublish >= PUBLISH_INTERVAL_MS) {
    g_lastPublish = now;
    if (mqtt.connected()) publishWeight();   // 連著才送，斷線時靜靜跳過
  }
}
