/*
 * esp32_env —— 環境感測節點（氣體 / 光照）
 *
 * 發布：cat/iot/env/<SOURCE_ID>   payload: {"gas_ppm":..,"lux":..}
 * 週期：每 PUBLISH_INTERVAL_MS 送一次目前讀數
 *
 * 溫濕度已改由 esp32_petbox（MLX90614 + DHT）負責，這支只留氣體與光照。
 *
 * ── 硬體 / 感測器型號 ──────────────────────────────────────────────
 *   MCU     : Espressif ESP32-WROOM-32 dev board（ESP32-DevKitC 類）
 *   氣體     : Winsen MQ-135 air quality / gas sensor module（類比 AOUT）
 *             ※ 想量更明確的 CO：Winsen MQ-7 carbon monoxide sensor module
 *   光照     : GL5528 photoresistor（CdS light-dependent resistor, LDR）+ 10kΩ 分壓
 *             ※ 想要真正校正過的 lux：Rohm BH1750FVI ambient light sensor（I2C）
 *
 * 接線（範例，依實際模組調整）：
 *   MQ-135 AOUT       -> GPIO 34 (ADC1_CH6, 類比輸入)
 *   GL5528 分壓中點    -> GPIO 35 (ADC1_CH7)
 *
 * 這是「範本」，MQ-135 的 ppm 換算只是粗略示意，實際使用請依你的模組
 * datasheet 校正（或直接送原始 ADC 值、在 hub 端再換算）。
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ─── 要改的設定 ──────────────────────────────────────────────
const char* WIFI_SSID     = "CBN-B4640-2.4G";
const char* WIFI_PASSWORD = "110106291208";
const char* MQTT_HOST     = "192.168.0.171";   // mosquitto / Node-RED 那台
const uint16_t MQTT_PORT  = 1883;
const char* MQTT_USER     = "";          // 匿名留空
const char* MQTT_PASS     = "";
const char* SOURCE_ID     = "living_room";
// ────────────────────────────────────────────────────────────

// ═══ 這支節點的 MQTT 身分與 topic（都集中在這，一眼看完）═══════════
//   發布 (publish)：
const String MQTT_CLIENT_ID = String("esp32-env-") + SOURCE_ID;      // 連線用戶端 ID
const String T_PUB_ENV      = String("cat/iot/env/") + SOURCE_ID;    // 氣體 + 光照讀數
//   訂閱 (subscribe)：無（這支只發不收）
// ════════════════════════════════════════════════════════════════════

#define MQ135_PIN 34   // Winsen MQ-135 gas sensor module — AOUT
#define LDR_PIN   35   // GL5528 CdS photoresistor (LDR) — 分壓中點
const unsigned long PUBLISH_INTERVAL_MS = 10000;

// 連線維護（非阻塞）：WiFi 或 broker 掛掉時 loop() 不會被卡住。
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long MQTT_RETRY_MS = 3000;
const uint16_t MQTT_SOCKET_TIMEOUT_SEC = 2;   // 縮短 PubSubClient 預設 15s，失敗的 connect() 快速返回
const uint16_t MQTT_KEEPALIVE_SEC      = 15;

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
    return;                       // WiFi 沒好就別碰 MQTT（避免卡在 DNS/socket）
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

// Winsen MQ-135 air quality / gas sensor module
float readGasPpm() {
  int raw = analogRead(MQ135_PIN);            // 0..4095
  return raw * (1000.0f / 4095.0f);           // 粗略映射到 0..1000 ppm（範本）
}

// GL5528 CdS photoresistor (LDR) — 分壓讀值
float readLux() {
  int raw = analogRead(LDR_PIN);
  return raw * (2000.0f / 4095.0f);           // 粗略映射（範本）
}

void publishReading() {
  StaticJsonDocument<128> doc;
  doc["gas_ppm"] = readGasPpm();
  doc["lux"] = readLux();

  char buf[128];
  size_t n = serializeJson(doc, buf);
  mqtt.publish(T_PUB_ENV.c_str(), buf, n);
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setSocketTimeout(MQTT_SOCKET_TIMEOUT_SEC);
  mqtt.setKeepAlive(MQTT_KEEPALIVE_SEC);
  // 不在 setup 阻塞等連線；由 loop() 的 handleConnections() 非阻塞處理
}

void loop() {
  handleConnections();

  unsigned long now = millis();
  if (now - g_lastPublish >= PUBLISH_INTERVAL_MS) {
    g_lastPublish = now;
    if (mqtt.connected()) publishReading();   // 連著才送，斷線時靜靜跳過
  }
}
