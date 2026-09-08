/*
 * esp32_motion —— 移動偵測節點（PIR）
 *
 * 發布：cat/iot/motion/<SOURCE_ID>
 * 時機：偵測狀態改變時才送 {"active": true/false}；另外每 HEARTBEAT_MS
 *       補送一次目前狀態，避免 hub 錯過邊緣事件。
 *
 * ── 硬體 / 感測器型號 ──────────────────────────────────────────────
 *   MCU  : Espressif ESP32-WROOM-32 dev board
 *   移動 : HC-SR501 passive infrared (PIR) motion sensor module
 *          （本專案採用 PIR：對準單一開放區域、只對溫體反應、不穿牆、
 *           供電寬容——理由見 iot/docs 討論。食盆/貓砂盆的「貓在不在」
 *           改由 esp32_weight 的 HX711 直接判斷，不靠這支。）
 *
 * HC-SR501 板上兩顆電位器 / 跳線：
 *   Sensitivity（Sx）：偵測距離，順時針增大（~3–7 m）
 *   Time-delay（Tx）：觸發後 OUT 保持 HIGH 的時間，逆時針最短（~3 s）
 *   跳線設 H（repeatable trigger）：期間內再偵測到會延長 HIGH，較適合這裡
 *   開機後有 ~30–60 s 暖機期，讀值不穩屬正常
 *
 * 接線：
 *   HC-SR501  VCC -> 5V   GND -> GND   OUT -> GPIO 27
 */

#include <WiFi.h>
#include <PubSubClient.h>

// ─── 要改的設定 ──────────────────────────────────────────────
const char* WIFI_SSID     = "CBN-B4640-2.4G";
const char* WIFI_PASSWORD = "110106291208";
const char* MQTT_HOST     = "192.168.0.171";   // mosquitto / Node-RED 那台
const uint16_t MQTT_PORT  = 1883;
const char* MQTT_USER     = "";
const char* MQTT_PASS     = "";
const char* SOURCE_ID     = "food_area";
// ────────────────────────────────────────────────────────────

// ═══ 這支節點的 MQTT 身分與 topic（都集中在這，一眼看完）═══════════
//   發布 (publish)：
const String MQTT_CLIENT_ID = String("esp32-motion-") + SOURCE_ID;    // 連線用戶端 ID
const String T_PUB_MOTION   = String("cat/iot/motion/") + SOURCE_ID;  // {"active":true/false}
//   訂閱 (subscribe)：無
// ════════════════════════════════════════════════════════════════════

#define PIR_PIN 27   // HC-SR501 PIR motion sensor — OUT（3.3V 邏輯，可直接讀）
const unsigned long HEARTBEAT_MS = 30000;
const unsigned long DEBOUNCE_MS  = 500;
const unsigned long WARMUP_MS    = 60000;  // HC-SR501 暖機期，期間 OUT 不穩，先不送狀態

// 連線維護（非阻塞）：WiFi 或 broker 掛掉時 loop() 不會被卡住。
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long MQTT_RETRY_MS = 3000;
const uint16_t MQTT_SOCKET_TIMEOUT_SEC = 2;
const uint16_t MQTT_KEEPALIVE_SEC      = 15;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

int  g_lastState = -1;
unsigned long g_lastChange = 0;
unsigned long g_lastHeartbeat = 0;
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

void publishState(int state) {
  const char* payload = state ? "{\"active\":true}" : "{\"active\":false}";
  mqtt.publish(T_PUB_MOTION.c_str(), payload);   // 未連線時 PubSubClient 直接回 false，不阻塞
}

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setSocketTimeout(MQTT_SOCKET_TIMEOUT_SEC);
  mqtt.setKeepAlive(MQTT_KEEPALIVE_SEC);
}

void loop() {
  handleConnections();

  unsigned long now = millis();
  if (now < WARMUP_MS) return;   // HC-SR501 暖機中，OUT 讀值不可靠，先不動作

  int state = digitalRead(PIR_PIN);

  // 狀態變化偵測照常跑（不管 MQTT 連不連得上）——重連後 heartbeat 會把
  // 目前狀態補送出去，所以斷線期間漏掉的邊緣事件恢復後會對齊。
  if (state != g_lastState && now - g_lastChange >= DEBOUNCE_MS) {
    g_lastState = state;
    g_lastChange = now;
    if (mqtt.connected()) publishState(state);
  }

  if (now - g_lastHeartbeat >= HEARTBEAT_MS) {
    g_lastHeartbeat = now;
    if (mqtt.connected() && g_lastState >= 0) publishState(g_lastState);
  }
}
