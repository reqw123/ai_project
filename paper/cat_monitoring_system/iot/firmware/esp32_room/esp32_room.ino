/*
 * esp32_room —— 房間節點：空氣品質＋光照＋移動偵測（一片 ESP32）
 *
 * 2026-09-27 由 esp32_env（氣體／光照）＋ esp32_motion（PIR）合併：兩者用的腳位不衝突（GPIO 34、35、27），
 * 放在同一個房間時省一片開發板。hub 端完全不用改——topic 跟原本兩支一樣：
 *   發布：cat/iot/env/<SOURCE_ID>     {"gas_ppm":..,"lux":..}      每 ENV_INTERVAL_MS 一次
 *         cat/iot/motion/<SOURCE_ID>  {"active":true/false}          狀態改變時送，另外每 HEARTBEAT_MS 補送一次
 * 原本分開的 esp32_env／esp32_motion 仍保留（感測器要放不同位置時用）；同一個 SOURCE_ID 不要兩種同時燒。
 *
 * ── 硬體 / 感測器型號 ──────────────────────────────────────────────
 *   MCU  : Espressif ESP32-WROOM-32 dev board（ESP32-DevKitC 類）
 *   氣體 : Winsen MQ-135 air quality / gas sensor module（類比 AOUT）
 *   光照 : GL5528 photoresistor（CdS LDR）+ 10kΩ 分壓
 *   移動 : HC-SR501 passive infrared (PIR) motion sensor module
 *
 * 接線：
 *   MQ-135   VCC -> 5V（加熱器要 5V、約 150 mA）  GND -> GND
 *            AOUT -> 10kΩ -> GPIO 34 -> 20kΩ -> GND
 *            ⚠ AOUT 最高接近 5V，ESP32 類比腳最多 3.3V：一定要經過這組分壓（5V × 20/30 ≈ 3.3V），不能直接接
 *   GL5528   3.3V -> GL5528 -> GPIO 35 -> 10kΩ -> GND（越亮 GPIO 35 電壓越高）
 *   HC-SR501 VCC -> 5V   GND -> GND   OUT -> GPIO 27（OUT 是 3.3V 邏輯，可直接接）
 *   電源：MQ-135 加熱器吃電，USB 電源建議 1A 以上
 *
 * 暖機：MQ-135 通電後要幾分鐘讀值才穩；HC-SR501 開機約 30–60 秒 OUT 不穩（WARMUP_MS 期間不送移動狀態，
 *       但空氣品質／光照照送——合併時注意不能讓 PIR 暖機卡住整個 loop()）。
 *
 * HC-SR501 板上兩顆電位器 / 跳線：
 *   Sensitivity（Sx）：偵測距離，順時針增大（~3–7 m）
 *   Time-delay（Tx）：觸發後 OUT 保持 HIGH 的時間，逆時針最短（~3 s）
 *   跳線設 H（repeatable trigger）：期間內再偵測到會延長 HIGH
 *
 * MQ-135 的 ppm 換算只是粗略示意（分壓後 ADC 0..4095 ≈ AOUT 0..5V → 0..1000 ppm），
 * 實際使用請依模組 datasheet 校正。
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
const char* SOURCE_ID     = "living_room";   // env、motion 兩個 topic 的最後一段（＝放在哪個房間）
String g_hostname;                           // 網路主機名稱 cat-room-<節點>（setup() 裡組好）
// ────────────────────────────────────────────────────────────

// ═══ 這支節點的 MQTT 身分與 topic（都集中在這，一眼看完）═══════════
//   發布 (publish)：
const String MQTT_CLIENT_ID = String("esp32-room-") + SOURCE_ID;      // 連線用戶端 ID（一片板子一條連線）
const String T_PUB_ENV      = String("cat/iot/env/") + SOURCE_ID;     // 氣體 + 光照
const String T_PUB_MOTION   = String("cat/iot/motion/") + SOURCE_ID;  // {"active":true/false}
//   訂閱 (subscribe)：無（這支只發不收）
// ════════════════════════════════════════════════════════════════════

#define MQ135_PIN 34   // Winsen MQ-135 AOUT（經 10k/20k 分壓）— ADC1_CH6
#define LDR_PIN   35   // GL5528 分壓中點 — ADC1_CH7
#define PIR_PIN   27   // HC-SR501 OUT

const unsigned long ENV_INTERVAL_MS = 10000;   // 空氣品質／光照多久送一次
const unsigned long HEARTBEAT_MS    = 30000;   // 移動狀態沒變時多久補送一次
const unsigned long DEBOUNCE_MS     = 500;
const unsigned long WARMUP_MS       = 60000;   // HC-SR501 暖機期，期間 OUT 不穩，先不送移動狀態

// 連線維護（非阻塞）：WiFi 或 broker 掛掉時 loop() 不會被卡住。
const unsigned long WIFI_RETRY_MS = 5000;
const unsigned long MQTT_RETRY_MS = 3000;
const uint16_t MQTT_SOCKET_TIMEOUT_SEC = 2;    // 縮短 PubSubClient 預設 15s，失敗的 connect() 快速返回
const uint16_t MQTT_KEEPALIVE_SEC      = 5;    // 斷電時 broker 約 1.5 倍（7.5 秒）後判定離線

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

unsigned long g_lastEnvPublish = 0;
int  g_lastMotion = -1;
unsigned long g_lastMotionChange = 0;
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

// ── 空氣品質＋光照 ───────────────────────────────────────────────
float readGasPpm() {
  int raw = analogRead(MQ135_PIN);            // 0..4095（分壓後 ≈ AOUT 0..5V）
  return raw * (1000.0f / 4095.0f);           // 粗略映射到 0..1000 ppm（範本）
}

float readLux() {
  int raw = analogRead(LDR_PIN);
  return raw * (2000.0f / 4095.0f);           // 粗略映射（範本）
}

void publishEnv() {
  StaticJsonDocument<128> doc;
  doc["gas_ppm"] = readGasPpm();
  doc["lux"] = readLux();

  char buf[128];
  size_t n = serializeJson(doc, buf);
  mqtt.publish(T_PUB_ENV.c_str(), buf, n);
}

// ── 移動偵測 ──────────────────────────────────────────────────────
void publishMotion(int state) {
  const char* payload = state ? "{\"active\":true}" : "{\"active\":false}";
  mqtt.publish(T_PUB_MOTION.c_str(), payload);   // 未連線時 PubSubClient 直接回 false，不阻塞
}

void updateMotion(unsigned long now) {
  if (now < WARMUP_MS) return;   // HC-SR501 暖機中：只跳過移動偵測，不影響空氣品質上傳

  int state = digitalRead(PIR_PIN);
  // 狀態變化偵測照常跑（不管 MQTT 連不連得上）——重連後 heartbeat 會把目前狀態補送出去
  if (state != g_lastMotion && now - g_lastMotionChange >= DEBOUNCE_MS) {
    g_lastMotion = state;
    g_lastMotionChange = now;
    if (mqtt.connected()) publishMotion(state);
  }
  if (now - g_lastHeartbeat >= HEARTBEAT_MS) {
    g_lastHeartbeat = now;
    if (mqtt.connected() && g_lastMotion >= 0) publishMotion(g_lastMotion);
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(PIR_PIN, INPUT);

  // 網路上的主機名稱（DHCP）：設定視窗「IoT 子系統」用 DNS 反查這個名稱，才知道哪個 IP 是哪一支。
  // cat-room-… 會同時顯示在「環境感測」和「移動偵測」兩張卡片。主機名稱不能有底線，換成連字號。
  // 要在 WiFi.mode() 之前設。
  g_hostname = String("cat-room-") + SOURCE_ID;
  g_hostname.replace("_", "-");
  WiFi.setHostname(g_hostname.c_str());
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
  if (now - g_lastEnvPublish >= ENV_INTERVAL_MS) {
    g_lastEnvPublish = now;
    if (mqtt.connected()) publishEnv();       // 連著才送，斷線時靜靜跳過
  }
  updateMotion(now);
}
