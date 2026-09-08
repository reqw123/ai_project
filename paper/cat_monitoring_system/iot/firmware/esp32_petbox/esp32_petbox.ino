/*
 * esp32_petbox —— 外出包 / 貓窩 環境 + 非接觸體溫監測節點
 *
 * 這份是舊 Arduino 專案「寵物包/mqtt_all」的**原樣復原版**（保留 yuanpei 開機
 * 圖、卡片式 TFT UI、警報閾值、Arduino OTA、非阻塞 WiFi/MQTT 重連）。
 *
 * ⚑ BluetoothSerial 全部以註解關閉（用不到、又佔掉一大塊 flash/RAM）。要恢復
 *   的話，把下面所有標了「// [BT]」的行取消註解即可。
 *
 * 相對原檔的修改：
 *
 *   1. 修掉原檔第 13/17 行的 bug：ssid 那行後面黏了一段 ngrok 網址、且
 *      mqtt_server 被宣告兩次。
 *   2. 除了原本的 esp32/sensors、esp32/status、esp32/alert 之外，**額外**
 *      publish 兩個對齊 cat_monitoring_system/iot/ hub 的 topic：
 *          cat/iot/env/<SOURCE_ID>       {"temp_c":..,"humidity_pct":..}
 *          cat/iot/bodytemp/<SOURCE_ID>  {"surface_temp_c":..,"ambient_temp_c":..}
 *      門檻告警仍由 hub 端 (output/alert_engine.py) 統一處理，這裡的
 *      esp32/alert 只是保留原專案行為。
 *   3. 修掉原檔 TFT 每秒「整片擦掉重畫」造成的閃爍：靜態版面（標題、卡片
 *      外框、標籤）改成 setup() 畫一次，loop 只用 setTextPadding 就地覆蓋
 *      會變動的數值，不再 fillRect 清背景。
 *   4. 三顆狀態燈加上固定文字標籤（WiFi / MQTT / ALERT），獨立成一列，
 *      不用再猜哪顆是哪個。綠 = 正常/已連線，紅 = 斷線/有告警。
 *   5. 連線改成完全非阻塞 + MQTT socket timeout 縮到 2 秒；loop() 先畫 TFT、
 *      最後才維護連線 → 斷線時螢幕不會被重連卡住（原本最壞卡 15 秒）。
 *
 * ⚠️ MLX90614 的 object temperature 對著貓量到的是「體表 / 毛髮表面」溫度，
 *    不等於肛溫 / 核心體溫，只作初步篩檢用（詳見 iot/config.py 的說明）。
 *
 * ── 硬體 / 感測器型號 ──────────────────────────────────────────────
 *   MCU     : Espressif ESP32-WROOM-32 dev board
 *   體表測溫 : Melexis MLX90614ESF-BAA infrared non-contact thermometer
 *             (single-zone, 90° FOV, I2C/SMBus)　※窄視角量小範圍可用 -DCI 版
 *   溫濕度   : Aosong DHT11 temperature & humidity sensor
 *             ※要更準可換 Aosong AM2302 / DHT22
 *   顯示     : Sitronix ST7789 1.3" 240x240 IPS TFT LCD module (SPI, no CS 常見)
 *   狀態燈   : 一般 5 mm LED + 220Ω
 *
 * 接線：
 *   MLX90614  SDA -> GPIO 21   SCL -> GPIO 22   (I2C)
 *   DHT11     DATA -> GPIO 15
 *   ST7789    -> 依 TFT_eSPI 的 User_Setup.h 設定（SCLK/MOSI/DC/RST/BLK）
 *   LED       -> GPIO 25
 *
 * 需要函式庫：WiFi(內建) / PubSubClient / Adafruit_MLX90614 / DHT sensor library
 *            / TFT_eSPI / ArduinoOTA(內建)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_MLX90614.h>   // Melexis MLX90614ESF-BAA 紅外線非接觸測溫
#include <DHT.h>                 // Aosong DHT11 溫濕度感測
#include <TFT_eSPI.h>            // Sitronix ST7789 240x240 IPS TFT
#include <SPI.h>
// #include "BluetoothSerial.h"   // [BT] 停用
#include "yuanpei.h"          // 元培校徽 80x80 RGB565；開機時放大鋪滿整個螢幕
#include <ArduinoOTA.h>

// === Wi-Fi 設定 ===
const char* ssid     = "CBN-B4640-2.4G";
const char* password = "110106291208";

// === MQTT ===
const char* mqtt_server = "192.168.0.171";   // mosquitto / Node-RED 那台
const int   mqtt_port   = 1883;
const char* SOURCE_ID   = "carrier";         // 節點名稱 → topic 最後一段

// ═══ 這支節點用到的 MQTT topic（都集中在這，一眼看完）═══════════════
const String MQTT_CLIENT_ID  = String("esp32-petbox-") + SOURCE_ID;

//   發布 — 對齊 iot/ hub（Python 端會訂閱這兩個）：
const String T_PUB_ENV       = String("cat/iot/env/") + SOURCE_ID;       // {"temp_c","humidity_pct"}
const String T_PUB_BODYTEMP  = String("cat/iot/bodytemp/") + SOURCE_ID;  // {"surface_temp_c","ambient_temp_c"}

//   發布 — 舊「寵物包」專案 topic（保留，Python hub 不訂閱，給舊 Node-RED flow）：
#define T_PUB_LEGACY_SENSOR "esp32/sensors"
#define T_PUB_LEGACY_STATUS "esp32/status"
#define T_PUB_LEGACY_ALERT  "esp32/alert"

//   訂閱 (subscribe)：無
// ══════════════════════════════════════════════════════════════════════

// === 感測器 ===
#define DHTPIN 15        // Aosong DHT11 溫濕度感測 — DATA
#define DHTTYPE DHT11    // 換 Aosong AM2302 / DHT22 時改成 DHT22
#define LED_PIN 25       // 狀態 LED
// Melexis MLX90614ESF-BAA 走 I2C（GPIO 21 SDA / GPIO 22 SCL，見 Wire.begin）
// Sitronix ST7789 走 SPI，腳位在 TFT_eSPI 的 User_Setup.h 設定

// === yuanpei 開機圖 ===
// yuanpei.h 是 80x80 RGB565 圖（6400 個 uint16）。開機時以最近鄰放大
// YUANPEI_SCALE 倍鋪滿 ST7789（80 x 3 = 240，正好塞滿 240x240）。
// 想要更清晰不要鋸齒 → 得用 240x240 的原圖重新產生 yuanpei.h（image2cpp 之類），
// 這裡只有 80x80 的資料，放大就是會有點塊狀。
#define YUANPEI_W     80
#define YUANPEI_H     80
#define YUANPEI_SCALE 3

// === UI 顏色 ===
#define CARD_BG    0x2104
#define CARD_FRAME 0x630C
#define TITLE_BG   0x0008

// === 警報閾值 ===
const float LOW_AMBIENT_LIMIT  = 10.0;
const float HIGH_AMBIENT_LIMIT = 40.0;
const float LOW_OBJECT_LIMIT   = 10.0;
const float HIGH_OBJECT_LIMIT  = 70.0;

unsigned long lastAlertTime = 0;
const unsigned long ALERT_INTERVAL = 5000;

// === 物件 ===
TFT_eSPI tft = TFT_eSPI();                        // Sitronix ST7789 240x240 IPS TFT
Adafruit_MLX90614 mlx = Adafruit_MLX90614();      // Melexis MLX90614ESF-BAA 紅外線測溫
// BluetoothSerial SerialBT;   // [BT] 停用
WiFiClient espClient;
PubSubClient client(espClient);
DHT dht(DHTPIN, DHTTYPE);                         // Aosong DHT11 溫濕度感測

// === 時間 ===
unsigned long lastUpdate = 0;
unsigned long lastLedToggle = 0;
unsigned long lastWifiRetry = 0;
unsigned long lastMqttRetry = 0;

const unsigned long SENSOR_INTERVAL = 1000;
const unsigned long LED_INTERVAL     = 1000;
const unsigned long WIFI_RETRY_INTERVAL = 5000;
const unsigned long MQTT_RETRY_INTERVAL = 3000;

// PubSubClient 預設 socket timeout 15 秒、keepalive 15 秒——斷線重連時
// client.connect() 這個阻塞呼叫最壞會卡住整個 loop()（含 TFT/感測/OTA）15 秒。
// 縮短 socket timeout → connect() 失敗時快速返回；loop() 也改成「先畫 TFT、
// 再維護連線」，讓螢幕不受重連拖累（見 loop()）。
const uint16_t MQTT_SOCKET_TIMEOUT_SEC = 2;
const uint16_t MQTT_KEEPALIVE_SEC      = 15;

bool ledState = false;


// =====================================================
// UI FUNCTION
// =====================================================
void drawCard(int x, int y, int w, int h, uint16_t frameColor, uint16_t bgColor) {
  tft.fillRoundRect(x, y, w, h, 8, bgColor);
  tft.drawRoundRect(x, y, w, h, 8, frameColor);
}

void drawStatusDot(int x, int y, bool ok, int radius) {
  uint16_t color = ok ? TFT_GREEN : TFT_RED;
  tft.fillCircle(x, y, radius, color);
  tft.drawCircle(x, y, radius, TFT_BLACK);
}

// 把 80x80 的 yuanpei 以最近鄰放大 YUANPEI_SCALE 倍鋪滿螢幕（80*3 = 240）。
// 一次算好一整列（outW 寬）再 pushImage，逐列往下推 → 比 6400 個 fillRect 快很多。
void drawYuanpeiSplash() {
  const int outW = YUANPEI_W * YUANPEI_SCALE;   // 240
  const int outH = YUANPEI_H * YUANPEI_SCALE;   // 240
  const int xOff = (tft.width()  - outW) / 2;   // 240x240 螢幕 → 0
  const int yOff = (tft.height() - outH) / 2;

  tft.fillScreen(TFT_BLACK);

  static uint16_t line[YUANPEI_W * YUANPEI_SCALE];   // 一列放大後的像素（240）
  for (int sy = 0; sy < YUANPEI_H; sy++) {
    for (int sx = 0; sx < YUANPEI_W; sx++) {
      uint16_t c = yuanpei[sy * YUANPEI_W + sx];     // ESP32 上 PROGMEM 可直接索引
      for (int k = 0; k < YUANPEI_SCALE; k++) line[sx * YUANPEI_SCALE + k] = c;
    }
    for (int k = 0; k < YUANPEI_SCALE; k++)
      tft.pushImage(xOff, yOff + sy * YUANPEI_SCALE + k, outW, 1, line);
  }
}

// 卡片外框 / 固定標籤只需要畫一次（setup 呼叫）。之後 loop 只更新數值，
// 用 setTextPadding 讓新數字自己蓋掉舊字，不必先擦背景 → 不會閃。
const int CARD_Y[3]         = {90, 140, 190};
const char* CARD_LABEL[3]   = {"Body", "Amb", "Hum"};   // 短標籤，跟右側數值不打架
const uint16_t CARD_COLOR[3] = {TFT_YELLOW, TFT_GREEN, TFT_CYAN};

// 狀態列：每顆燈旁邊有固定文字標籤（畫一次），loop 只改燈的顏色。
// 綠 = 正常 / 已連線；紅 = 斷線 / 有告警。
const char* STAT_LABEL[3] = {"WiFi", "MQTT", "ALERT"};
const int   STAT_LABEL_X[3] = { 8,  86, 158};   // 標籤左緣 x
const int   STAT_DOT_X[3]   = {52, 130, 214};   // 燈圓心 x（緊接在標籤後面）
const int   STAT_Y          = 47;               // 標籤/燈的 y（標題列下方、資料卡上方）

void drawStaticUI() {
  tft.fillScreen(TFT_BLACK);

  tft.fillRect(0, 0, tft.width(), 30, TITLE_BG);
  tft.setTextDatum(TL_DATUM);
  tft.setTextColor(TFT_CYAN, TITLE_BG);
  tft.drawString("Pet Box Monitor", 6, 8, 2);

  // 狀態列標籤
  tft.setTextColor(TFT_LIGHTGREY, TFT_BLACK);
  for (int i = 0; i < 3; i++)
    tft.drawString(STAT_LABEL[i], STAT_LABEL_X[i], STAT_Y - 8, 2);

  for (int i = 0; i < 3; i++) {
    drawCard(10, CARD_Y[i], 220, 45, CARD_FRAME, CARD_BG);
    tft.setTextColor(CARD_COLOR[i], CARD_BG);
    tft.drawString(CARD_LABEL[i], 18, CARD_Y[i] + 15, 2);
  }
}


// =====================================================
// WiFi + MQTT 連線管理
//   - WiFi 未連上：只 WiFi.begin() 後返回，不阻塞；也不會去碰 client.connect()
//     （避免在沒網路時卡在 DNS / socket）。
//   - WiFi 已連上、MQTT 斷線：每 MQTT_RETRY_INTERVAL 試一次 client.connect()。
//     這個呼叫本身是阻塞的，但 setup() 已把 socket timeout 縮到
//     MQTT_SOCKET_TIMEOUT_SEC 秒，最壞只卡這麼久，且本函式在 loop() 最後
//     才呼叫，該輪 TFT/感測已畫完。
// =====================================================
void handleConnections() {
  unsigned long now = millis();
  static bool wifiConnecting = false;

  if (WiFi.status() != WL_CONNECTED) {
    if (!wifiConnecting) {
      WiFi.begin(ssid, password);
      wifiConnecting = true;
    }
    if (now - lastWifiRetry > WIFI_RETRY_INTERVAL) {
      lastWifiRetry = now;
      WiFi.disconnect(true);
      wifiConnecting = false;
    }
    return;
  }

  wifiConnecting = false;

  if (!client.connected()) {
    if (now - lastMqttRetry > MQTT_RETRY_INTERVAL) {
      lastMqttRetry = now;

      if (client.connect(MQTT_CLIENT_ID.c_str())) {
        client.publish(T_PUB_LEGACY_STATUS, "Reconnected");
      }
    }
    return;
  }

  client.loop();
}


// =====================================================
// ⭐ 初始化 Arduino OTA
// =====================================================
void setupOTA() {

  ArduinoOTA.setHostname("ESP32-OTA");   // OTA 名稱

  ArduinoOTA.onStart([]() {
    Serial.println("OTA Update Started");
  });

  ArduinoOTA.onEnd([]() {
    Serial.println("OTA Update Finished");
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("OTA Progress: %u%%\r", (progress * 100) / total);
  });

  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("OTA Error[%u]\n", error);
  });

  ArduinoOTA.begin();
  Serial.println("✔ Arduino OTA Ready!");
}


// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);

  // SerialBT.begin("ESP32_TempMonitor");   // [BT] 停用

  // MLX90614 & DHT
  Wire.begin(21, 22);
  mlx.begin();
  dht.begin();

  pinMode(LED_PIN, OUTPUT);

  // TFT
  tft.init();
  tft.setRotation(1);
  tft.setSwapBytes(true);
  tft.fillScreen(TFT_BLACK);

  // ⭐ yuanpei 開機圖（放大鋪滿整個螢幕，顯示 1.5 秒）
  drawYuanpeiSplash();
  delay(1500);

  // 靜態 UI（標題列 + 卡片外框 + 標籤）只畫這一次
  drawStaticUI();

  // MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setSocketTimeout(MQTT_SOCKET_TIMEOUT_SEC);  // 失敗的 connect() 快速返回，不卡 loop
  client.setKeepAlive(MQTT_KEEPALIVE_SEC);

  // WiFi（非阻塞：begin 後直接返回，連線狀態由 handleConnections() 輪詢）
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);

  // ⭐ OTA初始化（必須放 setup）
  setupOTA();

  Serial.println("System Ready");
}


// =====================================================
// 上傳給 iot/ hub 的兩個 topic（cat/iot/env、cat/iot/bodytemp）
// =====================================================
void publishToHub(float ambient, float object, float humidity, bool dht_ok) {
  if (!client.connected()) return;

  String env = "{";
  env += "\"temp_c\":" + String(ambient, 2);
  if (dht_ok) env += ",\"humidity_pct\":" + String(humidity, 1);
  env += "}";
  client.publish(T_PUB_ENV.c_str(), env.c_str());

  if (!isnan(object)) {
    String body = "{\"surface_temp_c\":" + String(object, 2) +
                  ",\"ambient_temp_c\":" + String(ambient, 2) + "}";
    client.publish(T_PUB_BODYTEMP.c_str(), body.c_str());
  }
}


// =====================================================
// LOOP
// =====================================================
void loop() {

  // ⭐ 必須一直處理 OTA
  ArduinoOTA.handle();

  unsigned long now = millis();

  // LED 閃爍
  if (now - lastLedToggle >= LED_INTERVAL) {
    lastLedToggle = now;
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
  }

  // 感測器讀取
  if (now - lastUpdate >= SENSOR_INTERVAL) {
    lastUpdate = now;

    float ambient = mlx.readAmbientTempC();
    float object  = mlx.readObjectTempC();
    float humidity = dht.readHumidity();
    bool dht_ok = !isnan(humidity);

    bool alert =
      ambient < LOW_AMBIENT_LIMIT ||
      ambient > HIGH_AMBIENT_LIMIT ||
      object  < LOW_OBJECT_LIMIT  ||
      object  > HIGH_OBJECT_LIMIT;

    // ========== TFT：只更新會變動的部分（不清整片、不重畫外框）==========
    // 狀態燈：標籤在 drawStaticUI 畫過了，這裡只更新燈的顏色（綠=OK / 紅=異常）
    drawStatusDot(STAT_DOT_X[0], STAT_Y, WiFi.status() == WL_CONNECTED, 6);
    drawStatusDot(STAT_DOT_X[1], STAT_Y, client.connected(), 6);
    drawStatusDot(STAT_DOT_X[2], STAT_Y, !alert, 6);

    // 數值（卡片右側）：setTextPadding 讓新字用背景色填滿固定寬度，直接
    // 蓋掉舊字，不需要先 fillRect 清除 → 沒有「先黑一下再出現」的閃爍
    tft.setTextDatum(TL_DATUM);
    tft.setTextPadding(140);

    String vals[3] = {
      String(object, 2) + " C",
      String(ambient, 2) + " C",
      dht_ok ? String(humidity, 1) + " %" : "-- %",
    };
    for (int i = 0; i < 3; i++) {
      tft.setTextColor(CARD_COLOR[i], CARD_BG);
      tft.drawString(vals[i], 80, CARD_Y[i] + 10, 4);
    }
    tft.setTextPadding(0);

    // Serial
    Serial.printf("Amb: %.2f  Obj: %.2f  Hum: %.1f\n",
      ambient, object, humidity);
    // SerialBT.printf("Amb: %.2f  Obj: %.2f  Hum: %.1f\n",   // [BT] 停用
    //   ambient, object, humidity);

    // MQTT 上傳（原專案 topic）
    if (client.connected()) {
      String payload = "{";
      payload += "\"ambient\":" + String(ambient, 2) + ",";
      payload += "\"object\":" + String(object, 2) + ",";
      payload += dht_ok ? "\"humidity\":" + String(humidity)
                        : "\"humidity\":null";
      payload += "}";
      client.publish(T_PUB_LEGACY_SENSOR, payload.c_str());
    }

    // MQTT 上傳（對齊 iot/ hub 的 topic）
    publishToHub(ambient, object, humidity, dht_ok);

    // 警報（原專案行為，保留）
    if (client.connected() && (now - lastAlertTime >= ALERT_INTERVAL)) {
      if (ambient < LOW_AMBIENT_LIMIT || ambient > HIGH_AMBIENT_LIMIT) {
        lastAlertTime = now;
        client.publish(T_PUB_LEGACY_ALERT, "Ambient Out Of Range");
      }
      if (object < LOW_OBJECT_LIMIT || object > HIGH_OBJECT_LIMIT) {
        lastAlertTime = now;
        client.publish(T_PUB_LEGACY_ALERT, "Object Out Of Range");
      }
    }
  }

  // WiFi / MQTT 連線維護放最後：即使斷線時 client.connect() 偶爾卡住
  // ~MQTT_SOCKET_TIMEOUT_SEC 秒，這一輪的 TFT / 感測 / OTA 也都已經跑完了，
  // 螢幕更新不會被重連拖慢（連線正常時 client.loop() 很快，無影響）。
  handleConnections();
}
