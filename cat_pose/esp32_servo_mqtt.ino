#include <WiFi.h>
#include <PubSubClient.h>

// =========================================================
// Wi-Fi / MQTT Config
// =========================================================

const char* WIFI_SSID = "CBN-B4640-2.4G";
const char* WIFI_PASSWORD = "110106291208";

const char* MQTT_BROKER = "192.168.0.171";

const uint16_t MQTT_PORT = 1883;

const char* MQTT_CLIENT_ID = "cat-servo-esp32";

const char* MQTT_CMD_TOPIC = "cat/servo/cmd";
const char* MQTT_STATUS_TOPIC = "cat/servo/status";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// =========================================================
// Servo Config (ESP32 LEDC)
// =========================================================

const int servoPin = 4;

const int SERVO_CHANNEL = 0;
const int SERVO_FREQ = 50;
const int SERVO_RESOLUTION = 14;

// SG90 pulse range
const int SERVO_MIN_US = 500;
const int SERVO_MAX_US = 2500;

const int MIN_ANGLE = 10;
const int MAX_ANGLE = 170;

const int STEP_ANGLE = 2;

int angle = MIN_ANGLE;
int direction = 1;

bool catDetected = false;

unsigned long lastStepTime = 0;

const unsigned long STEP_INTERVAL_MS = 25;

unsigned long lastReconnectAttempt = 0;

const unsigned long MQTT_RECONNECT_INTERVAL_MS = 5000;

// =========================================================
// Servo Helper
// =========================================================

uint32_t angleToDuty(int angleDeg) {

  int pulseWidth = map(
    angleDeg,
    0,
    180,
    SERVO_MIN_US,
    SERVO_MAX_US
  );

  uint32_t duty = (pulseWidth * ((1 << SERVO_RESOLUTION) - 1)) / 20000;

  return duty;
}

void setServoAngle(int angleDeg) {

  angleDeg = constrain(angleDeg, 0, 180);

  uint32_t duty = angleToDuty(angleDeg);

 ledcWrite(servoPin, duty);
}

// =========================================================
// WiFi
// =========================================================

void connectWiFi() {

  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("Connecting WiFi...");

  WiFi.mode(WIFI_STA);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();

  while (
    WiFi.status() != WL_CONNECTED &&
    millis() - start < 15000
  ) {

    delay(300);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {

    Serial.println("WiFi Connected");
    Serial.println(WiFi.localIP());

  } else {

    Serial.println("WiFi Failed");
  }
}

// =========================================================
// MQTT
// =========================================================

void publishStatus(const char* msg) {

  if (mqttClient.connected()) {

    mqttClient.publish(
      MQTT_STATUS_TOPIC,
      msg,
      false
    );
  }
}

void handleCommand(const String& cmd) {

  if (cmd == "STOP") {

    catDetected = true;

    lastStepTime = millis();

    publishStatus("CAT DETECTED");

    return;
  }

  if (cmd == "START") {

    catDetected = false;

    lastStepTime = millis();

    publishStatus("SCAN START");

    return;
  }

  if (cmd == "LEFT") {

    catDetected = true;

    angle -= STEP_ANGLE;

    if (angle < MIN_ANGLE) {
      angle = MIN_ANGLE;
    }

    setServoAngle(angle);

    publishStatus("LEFT");

    return;
  }

  if (cmd == "RIGHT") {

    catDetected = true;

    angle += STEP_ANGLE;

    if (angle > MAX_ANGLE) {
      angle = MAX_ANGLE;
    }

    setServoAngle(angle);

    publishStatus("RIGHT");

    return;
  }
}

void mqttCallback(
  char* topic,
  byte* payload,
  unsigned int length
) {

  String incoming;

  for (unsigned int i = 0; i < length; i++) {

    incoming += (char)payload[i];
  }

  incoming.trim();

  Serial.print("MQTT: ");
  Serial.println(incoming);

  handleCommand(incoming);
}

void connectMQTT() {

  if (mqttClient.connected()) {
    return;
  }

  if (
    millis() - lastReconnectAttempt <
    MQTT_RECONNECT_INTERVAL_MS
  ) {
    return;
  }

  lastReconnectAttempt = millis();

  Serial.println("Connecting MQTT...");

  if (mqttClient.connect(MQTT_CLIENT_ID)) {

    Serial.println("MQTT Connected");

    mqttClient.subscribe(MQTT_CMD_TOPIC);

    publishStatus("ESP32 ONLINE");

  } else {

    Serial.println("MQTT Failed");
  }
}

// =========================================================
// Setup
// =========================================================

void setup() {

  Serial.begin(115200);

  // =====================================================
  // ESP32 LEDC Servo Setup
  // =====================================================

 ledcAttach(
  servoPin,
  SERVO_FREQ,
  SERVO_RESOLUTION
);

  setServoAngle(angle);

  // =====================================================
  // WiFi / MQTT
  // =====================================================

  connectWiFi();

  mqttClient.setServer(
    MQTT_BROKER,
    MQTT_PORT
  );

  mqttClient.setCallback(
    mqttCallback
  );

  Serial.println("System Ready");
}

// =========================================================
// Loop
// =========================================================

void loop() {

  // =====================================================
  // WiFi reconnect
  // =====================================================

  if (WiFi.status() != WL_CONNECTED) {

    connectWiFi();
  }

  // =====================================================
  // MQTT reconnect
  // =====================================================

  if (!mqttClient.connected()) {

    connectMQTT();

  } else {

    mqttClient.loop();
  }

  unsigned long now = millis();

  // =====================================================
  // Cat detected -> stop scan
  // =====================================================

  if (catDetected) {

    lastStepTime = now;

    return;
  }

  // =====================================================
  // Non-blocking scan sweep
  // =====================================================

  if (
    now - lastStepTime >= STEP_INTERVAL_MS
  ) {

    angle += direction * STEP_ANGLE;

    if (angle >= MAX_ANGLE) {

      angle = MAX_ANGLE;
      direction = -1;

    } else if (angle <= MIN_ANGLE) {

      angle = MIN_ANGLE;
      direction = 1;
    }

    setServoAngle(angle);

    lastStepTime = now;
  }
}