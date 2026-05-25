#include <Servo.h>

Servo myServo;

const int servoPin = 4;

const int MIN_ANGLE = 10;
const int MAX_ANGLE = 170;

int angle = MIN_ANGLE;
int direction = 1;

bool catDetected = false;

unsigned long lastStepTime = 0;
const unsigned long STEP_INTERVAL_MS = 25; // smooth movement interval

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(10);
  myServo.attach(servoPin);
  myServo.write(angle);
}

void loop() {
  // ===== 新增區塊：LEFT / RIGHT / STOP / START command handling =====
  // Read serial (non-blocking check)
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "STOP") {
      catDetected = true;
      Serial.println("CAT DETECTED");
    } else if (cmd == "LEFT") {
      catDetected = true;
      angle -= 2;
      if (angle < MIN_ANGLE) {
        angle = MIN_ANGLE;
      }
      myServo.write(angle);
      Serial.print("LEFT ANGLE: ");
      Serial.println(angle);
      delay(25);
      return;
    } else if (cmd == "RIGHT") {
      catDetected = true;
      angle += 2;
      if (angle > MAX_ANGLE) {
        angle = MAX_ANGLE;
      }
      myServo.write(angle);
      Serial.print("RIGHT ANGLE: ");
      Serial.println(angle);
      delay(25);
      return;
    } else if (cmd == "START") {
      catDetected = false;
      Serial.println("SCAN START");
    }
  }

  unsigned long now = millis();

  // If cat detected, do not sweep
  if (catDetected) {
    // keep servo at current angle
    lastStepTime = now;
    delay(25);
    return;
  }

  // ===== 新增區塊：non-blocking scan sweep =====
  if (now - lastStepTime >= STEP_INTERVAL_MS) {
    angle += direction;
    if (angle >= MAX_ANGLE) {
      angle = MAX_ANGLE;
      direction = -1;
    } else if (angle <= MIN_ANGLE) {
      angle = MIN_ANGLE;
      direction = 1;
    }
    myServo.write(angle);
    lastStepTime = now;
    delay(25);
  }
}
