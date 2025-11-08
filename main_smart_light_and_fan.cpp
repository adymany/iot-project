// Automatic Fan Speed Control using DHT11 + L298N + LCD (with enable pin control)

#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define DHTPIN 2
#define DHTTYPE DHT11

// Motor control pins
const uint8_t ENA = 9; // PWM
const uint8_t IN1 = 7;
const uint8_t IN2 = 6;

// Enable control pin
const uint8_t ENABLE_PIN = 4; // <<--- Connect switch or control signal here
const uint8_t read_light=12;
const uint8_t light_relay=13;

DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);

unsigned long lastRead = 0;
const unsigned long readInterval = 2000; // 2 seconds

const float TEMP_MIN = 27.0;
const float TEMP_MAX = 35.0;

void setup() {
  Serial.begin(9600);
  dht.begin();
  lcd.init();
  lcd.backlight();

  pinMode(read_light,INPUT);
  pinMode(light_relay,OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENABLE_PIN, INPUT); // Or INPUT_PULLUP depending on your wiring

  stopFan();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Fan Auto Control");
  lcd.setCursor(0, 1);
  lcd.print("Waiting for ON...");
  delay(1500);
  lcd.clear();
}

void loop() {
  if(digitalRead(read_light)){
    digitalWrite(light_relay,HIGH);
  }else{
    digitalWrite(light_relay,LOW);
  }
  bool enabled = digitalRead(ENABLE_PIN);

  if (!enabled) {
    // Turn everything off
    stopFan();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("System Disabled");
    lcd.setCursor(0, 1);
    lcd.print("Waiting for ON...");
    delay(500);
    return;
  }

  unsigned long now = millis();
  if (now - lastRead >= readInterval) {
    lastRead = now;

    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Sensor Error!");
      lcd.setCursor(0, 1);
      lcd.print("Fan Stopped");
      stopFan();
      delay(1500);
      return;
    }

    int pwm = tempToPwm(temperature);
    int percent = map(pwm, 0, 255, 0, 100);

    if (pwm == 0) stopFan();
    else runFan(pwm);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("T:");
    lcd.print(temperature, 1);
    lcd.print((char)223);
    lcd.print("C ");
    lcd.print("H:");
    lcd.print(humidity, 0);
    lcd.print("%");

    lcd.setCursor(0, 1);
    lcd.print("Fan:");
    lcd.print(percent);
    lcd.print("% PWM:");
    lcd.print(pwm);

    Serial.print("Temp: ");
    Serial.print(temperature);
    Serial.print(" C  Humidity: ");
    Serial.print(humidity);
    Serial.print(" %  PWM: ");
    Serial.println(pwm);
  }
}

void stopFan() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}

void runFan(int pwm) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, pwm);
}

int tempToPwm(float t) {
  if (t < TEMP_MIN) return 0;
  if (t >= TEMP_MAX) return 255;
  float ratio = (t - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
  int pwm = (int)(ratio * 255);
  return constrain(pwm, 0, 255);
}