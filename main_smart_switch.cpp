/*
  ESP8266 Blink with Blynk control
  Press a button in the Blynk app assigned to V1 to toggle blinking ON/OFF.
*/
#include <Arduino.h>
#include "secrets.h" // contains BLYNK_TEMPLATE_ID, BLYNK_TEMPLATE_NAME, BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS
#include <ESP8266WiFi.h>
#include <BlynkSimpleEsp8266.h>


// LED control
const int LED_PIN = 2; // GPIO2 (D4 on NodeMCU) - built-in LED
const int FAN_PIN = 5; // GPIO5 (D1 on NodeMCU) 
// ONLY ON/OFF control for FAN_PIN AND LED_PIN NO BRIGHTNESS CONTROL



// Blynk virtual pin handler (V1) - button in app sends 1 when pressed, 0 when released
// We use the button to set the LED ON/OFF directly instead of blinking.
BLYNK_WRITE(V1) {
bool pinValue = param.asInt(); // Get the value sent from the Blynk app
  if (pinValue) {
    // Button pressed - turn LED ON (active LOW)
    digitalWrite(LED_PIN, LOW); // LED ON
    
  } else {
    // Button released - turn LED OFF
    digitalWrite(LED_PIN, HIGH); // LED OFF
    
  }
}
BLYNK_WRITE(V2){
  bool pinValue = param.asInt(); // Get the value sent from the Blynk app
  if (pinValue) {
    // Button pressed - turn FAN ON
    digitalWrite(FAN_PIN, HIGH); // FAN ON
  } else {
    // Button released - turn FAN OFF
    digitalWrite(FAN_PIN, LOW); // FAN OFF
  }
}


void setup() {
  Serial.begin(115200);
  delay(10);
  pinMode(LED_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  digitalWrite(FAN_PIN, LOW); // Ensure FAN is OFF initially
  digitalWrite(LED_PIN, HIGH); // Ensure LED is OFF initially

  // Connect to Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);
}

void loop() {
  Blynk.run();
  // LED is controlled directly in BLYNK_WRITE(V1). Nothing else required here.
}
