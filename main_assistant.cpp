#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>
#include "credentials.h"

// --- HARDWARE PINS (updated for new, unused connections) ---
#define I2S_MIC_SCK 21  // New: Safe, unused GPIO for bit clock
#define I2S_MIC_WS  22  // New: Safe, unused GPIO for word select
#define I2S_MIC_SD  4   // New: Safe, unused GPIO for serial data in
#define I2S_SPK_BCLK 19
#define I2S_SPK_LRCLK 18
#define I2S_SPK_DIN 26
#define IR_PIN 23

// --- AUDIO SETTINGS ---
#define MIC_SAMPLE_RATE   16000
#define SPK_SAMPLE_RATE   22050
#define BITS_PER_SAMPLE   I2S_BITS_PER_SAMPLE_16BIT
#define I2S_MIC_PORT      I2S_NUM_0
#define I2S_SPK_PORT      I2S_NUM_1

// --- TRANSACTION SETTINGS ---
#define NETWORK_BUFFER_SIZE 1024
#define RESPONSE_TIMEOUT_MS 20000 // 20 seconds for server response
#define SPEECH_THRESHOLD 400 // Amplitude threshold for speech detection (adjust based on mic)
#define SILENCE_DURATION_MS 2500 // Stop after 2.5 seconds of silence
#define MAX_RECORD_DURATION_MS 20000 // Max 20 seconds to prevent infinite recording
#define DISCOVERY_PORT 12345 // UDP port for server broadcasts

WiFiClient client;
WiFiUDP udp;

char server_ip[16] = {'\0'};
int server_port = 5000;
bool server_found = false;

// Correctly configures microphone and speaker with their different sample rates and buffers
void i2s_install_separated() {
    // Microphone I2S Config
    i2s_config_t i2s_mic_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = MIC_SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 9,
        .dma_buf_len = 1024,
        .use_apll = true,
        .tx_desc_auto_clear = false
    };

    // Speaker I2S Config
    i2s_config_t i2s_spk_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SPK_SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 16,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = true
    };

    i2s_pin_config_t mic_pin_config = {.bck_io_num = I2S_MIC_SCK, .ws_io_num = I2S_MIC_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = I2S_MIC_SD};
    i2s_pin_config_t spk_pin_config = {.bck_io_num = I2S_SPK_BCLK, .ws_io_num = I2S_SPK_LRCLK, .data_out_num = I2S_SPK_DIN, .data_in_num = I2S_PIN_NO_CHANGE};
    
    i2s_driver_install(I2S_MIC_PORT, &i2s_mic_config, 0, NULL);
    i2s_set_pin(I2S_MIC_PORT, &mic_pin_config);
    i2s_driver_install(I2S_SPK_PORT, &i2s_spk_config, 0, NULL);
    i2s_set_pin(I2S_SPK_PORT, &spk_pin_config);
}

void setup() {
    Serial.begin(115200);
    pinMode(IR_PIN, INPUT_PULLUP);
    pinMode(2, OUTPUT);
    digitalWrite(2, LOW);
    
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n✅ Connected! ESP32 IP: " + WiFi.localIP().toString());
    
    udp.begin(DISCOVERY_PORT);
    Serial.println("🔍 Listening for server broadcasts on UDP port " + String(DISCOVERY_PORT) + "...");
    
    i2s_install_separated();
    Serial.println("🎤 Mic & 🔊 Speaker Initialized.");
    Serial.println("Ready and waiting for IR trigger... (Server discovery in progress)");
}

// Handles one full AI interaction with VAD
void perform_ai_transaction() {
    if (!server_found) {
        Serial.println("❌ No server discovered yet. Skipping transaction.");
        return;
    }

    Serial.printf("Connecting to %s:%d...\n", server_ip, server_port);
    if (!client.connect(server_ip, server_port)) {
        Serial.println("❌ Connection failed.");
        return;
    }
    Serial.println("✅ Connected to server!");

    // Record until silence is detected or max duration reached
    Serial.println("Recording until silence is detected...");
    digitalWrite(2, HIGH); // Turn on LED to indicate recording
    long startTime = millis();
    long lastSpeechTime = startTime; // Track last time speech was detected
    bool isSpeaking = false;

    while (millis() - startTime < MAX_RECORD_DURATION_MS && 
           millis() - lastSpeechTime < SILENCE_DURATION_MS) {
        uint8_t i2s_read_buffer[NETWORK_BUFFER_SIZE * 2];
        size_t bytes_read = 0;
        i2s_read(I2S_MIC_PORT, i2s_read_buffer, sizeof(i2s_read_buffer), &bytes_read, portMAX_DELAY);

        if (bytes_read > 0) {
            int16_t mono_samples[NETWORK_BUFFER_SIZE / 2];
            int sample_count = 0;
            bool speechDetected = false;

            // Convert stereo to mono and check for speech (now reading left channel for SEL=GND)
            for (int i = 0; i < bytes_read; i += 4) {
                int16_t sample = *(int16_t*)(i2s_read_buffer + i); // Left channel
                mono_samples[sample_count++] = sample;

                // Check amplitude for speech
                if (abs(sample) > SPEECH_THRESHOLD) {
                    speechDetected = true;
                    lastSpeechTime = millis();
                    if (!isSpeaking) {
                        isSpeaking = true;
                        digitalWrite(2, HIGH); // Ensure LED is on
                    }
                }
            }

            // Send audio data to server
            if (sample_count > 0) {
                client.write((uint8_t*)mono_samples, sample_count * 2);
            }

            // Blink LED if no speech but still recording
            if (!speechDetected && isSpeaking) {
                if (millis() % 500 < 250) { // Blink every 500ms
                    digitalWrite(2, LOW);
                } else {
                    digitalWrite(2, HIGH);
                }
            }
        }
    }
    digitalWrite(2, LOW); // Turn off LED when recording stops
    Serial.println("✅ Finished recording (silence detected or max duration reached).");

    // Wait for a response - BLINKING LIGHT
    Serial.println("Waiting for response...");
    long response_start_time = millis();
    while (!client.available() && millis() - response_start_time < RESPONSE_TIMEOUT_MS) {
        digitalWrite(2, !digitalRead(2)); // Blink light
        delay(200); // Control blink speed
    }
    digitalWrite(2, LOW); // Turn off light after waiting is done

    if (client.available()) {
        // Handle the audio response (raw PCM, no header)
        Serial.println("🔊 Playing audio...");

        uint8_t mono_buffer[NETWORK_BUFFER_SIZE];
        uint8_t stereo_buffer[NETWORK_BUFFER_SIZE * 2];
        size_t total_bytes = 0;

        while (client.connected() || client.available()) {
            if (client.available()) {
                size_t bytes_read_from_net = client.read(mono_buffer, sizeof(mono_buffer));
                if (bytes_read_from_net > 0) {
                    total_bytes += bytes_read_from_net;
                    int16_t* mono_samples = (int16_t*)mono_buffer;
                    int16_t* stereo_samples = (int16_t*)stereo_buffer;
                    size_t mono_sample_count = bytes_read_from_net / 2;

                    float volume = 0.5;
                    for (int i = 0; i < mono_sample_count; i++) {
                        int16_t sample = mono_samples[i] * volume;
                        stereo_samples[i * 2] = sample;
                        stereo_samples[i * 2 + 1] = sample;
                    }
                    
                    size_t bytes_written = 0;
                    i2s_write(I2S_SPK_PORT, stereo_buffer, bytes_read_from_net * 2, &bytes_written, portMAX_DELAY);
                }
            } else {
                delay(10); // Avoid busy-waiting
            }
        }
        Serial.printf("✅ Audio stream finished. Total bytes: %d\n", total_bytes);
    } else {
        Serial.println("❌ Response timeout.");
    }

    // Disconnect
    Serial.println("Transaction complete. Disconnecting.");
    client.stop();
}

void loop() {
    // Check for server discovery if not found yet (check every loop for more responsiveness)
    if (!server_found) {
        int packet_size = udp.parsePacket();
        if (packet_size) {
            char packet_buffer[255];
            int len = udp.read(packet_buffer, 255);
            if (len > 0) {
                packet_buffer[len] = '\0';
                Serial.printf("📡 Received UDP packet: %s\n", packet_buffer); // Debug: Log received packets
                if (strncmp(packet_buffer, "AI_SERVER:", 10) == 0) {
                    char *rest = packet_buffer + 10;
                    char *colon = strchr(rest, ':');
                    if (colon) {
                        *colon = '\0';
                        strncpy(server_ip, rest, sizeof(server_ip) - 1);
                        server_port = atoi(colon + 1);
                        server_found = true;
                        Serial.printf("🔍 Server discovered: %s:%d\n", server_ip, server_port);
                    }
                }
            }
        }
    }
    
    if (digitalRead(IR_PIN) == LOW) {
        Serial.println("\n--- IR Trigger Detected! ---");
        perform_ai_transaction();
        Serial.println("--- Transaction Over. Waiting for next trigger... ---");
        delay(2000);
    }
    
    delay(50);
}