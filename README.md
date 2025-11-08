# NEXA AI - Smart Voice Assistant IoT Project

NEXA AI is an intelligent voice assistant built with ESP32 microcontrollers and a Python server backend. This IoT project enables voice-controlled home automation with smart lighting and fan controls, powered by Google's Gemini AI for natural conversations.

## 📋 Project Overview

This project was developed as a Semester 5 IoT assignment by Adnan, Naresh, Urvik, and Chandan. It demonstrates a complete voice-controlled smart home system with the following features:

- Voice recognition using ESP32 microcontroller
- Speech-to-text conversion with Whisper AI
- Natural language processing with Google's Gemini AI
- Text-to-speech using Piper TTS engine
- Home automation control via Blynk IoT platform
- Real-time audio streaming between client and server

## 🏗️ System Architecture

```
[ESP32 Client] ←→ [Python Server] ←→ [Gemini AI] ←→ [Blynk IoT]
     ↑                                    ↓
   Voice                             Text Response
   Input                             ↓
                                     [Piper TTS]
                                           ↓
                                      Audio Output
```

## 📁 Project Structure

```
iot-project/
├── server_v4.py              # Main Python server with AI integration
├── main_assistant.cpp        # ESP32 client code for voice input/output
├── credentials.h             # WiFi credentials (not committed to repo)
├── .env                      # API keys (not committed to repo)
├── .gitignore                # Files excluded from git
├── piper/                    # Piper TTS executable files
├── voice_model/              # Voice model files for TTS
├── piper-tts-project/        # Standalone TTS project
├── recordings/               # Directory for recorded voice inputs
└── responses/                # Directory for AI responses
```

## 🚀 Getting Started

### Prerequisites

1. **Hardware Requirements:**
   - ESP32 development board
   - INMP441 MEMS microphone
   - MAX98357A I2S amplifier
   - Speaker (4Ω or 8Ω)
   - IR sensor for activation
   - LEDs for status indication
   - Jumper wires and breadboard

2. **Software Requirements:**
   - Python 3.8+
   - Arduino IDE
   - ESP32 board support for Arduino IDE
   - Required Python packages (see requirements)

### Installation

#### Server Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd iot-project
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is missing or incomplete, install the required packages manually:
   ```bash
   pip install faster-whisper google-generativeai soundfile noisereduce numpy requests python-dotenv
   ```

3. Set up environment variables:
   You can either:
   
   a. Manually rename and edit the files:
      - Rename `.env.example` to `.env`
      - Update the API keys in `.env` with your actual values:
        ```
        GEMINI_API_KEY=your_actual_gemini_api_key
        BLYNK_AUTH_TOKEN=your_actual_blynk_auth_token
        ```
   
   b. Or run the setup script:
      ```bash
      python setup_env.py
      ```
      This will automatically create the necessary files for you.

4. Set up ESP32 credentials:
   - Update the WiFi credentials in `credentials.h` with your actual values

4. Download voice models:
   - Download voice models from [Piper Voices](https://github.com/rhasspy/piper/releases/tag/v0.2.0)
   - Place the `.onnx` and `.onnx.json` files in the `voice_model/` directory

5. Run the server:
   ```bash
   python server_v4.py
   ```

#### ESP32 Client Setup

1. Open `main_assistant.cpp` in Arduino IDE
2. Install required libraries:
   - WiFi.h
   - WiFiClient.h
   - WiFiUdp.h
   - driver/i2s.h

3. Update credentials:
   - Edit `credentials.h` with your actual WiFi SSID and password

4. Flash the code to your ESP32 board

### Configuration

#### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
BLYNK_AUTH_TOKEN=your_blynk_auth_token_here
```

#### Hardware Pin Configuration

Update pin assignments in `main_assistant.cpp` according to your wiring:

```cpp
// Microphone I2S pins
#define I2S_MIC_SCK  21
#define I2S_MIC_WS   22
#define I2S_MIC_SD   4

// Speaker I2S pins
#define I2S_SPK_BCLK 19
#define I2S_SPK_LRCLK 18
#define I2S_SPK_DIN 26

// Other pins
#define IR_PIN 23
```

## 🔧 Usage

1. Power on the ESP32 client device
2. Ensure the Python server is running
3. Trigger voice input using the IR sensor
4. Speak your command clearly
5. Receive voice response from the AI assistant
6. Control connected devices through voice commands:
   - "Turn on the light"
   - "Turn off the fan"
   - "Toggle the light"
   - General questions and conversation

## 🌐 Supported Commands

### Home Automation
- Light control: "Turn on/off the light"
- Fan control: "Turn on/off the fan"
- Toggle devices: "Toggle the light/fan"

### General Queries
- Weather information
- Time and date
- General knowledge questions
- Conversational interactions

## 🔒 Security

API keys and credentials are stored in environment variables and credential files that are excluded from the repository:

- `.env` file contains API keys (Git ignored)
- `credentials.h` contains WiFi credentials (Git ignored)

Always ensure these files are not committed to public repositories.

## 🛠️ Troubleshooting

### Common Issues

1. **Server not detecting ESP32 client:**
   - Check WiFi connection on both devices
   - Verify network settings and firewall rules
   - Ensure both devices are on the same network

2. **Poor voice recognition quality:**
   - Adjust microphone sensitivity
   - Check ambient noise levels
   - Verify microphone wiring

3. **No audio output:**
   - Check speaker connections
   - Verify amplifier power supply
   - Confirm volume settings in code

### Logs and Debugging

The server outputs detailed logs to the console which can help diagnose issues:
- Connection status
- Audio processing steps
- AI interaction logs
- Blynk IoT communication

## 📦 Dependencies

### Python Libraries
- faster-whisper: Speech-to-text engine
- google-generativeai: Gemini AI integration
- soundfile: Audio file handling
- noisereduce: Audio noise reduction
- requests: HTTP requests to Blynk
- numpy: Numerical computations
- python-dotenv: Environment variable management
- tkinter: GUI interface

### ESP32 Libraries
- WiFi.h: WiFi connectivity
- I2S driver: Audio input/output
- Standard Arduino libraries

## 🤝 Contributing

This project was created as an academic assignment but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 👨‍💻 Authors

- **Adnan**
- **Naresh**
- **Urvik**
- **Chandan**

Developed as part of Semester 5 IoT curriculum.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google's Gemini AI team for the powerful language model
- Rhasspy community for the Piper TTS engine
- Blynk IoT platform for home automation integration
- Open-source community for various libraries and tools