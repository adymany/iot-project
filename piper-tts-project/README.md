# piper-tts-project

## Overview
This project utilizes the Piper TTS (Text-to-Speech) tool to synthesize speech from text input. It allows users to convert written text into spoken audio, which can be played back directly from the application.

## Project Structure
```
piper-tts-project
├── src
│   ├── tts_piper.py       # Main script for text-to-speech synthesis
│   └── utils.py           # Utility functions for the project
├── voice_model
│   └── README.md          # Instructions for setting up the voice model
├── requirements.txt       # List of required Python packages
└── README.md              # Project documentation
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd piper-tts-project
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Voice Model**
   - Visit [Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/main) to download a voice model.
   - Place the downloaded `.onnx` and `.onnx.json` files in the `voice_model` directory.

## Usage
1. Run the main script to start the text-to-speech synthesis:
   ```bash
   python src/tts_piper.py
   ```

2. Enter the text you wish to convert to speech when prompted.

3. The synthesized audio will be saved as `output.wav` and played automatically.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.