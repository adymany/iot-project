# voice_model/README.md

# Voice Model for Piper TTS

This directory contains the voice model files required for the Piper TTS tool to synthesize speech from text.

## Downloading the Voice Model

1. Visit the [Piper Voices Repository](https://huggingface.co/rhasspy/piper-voices/tree/main).
2. Choose a voice model that suits your needs. Each model typically consists of two files:
   - A `.onnx` file (the model itself).
   - A `.onnx.json` file (metadata for the model).

3. Download both files and place them in this `voice_model` directory.

## Usage

Ensure that the `VOICE_MODEL_ONNX_PATH` in the `src/tts_piper.py` script points to the correct `.onnx` file in this directory. The script will use this model to generate audio from the input text.

## Notes

- Make sure you have the necessary permissions to use the voice model.
- If you encounter any issues, refer to the main project documentation for troubleshooting tips.