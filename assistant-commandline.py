import os
import argparse
import pathlib
import whisper
import google.generativeai as genai
import re
import subprocess

# --- CONFIGURATION ---
# 1. Set your Gemini API Key.
#    Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyC5-yJxBy0yQIAm7ZWEiPfVtx-XC3TMLJ0"

# 2. Configure Piper TTS Voice Model
#    Download a voice model from: https://huggingface.co/rhasspy/piper-voices/tree/main
#    Place the .onnx file in a 'voice_model' directory next to this script.
VOICE_MODEL_ONNX_PATH = "voice_model/en_US-hfc_female-medium.onnx"

# 3. Configure Whisper Model
#    Options: 'tiny', 'base', 'small', 'medium', 'large'
WHISPER_MODEL_NAME = "base"

# --- MAIN SCRIPT LOGIC ---

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using OpenAI Whisper.
    """
    print(f"Loading Whisper model '{WHISPER_MODEL_NAME}'...")
    model = whisper.load_model(WHISPER_MODEL_NAME)
    
    print(f"Transcribing audio file: {file_path}...")
    result = model.transcribe(file_path)
    
    transcribed_text = result["text"]
    print(f"Whisper STT successful. Text: '{transcribed_text}'")
    return transcribed_text

def get_gemini_response(prompt: str) -> str:
    """
    Gets a response from the Google Gemini API.
    """
    print("Connecting to Google Gemini...")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        response = model.generate_content(prompt)
        
        ai_response_text = response.text if response.text else ""
        print(f"Raw Gemini response received: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "I encountered an error while processing your request."

def synthesize_and_play_speech(text_to_speak: str, output_path: str):
    """
    Sanitizes text, generates audio using the piper command-line tool, and plays it.
    """
    # --- Sanitize the text from Gemini ---
    sanitized_text = re.sub(r'```.*?```', '', text_to_speak, flags=re.DOTALL)
    sanitized_text = re.sub(r'`[^`]*`', '', sanitized_text)
    sanitized_text = re.sub(r'[\*#]', '', sanitized_text)
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()

    print(f"Sanitized text for TTS: '{sanitized_text}'")

    if not sanitized_text:
        print("ERROR: No valid text to synthesize after cleaning. Skipping audio generation.")
        return

    # --- Synthesize using the piped command-line method ---
    print("\nSynthesizing audio using the piper command-line tool...")
    
    command = [
        "piper",
        "--model", VOICE_MODEL_ONNX_PATH,
        "--output_file", output_path,
        "--stdin"
    ]

    try:
        process = subprocess.run(
            command, 
            input=sanitized_text, 
            encoding='utf-8', 
            check=True,
            capture_output=True
        )
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
             print(f"\nSuccessfully saved synthesized audio to '{output_path}'")
             
             if os.name == 'nt': # Check for Windows
                 try:
                     print("Playing audio...")
                     os.startfile(output_path)
                 except Exception as e:
                     print(f"Error playing audio: {e}")
             else:
                 print(f"Audio generated. Please play the file manually: {os.path.abspath(output_path)}")
        else:
            print(f"\nERROR: Synthesis failed or resulted in an empty file.")
            print(f"Piper stderr: {process.stderr}")

    except FileNotFoundError:
        print("\nERROR: The 'piper' command was not found.")
    except subprocess.CalledProcessError as e:
        print("\nERROR: The piper command failed to execute.")
        print(f"Stderr: {e.stderr}")

def main():
    """
    Main function to run the STT -> LLM -> TTS -> Playback pipeline.
    """
    parser = argparse.ArgumentParser(description="AI Assistant Pipeline")
    parser.add_argument("audio_file", type=str, help="Path to the input MP3 audio file.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Input file not found at '{args.audio_file}'")
        return

    # 1. Speech-to-Text
    user_prompt = transcribe_audio(args.audio_file)
    
    # 2. Get AI Response
    ai_response = get_gemini_response(user_prompt)
    
    # 3. Text-to-Speech and Playback
    output_filename = "response.wav"
    synthesize_and_play_speech(ai_response, output_filename)
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
