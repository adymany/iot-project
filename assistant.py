
import os
import pathlib
import whisper
import google.generativeai as genai
import re
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import numpy as np
import queue
import json
import keyboard

# --- CONFIGURATION ---
# 1. Set your Gemini API Key.
#    Get your key from Google AI Studio: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "                 " # <--- IMPORTANT: PASTE YOUR KEY HERE

# 2. Configure Piper TTS Voice Model
#    Place the .onnx file in a 'voice_model' directory next to this script.
VOICE_MODEL_ONNX_PATH = "voice_model/en_US-hfc_female-medium.onnx"

# 3. Configure Whisper Model
# --- OPTIMIZATION: Using the 'tiny' model for faster performance ---
WHISPER_MODEL_NAME = "tiny"

# 4. Recording Settings
SAMPLE_RATE = 16000  # for Whisper compatibility
# VAD (Voice Activity Detection) settings
SILENCE_THRESHOLD = 300  # Adjust this based on your microphone's sensitivity
SILENCE_DURATION = 30  # Number of silent chunks before stopping (30 chunks * 50ms/chunk = 1.5 seconds of silence)
CHUNK_SIZE = int(SAMPLE_RATE * 0.05) # 50ms chunks

# --- SCRIPT LOGIC ---

def record_audio_in_memory(sample_rate: int):
    """
    Records audio from the default microphone into memory using VAD.
    Returns the audio data as a NumPy array.
    """
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio chunk."""
        if status:
            print(status, flush=True)
        q.put(indata.copy())

    print("\nListening... (recording will start automatically when you speak)")
    
    recorded_frames = []
    is_recording = False
    silent_chunks = 0
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=CHUNK_SIZE, callback=callback):
        while True:
            chunk = q.get()
            volume_norm = np.linalg.norm(chunk) * 10
            
            if volume_norm > SILENCE_THRESHOLD:
                if not is_recording:
                    print("Recording started...")
                    is_recording = True
                silent_chunks = 0
                recorded_frames.append(chunk)
            elif is_recording:
                silent_chunks += 1
                recorded_frames.append(chunk)
                if silent_chunks > SILENCE_DURATION:
                    print("Recording stopped due to silence.")
                    break
    
    if not recorded_frames:
        print("No speech detected.")
        return None

    # Concatenate all recorded frames into a single NumPy array
    recording = np.concatenate(recorded_frames, axis=0)
    # Convert to the float32 format that Whisper expects
    return recording.flatten().astype(np.float32) / 32768.0


def transcribe_audio_from_memory(model, audio_data: np.ndarray) -> str:
    """
    Transcribes audio data from a NumPy array using a pre-loaded Whisper model.
    """
    print(f"\nTranscribing audio...")
    result = model.transcribe(audio_data, fp16=False)
    
    transcribed_text = result["text"]
    print(f"Whisper STT successful. Text: '{transcribed_text}'")
    return transcribed_text

def get_gemini_response(prompt: str, chat_session) -> str:
    """
    Sends a prompt to the ongoing chat session to get a context-aware response.
    The chat_session object automatically manages the history.
    """
    print("\nSending prompt to Google Gemini...")
    try:
        response = chat_session.send_message(prompt)
        ai_response_text = response.text if response.text else ""
        print(f"Gemini response received: '{ai_response_text}'")
        return ai_response_text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "I encountered an error while processing your request."

def synthesize_and_play_speech(text_to_speak: str, output_path: str):
    """
    Sanitizes text, generates audio using the piper command-line tool, and plays it.
    """
    sanitized_text = re.sub(r'```.*?```', '', text_to_speak, flags=re.DOTALL)
    sanitized_text = re.sub(r'`[^`]*`', '', sanitized_text)
    sanitized_text = re.sub(r'[\*#]', '', sanitized_text)
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()

    print(f"\nSanitized text for TTS: '{sanitized_text}'")

    if not sanitized_text:
        print("ERROR: No valid text to synthesize after cleaning. Skipping audio generation.")
        return

    print("\nSynthesizing audio using Piper TTS...")
    
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
             
             try:
                 print("Playing audio... (Press SPACE to interrupt)")
                 data, fs = sf.read(output_path, dtype='float32')
                 sd.play(data, fs)
                 while sd.get_stream().active:
                     if keyboard.is_pressed('space'):
                         sd.stop()
                         print("\nPlayback interrupted by user.")
                         break
                     sd.sleep(100)
                 print("Playback finished.")
             except Exception as e:
                 print(f"Error playing audio: {e}")
        else:
            print(f"\nERROR: Synthesis failed or resulted in an empty file.")
            print(f"Piper stderr: {process.stderr}")

    except FileNotFoundError:
        print("\nERROR: The 'piper' command was not found. Make sure it's installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print("\nERROR: The piper command failed to execute.")
        print(f"Stderr: {e.stderr}")

def main():
    """
    Main function to run the conversational assistant in a loop.
    """
    # --- Initialization ---
    print(f"Loading Whisper model '{WHISPER_MODEL_NAME}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("Whisper model loaded.")

    print("\nConfiguring Gemini API...")
    try:
        if not GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in GEMINI_API_KEY:
            print("ERROR: Please set your GEMINI_API_KEY in the script.")
            return
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # Initialize the chat session, which will hold the conversation history
        chat_session = model.start_chat(history=[])
        print("Gemini chat session started successfully.")
    except Exception as e:
        print(f"FATAL: Failed to initialize Gemini API: {e}")
        return # Exit if Gemini fails to start

    print("\n🚀 AI Assistant activated. Say 'goodbye' to exit. 🚀")
    
    # --- Main Conversation Loop ---
    while True:
        # 1. Record Audio from user
        audio_data = record_audio_in_memory(SAMPLE_RATE)
        
        if audio_data is None:
            continue # Listen again if no speech was detected

        # 2. Transcribe user's speech to text
        user_prompt = transcribe_audio_from_memory(whisper_model, audio_data)
        
        if "goodbye" in user_prompt.lower():
            print("Exit command received. Shutting down.")
            synthesize_and_play_speech("Goodbye!", "response.wav")
            break

        # 3. Get AI Response using the chat session with history
        ai_response = get_gemini_response(user_prompt, chat_session)
        
        # 4. Synthesize AI's text response to speech and play it
        output_filename = "response.wav"
        synthesize_and_play_speech(ai_response, output_filename)
        
        print("\n----------------------------------")

    # --- Cleanup ---
    if os.path.exists("response.wav"):
        os.remove("response.wav")
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
