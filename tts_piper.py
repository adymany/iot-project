import os
import pathlib
import subprocess
import re

# --- CONFIGURATION ---
# 1. Configure Piper TTS Voice Model
#    Download a voice model from: https://huggingface.co/rhasspy/piper-voices/tree/main
#    You need both the .onnx file and the .onnx.json file.
#    Place them in a 'voice_model' directory next to this script.
VOICE_MODEL_ONNX_PATH = "voice_model/en_US-hfc_female-medium.onnx"
OUTPUT_WAV_PATH = "output.wav"

def main():
    """
    Takes text input from the user, generates a WAV file by piping
    text to the piper command-line tool, and then plays the file.
    """
    
    # --- Check if voice model file exists ---
    voice_onnx_path = pathlib.Path(VOICE_MODEL_ONNX_PATH)

    if not voice_onnx_path.exists():
        print(f"ERROR: Piper voice model (.onnx file) not found at '{voice_onnx_path}'")
        print("Please ensure the 'voice_model' folder exists and contains the .onnx file.")
        return

    # --- Get user input ---
    text_to_speak = input("Please enter the text you want to convert to speech: ")

    if not text_to_speak.strip():
        print("No text entered. Exiting.")
        return

    # --- Synthesize the audio using the piped command-line method ---
    print("\nSynthesizing audio using the piper command-line tool...")
    
    # This command tells piper to use your voice model, read from standard input,
    # and save the output to a file.
    command = [
        "piper",
        "--model", str(voice_onnx_path),
        "--output_file", OUTPUT_WAV_PATH,
        "--stdin"
    ]

    try:
        # The 'input' argument sends our text to the command's standard input (the pipe).
        # 'check=True' will raise an error if the command fails.
        # 'capture_output=True' and 'text=True' help in debugging if something goes wrong.
        process = subprocess.run(
            command, 
            input=text_to_speak, 
            encoding='utf-8', 
            check=True,
            capture_output=True
        )
        
        # Final check to ensure the file was created and is not empty
        if os.path.exists(OUTPUT_WAV_PATH) and os.path.getsize(OUTPUT_WAV_PATH) > 44:
             print(f"\nSuccessfully saved synthesized audio to '{OUTPUT_WAV_PATH}'")
             
             # --- ADDED THIS BLOCK TO PLAY THE AUDIO ON WINDOWS ---
             if os.name == 'nt': # 'nt' is the name for Windows
                 try:
                     print("Playing audio...")
                     os.startfile(OUTPUT_WAV_PATH)
                 except Exception as e:
                     print(f"Error playing audio: {e}")
             else:
                 print(f"Audio generated. Please play the file manually: {os.path.abspath(OUTPUT_WAV_PATH)}")

        else:
            print(f"\nERROR: Synthesis failed or resulted in an empty file.")
            print(f"Piper stderr: {process.stderr}")

    except FileNotFoundError:
        print("\nERROR: The 'piper' command was not found.")
        print("Please ensure that the 'piper-tts' library is correctly installed and that its scripts are in your system's PATH.")
    except subprocess.CalledProcessError as e:
        print("\nERROR: The piper command failed to execute.")
        print(f"Return Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")


if __name__ == "__main__":
    main()
