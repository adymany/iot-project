import socket
import sounddevice as sd
import numpy as np
import threading
import queue
import io
import wave
import time
import speech_recognition as sr

# --- CONFIGURATION ---
class Config:
    SERVER_IP = "127.0.0.1"  # Change to your server's IP
    SERVER_PORT = 5000
    
    # --- Audio Settings ---
    RECORDING_SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = 'int16'
    
    # --- Silence Detection Settings ---
    SILENCE_DURATION = 2.5
    SILENCE_THRESHOLD = 350
    
    # --- Keyword Detection Settings ---
    KEYWORD = "hey nexa"
    
    # --- Playback Settings for RAW PCM ---
    PLAYBACK_SAMPLE_RATE = 22050  # Piper's default output rate
    PLAYBACK_CHANNELS = 1
    
# --- GLOBAL VARIABLES ---
audio_queue = queue.Queue()
stop_recording_event = threading.Event()

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(f"Audio callback status: {status}", flush=True)
    audio_queue.put(bytes(indata))

def play_raw_pcm_stream(client_socket):
    """
    Receives RAW PCM audio stream from server and plays it in real-time.
    No WAV header - just raw audio data.
    """
    print("🎤 Server is responding...", flush=True)
    
    try:
        # Create output stream for real-time playback
        stream = sd.OutputStream(
            samplerate=Config.PLAYBACK_SAMPLE_RATE,
            channels=Config.PLAYBACK_CHANNELS,
            dtype='int16'
        )
        stream.start()
        
        # Receive and play audio chunks as they arrive
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            
            # Convert bytes to numpy array and play
            audio_np = np.frombuffer(data, dtype=np.int16)
            stream.write(audio_np)
        
        stream.stop()
        stream.close()
        print("✅ Response finished.", flush=True)

    except Exception as e:
        print(f"Error during audio playback: {e}")

def record_audio_with_silence_detection():
    """Records from the microphone and stops after a period of silence."""
    stop_recording_event.clear()
    audio_queue.queue.clear()

    print(f"\n🔴 Recording... Speak now. (Stops after {Config.SILENCE_DURATION}s of silence)")

    stream = sd.InputStream(
        samplerate=Config.RECORDING_SAMPLE_RATE,
        channels=Config.CHANNELS,
        dtype=Config.DTYPE,
        callback=audio_callback
    )
    stream.start()

    full_recording = []
    silent_chunks = 0
    silence_limit = int((Config.SILENCE_DURATION * Config.RECORDING_SAMPLE_RATE) / 1024)

    while not stop_recording_event.is_set():
        try:
            chunk = audio_queue.get(timeout=Config.SILENCE_DURATION + 1)
            full_recording.append(chunk)

            audio_chunk_np = np.frombuffer(chunk, dtype=np.int16)
            volume = np.abs(audio_chunk_np).mean()
            
            # Uncomment to tune your threshold:
            # print(f"Current volume: {volume:.2f}")

            if volume < Config.SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > silence_limit:
                print("🤫 Silence detected, stopping recording.")
                break

        except queue.Empty:
            print("🤫 Recording timed out.")
            break

    stream.stop()
    stream.close()
    return b''.join(full_recording)

def listen_for_keyword():
    """Prompt user to say the keyword and press Enter"""
    print("\n🎤 Say 'Hey Nexa' out loud, then press Enter to start recording...")
    print("   Or press Ctrl+C to exit the application")
    
    try:
        input()  # Wait for user to press Enter
        return True
    except KeyboardInterrupt:
        return False

def main():
    """Main function to run the voice client."""
    print("--- AI Voice Assistant Client ---")
    print(f"Server: {Config.SERVER_IP}:{Config.SERVER_PORT}")
    print(f"⚠️  NOTE: This client expects RAW PCM audio streaming (no WAV header)")
    print("🗣️  To use: Say 'Hey Nexa' to start recording (keyword detection active)")

    while True:
        client_socket = None
        try:
            # Listen for keyword
            if not listen_for_keyword():
                break
            
            recorded_data = record_audio_with_silence_detection()
            
            if not recorded_data or len(recorded_data) < 2048:
                print("No audio recorded, please try again.")
                continue

            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((Config.SERVER_IP, Config.SERVER_PORT))
            print("🤝 Connected to server.")

            print(f"📤 Sending {len(recorded_data)} bytes of audio...")
            client_socket.sendall(recorded_data)
            
            # Signal that we're done sending
            client_socket.shutdown(socket.SHUT_WR)

            # Receive and play the RAW PCM stream
            play_raw_pcm_stream(client_socket)

        except ConnectionRefusedError:
            print("❌ Connection refused. Is the server script running?")
            break
        except ConnectionResetError:
            print("🔌 Server closed the connection unexpectedly.")
        except KeyboardInterrupt:
            print("\n🛑 Exiting client.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            if client_socket:
                client_socket.close()

if __name__ == "__main__":
    main()