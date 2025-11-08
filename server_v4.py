import os
import pathlib
import io
import soundfile as sf
import subprocess
import re
import datetime
import logging
import socket
import tempfile
import requests
import threading
import queue
import time
import numpy as np
import noisereduce as nr
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
from dotenv import load_dotenv

from faster_whisper import WhisperModel
from google import genai
from google.genai import types

# Global conversation history: {client_ip: list of message dicts}
conversation_history = {}

# Queue for GUI updates (thread-safe)
message_queue = queue.Queue()

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN")
    SCRIPT_DIR = pathlib.Path(__file__).parent
    PIPER_EXECUTABLE_PATH = SCRIPT_DIR / "piper" / "piper.exe"
    VOICE_MODEL_ONNX_PATH = SCRIPT_DIR / "voice_model" / "en_US-hfc_female-medium.onnx"

    WHISPER_MODEL_NAME = "medium.en"
    RECORDINGS_DIR = "recordings"
    RESPONSES_DIR = "responses"
    SAVE_RECORDINGS = True
    HOST_IP = '0.0.0.0'
    PORT = 5000
    RECORDING_TIMEOUT = 1.5
    BROADCAST_PORT = 12345
    BROADCAST_INTERVAL = 5
    GEMINI_MODEL_NAME = 'gemini-2.0-flash'
    MAX_HISTORY_LENGTH = 20

# --- INITIALIZATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)
os.makedirs(Config.RESPONSES_DIR, exist_ok=True)

# --- CHECK FOR REQUIRED FILES ---
if not Config.PIPER_EXECUTABLE_PATH.exists():
    logging.error(f"❌ CRITICAL: Piper executable not found at '{Config.PIPER_EXECUTABLE_PATH}'")
    exit()
if not Config.VOICE_MODEL_ONNX_PATH.exists():
    logging.error(f"❌ CRITICAL: Voice model not found at '{Config.VOICE_MODEL_ONNX_PATH}'")
    exit()
logging.info("✅ Piper executable and voice model found.")

# --- LOAD AI MODELS ---
logging.info("Loading Whisper model...")
whisper_model = WhisperModel(Config.WHISPER_MODEL_NAME, device='cuda', compute_type='float16')
logging.info("✅ Whisper model loaded.")

logging.info("Configuring Gemini API with new google-genai SDK...")
if not Config.GEMINI_API_KEY or "YOUR_GEMINI_API_KEY" in Config.GEMINI_API_KEY:
    raise ValueError("❌ Please set your GEMINI_API_KEY in the Config class.")

client = genai.Client(api_key=Config.GEMINI_API_KEY)

grounding_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(
    tools=[grounding_tool],
    system_instruction="You are NEXA AI, a smart voice assistant built by Adnan, Naresh, Urvik, and Chandan as a Sem 5 IoT project. You're helpful, witty, and always ready to assist with home automation or general queries, make sure you know you are in india and question will have context of india. For voice responses, keep it natural, concise (under 100 words), and engaging for smooth audio playback. Maintain context from past chats. For IoT commands: If the user wants to turn on the light, start your response exactly with 'OK, turning the light on.' If turning off the light, start with 'OK, turning the light off.' For toggling the light, start with 'OK, toggling the light.' For fan: 'OK, turning the fan on.', 'OK, turning the fan off.', or 'OK, toggling the fan.' Then, continue with any additional friendly response if needed, but keep the command phrase at the very beginning for easy parsing, also make sure you use the singular from for light!! and fan !! dont say lights or fans or i will kill you."
)
logging.info(f"✅ Gemini client configured with '{Config.GEMINI_MODEL_NAME}'.")

def sanitize_text(text):
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'[\*#]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def update_history(client_ip, user_msg, model_msg):
    history = conversation_history.get(client_ip, [])
    history.append({'role': 'user', 'parts': [{'text': user_msg}]})
    history.append({'role': 'model', 'parts': [{'text': model_msg}]})
    if len(history) > Config.MAX_HISTORY_LENGTH:
        history = history[-Config.MAX_HISTORY_LENGTH:]
    conversation_history[client_ip] = history

def get_contextual_contents(client_ip, user_prompt):
    history = conversation_history.get(client_ip, [])
    contents = history + [{'role': 'user', 'parts': [{'text': user_prompt}]}]
    return contents

def control_blynk_light(action):
    if not Config.BLYNK_AUTH_TOKEN:
        logging.error("Blynk Auth Token is not configured.")
        return
    base_url = "https://blynk.cloud/external/api/"
    token = Config.BLYNK_AUTH_TOKEN
    LED_pin = "v1"
    try:
        if action == 'on':
            requests.get(f"{base_url}update?token={token}&{LED_pin}=1", timeout=5)
        elif action == 'off':
            requests.get(f"{base_url}update?token={token}&{LED_pin}=0", timeout=5)
        elif action == 'toggle':
            get_response = requests.get(f"{base_url}get?token={token}&{LED_pin}", timeout=5)
            current_value = int(get_response.text)
            new_value = 0 if current_value > 0 else 255
            requests.get(f"{base_url}update?token={token}&{LED_pin}={new_value}", timeout=5)
    except requests.exceptions.RequestException as e:
        logging.error(f"Blynk error: {e}")

def control_blynk_fan(action):
    if not Config.BLYNK_AUTH_TOKEN:
        logging.error("Blynk Auth Token is not configured.")
        return
    base_url = "https://blynk.cloud/external/api/"
    token = Config.BLYNK_AUTH_TOKEN
    FAN_pin = "v2"
    try:
        if action == 'on':
            requests.get(f"{base_url}update?token={token}&{FAN_pin}=1", timeout=5)
        elif action == 'off':
            requests.get(f"{base_url}update?token={token}&{FAN_pin}=0", timeout=5)
        elif action == 'toggle':
            get_response = requests.get(f"{base_url}get?token={token}&{FAN_pin}", timeout=5)
            current_value = int(get_response.text)
            new_value = 0 if current_value > 0 else 255
            requests.get(f"{base_url}update?token={token}&{FAN_pin}={new_value}", timeout=5)
    except requests.exceptions.RequestException as e:
        logging.error(f"Blynk error: {e}")

def generate_and_send_blocking(text, client_conn):
    logging.info(f"Generating blocking TTS for: '{text}'")
    clean_text = sanitize_text(text)
    piper_command = [
        str(Config.PIPER_EXECUTABLE_PATH),
        "--model", str(Config.VOICE_MODEL_ONNX_PATH),
        "--output-raw"
    ]
    try:
        with subprocess.Popen(piper_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            stdout, stderr = process.communicate(input=clean_text.encode('utf-8'))
            if process.returncode != 0:
                logging.error(f"Piper failed: {stderr.decode()}")
                return
            if stdout:
                client_conn.sendall(stdout)
                client_conn.shutdown(socket.SHUT_WR)
    except Exception as e:
        logging.error(f"Piper error: {e}")

def stream_tts_response(user_prompt, client_conn, client_ip):
    logging.info(f"Starting Gemini and Piper TTS streaming for {client_ip}...")
    contents = get_contextual_contents(client_ip, user_prompt)
    piper_command = [
        str(Config.PIPER_EXECUTABLE_PATH),
        "--model", str(Config.VOICE_MODEL_ONNX_PATH),
        "--output-raw"
    ]
    try:
        piper_process = subprocess.Popen(piper_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logging.error(f"Failed to start Piper: {e}")
        return

    def piper_to_client():
        try:
            while True:
                chunk = piper_process.stdout.read(1024)
                if not chunk:
                    break
                client_conn.sendall(chunk)
        except Exception as e:
            logging.error(f"Error streaming to client: {e}")

    threading.Thread(target=piper_to_client, daemon=True).start()

    full_response_text = ""
    sentence_buffer = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=Config.GEMINI_MODEL_NAME,
            contents=contents,
            config=config
        ):
            chunk_text = chunk.text or ""
            full_response_text += chunk_text
            sentence_buffer += chunk_text
            while True:
                match = re.search(r'([.?!])', sentence_buffer)
                if not match:
                    break
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx]
                sentence_buffer = sentence_buffer[end_idx:].lstrip()
                clean_sentence = sanitize_text(sentence)
                if clean_sentence:
                    piper_process.stdin.write((clean_sentence + "\n").encode('utf-8'))
                    piper_process.stdin.flush()
        if sentence_buffer.strip():
            clean_sentence = sanitize_text(sentence_buffer)
            piper_process.stdin.write((clean_sentence + "\n").encode('utf-8'))
            piper_process.stdin.flush()
    except Exception as e:
        logging.error(f"Gemini streaming error: {e}")
        full_response_text = "Sorry, I had trouble processing that."
    finally:
        if piper_process.stdin and not piper_process.stdin.closed:
            piper_process.stdin.close()
        update_history(client_ip, user_prompt, full_response_text)

    # Parse response for IoT commands and trigger Blynk
    response_upper = full_response_text.upper()
    if "OK, TURNING THE LIGHT ON" in response_upper:
        control_blynk_light('on')
        logging.info("💡 Light turned ON via AI response.")
    elif "OK, TURNING THE LIGHT OFF" in response_upper:
        control_blynk_light('off')
        logging.info("💡 Light turned OFF via AI response.")
    elif "OK, TOGGLING THE LIGHT" in response_upper:
        control_blynk_light('toggle')
        logging.info("💡 Light toggled via AI response.")
    elif "OK, TURNING THE FAN ON" in response_upper:
        control_blynk_fan('on')
        logging.info("  Fan turned ON via AI response.")
    elif "OK, TURNING THE FAN OFF" in response_upper:
        control_blynk_fan('off')
        logging.info("  Fan turned OFF via AI response.")
    elif "OK, TOGGLING THE FAN" in response_upper:
        control_blynk_fan('toggle')
        logging.info("  Fan toggled via AI response.")

    print(f"\n🧠 Gemini response for {client_ip}:\n{full_response_text}\n")

    # Queue message for GUI
    message_queue.put(('ai', user_prompt, full_response_text))

    piper_process.wait()
    client_conn.shutdown(socket.SHUT_WR)
    logging.info(f"✅ TTS streaming finished for {client_ip}.")

def process_and_reply(audio_data, client_conn, addr):
    client_ip = addr[0]
    if not audio_data or len(audio_data) < 2048:
        logging.warning("No audio received.")
        return
    logging.info(f"Processing {len(audio_data)} bytes from {addr}.")

    try:
        audio_input, _ = sf.read(io.BytesIO(audio_data), dtype='float32', samplerate=16000, channels=1, format='RAW', subtype='PCM_16')

        # --- 🔇 Apply noise reduction ---
        try:
            logging.info("🔇 Applying noise reduction...")
            audio_input = np.asarray(audio_input, dtype=np.float32)
            reduced_audio = nr.reduce_noise(y=audio_input, sr=16000)
            audio_input = reduced_audio
            logging.info("✅ Noise reduction complete.")
        except Exception as e:
            logging.warning(f"Noise reduction failed: {e}. Using raw audio.")

        if Config.SAVE_RECORDINGS:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sf.write(f"{Config.RECORDINGS_DIR}/rec_{ts}.wav", audio_input, 16000)

        segments, _ = whisper_model.transcribe(audio_input, beam_size=7, temperature=0.0)
        user_prompt = "".join(seg.text for seg in segments).strip()
        logging.info(f"🎙️ Transcription: '{user_prompt}'")

        if not user_prompt or user_prompt.lower() in ["thank you", "thanks for watching"]:
            ai_response = "I didn't hear anything, please try again."
            generate_and_send_blocking(ai_response, client_conn)
            update_history(client_ip, "[silence]", ai_response)
            # Queue for GUI
            message_queue.put(('user', "[silence]", ai_response))
            return

        # Queue user message for GUI
        message_queue.put(('user', user_prompt, None))

        stream_tts_response(user_prompt, client_conn, client_ip)

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        fallback = "Sorry, something went wrong."
        generate_and_send_blocking(fallback, client_conn)
        update_history(client_ip, "[error]", fallback)
        # Queue for GUI
        message_queue.put(('user', "[error]", fallback))

def handle_client(conn, addr):
    try:
        audio_buffer = bytearray()
        conn.settimeout(Config.RECORDING_TIMEOUT)
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                audio_buffer.extend(data)
        except socket.timeout:
            logging.info("🎤 Recording timeout reached.")
        if audio_buffer:
            process_and_reply(bytes(audio_buffer), conn, addr)
    finally:
        conn.close()

def broadcast_server(local_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    message = f"AI_SERVER:{local_ip}:{Config.PORT}".encode()
    logging.info(f"📡 Broadcasting server info: {message.decode()} every {Config.BROADCAST_INTERVAL}s")
    while True:
        sock.sendto(message, ('255.255.255.255', Config.BROADCAST_PORT))
        time.sleep(Config.BROADCAST_INTERVAL)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Use a reliable external endpoint to detect the local IP (more robust than 10.255.255.255)
        s.connect(("8.8.8.8", 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# --- TKINTER GUI ---
class ChatWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("NEXA AI Chat")
        self.root.geometry("500x600")
        self.root.configure(bg='#f0f0f0')

        # Chat display
        self.chat_frame = tk.Frame(root, bg='#f0f0f0')
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.chat_text = scrolledtext.ScrolledText(
            self.chat_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED, 
            bg='#ffffff', 
            font=('Arial', 10),
            relief=tk.FLAT,
            bd=0,
            height=30
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for styling
        self.chat_text.tag_configure('user', justify='right', foreground='#007bff', font=('Arial', 10, 'bold'))
        self.chat_text.tag_configure('ai', justify='left', foreground='#6c757d', font=('Arial', 10))
        self.chat_text.tag_configure('timestamp', foreground='#999', font=('Arial', 8))

        # Check queue periodically
        self.check_queue()

    def add_message(self, sender, text, timestamp=None):
        self.chat_text.config(state=tk.NORMAL)
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        self.chat_text.insert(tk.END, f"[{timestamp}] ", 'timestamp')
        
        if sender == 'user':
            self.chat_text.insert(tk.END, f"You: {text}\n\n", 'user')
        else:
            self.chat_text.insert(tk.END, f"NEXA AI: {text}\n\n", 'ai')
        
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def check_queue(self):
        try:
            while True:
                msg_type, user_text, ai_text = message_queue.get_nowait()
                if msg_type == 'user':
                    self.add_message('user', user_text)
                else:  # 'ai'
                    self.add_message('ai', ai_text)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)  # Poll every 100ms

# Launch GUI
root = tk.Tk()
chat_window = ChatWindow(root)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((Config.HOST_IP, Config.PORT))
server_socket.listen(1)

local_ip = get_local_ip()
logging.info("----------------------------------------------------")
logging.info(f"🚀 Server started on {local_ip}:{Config.PORT}")
logging.info("----------------------------------------------------")

# Start server in a thread so GUI runs in main
def run_server():
    broadcast_thread = threading.Thread(target=broadcast_server, args=(local_ip,), daemon=True)
    broadcast_thread.start()

    try:
        while True:
            conn, addr = server_socket.accept()
            logging.info(f"🤝 Connected by {addr}")
            threading.Thread(target=handle_client, args=(conn, addr)).start()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down.")
    finally:
        server_socket.close()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Run GUI in main thread
root.mainloop()