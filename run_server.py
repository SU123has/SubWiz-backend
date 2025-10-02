import os
import tempfile
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
from pyngrok import ngrok
import threading
import time
import torch  # used only for device detection

# ---------- Config ----------
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")  # change to tiny/base/small/medium
FLASK_HOST = "127.0.0.1"
FLASK_PORT = int(os.environ.get("PORT", 5000))
MAX_UPLOAD_MB = 100  # limit uploads
NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
# ----------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper-api")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# Globals
model = None
model_device = None
model_ready = False
transcribe_lock = threading.Lock()

def pick_device_and_compute_type():
    """Detect best device and safe compute type."""
    # GPU
    if torch.cuda.is_available():
        return "cuda", "float16"
    # Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "float32"
    # CPU
    return "cpu", "int8"

def load_whisper_model_blocking():
    """Load faster-whisper model (blocking) with safe settings."""
    global model, model_ready, model_device
    try:
        device, compute_type = pick_device_and_compute_type()
        model_device = device
        logger.info(f"Loading Whisper model '{MODEL_SIZE}' on device={device} compute_type={compute_type} ...")
        model = WhisperModel(
            MODEL_SIZE,
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count() or 4,
            num_workers=1
        )
        model_ready = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        model_ready = False
        logger.error("Failed to load model:\n" + traceback.format_exc())

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if model_ready else "loading" if model is None else "unhealthy",
        "model_loaded": bool(model_ready),
        "device": model_device
    }), 200 if model_ready else 503

@app.route("/model/info", methods=["GET"])
def model_info():
    if not model_ready:
        return jsonify({"success": False, "error": "Model not loaded"}), 503
    return jsonify({
        "success": True,
        "model_size": MODEL_SIZE,
        "device": model_device,
        "compute_type": getattr(model, "compute_type", None)
    })

@app.route("/translate", methods=["POST"])
def translate_audio():
    if not model_ready:
        return jsonify({"success": False, "error": "Model not loaded"}), 503

    audio_file = request.files.get("audio")
    start_time = request.form.get("start_time", "0")

    if not audio_file:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    # Save file safely using mkstemp (avoids locked file issues)
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        audio_file.save(tmp_path)
        size = os.path.getsize(tmp_path)
        logger.info(f"Received audio file saved to {tmp_path} size={size} bytes start_time={start_time}")

        # simple sanity check
        if size == 0:
            return jsonify({"success": False, "error": "Empty audio file"}), 400

        # Serialize heavy work: limit concurrency to 1 to avoid OOMs
        with transcribe_lock:
            start_t = time.time()
            try:
                segments, info = model.transcribe(
                    tmp_path,
                    beam_size=1,
                    task='translate',
                    language=None,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    word_timestamps=True
                )
                segs = []
                for seg in segments:
                    segs.append({
                        "start": float(getattr(seg, "start", 0.0)),
                        "end": float(getattr(seg, "end", 0.0)),
                        "text": getattr(seg, "text", "").strip(),
                        "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                        "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                    })
                elapsed = time.time() - start_t
                return jsonify({
                    "success": True,
                    "segments": segs,
                    "info": {
                        "language": getattr(info, "language", None),
                        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                        "duration": float(getattr(info, "duration", 0.0) or 0.0),
                        "processing_time": elapsed
                    },
                    "start_time": float(start_time)
                })
            except Exception as e:
                logger.error("Transcription failed:\n" + traceback.format_exc())
                return jsonify({"success": False, "error": f"Transcription failed: {str(e)}"}), 500

    finally:
        # Always attempt to cleanup tmp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            logger.warning("Failed to delete temp file: " + tmp_path)

def setup_ngrok_tunnel():
    try:
        if NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(FLASK_PORT)
        logger.info(f"Ngrok tunnel: {public_url}")
        return str(public_url)
    except Exception:
        logger.error("Failed to start ngrok tunnel:\n" + traceback.format_exc())
        return None

def main():
    # 1) Load model (blocking). If you want Flask to start immediately and model load async,
    #    run load_whisper_model_blocking in a daemon thread instead.
    load_whisper_model_blocking()
    if not model_ready:
        logger.error("Model failed to load. Exiting.")
        return

    # 2) Start ngrok (optional)
    public_url = setup_ngrok_tunnel()
    if public_url:
        logger.info(f"Health: {public_url}/health")
        logger.info(f"Translate: {public_url}/translate")

    # 3) Start Flask
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()