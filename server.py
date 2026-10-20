"""Piper TTS server with CORS (allow all origins by default).

Replaces `python -m piper.http_server` and the separate proxy.

  POST /tts           JSON {"text","length_scale","gap_ms"}  -> finished WAV
  GET  /tts?text=...                                         -> finished WAV
  POST|GET /tts/stream   same fields                         -> WAV that starts playing early
  POST /              same as POST /tts (old Piper-style endpoint)
  GET  /health, /warm, /voices, /diag

Environment (all optional):
  MODEL           voice name, default en_US-lessac-medium
  ALLOWED_ORIGIN  default "*" (any website may call this server)
  API_KEY         if set, callers must send header "x-api-key" or ?key=
  MAX_CHARS       default 10000
  PORT            set automatically by Render
"""

import os
import struct
import threading

from flask import Flask, Response, jsonify, request
from piper import PiperVoice, SynthesisConfig

MODEL = os.environ.get("MODEL", "en_US-lessac-medium")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
API_KEY = os.environ.get("API_KEY", "")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "10000"))
PORT = int(os.environ.get("PORT", "5000"))


def find_model(name: str) -> str:
    """Look for the voice in the project folder and in ./voices."""
    name = name[:-5] if name.endswith(".onnx") else name
    for folder in (".", "voices", os.environ.get("VOICE_DIR", ".")):
        path = os.path.join(folder, name + ".onnx")
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"Voice '{name}' not found in . or ./voices. "
        f"Run: python -m piper.download_voices {name}"
    )


MODEL_PATH = find_model(MODEL)
voice = PiperVoice.load(MODEL_PATH)
RATE = voice.config.sample_rate
lock = threading.Lock()  # one synthesis at a time (protects RAM)

app = Flask(__name__)


# ---------- CORS + API key ----------
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, x-api-key"
    resp.headers["Access-Control-Expose-Headers"] = "Content-Type, Content-Length"
    return resp


@app.before_request
def preflight_and_auth():
    if request.method == "OPTIONS":
        return Response(status=204)
    if API_KEY and request.path not in ("/health",):
        key = request.headers.get("x-api-key") or request.args.get("key")
        if key != API_KEY:
            return Response("unauthorized", status=401)
    return None


# ---------- helpers ----------
def wav_header(rate: int, data_len=None) -> bytes:
    """data_len=None means unknown length (used for streaming)."""
    unknown = data_len is None
    big = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", big if unknown else 36 + data_len)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", big if unknown else data_len)
    )


def chunk_bytes(chunk) -> bytes:
    data = chunk.audio_int16_bytes
    return data() if callable(data) else data


def read_params():
    """Returns (paragraphs, SynthesisConfig, gap_ms) or raises ValueError."""
    if request.method == "GET":
        body = request.args
    else:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {"text": request.get_data(as_text=True)}

    text = str(body.get("text", "")).strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > MAX_CHARS:
        raise ValueError(f"max {MAX_CHARS} characters")

    cfg_kwargs = {}
    try:
        ls = float(body.get("length_scale", 0) or 0)
    except (TypeError, ValueError):
        ls = 0
    if 0.5 <= ls <= 2:
        cfg_kwargs["length_scale"] = ls
    cfg = SynthesisConfig(**cfg_kwargs)

    try:
        gap = float(body.get("gap_ms", 400))
    except (TypeError, ValueError):
        gap = 400
    gap = min(max(gap, 0), 1500)

    paragraphs = [p.strip() for p in text.replace("\r", "").split("\n") if p.strip()]
    return paragraphs, cfg, gap


def silence(gap_ms: float) -> bytes:
    return b"\x00\x00" * int(RATE * gap_ms / 1000)


# ---------- routes ----------
@app.route("/health")
def health():
    return "ok"


@app.route("/warm")
def warm():
    return "warm"


@app.route("/voices")
def voices():
    return jsonify({MODEL: {"sample_rate": RATE}})


@app.route("/diag")
def diag():
    return jsonify({"ok": True, "model": MODEL_PATH, "sample_rate": RATE,
                    "allowed_origin": ALLOWED_ORIGIN, "api_key_required": bool(API_KEY)})


@app.route("/", methods=["GET", "POST"])
@app.route("/tts", methods=["GET", "POST"])
def tts():
    try:
        paragraphs, cfg, gap = read_params()
    except ValueError as e:
        return Response(str(e), status=400)

    parts = []
    try:
        with lock:
            for para in paragraphs:
                for chunk in voice.synthesize(para, syn_config=cfg):
                    parts.append(chunk_bytes(chunk))
                parts.append(silence(gap))
    except Exception as e:  # noqa: BLE001
        return Response(f"synthesis failed: {e}", status=500)

    pcm = b"".join(parts)
    if not pcm:
        return Response("no audio produced", status=500)

    headers = {}
    if request.values.get("download"):
        headers["Content-Disposition"] = 'attachment; filename="speech.wav"'
    return Response(wav_header(RATE, len(pcm)) + pcm, mimetype="audio/wav", headers=headers)


@app.route("/tts/stream", methods=["GET", "POST"])
def tts_stream():
    try:
        paragraphs, cfg, gap = read_params()
    except ValueError as e:
        return Response(str(e), status=400)

    def generate():
        yield wav_header(RATE, None)
        with lock:
            for para in paragraphs:
                for chunk in voice.synthesize(para, syn_config=cfg):
                    yield chunk_bytes(chunk)
                yield silence(gap)

    return Response(
        generate(),
        mimetype="audio/wav",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print(f"Voice {MODEL_PATH} loaded, {RATE} Hz. Listening on :{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
