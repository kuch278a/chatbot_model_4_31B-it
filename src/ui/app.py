import os
import time
import threading
from datetime import datetime
from flask import Blueprint, Response, request, jsonify, current_app
from src.ui.components import UIComponents

ui_bp = Blueprint("ui", __name__)

# ── Global LLM lock — only one generation at a time (prevents CUDA OOM) ───────
_llm_lock = threading.Lock()

# ── Terminal color codes ───────────────────────────────────────────────────────
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "red":    "\033[91m",
    "dim":    "\033[2m",
    "blue":   "\033[94m",
}

def _ts():
    """Current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")

def _log_request(method, path, client_ip, session_id, prompt):
    preview = prompt[:80] + ('...' if len(prompt) > 80 else '')
    print(
        f"\n{_C['bold']}{_C['green']}▶ POST {path}{_C['reset']} "
        f"{_C['dim']}[{_ts()}]{_C['reset']}\n"
        f"  {_C['cyan']}Client :{_C['reset']} {client_ip}\n"
        f"  {_C['cyan']}Session:{_C['reset']} {session_id}\n"
        f"  {_C['cyan']}Prompt :{_C['reset']} {preview}",
        flush=True
    )

def _log_done(path, elapsed, token_count=None):
    extra = f" | {_C['yellow']}{token_count} tokens{_C['reset']}" if token_count else ""
    print(
        f"  {_C['green']}✔ {path} done{_C['reset']} in "
        f"{_C['yellow']}{elapsed:.2f}s{_C['reset']}{extra}\n",
        flush=True
    )

def _log_get(path, client_ip, params=None):
    param_str = f" | params: {params}" if params else ""
    print(
        f"\n{_C['bold']}{_C['yellow']}▶ GET  {path}{_C['reset']} "
        f"{_C['dim']}[{_ts()}]{_C['reset']}\n"
        f"  {_C['cyan']}Client :{_C['reset']} {client_ip}{param_str}",
        flush=True
    )

@ui_bp.route("/")
def index():
    """Serves the main chat UI HTML page."""
    return UIComponents.render_chat_page()

@ui_bp.route("/style.css")
def serve_css():
    """Serves the CSS styling file for the chat interface."""
    return Response(UIComponents.render_styles(), mimetype="text/css")

@ui_bp.route("/chat", methods=["POST"])
def chat():
    """Endpoint for generating chatbot responses (blocking)."""
    conv_service = current_app.config.get("CONVERSATION_SERVICE")
    if not conv_service:
        return jsonify({"error": "LLM client service not loaded yet. Please wait a moment."}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "default_session")
    client_ip = request.remote_addr

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    _log_request("POST", "/chat", client_ip, session_id, prompt)
    t0 = time.time()

    try:
        if not _llm_lock.acquire(timeout=300):  # Wait up to 5 min
            return jsonify({"error": "Server busy. Another request is being processed. Please try again."}), 503
        try:
            result = conv_service.chat(session_id, prompt)
        finally:
            _llm_lock.release()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        _log_done("/chat", time.time() - t0)
        return jsonify({
            "response": result["response"],
            "sources": result["sources"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  {_C['red']}✘ /chat error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500

@ui_bp.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Endpoint for streaming chatbot responses token-by-token."""
    from flask import stream_with_context
    conv_service = current_app.config.get("CONVERSATION_SERVICE")
    if not conv_service:
        return jsonify({"error": "LLM client service not loaded yet."}), 503

    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "default_session")
    client_ip = request.remote_addr

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    _log_request("POST", "/chat/stream", client_ip, session_id, prompt)
    t0 = time.time()

    # Check if LLM is already busy before starting the stream
    if not _llm_lock.acquire(timeout=300):  # Wait up to 5 min
        return jsonify({"error": "Server busy. Another request is being processed."}), 503

    print(f"  {_C['dim']}[queued — lock acquired]{_C['reset']}", flush=True)

    def generate():
        token_count = 0
        try:
            for token in conv_service.chat_stream(session_id, prompt):
                token_count += 1
                yield token
        finally:
            _llm_lock.release()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        _log_done("/chat/stream", time.time() - t0, token_count)

    return Response(stream_with_context(generate()), mimetype="text/plain")

@ui_bp.route("/api/stt", methods=["POST"])
def speech_to_text():
    """Transcribes audio using faster-whisper (distil-large-v3.5 int8 CPU)."""
    from src.stt.faster_whisper_transcriber import transcribe_webm

    client_ip = request.remote_addr
    audio_bytes = request.data  # Raw binary audio from browser MediaRecorder

    if not audio_bytes:
        return jsonify({"error": "No audio data received"}), 400

    _log_request("POST", "/api/stt", client_ip, "-", f"[audio {len(audio_bytes)//1024}KB]")
    t0 = time.time()

    try:
        transcript = transcribe_webm(audio_bytes)
        _log_done("/api/stt", time.time() - t0)
        print(f"  {_C['cyan']}Transcript:{_C['reset']} {transcript}", flush=True)
        return jsonify({"status": "success", "transcript": transcript})
    except Exception as e:
        print(f"  {_C['red']}✘ /api/stt error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500

@ui_bp.route("/api/voices", methods=["GET"])
def list_voices():
    """Returns available TTS voice profiles, focusing on Amharic (am-ET)."""
    from src.tts.synthesizer import VoiceSynthesizer
    lang = request.args.get("lang")
    synthesizer = VoiceSynthesizer()
    voices = synthesizer.get_available_voices(lang=lang)
    return jsonify({"status": "success", "voices": voices})

@ui_bp.route("/api/tts", methods=["POST"])
def synthesize_speech():
    """Endpoint to synthesize text into speech parameters for Amharic or target languages."""
    from src.tts.synthesizer import VoiceSynthesizer
    data = request.get_json() or {}
    text = data.get("text", "")
    lang = data.get("lang", "am-ET")
    rate = data.get("rate", 1.0)
    pitch = data.get("pitch", 1.0)

    _log_request("POST", "/api/tts", request.remote_addr, "-", text)
    t0 = time.time()

    if not text.strip():
        return jsonify({"error": "Text to speak cannot be empty"}), 400

    synthesizer = VoiceSynthesizer()
    result = synthesizer.synthesize(text=text, lang=lang, rate=rate, pitch=pitch)
    _log_done("/api/tts", time.time() - t0)
    return jsonify(result)

@ui_bp.route("/api/tts/audio", methods=["POST", "GET"])
def stream_speech_audio():
    """Endpoint that synthesizes text using edge-tts and streams the resulting MP3 audio file."""
    import asyncio
    import tempfile
    from flask import send_file
    from src.tts.synthesizer import VoiceSynthesizer

    if request.method == "GET":
        text = request.args.get("text", "እንኳን ወደ አማኒ ረዳት በደህና መጡ")
        lang = request.args.get("lang", "am-ET")
    else:
        data = request.get_json() or {}
        text = data.get("text", "እንኳን ወደ አማኒ ረዳት በደህና መጡ")
        lang = data.get("lang", "am-ET")
        _log_request("POST", "/api/tts/audio", request.remote_addr, "-", text)

    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    t0 = time.time()
    synthesizer = VoiceSynthesizer()
    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)

    try:
        asyncio.run(synthesizer.generate_audio_file(text=text, output_path=temp_mp3.name, lang=lang))
        _log_done("/api/tts/audio", time.time() - t0)
        return send_file(temp_mp3.name, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        print(f"  {_C['red']}✘ /api/tts/audio error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500



@ui_bp.route("/refresh", methods=["POST"])
def refresh():
    """Rescans and index the documents folder."""
    _log_request("POST", "/refresh", request.remote_addr, "-", "[rescan index]")
    t0 = time.time()
    rag_pipeline = current_app.config.get("RAG_PIPELINE")
    if not rag_pipeline:
        return jsonify({"error": "RAG pipeline not loaded"}), 503

    try:
        from config import settings
        rag_pipeline.index_directory(settings.INDEX_DIR)
        _log_done("/refresh", time.time() - t0)
        return jsonify({"status": "success", "message": "Rescanned and indexed document folder"})
    except Exception as e:
        print(f"  {_C['red']}✘ /refresh error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500

@ui_bp.route("/files", methods=["GET"])
def get_files():
    """Returns a list of all indexed files."""
    vector_store = current_app.config.get("VECTOR_STORE")
    if not vector_store:
        return jsonify({"error": "Vector store not loaded"}), 503

    try:
        files = vector_store.get_all_document_names()
        return jsonify({"files": files})
    except Exception as e:
        print(f"  {_C['red']}✘ /files error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500

@ui_bp.route("/clear", methods=["POST"])
def clear_history_route():
    """Clears the chat history for a session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default_session")
    _log_request("POST", "/clear", request.remote_addr, session_id, "[clear history]")
    t0 = time.time()
    try:
        from src.db.chat_history import clear_history
        clear_history(session_id)
        _log_done("/clear", time.time() - t0)
        return jsonify({"status": "success", "message": "Session history cleared"})
    except Exception as e:
        print(f"  {_C['red']}✘ /clear error: {e}{_C['reset']}", flush=True)
        return jsonify({"error": str(e)}), 500

def create_ui_app(flask_app=None):
    """
    Registers the UI blueprint on an existing Flask application
    or creates a standalone Flask application if none is provided.
    """
    if flask_app is None:
        from flask import Flask
        flask_app = Flask(__name__)

    flask_app.register_blueprint(ui_bp)
    from API.api_docs import docs_bp
    flask_app.register_blueprint(docs_bp, url_prefix="/API")
    return flask_app
