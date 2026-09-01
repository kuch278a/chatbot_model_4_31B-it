"""
End-to-end voice conversation endpoint.

POST /tesfansh-api/api/v1/chat/talk
  Body : multipart/form-data
    audio      (required) – WAV/WebM/MP4/OGG audio file
    session_id (optional) – conversation session ID (default: "default_session")
    lang       (optional) – "auto" | "am-ET" | "en-US"  (default: "auto")

Flow:
  1. Read audio bytes from the "audio" field
  2. STT  – auto mode: Ethio-ASR first, Whisper fallback if empty
  3. LLM  – ConversationService.chat() with session history
  4. TTS  – edge-tts → MP3 temp file
  5. Return MP3 audio stream  (mimetype: audio/mpeg)
     X-Transcript and X-Response headers carry the intermediate text
     for debugging / client display without a second request.
"""

import os
import time
import asyncio
import tempfile
import threading

from flask import Blueprint, Response, request, jsonify, current_app, after_this_request, send_file
from datetime import datetime

talk_bp = Blueprint("talk", __name__)

# Shared lock re-exported from ui.app so both blueprints queue against the same LLM
# We import lazily inside the route to avoid circular-import issues at module load time.

# ── Terminal colors (same palette as ui/app.py) ───────────────────────────────
_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "red":    "\033[91m",
    "dim":    "\033[2m",
}

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _get_client_ip():
    if request.headers.get("X-Forwarded-For"):
        return request.headers.get("X-Forwarded-For").split(",")[0].strip()
    if request.headers.get("X-Real-IP"):
        return request.headers.get("X-Real-IP").strip()
    return request.remote_addr or "127.0.0.1"


@talk_bp.route("/tesfansh-api/api/v1/chat/talk", methods=["POST"])
def talk():
    """
    Single-shot audio-in / audio-out endpoint.

    Accepts a WAV (preferred) or any ffmpeg-decodable audio file via
    multipart form field "audio", returns an MP3 audio response.
    """
    # ── 0. Guard: services must be ready ─────────────────────────────────────
    conv_service = current_app.config.get("CONVERSATION_SERVICE")
    if not conv_service:
        return jsonify({"error": "Services not ready yet. Please wait and retry."}), 503

    client_ip  = _get_client_ip()
    session_id = request.form.get("session_id", "default_session")
    lang       = request.form.get("lang", "auto")

    print(
        f"\n{_C['bold']}{_C['cyan']}▶ POST /tesfansh-api/api/v1/chat/talk{_C['reset']} "
        f"{_C['dim']}[{_ts()}]{_C['reset']}\n"
        f"  {_C['cyan']}Client :{_C['reset']} {client_ip}\n"
        f"  {_C['cyan']}Session:{_C['reset']} {session_id} | lang={lang}",
        flush=True,
    )

    # ── 1. Extract audio bytes ────────────────────────────────────────────────
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "Missing 'audio' field in form-data."}), 400

    audio_bytes = audio_file.read()
    if len(audio_bytes) < 64:
        return jsonify({"error": "Audio file is empty or too short."}), 400

    print(
        f"  {_C['cyan']}Audio  :{_C['reset']} {len(audio_bytes) // 1024} KB  "
        f"(filename={audio_file.filename or 'unknown'})",
        flush=True,
    )

    t_total = time.time()

    # ── 2. STT ────────────────────────────────────────────────────────────────
    t_stt = time.time()
    transcript = ""
    detected_lang = "am-ET"

    try:
        if lang.startswith("en"):
            # Caller forced English → go straight to Whisper
            from src.stt.faster_whisper_transcriber import transcribe_webm
            transcript = transcribe_webm(audio_bytes, force_language="en")
            detected_lang = "en-US"
        else:
            # Default / auto: try Ethio-ASR (Amharic) first
            from src.stt.ethio_asr_transcriber import transcribe_audio_blob
            transcript = transcribe_audio_blob(audio_bytes)
            detected_lang = "am-ET"

            # If nothing came back and we're in auto mode, fall back to Whisper
            if not transcript.strip() and lang == "auto":
                try:
                    from src.stt.faster_whisper_transcriber import transcribe_webm_with_info
                    whisper_text, w_lang, _ = transcribe_webm_with_info(audio_bytes)
                    if whisper_text.strip():
                        transcript = whisper_text
                        detected_lang = "en-US" if w_lang == "en" else "am-ET"
                except Exception as w_err:
                    print(
                        f"  {_C['yellow']}Whisper fallback failed: {w_err}{_C['reset']}",
                        flush=True,
                    )
    except Exception as stt_err:
        print(f"  {_C['red']}✘ STT error: {stt_err}{_C['reset']}", flush=True)
        return jsonify({"error": f"Speech recognition failed: {stt_err}"}), 500

    stt_elapsed = time.time() - t_stt
    print(
        f"  {_C['green']}STT done{_C['reset']} in {_C['yellow']}{stt_elapsed:.2f}s{_C['reset']} "
        f"| lang={detected_lang} | transcript: {transcript[:80]}{'...' if len(transcript) > 80 else ''}",
        flush=True,
    )

    if not transcript.strip():
        return jsonify({"error": "Could not transcribe audio. Please speak clearly and try again."}), 422

    # ── 3. LLM ────────────────────────────────────────────────────────────────
    # Import the shared LLM lock from ui.app so we queue properly
    from src.ui.app import _llm_lock

    t_llm = time.time()
    if not _llm_lock.acquire(timeout=300):
        return jsonify({"error": "Server busy. Another request is being processed. Please retry."}), 503

    try:
        result = conv_service.chat(session_id, transcript)
    except Exception as llm_err:
        print(f"  {_C['red']}✘ LLM error: {llm_err}{_C['reset']}", flush=True)
        return jsonify({"error": f"Language model failed: {llm_err}"}), 500
    finally:
        _llm_lock.release()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    llm_elapsed = time.time() - t_llm
    llm_response = result.get("response", "")
    print(
        f"  {_C['green']}LLM done{_C['reset']} in {_C['yellow']}{llm_elapsed:.2f}s{_C['reset']} "
        f"| response: {llm_response[:100]}{'...' if len(llm_response) > 100 else ''}",
        flush=True,
    )

    # ── 4. TTS ────────────────────────────────────────────────────────────────
    from src.tts.synthesizer import VoiceSynthesizer

    t_tts = time.time()
    synthesizer = VoiceSynthesizer()

    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_mp3.name
    temp_mp3.close()

    try:
        asyncio.run(
            synthesizer.generate_audio_file(
                text=llm_response,
                output_path=temp_path,
                lang=detected_lang,
            )
        )
    except Exception as tts_err:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        print(f"  {_C['red']}✘ TTS error: {tts_err}{_C['reset']}", flush=True)
        return jsonify({"error": f"Speech synthesis failed: {tts_err}"}), 500

    tts_elapsed = time.time() - t_tts
    total_elapsed = time.time() - t_total
    print(
        f"  {_C['green']}TTS done{_C['reset']} in {_C['yellow']}{tts_elapsed:.2f}s{_C['reset']}\n"
        f"  {_C['bold']}{_C['green']}✔ /talk total{_C['reset']} {_C['yellow']}{total_elapsed:.2f}s{_C['reset']}\n",
        flush=True,
    )

    # ── 5. Stream MP3 back ────────────────────────────────────────────────────
    @after_this_request
    def _cleanup(response):
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return response

    response = send_file(temp_path, mimetype="audio/mpeg", as_attachment=False)

    # Attach intermediate text as headers so clients can show transcript/response
    # without needing a separate API call.
    response.headers["X-Transcript"] = transcript.encode("utf-8", errors="replace").decode("latin-1", errors="replace")
    response.headers["X-Response"]   = llm_response[:500].encode("utf-8", errors="replace").decode("latin-1", errors="replace")
    response.headers["X-Detected-Language"] = detected_lang

    return response
