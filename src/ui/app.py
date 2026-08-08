import os
from flask import Blueprint, Response, request, jsonify, current_app
from src.ui.components import UIComponents

ui_bp = Blueprint("ui", __name__)

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
    """Endpoint for generating chatbot responses."""
    conv_service = current_app.config.get("CONVERSATION_SERVICE")
    if not conv_service:
        return jsonify({"error": "LLM client service not loaded yet. Please wait a moment."}), 503
        
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "default_session")
    
    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400
        
    try:
        result = conv_service.chat(session_id, prompt)
        return jsonify({
            "response": result["response"],
            "sources": result["sources"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
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

    if not prompt.strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400

    def generate():
        for token in conv_service.chat_stream(session_id, prompt):
            yield token

    return Response(stream_with_context(generate()), mimetype="text/plain")

@ui_bp.route("/api/voices", methods=["GET"])
def list_voices():
    """Returns available TTS voice profiles, focusing on Amharic (am-ET)."""
    from src.tts.synthesizer import VoiceSynthesizer
    synthesizer = VoiceSynthesizer()
    lang = request.args.get("lang")
    voices = synthesizer.get_available_voices(lang=lang)
    return jsonify({"status": "success", "voices": voices})

@ui_bp.route("/api/tts", methods=["POST"])
def synthesize_speech():
    """Endpoint to synthesize text into speech parameters for Amharic or target languages."""
    from src.tts.synthesizer import VoiceSynthesizer
    synthesizer = VoiceSynthesizer()
    data = request.get_json() or {}
    text = data.get("text", "")
    lang = data.get("lang", "am-ET")
    rate = data.get("rate", 1.0)
    pitch = data.get("pitch", 1.0)
    
    if not text.strip():
        return jsonify({"error": "Text to speak cannot be empty"}), 400

    result = synthesizer.synthesize(text=text, lang=lang, rate=rate, pitch=pitch)
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

    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400

    synthesizer = VoiceSynthesizer()
    temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    
    try:
        asyncio.run(synthesizer.generate_audio_file(text=text, output_path=temp_mp3.name, lang=lang))
        return send_file(temp_mp3.name, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@ui_bp.route("/refresh", methods=["POST"])
def refresh():
    """Rescans and index the documents folder."""
    rag_pipeline = current_app.config.get("RAG_PIPELINE")
    if not rag_pipeline:
        return jsonify({"error": "RAG pipeline not loaded"}), 503
        
    try:
        from config import settings
        rag_pipeline.index_directory(settings.INDEX_DIR)
        return jsonify({"status": "success", "message": "Rescanned and indexed document folder"})
    except Exception as e:
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
        return jsonify({"error": str(e)}), 500

@ui_bp.route("/clear", methods=["POST"])
def clear_history_route():
    """Clears the chat history for a session."""
    data = request.get_json() or {}
    session_id = data.get("session_id", "default_session")
    try:
        from src.db.chat_history import clear_history
        clear_history(session_id)
        return jsonify({"status": "success", "message": "Session history cleared"})
    except Exception as e:
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
    return flask_app
