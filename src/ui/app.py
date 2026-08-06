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
