import os
import sys
import time
import threading
from flask import Flask, jsonify

# Add current folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Dual logging to output.log and console
class LoggerTee:
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = LoggerTee("output.log")
sys.stderr = sys.stdout

from config import settings
from src.ui.app import create_ui_app

app = Flask(__name__)

# Register UI blueprints
create_ui_app(app)

# Global services container
services = {
    "llm_client": None,
    "conversation_service": None,
    "rag_pipeline": None,
    "vector_store": None
}

# Service loading state shared between the background loader and health endpoint
MAX_LOAD_ATTEMPTS = 5
LOAD_RETRY_DELAY_SECONDS = 15
service_state = {
    "status": "starting",  # starting | ready | error
    "attempts": 0,
    "last_error": None
}

def load_services_bg():
    while service_state["attempts"] < MAX_LOAD_ATTEMPTS:
        try:
            service_state["attempts"] += 1
            print(f"Starting background service initialization (attempt {service_state['attempts']}/{MAX_LOAD_ATTEMPTS})...")

            # 1. Initialize DB tables
            from src.db.connection import init_db
            init_db()

            # 2. Load Local Embedding Model
            from src.models.embeddings import LocalMultilingualEmbeddings
            print("Loading local multilingual sentence embeddings model...")
            embedding_model = LocalMultilingualEmbeddings(settings.EMBEDDING_MODEL_PATH, device="cuda:1")

            # 3. Load Vector Store and RAG pipeline
            from src.db.vector_store import SQLiteVectorStore
            from src.rag.pipeline import RAGPipeline
            vector_store = SQLiteVectorStore()
            rag_pipeline = RAGPipeline(embedding_model, vector_store=vector_store)

            # Index document directory
            rag_pipeline.index_directory(settings.INDEX_DIR)

            services["vector_store"] = vector_store
            services["rag_pipeline"] = rag_pipeline

            # Register intermediate RAG services in app context
            app.config["RAG_PIPELINE"] = rag_pipeline
            app.config["VECTOR_STORE"] = vector_store

            # 4. Load sharded Gemma 4 model
            if settings.LLM_BACKEND == "accelerate":
                from src.models.accelerate import AcceleratedGemma4LLMClient
                llm_client = AcceleratedGemma4LLMClient(settings.MODEL_PATH)
            else:
                from src.models.llm_client import Gemma4LLMClient
                llm_client = Gemma4LLMClient(settings.MODEL_PATH)
            services["llm_client"] = llm_client

            # 5. Load Conversation Service
            from src.services.conversation import ConversationService
            conv_service = ConversationService(llm_client, rag_pipeline)
            services["conversation_service"] = conv_service

            # Register full services in app context
            app.config["CONVERSATION_SERVICE"] = conv_service

            service_state["status"] = "ready"
            service_state["last_error"] = None
            print("All services loaded and ready!")
            return
        except Exception as e:
            service_state["status"] = "error"
            service_state["last_error"] = str(e)
            print(f"CRITICAL: Failed to load services: {e}")
            import traceback
            traceback.print_exc()
            if service_state["attempts"] < MAX_LOAD_ATTEMPTS:
                print(f"Retrying in {LOAD_RETRY_DELAY_SECONDS}s...")
                time.sleep(LOAD_RETRY_DELAY_SECONDS)

@app.route("/health")
def health():
    if services["conversation_service"] is not None:
        return jsonify({"status": "ready"}), 200
    if service_state["status"] == "error":
        return jsonify({
            "status": "error",
            "attempts": service_state["attempts"],
            "max_attempts": MAX_LOAD_ATTEMPTS,
            "error": service_state["last_error"]
        }), 500
    return jsonify({
        "status": "loading",
        "attempts": service_state["attempts"],
        "max_attempts": MAX_LOAD_ATTEMPTS
    }), 503

def main():
    # Start loading heavy model in a background thread to prevent server startup hang
    loading_thread = threading.Thread(target=load_services_bg)
    loading_thread.daemon = True
    loading_thread.start()

    # Start Flask development server
    print(f"Starting Amani AI server on {settings.HOST}:{settings.PORT}...")
    app.run(host=settings.HOST, port=settings.PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
