import os
import sys
import threading
from flask import Flask, jsonify

# Add current folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def load_services_bg():
    try:
        print("Starting background service initialization...")
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
        from src.models.llm_client import Gemma4LLMClient
        llm_client = Gemma4LLMClient(settings.MODEL_PATH)
        services["llm_client"] = llm_client
        
        # 5. Load Conversation Service
        from src.services.conversation import ConversationService
        conv_service = ConversationService(llm_client, rag_pipeline)
        services["conversation_service"] = conv_service
        
        # Register full services in app context
        app.config["CONVERSATION_SERVICE"] = conv_service
        
        print("All services loaded and ready!")
    except Exception as e:
        print(f"CRITICAL: Failed to load services: {e}")
        import traceback
        traceback.print_exc()

@app.route("/health")
def health():
    if services["conversation_service"] is not None:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({"status": "loading"}), 503

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
