import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.models.embeddings import LocalMultilingualEmbeddings
from src.db.connection import init_db
from src.rag.pipeline import RAGPipeline

def main():
    print("Starting document indexing...")
    
    # Initialize database tables
    init_db()
    
    # Load embedding model
    embedding_model = LocalMultilingualEmbeddings(settings.EMBEDDING_MODEL_PATH)
    
    # Initialize pipeline
    pipeline = RAGPipeline(embedding_model)
    
    # Index directory
    pipeline.index_directory(settings.INDEX_DIR)
    print("Indexing completed!")

if __name__ == "__main__":
    main()
