import os
import uuid
from .chunker import split_text
from src.db.vector_store import SQLiteVectorStore

class RAGPipeline:
    def __init__(self, embeddings_model, retriever=None, vector_store=None):
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store or SQLiteVectorStore()
        from .retriever import Retriever
        from .context_filter import ContextFilter
        self.retriever = retriever or Retriever(self.embeddings_model, self.vector_store)
        self.context_filter = ContextFilter(min_score_threshold=0.25, max_context_length=1500)

    def index_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
            
        filename = os.path.basename(filepath)
        print(f"Indexing file: {filename}")
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read file {filepath}: {e}")
            return False
            
        if not content.strip():
            print(f"File is empty: {filepath}")
            return False
            
        # Chunk the text
        chunks = split_text(content)
        if not chunks:
            return False

        # Replace any previously indexed chunks for this document (idempotent re-indexing)
        self.vector_store.delete_document_chunks(filename)

        # Embed the chunks
        embeddings = self.embeddings_model.embed_documents(chunks)
        
        # Add to vector store
        db_chunks = []
        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            db_chunks.append({
                "id": f"{filename}_{i}_{str(uuid.uuid4())[:8]}",
                "document_name": filename,
                "text": chunk_text,
                "embedding": emb,
                "metadata": {"source": filename, "chunk_index": i}
            })
            
        self.vector_store.add_document_chunks(db_chunks)
        print(f"Successfully indexed {len(db_chunks)} chunks from {filename}")
        return True

    def index_directory(self, directory_path):
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return
            
        active_filenames = set()
        indexed_count = 0
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            if os.path.isfile(filepath) and filename.endswith((".txt", ".md", ".json", ".csv")):
                active_filenames.add(filename)
                if self.index_file(filepath):
                    indexed_count += 1

        # Strict sync: Purge documents from DB that no longer exist in local_data directory
        existing_db_docs = self.vector_store.get_all_document_names()
        for doc in existing_db_docs:
            if doc not in active_filenames:
                print(f"Purging deleted document '{doc}' from vector store database...")
                self.vector_store.delete_document_chunks(doc)

        print(f"Indexed {indexed_count} files from {directory_path}")


    def get_context(self, query, k=4, min_score_threshold=None):
        """
        Retrieves and filters context for the given user query.
        Discards low-relevance documents, greetings, and duplicate chunks.
        """
        if min_score_threshold is not None:
            self.context_filter.min_score_threshold = min_score_threshold

        raw_results = self.retriever.retrieve(query, k=k)
        return self.context_filter.filter_contexts(query, raw_results)

