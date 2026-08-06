from src.db.vector_store import SQLiteVectorStore

class Retriever:
    def __init__(self, embeddings_model, vector_store=None):
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store or SQLiteVectorStore()

    def retrieve(self, query, k=3):
        # Compute query embedding
        query_emb = self.embeddings_model.embed_query(query)
        # Query vector store
        results = self.vector_store.similarity_search(query_emb, k=k)
        return results
