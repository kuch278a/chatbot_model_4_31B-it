import numpy as np
import json
from .connection import get_db_connection

class SQLiteVectorStore:
    def __init__(self):
        pass

    def add_document_chunks(self, chunks):
        """
        chunks is a list of dicts:
        {
            "id": str,
            "document_name": str,
            "text": str,
            "embedding": list of floats,
            "metadata": dict
        }
        """
        if not chunks:
            return
            
        conn = get_db_connection()
        try:
            for chunk in chunks:
                # Serialize embedding to float32 bytes
                emb_array = np.array(chunk["embedding"], dtype=np.float32)
                emb_bytes = emb_array.tobytes()
                meta_str = json.dumps(chunk.get("metadata", {}))
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO vector_store (id, document_name, text, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (chunk["id"], chunk["document_name"], chunk["text"], emb_bytes, meta_str)
                )
            conn.commit()
        finally:
            conn.close()

    def similarity_search(self, query_embedding, k=3):
        """
        query_embedding: list of floats or numpy array
        """
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT id, document_name, text, embedding, metadata FROM vector_store;")
            rows = cursor.fetchall()
        finally:
            conn.close()
            
        if not rows:
            return []
            
        # Parse query embedding
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            query_norm = 1.0
            
        results = []
        for row in rows:
            # Reconstruct embedding from bytes
            emb_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            emb_norm = np.linalg.norm(emb_vec)
            if emb_norm < 1e-9:
                emb_norm = 1.0
                
            # Cosine similarity
            similarity = np.dot(query_vec, emb_vec) / (query_norm * emb_norm)
            
            results.append({
                "id": row["id"],
                "document_name": row["document_name"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "score": float(similarity)
            })
            
        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
        
    def get_all_document_names(self):
        conn = get_db_connection()
        try:
            cursor = conn.execute("SELECT DISTINCT document_name FROM vector_store;")
            return [row["document_name"] for row in cursor.fetchall()]
        finally:
            conn.close()

    def clear_all(self):
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM vector_store;")
            conn.commit()
        finally:
            conn.close()
