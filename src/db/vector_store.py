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

    def similarity_search(self, query_embedding, keywords=None, k=4):
        """
        query_embedding: list of floats or numpy array
        keywords: optional list or set of search tokens for lexical hybrid weighting
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
            cosine = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm))
            
            # Lexical keyword boost
            lexical_score = 0.0
            if keywords:
                text_lower = row["text"].lower()
                matched_count = 0
                for kw in keywords:
                    if kw in text_lower:
                        matched_count += 1
                    elif kw.endswith('s') and kw[:-1] in text_lower:
                        matched_count += 1
                lexical_score = (matched_count / len(keywords)) * 0.40

            combined_score = min(1.0, cosine + lexical_score)
            
            results.append({
                "id": row["id"],
                "document_name": row["document_name"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "score": float(combined_score),
                "cosine": float(cosine)
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

    def delete_document_chunks(self, document_name):
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM vector_store WHERE document_name = ?;", (document_name,))
            conn.commit()
        finally:
            conn.close()

    def clear_all(self):
        conn = get_db_connection()
        try:
            conn.execute("DELETE FROM vector_store;")
            conn.commit()
        finally:
            conn.close()
