import re
from src.db.vector_store import SQLiteVectorStore

EN_AM_EXPANSIONS = {
    'president': 'ፕሬዝዳንት',
    'presidents': 'ፕሬዝዳንቶች',
    'prime minister': 'ጠቅላይ ሚኒስትር',
    'minister': 'ሚኒስትር',
    'ministers': 'ሚኒስትሮች',
    'ministry': 'ሚኒስቴር',
    'ethiopia': 'ኢትዮጵያ',
    'ethiopian': 'የኢትዮጵያ',
    'history': 'ታሪክ',
    'director': 'ዳይሬክተር',
    'institute': 'ኢንስቲትዩት',
    'amani': 'አማኒ',
    'axum': 'አክሱም',
    'aksum': 'አክሱም',
    'lalibela': 'ላሊበላ',
    'gondar': 'ጎንደር',
    'politics': 'ፖለቲካ',
    'leader': 'መሪ',
    'leaders': 'መሪዎች',
}

class Retriever:
    def __init__(self, embeddings_model, vector_store=None):
        self.embeddings_model = embeddings_model
        self.vector_store = vector_store or SQLiteVectorStore()

    def _expand_query(self, query: str) -> str:
        """Expands cross-lingual English terms to Amharic to maximize embedding cosine alignment."""
        q_lower = query.lower()
        expansions = []
        for en, am in EN_AM_EXPANSIONS.items():
            if re.search(r'\b' + re.escape(en) + r'\b', q_lower):
                expansions.append(am)
        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    def retrieve(self, query, k=4):
        # Expand cross-lingual concepts for optimal multilingual alignment
        expanded_query = self._expand_query(query)
        # Compute query embedding
        query_emb = self.embeddings_model.embed_query(expanded_query)
        # Query vector store
        results = self.vector_store.similarity_search(query_emb, k=k)
        return results

