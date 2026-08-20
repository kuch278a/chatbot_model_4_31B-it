"""
Context Filter for RAG responses.
Filters out low-relevance documents, greetings, chit-chat, and duplicate context chunks.
"""

import re
from typing import List, Dict, Tuple

# Common Amharic & English greeting / chit-chat regex patterns
GREETING_PATTERNS = [
    r'^(?:ሰላም|ሰላምታ|ሰላም\s*ነው|ታዲያስ|ጤና\s*ይስጥልኝ|እንደምን\s*(?:ነህ|ነሽ|ኖት|ነዎት|አላችሁ|አደርክ|አደርሽ|አደራችሁ|አደሩ|ዋልክ|ዋልሽ|ዋላችሁ|ዋሉ|አመሻችሁ|አመሹ)|ደህና\s*(?:ነህ|ነሽ|አላችሁ|ነዎት|ኖት|ዋልክ|ዋሉ|አደርክ|አደሩ)|እንዴት\s*(?:ነህ|ነሽ|ኖት|ናችሁ|ነዎት|አለህ|አለሽ|አላችሁ|ነው)|አለው|ደህና\s*ነኝ|አመሰግናለሁ|እናመሰግናለን|በጣም\s*አመሰግናለሁ|ባይ|ቻው|ደህና\s*ሁኑ|ደህና\s*ሁን|ደህና\s*ሁኚ)\b',
    r'^(?:hi|hello|hey|greetings|good\s*(?:morning|afternoon|evening|day)|how\s*are\s*you|thanks|thank\s*you|bye|goodbye)\b',
]

GREETING_REGEX = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)


class ContextFilter:
    """
    Intelligent context filtering to ensure only high-confidence,
    relevant context is passed to the LLM for response generation.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.25,
        max_context_length: int = 1500,
        filter_greetings: bool = True
    ):
        self.min_score_threshold = min_score_threshold
        self.max_context_length = max_context_length
        self.filter_greetings = filter_greetings

    def is_greeting(self, query: str) -> bool:
        """Checks if a user query is a greeting or general pleasantry."""
        cleaned = re.sub(r'[^\w\s]', '', query.strip())
        words = cleaned.split()
        if len(words) <= 6 and GREETING_REGEX.search(cleaned):
            return True
        return False

    def is_fragment_or_too_short(self, query: str) -> bool:
        """Checks if a user query is a fragmented syllable, single character, or empty punctuation."""
        cleaned = re.sub(r'[^\w\u1200-\u137F]', '', query.strip())
        return len(cleaned) < 2

    def filter_contexts(
        self,
        query: str,
        retrieved_chunks: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Filters and ranks retrieved document chunks.

        Returns:
            (formatted_context_string, filtered_metadata_list)
        """
        if not retrieved_chunks:
            return "", []

        # 1. If query is a single fragment or too short, bypass context
        if self.is_fragment_or_too_short(query):
            print(f"  [Context Filter] Short fragment query ('{query}') — bypassing RAG context injection.")
            return "", []

        # 2. If query is a greeting or pleasantry, bypass context
        if self.filter_greetings and self.is_greeting(query):
            print(f"  [Context Filter] Greeting detected ('{query}') — bypassing RAG context injection.")
            return "", []

        filtered = []
        seen_texts = set()

        for chunk in retrieved_chunks:
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            # 2. Relevance Score Threshold Filter
            if score < self.min_score_threshold:
                continue

            # 3. Deduplication Filter
            norm_text = re.sub(r'\s+', ' ', text[:100])
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)

            filtered.append(chunk)

        if not filtered:
            print(f"  [Context Filter] All retrieved chunks fell below relevance threshold ({self.min_score_threshold:.2f}) — no context injected.")
            return "", []

        # 4. Format and Truncate context parts within max length
        context_parts = []
        metadata_list = []
        total_len = 0

        for i, res in enumerate(filtered):
            chunk_str = f"Source {i+1} ({res['document_name']}):\n{res['text']}"
            if total_len + len(chunk_str) > self.max_context_length and len(context_parts) > 0:
                break
            
            context_parts.append(chunk_str)
            metadata_list.append({
                "document_name": res["document_name"],
                "score": res["score"],
                "text": res["text"]
            })
            total_len += len(chunk_str)

        print(f"  [Context Filter] Accepted {len(metadata_list)}/{len(retrieved_chunks)} context chunks (top score: {metadata_list[0]['score']:.3f}).")
        context_str = "\n\n".join(context_parts)
        return context_str, metadata_list
