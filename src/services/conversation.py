import re
from src.db.chat_history import add_message, get_history
from config import settings


def _is_amharic(text: str) -> bool:
    """Checks if text contains Ge'ez / Amharic Fidel script."""
    return bool(re.search(r'[\u1200-\u137F]', text))


class ConversationService:
    def __init__(self, llm_client, rag_pipeline):
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline
        self.base_system_prompt = (
            "You are አማኒ (Amani), a helpful and intelligent bilingual (Amharic and English) AI assistant.\n"
            "Answer the user's questions truthfully, accurately, and concisely. If context is provided below, use it to answer "
            "the question. If the answer cannot be found in the context, use your general knowledge.\n"
        )

    def _get_system_prompt(self) -> str:
        return (
            "You are አማኒ (Amani), a helpful, highly intelligent, and polite AI assistant created by the Ethiopian Artificial Intelligence Institute (EAII).\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. LANGUAGE: Write your entire response exclusively in natural, fluent Amharic script (በአማርኛ ፊደላት ብቻ).\n"
            "2. RAG CONTEXT PRIORITY: When 'አስፈላጊ መረጃዎች (Context)' is provided below, compare the user's question with it and treat the context as the absolute, authoritative source of truth. Always prioritize facts, names, dates, leadership info, and historical details from the context over any general pre-trained knowledge.\n"
            "3. ACCURACY & CONSISTENCY: If the context answers the user's question, base your response directly and faithfully on the context.\n"
            "4. FALLBACK: Only if the context is empty or completely uninformative, use your general knowledge to give a complete and accurate answer.\n"
            "5. CONCISENESS: Keep your answer concise, direct, and conversational (2-3 sentences max). Avoid writing unnecessarily long essays."
        )

    def chat(self, session_id, prompt):
        history = get_history(session_id, limit=6)
        context_str, sources = self.rag_pipeline.get_context(prompt, k=4)
        
        if context_str:
            llm_prompt = f"አስፈላጊ መረጃዎች (Context):\n{context_str}\n\nየተጠቃሚው ጥያቄ: {prompt}"
        else:
            llm_prompt = prompt
            
        system_prompt = self._get_system_prompt()

        response = self.llm_client.generate(
            prompt=llm_prompt,
            system_prompt=system_prompt,
            history=history
        )
        
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", response)
        
        return {
            "response": response,
            "sources": sources
        }

    def chat_stream(self, session_id, prompt):
        history = get_history(session_id, limit=6)
        context_str, sources = self.rag_pipeline.get_context(prompt, k=4)
        
        if context_str:
            llm_prompt = f"አስፈላጊ መረጃዎች (Context):\n{context_str}\n\nየተጠቃሚው ጥያቄ: {prompt}"
        else:
            llm_prompt = prompt
            
        system_prompt = self._get_system_prompt()

        full_response = []
        for token in self.llm_client.generate_stream(
            prompt=llm_prompt,
            system_prompt=system_prompt,
            history=history
        ):
            full_response.append(token)
            yield token
            
        complete_text = "".join(full_response)
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", complete_text)

