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

    def _get_system_prompt_for_query(self, prompt: str) -> str:
        if _is_amharic(prompt):
            return (
                self.base_system_prompt +
                "\nCRITICAL INSTRUCTION: The user is asking in Amharic. You MUST write your entire response in natural, fluent Amharic script (በአማርኛ ፊደላት). Do not switch to English."
            )
        else:
            return (
                self.base_system_prompt +
                "\nCRITICAL INSTRUCTION: The user is asking in English. You MUST write your response in clear, professional English."
            )

    def chat(self, session_id, prompt):
        # Retrieve history
        history = get_history(session_id, limit=10)
        
        # Retrieve RAG context
        context_str, sources = self.rag_pipeline.get_context(prompt, k=3)
        
        # Format user prompt with context
        if context_str:
            llm_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"
        else:
            llm_prompt = prompt
            
        system_prompt = self._get_system_prompt_for_query(prompt)

        # Generate response using LLM
        response = self.llm_client.generate(
            prompt=llm_prompt,
            system_prompt=system_prompt,
            history=history
        )
        
        # Save user message and assistant message to history
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", response)
        
        return {
            "response": response,
            "sources": sources
        }

    def chat_stream(self, session_id, prompt):
        history = get_history(session_id, limit=10)
        context_str, sources = self.rag_pipeline.get_context(prompt, k=3)
        
        if context_str:
            llm_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}"
        else:
            llm_prompt = prompt
            
        system_prompt = self._get_system_prompt_for_query(prompt)

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

