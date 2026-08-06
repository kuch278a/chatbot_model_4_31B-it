from src.db.chat_history import add_message, get_history
from config import settings

class ConversationService:
    def __init__(self, llm_client, rag_pipeline):
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline
        self.system_prompt = (
            "You are አማኒ (Amani), a helpful and intelligent bilingual (Amharic and English) AI assistant.\n"
            "Answer the user's questions truthfully and concisely. If context is provided below, use it to answer "
            "the question. If the answer cannot be found in the context, use your general knowledge.\n"
            "Always respond in the same language as the user's query (Amharic if the query is in Amharic, "
            "English if the query is in English).\n"
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
            
        # Generate response using LLM
        response = self.llm_client.generate(
            prompt=llm_prompt,
            system_prompt=self.system_prompt,
            history=history
        )
        
        # Save user message and assistant message to history
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", response)
        
        return {
            "response": response,
            "sources": sources
        }
