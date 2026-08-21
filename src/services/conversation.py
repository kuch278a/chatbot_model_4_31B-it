import os
import re
from src.db.chat_history import add_message, get_history


def is_amharic_text(text: str) -> bool:
    """Returns True if text contains any Ethiopic / Ge'ez script characters."""
    return bool(re.search(r"[ሀ-፿]", text))


def _is_too_short_or_fragment(text: str) -> bool:
    """Checks if text contains less than 2 valid alphanumeric characters or fidelat."""
    cleaned = re.sub(r"[^\wሀ-፿]", "", text.strip())
    return len(cleaned) < 2


UNCLEAR_QUESTION_FALLBACK_AM = "ይቅርታ፣ ጥያቄዎ አልገባኝም ወይም የተቋረጠ ይመስላል። እባክዎ ጥያቄዎን በግልጽ እንደገና ይጠይቁኝ።"
UNCLEAR_QUESTION_FALLBACK_EN = "I'm sorry, I couldn't understand your question or it seemed incomplete. Could you please rephrase or ask your question clearly?"


def _clean_repetitive_intros(text: str, user_prompt: str) -> str:
    """Removes repetitive self-introductions from responses unless explicitly asked."""
    user_lower = user_prompt.lower()
    who_patterns = [
        "ማን ነህ", "ማን ነሽ", "ስምህ ማን ነው", "ስምሽ ማን ነው", "ስምህ ማን", "ስምሽ ማን",
        "ማን ልበል", "ስምሽ ማን ልበል", "ስምህ ማን ልበል", "ስምህን ንገረኝ", "ስምሽን ንገረኝ",
        "ስምህን", "ስምሽን", "ምን ልበል", "ስለ ራስህ", "ስለ ራስሽ", "who are you",
        "what is your name", "tell me about yourself", "about amani", "ስለ አማኒ",
        "ስምህ", "ስምሽ", "ማን ነሽ?"
    ]
    if any(p in user_lower for p in who_patterns):
        return text.strip()

    # Immediate clean greeting for pure greetings based on input language
    am_greetings = ["ሰላም", "ሰላም ነህ", "ሰላም ነሽ", "እንደምን ነህ", "እንደምን ነሽ", "እንደምን አለህ", "እንደምን አለሽ"]
    en_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "greetings"]

    p_clean = user_lower.strip("!?,. ")
    if p_clean in am_greetings:
        return "ሰላም! እንዴት ነዎት? በምን ልርዳዎት?"
    if p_clean in en_greetings:
        return "Hello! How can I assist you today?"

    # Clean Amharic intros
    cleaned = re.sub(r"^(?:ሰላም[!፣,\s]*)?(?:እኔ\s+)?አማኒ\s+(?:AI\s+|ኤአይ\s+)?ነኝ[፣,፤;።\s]*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:በኢትዮጵያ\s+አርቴ?ፊሻል\s+ኢንተለጀንስ\s+ኢንስቲትዩት\s*(?:\(EAII\))?\s*የተገነባሁ\s*ረዳት\s*ነኝ[፣,፤;።\s]*)+", "", cleaned, flags=re.IGNORECASE)

    # Clean English intros
    cleaned = re.sub(r"^(?:Hello[!,\s]*)?(?:I am|I'm)\s+Amani,?\s*(?:an?\s+AI\s+assistant\s+(?:developed\s+by\s+EAII)?)?[,\.\s]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[፣,፤;።\s\.\-]+", "", cleaned).strip()
    return cleaned if cleaned else text.strip()


class ConversationService:
    def __init__(self, llm_client, rag_pipeline):
        self.llm_client = llm_client
        self.rag_pipeline = rag_pipeline

    def _get_system_prompt(self) -> str:
        prompt_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "local_data", "system_prompt.txt"),
            "/mnt/data/local_data/system_prompt.txt"
        ]
        for path in prompt_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            return content
                except Exception as e:
                    print(f"Warning: Could not read system prompt from {path}: {e}")

        return "The assistant is Amani (አማኒ), an intelligent bilingual (Amharic & English) AI assistant developed by EAII."

    def chat(self, session_id, prompt):
        is_am = is_amharic_text(prompt)

        # Guard for single-letter or empty audio fragments
        if _is_too_short_or_fragment(prompt):
            fallback = UNCLEAR_QUESTION_FALLBACK_AM if is_am else UNCLEAR_QUESTION_FALLBACK_EN
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", fallback)
            return {
                "response": fallback,
                "sources": []
            }

        p_clean = prompt.lower().strip("!?,. ")
        am_greetings = ["ሰላም", "ሰላም ነህ", "ሰላም ነሽ", "እንደምን ነህ", "እንደምን ነሽ"]
        en_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "greetings"]

        if p_clean in am_greetings:
            greeting_resp = "ሰላም! እንዴት ነዎት? በምን ልርዳዎት?"
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", greeting_resp)
            return {"response": greeting_resp, "sources": []}

        if p_clean in en_greetings:
            greeting_resp = "Hello! How can I assist you today?"
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", greeting_resp)
            return {"response": greeting_resp, "sources": []}

        history = get_history(session_id, limit=6)
        context_str, sources = self.rag_pipeline.get_context(prompt, k=4)

        if context_str:
            if is_am:
                llm_prompt = f"አስፈላጊ መረጃዎች (Context):\n{context_str}\n\nየተጠቃሚው ጥያቄ: {prompt}"
            else:
                llm_prompt = f"Relevant Context Information:\n{context_str}\n\nUser Question: {prompt}"
        else:
            llm_prompt = prompt

        system_prompt = self._get_system_prompt()

        response = self.llm_client.generate(
            prompt=llm_prompt,
            system_prompt=system_prompt,
            history=history
        )

        cleaned_response = _clean_repetitive_intros(response, prompt)
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", cleaned_response)

        return {
            "response": cleaned_response,
            "sources": sources
        }

    def chat_stream(self, session_id, prompt):
        is_am = is_amharic_text(prompt)

        # Guard for single-letter or empty audio fragments
        if _is_too_short_or_fragment(prompt):
            fallback = UNCLEAR_QUESTION_FALLBACK_AM if is_am else UNCLEAR_QUESTION_FALLBACK_EN
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", fallback)
            yield fallback
            return

        p_clean = prompt.lower().strip("!?,. ")
        am_greetings = ["ሰላም", "ሰላም ነህ", "ሰላም ነሽ", "እንደምን ነህ", "እንደምን ነሽ"]
        en_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you", "greetings"]

        if p_clean in am_greetings:
            greeting_resp = "ሰላም! እንዴት ነዎት? በምን ልርዳዎት?"
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", greeting_resp)
            yield greeting_resp
            return

        if p_clean in en_greetings:
            greeting_resp = "Hello! How can I assist you today?"
            add_message(session_id, "user", prompt)
            add_message(session_id, "assistant", greeting_resp)
            yield greeting_resp
            return

        history = get_history(session_id, limit=6)
        context_str, sources = self.rag_pipeline.get_context(prompt, k=4)

        if context_str:
            if is_am:
                llm_prompt = f"አስፈላጊ መረጃዎች (Context):\n{context_str}\n\nየተጠቃሚው ጥያቄ: {prompt}"
            else:
                llm_prompt = f"Relevant Context Information:\n{context_str}\n\nUser Question: {prompt}"
        else:
            llm_prompt = prompt

        system_prompt = self._get_system_prompt()

        buffer = []
        is_buffered = True
        full_response = []

        for token in self.llm_client.generate_stream(
            prompt=llm_prompt,
            system_prompt=system_prompt,
            history=history
        ):
            full_response.append(token)
            if is_buffered:
                buffer.append(token)
                buf_str = "".join(buffer)
                if len(buf_str) > 45 or "\n" in buf_str:
                    cleaned_head = _clean_repetitive_intros(buf_str, prompt)
                    is_buffered = False
                    if cleaned_head:
                        yield cleaned_head
            else:
                yield token

        if is_buffered and buffer:
            cleaned_head = _clean_repetitive_intros("".join(buffer), prompt)
            if cleaned_head:
                yield cleaned_head

        complete_text = _clean_repetitive_intros("".join(full_response), prompt)
        add_message(session_id, "user", prompt)
        add_message(session_id, "assistant", complete_text)
