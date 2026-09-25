"""
LLM Post-Transcription Correction for Amharic STT.
Uses local Gemma 4 model to correct ASR errors in Amharic Ge'ez script.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global singleton instance
_corrector_instance = None
_DEFAULT_MODEL_PATH = os.environ.get("AMHARIC_CORRECTOR_MODEL_PATH", "/mnt/data/biruk/models/gemma-4-31b-it")


def _load_asr_correction_prompt() -> str:
    """Load ASR correction prompt from local_data/system_prompt.txt."""
    prompt_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "local_data", "system_prompt.txt"),
        "/mnt/data/local_data/system_prompt.txt"
    ]
    for path in prompt_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    # Extract the ASR correction section (after the main prompt)
                    if "<system_integrity_and_security>" in content:
                        parts = content.split("<system_integrity_and_security>")
                        if len(parts) >= 2:
                            # Get the part after system_integrity_and_security which contains the ASR prompt
                            asr_part = parts[1].strip()
                            if asr_part:
                                return asr_part
                    return content
            except Exception as e:
                logger.warning(f"Could not read ASR correction prompt from {path}: {e}")
    
    # Fallback
    return """You are an advanced Amharic AI language corrector. 
You will receive a raw, phonetic transcription from an Amharic speech-to-text model. 
Your only job is to fix grammatical errors, correct typos, insert proper spacing, and add punctuation (፦ ፣ ？ ！). 
Preserve the user's exact dialect and intent. 
Do not translate to English. Respond ONLY with the corrected Amharic text.

Example Input: ሰላምነክ እንደትነክ ባላንስ ማየትፈጋለዉ
Example Output: ሰላም ነህ? እንዴት ነህ? ባላንስ ማየት እፈልጋለሁ።"""


class AmharicLLMCorrector:
    """LLM-based post-correction for Amharic ASR transcripts."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = None
        self._asr_system_prompt = _load_asr_correction_prompt()
        self._load_client()

    def _load_client(self):
        """Lazy-load the Gemma LLM client."""
        try:
            from src.models.llm_client import Gemma4LLMClient
            logger.info(f"[LLM-Corrector] Loading Gemma 4 from {self.model_path}...")
            self._client = Gemma4LLMClient(self.model_path)
            logger.info("[LLM-Corrector] Model loaded successfully ✔")
        except Exception as e:
            logger.error(f"[LLM-Corrector] Failed to load model: {e}")
            self._client = None

    def _build_correction_prompt(self, transcript: str) -> str:
        """Build the correction prompt for Amharic ASR output."""
        return f"""የተጠቀሰው የድምጽ ጽሑፍ (ASR output) አስተካክል ያለበትን ሁኔታ ያስተካክሉ። በግልጽ አማርኛ (ግዕዝ ፊደል) ስርዓት ያስቀሩት። ከፊደሎች በተጨማሪ ምንም ነገር አትጨምሩ።

ASR Output: {transcript}

የተሻሻለ ውጤት:"""

    def correct(self, transcript: str) -> str:
        """
        Correct an Amharic ASR transcript using the LLM.
        
        Args:
            transcript: Raw ASR output in Amharic Ge'ez script
            
        Returns:
            Corrected transcript, or original if correction fails
        """
        if not transcript or not transcript.strip():
            return transcript
            
        if self._client is None:
            logger.warning("[LLM-Corrector] Client not available, returning original transcript")
            return transcript

        try:
            prompt = self._build_correction_prompt(transcript.strip())
            corrected = self._client.generate(
                prompt=prompt,
                system_prompt=self._asr_system_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
            )
            
            # Clean up the response
            corrected = corrected.strip()
            if corrected and len(corrected) > 0:
                logger.info(f"[LLM-Corrector] '{transcript[:50]}...' → '{corrected[:50]}...'")
                return corrected
            else:
                logger.warning("[LLM-Corrector] Empty correction, returning original")
                return transcript
                
        except Exception as e:
            logger.error(f"[LLM-Corrector] Correction failed: {e}")
            return transcript

    def correct_batch(self, transcripts: list[str]) -> list[str]:
        """Correct multiple transcripts."""
        return [self.correct(t) for t in transcripts]


def _get_corrector_instance() -> Optional[AmharicLLMCorrector]:
    """Lazy-load global singleton corrector instance."""
    global _corrector_instance
    if _corrector_instance is None:
        try:
            _corrector_instance = AmharicLLMCorrector()
        except Exception as e:
            logger.error(f"[LLM-Corrector] Failed to initialize: {e}")
            _corrector_instance = None
    return _corrector_instance


def correct_transcript(transcript: str) -> str:
    """
    Main entry point for LLM-based Amharic transcript correction.
    
    Args:
        transcript: Raw ASR output in Amharic Ge'ez script
        
    Returns:
        Corrected transcript (or original if correction unavailable/failed)
    """
    corrector = _get_corrector_instance()
    if corrector is None:
        return transcript
    return corrector.correct(transcript)


# Backward compatibility
def correct_amharic_transcript(transcript: str) -> str:
    """Alias for correct_transcript."""
    return correct_transcript(transcript)