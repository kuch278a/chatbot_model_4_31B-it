import re
import asyncio
import edge_tts
from typing import Dict, List, Optional

class VoiceSynthesizer:
    """
    Text-to-Speech (TTS) voice synthesizer and voice manager supporting Amharic (am-ET)
    and multilingual audio synthesis via edge-tts and Web Speech API fallback.
    """
    
    DEFAULT_VOICES = [
        {
            "name": "Microsoft Mekdes Online (Natural) - Amharic (Ethiopia)",
            "short_name": "am-ET-MekdesNeural",
            "lang": "am-ET",
            "gender": "Female",
            "type": "Neural"
        },
        {
            "name": "Microsoft Ameha Online (Natural) - Amharic (Ethiopia)",
            "short_name": "am-ET-AmehaNeural",
            "lang": "am-ET",
            "gender": "Male",
            "type": "Neural"
        },
        {
            "name": "Google Amharic (Ethiopia)",
            "short_name": "am-ET-Standard",
            "lang": "am-ET",
            "gender": "Female",
            "type": "Standard"
        },
        {
            "name": "Microsoft Jenny Online (Natural) - English (United States)",
            "short_name": "en-US-JennyNeural",
            "lang": "en-US",
            "gender": "Female",
            "type": "Neural"
        }
    ]

    def __init__(self, voices: Optional[List[Dict[str, str]]] = None):
        self.voices = voices if voices is not None else self.DEFAULT_VOICES

    def get_available_voices(self, lang: Optional[str] = None) -> List[Dict[str, str]]:
        """Returns all available voices, optionally filtered by language locale."""
        if not lang:
            return self.voices
        return [v for v in self.voices if v["lang"].lower() == lang.lower() or v["lang"].startswith(lang.split("-")[0])]

    def select_voice(self, lang: str = "am-ET", preferred_name: Optional[str] = None) -> Dict[str, str]:
        """
        Selects the best matching voice using a prioritized multi-tier fallback strategy.
        """
        if preferred_name:
            exact = next((v for v in self.voices if v["name"] == preferred_name or v["short_name"] == preferred_name), None)
            if exact:
                return exact

        if lang.startswith("am"):
            mekdes_exact = next((v for v in self.voices if v["name"] == "Microsoft Mekdes Online (Natural) - Amharic (Ethiopia)"), None)
            if mekdes_exact:
                return mekdes_exact
                
            mekdes_neural = next((v for v in self.voices if "am-ET-MekdesNeural" in v.get("short_name", "")), None)
            if mekdes_neural:
                return mekdes_neural

            am_fallback = next((v for v in self.voices if v["lang"] == "am-ET"), None)
            if am_fallback:
                return am_fallback

        generic = next((v for v in self.voices if v["lang"].startswith(lang.split("-")[0])), None)
        return generic or self.voices[0]

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Cleans markdown, HTML tags, and code blocks for speech synthesis."""
        if not text:
            return ""
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'\*{1,2}|_{1,2}', '', text)
        text = re.sub(r'#+\s*', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    async def generate_audio_file(self, text: str, output_path: str, lang: str = "am-ET") -> str:
        """Generates an MP3 audio file using edge-tts with Neural Amharic voice."""
        cleaned_text = self.clean_text_for_speech(text)
        voice_info = self.select_voice(lang=lang)
        voice_short_name = voice_info.get("short_name", "am-ET-MekdesNeural")

        communicate = edge_tts.Communicate(cleaned_text, voice_short_name)
        await communicate.save(output_path)
        return output_path

    def synthesize(self, text: str, lang: str = "am-ET", rate: float = 1.0, pitch: float = 1.0) -> Dict[str, str]:
        """Synthesizes speech metadata for a given text."""
        cleaned_text = self.clean_text_for_speech(text)
        selected_voice = self.select_voice(lang=lang)
        
        return {
            "status": "success",
            "text": cleaned_text,
            "voice": selected_voice["name"],
            "short_name": selected_voice["short_name"],
            "lang": selected_voice["lang"],
            "rate": rate,
            "pitch": pitch,
            "wpm": int(117 * rate)
        }
