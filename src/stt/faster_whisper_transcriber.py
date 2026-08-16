"""
High-performance Speech-to-Text transcriber using faster-whisper (CTranslate2).
Model: large-v3 with int8 CPU quantization & Silero VAD.
"""

import os
import tempfile
import subprocess
import numpy as np
import torch
from faster_whisper import WhisperModel

# Global singleton instance
_model_instance = None
_MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "distil-large-v3.5")
_DEFAULT_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_DEFAULT_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Domain vocabulary initial prompt to guide Whisper tokenizer for Amharic and AI terminology
DEFAULT_INITIAL_PROMPT = "የኢትዮጵያ አርቴፊሻል ኢንተለጀንስ ኢንስቲትዩት, አማኒ, EAII, Amani AI assistant, Gemma."


class FasterWhisperTranscriber:
    """Wrapper class for faster-whisper CTranslate2 STT model."""

    def __init__(
        self,
        model_size_or_path: str = _MODEL_ID,
        device: str = _DEFAULT_DEVICE,
        device_index: int = 0,
        compute_type: str = _DEFAULT_COMPUTE_TYPE,
        use_vad: bool = True,
        vad_parameters: dict = None,
        language: str = None,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT
    ):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.use_vad = use_vad
        self.vad_parameters = vad_parameters or {
            "threshold": 0.40,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 300
        }
        self.language = language
        self.initial_prompt = initial_prompt

        self._load_model()

    def _load_model(self):
        try:
            if self.device == "cuda":
                print(f"[FasterWhisper] Loading {self.model_size_or_path} on GPU cuda:{self.device_index} ({self.compute_type})...")
                self.model = WhisperModel(
                    self.model_size_or_path,
                    device=self.device,
                    device_index=self.device_index,
                    compute_type=self.compute_type
                )
            else:
                print(f"[FasterWhisper] Loading {self.model_size_or_path} on {self.device} ({self.compute_type})...")
                self.model = WhisperModel(
                    self.model_size_or_path,
                    device=self.device,
                    compute_type=self.compute_type
                )
            print(f"[FasterWhisper] Model loaded successfully on {self.device} ✔")
        except Exception as e:
            if self.device == "cuda":
                print(f"[FasterWhisper] ⚠️ GPU load failed ({e}). Falling back to CPU (int8)...")
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_size_or_path,
                    device="cpu",
                    compute_type="int8"
                )
                print("[FasterWhisper] CPU fallback loaded successfully ✔")
            else:
                raise e

    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a 1D float32 numpy audio array with normalization and beam search.

        Args:
            audio_array: 1D float32 numpy array normalized to [-1.0, 1.0].
            sample_rate: Audio sample rate in Hz (default 16000).

        Returns:
            Transcribed text string.
        """
        if len(audio_array) == 0:
            return ""

        # Audio Amplitude Peak Normalization (brings low-gain audio to clear audible range)
        max_val = np.max(np.abs(audio_array))
        if max_val > 1e-5:
            audio_array = audio_array / max_val * 0.95

        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            initial_prompt=self.initial_prompt,
            beam_size=5,
            best_of=5,
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None
        )

        full_text = " ".join([segment.text.strip() for segment in segments]).strip()
        return full_text

    def transcribe_audio_array_with_info(self, audio_array: np.ndarray, sample_rate: int = 16000, force_language: str = None) -> tuple:
        """
        Transcribe audio array with auto language detection.

        Returns:
            (transcript, detected_language, probability)
        """
        if len(audio_array) == 0:
            return "", "en", 0.0

        max_val = np.max(np.abs(audio_array))
        if max_val > 1e-5:
            audio_array = audio_array / max_val * 0.95

        segments, info = self.model.transcribe(
            audio_array,
            language=force_language if force_language else self.language,
            initial_prompt=self.initial_prompt if (force_language or self.language) == "am" else None,
            beam_size=5,
            best_of=5,
            repetition_penalty=1.2,
            condition_on_previous_text=False,
            vad_filter=self.use_vad,
            vad_parameters=self.vad_parameters if self.use_vad else None
        )

        full_text = " ".join([segment.text.strip() for segment in segments]).strip()
        detected_lang = getattr(info, "language", "en") or "en"
        prob = getattr(info, "language_probability", 1.0) or 1.0
        return full_text, detected_lang, prob


def _get_transcriber_instance() -> FasterWhisperTranscriber:
    """Lazy-load global singleton transcriber instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FasterWhisperTranscriber()
    return _model_instance


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit PCM audio bytes.

    Args:
        audio_bytes: Raw 16-bit signed PCM audio bytes.
        sample_rate: Audio sampling frequency (default 16000 Hz).

    Returns:
        Transcribed text string.
    """
    transcriber = _get_transcriber_instance()

    # Convert 16-bit PCM bytes to float32 numpy array [-1.0, 1.0]
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if sample_rate != 16000:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * 16000 / sample_rate)
        audio_array = signal.resample(audio_array, num_samples)

    return transcriber.transcribe_audio_array(audio_array, sample_rate=16000)


def transcribe_webm(audio_bytes: bytes, force_language: str = None) -> str:
    """
    Transcribe WebM/Opus audio blob from browser MediaRecorder using Distil-Whisper.
    """
    text, _, _ = transcribe_webm_with_info(audio_bytes, force_language=force_language)
    return text


def transcribe_webm_with_info(audio_bytes: bytes, force_language: str = None) -> tuple:
    """
    Transcribe WebM/Opus audio blob and return (text, detected_lang, probability).
    """
    transcriber = _get_transcriber_instance()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
        webm_file.write(audio_bytes)
        webm_path = webm_file.name

    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        with open(wav_path, "rb") as f:
            wav_data = f.read()

        pcm_bytes = wav_data[44:]  # Skip WAV header
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        return transcriber.transcribe_audio_array_with_info(audio_array, sample_rate=16000, force_language=force_language)

    finally:
        for p in [webm_path, wav_path]:
            if os.path.exists(p):
                os.unlink(p)

