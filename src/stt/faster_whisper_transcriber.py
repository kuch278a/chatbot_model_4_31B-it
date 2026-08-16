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
_MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "large-v3")
_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")  # 8-bit quantization for maximum CPU speed


# Domain vocabulary initial prompt to guide Whisper tokenizer for Amharic and AI terminology
DEFAULT_INITIAL_PROMPT = "የኢትዮጵያ አርቴፊሻል ኢንተለጀንስ ኢንስቲትዩት, አማኒ, EAII, Amani AI assistant, Gemma."


class FasterWhisperTranscriber:
    """Wrapper class for faster-whisper CTranslate2 STT model with enhanced decoding."""

    def __init__(
        self,
        model_size_or_path: str = _MODEL_ID,
        device: str = _DEVICE,
        compute_type: str = _COMPUTE_TYPE,
        use_vad: bool = True,
        vad_parameters: dict = None,
        language: str = None,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT
    ):
        self.model_size_or_path = model_size_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or ("int8" if self.device == "cpu" else "float16")
        self.use_vad = use_vad
        self.vad_parameters = vad_parameters or {
            "threshold": 0.40,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 300
        }
        self.language = language
        self.initial_prompt = initial_prompt

        print(f"[FasterWhisper] Loading {self.model_size_or_path} on {self.device} ({self.compute_type})...")
        self.model = WhisperModel(
            self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type
        )
        print("[FasterWhisper] Model loaded successfully ✔")

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


def transcribe_webm(audio_bytes: bytes) -> str:
    """
    Transcribe WebM/Opus audio blob from browser MediaRecorder.

    Args:
        audio_bytes: WebM binary audio blob.

    Returns:
        Transcribed text string.
    """
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
        return transcribe_audio(pcm_bytes, sample_rate=16000)

    finally:
        for p in [webm_path, wav_path]:
            if os.path.exists(p):
                os.unlink(p)
