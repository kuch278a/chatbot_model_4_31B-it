"""
Speech-to-Text transcriber using distil-whisper/distil-large-v3.5.
Model stored locally in: /mnt/data/chatbot_model_4_31B-it/src/models/distil-large-v3.5
Runs on CPU to preserve 31B Gemma LLM GPU VRAM.
"""

import torch
import numpy as np
from src.stt.vad import apply_vad, DEFAULT_SAMPLE_RATE

# Model state (lazy-loaded on first call)
_processor = None
_model = None
_MODEL_ID = "/mnt/data/chatbot_model_4_31B-it/src/models/distil-large-v3.5"
_DEVICE = "cpu"


def _load_model():
    """Lazy-load distil-large-v3.5 processor and model on first call."""
    global _processor, _model
    if _model is not None:
        return

    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    print(f"[STT] Loading distil-large-v3.5 from local models directory ({_MODEL_ID})...")
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(_MODEL_ID, low_cpu_mem_usage=True)
    _model.eval()
    _model.to(_DEVICE)
    print("[STT] distil-large-v3.5 loaded ✔")


def transcribe_audio(audio_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """
    Transcribe raw audio bytes using distil-large-v3.5 with VAD pre-filtering.

    Args:
        audio_bytes: Raw PCM audio bytes (mono, 16-bit, 16kHz preferred).
        sample_rate: Sample rate of the audio (defaults to VAD DEFAULT_SAMPLE_RATE).

    Returns:
        Transcribed text from speech input.
    """
    _load_model()

    # Convert raw bytes to float32 numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

    # Resample to target sample rate if needed
    if sample_rate != DEFAULT_SAMPLE_RATE:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * DEFAULT_SAMPLE_RATE / sample_rate)
        audio_array = signal.resample(audio_array, num_samples)

    # Apply VAD pre-filter
    audio_array, has_speech = apply_vad(audio_array, sample_rate=DEFAULT_SAMPLE_RATE)
    if not has_speech or len(audio_array) == 0:
        print("[STT VAD] Silence detected — skipping model inference.")
        return ""

    # Process audio input with distil-large-v3.5 processor
    inputs = _processor(
        audio_array,
        sampling_rate=DEFAULT_SAMPLE_RATE,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(_DEVICE)

    # Generate transcription using distil-whisper-large-v3.5
    with torch.no_grad():
        predicted_ids = _model.generate(input_features)

    transcription = _processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def transcribe_webm(audio_bytes: bytes) -> str:
    """
    Transcribe audio from a browser MediaRecorder blob (WebM/Opus format).
    Converts WebM → raw PCM using ffmpeg, then transcribes with distil-large-v3.5.

    Args:
        audio_bytes: Raw WebM/Opus bytes from browser MediaRecorder.

    Returns:
        Transcribed text.
    """
    import subprocess
    import tempfile
    import os

    # Write WebM to a temp file
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
        webm_file.write(audio_bytes)
        webm_path = webm_file.name

    # Convert to 16kHz mono PCM WAV using ffmpeg
    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ar", str(DEFAULT_SAMPLE_RATE),   # 16kHz
                "-ac", "1",                      # mono
                "-f", "wav",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Read converted WAV as raw PCM (skip 44-byte WAV header)
        with open(wav_path, "rb") as f:
            wav_data = f.read()
        pcm_bytes = wav_data[44:]  # Skip WAV header

        return transcribe_audio(pcm_bytes, sample_rate=DEFAULT_SAMPLE_RATE)

    finally:
        # Clean up temp files
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.unlink(path)
