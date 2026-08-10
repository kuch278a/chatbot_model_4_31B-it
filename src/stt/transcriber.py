"""
Amharic Speech-to-Text transcriber using badrex/Ethio-ASR-amharic.
Architecture: wav2vec2-BERT 2.0 | Params: 606M | WER: ~22.4% | Output: native Fidel script
Runs on CPU to avoid competing with the 31B LLM on GPU.
"""

import torch
import numpy as np

# Model is loaded once at module level (lazy, on first use)
_processor = None
_model = None
_MODEL_ID = "/mnt/data/ethio-asr"   # Local copy — no internet needed
_DEVICE = "cpu"   # Keep off GPU — LLM occupies all VRAM


def _load_model():
    """Lazy-load the Ethio-ASR model on first call."""
    global _processor, _model
    if _model is not None:
        return

    from transformers import AutoProcessor, AutoModelForCTC
    print(f"[STT] Loading Ethio-ASR-amharic from HuggingFace ({_MODEL_ID})...")
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    _model = AutoModelForCTC.from_pretrained(_MODEL_ID)
    _model.eval()
    _model.to(_DEVICE)
    print("[STT] Ethio-ASR-amharic loaded ✔")


from src.stt.vad import apply_vad



def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw audio bytes to Amharic text with VAD pre-filtering.

    Args:
        audio_bytes: Raw PCM audio bytes (mono, 16-bit, 16kHz preferred).
        sample_rate: Sample rate of the audio (default 16000 Hz).

    Returns:
        Transcribed Amharic text (Fidel script).
    """
    _load_model()

    # Convert raw bytes to float32 numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * 16000 / sample_rate)
        audio_array = signal.resample(audio_array, num_samples)

    # Apply VAD pre-filter
    audio_array, has_speech = apply_vad(audio_array, sample_rate=16000)
    if not has_speech or len(audio_array) == 0:
        print("[STT VAD] Silence detected — skipping model inference.")
        return ""

    # Tokenize
    inputs = _processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    # Run inference (CPU)
    with torch.no_grad():
        logits = _model(**inputs).logits

    # Decode CTC output to Amharic text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = _processor.batch_decode(predicted_ids)[0]

    return transcription.strip()


def transcribe_webm(audio_bytes: bytes) -> str:
    """
    Transcribe audio from a browser MediaRecorder blob (WebM/Opus format).
    Converts WebM → raw PCM using ffmpeg, then transcribes.

    Args:
        audio_bytes: Raw WebM/Opus bytes from browser MediaRecorder.

    Returns:
        Transcribed Amharic text.
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
                "-ar", "16000",   # 16kHz
                "-ac", "1",       # mono
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

        return transcribe_audio(pcm_bytes, sample_rate=16000)

    finally:
        # Clean up temp files
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.unlink(path)
