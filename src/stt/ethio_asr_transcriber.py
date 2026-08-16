"""
Dedicated Amharic Speech-to-Text transcriber using badrex/Ethio-ASR-amharic (w2v-bert-2.0).
Outputs 100% native Amharic Ge'ez Fidel script with GPU acceleration on cuda:1 and CPU fallback.
"""

import os
import tempfile
import subprocess
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCTC

# Global singleton instance
_model_instance = None
_MODEL_ID = os.environ.get("AMHARIC_ASR_MODEL_ID", "badrex/Ethio-ASR-amharic")
_DEFAULT_DEVICE = os.environ.get("AMHARIC_ASR_DEVICE", "cpu")


class EthioASRTranscriber:
    """Wrapper class for Ethio-ASR Amharic speech recognition."""

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device: str = _DEFAULT_DEVICE,
    ):
        self.model_id = model_id
        self.device = device
        self._load_model()

    def _load_model(self):
        print(f"[Ethio-ASR] Loading {self.model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            self.model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            print(f"[Ethio-ASR] Model loaded successfully on {self.device} ✔")
        except Exception as e:
            if "cuda" in str(self.device):
                print(f"[Ethio-ASR] ⚠️ GPU load failed ({e}). Falling back to CPU...")
                self.device = "cpu"
                self.model = AutoModelForCTC.from_pretrained(self.model_id).to("cpu")
                self.model.eval()
                print("[Ethio-ASR] CPU fallback loaded successfully ✔")
            else:
                raise e

    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe 1D float32 numpy audio array to Amharic Ge'ez text.
        """
        if len(audio_array) == 0:
            return ""

        # Normalize audio amplitude
        max_val = np.max(np.abs(audio_array))
        if max_val > 1e-5:
            audio_array = audio_array / max_val * 0.95

        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        return transcription.strip()


def _get_transcriber_instance() -> EthioASRTranscriber:
    """Lazy-load global singleton transcriber instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = EthioASRTranscriber()
    return _model_instance


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit PCM audio bytes into Amharic text.
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
    Transcribe WebM/Opus audio blob from browser MediaRecorder to Amharic text.
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
