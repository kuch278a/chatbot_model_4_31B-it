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
            
            if "cuda" in str(self.device):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
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
        Splits long audio into safe chunks (<= 20s) to prevent quadratic self-attention memory blowup.
        """
        if len(audio_array) == 0:
            return ""

        # Limit total audio duration to 60 seconds max to protect system RAM
        max_total_samples = sample_rate * 60
        if len(audio_array) > max_total_samples:
            # Keep the most recent 60 seconds
            audio_array = audio_array[-max_total_samples:]

        # Safe chunk size: 20 seconds (320,000 samples @ 16kHz)
        max_chunk_samples = sample_rate * 20

        if len(audio_array) > max_chunk_samples:
            transcripts = []
            for start in range(0, len(audio_array), max_chunk_samples):
                chunk = audio_array[start : start + max_chunk_samples]
                if len(chunk) < int(sample_rate * 0.3):
                    continue
                sub_text = self._transcribe_single_chunk(chunk, sample_rate=sample_rate)
                if sub_text:
                    transcripts.append(sub_text)
            return " ".join(transcripts)

        return self._transcribe_single_chunk(audio_array, sample_rate=sample_rate)

    def _transcribe_single_chunk(self, chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a single audio chunk within safe memory bounds."""
        try:
            max_val = np.max(np.abs(chunk))
            if max_val > 1e-5:
                chunk = chunk / max_val * 0.95

            inputs = self.processor(
                chunk,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            pred_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(pred_ids)[0]
            return transcription.strip()
        except (RuntimeError, MemoryError) as e:
            print(f"[Ethio-ASR] Chunk transcription memory warning: {e}", flush=True)
            return ""



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
    if not audio_bytes or len(audio_bytes) < 4:
        return ""

    transcriber = _get_transcriber_instance()

    # Ensure byte count is even for int16
    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes[:len(audio_bytes) - (len(audio_bytes) % 2)]

    # Convert 16-bit PCM bytes to float32 numpy array [-1.0, 1.0]
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if sample_rate != 16000 and len(audio_array) > 0:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * 16000 / sample_rate)
        audio_array = signal.resample(audio_array, num_samples)

    return transcriber.transcribe_audio_array(audio_array, sample_rate=16000)


def transcribe_audio_blob(audio_bytes: bytes) -> str:
    """
    Transcribe WebM/MP4/OGG/WAV audio blob from browser MediaRecorder to Amharic text.
    Uses ffmpeg in-memory to universally normalize any client container into 16kHz mono PCM.
    """
    if not audio_bytes or len(audio_bytes) < 64:
        return ""

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", "pipe:0",
                "-ar", "16000",
                "-ac", "1",
                "-f", "s16le",
                "pipe:1"
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False
        )

        if proc.returncode != 0 or not proc.stdout:
            return ""

        return transcribe_audio(proc.stdout, sample_rate=16000)
    except Exception as e:
        print(f"[Ethio-ASR] FFMPEG pipe error: {e}", flush=True)
        return ""


# Backward compatibility alias
def transcribe_webm(audio_bytes: bytes) -> str:
    """Legacy wrapper for transcribe_audio_blob."""
    return transcribe_audio_blob(audio_bytes)

