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
from src.stt.vad import apply_vad
from src.stt.llm_corrector import correct_transcript
from src.stt.audio_quality import ensure_audio_constraints, validate_for_ctc

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

    def transcribe_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000, use_llm_correction: bool = True) -> str:
        """
        Transcribe 1D float32 numpy audio array to Amharic Ge'ez text.
        Splits long audio into safe chunks (<= 20s) to prevent quadratic self-attention memory blowup.
        
        Args:
            audio_array: 1D float32 numpy audio array
            sample_rate: Audio sample rate (default 16000)
            use_llm_correction: Whether to apply LLM post-correction (default True)
        """
        if len(audio_array) == 0:
            return ""

        # ─── Audio Quality Validation & Normalization for CTC Encoder ────────────
        is_valid, violation_msg = validate_for_ctc(audio_array, sample_rate)
        if not is_valid:
            print(f"[Ethio-ASR][WARN] Audio quality issues: {violation_msg}. Attempting auto-fix...", flush=True)
            audio_array, metrics = ensure_audio_constraints(
                audio_array, 
                sample_rate=sample_rate,
                auto_normalize=True,
                auto_resample=True,
            )
            sample_rate = 16000  # After ensure_audio_constraints, it's always 16kHz
            if not metrics.passes_constraints:
                print(f"[Ethio-ASR][WARN] Auto-fix incomplete: {metrics.violations}", flush=True)
        else:
            print(f"[Ethio-ASR][DEBUG] Audio quality OK: {len(audio_array)} samples @ {sample_rate}Hz", flush=True)

        # Apply VAD to trim leading/trailing silence before processing
        pre_vad_len = len(audio_array)
        audio_array, has_speech = apply_vad(audio_array, sample_rate)
        print(f"[Ethio-ASR][DEBUG] VAD: {pre_vad_len} samples → {len(audio_array)} samples, "
              f"has_speech={has_speech}, "
              f"trimmed={pre_vad_len - len(audio_array)} samples ({(pre_vad_len - len(audio_array))/sample_rate:.2f}s removed)", flush=True)
        if not has_speech or len(audio_array) == 0:
            print(f"[Ethio-ASR][DEBUG] VAD rejected audio — no speech detected!", flush=True)
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
            result = " ".join(transcripts)
            print(f"[Ethio-ASR][DEBUG] Multi-chunk result: '{result}'", flush=True)
        else:
            result = self._transcribe_single_chunk(audio_array, sample_rate=sample_rate)
            print(f"[Ethio-ASR][DEBUG] Single-chunk result: '{result}'", flush=True)

        # Apply LLM post-correction if enabled and we have a result
        if use_llm_correction and result:
            print(f"[Ethio-ASR][DEBUG] Applying LLM correction...", flush=True)
            result = correct_transcript(result)
            print(f"[Ethio-ASR][DEBUG] Corrected result: '{result}'", flush=True)

        return result

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


def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000, use_llm_correction: bool = True) -> str:
    """
    Transcribe raw 16-bit PCM audio bytes into Amharic text.
    
    Args:
        audio_bytes: Raw 16-bit PCM audio bytes
        sample_rate: Audio sample rate (default 16000)
        use_llm_correction: Whether to apply LLM post-correction (default True)
    """
    if not audio_bytes or len(audio_bytes) < 4:
        return ""

    transcriber = _get_transcriber_instance()

    # Ensure byte count is even for int16
    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes[:len(audio_bytes) - (len(audio_bytes) % 2)]

    # Convert 16-bit PCM bytes to float32 numpy array [-1.0, 1.0]
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[Ethio-ASR][DEBUG] PCM→float32: {len(audio_array)} samples, "
          f"duration={len(audio_array)/sample_rate:.2f}s, "
          f"max={np.max(np.abs(audio_array)):.4f}, "
          f"rms={np.sqrt(np.mean(audio_array**2)):.4f}", flush=True)

    if sample_rate != 16000 and len(audio_array) > 0:
        import scipy.signal as signal
        num_samples = int(len(audio_array) * 16000 / sample_rate)
        audio_array = signal.resample(audio_array, num_samples)

    return transcriber.transcribe_audio_array(audio_array, sample_rate=16000, use_llm_correction=use_llm_correction)


def transcribe_audio_blob(audio_bytes: bytes, use_llm_correction: bool = True) -> str:
    """
    Transcribe WebM/MP4/OGG/WAV audio blob from browser MediaRecorder to Amharic text.
    Uses ffmpeg in-memory to universally normalize any client container into 16kHz mono PCM.
    
    Args:
        audio_bytes: Audio blob bytes from browser MediaRecorder
        use_llm_correction: Whether to apply LLM post-correction (default True)
    """
    if not audio_bytes or len(audio_bytes) < 64:
        print(f"[Ethio-ASR][DEBUG] Audio too short: {len(audio_bytes) if audio_bytes else 0} bytes", flush=True)
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
            stderr=subprocess.PIPE,
            check=False
        )

        if proc.returncode != 0 or not proc.stdout:
            stderr_msg = proc.stderr.decode("utf-8", errors="replace")[-500:] if proc.stderr else "no stderr"
            print(f"[Ethio-ASR][DEBUG] ffmpeg failed (rc={proc.returncode}): {stderr_msg}", flush=True)
            return ""

        pcm_bytes = proc.stdout
        print(f"[Ethio-ASR][DEBUG] ffmpeg OK: {len(audio_bytes)} input bytes → {len(pcm_bytes)} PCM bytes ({len(pcm_bytes)/32000:.2f}s)", flush=True)

        return transcribe_audio(pcm_bytes, sample_rate=16000, use_llm_correction=use_llm_correction)
    except Exception as e:
        print(f"[Ethio-ASR] FFMPEG pipe error: {e}", flush=True)
        return ""


# Backward compatibility alias
def transcribe_webm(audio_bytes: bytes, use_llm_correction: bool = True) -> str:
    """Legacy wrapper for transcribe_audio_blob."""
    return transcribe_audio_blob(audio_bytes, use_llm_correction=use_llm_correction)

