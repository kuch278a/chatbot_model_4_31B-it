"""
Voice Activity Detection (VAD) module for Amani AI STT.
Centralized source of truth for all VAD configurations, thresholds,
frame parameters, and silence trimming algorithms.

Upgrade: Silero VAD neural backend (v5) is used for human-voice detection
when PyTorch is available. Falls back to RMS energy thresholding if Silero
cannot be loaded (offline / import error), preserving full backward compat.
"""

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Centralized VAD Settings ────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE: int = 16000        # Target audio sample rate (16kHz)
DEFAULT_ENERGY_THRESHOLD: float = 0.003 # RMS volume threshold for speech detection
DEFAULT_FRAME_MS: int = 20              # Processing frame window in milliseconds
DEFAULT_PRE_PADDING_MS: int = 200       # Pre-speech audio margin to preserve (ms)
DEFAULT_POST_PADDING_MS: int = 200      # Post-speech audio margin to preserve (ms)
VAD_ENABLED: bool = True                # Global toggle to enable/disable VAD filtering

# ─── Silero Neural VAD Settings ───────────────────────────────────────────────
SILERO_ENABLED: bool = False             # Toggle Silero neural backend on/off
SILERO_THRESHOLD: float = 0.40          # Voice probability cutoff (0.0 – 1.0)
SILERO_REPO: str = "snakers4/silero-vad"  # Torch Hub model source


# ─── Silero VAD Backend (lazy-loaded singleton) ────────────────────────────────

class SileroVADBackend:
    """
    Lazy-loaded singleton wrapper around the Silero VAD v5 neural model.

    The model (~2 MB ONNX) is downloaded once from PyTorch Hub and cached
    locally. If loading fails for any reason (offline, missing torch, etc.),
    the backend silently marks itself as unavailable so callers can fall back
    to RMS energy detection without raising exceptions.
    """

    _instance: Optional["SileroVADBackend"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._utils = None
        self._available: bool = False
        self._load_attempted: bool = False

    @classmethod
    def get(cls) -> "SileroVADBackend":
        """Return the global singleton, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _try_load(self):
        """Attempt to load Silero VAD from Torch Hub (once)."""
        if self._load_attempted:
            return
        self._load_attempted = True

        if not SILERO_ENABLED:
            logger.info("[VAD] Silero backend disabled via SILERO_ENABLED=False. Using RMS fallback.")
            return

        try:
            import torch
            logger.info("[VAD] Loading Silero VAD neural model from Torch Hub...")
            self._model, self._utils = torch.hub.load(
                repo_or_dir=SILERO_REPO,
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            self._model.eval()
            self._available = True
            logger.info("[VAD] Silero VAD loaded successfully. Human-voice detection active.")
        except Exception as exc:
            logger.warning(
                "[VAD] Silero VAD could not be loaded (%s). "
                "Falling back to RMS energy thresholding.", exc
            )
            self._available = False

    @property
    def available(self) -> bool:
        self._try_load()
        return self._available

    def is_speech_frame(self, frame: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
        """
        Run a single audio frame through the Silero neural classifier.

        Args:
            frame: 1D float32 numpy array for one frame (512 samples at 16kHz).
            sample_rate: Sample rate of the audio (must be 8000 or 16000).

        Returns:
            True if Silero detects a human voice in the frame, False otherwise.
        """
        if not self.available:
            return False
        try:
            import torch
            tensor = torch.from_numpy(frame.copy()).float()
            with torch.no_grad():
                prob = self._model(tensor, sample_rate).item()
            return prob >= SILERO_THRESHOLD
        except Exception as exc:
            logger.debug("[VAD] Silero frame inference error: %s", exc)
            return False

    def voice_probabilities(
        self,
        audio_array: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_size: int = 512
    ) -> np.ndarray:
        """
        Compute per-frame Silero voice probabilities for an entire audio buffer.

        Args:
            audio_array: 1D float32 audio array.
            sample_rate: Audio sample rate (8000 or 16000 Hz).
            frame_size: Samples per frame (512 recommended for 16kHz Silero v5).

        Returns:
            1D float32 numpy array of voice probability per frame.
        """
        if not self.available:
            return np.array([], dtype=np.float32)
        try:
            import torch
            num_frames = len(audio_array) // frame_size
            if num_frames == 0:
                return np.array([], dtype=np.float32)

            frames = audio_array[: num_frames * frame_size].reshape(num_frames, frame_size)
            probs = []
            with torch.no_grad():
                for frame in frames:
                    tensor = torch.from_numpy(frame.copy()).float()
                    prob = self._model(tensor, sample_rate).item()
                    probs.append(prob)
            return np.array(probs, dtype=np.float32)
        except Exception as exc:
            logger.debug("[VAD] Silero batch inference error: %s", exc)
            return np.array([], dtype=np.float32)

    def reset_state(self):
        """Reset internal GRU hidden state between utterances (Silero v5 is stateful)."""
        if self._available and self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass


# ─── VAD Configuration ────────────────────────────────────────────────────────

class VADConfig:
    """Configuration container for Voice Activity Detection."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        frame_ms: int = DEFAULT_FRAME_MS,
        pre_padding_ms: int = DEFAULT_PRE_PADDING_MS,
        post_padding_ms: int = DEFAULT_POST_PADDING_MS,
        enabled: bool = VAD_ENABLED,
        use_silero: bool = SILERO_ENABLED,
        silero_threshold: float = SILERO_THRESHOLD,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_ms = frame_ms
        self.pre_padding_ms = pre_padding_ms
        self.post_padding_ms = post_padding_ms
        self.enabled = enabled
        self.use_silero = use_silero
        self.silero_threshold = silero_threshold


# ─── Voice Activity Detector ──────────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Detects human voice presence and trims leading/trailing silence.

    Detection strategy (dual-gate when Silero is available):
      1. Neural gate  -- Silero VAD voice probability >= silero_threshold
      2. Energy gate  -- RMS energy >= energy_threshold (guards against very
                         quiet artefacts that Silero might classify as voice)

    When Silero is unavailable the system transparently falls back to
    RMS energy thresholding only (identical to previous behavior).
    """

    def __init__(
        self,
        config: VADConfig = None,
        sample_rate: int = None,
        energy_threshold: float = None,
    ):
        """
        Initialize the Voice Activity Detector with centralized settings.

        Args:
            config: Optional VADConfig instance.
            sample_rate: Optional override for sample_rate.
            energy_threshold: Optional override for energy_threshold.
        """
        self.config = config or VADConfig()
        if sample_rate is not None:
            self.config.sample_rate = sample_rate
        if energy_threshold is not None:
            self.config.energy_threshold = energy_threshold

        self.frame_size = int(self.config.sample_rate * (self.config.frame_ms / 1000.0))

        # Silero v5 requires exactly 512-sample frames at 16kHz
        self._silero_frame_size: int = 512
        self._silero = SileroVADBackend.get()

    # ── Low-level frame helpers ──────────────────────────────────────────────

    def compute_frame_rms(self, frame: np.ndarray) -> float:
        """Calculates Root-Mean-Square (RMS) energy for a single audio frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame ** 2)))

    def _rms_speech_frame(self, frame: np.ndarray) -> bool:
        """Returns True if RMS energy exceeds the configured threshold."""
        return self.compute_frame_rms(frame) > self.config.energy_threshold

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """
        Determines whether a single frame contains active human speech.

        Uses Silero neural detection when available (dual-gated with RMS).
        Falls back to pure RMS energy check when Silero is not loaded.
        """
        if self.config.use_silero and self._silero.available:
            # Dual-gate: both neural AND energy must agree
            return (
                self._silero.is_speech_frame(frame, self.config.sample_rate)
                and self._rms_speech_frame(frame)
            )
        return self._rms_speech_frame(frame)

    # ── Main processing ──────────────────────────────────────────────────────

    def process_audio(
        self,
        audio_array: np.ndarray,
        pre_padding_ms: int = None,
        post_padding_ms: int = None,
    ) -> tuple:
        """
        Detect speech segments and trim leading/trailing silence from audio buffer.

        Uses Silero neural VAD (when available) for human-voice detection,
        falling back to RMS energy thresholding if Silero is unavailable.

        Args:
            audio_array: 1D float32 numpy array normalized to [-1.0, 1.0].
            pre_padding_ms: Milliseconds of audio buffer to preserve before speech start.
            post_padding_ms: Milliseconds of audio buffer to preserve after speech end.

        Returns:
            Tuple of (trimmed_audio_array, has_speech_bool, stats_dict)
        """
        pre_padding_ms = pre_padding_ms if pre_padding_ms is not None else self.config.pre_padding_ms
        post_padding_ms = post_padding_ms if post_padding_ms is not None else self.config.post_padding_ms

        original_sec = round(len(audio_array) / self.config.sample_rate, 3)

        if not self.config.enabled:
            # VAD disabled -- pass-through audio as-is
            return audio_array, True, {
                "original_sec": original_sec,
                "trimmed_sec": original_sec,
                "has_speech": True,
                "backend": "disabled",
            }

        if len(audio_array) == 0:
            return audio_array, False, {"original_sec": 0, "trimmed_sec": 0, "has_speech": False}

        # ── Choose frame size based on available backend ─────────────────────
        use_silero = self.config.use_silero and self._silero.available

        if use_silero:
            # Silero v5 requires exactly 512-sample frames at 16kHz
            effective_frame_size = self._silero_frame_size
            frame_ms_effective = int(effective_frame_size / self.config.sample_rate * 1000)
        else:
            effective_frame_size = self.frame_size
            frame_ms_effective = self.config.frame_ms

        num_frames = len(audio_array) // effective_frame_size
        if num_frames == 0:
            # Audio shorter than one frame -- pass through unchanged
            return audio_array, True, {
                "original_sec": original_sec,
                "trimmed_sec": original_sec,
                "has_speech": True,
                "backend": "short_audio",
            }

        frames = audio_array[: num_frames * effective_frame_size].reshape(num_frames, effective_frame_size)
        rms_energies = np.sqrt(np.mean(frames ** 2, axis=1))

        # ── Build speech mask ─────────────────────────────────────────────────
        if use_silero:
            # Reset Silero GRU hidden state for a fresh utterance
            self._silero.reset_state()

            silero_probs = self._silero.voice_probabilities(
                audio_array, self.config.sample_rate, effective_frame_size
            )

            if len(silero_probs) == num_frames:
                # Dual-gate: neural probability AND RMS energy must both pass
                speech_mask = (
                    (silero_probs >= self.config.silero_threshold)
                    & (rms_energies > self.config.energy_threshold)
                )
                backend = "silero+rms"
            else:
                # Silero returned unexpected shape -- fall back to RMS only
                speech_mask = rms_energies > self.config.energy_threshold
                silero_probs = None
                backend = "rms_fallback"
        else:
            speech_mask = rms_energies > self.config.energy_threshold
            silero_probs = None
            backend = "rms"

        # ── No speech detected ────────────────────────────────────────────────
        if not np.any(speech_mask):
            logger.debug("[VAD] No speech detected. backend=%s original=%.2fs", backend, original_sec)
            return np.array([], dtype=np.float32), False, {
                "original_sec": original_sec,
                "trimmed_sec": 0.0,
                "has_speech": False,
                "backend": backend,
            }

        speech_indices = np.where(speech_mask)[0]

        # ── Apply pre/post padding around detected speech ─────────────────────
        pre_frames = int(pre_padding_ms / frame_ms_effective)
        post_frames = int(post_padding_ms / frame_ms_effective)

        start_frame = max(0, speech_indices[0] - pre_frames)
        end_frame = min(num_frames, speech_indices[-1] + 1 + post_frames)

        start_sample = start_frame * effective_frame_size
        end_sample = min(len(audio_array), end_frame * effective_frame_size)

        trimmed_audio = audio_array[start_sample:end_sample]
        trimmed_sec = round(len(trimmed_audio) / self.config.sample_rate, 3)

        stats = {
            "original_sec": original_sec,
            "trimmed_sec": trimmed_sec,
            "has_speech": True,
            "active_ratio": round(float(np.sum(speech_mask)) / num_frames, 3),
            "backend": backend,
        }

        if silero_probs is not None and len(silero_probs):
            stats["silero_mean_prob"] = round(float(np.mean(silero_probs[speech_indices])), 3)

        logger.debug(
            "[VAD] Speech detected. backend=%s original=%.2fs trimmed=%.2fs active=%.1f%%",
            backend, original_sec, trimmed_sec, stats["active_ratio"] * 100
        )

        return trimmed_audio, True, stats


# ─── Global default detector instance using centralized settings ──────────────
_default_detector = VoiceActivityDetector()


def apply_vad(audio_array: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple:
    """
    Applies VAD silence trimming using centralized settings from src/stt/vad.py.

    Uses Silero neural VAD (human-voice detection) when available, with
    automatic fallback to RMS energy thresholding.

    Args:
        audio_array: 1D float32 audio numpy array.
        sample_rate: Audio sample rate in Hz (defaults to DEFAULT_SAMPLE_RATE = 16000).

    Returns:
        Tuple of (trimmed_audio_array, has_speech_bool).
    """
    detector = VoiceActivityDetector(sample_rate=sample_rate)
    trimmed, has_speech, _ = detector.process_audio(audio_array)
    return trimmed, has_speech
