"""
Voice Activity Detection (VAD) module for Amani AI STT.
Centralized source of truth for all VAD configurations, thresholds,
frame parameters, and silence trimming algorithms.
"""

import numpy as np

# ─── Centralized VAD Settings ────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE: int = 16000       # Target audio sample rate (16kHz)
DEFAULT_ENERGY_THRESHOLD: float = 0.008 # RMS volume threshold for speech detection
DEFAULT_FRAME_MS: int = 20             # Processing frame window in milliseconds
DEFAULT_PRE_PADDING_MS: int = 200      # Pre-speech audio margin to preserve (ms)
DEFAULT_POST_PADDING_MS: int = 200     # Post-speech audio margin to preserve (ms)
VAD_ENABLED: bool = True               # Global toggle to enable/disable VAD filtering


class VADConfig:
    """Configuration container for Voice Activity Detection."""
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        frame_ms: int = DEFAULT_FRAME_MS,
        pre_padding_ms: int = DEFAULT_PRE_PADDING_MS,
        post_padding_ms: int = DEFAULT_POST_PADDING_MS,
        enabled: bool = VAD_ENABLED
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_ms = frame_ms
        self.pre_padding_ms = pre_padding_ms
        self.post_padding_ms = post_padding_ms
        self.enabled = enabled


class VoiceActivityDetector:
    def __init__(self, config: VADConfig = None, sample_rate: int = None, energy_threshold: float = None):
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

    def compute_frame_rms(self, frame: np.ndarray) -> float:
        """Calculates Root-Mean-Square (RMS) energy for a single audio frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame ** 2)))

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Determines whether a single frame contains active speech."""
        return self.compute_frame_rms(frame) > self.config.energy_threshold

    def process_audio(
        self,
        audio_array: np.ndarray,
        pre_padding_ms: int = None,
        post_padding_ms: int = None
    ) -> tuple[np.ndarray, bool, dict]:
        """
        Detect speech segments and trim leading/trailing silence from audio buffer.

        Args:
            audio_array: 1D float32 numpy array normalized to [-1.0, 1.0].
            pre_padding_ms: Milliseconds of audio buffer to preserve before speech start.
            post_padding_ms: Milliseconds of audio buffer to preserve after speech end.

        Returns:
            Tuple of (trimmed_audio_array, has_speech_bool, stats_dict)
        """
        pre_padding_ms = pre_padding_ms if pre_padding_ms is not None else self.config.pre_padding_ms
        post_padding_ms = post_padding_ms if post_padding_ms is not None else self.config.post_padding_ms

        if not self.config.enabled:
            # VAD disabled — pass-through audio as-is
            return audio_array, True, {"original_sec": len(audio_array)/self.config.sample_rate, "trimmed_sec": len(audio_array)/self.config.sample_rate, "has_speech": True}

        if len(audio_array) == 0:
            return audio_array, False, {"original_sec": 0, "trimmed_sec": 0, "has_speech": False}

        num_frames = len(audio_array) // self.frame_size
        if num_frames == 0:
            return audio_array, True, {
                "original_sec": round(len(audio_array)/self.config.sample_rate, 3),
                "trimmed_sec": round(len(audio_array)/self.config.sample_rate, 3),
                "has_speech": True
            }

        # Reshape into contiguous frames
        frames = audio_array[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)
        rms_energies = np.sqrt(np.mean(frames ** 2, axis=1))
        speech_mask = rms_energies > self.config.energy_threshold

        if not np.any(speech_mask):
            # Pure silence detected
            return np.array([], dtype=np.float32), False, {
                "original_sec": round(len(audio_array) / self.config.sample_rate, 3),
                "trimmed_sec": 0.0,
                "has_speech": False
            }

        speech_indices = np.where(speech_mask)[0]

        # Calculate padding in frame counts
        pre_frames = int(pre_padding_ms / self.config.frame_ms)
        post_frames = int(post_padding_ms / self.config.frame_ms)

        start_frame = max(0, speech_indices[0] - pre_frames)
        end_frame = min(num_frames, speech_indices[-1] + post_frames)

        start_sample = start_frame * self.frame_size
        end_sample = min(len(audio_array), end_frame * self.frame_size)

        trimmed_audio = audio_array[start_sample:end_sample]

        stats = {
            "original_sec": round(len(audio_array) / self.config.sample_rate, 3),
            "trimmed_sec": round(len(trimmed_audio) / self.config.sample_rate, 3),
            "has_speech": True,
            "active_ratio": round(len(speech_indices) / num_frames, 3)
        }

        return trimmed_audio, True, stats


# Global default detector instance using centralized settings
_default_detector = VoiceActivityDetector()


def apply_vad(audio_array: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[np.ndarray, bool]:
    """
    Applies VAD silence trimming using centralized settings from src/stt/vad.py.

    Args:
        audio_array: 1D float32 audio numpy array.
        sample_rate: Audio sample rate in Hz (defaults to DEFAULT_SAMPLE_RATE = 16000).

    Returns:
        Tuple of (trimmed_audio_array, has_speech_bool).
    """
    detector = VoiceActivityDetector(sample_rate=sample_rate)
    trimmed, has_speech, _ = detector.process_audio(audio_array)
    return trimmed, has_speech
