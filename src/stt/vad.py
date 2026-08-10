"""
Voice Activity Detection (VAD) module for Amani AI STT.
Provides RMS energy-based and frame-level silence filtering, speech trimming,
and activity detection for incoming audio streams and recorded audio buffers.
"""

import numpy as np


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, energy_threshold: float = 0.008, frame_ms: int = 20):
        """
        Initialize the Voice Activity Detector.

        Args:
            sample_rate: Audio sampling frequency in Hz (default: 16000).
            energy_threshold: Minimum RMS energy to classify frame as active speech.
            frame_ms: Frame window duration in milliseconds (default: 20ms).
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.frame_ms = frame_ms
        self.frame_size = int(self.sample_rate * (self.frame_ms / 1000.0))

    def compute_frame_rms(self, frame: np.ndarray) -> float:
        """Calculates Root-Mean-Square (RMS) energy for a single frame."""
        if len(frame) == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame ** 2)))

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """Determines whether a single frame contains active speech."""
        return self.compute_frame_rms(frame) > self.energy_threshold

    def process_audio(
        self,
        audio_array: np.ndarray,
        pre_padding_ms: int = 200,
        post_padding_ms: int = 200
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
        if len(audio_array) == 0:
            return audio_array, False, {"original_sec": 0, "trimmed_sec": 0, "has_speech": False}

        num_frames = len(audio_array) // self.frame_size
        if num_frames == 0:
            return audio_array, True, {"original_sec": len(audio_array)/self.sample_rate, "trimmed_sec": len(audio_array)/self.sample_rate, "has_speech": True}

        # Reshape into contiguous frames
        frames = audio_array[:num_frames * self.frame_size].reshape(num_frames, self.frame_size)
        rms_energies = np.sqrt(np.mean(frames ** 2, axis=1))
        speech_mask = rms_energies > self.energy_threshold

        if not np.any(speech_mask):
            # Pure silence detected
            return np.array([], dtype=np.float32), False, {
                "original_sec": round(len(audio_array) / self.sample_rate, 3),
                "trimmed_sec": 0.0,
                "has_speech": False
            }

        speech_indices = np.where(speech_mask)[0]

        # Calculate padding in frame counts
        pre_frames = int(pre_padding_ms / self.frame_ms)
        post_frames = int(post_padding_ms / self.frame_ms)

        start_frame = max(0, speech_indices[0] - pre_frames)
        end_frame = min(num_frames, speech_indices[-1] + post_frames)

        start_sample = start_frame * self.frame_size
        end_sample = min(len(audio_array), end_frame * self.frame_size)

        trimmed_audio = audio_array[start_sample:end_sample]

        stats = {
            "original_sec": round(len(audio_array) / self.sample_rate, 3),
            "trimmed_sec": round(len(trimmed_audio) / self.sample_rate, 3),
            "has_speech": True,
            "active_ratio": round(len(speech_indices) / num_frames, 3)
        }

        return trimmed_audio, True, stats


# Convenience module-level function
_default_vad = VoiceActivityDetector()

def apply_vad(audio_array: np.ndarray, sample_rate: int = 16000) -> tuple[np.ndarray, bool]:
    """Trims silence using standard VAD parameters."""
    vad = VoiceActivityDetector(sample_rate=sample_rate)
    trimmed, has_speech, _ = vad.process_audio(audio_array)
    return trimmed, has_speech
