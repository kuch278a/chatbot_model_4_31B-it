"""
Audio Quality Constraints & Validation for CTC-based ASR Encoders.
Ensures input audio meets requirements for wav2vec2/CTC models (Ethio-ASR).
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ─── CTC Encoder Requirements (wav2vec2 / Ethio-ASR) ──────────────────────────
REQUIRED_SAMPLE_RATE: int = 16000
REQUIRED_CHANNELS: int = 1  # Mono only
MIN_DURATION_SEC: float = 0.3   # Too short = unreliable CTC alignment
MAX_DURATION_SEC: float = 60.0  # CTC quadratic memory blowup protection
TARGET_RMS_DBFS: float = -20.0  # Target loudness (dBFS)
MAX_PEAK_DBFS: float = -1.0     # Headroom to prevent clipping
MIN_SNR_DB: float = 10.0        # Minimum signal-to-noise ratio


@dataclass
class AudioQualityMetrics:
    """Container for audio quality analysis results."""
    sample_rate: int
    channels: int
    duration_sec: float
    num_samples: int
    rms_dbfs: float
    peak_dbfs: float
    snr_db: Optional[float]
    is_clipped: bool
    is_silent: bool
    passes_constraints: bool
    violations: list[str]


def analyze_audio_quality(
    audio_array: np.ndarray,
    sample_rate: int = REQUIRED_SAMPLE_RATE,
) -> AudioQualityMetrics:
    """
    Comprehensive audio quality analysis for CTC encoder compatibility.
    
    Args:
        audio_array: 1D or 2D float32 numpy array (normalized to [-1, 1])
        sample_rate: Audio sample rate in Hz
        
    Returns:
        AudioQualityMetrics with pass/fail and detailed measurements
    """
    violations = []
    
    # Handle stereo → mono conversion for analysis
    if audio_array.ndim == 2:
        channels = audio_array.shape[1]
        audio_mono = np.mean(audio_array, axis=1) if channels > 1 else audio_array[:, 0]
    else:
        channels = 1
        audio_mono = audio_array
    
    num_samples = len(audio_mono)
    duration_sec = num_samples / sample_rate
    
    # RMS energy (dBFS)
    rms = np.sqrt(np.mean(audio_mono ** 2))
    rms_dbfs = 20 * np.log10(rms + 1e-10)
    
    # Peak level (dBFS)
    peak = np.max(np.abs(audio_mono))
    peak_dbfs = 20 * np.log10(peak + 1e-10)
    
    # Clipping detection (samples at ±1.0)
    clipped_samples = np.sum(np.abs(audio_mono) >= 0.99)
    is_clipped = clipped_samples > (num_samples * 0.001)  # >0.1% clipped
    
    # Silence detection
    is_silent = rms < 1e-4
    
    # SNR estimation (simple: ratio of signal power to noise floor estimate)
    snr_db = None
    if not is_silent:
        # Estimate noise floor from bottom 10% of frame energies
        frame_size = 512
        num_frames = max(1, len(audio_mono) // frame_size)
        if num_frames > 10:
            frames = audio_mono[:num_frames * frame_size].reshape(num_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            noise_floor = np.percentile(frame_rms, 10)
            signal_level = np.percentile(frame_rms, 90)
            if noise_floor > 1e-10:
                snr_db = 20 * np.log10(signal_level / noise_floor)
    
    # ─── Constraint Checks ────────────────────────────────────────────────────
    
    # Sample rate
    if sample_rate != REQUIRED_SAMPLE_RATE:
        violations.append(f"sample_rate={sample_rate}Hz (required {REQUIRED_SAMPLE_RATE}Hz)")
    
    # Channels
    if channels != REQUIRED_CHANNELS:
        violations.append(f"channels={channels} (required mono={REQUIRED_CHANNELS})")
    
    # Duration
    if duration_sec < MIN_DURATION_SEC:
        violations.append(f"duration={duration_sec:.2f}s < {MIN_DURATION_SEC}s minimum")
    if duration_sec > MAX_DURATION_SEC:
        violations.append(f"duration={duration_sec:.2f}s > {MAX_DURATION_SEC}s maximum")
    
    # Loudness
    if rms_dbfs < TARGET_RMS_DBFS - 15:  # Allow 15dB below target
        violations.append(f"rms={rms_dbfs:.1f}dBFS too quiet (target ~{TARGET_RMS_DBFS}dBFS)")
    if rms_dbfs > TARGET_RMS_DBFS + 10:
        violations.append(f"rms={rms_dbfs:.1f}dBFS too loud (target ~{TARGET_RMS_DBFS}dBFS)")
    
    # Headroom / clipping
    if peak_dbfs > MAX_PEAK_DBFS:
        violations.append(f"peak={peak_dbfs:.1f}dBFS exceeds headroom limit {MAX_PEAK_DBFS}dBFS")
    if is_clipped:
        violations.append(f"clipping detected ({clipped_samples} samples at ±1.0)")
    
    # SNR
    if snr_db is not None and snr_db < MIN_SNR_DB:
        violations.append(f"snr={snr_db:.1f}dB < {MIN_SNR_DB}dB minimum")
    
    passes = len(violations) == 0
    
    if not passes:
        logger.warning(
            "[AudioQuality] Constraints violated: %s | "
            "sr=%d ch=%d dur=%.2fs rms=%.1fdBFS peak=%.1fdBFS snr=%s clipped=%s silent=%s",
            "; ".join(violations), sample_rate, channels, duration_sec,
            rms_dbfs, peak_dbfs, f"{snr_db:.1f}" if snr_db else "N/A",
            is_clipped, is_silent
        )
    
    return AudioQualityMetrics(
        sample_rate=sample_rate,
        channels=channels,
        duration_sec=duration_sec,
        num_samples=num_samples,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        snr_db=snr_db,
        is_clipped=is_clipped,
        is_silent=is_silent,
        passes_constraints=passes,
        violations=violations,
    )


def normalize_audio(
    audio_array: np.ndarray,
    target_rms_dbfs: float = TARGET_RMS_DBFS,
    max_peak_dbfs: float = MAX_PEAK_DBFS,
) -> np.ndarray:
    """
    Normalize audio to target loudness with peak limiting.
    
    Args:
        audio_array: Input float32 audio array
        target_rms_dbfs: Target RMS level in dBFS
        max_peak_dbfs: Maximum peak level in dBFS (headroom)
        
    Returns:
        Normalized audio array
    """
    if len(audio_array) == 0:
        return audio_array
    
    # Current RMS and peak
    rms = np.sqrt(np.mean(audio_array ** 2))
    peak = np.max(np.abs(audio_array))
    
    if rms < 1e-10:
        logger.debug("[AudioQuality] Silent audio, skipping normalization")
        return audio_array
    
    current_rms_dbfs = 20 * np.log10(rms)
    current_peak_dbfs = 20 * np.log10(peak)
    
    # Calculate gain to reach target RMS
    gain_db = target_rms_dbfs - current_rms_dbfs
    gain_linear = 10 ** (gain_db / 20)
    
    # Limit gain to prevent peak exceeding max_peak_dbfs
    max_gain_linear = 10 ** ((max_peak_dbfs - current_peak_dbfs) / 20)
    gain_linear = min(gain_linear, max_gain_linear)
    
    # Apply gain
    normalized = audio_array * gain_linear
    
    # Final safety clip
    normalized = np.clip(normalized, -1.0, 1.0)
    
    new_rms_dbfs = 20 * np.log10(np.sqrt(np.mean(normalized ** 2)) + 1e-10)
    new_peak_dbfs = 20 * np.log10(np.max(np.abs(normalized)) + 1e-10)
    
    logger.debug(
        "[AudioQuality] Normalized: rms %.1f→%.1f dBFS, peak %.1f→%.1f dBFS, gain=%.2fx",
        current_rms_dbfs, new_rms_dbfs, current_peak_dbfs, new_peak_dbfs, gain_linear
    )
    
    return normalized.astype(np.float32)


def ensure_audio_constraints(
    audio_array: np.ndarray,
    sample_rate: int = REQUIRED_SAMPLE_RATE,
    auto_normalize: bool = True,
    auto_resample: bool = True,
) -> Tuple[np.ndarray, AudioQualityMetrics]:
    """
    Enforce CTC encoder audio constraints with optional auto-fix.
    
    Args:
        audio_array: Input audio (float32, mono or stereo)
        sample_rate: Input sample rate
        auto_normalize: Apply loudness normalization if needed
        auto_resample: Resample to 16kHz if needed
        
    Returns:
        Tuple of (processed_audio_array, quality_metrics)
    """
    from scipy.signal import resample
    
    # Initial analysis
    metrics = analyze_audio_quality(audio_array, sample_rate)
    
    processed = audio_array.copy()
    
    # Convert stereo to mono
    if processed.ndim == 2 and processed.shape[1] > 1:
        processed = np.mean(processed, axis=1)
        logger.debug("[AudioQuality] Converted stereo to mono")
    
    # Resample if needed
    if auto_resample and sample_rate != REQUIRED_SAMPLE_RATE:
        num_samples = int(len(processed) * REQUIRED_SAMPLE_RATE / sample_rate)
        processed = resample(processed, num_samples)
        sample_rate = REQUIRED_SAMPLE_RATE
        logger.debug("[AudioQuality] Resampled to %dHz", REQUIRED_SAMPLE_RATE)
    
    # Normalize loudness
    if auto_normalize and not metrics.passes_constraints:
        processed = normalize_audio(processed)
        # Re-analyze after normalization
        metrics = analyze_audio_quality(processed, sample_rate)
    
    # Final duration clamp (hard limit for CTC memory)
    max_samples = REQUIRED_SAMPLE_RATE * MAX_DURATION_SEC
    if len(processed) > max_samples:
        processed = processed[-max_samples:]  # Keep most recent
        logger.warning("[AudioQuality] Clamped audio to %.1fs (CTC memory limit)", MAX_DURATION_SEC)
    
    return processed, metrics


# Convenience function for transcriber integration
def validate_for_ctc(audio_array: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, str]:
    """
    Quick validation for CTC encoder readiness.
    
    Returns:
        (is_valid, error_message)
    """
    metrics = analyze_audio_quality(audio_array, sample_rate)
    if metrics.passes_constraints:
        return True, ""
    return False, "; ".join(metrics.violations)