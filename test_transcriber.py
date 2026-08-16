"""
Test the STT transcriber with a recorded audio file.
Usage: python3 test_transcriber.py [path/to/audio]
Converts any audio (mp3/webm/wav/m4a...) to 16kHz mono PCM via ffmpeg, then transcribes.
"""

import os
import sys
import subprocess
import tempfile

from src.stt.ethio_asr_transcriber import transcribe_audio

DEFAULT_AUDIO = os.path.join(os.path.dirname(__file__), "data", "amharic_sample.mp3")


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                wav_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with open(wav_path, "rb") as f:
            pcm_bytes = f.read()[44:]

        print(f"Transcribing: {audio_path} ({len(pcm_bytes) * 2 / 16000:.1f}s audio)")
        text = transcribe_audio(pcm_bytes, sample_rate=16000)
        print("-" * 60)
        print(f"TRANSCRIPTION: {text!r}")
        print("-" * 60)
    finally:
        os.unlink(wav_path)


if __name__ == "__main__":
    main()