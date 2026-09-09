import numpy as np
from io import BytesIO
import wave
import sys
import os

sys.path.append(os.path.abspath('.'))

from src.stt.ethio_asr_transcriber import transcribe_audio_blob

# Generate a 1 second 440Hz sine wave as a wav file
sample_rate = 16000
t = np.linspace(0, 1, sample_rate, False)
audio_data = np.sin(2 * np.pi * 440 * t) * 32767
audio_data = audio_data.astype(np.int16)

wav_io = BytesIO()
with wave.open(wav_io, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

audio_bytes = wav_io.getvalue()
print("Starting transcription test...")
result = transcribe_audio_blob(audio_bytes)
print(f"Result (should be empty for sine wave or some random characters): '{result}'")

