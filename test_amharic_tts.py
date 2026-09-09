import asyncio
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath('.'))

from src.tts.synthesizer import VoiceSynthesizer

async def test_stream():
    synth = VoiceSynthesizer()
    text = "ሰላም፣ ይህ የድምፅ ሙከራ ነው።"
    lang = "am-ET"
    
    print(f"Testing Amharic stream for: {text}")
    chunk_count = 0
    total_bytes = 0
    
    async for chunk in synth.generate_audio_stream(text=text, lang=lang):
        chunk_count += 1
        total_bytes += len(chunk)
    
    print(f"Success! Received {chunk_count} chunks, totaling {total_bytes} bytes.")

if __name__ == "__main__":
    asyncio.run(test_stream())
