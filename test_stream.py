import asyncio
import edge_tts

async def stream_audio():
    comm = edge_tts.Communicate("Hello there", "en-US-JennyNeural")
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]

def generate():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gen = stream_audio()
    try:
        while True:
            chunk = loop.run_until_complete(gen.__anext__())
            yield chunk
    except StopAsyncIteration:
        pass
    finally:
        loop.close()

chunks = list(generate())
print(f"Generated {len(chunks)} chunks")
