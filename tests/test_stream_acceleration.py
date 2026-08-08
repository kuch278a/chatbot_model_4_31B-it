import time
import requests
import json
import unittest

BASE_URL = "http://127.0.0.1:5000"

def test_streaming_acceleration(prompt="Tell me about the history of Ethiopia in detail."):
    print(f"\n==================================================")
    print(f"Testing Prompt: '{prompt[:50]}...'")
    print(f"==================================================")
    
    # 1. Non-Streaming Standard /chat
    print("Testing Standard Non-Streaming POST /chat...")
    start_time = time.perf_counter()
    res = requests.post(f"{BASE_URL}/chat", json={"prompt": prompt, "session_id": "accel_test_1"}, timeout=120)
    total_non_stream_time = time.perf_counter() - start_time
    
    if res.status_code == 200:
        non_stream_text = res.json().get("response", "")
        print(f"-> Non-Streaming Total Wait Time: {total_non_stream_time:.4f} seconds")
        print(f"-> Output Length: {len(non_stream_text)} chars")
    else:
        print(f"-> Non-Streaming Error: {res.status_code} - {res.text}")
        return

    # 2. Streaming POST /chat/stream
    print("\nTesting Token-Streaming POST /chat/stream...")
    start_time = time.perf_counter()
    ttft = None
    streamed_text = []
    
    response = requests.post(f"{BASE_URL}/chat/stream", json={"prompt": prompt, "session_id": "accel_test_2"}, stream=True, timeout=120)
    
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - start_time
                streamed_text.append(chunk)
        
        total_stream_time = time.perf_counter() - start_time
        full_text = "".join(streamed_text)
        
        print(f"-> Time To First Token (TTFT) : {ttft:.4f} seconds!")
        print(f"-> Stream Total Completion Time: {total_stream_time:.4f} seconds")
        print(f"-> Output Length               : {len(full_text)} chars")
        
        if total_non_stream_time > 0 and ttft is not None:
            perceived_speedup = total_non_stream_time / ttft
            print(f"\n🚀 PERCEIVED SPEEDUP FACTOR: {perceived_speedup:.2f}x FASTER INITIAL RESPONSE!")
            print(f"User waits {ttft:.2f}s instead of {total_non_stream_time:.2f}s to see the response begin!")
    else:
        print(f"-> Streaming Error: {response.status_code} - {response.text}")

class TestStreamingAcceleration(unittest.TestCase):
    def test_streaming_first_token_speedup(self):
        start_time = time.perf_counter()
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json={"prompt": "Hello! Introduce yourself.", "session_id": "unit_test_stream"},
            stream=True,
            timeout=60
        )
        self.assertEqual(response.status_code, 200)
        first_chunk = None
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                first_chunk = chunk
                break
        ttft = time.perf_counter() - start_time
        print(f"\n[Unit Test Stream] Time to First Token: {ttft:.4f}s")
        self.assertIsNotNone(first_chunk)
        self.assertLess(ttft, 4.0, f"Time to first token too slow: {ttft:.4f}s")

if __name__ == "__main__":
    test_streaming_acceleration("What is the capital of Ethiopia?")
    test_streaming_acceleration("ሰላም! እባክዎን ስለ ኢትዮጵያ ታሪክ በአጭሩ ይንገሩኝ::")
