import os
import sys
import time
import requests
import unittest
import statistics

BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:5000")

def calculate_score(time_sec, success, target_sec=2.0):
    """
    Calculates a performance score from 0.0 to 100.0.
    - 100.0 = response time <= 0.5s
    - 90-100 = response time <= target_sec
    - 70-90 = response time <= 5s
    - 50-70 = response time <= 10s
    """
    if not success:
        return 0.0
    
    if time_sec <= 0.5:
        return 100.0
    elif time_sec <= target_sec:
        return 100.0 - (time_sec - 0.5) / (target_sec - 0.5) * 10.0
    elif time_sec <= 5.0:
        return 90.0 - (time_sec - target_sec) / (5.0 - target_sec) * 20.0
    elif time_sec <= 10.0:
        return 70.0 - (time_sec - 5.0) / 5.0 * 20.0
    else:
        return max(10.0, 50.0 - (time_sec - 10.0) * 3.0)

class TestResponseTimeAndScore(unittest.TestCase):
    """
    Response time performance test suite with scoring and latency assertions.
    """

    def setUp(self):
        # Verify server availability before running latency tests
        try:
            res = requests.get(f"{BASE_URL}/health", timeout=5)
            self.assertTrue(res.status_code == 200, "Server health check failed")
        except Exception as e:
            self.fail(f"Server is not reachable at {BASE_URL}. Error: {e}")

    def test_01_health_check_latency(self):
        """Test health endpoint latency and score."""
        start = time.perf_counter()
        res = requests.get(f"{BASE_URL}/health", timeout=10)
        elapsed = time.perf_counter() - start
        
        self.assertEqual(res.status_code, 200)
        score = calculate_score(elapsed, True, target_sec=0.1)
        print(f"\n[Test Health Check] Time: {elapsed:.4f}s | Score: {score:.1f}/100")
        self.assertLess(elapsed, 1.0, f"Health check too slow: {elapsed:.4f}s")
        self.assertGreaterEqual(score, 90.0)

    def test_02_tts_voices_latency(self):
        """Test TTS voices list endpoint response time and score."""
        start = time.perf_counter()
        res = requests.get(f"{BASE_URL}/api/voices?lang=am-ET", timeout=10)
        elapsed = time.perf_counter() - start
        
        self.assertEqual(res.status_code, 200)
        score = calculate_score(elapsed, True, target_sec=0.3)
        print(f"[Test TTS Voices] Time: {elapsed:.4f}s | Score: {score:.1f}/100")
        self.assertLess(elapsed, 2.0, f"TTS voices endpoint too slow: {elapsed:.4f}s")

    def test_03_tts_audio_streaming_latency(self):
        """Test TTS audio streaming response time and score."""
        payload = {"text": "እንኳን ወደ አማኒ ረዳት በደህና መጡ", "lang": "am-ET"}
        start = time.perf_counter()
        res = requests.post(f"{BASE_URL}/api/tts/audio", json=payload, timeout=30)
        elapsed = time.perf_counter() - start
        
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Type"), "audio/mpeg")
        score = calculate_score(elapsed, True, target_sec=1.5)
        print(f"[Test TTS Streaming] Time: {elapsed:.4f}s | Stream Size: {len(res.content)} bytes | Score: {score:.1f}/100")
        self.assertLess(elapsed, 5.0, f"TTS audio streaming too slow: {elapsed:.4f}s")

    def test_04_chat_llm_short_prompt_latency(self):
        """Test Chat LLM short query response time and score."""
        payload = {"prompt": "What is the capital of Ethiopia?", "session_id": "test_perf_1"}
        start = time.perf_counter()
        res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        elapsed = time.perf_counter() - start
        
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response", data)
        score = calculate_score(elapsed, True, target_sec=3.0)
        print(f"[Test Chat LLM Short] Time: {elapsed:.4f}s | Response: '{data['response'][:60]}...' | Score: {score:.1f}/100")

    def test_05_chat_llm_amharic_prompt_latency(self):
        """Test Chat LLM Amharic query response time and score."""
        payload = {"prompt": "ሰላም! እባክዎን ስለ ኢትዮጵያ በአጭሩ ይንገሩኝ::", "session_id": "test_perf_2"}
        start = time.perf_counter()
        res = requests.post(f"{BASE_URL}/chat", json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response", data)
        score = calculate_score(elapsed, True, target_sec=10.0)
        print(f"[Test Chat LLM Amharic] Time: {elapsed:.4f}s | Length: {len(data['response'])} chars | Score: {score:.1f}/100")

def run_benchmark_summary():
    """Prints a formatted benchmark summary table with times in seconds and scores."""
    test_cases = [
        {"name": "Health Check", "url": f"{BASE_URL}/health", "method": "GET", "payload": None, "target": 0.1},
        {"name": "TTS Voices List", "url": f"{BASE_URL}/api/voices?lang=am-ET", "method": "GET", "payload": None, "target": 0.3},
        {"name": "TTS Audio Streaming", "url": f"{BASE_URL}/api/tts/audio", "method": "POST", "payload": {"text": "ሰላም", "lang": "am-ET"}, "target": 1.0},
        {"name": "LLM Short Query", "url": f"{BASE_URL}/chat", "method": "POST", "payload": {"prompt": "What is the capital of Ethiopia?"}, "target": 3.0},
        {"name": "LLM Amharic Query", "url": f"{BASE_URL}/chat", "method": "POST", "payload": {"prompt": "ሰላም! እባክዎን ስለ ኢትዮጵያ ይንገሩኝ::"}, "target": 10.0},
    ]

    print("\n==========================================================================")
    print("              RESPONSE TIME & SCORE PERFORMANCE BENCHMARK                 ")
    print("==========================================================================")
    print(f"{'Test Description':<25} | {'Response Time (s)':<18} | {'Score (0-100)':<13} | {'Status'}")
    print("-" * 74)

    scores = []
    times = []

    for test in test_cases:
        start = time.perf_counter()
        try:
            if test["method"] == "POST":
                res = requests.post(test["url"], json=test["payload"], timeout=120)
            else:
                res = requests.get(test["url"], timeout=120)
            elapsed = time.perf_counter() - start
            success = (res.status_code == 200)
            score = calculate_score(elapsed, success, target_sec=test["target"])
            status_str = "PASS" if success else "FAIL"
        except Exception as e:
            elapsed = time.perf_counter() - start
            score = 0.0
            status_str = f"ERROR: {e}"

        scores.append(score)
        times.append(elapsed)
        print(f"{test['name']:<25} | {elapsed:<18.4f} | {score:<13.1f} | {status_str}")

    avg_time = statistics.mean(times)
    avg_score = statistics.mean(scores)
    print("-" * 74)
    print(f"OVERALL AVERAGE RESPONSE TIME: {avg_time:.4f} SECONDS")
    print(f"OVERALL PERFORMANCE SCORE:    {avg_score:.1f} / 100")
    print("==========================================================================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary-only":
        run_benchmark_summary()
    else:
        run_benchmark_summary()
        unittest.main()
