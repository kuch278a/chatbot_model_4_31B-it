import time
import requests
import json
import statistics
import unittest

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(name, url, method="GET", payload=None):
    print(f"\n--- Testing {name} ---")
    headers = {"Content-Type": "application/json"}
    
    start_time = time.perf_counter()
    try:
        if method == "POST":
            response = requests.post(url, json=payload, headers=headers, timeout=120)
        else:
            response = requests.get(url, headers=headers, timeout=120)
        end_time = time.perf_counter()
        
        elapsed_seconds = end_time - start_time
        status_code = response.status_code
        
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            data = response.json()
            response_text = data.get("response", str(data))
            char_count = len(response_text)
        elif "audio" in content_type:
            data_bytes = response.content
            char_count = len(data_bytes)
            response_text = f"[Audio Stream: {len(data_bytes)} bytes]"
        else:
            response_text = response.text
            char_count = len(response_text)
            
        print(f"Status Code: {status_code}")
        print(f"Response Time: {elapsed_seconds:.4f} seconds")
        print(f"Response Length: {char_count} characters/bytes")
        print(f"Snippet: {response_text[:120]}...")
        
        return {
            "name": name,
            "success": status_code == 200,
            "status_code": status_code,
            "time_seconds": elapsed_seconds,
            "char_count": char_count,
            "response_text": response_text
        }
    except Exception as e:
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        print(f"Error: {e}")
        return {
            "name": name,
            "success": False,
            "status_code": 500,
            "time_seconds": elapsed_seconds,
            "char_count": 0,
            "error": str(e)
        }

def calculate_score(time_sec, success, target_sec=2.0):
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

def run_benchmark():
    print("==================================================")
    print("       Amani AI Assistant Benchmark Tool          ")
    print("==================================================")
    
    test_cases = [
        {
            "name": "Health Check Endpoint",
            "url": f"{BASE_URL}/health",
            "method": "GET",
            "payload": None,
            "target_sec": 0.1
        },
        {
            "name": "Simple Greeting Prompt (/chat)",
            "url": f"{BASE_URL}/chat",
            "method": "POST",
            "payload": {"prompt": "Hello! Introduce yourself briefly.", "session_id": "bench_1"},
            "target_sec": 2.5
        },
        {
            "name": "Amharic Prompt (/chat)",
            "url": f"{BASE_URL}/chat",
            "method": "POST",
            "payload": {"prompt": "ሰላም! እባክዎን ስለ ኢትዮጵያ በአጭሩ ይንገሩኝ::", "session_id": "bench_2"},
            "target_sec": 3.0
        },
        {
            "name": "TTS Voice List Endpoint (/api/voices)",
            "url": f"{BASE_URL}/api/voices?lang=am-ET",
            "method": "GET",
            "payload": None,
            "target_sec": 0.2
        },
        {
            "name": "TTS Audio Streaming (/api/tts/audio)",
            "url": f"{BASE_URL}/api/tts/audio",
            "method": "POST",
            "payload": {"text": "እንኳን ወደ አማኒ ረዳት በደህና መጡ", "lang": "am-ET"},
            "target_sec": 1.5
        }
    ]

    results = []
    for test in test_cases:
        res = test_endpoint(test["name"], test["url"], test["method"], test["payload"])
        score = calculate_score(res["time_seconds"], res["success"], target_sec=test["target_sec"])
        res["score"] = score
        results.append(res)

    print("\n==================================================")
    print("                 BENCHMARK SUMMARY                ")
    print("==================================================")
    print(f"{'Test Case':<35} | {'Time (s)':<10} | {'Score':<8} | {'Status'}")
    print("-" * 65)

    scores = []
    times = []
    for r in results:
        status_str = "PASS" if r["success"] else "FAIL"
        print(f"{r['name']:<35} | {r['time_seconds']:<10.3f} | {r['score']:<8.1f} | {status_str}")
        if r["success"]:
            scores.append(r["score"])
            times.append(r["time_seconds"])

    avg_score = statistics.mean(scores) if scores else 0.0
    avg_time = statistics.mean(times) if times else 0.0

    print("-" * 65)
    print(f"Overall Average Response Time: {avg_time:.3f} seconds")
    print(f"Overall Performance Score:     {avg_score:.1f} / 100")
    print("==================================================")
    return results

class TestBenchmark(unittest.TestCase):
    def test_run_benchmark(self):
        results = run_benchmark()
        for r in results:
            self.assertTrue(r["success"], f"Test failed for {r['name']}")

if __name__ == "__main__":
    run_benchmark()
