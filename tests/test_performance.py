import time
import requests
import json
import unittest
import statistics

BASE_URL = "http://127.0.0.1:5000"

def benchmark_llm():
    prompts = [
        {"name": "Short English Query", "prompt": "What is the capital of Ethiopia?", "target_sec": 3.0},
        {"name": "Standard Greeting", "prompt": "Hi, who are you and what can you help me with?", "target_sec": 5.0},
        {"name": "Amharic Knowledge Query", "prompt": "የኢትዮጵያ ታላቅ ህዳሴ ግድብ የት ይገኛል?", "target_sec": 8.0},
    ]
    
    llm_results = []
    for p in prompts:
        start = time.perf_counter()
        res = requests.post(f"{BASE_URL}/chat", json={"prompt": p["prompt"], "session_id": "perf_test"})
        elapsed = time.perf_counter() - start
        
        if res.status_code == 200:
            data = res.json()
            text = data.get("response", "")
            char_count = len(text)
            est_tokens = len(text.split()) if " " in text else max(1, char_count // 4)
            tps = est_tokens / elapsed if elapsed > 0 else 0
            speed_score = max(0, min(100, 100 - max(0, (elapsed - p["target_sec"]) * 5)))
            
            llm_results.append({
                "name": p["name"],
                "time_sec": elapsed,
                "length_chars": char_count,
                "est_tokens": est_tokens,
                "tps": tps,
                "score": speed_score,
                "status": "PASS"
            })
        else:
            llm_results.append({
                "name": p["name"],
                "time_sec": elapsed,
                "length_chars": 0,
                "est_tokens": 0,
                "tps": 0,
                "score": 0.0,
                "status": "FAIL"
            })
            
    return llm_results

def benchmark_tts():
    tts_tests = [
        {"name": "TTS Voice List", "url": f"{BASE_URL}/api/voices?lang=am-ET", "method": "GET", "target_sec": 0.3},
        {"name": "TTS Short Audio Stream", "url": f"{BASE_URL}/api/tts/audio", "method": "POST", "payload": {"text": "ሰላም", "lang": "am-ET"}, "target_sec": 1.0},
        {"name": "TTS Long Audio Stream", "url": f"{BASE_URL}/api/tts/audio", "method": "POST", "payload": {"text": "እንኳን ወደ አማኒ ረዳት በደህና መጡ! ዛሬ እንዴት ልረዳዎት እችላለሁ?", "lang": "am-ET"}, "target_sec": 2.0},
    ]
    
    tts_results = []
    for test in tts_tests:
        start = time.perf_counter()
        if test["method"] == "GET":
            res = requests.get(test["url"])
        else:
            res = requests.post(test["url"], json=test["payload"])
        elapsed = time.perf_counter() - start
        
        if res.status_code == 200:
            bytes_size = len(res.content)
            score = max(0, min(100, 100 - max(0, (elapsed - test["target_sec"]) * 20)))
            tts_results.append({
                "name": test["name"],
                "time_sec": elapsed,
                "bytes": bytes_size,
                "score": score,
                "status": "PASS"
            })
        else:
            tts_results.append({
                "name": test["name"],
                "time_sec": elapsed,
                "bytes": 0,
                "score": 0.0,
                "status": "FAIL"
            })
            
    return tts_results

class TestPerformanceMetrics(unittest.TestCase):
    def test_llm_performance(self):
        results = benchmark_llm()
        for r in results:
            self.assertEqual(r["status"], "PASS")

    def test_tts_performance(self):
        results = benchmark_tts()
        for r in results:
            self.assertEqual(r["status"], "PASS")

if __name__ == "__main__":
    print("Running LLM Benchmark...")
    llm_res = benchmark_llm()
    print("Running TTS Benchmark...")
    tts_res = benchmark_tts()
    
    print("\n--- LLM Performance ---")
    for r in llm_res:
        print(f"{r['name']}: {r['time_sec']:.2f}s | {r['est_tokens']} tokens ({r['tps']:.1f} tps) | Score: {r['score']:.1f}/100")
        
    print("\n--- TTS Performance ---")
    for r in tts_res:
        print(f"{r['name']}: {r['time_sec']:.2f}s | {r['bytes']} bytes | Score: {r['score']:.1f}/100")
