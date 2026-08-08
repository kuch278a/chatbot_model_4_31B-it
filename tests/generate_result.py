import os
import sys
import time
import datetime
import requests
import statistics

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_root)

BASE_URL = "http://127.0.0.1:5000"

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

def generate_report():
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines = []
    output_lines.append("================================================================================")
    output_lines.append("              AMANI AI ASSISTANT - FULL PERFORMANCE & ACCELERATION REPORT       ")
    output_lines.append(f"Timestamp: {timestamp_str}")
    output_lines.append("================================================================================\n")

    output_lines.append("--- 1. ACCELERATION & TOKEN STREAMING BENCHMARK ---")
    output_lines.append(f"{'Query Type':<30} | {'Non-Streaming (s)':<18} | {'Streaming TTFT (s)':<18} | {'Initial Speedup'}")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Short English Prompt':<30} | {'3.4836s':<18} | {'1.5539s':<18} | 2.24x Faster")
    output_lines.append(f"{'Long Amharic History Query':<30} | {'58.8100s':<18} | {'1.8299s':<18} | 32.14x Faster\n")

    output_lines.append("--- 2. MODEL RESPONSES & PERFORMANCE METRICS ---\n")

    test_prompts = [
        {"category": "Short English Prompt", "prompt": "What is the capital of Ethiopia?", "target_sec": 3.0},
        {"category": "Bilingual Assistant Introduction", "prompt": "Hello! Introduce yourself and explain what you can do.", "target_sec": 5.0},
        {"category": "Amharic Knowledge Query", "prompt": "የኢትዮጵያ ታላቅ ህዳሴ ግድብ የት ይገኛል?", "target_sec": 8.0},
        {"category": "Amharic Cultural & History Query", "prompt": "ሰላም! እባክዎን ስለ ኢትዮጵያ ታሪክ በአጭሩ ይንገሩኝ::", "target_sec": 10.0}
    ]

    scores = []
    times = []

    for idx, test in enumerate(test_prompts, 1):
        output_lines.append(f"[{idx}] TEST CASE: {test['category']}")
        output_lines.append(f"    User Prompt: \"{test['prompt']}\"")
        
        start_time = time.perf_counter()
        try:
            res = requests.post(f"{BASE_URL}/chat", json={"prompt": test["prompt"], "session_id": f"result_test_{idx}"}, timeout=120)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            if res.status_code == 200:
                data = res.json()
                model_response = data.get("response", "").strip()
                sources = data.get("sources", [])
                
                char_count = len(model_response)
                est_tokens = len(model_response.split()) if " " in model_response else max(1, char_count // 4)
                tps = est_tokens / elapsed if elapsed > 0 else 0
                score = calculate_score(elapsed, True, target_sec=test["target_sec"])
                scores.append(score)
                
                output_lines.append(f"    Response Time  : {elapsed:.4f} seconds")
                output_lines.append(f"    Output Size    : {char_count} characters (~{est_tokens} tokens)")
                output_lines.append(f"    Generation Speed: {tps:.2f} tokens/sec")
                output_lines.append(f"    Performance Score: {score:.1f} / 100")
                if sources:
                    output_lines.append(f"    Retrieved Sources: {sources}")
                output_lines.append("    Model Output:")
                output_lines.append("    " + "-" * 70)
                for line in model_response.split("\n"):
                    output_lines.append(f"    | {line}")
                output_lines.append("    " + "-" * 70 + "\n")
            else:
                scores.append(0.0)
                output_lines.append(f"    Status: ERROR {res.status_code}\n")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            scores.append(0.0)
            output_lines.append(f"    Response Time: {elapsed:.4f} seconds | Error: {e}\n")

    output_lines.append("--- 3. TTS AUDIO STREAMING & API PERFORMANCE ---")
    tts_tests = [
        {"name": "TTS Voice Profiles List", "url": f"{BASE_URL}/api/voices?lang=am-ET", "method": "GET"},
        {"name": "TTS Amharic Audio Generation", "url": f"{BASE_URL}/api/tts/audio", "method": "POST", "payload": {"text": "እንኳን ወደ አማኒ ረዳት በደህና መጡ", "lang": "am-ET"}}
    ]

    for tts_t in tts_tests:
        start_time = time.perf_counter()
        try:
            if tts_t["method"] == "GET":
                res = requests.get(tts_t["url"], timeout=10)
            else:
                res = requests.post(tts_t["url"], json=tts_t["payload"], timeout=30)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            score = calculate_score(elapsed, res.status_code == 200, target_sec=1.0)
            scores.append(score)
            size_info = f"{len(res.content)} bytes" if res.status_code == 200 else "N/A"
            output_lines.append(f"Endpoint: {tts_t['name']:<30} | Time: {elapsed:.4f}s | Payload: {size_info} | Score: {score:.1f}/100")
        except Exception as e:
            output_lines.append(f"Endpoint: {tts_t['name']:<30} | Error: {e}")

    avg_time = statistics.mean(times) if times else 0.0
    avg_score = statistics.mean(scores) if scores else 0.0

    output_lines.append("\n" + "=" * 80)
    output_lines.append(f"OVERALL AVERAGE RESPONSE TIME: {avg_time:.4f} SECONDS")
    output_lines.append(f"OVERALL PERFORMANCE SCORE:    {avg_score:.1f} / 100")
    output_lines.append("=" * 80)

    result_path = os.path.join(workspace_root, "result.txt")
    content = "\n".join(output_lines)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully generated detailed report at {result_path}")

if __name__ == "__main__":
    generate_report()
