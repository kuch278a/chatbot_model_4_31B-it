# አማኒ (Amani) AI — Multimodal Amharic/English Conversational Assistant

Amani AI is a modular, high-performance, multimodal conversational AI platform built for Amharic-first and English enterprise workflows. The system is specifically engineered for resource-optimized multi-GPU execution (e.g., dual-GPU NVIDIA workstations like Tesla V100/A100) running the **Gemma 4 31B-it** model with dynamic layer sharding.

The platform integrates real-time token-by-token streaming, neural Speech-to-Text (STT), high-fidelity Text-to-Speech (TTS), local multilingual RAG (Retrieval-Augmented Generation), an interactive Swagger UI, and a 4-in-1 enterprise Nginx Gateway.

---

## 🌟 Core Highlights

* **Dynamic GPU-Sharded LLM Inference**: Custom layer partitioner that splits the 60-layer Gemma 4 visual-language model across two GPUs (`cuda:0` and `cuda:1`). Vision tower modules remain on CPU, preserving GPU VRAM for extended context windows and high-throughput generation (~6.4 tok/s).
* **Real-Time Token Streaming**: Server-Sent Events (SSE) `/chat/stream` endpoint with zero buffer latency, delivering word-by-word typing responses.
* **Native Ge'ez Speech Recognition (STT)**:
  * **Primary:** `badrex/Ethio-ASR-amharic` (w2v-bert-2.0) producing 100% native Ge'ez Fidel script.
  * **Secondary:** `faster-whisper` (CTranslate2 `distil-large-v3.5` with int8 quantization).
* **Neural Text-to-Speech (TTS)**: Server-side high-fidelity voice synthesis powered by `edge-tts` featuring Microsoft Mekdes Neural (`am-ET-MekdesNeural`) and Ameha Neural (`am-ET-AmehaNeural`) voices, with browser-native Web Speech API fallback.
* **Local Semantic RAG Pipeline**: Fully offline sentence embedding pipeline (`rasyosef/bert-amharic-text-embedding-medium`) coupled with a lightweight NumPy/SQLite vector database for instant cosine similarity document search.
* **Enterprise Nginx Gateway**:
  * **Reverse Proxy:** Full TLS/SSL termination with SSE streaming pass-through (`X-Accel-Buffering: no`).
  * **Load Balancer:** `least_conn` upstream algorithm with connection pooling.
  * **Content Cache:** 1GB high-speed disk cache for static assets and synthesized TTS audio responses.
  * **API Gateway:** Rate limiting (5 req/s chat, 30 req/s API), connection limits, security headers, and route filtering.
* **Interactive API Documentation**: Embedded OpenAPI 3.0 specification and Swagger UI at `/API/docs`.

---

## 🛠️ System Architecture

```
chatbot_model_4_31B-it/
├── main.py                     # Application entrypoint & background service loader
├── amani_nginx.conf            # 4-in-1 Nginx Gateway configuration
├── apply_nginx_config.sh       # Automated Nginx & firewall deployment script
├── requirements.txt            # Python dependencies
│
├── config/
│   └── settings.py             # Environment configuration & .env dynamic loader
│
├── data/
│   └── db.sqlite               # Vector embeddings store & chat history database
│
├── API/
│   └── api_docs.py             # OpenAPI 3.0 specification & Swagger UI blueprint
│
├── src/
│   ├── models/
│   │   ├── base.py             # Abstract base classes for LLM & Embeddings
│   │   ├── llm_client.py       # Custom layer-sharded inference engine (2x GPU)
│   │   ├── accelerate.py       # HF Accelerate dynamic device dispatcher
│   │   └── embeddings.py       # Local BERT Amharic sentence embeddings
│   │
│   ├── rag/
│   │   ├── chunker.py          # Multilingual text document splitter
│   │   ├── retriever.py        # Vector similarity matcher & scorer
│   │   ├── context_filter.py   # RAG prompt context builder & filter
│   │   └── pipeline.py         # Directory indexing & document store manager
│   │
│   ├── stt/
│   │   ├── ethio_asr_transcriber.py      # Native Amharic ASR (w2v-bert-2.0)
│   │   ├── faster_whisper_transcriber.py # CTranslate2 int8 quantized Whisper
│   │   └── vad.py                        # Voice activity detection
│   │
│   ├── tts/
│   │   └── synthesizer.py      # Neural voice synthesizer & audio generator
│   │
│   ├── db/
│   │   ├── connection.py       # SQLite connection manager
│   │   ├── chat_history.py     # Session conversation CRUD operations
│   │   └── vector_store.py     # Cosine similarity vector index
│   │
│   ├── services/
│   │   └── conversation.py     # Context orchestration, history, & LLM streaming
│   │
│   └── ui/
│       ├── app.py              # Flask Blueprint & route definitions
│       ├── chat.html           # Dark-mode glassmorphic chat interface
│       ├── style.css           # UI styling & animations
│       └── components.py       # Server-side HTML render engine
│
└── scripts/
    ├── amani_backend.service   # Systemd service unit for 24/7 background uptime
    ├── index_documents.py      # Manual document ingestion script
    └── start_server.sh         # Startup shell script
```

---

## 🚀 Getting Started

### 1. Environment Requirements
* **OS:** Linux (Ubuntu 20.04/22.04 LTS recommended)
* **Python:** 3.10+
* **Hardware:** 2x GPUs with $\ge$ 32GB VRAM each (or 1x 80GB GPU)
* **Web Server:** Nginx 1.18+

### 2. Installation
Clone the repository and install the Python dependencies into a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration (`.env`)
Create a `.env` file in the root directory:
```ini
MODEL_PATH=/mnt/data/chatbot_model_4_31B-it/src/models/model
DB_PATH=/mnt/data/chatbot_model_4_31B-it/data/db.sqlite
INDEX_DIR=/mnt/data/local_data
EMBEDDING_MODEL_PATH=rasyosef/bert-amharic-text-embedding-medium
HOST=0.0.0.0
PORT=5000
LLM_BACKEND=manual
```

---

## 🚦 Running the Application

### Option A: Interactive Development Mode
```bash
python3 main.py
```

### Option B: Production with Gunicorn
```bash
./.venv/bin/gunicorn --workers 1 --threads 4 --bind 127.0.0.1:5000 --timeout 300 main:app
```

### Option C: Production 24/7 Background Daemon (Systemd)
Install the systemd service unit:
```bash
sudo cp scripts/amani_backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now amani_backend
```

Check status:
```bash
sudo systemctl status amani_backend
```

---

## 🌐 Nginx Gateway Configuration

Deploy the production Nginx reverse proxy, load balancer, cache, and API gateway:

```bash
sudo ./apply_nginx_config.sh
```

Once applied, the application will be accessible via:
* **Frontend Web UI:** `http://<SERVER_IP>/` or `https://<SERVER_IP>/`
* **Swagger API Docs:** `http://<SERVER_IP>/API/docs`
* **Health Check:** `http://<SERVER_IP>/health`

---

## 📡 API Reference & Endpoints

| Endpoint | Method | Content-Type | Description |
| :--- | :--- | :--- | :--- |
| **`/`** | `GET` | `text/html` | Interactive Web Chat Interface |
| **`/chat/stream`** | `POST` | `application/json` | Real-time SSE token-by-token streaming |
| **`/chat`** | `POST` | `application/json` | Synchronous blocking chat response |
| **`/api/stt`** | `POST` | `multipart/form-data` | Transcribes audio files to Amharic Ge'ez text |
| **`/api/tts/audio`** | `GET` | `audio/mpeg` | Synthesizes speech from text (12h Nginx cache) |
| **`/api/voices`** | `GET` | `application/json` | Returns list of available neural voices |
| **`/files`** | `GET` | `application/json` | Lists all indexed RAG knowledge documents |
| **`/refresh`** | `POST` | `application/json` | Re-indexes documents in `INDEX_DIR` |
| **`/clear`** | `POST` | `application/json` | Clears conversation history for active session |
| **`/health`** | `GET` | `application/json` | Service health status (`ready` or `loading`) |
| **`/API/docs`** | `GET` | `text/html` | Interactive Swagger UI API Explorer |

---

### cURL Examples

#### 1. Real-Time Token Streaming
```bash
curl -N -X POST http://127.0.0.1:5000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ሰላም! አማኒ ማን ነው?", "session_id": "user_001"}'
```

#### 2. Speech-to-Text (STT)
```bash
curl -X POST http://127.0.0.1:5000/api/stt \
  -F "audio=@recording.wav"
```

#### 3. Text-to-Speech (TTS)
```bash
curl "http://127.0.0.1:5000/api/tts/audio?text=ሰላም&lang=am-ET" --output speech.mp3
```

---

## 📄 License & Attribution
Developed for high-efficiency bilingual AI research and deployment at the Ethiopian Artificial Intelligence Institute (EAII).
