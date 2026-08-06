# አማኒ (Amani) AI: Bilingual English/Amharic RAG Chatbot

Amani AI is a modular, high-performance, bilingual retrieval-augmented generation (RAG) chatbot specifically optimized for sharded execution on resource-constrained multi-GPU environments (e.g., dual-Tesla V100 workstations). 

The system operates offline using the **Gemma 4 31B-it** model sharded manually across two GPUs, combined with a local multilingual Amharic sentence embedding pipeline, SQLite-backed conversation persistence, and a browser-native voice (STT/TTS) interface.

---

## 🌟 Core Features

- **Dynamic GPU-Sharded Inference**: Custom layer partitioning that splits the 60-layer Gemma 4 visual-language model across two GPUs (`cuda:0` and `cuda:1`), featuring a dynamic device dispatcher that intercepts model operations to route boundary tensors (like `embed_scale` and `rotary_emb`) to the correct GPU.
- **Multilingual Sentence Embeddings**: Fully local embedding pipeline utilizing `rasyosef/bert-amharic-text-embedding-medium` for semantic indexing of local documents.
- **Local SQLite Vector Store**: A lightweight NumPy-backed vector database that performs rapid cosine similarity search and stores retrieved snippets persistently.
- **Bilingual Voice Studio**: Beautiful, interactive dark-themed UI that leverages browser-native **Web Speech API** for real-time speech-to-text (STT) and text-to-speech (TTS) synthesis in English and Amharic.
- **Context-Aware Retrieval**: Full RAG pipeline featuring recursive character chunking and a source transparency interface showing retrieval match scores and snippets.

---

## 🛠️ Architecture Overview

```
├── config/
│   └── settings.py          # Pydantic settings loading from .env
│
├── data/
│   └── db.sqlite            # Persisted vector index and chat history tables
│
├── scripts/
│   └── index_documents.py   # Document ingestion and indexing pipeline
│
└── src/
    ├── models/
    │   ├── base.py          # Abstract interfaces for LLM and embeddings
    │   ├── embeddings.py    # Local bert-amharic model & PyTorch pooling
    │   └── llm_client.py    # Gemma4ForConditionalGeneration client & sharder
    │
    ├── db/
    │   ├── connection.py    # SQLite connections initialization
    │   ├── chat_history.py  # Chat history CRUD operations
    │   └── vector_store.py  # NumPy cosine-similarity vector store
    │
    ├── rag/
    │   ├── chunker.py       # Recursive text character chunking
    │   ├── retriever.py     # Embedding search matcher
    │   └── pipeline.py      # Folder document index manager
    │
    ├── services/
    │   └── conversation.py  # Context orchestration and model memory
    │
    └── ui/
        ├── app.py           # Flask Blueprint API routes
        ├── chat.html        # Bilingual interface templates & script controls
        └── style.css        # Premium glassmorphic styling & animation rules
```

---

## 🚀 Setup & Installation

### 1. Requirements
Ensure your environment is running Python 3.10+ and PyTorch with CUDA support. Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```ini
MODEL_PATH=/path/to/gemma4-31b-it-model
DB_PATH=./data/db.sqlite
INDEX_DIR=/path/to/local_knowledge_documents
EMBEDDING_MODEL_PATH=/path/to/cached/bert-amharic-model
HOST=0.0.0.0
PORT=5000
```

### 3. Ingest Documents
Place your text files (`.txt`) in your configured `INDEX_DIR` directory and index them:
```bash
python scripts/index_documents.py
```

### 4. Run the Server
Launch the main Flask application:
```bash
python main.py
```

---

## 🗣️ Voice Integration Notes
The voice interface operates directly inside the browser using standard web APIs. No additional cloud transcription API keys or server-side speech modules are required:
- **Speech Recognition**: Utilizes `webkitSpeechRecognition` to dictate inputs.
- **Speech Synthesis**: Utilizes `speechSynthesis` to speak responses, automatically mapping Amharic texts to compatible voices.
