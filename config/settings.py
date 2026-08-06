import os

def load_dotenv(dotenv_path=".env"):
    # Look for .env in the parent directory or current directory
    possible_paths = [dotenv_path, os.path.join(os.path.dirname(os.path.dirname(__file__)), dotenv_path)]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        # Clean up quotes if present
                        val = val.strip().strip("'\"")
                        os.environ[key.strip()] = val
            break

# Load env variables at module import time
load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/data/chatbot_model_4_31B-it/src/models/model")
    DB_PATH = os.getenv("DB_PATH", "/mnt/data/chatbot_model_4_31B-it/data/db.sqlite")
    INDEX_DIR = os.getenv("INDEX_DIR", "/mnt/data/local_data")
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/home/iotadmin/.cache/huggingface/hub/models--rasyosef--bert-amharic-text-embedding-medium/snapshots/744955b57f0f6bb689255bb15fb9ec87b9275460")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))

settings = Settings()
