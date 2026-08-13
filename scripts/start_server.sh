#!/bin/bash
# Start script for Amani AI with Gunicorn WSGI Server

PROJECT_DIR="/mnt/data/chatbot_model_4_31B-it"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/gunicorn"

cd "$PROJECT_DIR" || exit 1

echo "Starting Amani AI server via Gunicorn (1 worker, 4 threads)..."
exec "$VENV_PYTHON" \
    --workers 1 \
    --threads 4 \
    --bind 127.0.0.1:5000 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    main:app
