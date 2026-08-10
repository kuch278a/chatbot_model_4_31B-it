#!/usr/bin/env bash
# =============================================================
#  Amani AI — Quick Test: Gunicorn + HTTPS (No Nginx needed)
#  Usage: cd /mnt/data/chatbot_model_4_31B-it && bash streaming/start_gunicorn_https.sh
#
#  Use this for quick testing without setting up Nginx.
#  Access the server at: https://<SERVER_IP>:5000
# =============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Project root (one level above streaming/) ─────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Detect virtual env ────────────────────────────────────────
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -f "$VENV_PYTHON" ]]; then
    error ".venv not found at $PROJECT_ROOT/.venv. Activate your environment first."
fi

# ── Check/install gunicorn ────────────────────────────────────
if ! "$VENV_PYTHON" -m gunicorn --version &>/dev/null; then
    info "Installing gunicorn..."
    "$VENV_PYTHON" -m pip install gunicorn --quiet
fi

# ── SSL cert paths ────────────────────────────────────────────
CERT_FILE="/etc/ssl/certs/amani.crt"
KEY_FILE="/etc/ssl/private/amani.key"

# Auto-generate cert if missing (requires sudo)
if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
    info "SSL certificate not found. Generating self-signed cert (requires sudo)..."
    SERVER_IP=$(hostname -I | awk '{print $1}')
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -subj "/CN=${SERVER_IP}/O=AmaniAI/C=ET" 2>/dev/null
    sudo chmod 644 "$CERT_FILE"
    sudo chmod 600 "$KEY_FILE"
fi

# ── Detect IP ─────────────────────────────────────────────────
SERVER_IP=$(hostname -I | awk '{print $1}')
PORT=5000

# ── Launch Gunicorn ───────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Starting Amani AI with HTTPS${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Access at: ${YELLOW}https://${SERVER_IP}:${PORT}${NC}"
echo -e "  (Accept the self-signed cert warning in browser)"
echo -e "${GREEN}============================================${NC}"
echo ""

"$VENV_PYTHON" -m gunicorn main:app \
    --bind "0.0.0.0:${PORT}" \
    --certfile "$CERT_FILE" \
    --keyfile  "$KEY_FILE" \
    --workers  1 \
    --threads  4 \
    --timeout  300 \
    --log-level info \
    --access-logfile logs/access.log \
    --error-logfile  logs/error.log
