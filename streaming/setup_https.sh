#!/usr/bin/env bash
# =============================================================
#  Amani AI — Automated HTTPS Setup Script
#  Usage: cd streaming && sudo bash setup_https.sh
#
#  What this script does:
#   1. Detects the server's LAN IP automatically
#   2. Installs Nginx
#   3. Generates a self-signed SSL certificate
#   4. Deploys the Nginx config with the real IP substituted
#   5. Enables and restarts Nginx
#   6. Opens port 443 in the firewall
# =============================================================

set -e  # Exit on any error

# ── Colors ────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

info()    { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Must run as root ──────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "Please run with sudo: sudo bash setup_https.sh"
fi

# ── Detect LAN / Public IP ──────────────────────────────────────
info "Detecting server IP..."
SERVER_IP=${SERVER_IP:-"196.188.240.106"}
if [[ -z "$SERVER_IP" ]]; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
fi
info "Configuring for IP: ${SERVER_IP}"

# ── Install Nginx ─────────────────────────────────────────────
info "Installing Nginx..."
apt-get update -qq
apt-get install -y nginx > /dev/null
info "Nginx installed."

# ── Generate Self-Signed SSL Certificate ─────────────────────
CERT_DIR="/etc/ssl/certs"
KEY_DIR="/etc/ssl/private"
CERT_FILE="${CERT_DIR}/amani.crt"
KEY_FILE="${KEY_DIR}/amani.key"

if [[ -f "$CERT_FILE" && -f "$KEY_FILE" ]]; then
    warn "SSL certificate already exists, skipping generation."
else
    info "Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -subj "/CN=${SERVER_IP}/O=AmaniAI/C=ET" \
        2>/dev/null
    chmod 600 "$KEY_FILE"
    info "Certificate created: ${CERT_FILE}"
fi

# ── Deploy Nginx Config ───────────────────────────────────────
CONF_SRC="$(dirname "$0")/nginx/amani.conf"
CONF_DEST="/etc/nginx/sites-available/amani"
CONF_LINK="/etc/nginx/sites-enabled/amani"

if [[ ! -f "$CONF_SRC" ]]; then
    error "Nginx config not found at: ${CONF_SRC}. Run from the streaming/ folder."
fi

info "Deploying Nginx config to ${CONF_DEST}..."
sed "s/SERVER_IP_PLACEHOLDER/${SERVER_IP}/g" "$CONF_SRC" > "$CONF_DEST"

# Remove default site if it conflicts on port 80/443
if [[ -f "/etc/nginx/sites-enabled/default" ]]; then
    warn "Removing default Nginx site to avoid port conflicts."
    rm /etc/nginx/sites-enabled/default
fi

# Create symlink
ln -sf "$CONF_DEST" "$CONF_LINK"
info "Nginx config linked."

# ── Test and Restart Nginx ────────────────────────────────────
info "Testing Nginx configuration..."
nginx -t || error "Nginx config test failed. Check ${CONF_DEST}"

info "Restarting Nginx..."
systemctl restart nginx
systemctl enable nginx

# ── Open Firewall Ports ───────────────────────────────────────
info "Configuring firewall (UFW)..."
if command -v ufw &>/dev/null; then
    ufw allow 80/tcp  > /dev/null 2>&1 || true
    ufw allow 443/tcp > /dev/null 2>&1 || true
    ufw --force reload > /dev/null 2>&1 || true
    info "Firewall ports 80 and 443 opened."
else
    warn "UFW not found. Manually open ports 80 and 443 if needed."
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  HTTPS Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  Your server is now accessible at:"
echo -e "  ${YELLOW}https://${SERVER_IP}${NC}"
echo ""
echo -e "  Flask must be running on port 5000:"
echo -e "  ${YELLOW}cd /mnt/data/chatbot_model_4_31B-it && python main.py${NC}"
echo ""
echo -e "  Browser note: Accept the self-signed cert warning"
echo -e "  (Advanced → Proceed to ${SERVER_IP})"
echo -e "${GREEN}============================================${NC}"
