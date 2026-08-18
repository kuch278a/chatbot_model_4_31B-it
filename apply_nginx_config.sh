#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_SRC="$SCRIPT_DIR/amani_nginx.conf"
CONF_DEST="/etc/nginx/sites-available/amani"
LINK_DEST="/etc/nginx/sites-enabled/amani"
CACHE_DIR="/var/cache/nginx/amani_cache"

echo "==> [1/5] Creating cache directory at $CACHE_DIR..."
mkdir -p "$CACHE_DIR"
chown -R www-data:www-data "$CACHE_DIR" 2>/dev/null || true

echo "==> [2/5] Copying new Nginx configuration to $CONF_DEST..."
cp "$CONF_SRC" "$CONF_DEST"

echo "==> [3/5] Ensuring enabled site symlink exists..."
ln -sf "$CONF_DEST" "$LINK_DEST"

# Remove duplicate or conflicting site configs in sites-enabled
rm -f /etc/nginx/sites-enabled/default
rm -f /etc/nginx/sites-enabled/amani_nginx.conf

echo "==> [4/5] Testing Nginx configuration..."
nginx -t

echo "==> [5/5] Opening firewall ports (80 & 443) and reloading Nginx..."
if command -v ufw >/dev/null 2>&1; then
    ufw allow 80/tcp || true
    ufw allow 443/tcp || true
fi

systemctl reload nginx

echo ""
echo "=========================================================================="
echo " SUCCESS: Nginx is ready for access across all networks!"
echo ""
echo " [1] Same WiFi / LAN     : http://10.100.9.209/"
echo " [2] Via Tailscale VPN    : http://100.114.122.26/"
echo " [3] Public IP (Internet) : http://196.188.240.106/"
echo "=========================================================================="
