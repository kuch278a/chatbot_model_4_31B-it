#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_SRC="$SCRIPT_DIR/amani_nginx.conf"
CONF_DEST="/etc/nginx/sites-available/amani"
LINK_DEST="/etc/nginx/sites-enabled/amani"

echo "==> [1/4] Copying new Nginx configuration to $CONF_DEST..."
cp "$CONF_SRC" "$CONF_DEST"

echo "==> [2/4] Ensuring enabled site symlink exists..."
ln -sf "$CONF_DEST" "$LINK_DEST"

# Remove default site if it conflicts on port 80
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "==> Removing default conflicting site..."
    rm -f /etc/nginx/sites-enabled/default
fi

echo "==> [3/4] Testing Nginx configuration..."
nginx -t

echo "==> [4/4] Opening firewall ports (80 & 443) and reloading Nginx..."
if command -v ufw >/dev/null 2>&1; then
    ufw allow 80/tcp || true
    ufw allow 443/tcp || true
fi

systemctl reload nginx

echo ""
echo "============================================================="
echo " SUCCESS: Nginx is configured exclusively for Public IP!"
echo " Public HTTP  : http://196.188.240.106/"
echo " Public HTTPS : https://196.188.240.106/"
echo "============================================================="
