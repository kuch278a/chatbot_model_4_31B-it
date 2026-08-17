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

# Remove default site if it conflicts on port 80
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "==> Removing default conflicting site..."
    rm -f /etc/nginx/sites-enabled/default
fi

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
echo " SUCCESS: Nginx is running as:"
echo "   [1] Reverse Proxy : Forwarding requests to Gunicorn backend"
echo "   [2] Load Balancer : Upstream pool with least_conn algorithm"
echo "   [3] Content Cache : 1GB cache zone at $CACHE_DIR"
echo "   [4] API Gateway   : Rate limiting, security headers, route filtering"
echo ""
echo " Public HTTP  : http://196.188.240.106/"
echo " Public HTTPS : https://196.188.240.106/"
echo " API Docs     : http://196.188.240.106/API/docs"
echo " Health Check : http://196.188.240.106/health"
echo "=========================================================================="
