# Amani AI — HTTPS Remote Access

This folder contains everything needed to expose the Amani AI server
to other computers on your network via **HTTPS**.

---

## Folder Structure

```
streaming/
├── nginx/
│   └── amani.conf              ← Nginx reverse proxy config (HTTPS + SSE streaming)
├── setup_https.sh              ← Full automated setup (Nginx + SSL) — RECOMMENDED
├── start_gunicorn_https.sh     ← Quick HTTPS test without Nginx
└── README.md                   ← This file
```

---

## Option A — Full Setup with Nginx (Recommended)

Nginx handles HTTPS and correctly proxies SSE streaming to Flask.

```bash
# 1. Run the setup script from the streaming/ folder
cd /mnt/data/chatbot_model_4_31B-it/streaming
sudo bash setup_https.sh
```

The script will:
- Auto-detect your server's LAN IP
- Install Nginx
- Generate a self-signed SSL certificate
- Deploy the Nginx config
- Open ports 80 and 443 in the firewall

```bash
# 2. Start your Flask server (in another terminal)
cd /mnt/data/chatbot_model_4_31B-it
python main.py
```

```bash
# 3. From any other computer on the network, open:
https://<YOUR_SERVER_IP>
```

> **Browser warning**: Click **Advanced → Proceed** to bypass the self-signed cert warning.

---

## Option B — Quick Test with Gunicorn (No Nginx)

Use this if you just want to test quickly without Nginx setup.

```bash
cd /mnt/data/chatbot_model_4_31B-it
bash streaming/start_gunicorn_https.sh
```

Access at:
```
https://<YOUR_SERVER_IP>:5000
```

---

## Finding Your Server IP

Run this on the **server machine**:
```bash
hostname -I
# Example: 192.168.1.50
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `502 Bad Gateway` | Flask is not running. Start: `python main.py` |
| SSE tokens don't stream | Ensure `proxy_buffering off` is in nginx config |
| Can't connect from other PC | Check firewall: `sudo ufw status` |
| `nginx -t` fails | Check `/etc/nginx/sites-available/amani` for IP typos |
| Browser shows "Not Secure" | Normal for self-signed certs — click Advanced → Proceed |

---

## Nginx Management Commands

```bash
sudo systemctl status nginx     # Check if running
sudo systemctl restart nginx    # Restart after config changes
sudo nginx -t                   # Test config before restarting
sudo tail -f /var/log/nginx/error.log  # View errors
```
