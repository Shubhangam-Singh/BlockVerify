#!/usr/bin/env bash
# One-time setup of BlockVerify on an Oracle Cloud Ubuntu VM (24.04 or 22.04).
# Run it as the default "ubuntu" user:
#
#   bash setup.sh <your-domain> [branch]
#   e.g.  bash setup.sh blockverify.duckdns.org
#
# It installs Python and Caddy, clones the repository to /opt/blockverify, runs the
# backend as a systemd service on 127.0.0.1:5000, serves the frontend and proxies
# /api and /socket.io through Caddy with automatic HTTPS, and opens ports 80 and 443.
# Safe to re-run.
set -euo pipefail

DOMAIN="${1:?usage: bash setup.sh <domain> [branch]}"
BRANCH="${2:-ss2005/modest-fermat-hflkcl}"
REPO_URL="https://github.com/Shubhangam-Singh/BlockVerify.git"
APP=/opt/blockverify
WEB=/var/www/blockverify
ENVFILE=/etc/blockverify.env
RUN_USER="$(id -un)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1/7 System packages"
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip rsync curl gnupg \
  debian-keyring debian-archive-keyring apt-transport-https

step "2/7 Caddy (web server with automatic HTTPS)"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
  sudo apt-get update -y
  sudo apt-get install -y caddy
fi

step "3/7 Code in $APP (branch $BRANCH)"
sudo mkdir -p "$APP"
sudo chown "$RUN_USER":"$RUN_USER" "$APP"
if [ ! -d "$APP/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP"
else
  git -C "$APP" fetch origin "$BRANCH"
  git -C "$APP" checkout "$BRANCH"
  git -C "$APP" pull --ff-only origin "$BRANCH"
fi

step "4/7 Python environment"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --upgrade pip
"$APP/.venv/bin/pip" install -r "$APP/backend/requirements.txt"

step "5/7 Settings and the backend service"
if ! sudo test -f "$ENVFILE"; then
  sudo install -m 600 -o root -g root "$APP/deploy/oracle/blockverify.env.example" "$ENVFILE"
  echo "Created $ENVFILE. Put your ALGO_PRIVATE_KEY in it (see the README)."
fi
sudo tee /etc/systemd/system/blockverify.service >/dev/null <<UNIT
[Unit]
Description=BlockVerify backend (Flask + Socket.IO)
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP
EnvironmentFile=$ENVFILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP/.venv/bin/python backend/app.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable blockverify
# Start only once a wallet is configured: without one the backend makes a new, empty
# wallet on first boot and every registration fails.
if sudo grep -Eq '^(ALGO_PRIVATE_KEY|ALGO_MNEMONIC)=.+' "$ENVFILE"; then
  sudo systemctl restart blockverify
  WALLET_SET=1
else
  WALLET_SET=0
fi

step "6/7 Frontend and Caddy for $DOMAIN"
bash "$APP/deploy/oracle/publish-frontend.sh" "$APP" "$WEB"
sudo tee /etc/caddy/Caddyfile >/dev/null <<CADDY
$DOMAIN {
	encode gzip zstd
	handle /api/* {
		reverse_proxy 127.0.0.1:5000
	}
	handle /socket.io/* {
		reverse_proxy 127.0.0.1:5000
	}
	handle {
		root * $WEB
		file_server
	}
}
CADDY
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl enable caddy
sudo systemctl reload caddy || sudo systemctl restart caddy

step "7/7 Open ports 80 and 443 in the VM's own firewall"
# Oracle's Ubuntu images ship iptables rules that reject everything except SSH.
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 1 -p tcp -m state --state NEW --dport "$port" -j ACCEPT
  fi
done
if command -v netfilter-persistent >/dev/null; then
  sudo netfilter-persistent save
else
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
  sudo netfilter-persistent save
fi

step "Done"
if [ "$WALLET_SET" = "1" ]; then
  sleep 3
  if curl -fsS -o /dev/null http://127.0.0.1:5000/api/stats; then
    echo "Backend is running and answers on 127.0.0.1:5000."
  else
    echo "Backend is not answering yet. Check: sudo journalctl -u blockverify -n 50"
  fi
else
  echo "Backend NOT started yet: no wallet in $ENVFILE."
  echo "  sudo nano $ENVFILE          (set ALGO_PRIVATE_KEY=...)"
  echo "  sudo systemctl restart blockverify"
fi
echo
echo "Then open https://$DOMAIN  (the first visit can take a minute while Caddy gets the certificate)."
