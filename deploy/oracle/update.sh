#!/usr/bin/env bash
# Pull the latest code and restart. Run on the server:  bash /opt/blockverify/deploy/oracle/update.sh
# Your data (backend/data: accounts, models, photos, wallet) is not touched.
set -euo pipefail
APP=/opt/blockverify
BRANCH="$(git -C "$APP" rev-parse --abbrev-ref HEAD)"

git -C "$APP" pull --ff-only origin "$BRANCH"
"$APP/.venv/bin/pip" install -r "$APP/backend/requirements.txt"
bash "$APP/deploy/oracle/publish-frontend.sh" "$APP" /var/www/blockverify
sudo systemctl restart blockverify
sleep 3
curl -fsS -o /dev/null http://127.0.0.1:5000/api/stats && echo "Updated to $(git -C "$APP" log -1 --format='%h %s')"
