#!/usr/bin/env bash
# Copy frontend/ to the web root and point it at this server's own /api.
#   publish-frontend.sh <repo dir> <web root>
# The repository copy is left alone: it keeps calling the Render backend.
set -euo pipefail
REPO="${1:?repo dir}"
WEB="${2:?web root}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -r "$REPO/frontend/." "$STAGE/"

OLD="  : 'https://blockverify-api-0ojc.onrender.com/api';"
NEW="  : window.location.origin + '/api';   // served by the Oracle server (deploy/oracle)"
COUNT="$(grep -cF "$OLD" "$STAGE/index.html" || true)"
if [ "$COUNT" != "1" ]; then
  echo "publish-frontend: expected the Render API line once in index.html, found $COUNT." >&2
  echo "The API line changed; update OLD in deploy/oracle/publish-frontend.sh." >&2
  exit 1
fi
python3 - "$STAGE/index.html" "$OLD" "$NEW" <<'PY'
import sys
path, old, new = sys.argv[1:4]
s = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
PY

sudo mkdir -p "$WEB"
if command -v rsync >/dev/null; then
  sudo rsync -a --delete "$STAGE/" "$WEB/"
else
  sudo find "$WEB" -mindepth 1 -delete
  sudo cp -r "$STAGE/." "$WEB/"
fi
sudo chmod -R a+rX "$WEB"
echo "publish-frontend: $WEB now calls $(grep -o "window.location.origin + '/api'" "$WEB/index.html")"
