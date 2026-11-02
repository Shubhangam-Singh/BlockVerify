# Browser end-to-end check (manual, not part of pytest)

Drives the Photo Shield tab in headless Chromium against a local backend: key creation, a
provenance-only registration (PoW, signature, anchor, manifest download), the public-ledger
check, verification, encrypted key export and the ownership proof.

```bash
PORT=5000 PHOTOSHIELD_ANCHOR=local PHOTOSHIELD_LEDGER_SYNC=0 python3 backend/app.py &
(cd frontend && python3 -m http.server 8080) &
python3 -c "from PIL import Image; Image.open('photoshield/tests/vectors/photo_chelsea.png').convert('RGB').save('/tmp/chelsea_q60.jpg', quality=60)"
node photoshield/tests/e2e/browser_flow.js        # needs `npm i playwright` (or a global install)
```

It prints a JSON summary; `errors` should hold nothing but CDN loads blocked by your network.
