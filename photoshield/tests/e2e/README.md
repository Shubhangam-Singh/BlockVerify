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

## Origin reports (PhotoShield v2)

`origin_flow.js` covers:
- registering a photo with its region map
- making a report from the original plus an edited copy, with no manifest (rebuilt from the
  owner's registry record)
- downloading the report and checking it in the page, bound to the copy
- the routing of a cropped copy to the CLI

Write the test images first, then run it against a fresh local registry:

```bash
mkdir -p /tmp/ps_e2e && python3 -c "
import skimage.data as sd; from PIL import Image
o = sd.astronaut(); Image.fromarray(o).save('/tmp/ps_e2e/me.png')
f = o.copy(); f[40:200,150:320] = sd.coffee()[100:260,200:370]
Image.fromarray(f).resize((400,400)).save('/tmp/ps_e2e/fake.jpg', quality=82)
Image.fromarray(f).crop((30,30,480,470)).save('/tmp/ps_e2e/fake_cropped.jpg', quality=82)"
```

```bash
PS_E2E=/tmp/ps_e2e node photoshield/tests/e2e/origin_flow.js
```

Expected:
- the edited copy gets **DERIVED FROM YOUR REGISTERED PHOTO**
- the check says **REPORT VALID**
- the cropped copy is sent to `python -m photoshield origin`
