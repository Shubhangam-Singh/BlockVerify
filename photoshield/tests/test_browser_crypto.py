"""The shipped browser crypto (frontend/index.html) agrees with the Python side.

Run under Node 22 (WebCrypto with Ed25519): canonical manifest digests, photo ids, Ed25519
signatures both ways, and the passphrase-encrypted key backup round trip (design §9.2).
"""
import json
import shutil
import subprocess

import pytest

from photoshield.provenance import manifest as M
from photoshield.provenance import signing as S

from .conftest import extract_js
from .helpers import make_manifest

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")

NAMES = ("PS_REGISTER", "PS_ALBUM", "PS_PROVE", "PS_PKCS8_PREFIX", "PS_KDF_ITER", "_psWebEd",
         "sha256Fallback", "bvSha256Hex", "psHex", "psUnhex", "psB64", "psUnb64", "psCanonical",
         "psDigestOf", "psPhotoIdOf", "psSignRaw", "psVerifyRaw", "psHasWebEd25519", "psPkcs8Of",
         "psKeyFromPkcs8", "psEncryptBackup", "psDecryptBackup")


def _node(body: str, tmp_path):
    js = extract_js(*NAMES).replace("let _psWebEd", "var _psWebEd")
    src = tmp_path / "t.mjs"
    src.write_text("globalThis.window = globalThis;\n" + js + "\n" + body)
    r = subprocess.run(["node", str(src)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_canonical_digest_and_photo_id_match_python(tmp_path):
    mans = [make_manifest(), make_manifest(title="Café ☕ — ünïcode \"quoted\"", seed="u"),
            make_manifest(cloaked=False, seed="p")]
    got = _node("(async () => { const ms = %s; const out = [];"
                " for (const m of ms) { const d = await psDigestOf(m); out.push([d, await psPhotoIdOf(d)]); }"
                " console.log(JSON.stringify(out)); })();" % json.dumps(mans, ensure_ascii=False), tmp_path)
    for m, (d, pid) in zip(mans, got):
        assert d == M.manifest_digest(m) and pid == M.photo_id(d)


def test_js_rejects_floats_like_python(tmp_path):
    got = _node("try { psCanonical({a: 1.5}); console.log('\"accepted\"'); }"
                " catch (e) { console.log(JSON.stringify(e.message)); }", tmp_path)
    assert "floats" in got


def test_browser_signature_verifies_in_python_and_vice_versa(tmp_path):
    d = M.manifest_digest(make_manifest())
    sk, py_pk = S.new_key(b"\x42" * 32)
    py_sig = S.sign(sk, S.register_message(d))
    got = _node("""(async () => {
      const kp = await crypto.subtle.generateKey({name: 'Ed25519'}, true, ['sign', 'verify']);
      const pk = psHex(await crypto.subtle.exportKey('raw', kp.publicKey));
      const sig = await psSignRaw({impl: 'webcrypto', privateKey: kp.privateKey}, PS_REGISTER + '%s');
      const pyOk = await psVerifyRaw('%s', '%s', PS_REGISTER + '%s');
      const pyBad = await psVerifyRaw('%s', '%s', PS_ALBUM + '%s');
      console.log(JSON.stringify({pk, sig, pyOk, pyBad}));
    })();""" % (d, py_pk, py_sig, d, py_pk, py_sig, d), tmp_path)
    assert S.verify(got["pk"], got["sig"], S.register_message(d))
    assert got["pyOk"] is True and got["pyBad"] is False


def test_encrypted_backup_round_trip_and_wrong_passphrase(tmp_path):
    got = _node("""(async () => {
      const kp = await crypto.subtle.generateKey({name: 'Ed25519'}, true, ['sign', 'verify']);
      const pk = psHex(await crypto.subtle.exportKey('raw', kp.publicKey));
      const key = {impl: 'webcrypto', privateKey: kp.privateKey};
      const b = await psEncryptBackup(key, pk, 'correct horse battery');
      const back = await psDecryptBackup(JSON.parse(JSON.stringify(b)), 'correct horse battery');
      let wrong = null;
      try { await psDecryptBackup(b, 'wrong passphrase!!'); } catch (e) { wrong = e.message; }
      const sig = await psSignRaw(back, PS_PROVE + 'x');
      console.log(JSON.stringify({pk, back: back.pkHex, wrong, iters: b.kdf.iterations,
                                  plainLeak: JSON.stringify(b).includes('302e0201'), sig}));
    })();""", tmp_path)
    assert got["back"] == got["pk"] and "Wrong passphrase" in got["wrong"]
    assert got["iters"] == 600000 and not got["plainLeak"]
    assert S.verify(got["pk"], got["sig"], b"photoshield/prove@1|x")


def test_pkcs8_import_of_a_python_seed_gives_the_same_public_key(tmp_path):
    seed = b"\x07" * 32
    _, py_pk = S.new_key(seed)
    pkcs8 = "302e020100300506032b657004220420" + seed.hex()
    got = _node("(async () => { const k = await psKeyFromPkcs8(psUnhex('%s'));"
                " console.log(JSON.stringify(k.pkHex)); })();" % pkcs8, tmp_path)
    assert got == py_pk
