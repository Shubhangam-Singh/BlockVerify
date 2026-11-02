"""PhotoShield command line. Run from the repository root:

    python -m photoshield protect photo.jpg --out shielded/          # cloak + manifest
    python -m photoshield hash a.jpg b.png ...                        # sha256 + pHash
    python -m photoshield compare a.jpg b.jpg                         # pHash Hamming distance
    python -m photoshield check shielded/manifest.json photo.jpg shielded/cloaked.png
    python -m photoshield keygen --out my.pskey                       # CLI signing key
    python -m photoshield register shielded/manifest.json --api URL --token JWT --key my.pskey
    python -m photoshield study ...                                   # experiments (P6)

`protect` runs entirely on this machine. Nothing is uploaded; `register` sends only the
manifest, a proof-of-work nonce and a signature.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _cmd_protect(ns) -> int:
    from .cloak.protect import protect
    from .config import from_args
    cfg = from_args(ns)
    res = protect(ns.photo, cfg, ns.out, title=ns.title)
    m = res["metrics"]
    print(f"cloaked  -> {res['cloaked_path']}  ({res['working_size'][0]}x{res['working_size'][1]})")
    print(f"manifest -> {os.path.join(ns.out, 'manifest.json')}  digest {res['manifest_digest']}")
    print(f"PSNR {m['psnr']:.2f} dB · SSIM {m['ssim']:.4f} · L-inf {m['linf']:.1f}/255 · "
          f"latent shift {m['latent_shift']:.4f}"
          + (f" · VAE recon drop {m['recon_psnr_drop']:.2f} dB" if m['recon_psnr_drop'] is not None else ""))
    print(f"attack time {res['attack_wall_s']:.1f} s on {cfg.threads} thread(s), model {cfg.model}")
    print("Keep manifest.json: it is your backup of everything not stored on chain.")
    return 0


def _cmd_hash(ns) -> int:
    from .provenance.hashing import Unhashable, phash, sha256_file
    for p in ns.files:
        try:
            ph = phash(p)
        except Unhashable as e:
            ph = f"UNHASHABLE ({e})"
        print(json.dumps({"file": p, "sha256": sha256_file(p), "phash": ph}))
    return 0


def _cmd_compare(ns) -> int:
    from .provenance.hashing import hamming, phash
    a, b = phash(ns.a), phash(ns.b)
    print(json.dumps({"a": a, "b": b, "hamming": hamming(a, b)}))
    return 0


def _cmd_check(ns) -> int:
    from .provenance import manifest as M
    from .provenance.hashing import phash, sha256_file
    man = M.validate(json.load(open(ns.manifest, encoding="utf-8")))
    ok = sha256_file(ns.original) == man["original"]["sha256"] and phash(ns.original) == man["original"]["phash"]
    print(f"original : {'OK' if ok else 'MISMATCH'}")
    good = ok
    if ns.cloaked:
        okc = man["cloaked"] is not None and sha256_file(ns.cloaked) == man["cloaked"]["sha256"]
        print(f"cloaked  : {'OK' if okc else 'MISMATCH'}")
        good = good and okc
    print(f"digest   : {M.manifest_digest(man)}")
    return 0 if good else 1


def _cmd_keygen(ns) -> int:
    from .chain.client import keygen
    pk = keygen(ns.out)
    print(f"public key {pk}\nwrote {ns.out} (keep it private; anyone with it can sign as you)")
    return 0


def _cmd_register(ns) -> int:
    from .chain.client import register_manifest
    res = register_manifest(ns.manifest, ns.api, ns.token, ns.key, privacy=ns.privacy)
    print(json.dumps(res, indent=2))
    return 0 if res.get("success") else 1


def _cmd_study(ns) -> int:
    from .experiments.study import main as study_main
    return study_main(ns.rest)


def build_parser() -> argparse.ArgumentParser:
    from .config import add_args
    p = argparse.ArgumentParser(prog="python -m photoshield", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("protect", help="cloak a photo and write manifest.json")
    pr.add_argument("photo")
    pr.add_argument("--out", required=True)
    pr.add_argument("--title")
    add_args(pr)
    pr.set_defaults(fn=_cmd_protect)

    h = sub.add_parser("hash", help="print sha256 and pHash")
    h.add_argument("files", nargs="+")
    h.set_defaults(fn=_cmd_hash)

    c = sub.add_parser("compare", help="pHash Hamming distance of two images")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(fn=_cmd_compare)

    ck = sub.add_parser("check", help="check image files against a manifest")
    ck.add_argument("manifest")
    ck.add_argument("original")
    ck.add_argument("cloaked", nargs="?")
    ck.set_defaults(fn=_cmd_check)

    k = sub.add_parser("keygen", help="create a CLI signing key file")
    k.add_argument("--out", required=True)
    k.set_defaults(fn=_cmd_keygen)

    r = sub.add_parser("register", help="anchor a manifest through the BlockVerify API")
    r.add_argument("manifest")
    r.add_argument("--api", required=True, help="e.g. https://blockverify-api-0ojc.onrender.com/api")
    r.add_argument("--token", required=True, help="JWT from /api/auth/login")
    r.add_argument("--key", required=True, help="key file from `keygen`")
    r.add_argument("--privacy", choices=("full", "hash-only"), default="full")
    r.set_defaults(fn=_cmd_register)

    s = sub.add_parser("study", help="run experiments (see photoshield/experiments/study.py)")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=_cmd_study)
    return p


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    sys.exit(main())
