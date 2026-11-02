"""PhotoShield command line. Run from the repository root:

    python -m photoshield protect photo.jpg --out shielded/          # cloak + manifest
    python -m photoshield hash a.jpg b.png ...                        # sha256 + pHash
    python -m photoshield compare a.jpg b.jpg                         # pHash Hamming distance
    python -m photoshield check shielded/manifest.json photo.jpg shielded/cloaked.png
    python -m photoshield keygen --out my.pskey                       # CLI signing key
    python -m photoshield register shielded/manifest.json --api URL --token JWT --key my.pskey
    python -m photoshield study ...                                   # experiments (P6)
    python -m photoshield origin ORIGINAL SUSPECT --manifest manifest.json --out report/ --key my.pskey
    python -m photoshield origin-verify report/origin-report.json [SUSPECT] [--recheck] [--api URL]

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


def _cmd_origin(ns) -> int:
    from .chain.client import load_key
    from .origin import heatmap as HM, report as OR
    with open(ns.manifest, encoding="utf-8") as f:
        man = json.load(f)
    with open(ns.original, "rb") as f:
        ob = f.read()
    with open(ns.suspect, "rb") as f:
        sb = f.read()
    try:
        thr = OR.load_thresholds(ns.calibration)
        sk = load_key(ns.key)[0] if ns.key else None
        out = OR.build_report(ob, sb, man, thr, sk, mode=ns.align)
    except (OR.OriginError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rep = out["report"]
    os.makedirs(ns.out, exist_ok=True)
    jp = os.path.join(ns.out, "origin-report.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
        f.write("\n")
    heat = HM.heatmap(out["original"], out["alignment"]["warped"], rep["tiles"], rep["grid"])
    hp = os.path.join(ns.out, "heatmap.png")
    heat.save(hp)
    pp = os.path.join(ns.out, "origin-report.pdf")
    HM.pdf(rep, heat, pp, include_original=ns.include_original)
    s = rep["summary"]
    print(f"{s['verdict']}: {s['match']} regions match, {s['altered']} altered, {s['uncertain']} uncertain, "
          f"{s['uncovered']} not in the suspect, {s['flat']} flat ({rep['grid']}x{rep['grid']} grid)")
    print(f"alignment {rep['alignment']['method']} ({rep['alignment']['inliers']} inliers)")
    print(f"report  -> {jp}" + ("" if rep.get("sig") else "   (UNSIGNED: pass --key to sign it)"))
    print(f"heatmap -> {hp}  (shows your original: share the PDF instead)")
    print(f"pdf     -> {pp}")
    return 0


def _cmd_origin_verify(ns) -> int:
    from .origin import report as OR
    with open(ns.report, encoding="utf-8") as f:
        rep = json.load(f)
    sb = None
    if ns.suspect:
        with open(ns.suspect, "rb") as f:
            sb = f.read()
    anchored = None
    if ns.api:
        import requests
        pid = rep.get("photo", {}).get("photoId", "")
        try:
            p = requests.get(ns.api.rstrip("/") + "/photo/proof/" + pid, timeout=20).json()
            if p.get("success"):
                anchored = {"manifestDigest": p.get("manifestDigest"), "ownerPk": p.get("ownerPk")}
            else:
                print(f"registry: {p.get('error')}", file=sys.stderr)
        except Exception as e:                       # noqa: BLE001 - report and continue offline
            print(f"registry unreachable: {e}", file=sys.stderr)
    res = OR.verify_report(rep, sb, recheck=ns.recheck, anchored=anchored)
    for c in res["checks"]:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    if anchored is None:
        print("note: the anchor was not checked (pass --api URL, or verify in the Photo Shield tab)")
    print("REPORT VALID" if res["ok"] else "REPORT INVALID")
    return 0 if res["ok"] else 1


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

    o = sub.add_parser("origin", help="origin report: which regions of SUSPECT match your registered ORIGINAL")
    o.add_argument("original")
    o.add_argument("suspect")
    o.add_argument("--manifest", required=True, help="the manifest.json of the registration")
    o.add_argument("--out", required=True)
    o.add_argument("--key", help="key file from `keygen` (the registering key) to sign the report")
    o.add_argument("--calibration", help="threshold file (default photoshield/calibration.json)")
    o.add_argument("--align", choices=("auto", "features", "resize"), default="auto")
    o.add_argument("--include-original", action="store_true", help="put your original in the PDF too")
    o.set_defaults(fn=_cmd_origin)

    ov = sub.add_parser("origin-verify", help="check an origin report (proofs, signature, anchor)")
    ov.add_argument("report")
    ov.add_argument("suspect", nargs="?")
    ov.add_argument("--recheck", action="store_true", help="recompute the suspect's tile hashes")
    ov.add_argument("--api", help="registry API to read the anchored digest and key from")
    ov.set_defaults(fn=_cmd_origin_verify)

    s = sub.add_parser("study", help="run experiments (see photoshield/experiments/study.py)")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=_cmd_study)
    return p


def main(argv=None) -> int:
    ns = build_parser().parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    sys.exit(main())
