"""Small builders shared by the tests."""
import hashlib

from photoshield.provenance import manifest as M


def hx(seed: str, n: int = 64) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:n]


def make_manifest(title="Test photo", seed="a", cloaked=True, **cloak_over):
    original = {"sha256": hx("orig" + seed), "phash": hx("po" + seed, 16), "width": 640, "height": 480}
    cl = {"sha256": hx("clk" + seed), "phash": hx("pc" + seed, 16), "format": "png"} if cloaked else None
    cloak = None
    if cloaked:
        cloak = {"model": "tiny", "eps": 8, "steps": 10, "size": 64, "seed": 0, "eot": ["resize"],
                 "psnr": "40.1234", "tool_version": "photoshield-0.1.0"}
        cloak.update(cloak_over)
    return M.build(title, original, cl, cloak)
