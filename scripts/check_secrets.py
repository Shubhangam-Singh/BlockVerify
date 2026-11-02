"""Block API keys and other secrets from being committed.

Runs as a pre-commit hook (see scripts/install-hooks.sh) and in CI. It scans the STAGED
content, not the working tree, because that is what would actually be pushed.

This exists because two live Google API keys reached the public GitHub history of this
repository via backend/test_gemini.py before it was noticed.
"""
from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [
    ("Google API key",      re.compile(r"AIza[0-9A-Za-z_\-]{30,45}")),
    ("Google OAuth token",  re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}")),
    ("AWS access key",      re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Private key block",   re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Slack token",         re.compile(r"\bxox[abprs]-[0-9A-Za-z\-]{10,}")),
    ("Bitcoin WIF",         re.compile(r"\b[5KL][1-9A-HJ-NP-Za-km-z]{50,51}\b")),
    ("Algorand mnemonic",   re.compile(r"(?:\b[a-z]{3,8}\b[ ,]+){24}\b[a-z]{3,8}\b")),
]

# Files that legitimately contain key-SHAPED text: templates and this scanner itself.
ALLOW_FILES = {"backend/.env.example", "scripts/check_secrets.py", ".gitignore"}


def staged_files():
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                         capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f.strip()]


def staged_content(path):
    """Staged blob as text, or None if it is missing or not UTF-8.

    ``text=True`` decodes stdout as UTF-8 before the caller can skip binaries,
    so a staged .xlsx or .png aborts the hook with UnicodeDecodeError.
    """
    r = subprocess.run(["git", "show", f":{path}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    raw = r.stdout
    if b"\0" in raw[:2048]:
        return None                                    # binary
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None                                    # xlsx, png, and other non-text


def scan(paths):
    hits = []
    for path in paths:
        if path in ALLOW_FILES:
            continue
        content = staged_content(path)
        if content is None:
            continue
        for lineno, line in enumerate(content.splitlines(), 1):
            for name, pat in PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                tok = m.group(0)
                # a placeholder is not a secret
                if any(w in line.lower() for w in ("your_", "example", "placeholder",
                                                   "changeme", "<key>", "redacted")):
                    continue
                hits.append((path, lineno, name, tok[:8] + "…"))
    return hits


def main():
    files = staged_files()
    if not files:
        return 0
    hits = scan(files)
    if not hits:
        return 0
    print("\n  COMMIT BLOCKED — possible secret(s) staged:\n", file=sys.stderr)
    for path, lineno, name, frag in hits:
        print(f"    {path}:{lineno}  {name}  ({frag})", file=sys.stderr)
    print("\n  Move the value into backend/.env (gitignored) and read it from the",
          "\n  environment. If this is genuinely a false positive, commit with --no-verify.\n",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
