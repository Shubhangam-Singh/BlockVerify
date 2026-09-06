"""Check that the Gemini key pool is configured and reachable.

    python3 backend/test_gemini.py

Reads GEMINI_API_KEYS (or GEMINI_API_KEY) from the environment / backend/.env.
It NEVER prints a key — only the masked id, so this is safe to run while screen-sharing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from gemini_keys import KeyPool, mask   # noqa: E402


def main():
    pool = KeyPool()
    if not pool.configured:
        print("No keys found.\n"
              "  cp backend/.env.example backend/.env\n"
              "  then set GEMINI_API_KEYS=key1,key2,key3")
        return 2

    print(f"{len(pool)} key(s) configured. Probing each…\n")
    import requests
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    ok = 0
    for _ in range(len(pool)):
        k = pool.acquire()
        if k is None:
            print("  (remaining keys are cooling down)")
            break
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={k.value}",
                json={"contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                      "generationConfig": {"maxOutputTokens": 8}},
                timeout=20)
            msg = ""
            if r.status_code != 200:
                try:
                    msg = (r.json().get("error") or {}).get("message", "")[:90]
                except Exception:
                    msg = r.text[:90]
            pool.report(k, r.status_code, msg)
            mark = "OK  " if r.status_code == 200 else "FAIL"
            print(f"  [{mark}] {mask(k.value)}  HTTP {r.status_code}" + (f"  {msg}" if msg else ""))
            ok += r.status_code == 200
        except Exception as e:
            pool.report(k, 503, str(e))
            print(f"  [FAIL] {mask(k.value)}  {type(e).__name__}: {e}")

    print(f"\n{ok}/{len(pool)} key(s) working.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
