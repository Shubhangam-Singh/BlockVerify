"""Shared fixtures for the PhotoShield suite.

Tests never touch the network or real model weights (CLAUDE.md A15): Algorand is mocked, the
indexer is mocked, and the cloak tests use the `tiny` random-weight encoder.
"""
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
BACKEND = os.path.join(REPO, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
VECTORS = os.path.join(REPO, "photoshield", "tests", "vectors")


def extract_js(*names, html_path=None):
    """Pull functions and const blocks VERBATIM out of frontend/index.html.

    Same technique as fedverify/tests/test_chain.py::_extract_js: the test exercises the
    shipped browser code, not a re-typed copy of it.
    """
    html = open(html_path or os.path.join(REPO, "frontend", "index.html"), encoding="utf-8").read()
    out = []
    for n in names:
        m = re.search(r"(async\s+)?function\s+" + re.escape(n) + r"\s*\([^)]*\)\s*\{", html)
        if m:
            open_ch, close_ch = "{", "}"
        else:
            m = re.search(r"const\s+" + re.escape(n) + r"\s*=\s*\[", html)
            open_ch, close_ch = "[", "]"
        if not m:                                   # a one-line scalar constant
            m1 = re.search(r"(?:const|let)\s+" + re.escape(n) + r"\s*=[^;\n]*;", html)
            assert m1, f"{n} not found in frontend/index.html"
            out.append(m1.group(0))
            continue
        i, depth = m.end() - 1, 0
        while i < len(html):
            if html[i] == open_ch:
                depth += 1
            elif html[i] == close_ch:
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = i + 1
        if open_ch == "[" and html[end:end + 1] == ";":
            end += 1
        out.append(html[m.start():end])
    return "\n".join(out)


@pytest.fixture
def vectors_dir():
    return VECTORS
