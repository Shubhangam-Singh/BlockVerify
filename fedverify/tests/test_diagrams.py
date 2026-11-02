"""Phase 10.4: every module, class and function named in docs/diagrams/*.mmd exists in the repo.

The diagrams name code in four ways, and each is checked against an index built from the source itself (Python by
its syntax tree, the dashboard's JavaScript by its `function` declarations):
  a path       survival/runner.py, make_fl_healthcare_all.py   -> a repo file ending with that path
  a call       Class.method()  or  function()                  -> that method (on the class or a repo base class)
                                                                   or that function
  a class      CamelCase words such as KeyRegistry              -> a class defined in the repo
  classDiagram `class X { +m() }` and every relationship        -> class X exists and has method m
Words that are product names, not code, are listed in NOT_CODE.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIAGRAMS = REPO / "docs" / "diagrams"
EXPECTED = ("architecture", "dfd_level0", "dfd_level1", "use_case", "class_diagram", "sequence_round")
SOURCE_DIRS = ("fedverify", "backend", "evaluation")
NOT_CODE = {"BlockVerify", "FedVerify", "FedXGBBagging", "PoW", "TestNet"}
CALL = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\(\)")
PATH = re.compile(r"[\w./-]+\.(?:py|html)\b")
CAMEL = re.compile(r"\b[A-Z][A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*\b")


def build_index():
    files, classes, functions = [], {}, set()
    for d in SOURCE_DIRS:
        for p in (REPO / d).rglob("*.py"):
            if "__pycache__" in p.parts or "node_modules" in p.parts:
                continue
            files.append(p.relative_to(REPO).as_posix())
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    methods = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                    bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", None) for b in node.bases]
                    entry = classes.setdefault(node.name, {"methods": set(), "bases": set()})
                    entry["methods"] |= methods
                    entry["bases"] |= {b for b in bases if b}
                    functions |= methods
    html = REPO / "frontend" / "index.html"
    files.append("frontend/index.html")
    js = set(re.findall(r"(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(", html.read_text(encoding="utf-8")))
    return files, classes, functions | js


FILES, CLASSES, FUNCTIONS = build_index()


def has_method(cls: str, method: str, seen=None) -> bool:
    seen = seen or set()
    if cls in seen or cls not in CLASSES:
        return False
    seen.add(cls)
    c = CLASSES[cls]
    return method in c["methods"] or any(has_method(b, method, seen) for b in c["bases"])


def body(path: Path) -> str:
    return "\n".join(line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("%%"))


def names_in(text: str) -> dict:
    return {"paths": set(PATH.findall(text)), "calls": set(CALL.findall(text)),
            "classes": set(CAMEL.findall(text)) - NOT_CODE}


def test_every_required_diagram_exists():
    missing = [n for n in EXPECTED if not (DIAGRAMS / f"{n}.mmd").exists()]
    assert not missing, f"missing diagrams: {missing}"


def test_every_named_module_class_and_function_exists():
    problems = []
    for p in sorted(DIAGRAMS.glob("*.mmd")):
        found = names_in(body(p))
        for path in found["paths"]:
            if not any(f == path or f.endswith("/" + path) for f in FILES):
                problems.append(f"{p.name}: no file {path}")
        for call in found["calls"]:
            if "." in call:
                cls, method = call.split(".", 1)
                if not has_method(cls, method):
                    problems.append(f"{p.name}: no method {call}")
            elif call not in FUNCTIONS and call not in CLASSES:
                problems.append(f"{p.name}: no function {call}")
        for cls in found["classes"]:
            if cls not in CLASSES:
                problems.append(f"{p.name}: no class {cls}")
    assert not problems, "\n".join(problems)


def test_class_diagram_members_and_relationships_exist():
    text = body(DIAGRAMS / "class_diagram.mmd")
    problems = []
    for cls, members in re.findall(r"class\s+(\w+)\s*\{([^}]*)\}", text):
        for m in re.findall(r"[+\-#~]?\s*(\w+)\(\)", members):
            if not has_method(cls, m):
                problems.append(f"{cls}.{m}() does not exist")
    for cls in re.findall(r"^\s*class\s+(\w+)", text, re.M):
        if cls not in CLASSES:
            problems.append(f"class {cls} does not exist")
    for line in text.splitlines():
        rel = re.match(r'\s*(\w+)\s*(?:"[^"]*"\s*)?(?:<\|--|\*--|o--|-->|\.\.>|--|\.\.)\s*(?:"[^"]*"\s*)?(\w+)', line)
        if rel:
            for cls in rel.groups():
                if cls not in CLASSES:
                    problems.append(f"relationship names {cls}, which is not a class")
    assert not problems, "\n".join(problems)


def test_the_checker_catches_a_made_up_name():
    fake = names_in('X["Train (made_up_function()), FoldCommitter.fly(), NoSuchThing, survival/nowhere.py"]')
    assert "made_up_function" in fake["calls"] and "made_up_function" not in FUNCTIONS
    assert not has_method("FoldCommitter", "fly") and has_method("FoldCommitter", "process")
    assert "NoSuchThing" in fake["classes"] and "NoSuchThing" not in CLASSES
    assert not any(f.endswith("survival/nowhere.py") for f in FILES)
    assert has_method("MeteredAnchor", "anchor_round")                  # inherited from ChainAnchor
