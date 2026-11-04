"""Guard for the docstring conventions of docs/contributing_docstrings.md.

Three checks run over every module of ``hcwf``:

* ruff's pydocstyle rules (``D``, numpy convention, plus D401 imperative mood and D417 documented arguments), configured in
  ``[tool.ruff.lint]`` of pyproject.toml;
* pydoclint (numpy style), configured in ``[tool.pydoclint]``: Parameters match the signature in names and order,
  Returns / Yields / Raises exist when the code returns / yields / raises, also for short docstrings and private functions;
* an AST check that every module, class, function and method (nested ones included) has a docstring.

Run it as a report with ``uv run --extra dev python tests/test_docstrings.py [PATH ...]``: it prints every violation
and, per package, how many there are.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "hcwf"

# pydoclint cannot allow optional signature annotations: with ``arg-type-hints-in-signature = false`` it reports DOC108 for
# every annotated argument. Annotations are optional in HCWF (docstring types are the rendered ones), so DOC108 is ignored.
IGNORED_PYDOCLINT = {"DOC108"}


def python_files(paths: list[Path] | None = None) -> list[Path]:
    """Return the ``hcwf`` source files to check (all of them, or those under `paths`), sorted."""
    roots = paths or [PACKAGE]
    return sorted(f for root in roots for f in ([root] if root.is_file() else root.rglob("*.py"))
                  if "__pycache__" not in f.parts)


def package_of(path: str) -> str:
    """Return the report group of a file path: ``hcwf`` top level or its subpackage."""
    rel = Path(path).resolve().relative_to(ROOT).as_posix() if Path(path).is_absolute() else Path(path).as_posix()
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else "hcwf (top level)"


def _tool_available(module: str) -> bool:
    """Return True when the module can be imported in this environment."""
    return importlib.util.find_spec(module) is not None


# -- ruff ---------------------------------------------------------------------------------------------------------------


def ruff_violations(files: list[Path]) -> list[str]:
    """Return ruff's ``D`` findings of the files as ``path:line:col: CODE message`` lines."""
    cmd = [sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format=concise", "--force-exclude"]
    out = subprocess.run([*cmd, *map(str, files)], cwd=ROOT, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if re.match(r"^\S+:\d+:\d+: ", line)]


# -- pydoclint ----------------------------------------------------------------------------------------------------------


def _nested_functions_without_sections(files: list[Path]) -> set[tuple[str, int]]:
    """Return ``(path, line)`` of nested functions whose docstring has no NumPy section (a summary line is enough there)."""
    found: set[tuple[str, int]] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(ROOT).as_posix()

        def visit(node: ast.AST, in_function: bool) -> None:
            """Walk the tree remembering whether the nearest enclosing scope is a function."""
            for child in ast.iter_child_nodes(node):
                is_function = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                if is_function and in_function:
                    doc = ast.get_docstring(child) or ""
                    if not re.search(r"^\s*-{3,}\s*$", doc, re.MULTILINE):
                        found.add((rel, child.lineno))
                visit(child, is_function or (in_function and not isinstance(child, ast.ClassDef)))

        visit(tree, False)
    return found


def pydoclint_violations(files: list[Path]) -> list[str]:
    """Return pydoclint's findings of the files as ``path:line: CODE: message`` lines (see `IGNORED_PYDOCLINT`)."""
    out = subprocess.run([sys.executable, "-m", "pydoclint.main", "--config", "pyproject.toml",
                          *[f.relative_to(ROOT).as_posix() for f in files]],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(f"pydoclint failed ({out.returncode}): {out.stderr.strip()[:500]}")
    if out.returncode == 1 and not re.search(r"DOC\d+:", out.stdout + out.stderr):
        raise RuntimeError(f"pydoclint failed without a report: {out.stderr.strip()[:500]}")
    nested = _nested_functions_without_sections(files)
    lines = []
    for line in (out.stdout + "\n" + out.stderr).splitlines():   # pydoclint reports on stderr
        match = re.match(r"^(\S+?):(\d+): (DOC\d+): ", line)
        if not match:
            continue
        path, lineno, code = match.group(1), int(match.group(2)), match.group(3)
        if code in IGNORED_PYDOCLINT or (path, lineno) in nested:
            continue
        lines.append(line)
    return lines


# -- AST ----------------------------------------------------------------------------------------------------------------


def _exempt(node: ast.AST) -> bool:
    """Return True for ``__init__`` (the class docstring documents it), property setters and deleters, and overloads."""
    if node.name == "__init__":
        return True
    for dec in node.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in {"setter", "deleter"}:
            return True
        if (isinstance(dec, ast.Name) and dec.id == "overload") or (isinstance(dec, ast.Attribute) and dec.attr == "overload"):
            return True
    return False


def missing_docstrings(path: Path) -> list[str]:
    """Return ``path:line: name`` for every module, class, function and method of the file without a docstring."""
    rel = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing = []
    if ast.get_docstring(tree) is None:
        missing.append(f"{rel}:1: module")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif _exempt(node):
                continue
            else:
                kind = "function"
            if ast.get_docstring(node) is None:
                missing.append(f"{rel}:{node.lineno}: {kind} {node.name}")
    return missing


# -- tests --------------------------------------------------------------------------------------------------------------


def test_ruff_docstring_rules():
    """Ruff's D rules (numpy convention, with D401 and D417) pass on every module."""
    if not _tool_available("ruff"):
        pytest.skip("ruff is not installed: run with `uv run --extra dev`")
    found = ruff_violations(python_files())
    assert not found, "ruff docstring violations:\n" + "\n".join(found)


def test_pydoclint():
    """Pydoclint (numpy style, see [tool.pydoclint]) passes on every module."""
    if not _tool_available("pydoclint"):
        pytest.skip("pydoclint is not installed: run with `uv run --extra dev`")
    found = pydoclint_violations(python_files())
    assert not found, "pydoclint violations:\n" + "\n".join(found)


def test_every_object_has_a_docstring():
    """Every module, class, function and method (nested included) has a docstring."""
    missing = [m for f in python_files() for m in missing_docstrings(f)]
    assert not missing, "missing docstrings:\n" + "\n".join(missing)


def main(argv: list[str]) -> int:
    """Print every docstring violation with counts per package; return the exit code."""
    paths = [Path(a).resolve() for a in argv if not a.startswith("--")] or None
    files = python_files(paths)
    findings: list[tuple[str, str]] = []
    if _tool_available("ruff"):
        findings += [("ruff", line) for line in ruff_violations(files)]
    if _tool_available("pydoclint"):
        findings += [("pydoclint", line) for line in pydoclint_violations(files)]
    findings += [("missing", m) for f in files for m in missing_docstrings(f)]
    for tool, line in findings:
        print(f"[{tool}] {line}")
    counts: Counter = Counter()
    for tool, line in findings:
        counts[(package_of(line.split(":")[0]), tool)] += 1
    print("\nviolations per package and tool")
    for (package, tool), n in sorted(counts.items()):
        print(f"  {package:28s} {tool:10s} {n}")
    print(f"total {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
