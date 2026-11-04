"""Guard for the docstring conventions of docs/contributing_docstrings.md.

Three checks run over every module of ``hcwf`` that is not in `PENDING`:

* ruff's pydocstyle rules (``D``, numpy convention, plus D401 imperative mood and D417 documented arguments), configured in
  ``[tool.ruff.lint]`` of pyproject.toml;
* pydoclint (numpy style), configured in ``[tool.pydoclint]``: Parameters match the signature in names and order,
  Returns / Yields / Raises exist when the code returns / yields / raises, also for short docstrings and private functions;
* an AST check that every module, class, function and method (nested ones included) has a docstring.

`PENDING` lists the packages whose docstrings have not been converted yet; they are skipped. A package agent removes its
lines from `PENDING` and from ``[tool.ruff.lint.per-file-ignores]`` in pyproject.toml (a test keeps the two equal).

Run it as a report with ``uv run --extra dev python tests/test_docstrings.py [--all] [PATH ...]``: it prints every
violation (``--all`` includes the pending packages) and, per package, how many there are.
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

# Packages not converted to the docstring conventions yet, one per line, alphabetic. Remove a line when its package passes.
PENDING = {
}

# pydoclint cannot allow optional signature annotations: with ``arg-type-hints-in-signature = false`` it reports DOC108 for
# every annotated argument. Annotations are optional in HCWF (docstring types are the rendered ones), so DOC108 is ignored.
IGNORED_PYDOCLINT = {"DOC108"}


def python_files(include_pending: bool = False, paths: list[Path] | None = None) -> list[Path]:
    """Return the ``hcwf`` source files to check, sorted; pending packages only when asked."""
    roots = paths or [PACKAGE]
    files = sorted(f for root in roots for f in ([root] if root.is_file() else root.rglob("*.py"))
                   if "__pycache__" not in f.parts)
    if include_pending:
        return files
    return [f for f in files if not is_pending(f)]


def is_pending(path: Path) -> bool:
    """Return True when the file lies inside a package of `PENDING`."""
    rel = path.resolve().relative_to(ROOT).as_posix()
    return any(rel == p or rel.startswith(p + "/") for p in PENDING)


def package_of(path: str) -> str:
    """Return the report group of a file path: its pending package, else ``hcwf`` top level or its subpackage."""
    rel = Path(path).resolve().relative_to(ROOT).as_posix() if Path(path).is_absolute() else Path(path).as_posix()
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else "hcwf (top level)"


def _tool_available(module: str) -> bool:
    """Return True when the module can be imported in this environment."""
    return importlib.util.find_spec(module) is not None


# -- ruff ---------------------------------------------------------------------------------------------------------------


def ruff_violations(files: list[Path], honor_pending_ignores: bool = True) -> list[str]:
    """Return ruff's ``D`` findings of the files as ``path:line:col: CODE message`` lines."""
    cmd = [sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format=concise", "--force-exclude"]
    if not honor_pending_ignores:
        cmd += ["--config", "lint.per-file-ignores = {}"]
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


def test_pending_packages_exist():
    """Every PENDING entry is an existing package directory (stale entries are removed, not left behind)."""
    for entry in PENDING:
        assert (ROOT / entry).is_dir(), f"PENDING lists {entry}, which does not exist"


def test_pending_matches_ruff_ignores():
    """PENDING equals the packages ignored for the D rules in ``[tool.ruff.lint.per-file-ignores]``."""
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = config["tool"]["ruff"]["lint"].get("per-file-ignores", {})
    listed = {pattern.removesuffix("/**") for pattern, rules in ignores.items() if rules == ["D"]}
    assert listed == PENDING, (
        "PENDING (tests/test_docstrings.py) and per-file-ignores (pyproject.toml) differ: "
        f"only in PENDING {sorted(PENDING - listed)}, only in pyproject {sorted(listed - PENDING)}")


def test_ruff_docstring_rules():
    """Ruff's D rules (numpy convention, with D401 and D417) pass on every converted module."""
    if not _tool_available("ruff"):
        pytest.skip("ruff is not installed: run with `uv run --extra dev`")
    found = ruff_violations(python_files())
    assert not found, "ruff docstring violations:\n" + "\n".join(found)


def test_pydoclint():
    """Pydoclint (numpy style, see [tool.pydoclint]) passes on every converted module."""
    if not _tool_available("pydoclint"):
        pytest.skip("pydoclint is not installed: run with `uv run --extra dev`")
    found = pydoclint_violations(python_files())
    assert not found, "pydoclint violations:\n" + "\n".join(found)


def test_every_object_has_a_docstring():
    """Every module, class, function and method (nested included) of a converted module has a docstring."""
    missing = [m for f in python_files() for m in missing_docstrings(f)]
    assert not missing, "missing docstrings:\n" + "\n".join(missing)


def main(argv: list[str]) -> int:
    """Print every docstring violation (``--all`` includes pending packages) with counts per package; return the exit code."""
    include_pending = "--all" in argv
    paths = [Path(a).resolve() for a in argv if not a.startswith("--")] or None
    files = python_files(include_pending, paths)
    findings: list[tuple[str, str]] = []
    if _tool_available("ruff"):
        findings += [("ruff", line) for line in ruff_violations(files, honor_pending_ignores=not include_pending)]
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
