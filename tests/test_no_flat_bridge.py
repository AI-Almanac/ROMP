"""The end of the flat bridge (step S11 of docs/review/runconfig_design.md): nothing imports the transitional adapter,
no reader, detector, scorer, figure or app takes `**kwargs`, and only the config layer, the run-folder and resolved-config
writers, the `--check` report and `RunResult` read the flat record of a `RunConfig`."""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "hcwf"

KWARGS_AREAS = ("io", "stats", "metrics", "graphics", "app")
KWARGS_FILES = ("export.py",)
KWARGS_ALLOWED = {("graphics/field.py", "draw_field"): "style options passed on to matplotlib",
                  ("stats/parallel.py", "map_items"): "options of start_items"}
"""The functions of those areas that may take a `**` parameter, and why (plot-style and worker-pool options)."""

VALUES_READERS = {"api.py", "lib/run_folder.py", "lib/resolved.py", "lib/check.py"}
"""Outside hcwf/config/, the files that read `RunConfig.values` (the flat record): `RunResult.config` (api.py), the run
folder and the resolved config it writes, and the `--check` report."""
RUN_NAMES = {"rc", "run_config", "run"}
"""The names a RunConfig has in hcwf."""


def _python_files(root):
    return sorted(p for p in root.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts)


def test_the_bridge_module_is_gone():
    assert not (PACKAGE / "config" / "flat_bridge.py").exists()


@pytest.mark.parametrize("path", _python_files(PACKAGE) + _python_files(REPO / "tests") + _python_files(REPO / "scripts"),
                         ids=lambda p: p.relative_to(REPO).as_posix())
def test_nothing_imports_the_flat_bridge(path):
    if path == Path(__file__).resolve():
        return
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        assert not any("flat_bridge" in name for name in names), f"{path} imports the flat bridge"
    assert "R2-bridge" not in path.read_text() or path.parent.name == "tests", f"{path} still has an R2-bridge wrapper"


def _kwargs_files():
    for area in KWARGS_AREAS:
        yield from _python_files(PACKAGE / area)
    for name in KWARGS_FILES:
        yield PACKAGE / name


@pytest.mark.parametrize("path", list(_kwargs_files()), ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_no_function_takes_double_star_kwargs(path):
    rel = path.relative_to(PACKAGE).as_posix()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node.args.kwarg is not None:
            name = getattr(node, "name", "<lambda>")
            assert (rel, name) in KWARGS_ALLOWED, f"{rel}:{node.lineno} {name}(**{node.args.kwarg.arg})"


def test_the_allowlist_names_functions_that_exist():
    for (rel, name), _ in KWARGS_ALLOWED.items():
        found = [n for n in ast.walk(ast.parse((PACKAGE / rel).read_text()))
                 if isinstance(n, ast.FunctionDef) and n.name == name and n.args.kwarg is not None]
        assert found, f"{rel} has no {name}(**...)"


def _reads_values(tree):
    """The line numbers of `<run>.values` and of any `._values` read in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        owner = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
        if node.attr == "values" and owner in RUN_NAMES or node.attr == "_values":
            yield node.lineno


@pytest.mark.parametrize("path", [p for p in _python_files(PACKAGE) if p.relative_to(PACKAGE).parts[0] != "config"],
                         ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_only_the_record_writers_read_the_flat_values(path):
    rel = path.relative_to(PACKAGE).as_posix()
    lines = list(_reads_values(ast.parse(path.read_text())))
    if rel in VALUES_READERS:
        return
    assert not lines, f"{rel} reads RunConfig.values at line(s) {lines}"
