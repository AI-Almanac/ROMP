"""Reuse onset_blending's outputs between verification rounds: the oracle is deterministic, hcwf is what is verified.

onset_blending (frozen at one commit) writes the same outputs for the same inputs every time, yet each round used to
recompute them (the differential tests, tests/blend/realdata/run_parity.py). `reuse` keeps each oracle result once in
the ``oracle`` kind of the shared hcwf cache (`hcwf.cache.Store`; listed and deleted by ``hcwf-run cache info|clean
--kind oracle``) and runs the oracle again only when the key changes. The hcwf side is never cached here: it always runs
cold, in its own temporary cache folder.

The key of an entry is made of these parts (any change is a miss; `hcwf.cache.SCHEMA` versions this logic):

* ``commit``: the git commit of the onset_blending checkout that the oracle interpreter imports. A tree with
  uncommitted changes (or untracked files) is *dirty* and disables reuse altogether (nothing is read or stored).
* ``source``: a hash of the contents of the imported ``onset_blending`` package files, so a package that is installed
  rather than read from the checkout is still keyed by the code that runs.
* ``environment``: the interpreter path and the hash of its Python version and installed distributions and their
  versions (numpy and scipy versions change the last bits of some numbers, for example the Pietra index).
* ``driver``: the hash of the script that runs the oracle and of every ``tests`` file it imports (they generate the
  synthetic inputs and configs), or for the real-data driver the oracle command line and the config text.
* ``inputs``: the identity (path, size, mtime) of every input data file of a real-data run, or the dump arguments.

Input files are keyed by path, size and mtime (`hcwf.cache.file_identity`) rather than by contents: contents of the
real-data NetCDF files are gigabytes and hashing them costs more than it saves, while a file that changes in place
without changing its size or its modification time does not happen in practice; a file that is only touched is
recomputed, which is the safe direction. Whenever the key cannot be established (dirty tree, no git, a package
imported from elsewhere than the checkout, an unreadable input) nothing is reused.

Where: ``$HCWF_ORACLE_CACHE_DIR`` if set, else the shared root (``$HCWF_CACHE_DIR``, default ``~/.cache/hcwf``), read
once when this module is imported, so that the per-test cache folder of tests/conftest.py (set later, per test) is not
the one used. Force a recompute (and overwrite the entry) with ``HCWF_ORACLE_REFRESH=1``, or delete the entries with
``hcwf-run cache clean --kind oracle``. Writes are atomic (`hcwf.cache.atomic_write`): parallel workers that compute
one key at once each write the same content and a reader sees a whole entry or none.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hcwf.cache import Codec, Store, file_identity, global_root  # noqa: E402

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent


def oracle_root() -> Path:
    """Return the folder of the oracle cache: ``$HCWF_ORACLE_CACHE_DIR``, else the shared hcwf cache root.

    Returns
    -------
    pathlib.Path
        The cache root (the entries are under ``<root>/oracle``).
    """
    return Path(os.environ["HCWF_ORACLE_CACHE_DIR"]) if os.environ.get("HCWF_ORACLE_CACHE_DIR") else global_root()


SHARED_ROOT = oracle_root()
"""The oracle cache root, fixed at import (before a test's own HCWF_CACHE_DIR is set)."""


def _write_entry(path: Path, value) -> None:
    """Write an entry payload: a folder with the oracle's output (``dump``) and its metadata (``meta.json``).

    Parameters
    ----------
    path : pathlib.Path
        The folder to create.
    value : tuple of (pathlib.Path, dict)
        The oracle output (a file or a folder) and a JSON-able dict about how it was made.
    """
    dump, meta = value
    path.mkdir()
    if dump.is_dir():
        shutil.copytree(dump, path / "dump", symlinks=True)
    else:
        shutil.copy2(dump, path / "dump")
    (path / "meta.json").write_text(json.dumps(meta))


def _read_entry(path: Path):
    """Read an entry payload without copying it.

    Parameters
    ----------
    path : pathlib.Path
        The payload folder.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The stored output (read-only: compare it, never write into it) and its metadata.
    """
    return path / "dump", json.loads((path / "meta.json").read_text())


CODEC = Codec("payload", _read_entry, _write_entry)
"""An oracle output folder or file, with metadata."""


def _hash_files(paths) -> str:
    """Hash the names and contents of files.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        The files.

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    h = hashlib.sha256()
    for path in sorted(set(paths)):
        h.update(str(path).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _imported_files(script: Path) -> set[Path]:
    """Return a script and the ``tests`` files it imports, transitively (by reading the import statements).

    Parameters
    ----------
    script : pathlib.Path
        The script to start from.

    Returns
    -------
    set of pathlib.Path
        The script and every module file under ``tests`` that it (or they) import, including imports inside functions.
    """
    seen: set[Path] = set()
    todo = [Path(script).resolve()]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                if name == "_oracle_cache":
                    continue  # how outputs are stored (hcwf.cache.SCHEMA) is not an input of the oracle
                parts = name.split(".")
                bases = [ROOT / Path(*parts)] if parts[0] == "tests" else [HERE / Path(*parts), TESTS / Path(*parts)]
                for base in bases:
                    for found in (base.with_suffix(".py"), base / "__init__.py"):
                        if found.is_file():
                            todo.append(found.resolve())
    return seen


def driver_hash(script) -> str:
    """Return the hash of a differential script and of every tests file it imports.

    Parameters
    ----------
    script : path-like
        The script that runs the oracle (tests/blend/differential_*.py).

    Returns
    -------
    str
        A SHA-256 hex digest of the names and contents.
    """
    return _hash_files(_imported_files(Path(script)))


_PROBE = """
import hashlib, importlib.metadata as md, json, os, sys
sys.path[0] = {path0!r}
if {src!r}:
    sys.path.insert(0, {src!r})
import onset_blending
pkg = os.path.dirname(os.path.abspath(onset_blending.__file__))
h = hashlib.sha256()
for dirpath, dirnames, files in sorted(os.walk(pkg)):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for name in sorted(files):
        if name.endswith(".py"):
            full = os.path.join(dirpath, name)
            h.update(os.path.relpath(full, pkg).encode() + b"\\0" + open(full, "rb").read() + b"\\0")
dists = sorted((d.metadata["Name"] or "", d.version) for d in md.distributions())
print(json.dumps({{"package": pkg, "source": h.hexdigest(), "python": sys.version, "dists": dists}}))
"""

_STATES: dict = {}


def _git(src: Path, *args) -> str | None:
    """Return the stdout of a git command in a checkout, or None when it fails.

    Parameters
    ----------
    src : pathlib.Path
        The checkout.
    *args : str
        The git arguments.

    Returns
    -------
    str or None
        The stripped stdout, or None on a failure.
    """
    done = subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def oracle_state(src, python, *, src_first: bool, path0, env=None) -> dict | None:
    """Return the key parts of the oracle (commit, source, environment), or None when reuse is not safe.

    Parameters
    ----------
    src : path-like
        The onset_blending checkout.
    python : path-like
        The interpreter that runs the oracle.
    src_first : bool
        Whether the oracle script puts ``src`` first on ``sys.path`` (the differential scripts do; the ``-m
        onset_blending`` command of run_parity does not, it uses what the interpreter has installed).
    path0 : path-like
        The directory that is first on ``sys.path`` in the oracle run (the script's folder, or the working folder
        of a ``-m`` run).
    env : dict, optional
        The environment of the oracle run; default this process's.

    Returns
    -------
    dict or None
        The parts, or None when the checkout is dirty, is not a git checkout, or the interpreter imports the package
        from outside it (recompute then).
    """
    src, python = Path(src), Path(python)
    memo_key = (str(src), str(python), src_first, str(path0), tuple(sorted((env or {}).items())))
    if memo_key in _STATES:
        return _STATES[memo_key]
    state = None
    commit = _git(src, "rev-parse", "HEAD")
    dirty = _git(src, "status", "--porcelain", "--untracked-files=normal")
    if commit and dirty == "":
        with tempfile.TemporaryDirectory() as scratch:
            probe = _PROBE.format(path0=str(path0), src=str(src) if src_first else "")
            done = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, cwd=scratch,
                                  env=env if env is None else {**os.environ, **env})
        if done.returncode == 0:
            info = json.loads(done.stdout.strip().splitlines()[-1])
            if Path(info["package"]).resolve().is_relative_to(src.resolve()):
                state = {"commit": commit, "source": info["source"],
                         "environment": {"python": str(python), "version": info["python"], "dists": info["dists"]}}
    _STATES[memo_key] = state
    return state


def inputs_identity(paths) -> list:
    """Return the identity of input files and folders: each file under a folder, by path, size and mtime.

    Parameters
    ----------
    paths : iterable of path-like
        Input files or folders.

    Returns
    -------
    list
        ``[path, size, mtime_ns]`` of every file, sorted; a missing path stays in the list with ``None`` (so it
        differs from a present one).
    """
    out = []
    for path in sorted({str(Path(p)) for p in paths}):
        if Path(path).is_dir():
            out += [file_identity(f) for f in sorted(Path(path).rglob("*")) if f.is_file()]
        else:
            out.append(file_identity(path))
    return out


def reuse(parts: dict, produce, target, *, src, python, src_first: bool, path0, env=None, label: str = "oracle",
          reusable: bool = True):
    """Return the oracle's output for these key parts: the stored one, else compute it, store it and return it.

    Parameters
    ----------
    parts : dict
        The call-specific key parts (``driver``, ``inputs``, ...); the commit, source and environment are added.
    produce : callable
        ``produce(target)`` runs the oracle and writes its output at ``target`` (a path that does not exist yet).
    target : pathlib.Path
        Where a fresh output goes (not used on a reuse).
    src : path-like
        The onset_blending checkout.
    python : path-like
        The interpreter that runs the oracle.
    src_first : bool
        See `oracle_state`.
    path0 : path-like
        See `oracle_state`.
    env : dict, optional
        See `oracle_state`.
    label : str, default "oracle"
        Names the entry in the line printed to stderr.
    reusable : bool, default True
        False when the caller could not establish all the inputs of the key: the oracle is run, nothing is read or
        stored.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The output (a stored one is read-only) and ``{"reused": bool, "seconds": float, "fingerprint": str or None}``
        (``seconds``: the time the oracle took, now or when stored).
    """
    target = Path(target)
    state = oracle_state(src, python, src_first=src_first, path0=path0, env=env) if reusable else None
    if state is None:
        print(f"[oracle cache] {label}: not reusable (dirty tree, no git, or imported from elsewhere): computing",
              file=sys.stderr)
        started = time.time()
        produce(target)
        return target, {"reused": False, "seconds": round(time.time() - started, 2), "fingerprint": None}
    store = Store("oracle", SHARED_ROOT, codec=CODEC, label="oracle", memory=False)
    key = {**state, **parts}
    fingerprint = store.fingerprint(key)
    refresh = os.environ.get("HCWF_ORACLE_REFRESH") == "1"
    if not refresh:
        hit = store.get(key, quiet=True)
        if hit is not None:
            dump, meta = hit
            print(f"[oracle cache] {label}: reused {fingerprint} (computed in {meta['seconds']} s)", file=sys.stderr)
            return dump, {"reused": True, "seconds": meta["seconds"], "fingerprint": fingerprint}
    else:
        shutil.rmtree(store.entry(key), ignore_errors=True)
    started = time.time()
    produce(target)
    seconds = round(time.time() - started, 2)
    store.put(key, (target, {"seconds": seconds, "label": label}))
    print(f"[oracle cache] {label}: computed in {seconds} s, stored {fingerprint}", file=sys.stderr)
    return target, {"reused": False, "seconds": seconds, "fingerprint": fingerprint}
