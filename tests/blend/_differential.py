"""The step shared by the onset_blending differential tests: dump the same seeded inputs with both packages (each in
its own interpreter), then let the script compare the two dumps. The onset_blending dump is computed once and then
reused while its key (commit, environment, script, inputs) is unchanged (tests/blend/_oracle_cache.py); the hcwf dump
is always computed. Only the script, the dump arguments and the interpreters' working directory and environment differ
between the tests."""
import subprocess
import sys
from pathlib import Path

import _oracle_cache

ORACLE_SRC = Path("/home/hcwf/code/onset_blending")
"""The onset_blending checkout the differential scripts import."""


def dump(script, python, package, target, *extra, **run_kwargs):
    """Run ``script dump <package> <target> [extra...]`` with ``python`` and fail with its stderr on a bad exit.

    Parameters
    ----------
    script : path-like
        The differential script (tests/blend/differential_*.py).
    python : path-like
        The interpreter to run it with.
    package : str
        ``"hcwf"`` or ``"onset_blending"``: the package the script dumps.
    target : path-like
        Where the script writes the dump.
    *extra : str
        More arguments for the script.
    **run_kwargs
        Passed to ``subprocess.run`` (``cwd``, ``env``).
    """
    done = subprocess.run([str(python), str(script), "dump", package, str(target), *extra], capture_output=True,
                          text=True, **run_kwargs)
    assert done.returncode == 0, done.stderr[-3000:]


def oracle_dump(script, python, target, *extra, src=ORACLE_SRC, src_first=True, **run_kwargs):
    """Return the onset_blending dump: the stored one when its key is unchanged, else run it (and store it).

    The key is in tests/blend/_oracle_cache.py: the oracle's commit (a dirty checkout is never reused), source and
    interpreter environment, the hash of ``script`` and the tests files it imports, and the arguments. Set
    ``HCWF_ORACLE_REFRESH=1`` to recompute.

    Parameters
    ----------
    script : path-like
        The differential script.
    python : path-like
        The onset_blending interpreter.
    target : pathlib.Path
        Where a fresh dump is written (must not exist).
    *extra : str
        More arguments for the script.
    src : path-like, default `ORACLE_SRC`
        The onset_blending checkout.
    src_first : bool, default True
        Whether the script puts the checkout first on ``sys.path`` (False: the interpreter's installed package).
    **run_kwargs
        Passed to ``subprocess.run`` (``cwd``, ``env``).

    Returns
    -------
    pathlib.Path
        The dump to compare (read-only).
    """
    parts = {"driver": _oracle_cache.driver_hash(script), "script": Path(script).name, "extra": list(extra),
             "cwd": str(run_kwargs.get("cwd"))}
    path, _ = _oracle_cache.reuse(
        parts, lambda out: dump(script, python, "onset_blending", out, *extra, **run_kwargs), target, src=src,
        python=python, src_first=src_first, path0=Path(script).resolve().parent, env=run_kwargs.get("env"),
        label=f"{Path(script).stem} {' '.join(extra)}".strip())
    return path


def compare(script, first, second):
    """The stripped stdout of ``script compare <first> <second>``: it starts with ``identical`` when nothing differs."""
    done = subprocess.run([sys.executable, str(script), "compare", str(first), str(second)], capture_output=True,
                          text=True)
    return (done.stdout + done.stderr).strip() if done.returncode else done.stdout.strip()


def assert_dumps_identical(script, oracle_python, directory, *, extra=(), hcwf_kwargs=None, oracle_kwargs=None,
                           src_first=True):
    """Dump with hcwf (this interpreter) and onset_blending (``oracle_python``) into ``directory``, then compare.

    Parameters
    ----------
    script : path-like
        The differential script.
    oracle_python : path-like
        The onset_blending interpreter.
    directory : pathlib.Path
        Where the two dumps go (``hcwf.pkl`` and ``onset_blending.pkl``).
    extra : tuple of str
        More arguments for both dumps.
    hcwf_kwargs, oracle_kwargs : dict, optional
        ``subprocess.run`` options (``cwd``, ``env``) of each dump.
    src_first : bool, default True
        Whether the script puts the onset_blending checkout first on ``sys.path`` (see `oracle_dump`).
    """
    dump(script, sys.executable, "hcwf", directory / "hcwf.pkl", *extra, **(hcwf_kwargs or {}))
    oracle = oracle_dump(script, oracle_python, directory / "onset_blending.pkl", *extra, src_first=src_first,
                         **(oracle_kwargs or {}))
    result = compare(script, directory / "hcwf.pkl", oracle)
    assert result.startswith("identical"), result
