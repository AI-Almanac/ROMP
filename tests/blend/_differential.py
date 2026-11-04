"""The step shared by the onset_blending differential tests: dump the same seeded inputs with both packages (each in
its own interpreter), then let the script compare the two dumps. Only the script, the dump arguments and the
interpreters' working directory and environment differ between the tests."""
import subprocess
import sys


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


def compare(script, first, second):
    """The stripped stdout of ``script compare <first> <second>``: it starts with ``identical`` when nothing differs."""
    done = subprocess.run([sys.executable, str(script), "compare", str(first), str(second)], capture_output=True,
                          text=True)
    return (done.stdout + done.stderr).strip() if done.returncode else done.stdout.strip()


def assert_dumps_identical(script, oracle_python, directory, *, extra=(), hcwf_kwargs=None, oracle_kwargs=None):
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
    """
    dump(script, sys.executable, "hcwf", directory / "hcwf.pkl", *extra, **(hcwf_kwargs or {}))
    dump(script, oracle_python, "onset_blending", directory / "onset_blending.pkl", *extra, **(oracle_kwargs or {}))
    result = compare(script, directory / "hcwf.pkl", directory / "onset_blending.pkl")
    assert result.startswith("identical"), result
