"""hcwf's input path against onset_blending's own readers on the same synthetic files (see differential_input.py)."""
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("differential_input.py")
ROOT = Path(__file__).resolve().parents[2]
BLENDING = Path("/home/hcwf/code/onset_blending")


@pytest.mark.skipif(not (BLENDING / ".venv/bin/python").exists(), reason="the onset_blending environment is not there")
def test_the_arrays_read_are_bit_identical(tmp_path):
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    for package, python, cwd in (("hcwf", sys.executable, ROOT), ("onset_blending", BLENDING / ".venv/bin/python", BLENDING)):
        done = subprocess.run([str(python), str(SCRIPT), "dump", package, str(tmp_path / f"{package}.pkl")], cwd=cwd,
                              env=env if package == "hcwf" else None, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-2000:]
    done = subprocess.run([sys.executable, str(SCRIPT), "compare", str(tmp_path / "hcwf.pkl"),
                           str(tmp_path / "onset_blending.pkl")], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
