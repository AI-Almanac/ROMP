"""onset_blending's `train(config).save(dir)` and momp's `train(*converted(config)).save(dir)` write the same artifacts
on the same synthetic files and configs (tests/blend/differential_training.py): CSV and JSON byte for byte, NetCDF
variable for variable. The oracle is imported from its checkout and run in ROMP's environment (exact), and in its own
environment when that exists (identical but for the last bit of the Pietra index, which depends on the scipy version).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("differential_training.py")
ORACLE_SRC = Path(os.environ.get("ONSET_BLENDING_SRC", "/home/hcwf/code/onset_blending"))
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", ORACLE_SRC / ".venv/bin/python"))


def dump(python, package, target):
    subprocess.run([str(python), str(SCRIPT), "dump", package, str(target), "2"], check=True, capture_output=True,
                   text=True)


def compared(first, second):
    return subprocess.run([sys.executable, str(SCRIPT), "compare", str(first), str(second)], capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture(scope="module")
def momp_dump(tmp_path_factory):
    target = tmp_path_factory.mktemp("training") / "momp"
    dump(sys.executable, "momp", target)
    return target


@pytest.mark.skipif(not ORACLE_SRC.exists(), reason="the onset_blending checkout is not available")
def test_momp_training_writes_onset_blendings_artifacts_exactly(momp_dump, tmp_path):
    dump(sys.executable, "onset_blending", tmp_path / "onset_blending")

    assert compared(momp_dump, tmp_path / "onset_blending") == "identical"


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_momp_training_matches_onset_blendings_own_environment(momp_dump, tmp_path):
    dump(ORACLE_PYTHON, "onset_blending", tmp_path / "onset_blending")

    assert compared(momp_dump, tmp_path / "onset_blending") == "identical"
