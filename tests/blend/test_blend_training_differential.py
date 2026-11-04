"""onset_blending's `train(config).save(dir)` and hcwf's `train(*converted(config)).save(dir)` write the same artifacts
on the same synthetic files and configs (tests/blend/differential_training.py): CSV and JSON byte for byte, NetCDF
variable for variable. The oracle is imported from its checkout and run in HCWF's environment (exact), and in its own
environment when that exists (identical but for the last bit of the Pietra index, which depends on the scipy version).
"""

import os
import sys
from pathlib import Path

import _differential as differential
import pytest

pytestmark = pytest.mark.slow  # full run against onset_blending: --run-slow

SCRIPT = Path(__file__).with_name("differential_training.py")
ORACLE_SRC = Path(os.environ.get("ONSET_BLENDING_SRC", "/home/hcwf/code/onset_blending"))
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", ORACLE_SRC / ".venv/bin/python"))


def dump(python, package, target):
    differential.dump(SCRIPT, python, package, target, "2")


def compared(first, second):
    return differential.compare(SCRIPT, first, second)


@pytest.fixture(scope="module")
def hcwf_dump(tmp_path_factory):
    target = tmp_path_factory.mktemp("training") / "hcwf"
    dump(sys.executable, "hcwf", target)
    return target


@pytest.mark.skipif(not ORACLE_SRC.exists(), reason="the onset_blending checkout is not available")
def test_hcwf_training_writes_onset_blendings_artifacts_exactly(hcwf_dump, tmp_path):
    dump(sys.executable, "onset_blending", tmp_path / "onset_blending")

    assert compared(hcwf_dump, tmp_path / "onset_blending") == "identical"


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_hcwf_training_matches_onset_blendings_own_environment(hcwf_dump, tmp_path):
    dump(ORACLE_PYTHON, "onset_blending", tmp_path / "onset_blending")

    assert compared(hcwf_dump, tmp_path / "onset_blending") == "identical"
