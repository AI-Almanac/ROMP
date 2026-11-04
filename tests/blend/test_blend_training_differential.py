"""onset_blending's `train(config).save(dir)` and hcwf's `train(*converted(config)).save(dir)` write the same artifacts
on the same synthetic files and configs (tests/blend/differential_training.py): CSV and JSON byte for byte, NetCDF
variable for variable. The oracle is imported from its checkout and run in HCWF's environment (exact), and in its own
environment when that exists (identical but for the last bit of the Pietra index, which depends on the scipy version).
Each onset_blending dump is reused from the oracle cache while its key is unchanged (tests/blend/_oracle_cache.py); the
hcwf dump is always computed.
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


def oracle(python, target):
    return differential.oracle_dump(SCRIPT, python, target, "2", src=ORACLE_SRC)


def compared(first, second):
    return differential.compare(SCRIPT, first, second)


@pytest.fixture(scope="module")
def hcwf_dump(tmp_path_factory):
    target = tmp_path_factory.mktemp("training") / "hcwf"
    cache = tmp_path_factory.mktemp("hcwf_cache")  # hcwf's side always runs cold: not the shared cache folder
    differential.dump(SCRIPT, sys.executable, "hcwf", target, "2", env={**os.environ, "HCWF_CACHE_DIR": str(cache)})
    return target


@pytest.mark.skipif(not ORACLE_SRC.exists(), reason="the onset_blending checkout is not available")
def test_hcwf_training_writes_onset_blendings_artifacts_exactly(hcwf_dump, tmp_path):
    reference = oracle(sys.executable, tmp_path / "onset_blending")

    assert compared(hcwf_dump, reference) == "identical"


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_hcwf_training_matches_onset_blendings_own_environment(hcwf_dump, tmp_path):
    reference = oracle(ORACLE_PYTHON, tmp_path / "onset_blending")

    assert compared(hcwf_dump, reference) == "identical"
