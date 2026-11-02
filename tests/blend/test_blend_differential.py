"""The same seeded inputs through onset_blending's blending methods (its own environment) and momp.blend's give
bit-identical coefficients, CV predictions, forest trees and MME weights (tests/blend/differential_blend.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("differential_blend.py")
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", "/home/hcwf/code/onset_blending/.venv/bin/python"))


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_momp_blend_is_onset_blending_bit_for_bit(tmp_path):
    for python, package in ((sys.executable, "momp"), (str(ORACLE_PYTHON), "onset_blending")):
        subprocess.run([python, str(SCRIPT), "dump", package, str(tmp_path / f"{package}.pkl")], check=True,
                       capture_output=True, text=True)

    compared = subprocess.run([sys.executable, str(SCRIPT), "compare", str(tmp_path / "momp.pkl"),
                               str(tmp_path / "onset_blending.pkl")], capture_output=True, text=True)

    assert compared.stdout.strip() == "identical", compared.stdout
