"""The same seeded inputs through onset_blending's blending methods (its own environment) and hcwf.blend's give
bit-identical coefficients, CV predictions, forest trees and MME weights (tests/blend/differential_blend.py)."""

import os
from pathlib import Path

import pytest
from _differential import assert_dumps_identical

pytestmark = pytest.mark.slow  # full run against onset_blending: --run-slow

SCRIPT = Path(__file__).with_name("differential_blend.py")
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", "/home/hcwf/code/onset_blending/.venv/bin/python"))


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_hcwf_blend_is_onset_blending_bit_for_bit(tmp_path):
    assert_dumps_identical(SCRIPT, ORACLE_PYTHON, tmp_path)
