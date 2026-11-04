"""The same synthetic files and configs through onset_blending's onsets and features (its own environment) and
hcwf's (hcwf.onset.detect, hcwf.blend.features) give bit-identical observed onsets, forecast onset counts, climatology
and weekly features, legacy tables included (tests/blend/differential_onsets.py)."""

import os
from pathlib import Path

import pytest
from _differential import assert_dumps_identical

pytestmark = pytest.mark.slow  # full run against onset_blending: --run-slow

SCRIPT = Path(__file__).with_name("differential_onsets.py")
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", "/home/hcwf/code/onset_blending/.venv/bin/python"))


@pytest.mark.skipif(not ORACLE_PYTHON.exists(), reason="onset_blending's environment is not available")
def test_hcwf_onsets_and_features_are_onset_blending_bit_for_bit(tmp_path):
    assert_dumps_identical(SCRIPT, ORACLE_PYTHON, tmp_path)
