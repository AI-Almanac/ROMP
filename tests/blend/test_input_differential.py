"""hcwf's input path against onset_blending's own readers on the same synthetic files (see differential_input.py).
The onset_blending dump is reused from the oracle cache while its key is unchanged (tests/blend/_oracle_cache.py)."""
from pathlib import Path

import pytest
from _differential import assert_dumps_identical

pytestmark = pytest.mark.slow  # full run against onset_blending: --run-slow

SCRIPT = Path(__file__).with_name("differential_input.py")
ROOT = Path(__file__).resolve().parents[2]
BLENDING = Path("/home/hcwf/code/onset_blending")


@pytest.mark.skipif(not (BLENDING / ".venv/bin/python").exists(), reason="the onset_blending environment is not there")
def test_the_arrays_read_are_bit_identical(tmp_path):
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
    assert_dumps_identical(SCRIPT, BLENDING / ".venv/bin/python", tmp_path,
                           hcwf_kwargs={"cwd": ROOT, "env": env}, oracle_kwargs={"cwd": BLENDING}, src_first=False)
