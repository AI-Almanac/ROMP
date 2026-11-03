"""onset_blending's forecast and momp's agree exactly (tests/blend/differential_forecast.py): the same trained blends
(onset_blending's own file read by momp, and momp's own file) applied to the same live files give identical CSVs, and
every live file the blends cannot take (fewer members, shorter leads, another grid, other units, another issue hour,
an issue date outside the season) is refused with the same reasons. The oracle runs in its own environment when that
exists (exact: the forecast has no scipy-version-dependent number), else in this one."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("differential_forecast.py")
ORACLE_SRC = Path(os.environ.get("ONSET_BLENDING_SRC", "/home/hcwf/code/onset_blending"))
ORACLE_PYTHON = Path(os.environ.get("ONSET_BLENDING_PYTHON", ORACLE_SRC / ".venv/bin/python"))


@pytest.mark.skipif(not ORACLE_SRC.exists(), reason="the onset_blending checkout is not available")
@pytest.mark.parametrize("environment", ["own", "oracle"])
def test_momp_forecasts_exactly_like_onset_blending(tmp_path, environment):
    if environment == "oracle" and not ORACLE_PYTHON.exists():
        pytest.skip("onset_blending's environment is not available")
    python = str(ORACLE_PYTHON) if environment == "oracle" else sys.executable
    done = subprocess.run([sys.executable, str(SCRIPT), "run", str(tmp_path / "work"), python],
                          capture_output=True, text=True,
                          env={**os.environ, "MOMP_CACHE_DIR": str(tmp_path / "cache")})

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip() == "identical"
