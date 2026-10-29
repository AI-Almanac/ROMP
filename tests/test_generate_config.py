"""generate_config.py turns ROMP_* env vars into a config.in that ROMP exec()s."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_config.py"
PAYLOAD = "__import__('pathlib').Path({marker!r}).touch()"


def _generate(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    """Run generate_config.py in a subprocess with the required ROMP_* env vars plus overrides."""
    required = {
        "ROMP_OBS_DIR": str(tmp_path / "obs"),
        "ROMP_MODEL_DIR": str(tmp_path / "model"),
        "ROMP_MODEL_NAME": "fuxi",
        "ROMP_DIR_OUT": str(tmp_path / "out"),
        "ROMP_DIR_FIG": str(tmp_path / "fig"),
        "ROMP_CONFIG_PATH": str(tmp_path / "config.in"),
    }
    clean = {k: v for k, v in os.environ.items() if not k.startswith("ROMP_")}
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**clean, **required, **env},
        capture_output=True,
        text=True,
    )


def _string_values(config: dict) -> set:
    """Collect every string found in the config values, including inside tuples."""
    values = (v if isinstance(v, tuple) else (v,) for v in config.values())
    return {item for group in values for item in group if isinstance(item, str)}


def _exec_config(tmp_path: Path) -> dict:
    """Exec the generated config.in and return the resulting namespace."""
    namespace: dict = {}
    exec((tmp_path / "config.in").read_text(), {}, namespace)
    return namespace


def test_defaults_produce_the_demo_config(tmp_path):
    """With only required variables set, the generated config holds the demo defaults."""
    assert _generate(tmp_path).returncode == 0

    config = _exec_config(tmp_path)

    assert config["model_list"] == ("fuxi",)
    assert config["start_date"] == (2019, 5, 1)
    assert config["init_days"] == (0, 3)
    assert config["members"] == "All"
    assert config["probabilistic"] is False
    assert config["nc_mask"] is None
    assert config["wet_threshold"] == 20


def test_job_settings_are_parsed_into_typed_values(tmp_path):
    """Environment settings are parsed into typed Python values in the generated config."""
    result = _generate(
        tmp_path,
        ROMP_INIT_DAYS="2, 5",
        ROMP_MEMBERS="0,1,2",
        ROMP_WET_THRESHOLD="25.0",
        ROMP_PROBABILISTIC="True",
        ROMP_NC_MASK="/data/masks/ethiopia mask.nc",
        ROMP_DATE_FILTER_YEAR="2020",
    )
    assert result.returncode == 0

    config = _exec_config(tmp_path)

    assert config["init_days"] == (2, 5)
    assert config["members"] == (0, 1, 2)
    assert config["wet_threshold"] == 25.0
    assert config["probabilistic"] is True
    assert config["nc_mask"] == "/data/masks/ethiopia mask.nc"
    assert config["date_filter_year"] == 2020


@pytest.mark.parametrize(
    "name",
    ["ROMP_OBS_VAR", "ROMP_MODEL_NAME", "ROMP_FILE_PATTERN", "ROMP_NC_MASK", "ROMP_REGION"],
)
def test_text_settings_stay_inert_strings(tmp_path, name):
    """Text settings containing code-like payloads are written as plain strings and never executed."""
    marker = tmp_path / "executed"
    value = f'x"\n{PAYLOAD.format(marker=str(marker))}\n"'
    assert _generate(tmp_path, **{name: value}).returncode == 0

    config = _exec_config(tmp_path)

    assert not marker.exists()
    assert value in _string_values(config)


@pytest.mark.parametrize(
    "name",
    [
        "ROMP_INIT_DAYS",
        "ROMP_MEMBERS",
        "ROMP_START_DATE",
        "ROMP_WET_THRESHOLD",
        "ROMP_DRY_SPELL",
        "ROMP_PROBABILISTIC",
        "ROMP_PARALLEL",
        "ROMP_UNIT_CVT",
        "ROMP_MODEL_DIMS",
    ],
)
def test_non_text_settings_that_are_not_values_are_rejected(tmp_path, name):
    """Non-text settings given code-like payloads make the script fail without writing a config or running the payload."""
    marker = tmp_path / "executed"
    value = f"0) or {PAYLOAD.format(marker=str(marker))} or (1"

    result = _generate(tmp_path, **{name: value})

    assert result.returncode == 1
    assert name in result.stderr
    assert not (tmp_path / "config.in").exists()
    assert not marker.exists()


def test_model_unit_conversion_and_dims_reach_the_per_model_lists(tmp_path):
    """ROMP_UNIT_CVT and ROMP_MODEL_DIMS end up in unit_cvt_list and model_dims_list."""
    result = _generate(
        tmp_path,
        ROMP_UNIT_CVT="1000",
        ROMP_MODEL_DIMS='{"init_time": "time", "step": "prediction_timedelta_daily"}',
    )
    assert result.returncode == 0

    config = _exec_config(tmp_path)

    assert config["unit_cvt_list"] == (1000,)
    assert config["model_dims_list"] == (
        {"init_time": "time", "step": "prediction_timedelta_daily"},
    )


def test_model_dims_default_to_name_inference(tmp_path):
    """Without ROMP_UNIT_CVT and ROMP_MODEL_DIMS, the per-model lists hold None."""
    assert _generate(tmp_path).returncode == 0

    config = _exec_config(tmp_path)

    assert config["unit_cvt_list"] == (None,)
    assert config["model_dims_list"] == (None,)


@pytest.mark.parametrize(
    "value", ['["time"]', '{"valid_time": "time"}', '{"init_time": ""}']
)
def test_malformed_model_dims_are_rejected(tmp_path, value):
    """Malformed ROMP_MODEL_DIMS values make the script fail with an error naming the variable."""
    result = _generate(tmp_path, ROMP_MODEL_DIMS=value)

    assert result.returncode == 1
    assert "ROMP_MODEL_DIMS" in result.stderr
