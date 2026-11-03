"""generate_config.py turns HCWF_* env vars into a YAML run config."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_config.py"
PAYLOAD = "__import__('pathlib').Path({marker!r}).touch()"


def _generate(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    """Run generate_config.py in a subprocess with the required HCWF_* env vars plus overrides."""
    required = {
        "HCWF_OBS_DIR": str(tmp_path / "obs"),
        "HCWF_MODEL_DIR": str(tmp_path / "model"),
        "HCWF_MODEL_NAME": "fuxi",
        "HCWF_DIR_OUT": str(tmp_path / "out"),
        "HCWF_DIR_FIG": str(tmp_path / "fig"),
        "HCWF_CONFIG_PATH": str(tmp_path / "config.yaml"),
    }
    clean = {k: v for k, v in os.environ.items() if not k.startswith("HCWF_")}
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
    """Load the generated YAML and return the flat config dict the pipeline would use."""
    import yaml

    from hcwf.lib.config_yaml import build_config

    return build_config(yaml.safe_load((tmp_path / "config.yaml").read_text()))


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
        HCWF_INIT_DAYS="2, 5",
        HCWF_MEMBERS="0,1,2",
        HCWF_WET_THRESHOLD="25.0",
        HCWF_PROBABILISTIC="True",
        HCWF_NC_MASK="/data/masks/ethiopia mask.nc",
        HCWF_DATE_FILTER_YEAR="2020",
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
    ["HCWF_OBS_VAR", "HCWF_MODEL_NAME", "HCWF_FILE_PATTERN", "HCWF_NC_MASK", "HCWF_REGION"],
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
        "HCWF_INIT_DAYS",
        "HCWF_MEMBERS",
        "HCWF_START_DATE",
        "HCWF_WET_THRESHOLD",
        "HCWF_DRY_SPELL",
        "HCWF_PROBABILISTIC",
        "HCWF_PARALLEL",
        "HCWF_UNIT_CVT",
        "HCWF_MODEL_DIMS",
    ],
)
def test_non_text_settings_that_are_not_values_are_rejected(tmp_path, name):
    """Non-text settings given code-like payloads make the script fail without writing a config or running the payload."""
    marker = tmp_path / "executed"
    value = f"0) or {PAYLOAD.format(marker=str(marker))} or (1"

    result = _generate(tmp_path, **{name: value})

    assert result.returncode == 1
    assert name in result.stderr
    assert not (tmp_path / "config.yaml").exists()
    assert not marker.exists()


def test_model_unit_conversion_and_dims_reach_the_per_model_lists(tmp_path):
    """HCWF_UNIT_CVT and HCWF_MODEL_DIMS end up in unit_cvt_list and model_dims_list."""
    result = _generate(
        tmp_path,
        HCWF_UNIT_CVT="1000",
        HCWF_MODEL_DIMS='{"init_time": "time", "step": "prediction_timedelta_daily"}',
    )
    assert result.returncode == 0

    config = _exec_config(tmp_path)

    assert config["unit_cvt_list"] == (1000,)
    assert config["model_dims_list"] == (
        {"init_time": "time", "step": "prediction_timedelta_daily"},
    )


def test_model_dims_default_to_name_inference(tmp_path):
    """Without HCWF_UNIT_CVT and HCWF_MODEL_DIMS, the per-model lists hold None."""
    assert _generate(tmp_path).returncode == 0

    config = _exec_config(tmp_path)

    assert config["unit_cvt_list"] == (None,)
    assert config.get("model_dims_list") in (None, (None,))


@pytest.mark.parametrize(
    "value", ['["time"]', '{"valid_time": "time"}', '{"init_time": ""}']
)
def test_malformed_model_dims_are_rejected(tmp_path, value):
    """Malformed HCWF_MODEL_DIMS values make the script fail with an error naming the variable."""
    result = _generate(tmp_path, HCWF_MODEL_DIMS=value)

    assert result.returncode == 1
    assert "HCWF_MODEL_DIMS" in result.stderr


def test_onset_definition_variables_replace_the_flat_keys(tmp_path):
    """Setting HCWF_ONSET_DEFINITION writes the definition keys and no flat onset keys."""
    result = _generate(tmp_path, HCWF_ONSET_DEFINITION="icpac", HCWF_THRESHOLD_YEARS="1998-2024",
                       HCWF_CHECK_ALIGNMENT="False")
    assert result.returncode == 0

    config = _exec_config(tmp_path)

    assert config["onset_definition"] == "icpac"
    assert config["threshold_years"] == (1998, 2024)
    assert config["check_alignment"] is False
    assert config["onset_backend"] == "numba"
    assert "wet_init" not in config and "thresh_file" not in config


def test_flat_onset_variables_cannot_be_mixed_with_a_definition(tmp_path):
    """A flat onset variable beside HCWF_ONSET_DEFINITION fails with a message naming it."""
    result = _generate(tmp_path, HCWF_ONSET_DEFINITION="icpac", HCWF_WET_SPELL="4")

    assert result.returncode == 1
    assert "HCWF_WET_SPELL" in result.stderr


def test_defaults_keep_the_flat_onset_keys(tmp_path):
    """Without onset definition variables the config has the flat keys and no onset_definition (legacy mode)."""
    assert _generate(tmp_path).returncode == 0

    config = _exec_config(tmp_path)

    assert "onset_definition" not in config
    assert config["wet_spell"] == 3
