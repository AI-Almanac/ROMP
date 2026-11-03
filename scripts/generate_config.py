"""
Generate an HCWF YAML run config from environment variables.

Every value is parsed into a typed Python value first and written with a YAML
dumper; no environment text is ever spliced into the file as code.

Required env vars:
  HCWF_OBS_DIR          - path to observation NetCDF files
  HCWF_MODEL_DIR        - path to forecast model NetCDF files
  HCWF_MODEL_NAME       - model identifier (e.g. AIFS)
  HCWF_DIR_OUT          - output directory for CSVs / NetCDFs
  HCWF_DIR_FIG          - output directory for figures

Optional env vars (with defaults matching the demo config):
  HCWF_OBS              - observation dataset name (default: CHIRPS_IMERG)
  HCWF_OBS_FILE_PATTERN - file naming pattern (default: {}.nc)
  HCWF_OBS_VAR          - rainfall variable name in obs file (default: RAINFALL)
  HCWF_MODEL_VAR        - rainfall variable name in forecast file (default: tp)
  HCWF_FILE_PATTERN     - forecast file naming pattern (default: {}.nc)
  HCWF_UNIT_CVT         - multiplier converting forecast rainfall to mm (default: None)
  HCWF_MODEL_DIMS       - JSON object naming the forecast file's dims, e.g.
                          {"init_time": "time", "step": "prediction_timedelta_daily"}
                          (default: None, infer from names)
  HCWF_REGION           - target region (default: Ethiopia)
  HCWF_NC_MASK          - path to land/region mask NetCDF (default: None)
  (variables marked [legacy] apply only when no definition-mode variable is set)
  HCWF_THRESH_FILE      - path to spatial threshold NetCDF (default: None) [legacy]
  HCWF_WET_THRESHOLD    - scalar rainfall threshold mm (default: 20) [legacy]
  HCWF_WET_INIT         - minimum wet day threshold mm (default: 1) [legacy]
  HCWF_WET_SPELL        - wet spell length days (default: 3) [legacy]
  HCWF_DRY_SPELL        - dry spell length days (default: 7) [legacy]
  HCWF_DRY_EXTENT       - dry spell search window days (default: 0) [legacy]
  HCWF_ONSET_DEFINITION - onset definition id or YAML path (default: unset, flat keys below).
                          Setting this, or any other variable of the onset group (HCWF_ONSET_SEASON,
                          HCWF_THRESHOLD_YEARS, HCWF_THRESHOLDS, HCWF_OBS_DAY_BOUNDARY, HCWF_MODEL_LEAD_LABEL,
                          HCWF_CHECK_ALIGNMENT, HCWF_ONSET_BACKEND, HCWF_FORECAST_HISTORY, HCWF_THRESHOLD_SOURCE),
                          selects definition mode; the flat HCWF_WET_* / HCWF_DRY_* (except HCWF_DRY_SPELL_CHECK) /
                          HCWF_THRESH_FILE variables must then be unset. With none of the group set, a legacy
                          flat-key config is written
  HCWF_ONSET_SEASON     - season name for definitions with several seasons (default: None)
  HCWF_THRESHOLD_YEARS  - first-last years for climatological thresholds, e.g. 1998-2024 (default: region)
  HCWF_THRESHOLDS       - JSON object {"name": "path.nc:variable"} of precomputed thresholds (default: None)
  HCWF_OBS_DAY_BOUNDARY - e.g. "00 UTC, labelled by start" (default: region)
  HCWF_MODEL_LEAD_LABEL - start or end (default: from the day boundary)
  HCWF_CHECK_ALIGNMENT  - True/False (default: True)
  HCWF_DRY_SPELL_CHECK  - False drops the definition's dry-spell condition (default: True; both modes)
  HCWF_FORECAST_HISTORY - observed or none (default: observed; definition mode only)
  HCWF_THRESHOLD_SOURCE - observations or model (default: observations; definition mode only)
  HCWF_ONSET_BACKEND    - numba, reference or gpu (default: numba)
  HCWF_START_DATE       - evaluation start as YYYY-MM-DD (default: 2019-05-01)
  HCWF_END_DATE         - evaluation end as YYYY-MM-DD (default: 2024-07-31)
  HCWF_START_YEAR_CLIM  - climatology start year (default: 1998)
  HCWF_END_YEAR_CLIM    - climatology end year (default: 2024)
  HCWF_MAX_FORECAST_DAY - max forecast lead day (default: 30)
  HCWF_PROBABILISTIC    - True/False (default: False)
  HCWF_MEMBERS          - ensemble members, All or comma-separated ints (default: All)
  HCWF_PARALLEL         - True/False (default: True)
  HCWF_REF_MODEL        - reference model name (default: climatology)
  HCWF_REF_MODEL_DIR    - reference model data dir (default: same as obs dir)
  HCWF_INIT_DAYS        - forecast init weekdays as comma-sep ints (0=Mon, default: 0,3)
  HCWF_DATE_FILTER_YEAR - reference year for init-day calendar alignment (default: start_date year)
  HCWF_CONFIG_PATH      - where to write the config (default: /tmp/hcwf_job.yaml); run it with hcwf-run -p
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path


class ConfigError(ValueError):
    """Raised when an environment variable is missing or has an invalid value."""
    pass


def require(name: str) -> str:
    """Return the value of a required environment variable, raising ConfigError if it is unset or empty."""
    val = os.environ.get(name)
    if not val:
        raise ConfigError(f"required environment variable {name} is not set")
    return val


def opt(name: str, default: str) -> str:
    """Return the value of an optional environment variable, or `default` if it is unset."""
    return os.environ.get(name, default)


def _parsed(name: str, default: str, parse):
    """Read an environment variable (or `default`) and apply `parse`, wrapping parse errors in ConfigError."""
    raw = opt(name, default)
    try:
        return parse(raw.strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}={raw!r} is invalid: {exc}") from exc


def parse_date(val: str) -> tuple:
    """Parse a YYYY-MM-DD string into a (year, month, day) tuple."""
    dt = datetime.strptime(val, "%Y-%m-%d")
    return (dt.year, dt.month, dt.day)


def parse_number(val: str) -> int | float:
    """Parse a finite number, as an int if possible, otherwise as a float."""
    try:
        return int(val)
    except ValueError:
        number = float(val)
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    return number


def parse_bool(val: str) -> bool:
    """Parse true/1/yes or false/0/no (case-insensitive) into a bool."""
    lowered = val.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ValueError("must be True or False")


def parse_int_tuple(val: str) -> tuple:
    """Parse a comma-separated list of integers into a non-empty tuple."""
    values = tuple(int(part) for part in val.split(",") if part.strip())
    if not values:
        raise ValueError("must list at least one integer")
    return values


def parse_members(val: str) -> str | tuple:
    """Parse ensemble members as "All" or a tuple of integers."""
    return "All" if val.lower() == "all" else parse_int_tuple(val)


def optional_path(val: str) -> str | None:
    """Return the string, or None if it is empty."""
    return val or None


def optional_number(val: str) -> int | float | None:
    """Parse a number, or return None if the string is empty."""
    return parse_number(val) if val else None


MODEL_DIM_NAMES = frozenset({"init_time", "step", "member", "lat", "lon"})


def parse_model_dims(val: str) -> dict | None:
    """Parse a JSON object mapping canonical dim names to file dim names, or None if empty."""
    if not val:
        return None
    dims = json.loads(val)
    if not isinstance(dims, dict):
        raise ValueError("must be a JSON object")
    unknown = sorted(set(dims) - MODEL_DIM_NAMES)
    if unknown:
        raise ValueError(f"unknown dims {unknown}; expected some of {sorted(MODEL_DIM_NAMES)}")
    if not all(isinstance(name, str) and name for name in dims.values()):
        raise ValueError("file dim names must be non-empty strings")
    return dims


ONSET_ENV = ("HCWF_ONSET_DEFINITION", "HCWF_ONSET_SEASON", "HCWF_THRESHOLD_YEARS", "HCWF_THRESHOLDS",
             "HCWF_OBS_DAY_BOUNDARY", "HCWF_MODEL_LEAD_LABEL", "HCWF_CHECK_ALIGNMENT", "HCWF_ONSET_BACKEND",
             "HCWF_FORECAST_HISTORY", "HCWF_THRESHOLD_SOURCE")
FLAT_ONSET_ENV = ("HCWF_WET_INIT", "HCWF_WET_THRESHOLD", "HCWF_WET_SPELL", "HCWF_DRY_SPELL", "HCWF_DRY_EXTENT",
                  "HCWF_THRESH_FILE")


def parse_years(val: str) -> tuple | None:
    """Parse "first-last" into a (first, last) tuple, or None if empty."""
    if not val:
        return None
    first, last = (int(part) for part in val.split("-"))
    return (first, last)


def parse_thresholds(val: str) -> dict | None:
    """Parse a JSON object mapping threshold names to "path.nc:variable", or None if empty."""
    if not val:
        return None
    mapping = json.loads(val)
    if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
        raise ValueError('must be a JSON object like {"name": "path.nc:variable"}')
    return mapping


def parse_choice(*allowed):
    """A parser accepting only the allowed strings."""
    def parse(val: str) -> str:
        """Return val if it is one of the allowed strings."""
        if val not in allowed:
            raise ValueError(f"must be one of {', '.join(allowed)}")
        return val
    return parse


def onset_values() -> dict:
    """The onset part of the config: flat legacy keys, or the onset_definition keys when any of those is set."""
    dry_spell_check = {"dry_spell_check": _parsed("HCWF_DRY_SPELL_CHECK", "True", parse_bool)}
    if not any(name in os.environ for name in ONSET_ENV):
        return {**dry_spell_check,
            "wet_init": _parsed("HCWF_WET_INIT", "1", parse_number),
            "wet_threshold": _parsed("HCWF_WET_THRESHOLD", "20", parse_number),
            "wet_spell": _parsed("HCWF_WET_SPELL", "3", int),
            "dry_threshold": 1,
            "dry_spell": _parsed("HCWF_DRY_SPELL", "7", int),
            "dry_extent": _parsed("HCWF_DRY_EXTENT", "0", int),
            "thresh_file": _parsed("HCWF_THRESH_FILE", "", optional_path),
            "thresh_var": None,
        }
    clash = [name for name in FLAT_ONSET_ENV if name in os.environ]
    if clash:
        raise ConfigError(f"{', '.join(clash)} cannot be combined with the onset definition variables; "
                          "put the rule in a definition file (extends: <id>) instead")
    return {
        "onset_definition": _parsed("HCWF_ONSET_DEFINITION", "", optional_path),
        "onset_season": _parsed("HCWF_ONSET_SEASON", "", optional_path),
        "threshold_years": _parsed("HCWF_THRESHOLD_YEARS", "", parse_years),
        "thresholds": _parsed("HCWF_THRESHOLDS", "", parse_thresholds),
        "obs_day_boundary": _parsed("HCWF_OBS_DAY_BOUNDARY", "", optional_path),
        "model_lead_label": _parsed("HCWF_MODEL_LEAD_LABEL", "", optional_path),
        "check_alignment": _parsed("HCWF_CHECK_ALIGNMENT", "True", parse_bool),
        "onset_backend": opt("HCWF_ONSET_BACKEND", "numba"),
        "forecast_history": _parsed("HCWF_FORECAST_HISTORY", "observed", parse_choice("observed", "none")),
        "threshold_source": _parsed("HCWF_THRESHOLD_SOURCE", "observations", parse_choice("observations", "model")),
        **dry_spell_check,
    }


def config_values() -> dict:
    """Build the config dictionary from environment variables, applying the documented defaults."""
    obs_dir = require("HCWF_OBS_DIR")
    model_dir = require("HCWF_MODEL_DIR")
    model_name = require("HCWF_MODEL_NAME")
    dir_out = require("HCWF_DIR_OUT")
    dir_fig = require("HCWF_DIR_FIG")

    obs_file_pat = opt("HCWF_OBS_FILE_PATTERN", "{}.nc")
    obs_var = opt("HCWF_OBS_VAR", "RAINFALL")
    start_date = _parsed("HCWF_START_DATE", "2019-05-01", parse_date)
    end_date = _parsed("HCWF_END_DATE", "2024-07-31", parse_date)
    date_filter_year = _parsed("HCWF_DATE_FILTER_YEAR", "", lambda v: int(v) if v else start_date[0])

    return {
        "project_name": "HCWF container run",
        "work_dir": dir_out,
        "pkg_dir": "/app",
        "layout": ("model", "verification_window"),
        "model_list": (model_name,),
        "obs": opt("HCWF_OBS", "CHIRPS_IMERG"),
        "obs_dir": obs_dir,
        "obs_file_pattern": (obs_file_pat,),
        "obs_var": obs_var,
        "obs_unit_cvt": None,
        "ref_model": opt("HCWF_REF_MODEL", "climatology"),
        "ref_model_dir": opt("HCWF_REF_MODEL_DIR", obs_dir),
        "ref_model_file_pattern": obs_file_pat,
        "ref_model_var": obs_var,
        "ref_model_unit_cvt": None,
        "model_dir_list": (model_dir,),
        "model_var_list": (opt("HCWF_MODEL_VAR", "tp"),),
        "unit_cvt_list": (_parsed("HCWF_UNIT_CVT", "", optional_number),),
        "file_pattern_list": (opt("HCWF_FILE_PATTERN", "{}.nc"),),
        "model_dims_list": (_parsed("HCWF_MODEL_DIMS", "", parse_model_dims),),
        "obs_day_boundary": None,
        "model_lead_label": None,
        "region": opt("HCWF_REGION", "Ethiopia"),
        "nc_mask": _parsed("HCWF_NC_MASK", "", optional_path),
        "shpfile_dir": None,
        "polygon": False,
        **onset_values(),
        "onset_percentage_threshold": 0.5,
        "start_date": start_date,
        "end_date": end_date,
        "start_year_clim": _parsed("HCWF_START_YEAR_CLIM", "1998", int),
        "end_year_clim": _parsed("HCWF_END_YEAR_CLIM", "2024", int),
        "init_days": _parsed("HCWF_INIT_DAYS", "0,3", parse_int_tuple),
        "date_filter_year": date_filter_year,
        "verification_window_list": ((1, 15), (16, 30)),
        "tolerance_days_list": (3, 5),
        "max_forecast_day": _parsed("HCWF_MAX_FORECAST_DAY", "30", int),
        "day_bins": ((1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)),
        "FAR": True,
        "MAE": True,
        "MR": True,
        "probabilistic": _parsed("HCWF_PROBABILISTIC", "False", parse_bool),
        "members": _parsed("HCWF_MEMBERS", "All", parse_members),
        "BS": True,
        "RPS": True,
        "AUC": True,
        "Reliability": True,
        "skill_score": True,
        "dir_out": dir_out,
        "dir_fig": dir_fig,
        "save_fig": True,
        "save_nc_spatial_far_mr_mae": True,
        "save_csv_score": True,
        "save_nc_climatology": True,
        "plot_spatial_far_mr_mae": True,
        "plot_heatmap_bss_auc": True,
        "plot_reliability": True,
        "plot_climatology_onset": True,
        "plot_panel_heatmap_error": True,
        "plot_panel_heatmap_skill": True,
        "plot_bar_bss_rpss_auc": True,
        "show_plot": False,
        "show_panel": False,
        "parallel": _parsed("HCWF_PARALLEL", "True", parse_bool),
        "debug": False,
    }


def render_config(values: dict) -> str:
    """Render the flat config dictionary as a nested YAML run config."""
    from hcwf.lib.convert import flat_to_mapping, render_yaml

    mapping, _ = flat_to_mapping(values)
    return render_yaml(mapping, header="# Generated by scripts/generate_config.py from HCWF_* environment variables.\n")


def main():
    """Write the generated config file and exit with status 1 on invalid configuration."""
    try:
        config = render_config(config_values())
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path = os.environ.get("HCWF_CONFIG_PATH", "/tmp/hcwf_job.yaml")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(config)
    print(f"Config written to {output_path}")


if __name__ == "__main__":
    main()
