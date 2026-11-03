"""Test helper: the typed inputs of a run written in the flat names of the old code (`{**Setting, **Case}`).

The oracle (tests/_oracle_config.py) is the frozen old code, which passes a case as one flat dict. These functions flatten the
typed RunConfig, Case and Selection to that dict, so a test compares the oracle's dict with the typed objects' values.
Nothing in hcwf imports this.

    case_kwargs(rc, case, flow)       the flat names of one case: the run's settings and the case's own values
    model_kwargs(rc, model, ...)      the same for a model's first window (export, --check)
    selection_kwargs(selection)       the flat names a Selection sets
"""
from __future__ import annotations

import copy
from typing import Mapping

from hcwf.config.model import Case, RunConfig, Selection
from hcwf.config import schema

FLOWS = ("deterministic", "ensemble")
"""The two scoring flows. They build the reference the same way (its own settings); the ensemble flow names its cases after
the forecast model (RunConfig.reference_cases)."""
SELECTION_KWARGS = {"keep_unit_ids": "keep_unit_ids", "member_ids": "member_ids", "max_member": "max_member",
                    "issue_dates": "issue_dates", "first_lead": "first_lead", "last_lead": "last_lead", "model_files_years": "model_files_years",
                    "obs_files_years": "obs_files_years", "model_paths": "model_paths", "grid_point": "grid_point",
                    "lat": "lat_select", "lon": "lon_select", "land_only": "land_only", "shp_only": "shp_only",
                    "reference_onset": "reference_onset", "rain_horizon": "rain_horizon", "check_units": "check_units"}
"""Selection field -> the kwargs name the readers use."""
CASE_FIELDS = ("case_name", "model", "model_var", "model_dir", "obs", "obs_var", "ref_model", "ref_model_var",
               "probabilistic", "members", "onset_percentage_threshold", "verification_window", "tolerance_days",
               "max_forecast_day", "day_bins", "years", "years_clim", "region", "file_pattern", "unit_cvt",
               "accumulation", "daily_boundary", "model_dims")
"""The keys a case owns, in the order of the kwargs (the old Case dataclass); the run's settings are every other key."""
_RUNTIME = {"cache_dir": None, "short_names": False, "lead_window": None}
"""The settings that are not schema keys, with their defaults."""
_RECORD = ("accumulation_detected", "config_file", "config_sources", "dataset_report", "onset_lead_lines", "run_paths")
"""The other keys of a resolved flat record (what the resolve step adds); a blend's config carries them through."""


def _setting_part(values: Mapping) -> dict:
    """The run's settings: every schema key that is not a case field (in schema order) and the run's own settings, each
    from `values` or else its schema default (None for a required or computed one)."""
    out = {}
    for item in schema.FIELDS:
        if item.key in CASE_FIELDS:
            continue
        default = None if callable(item.default) or item.default is schema.REQUIRED else item.default
        out[item.key] = values[item.key] if item.key in values else copy.deepcopy(default)
    for key, default in _RUNTIME.items():
        out[key] = values[key] if key in values else default
    return out


def case_kwargs(rc: RunConfig, case: Case, flow: str, selection: Selection | None = None) -> dict:
    """The settings and the case's own values as one dict, as the old code's `**case_cfg`.

    The settings part is the run's flat record (`_setting_part`); the case part comes from the typed case. For a reference case `flow` is "deterministic" or "ensemble" and must be the flow
    that built it with `rc.reference_cases` (no `for_model` or `for_model`): the cases differ in their name only. day_bins are the
    run's whole bins; the ensemble flow's create_score_results narrows them to the window with `Window.bins`.
    A `selection` adds its keys that are set.
    """
    if flow not in FLOWS:
        raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")
    ev, ref, model = rc.evaluation, rc.reference, case.model
    mine = {
        "case_name": case.name, "model": model.name, "model_var": model.var, "model_dir": model.dir,
        "obs": rc.observations.name, "obs_var": rc.observations.var, "ref_model": ref.model, "ref_model_var": ref.var,
        "probabilistic": ev.probabilistic, "members": ev.members,
        "onset_percentage_threshold": rc.onset_options.percentage_threshold, "verification_window": case.window.span,
        "tolerance_days": case.window.tolerance, "max_forecast_day": ev.max_forecast_day, "day_bins": ev.day_bins,
        "years": ev.years, "years_clim": ev.climatology_years, "region": rc.area.region,
        "file_pattern": model.file_pattern, "unit_cvt": model.unit_cvt, "accumulation": model.accumulation,
        "daily_boundary": model.daily_boundary, "model_dims": model.dims_map()}
    out = {**_setting_part(rc._values), **{name: mine[name] for name in CASE_FIELDS}}
    if selection is not None:
        out.update(selection_kwargs(selection))
    return out


def model_kwargs(rc: RunConfig, model: str, *, reference_as: str | None = None, selection: Selection | None = None) -> dict:
    """The kwargs of one model's first window (the detection does not depend on the window, so it stands for all).

    With `reference_as` (a source name: "climatology" or "conditional_climatology") the case is the reference's, under
    that name, built by `RunConfig.reference_cases()`; `model` is then ignored."""
    return case_kwargs(rc, rc.first_case(model, reference_as=reference_as), "deterministic", selection)


def reference_settings(values: Mapping) -> dict:
    """The case keys a reference has of its own, from a flat record: its accumulation, daily boundary and dims (None for a
    climatology reference, which has none). The oracle gave the reference models[0]'s accumulation and boundary and, in the
    deterministic flow, no dims; tests lay these over the oracle's reference kwargs."""
    return {"accumulation": values.get("ref_model_accumulation"), "daily_boundary": values.get("ref_model_daily_boundary"),
            "model_dims": values.get("ref_model_dims")}


def selection_kwargs(selection: Selection) -> dict:
    """The kwargs a Selection sets (the fields that are not None)."""
    return {kwarg: getattr(selection, name) for name, kwarg in SELECTION_KWARGS.items()
            if getattr(selection, name) is not None}
