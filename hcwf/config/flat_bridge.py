"""R2-bridge: the transitional adapter between RunConfig and the code that still takes the flat dict or `**case_cfg`.

Design: docs/review/runconfig_design.md section 3 ("Transitional adapter"). Step S11 deletes this module and every
`# R2-bridge` wrapper that calls it. Nothing outside tests and these wrappers may import it.

    to_flat(rc)                       a copy of the resolved flat record
    case_kwargs(rc, case, flow)       the flat keyword arguments of one case: the run's settings and the case's own values
    model_kwargs(rc, model, ...)      the same for a model's first window (export, --check)
    from_case_kwargs(kw)              (rc, case, selection): the typed inputs a migrated function used to read from kw
    reader_inputs(kw)                 the typed inputs of the io readers, from a full or partial flat config
"""
# R2-bridge
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping

from hcwf.config.model import (AreaSpec, Case, EvaluationSpec, ModelSpec, ObsSpec, OnsetOptions, OutputSpec,
                               ParallelSpec, RunConfig, Selection, Window, _project_area, _project_evaluation,
                               _project_generated, _project_obs, _project_onset_options, _project_parallel)
from hcwf.lib import config_schema as schema

FLOWS = ("deterministic", "ensemble")
"""The two scoring flows. They differ in how the reference model is built (RunConfig.reference_cases)."""
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


def to_flat(rc: RunConfig) -> dict:
    """A copy of the resolved flat record, for callees that take the run dict."""
    return dict(rc.values)


def case_kwargs(rc: RunConfig, case: Case, flow: str, selection: Selection | None = None) -> dict:
    """The settings and the case's own values as one dict, for callees that take `**case_cfg`.

    The settings part is the run's flat record (`_setting_part`); the case part comes from the typed case. For a reference case `flow` is "deterministic" or "ensemble" and must be the flow
    that built it with `rc.reference_cases` (no `like` or `like`): the case then has that flow's rule. day_bins are the
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
    that name, built with the deterministic rule (`RunConfig.reference_cases()`); `model` is then ignored."""
    if reference_as is None:
        case = next(c for c in rc.cases() if c.model.name == model)
    else:
        case = rc.reference_cases()[0]
        case = replace(case, model=replace(case.model, name=reference_as))
    return case_kwargs(rc, case, "deterministic", selection)


def selection_kwargs(selection: Selection) -> dict:
    """The kwargs a Selection sets (the fields that are not None)."""
    return {kwarg: getattr(selection, name) for name, kwarg in SELECTION_KWARGS.items()
            if getattr(selection, name) is not None}


def _completed(kw: Mapping[str, Any]) -> dict:
    """The kwargs with what a partial dict leaves out filled in (blend configs and unit tests pass fewer keys than a run
    holds): each missing schema key a resolved record always holds takes its default (None when it is required or computed), and the case's keys and the
    per-model lists a missing source needs take placeholders that no reader of the partial dict looks at."""
    out = dict(kw)
    for item in schema.FIELDS:
        if item.fill and item.key not in out:
            out[item.key] = None if callable(item.default) or item.default is schema.REQUIRED else copy.deepcopy(item.default)
    if not out.get("years_clim") and None in (out.get("start_year_clim"), out.get("end_year_clim")):
        out["start_year_clim"], out["end_year_clim"] = 1, 0   # no climatology years
    for key, placeholder in (("start_date", (1, 1, 1)), ("end_date", (1, 1, 1)), ("init_days", ()), ("day_bins", ())):
        if out.get(key) is None:
            out[key] = placeholder
    out.setdefault("model", (out.get("model_list") or ["model"])[0])
    out.setdefault("model_dir", "")
    for key in ("model_var", "unit_cvt", "file_pattern", "accumulation", "daily_boundary", "model_dims"):
        out.setdefault(key, None)
    span = out.get("verification_window") or (1, out.get("max_forecast_day") or 1)
    out["verification_window"] = span
    out.setdefault("tolerance_days", 0)
    out.setdefault("case_name", Case(ModelSpec(out["model"], "", None, None, None, None, None, None, None, None, None),
                                     Window(*span, out["tolerance_days"])).name)
    for key, value in (("model_list", [out["model"]]), ("model_dir_list", [out["model_dir"]]),
                       ("model_var_list", [out["model_var"]]), ("unit_cvt_list", [out["unit_cvt"]]),
                       ("file_pattern_list", [out["file_pattern"]]), ("verification_window_list", [tuple(span)]),
                       ("tolerance_days_list", [out["tolerance_days"]])):
        if out.get(key) is None:
            out[key] = value
    return out


def _model_of(kw: Mapping[str, Any], units) -> ModelSpec:
    """The source a case reads: the kwargs' own `model*`, `file_pattern`, `unit_cvt`, `accumulation`, `daily_boundary` and
    `model_dims`, with its resolved `units`."""
    return ModelSpec(
        name=kw["model"], dir=kw["model_dir"], var=kw["model_var"], unit_cvt=kw["unit_cvt"],
        file_pattern=kw["file_pattern"], accumulation=kw["accumulation"], accumulation_detected=None,
        daily_boundary=kw["daily_boundary"],
        dims=None if kw["model_dims"] is None else tuple(kw["model_dims"].items()), paths=None, units=units)


def _selection_of(kw: Mapping[str, Any]) -> Selection:
    """The Selection of the kwargs: each selection key that is present."""
    return Selection(**{name: kw[kwarg] for name, kwarg in SELECTION_KWARGS.items() if kwarg in kw})


def from_case_kwargs(kw: Mapping[str, Any]) -> tuple[RunConfig, Case, Selection]:
    """The typed inputs behind one case's kwargs: (RunConfig, Case, Selection).

    The flat record is rebuilt from the kwargs' schema keys, so a value edited in place (years, max_forecast_day, the
    reference keys) is kept. The case's source and window are the kwargs' own `model*`, `file_pattern`, `unit_cvt`,
    `accumulation`, `daily_boundary`, `model_dims`, `verification_window` and `tolerance_days`, so a case need not be one of
    the record's models (a reference case is not). Keys that are neither config nor selection raise.
    """
    case_only = set(CASE_FIELDS) - set(schema.BY_KEY)
    selection_kwargs = set(SELECTION_KWARGS.values())
    unknown = sorted(set(kw) - set(schema.BY_KEY) - case_only - set(_RUNTIME) - set(_RECORD) - selection_kwargs)
    if unknown:
        raise ValueError(f"not config keys or selections: {', '.join(unknown)}")
    kw = _completed(kw)
    values = {k: v for k, v in kw.items() if k in schema.BY_KEY or k in _RUNTIME or k in _RECORD}
    rc = RunConfig.from_flat(values, check=False)
    model = _model_of(kw, rc.reference.units if kw["model"] == rc.reference.model else
                      next((m.units for m in rc.models if m.name == kw["model"]), None))
    window = Window(*kw["verification_window"], kw["tolerance_days"])
    # a reference that is also one of the forecast models (same name) cannot be told apart, and is read the same way
    reference = kw["model"] == kw["ref_model"] and kw["model"] not in {m.name for m in rc.models}
    for_model = None
    if kw["case_name"] != Case(model, window).name:   # the ensemble flow names the reference's case after its model
        for_model = next((m.name for m in rc.models if Case(m, window).name == kw["case_name"]), None)
    case = Case(model, window, reference=reference, for_model=for_model)
    return rc, case, _selection_of(kw)


@dataclass(frozen=True)
class ReaderInputs:
    """The typed inputs of the io readers behind a flat config (see `reader_inputs`)."""

    obs: ObsSpec
    area: AreaSpec
    options: OnsetOptions
    ev: EvaluationSpec
    model: ModelSpec
    parallel: ParallelSpec
    output: OutputSpec
    select: Selection
    dir_fig: Any
    cache_dir: Any
    kwargs: Mapping

    def onset(self):
        """The onset setup of the config (memoised by `onset_setup`); an invalid or incomplete config raises."""
        from hcwf.lib.onset_config import onset_setup

        return onset_setup(self.kwargs)

    def onset_if_units(self):
        """The onset setup when the observations are on units (it labels their dates), else None."""
        return self.onset() if self.obs.units is not None else None

    def onset_if_steps(self):
        """The onset setup when the config has a max_forecast_day (it sets how many leads are read), else None."""
        return self.onset() if isinstance(self.ev.max_forecast_day, int) else None


def reader_inputs(kw: Mapping[str, Any]) -> ReaderInputs:
    """The typed inputs the io readers take, from a flat config: a run's case kwargs, or fewer keys (blend configs and
    unit tests pass what the reader needs). Nothing is looked up that the old reader did not look up: a missing key is
    what `kwargs.get` gave it, so the onset setup is built only when called for (`ReaderInputs.onset`). Values edited in
    place are kept. Unlike `from_case_kwargs` this builds no RunConfig, so it works without an onset definition."""
    from hcwf.spatial.units import UnitsSpec

    v = _completed(kw)
    for key, empty in (("day_bins", ()), ("init_days", ()), ("start_date", (1, 1, 1)), ("end_date", (1, 1, 1))):
        if v.get(key) is None:   # a reader that needs them was given them; the projection wants a sequence
            v[key] = empty
    run_years = tuple(kw.get("years") or ())
    ev = replace(_project_evaluation(v), years=run_years, max_forecast_day=kw.get("max_forecast_day"))
    if not kw.get("verification_window_list"):
        ev = replace(ev, windows=())   # no windows: one threshold pools every lead
    output = replace(_project_generated(v, OutputSpec, "output"), data_dir=kw.get("dir_out"),
                     figure_dir=kw.get("dir_fig"))
    obs_units = UnitsSpec.from_flat(v, "observations")
    return ReaderInputs(
        obs=_project_obs(v, {"observations": obs_units}), area=_project_area(v), options=_project_onset_options(v),
        ev=ev, model=_model_of(v, UnitsSpec.from_flat(v, v["model"])), parallel=_project_parallel(v), output=output,
        select=_selection_of(kw), dir_fig=kw.get("dir_fig"), cache_dir=kw.get("cache_dir"), kwargs=kw)
