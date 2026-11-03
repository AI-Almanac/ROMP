"""Test helpers: the typed API of hcwf called with the specs of a flat test config.

`case_kwargs(rc, case, flow)` flattens a typed case to the flat names the tests use (`{**settings, **case}`);
`selection_kwargs` does the same for a Selection.

The tests keep their configs as flat keyword dicts (what a resolved run holds, or a part of it). These helpers build the
typed inputs from them (`from_case_kwargs`: run, case and selection; `reader_inputs`: the specs the io readers take) and
call the typed API (`load_observations`, `load_forecast`, `load_thresholds`, ...). `flat_config` is a resolved run's flat
record, the form the tests start from.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import copy

from hcwf.config.build import resolve
from hcwf.config.model import complete_flat as _completed
from hcwf.config.model import (AreaSpec, Case, EvaluationSpec, FigureSpec, ModelSpec, ObsSpec, OnsetOptions, OutputSpec,
                               ParallelSpec, RunConfig, Selection, Window, _project_area, _project_evaluation,
                               _project_generated, _project_obs, _project_onset_options, _project_parallel)
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.io import thresholds as thresholds_io
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


def selection_kwargs(selection: Selection) -> dict:
    """The kwargs a Selection sets (the fields that are not None)."""
    return {kwarg: getattr(selection, name) for name, kwarg in SELECTION_KWARGS.items()
            if getattr(selection, name) is not None}


def skip_region_cut(monkeypatch):
    """Read the fixtures whole: `region_select` (the region's box and cell mask) does nothing in every reader."""
    for module in (observations_io, forecasts_io, thresholds_io):
        monkeypatch.setattr(module, "region_select", lambda ds, **kwargs: ds)


def flat_config(source, overrides=None, **options) -> dict:
    """The resolved flat record of a run (a copy of `resolve(...).values`); the arguments are those of `hcwf.config.build.resolve`.
    With check=True, (record or None, problems)."""
    got = resolve(source, overrides, **options)
    if options.get("check"):
        rc, found = got
        return (None if rc is None else dict(rc.values)), found
    return dict(got.values)


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
        from hcwf.config.onset import onset_setup

        return onset_setup(self.kwargs)

    def onset_if_units(self):
        """The onset setup when the observations are on units (it labels their dates), else None."""
        return self.onset() if self.obs.units is not None else None

    def onset_if_steps(self):
        """The onset setup when the config has a max_forecast_day (it sets how many leads are read), else None."""
        return self.onset() if isinstance(self.ev.max_forecast_day, int) else None


def figure_spec(kw: Mapping[str, Any]) -> FigureSpec:
    """The FigureSpec a flat config gives (the keys it lacks are what the old figure code's defaults were: show and save on)."""
    inputs = reader_inputs(kw)
    return FigureSpec(dir_fig=kw.get("dir_fig"), region=inputs.area.region, shpfile_dir=inputs.area.shpfile_dir,
                      polygon=inputs.area.polygon, save_fig=kw.get("save_fig", True), show_plot=kw.get("show_plot", True),
                      show_panel=kw.get("show_panel", True), date_filter_year=kw.get("date_filter_year"),
                      short_names=bool(kw.get("short_names")), models=tuple(kw.get("model_list") or ()),
                      max_forecast_day=inputs.ev.max_forecast_day, units=inputs.obs.units)


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

def inputs(kwargs):
    """The typed inputs of a flat config (see `ReaderInputs`)."""
    return reader_inputs(kwargs)


def observations(year, kwargs):
    """`load_observations` of a flat config."""
    got = inputs(kwargs)
    return observations_io.load_observations(year, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                                      run_years=got.ev.years, dir_fig=got.dir_fig)


def obs_years(years, kwargs, hold=True):
    """`read_obs_years` of a flat config."""
    got = inputs(kwargs)
    return observations_io.read_obs_years(years, got.obs, got.area, got.options, got.select, onset=got.onset_if_units(),
                                   run_years=got.ev.years, dir_fig=got.dir_fig, parallel=got.parallel, hold=hold)


def forecast(year, kwargs, ensemble):
    """`load_forecast` of a flat config."""
    got = inputs(kwargs)
    return forecasts_io.load_forecast(year, got.model, got.ev, got.area, got.options, got.onset_if_steps(),
                                  ensemble=ensemble, select=got.select, dir_fig=got.dir_fig)


def thresholds(kwargs):
    """`load_thresholds` of a flat config."""
    got = inputs(kwargs)
    return thresholds_io.load_thresholds(got.onset(), got.obs, got.area, got.options, got.ev, got.output, model=got.model,
                                    select=got.select, parallel=got.parallel, cache_dir=got.cache_dir)


def history(year, current, kwargs):
    """`observed_history` of a flat config."""
    got = inputs(kwargs)
    return observations_io.observed_history(year, current, got.onset(), got.obs, got.area, got.options, got.select,
                                     run_years=got.ev.years, dir_fig=got.dir_fig)


def steps_needed(kwargs):
    """`steps_needed` of a flat config."""
    got = inputs(kwargs)
    return forecasts_io.steps_needed(kwargs.get("max_forecast_day"), got.onset_if_steps())


def initialization_dates(year, kwargs):
    """`initialization_dates` of a flat config."""
    return forecasts_io.initialization_dates(year, inputs(kwargs).ev)


def flat_command_line(args, **options) -> dict:
    """The flat record of the config a command line names (`hcwf.config.build.resolve_command_line`)."""
    from hcwf.config.build import resolve_command_line

    return dict(resolve_command_line(args, **options).values)


def ingest_options(kwargs, forecast=False) -> dict:
    """`OnsetOptions.ingest` of a flat config (the keyword options of `ingest`, or of `ingest_forecast_info`)."""
    got = inputs(kwargs)
    return got.options.ingest(forecast, got.obs.units, check_units=got.select.check_units)


def ingest_forecast(cube, setup, kwargs, source=None):
    """`ingest_forecast` of a flat config (its model's accumulation and boundary, its options and selection)."""
    got = inputs(kwargs)
    return forecasts_io.ingest_forecast(cube, setup, got.model, got.options, got.select, source)
