"""R2-bridge: the transitional adapter between RunConfig and the code that still takes the flat dict or `**case_cfg`.

Design: docs/review/runconfig_design.md section 3 ("Transitional adapter"). Step S11 deletes this module and every
`# R2-bridge` wrapper that calls it. Nothing outside tests and these wrappers may import it.

    to_flat(rc)                       a copy of the resolved flat record
    case_kwargs(rc, case, flow)       today's {**asdict(Setting), **asdict(Case)} of one case, as the flow builds it
    from_case_kwargs(kw)              (rc, case, selection): the typed inputs a migrated function used to read from kw
"""
# R2-bridge
from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Mapping

from hcwf.config.model import Case, ModelSpec, RunConfig, Selection, Window
from hcwf.lib import config_schema as schema

FLOWS = ("deterministic", "ensemble")
"""The two scoring flows. They differ in how the reference model is built (RunConfig.reference_cases)."""
SELECTION_KWARGS = {"keep_unit_ids": "keep_unit_ids", "member_ids": "member_ids", "max_member": "max_member",
                    "issue_dates": "issue_dates", "last_lead": "last_lead", "model_files_years": "model_files_years",
                    "obs_files_years": "obs_files_years", "model_paths": "model_paths", "grid_point": "grid_point",
                    "lat": "lat_select", "lon": "lon_select", "land_only": "land_only", "shp_only": "shp_only",
                    "reference_onset": "reference_onset", "rain_horizon": "rain_horizon"}
"""Selection field -> the kwargs name the readers use."""
_RUNTIME = ("cache_dir", "short_names", "lead_window")
"""The settings that are not schema keys (convention._RUNTIME_SETTINGS)."""


def to_flat(rc: RunConfig) -> dict:
    """A copy of the resolved flat record, for callees that take the run dict."""
    return dict(rc.values)


def case_kwargs(rc: RunConfig, case: Case, flow: str, selection: Selection | None = None) -> dict:
    """{**asdict(Setting), **asdict(Case)} of one case, for callees that take `**case_cfg`.

    The Setting part is the run's flat record (keys in the order and with the defaults of convention.Setting); the Case
    part comes from the typed case. For a reference case `flow` is "deterministic" or "ensemble" and must be the flow
    that built it with `rc.reference_cases` (no `like` or `like`): the case then has that flow's rule. day_bins are the
    run's whole bins; the ensemble flow's create_score_results narrows them to the window with `Window.bins`.
    A `selection` adds its keys that are set.
    """
    from hcwf.lib.control import init_dataclass
    from hcwf.lib.convention import Case as OldCase, Setting

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
    names = [f.name for f in fields(OldCase)]
    if set(names) != set(mine):
        raise RuntimeError(f"convention.Case and the bridge disagree: {sorted(set(names) ^ set(mine))}")
    out = {**asdict(init_dataclass(Setting, rc._values)), **{name: mine[name] for name in names}}
    if selection is not None:
        out.update(_selection_kwargs(selection))
    return out


def _selection_kwargs(selection: Selection) -> dict:
    """The kwargs a Selection sets (the fields that are not None)."""
    return {kwarg: getattr(selection, name) for name, kwarg in SELECTION_KWARGS.items()
            if getattr(selection, name) is not None}


def from_case_kwargs(kw: Mapping[str, Any]) -> tuple[RunConfig, Case, Selection]:
    """The typed inputs behind one case's kwargs: (RunConfig, Case, Selection).

    The flat record is rebuilt from the kwargs' schema keys, so a value edited in place (years, max_forecast_day, the
    reference keys) is kept. The case's source and window are the kwargs' own `model*`, `file_pattern`, `unit_cvt`,
    `accumulation`, `daily_boundary`, `model_dims`, `verification_window` and `tolerance_days`, so a case need not be one of
    the record's models (a reference case is not). Keys that are neither config nor selection raise.
    """
    from hcwf.lib.convention import Case as OldCase

    case_only = {f.name for f in fields(OldCase)} - set(schema.BY_KEY)
    selection_kwargs = set(SELECTION_KWARGS.values())
    unknown = sorted(set(kw) - set(schema.BY_KEY) - case_only - set(_RUNTIME) - selection_kwargs)
    if unknown:
        raise ValueError(f"not config keys or selections: {', '.join(unknown)}")
    values = {k: v for k, v in kw.items() if k in schema.BY_KEY or k in _RUNTIME}
    rc = RunConfig.from_flat(values, check=False)
    model = ModelSpec(
        name=kw["model"], dir=kw["model_dir"], var=kw["model_var"], unit_cvt=kw["unit_cvt"],
        file_pattern=kw["file_pattern"], accumulation=kw["accumulation"], accumulation_detected=None,
        daily_boundary=kw["daily_boundary"],
        dims=None if kw["model_dims"] is None else tuple(kw["model_dims"].items()), paths=None,
        units=rc.reference.units if kw["model"] == rc.reference.model else
        next((m.units for m in rc.models if m.name == kw["model"]), None))
    window = Window(*kw["verification_window"], kw["tolerance_days"])
    # a reference that is also one of the forecast models (same name) cannot be told apart, and is read the same way
    reference = kw["model"] == kw["ref_model"] and kw["model"] not in {m.name for m in rc.models}
    for_model = None
    if kw["case_name"] != Case(model, window).name:   # the ensemble flow names the reference's case after its model
        for_model = next((m.name for m in rc.models if Case(m, window).name == kw["case_name"]), None)
    case = Case(model, window, reference=reference, for_model=for_model)
    selection = Selection(**{name: kw[kwarg] for name, kwarg in SELECTION_KWARGS.items() if kwarg in kw})
    return rc, case, selection
