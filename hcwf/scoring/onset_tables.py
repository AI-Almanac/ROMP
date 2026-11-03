"""The onset tables of a run: per year and case, read the data, detect onsets and score them (`onset_metrics`, in
hcwf.metrics.contingency, is the pure part), cached across the cases that share them."""

from hcwf.cache import memo
from hcwf.config.model import Case, Selection
from hcwf.metrics.contingency import onset_metrics
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.io import thresholds as thresholds_io
from hcwf.io import warmup as warmup_io
from hcwf.io.forecasts import initialization_dates
from hcwf.spatial.onto_units import point_dims
from hcwf.onset.detect import check_alignment, detect_deterministic, detect_ensemble, detect_observed
from hcwf.onset.observed_cache import threshold_hash
from hcwf.references.climatology import climatological_onset, compute_climatology_as_forecast
from hcwf.references.climatology import climatological_onset_dataset
from hcwf.references.conditional import (
    CLIMATOLOGY_REFERENCES,
    CONDITIONAL_CLIMATOLOGY,
    compute_conditional_climatology_as_forecast,
)
from hcwf.parallel import map_year_args

import logging
from hcwf.log import progress

log = logging.getLogger(__name__)


_ONSET_TABLES = memo("scoring.onset_tables", maxsize=1)
"""The onset tables of the source being scored: {family key: {year: (onset table, observed onsets)}} (see onset_tables);
one family is held, the next one's tables replace it."""


_SHARED = memo("scoring.onset_tables.shared")
"""Ensemble onset tables the skill pass already detected, for the spatial pass: {family key: {year: (onset table, observed
onsets)}}. onset_tables takes a family out of here (the first time it needs it), so each is held once."""


def reports_back_dated(onset, model_name):
    """Whether score outputs report back-dated forecast onsets: forecast_history "observed" in `onset` (the OnsetSetup).

    The climatology baselines (whose onsets come from no forecast; `model_name` says which source is scored) return
    False, so their outputs stay as they are.
    """
    if model_name in CLIMATOLOGY_REFERENCES:
        return False
    return onset.forecast_history == "observed"


def _family_key(rc, case, ens, thresh_da, years, select) -> tuple:
    """The onset tables a case needs: the run, the source, the thresholds (by content, so reloaded equal ones match), the
    years and the selection. Windows and tolerance are not in it: they only change the metrics."""
    return (bool(ens), threshold_hash(thresh_da), tuple(years), repr(case.model), repr(select), rc)


def _tables_of_years(years, rc, case, select, ens, extra):
    """The onset table and the observed onsets of each year in years: {year: (onset_df, onset_da)}.

    This is the part of the evaluation that reads and detects (the observed onsets, the forecast or climatology
    onsets paired with them); the metrics of a verification window are computed from it, so one table serves every
    window. The onset table has one row per (init, grid point) as detect_deterministic (or, for ensembles,
    detect_ensemble's ensemble table) gives. ens selects the ensemble variant of the evaluation (a climatology source
    then never reads a forecast). `extra` holds the climatology's onsets (see onset_tables).
    """
    model, ev = case.model, rc.evaluation
    ref_clim = (model.name == "climatology")
    ref_cond = (model.name == CONDITIONAL_CLIMATOLOGY)
    use_ensemble = (ev.probabilistic and not ref_clim) if ens else ev.probabilistic
    thresh_da = thresholds_io.thresholds_of(rc, model, select)
    out = {}
    for year in years:
        log.debug(f"\n{'-'*50}")
        log.debug(f"Processing year {year}")

        # obs onset
        imd = observations_io.observations_of(year, rc, select)
        onset_da = detect_observed(imd, thresh_da, year, rc, select)
        history = observations_io.history_of(year, imd, rc, select)
        # the forecast at the appropriate init dates: read inside the detection, which drops it once it is ingested
        read_forecast = None
        if ref_cond:
            pass
        elif use_ensemble or (not ref_clim and not ens):
            def read_forecast(year=year, imd=imd, ensemble=use_ensemble):
                """The year's forecast of the cells the masks keep, checked against the observations (once per run)."""
                log.debug("Extracting ensemble forecast data ..." if ensemble else "Extracting model forecast data ...")
                cube = forecasts_io.forecast_of(year, rc, model, ensemble=ensemble, select=select, cells=True)
                check_alignment(imd, cube, rc, case)
                return cube

        # detect onset dates
        if ref_cond:  # conditional climatology as forecast
            log.debug("Computing onset for conditional climatology as forecast ...")
            onset_df = compute_conditional_climatology_as_forecast(
                extra["clim_onset"], year, initialization_dates(year, ev), onset_da,
                max_forecast_day=ev.max_forecast_day, climatology_cv=rc.reference.climatology_cv)
        elif use_ensemble:  # ensemble forecast onset
            log.debug("Computing onset for ensemble forecast ...")
            _, onset_df = detect_ensemble(read_forecast, thresh_da, onset_da, rc, case, select, history=history)
        elif not ref_clim:  # deterministic model onset
            log.debug("Computing onset for model forecast ...")
            onset_df = detect_deterministic(read_forecast, thresh_da, onset_da, rc, case, select, history=history)
        else:  # climatology as forecast
            log.debug("Computing onset for climatology as forecast ...")
            onset_df = compute_climatology_as_forecast(
                extra["climatological_onset_doy"], year, initialization_dates(year, ev), onset_da,
                max_forecast_day=ev.max_forecast_day)
        del read_forecast, imd, history
        out[year] = (onset_df, onset_da)
    return out


def share_ensemble_tables(rc, model, tables, select=Selection()):
    """Hand the ensemble onset tables a pass over the run's years has already made to `onset_tables`, so the spatial
    metrics of `model` (ModelSpec) read and detect nothing again.

    `tables` is {year: (detect_ensemble's ensemble table, observed onsets)} made with this run's thresholds, history and
    members, as _tables_of_years makes them. Only a set that covers every year of the run is kept (a year skipped
    for a missing file is left to onset_tables, which reports it).
    """
    years = tuple(rc.evaluation.years)
    if set(tables) != set(years):
        return
    case = Case(model, rc.evaluation.windows[0])
    thresh_da = thresholds_io.thresholds_of(rc, model, select)
    _SHARED[_family_key(rc, case, True, thresh_da, years, select)] = tables


def onset_tables(rc, case, select=Selection(), *, ens=False, years=None):
    """The onset tables of the years (default the run's), ({year: (onset_df, onset_da)}, extra), detected once for all
    verification windows.

    The first case of a source (model, climatology, ...) detects every year, in worker processes unless
    ``parallel`` is off; the cases of its other verification windows get the same tables back (they differ only in the
    metrics). extra holds the climatological onset (a climatology source) and the climatology years' onsets
    (conditional climatology). Only the source being scored is kept: a new source drops the previous one's tables.
    """
    years = rc.evaluation.years if years is None else years
    thresh_da = thresholds_io.thresholds_of(rc, case.model, select)
    key = _family_key(rc, case, ens, thresh_da, years, select)
    if key not in _ONSET_TABLES:
        _ONSET_TABLES.clear()
        extra = {}
        tables = _SHARED.pop(key, None)   # the skill pass detected these years already
        if tables is None:
            if case.model.name == "climatology":
                log.debug("Generating climatology onset doy for each year as onset_da_dict")
                extra["climatological_onset_doy"] = climatological_onset(rc, case, select)
            elif case.model.name == CONDITIONAL_CLIMATOLOGY:
                log.debug("Computing observed onsets of the climatology years for the conditional climatology")
                extra["clim_onset"] = climatological_onset_dataset(rc, case, select)
            reads = case.model.name not in (CONDITIONAL_CLIMATOLOGY, "climatology") and len(years) > 1
            chunks = map_year_args(_tables_of_years, years, (rc, case, select, ens, extra), rc.parallel, skip=None,
                                   prewarm=lambda: warmup_io.warm_up_of(rc, case.model, select),
                                   item_bytes=forecasts_io.forecast_year_bytes(
                                       years[0], rc, case.model, ensemble=bool(rc.evaluation.probabilistic),
                                       select=select) if reads else None)
            tables = {}
            for chunk in chunks:
                tables.update(chunk)
        _ONSET_TABLES[key] = {"tables": tables, "extra": extra, "pin": thresh_da}
    held = _ONSET_TABLES[key]
    wanted = set(years)
    return ({year: (df, da.copy(deep=True)) for year, (df, da) in held["tables"].items() if year in wanted},
            held["extra"])


def _metrics_of_tables(tables, extra, rc, case, years=None):
    """The per-year metrics tables and observed onsets of one verification window, from the shared onset tables."""
    years = rc.evaluation.years if years is None else years
    metrics_df_dict = {}
    onset_da_dict = {}
    ref_clim = (case.model.name == "climatology")
    back_dated = reports_back_dated(rc.onset, case.model.name)
    dims = point_dims(rc.observations)
    if ref_clim:
        # climatological onset as "obs" for climatology baseline metrics
        climatological_onset_doy = extra["climatological_onset_doy"].copy(deep=True)
        onset_da_dict = {year: climatological_onset_doy for year in years}
    for year in progress(years, desc=f"{case.model.name} {case.window.span} years"):
        onset_df, onset_da = tables[year]
        # onset-obs metrics TP,TN,FP,FN for all locations and init times
        log.debug("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        metrics_df, summary_stats = onset_metrics(onset_df, case.window, back_dated=back_dated, default_dims=dims)
        metrics_df_dict[year] = metrics_df
        if not ref_clim:
            onset_da_dict[year] = onset_da
        log.debug(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        log.debug(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
                  f"FP={summary_stats['overall_false_positive']}, "
                  f"FN={summary_stats['overall_false_negative']}, "
                  f"TN={summary_stats['overall_true_negative']}")
    return metrics_df_dict, onset_da_dict


def compute_metrics(rc, case, select=Selection()):
    """Compute onset metrics for the case over the run's years (the onset dates are detected once per source, see
    onset_tables): ({year: metrics table}, {year: observed onsets}).

    Probabilistic runs are evaluated from the ensemble onset (the member-agreement onset of detect_ensemble),
    deterministic ones from the model's own onset.
    """
    tables, extra = onset_tables(rc, case, select, ens=bool(rc.evaluation.probabilistic))
    return _metrics_of_tables(tables, extra, rc, case)

