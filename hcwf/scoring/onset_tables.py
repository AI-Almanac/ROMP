"""The onset tables of a run: per year and case, read the data, detect onsets and score them.

`hcwf.metrics.contingency.onset_metrics` is the pure part of the scoring. The tables are cached across the cases that
share them (`onset_tables`), and the skill pass of `hcwf.scoring.pairs` can hand its ensemble tables over
(`share_ensemble_tables`) so that the spatial pass of `hcwf.scoring.deterministic` detects nothing twice.
"""

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
from hcwf.config.model import CLIMATOLOGY, CLIMATOLOGY_REFERENCES, CONDITIONAL_CLIMATOLOGY
from hcwf.references.conditional import compute_conditional_climatology_as_forecast
from hcwf.parallel import map_year_args

import logging
from hcwf.log import progress

log = logging.getLogger(__name__)


_ONSET_TABLES = memo("scoring.onset_tables", maxsize=1)
"""The onset tables of the source being scored: ``{family key: {"tables": ..., "extra": ..., "pin": ...}}`` (see
`onset_tables`); one family is held, the next one's tables replace it."""


_SHARED = memo("scoring.onset_tables.shared")
"""Ensemble onset tables the skill pass already detected, for the spatial pass: ``{family key: {year: (onset table,
observed onsets)}}``. `onset_tables` takes a family out of here (the first time it needs it), so each is held once."""


def reports_back_dated(onset, model_name):
    """Return whether score outputs report back-dated forecast onsets (``forecast_history: observed``).

    The climatology baselines (whose onsets come from no forecast) return False, so their outputs stay as they are.

    Parameters
    ----------
    onset : hcwf.config.onset.OnsetSetup
        The onset setup of the run; its ``forecast_history`` is read.
    model_name : str
        The source being scored.

    Returns
    -------
    bool
        True when ``model_name`` is not a climatology reference and ``forecast_history`` is ``"observed"``.
    """
    if model_name in CLIMATOLOGY_REFERENCES:
        return False
    return onset.forecast_history == "observed"


def _family_key(rc, case, ens, thresh_da, years, select) -> tuple:
    """Return the key of the onset tables a case needs.

    The key holds the run, the source, the thresholds (by content, so reloaded equal ones match), the years and the
    selection. Windows and tolerance are not in it: they only change the metrics.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case whose source is scored.
    ens : bool
        Whether the ensemble variant of the evaluation is meant.
    thresh_da : xarray.DataArray
        The thresholds of the source.
    years : iterable of int
        The years the tables cover.
    select : hcwf.config.model.Selection
        The selection of the cells.

    Returns
    -------
    tuple
        The hashable key.
    """
    return (bool(ens), threshold_hash(thresh_da), tuple(years), repr(case.model), repr(select), rc)


def _tables_of_years(years, rc, case, select, ens, extra):
    """Return the onset table and the observed onsets of each year.

    This is the part of the evaluation that reads and detects (the observed onsets, the forecast or climatology
    onsets paired with them); the metrics of a verification window are computed from it, so one table serves every
    window. The onset table has one row per (init, grid point) as `hcwf.onset.detect.detect_deterministic` gives (or,
    for ensembles, the ensemble table of `hcwf.onset.detect.detect_ensemble`). It is the function that the worker
    processes run for a chunk of years.

    Parameters
    ----------
    years : iterable of int
        The years to process.
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case whose source is read.
    select : hcwf.config.model.Selection
        The selection of the cells.
    ens : bool
        Select the ensemble variant of the evaluation (a climatology source then never reads a forecast).
    extra : dict
        The climatology's onsets: ``climatological_onset_doy`` (climatology source) or ``clim_onset`` (conditional
        climatology), see `onset_tables`.

    Returns
    -------
    dict of int to tuple of (pandas.DataFrame, xarray.DataArray)
        Maps each year to ``(onset_df, onset_da)``: the onset table and the observed onsets.
    """
    model, ev = case.model, rc.evaluation
    ref_clim = (model.name == CLIMATOLOGY)
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
                """Read the year's forecast of the cells the masks keep and check it against the observations."""
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
                extra["climatological_onset_doy"], initialization_dates(year, ev), onset_da,
                max_forecast_day=ev.max_forecast_day)
        del read_forecast, imd, history
        out[year] = (onset_df, onset_da)
    return out


def share_ensemble_tables(rc, model, tables, select=Selection()):
    """Hand the ensemble onset tables that a pass over the run's years has made to `onset_tables`.

    The spatial metrics of ``model`` then read and detect nothing again. ``tables`` are made with this run's thresholds,
    history and members, as `_tables_of_years` makes them. Only a set that covers every year of the run is kept (a year
    skipped for a missing file is left to `onset_tables`, which reports it).

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The source the tables belong to.
    tables : dict of int to tuple
        Maps each year to ``(ensemble table of detect_ensemble, observed onsets)``.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.

    Returns
    -------
    None
        Nothing is returned; the tables are stored for `onset_tables`.
    """
    years = tuple(rc.evaluation.years)
    if set(tables) != set(years):
        return
    case = Case(model, rc.evaluation.windows[0])
    thresh_da = thresholds_io.thresholds_of(rc, model, select)
    _SHARED[_family_key(rc, case, True, thresh_da, years, select)] = tables


def onset_tables(rc, case, select=Selection(), *, ens=False, years=None):
    """Return the onset tables of the years, detected once for all verification windows.

    The first case of a source (model, climatology, ...) detects every year, in worker processes unless
    ``parallel`` is off; the cases of its other verification windows get the same tables back (they differ only in the
    metrics). Only the source being scored is kept: a new source drops the previous one's tables. The returned
    observed onsets are deep copies, so the caller may change them.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case whose source is scored.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.
    ens : bool, default False
        Whether the ensemble variant of the evaluation is meant.
    years : iterable of int, optional
        The years to return; the run's years when None.

    Returns
    -------
    tables : dict of int to tuple of (pandas.DataFrame, xarray.DataArray)
        Maps each year to ``(onset_df, onset_da)``.
    extra : dict
        The climatological onset (``climatological_onset_doy``, a climatology source) or the climatology years' onsets
        (``clim_onset``, conditional climatology); empty otherwise.
    """
    years = rc.evaluation.years if years is None else years
    thresh_da = thresholds_io.thresholds_of(rc, case.model, select)
    key = _family_key(rc, case, ens, thresh_da, years, select)
    if key not in _ONSET_TABLES:
        _ONSET_TABLES.clear()
        extra = {}
        tables = _SHARED.pop(key, None)   # the skill pass detected these years already
        if tables is None:
            if case.model.name == CLIMATOLOGY:
                log.debug("Generating climatology onset doy for each year as onset_da_dict")
                extra["climatological_onset_doy"] = climatological_onset(rc, case, select)
            elif case.model.name == CONDITIONAL_CLIMATOLOGY:
                log.debug("Computing observed onsets of the climatology years for the conditional climatology")
                extra["clim_onset"] = climatological_onset_dataset(rc, case, select)
            reads = case.model.name not in CLIMATOLOGY_REFERENCES and len(years) > 1
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
    """Return the per-year metrics tables and observed onsets of one verification window.

    Parameters
    ----------
    tables : dict of int to tuple of (pandas.DataFrame, xarray.DataArray)
        The shared onset tables (`onset_tables`).
    extra : dict
        The climatological onsets (`onset_tables`); a climatology source uses them as its observed onsets.
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case, which carries the window.
    years : iterable of int, optional
        The years to score; the run's years when None.

    Returns
    -------
    metrics_df_dict : dict of int to pandas.DataFrame
        Maps each year to its metrics table (`hcwf.metrics.contingency.onset_metrics`).
    onset_da_dict : dict of int to xarray.DataArray
        Maps each year to its observed onsets (the climatological onset for a climatology source).
    """
    years = rc.evaluation.years if years is None else years
    metrics_df_dict = {}
    onset_da_dict = {}
    ref_clim = (case.model.name == CLIMATOLOGY)
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
    """Compute the onset metrics of the case over the run's years.

    The onset dates are detected once per source (see `onset_tables`). Probabilistic runs are evaluated from the
    ensemble onset (the member-agreement onset of `hcwf.onset.detect.detect_ensemble`), deterministic ones from the
    model's own onset.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case to score.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.

    Returns
    -------
    metrics_df_dict : dict of int to pandas.DataFrame
        Maps each year to its metrics table.
    onset_da_dict : dict of int to xarray.DataArray
        Maps each year to its observed onsets.
    """
    tables, extra = onset_tables(rc, case, select, ens=bool(rc.evaluation.probabilistic))
    return _metrics_of_tables(tables, extra, rc, case)

