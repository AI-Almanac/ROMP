"""Onset contingency metrics (TP/FP/FN/TN, MAE) for model or climatology forecasts across years."""

from hcwf.config.model import Selection
from hcwf.io import input as io_input
from hcwf.io.input import initialization_dates, point_dims
from hcwf.stats.detect import check_alignment, detect_deterministic, detect_ensemble, detect_observed
from hcwf.stats.observed_cache import threshold_hash
from hcwf.stats.climatology import climatological_onset, compute_climatology_as_forecast
from hcwf.stats.bins import reports_back_dated
from hcwf.stats.climatology import climatological_onset_dataset
from hcwf.stats.conditional_climatology import CONDITIONAL_CLIMATOLOGY, compute_conditional_climatology_as_forecast
from hcwf.spatial import axes
from hcwf.stats.parallel import map_year_args

import numpy as np
import xarray as xr
import pandas as pd
import logging
from hcwf.utils.log import progress

log = logging.getLogger(__name__)


def onset_metrics(onset_df, window, *, back_dated, default_dims):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows.

    Returns one row per grid point (or evaluation unit, with a unit column instead of lat and lon), in order of first
    appearance in onset_df, and summary totals.
    With forecast_history "observed" (`back_dated`, see bins.reports_back_dated) the rows also have `n_back_dated`: the
    cases whose forecast onset is dated on or before the init, which count as the first lead; 0 unless the window
    starts at lead 1. "none" runs get no such column. `window` is the Window (span and tolerance); `default_dims` are the
    spatial dims of a table with no rows (point_dims).
    """
    report_back_dated = back_dated
    tolerance_days, verification_window = window.tolerance, window.span
    log.debug(f"Computing onset metrics with tolerance = {tolerance_days} days")
    log.debug(f"Verification window starts {verification_window} days after initialization")
    log.debug(f"Forecast window length: {verification_window[1]} days")

    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]
    points = list(axes.spatial_dims(onset_df.columns) if len(onset_df.columns) else default_dims)
    columns = [*points, 'total_forecasts', 'true_positive', 'true_negative', 'false_positive', 'false_negative',
               'num_onset', 'num_no_onset', 'mae_combined', 'mae_tp_only', 'num_tp_errors', 'num_fp_errors',
               'tolerance_days', 'verification_window', 'forecast_days']

    if onset_df.empty:
        # Small domains (e.g. a user focus area) can have no forecast onsets in a year.
        onset_df = pd.DataFrame(columns=points)
    unique_locations = onset_df[points].drop_duplicates()

    log.debug(f"Processing {len(unique_locations)} unique grid points...")

    if len(onset_df):
        day = np.timedelta64(1, "D")
        code = onset_df.groupby(points, sort=False).ngroup().to_numpy()
        n_points = len(unique_locations)
        first_row = np.unique(code, return_index=True)[1]

        obs_onset = pd.to_datetime(onset_df['obs_onset_date']).to_numpy().astype("datetime64[ns]")
        model_onset = pd.to_datetime(onset_df['onset_date']).to_numpy().astype("datetime64[ns]")
        t_init = pd.to_datetime(onset_df['init_time']).to_numpy().astype("datetime64[ns]")
        # every row of a grid point shares its observed onset
        gt_grd = obs_onset[first_row][code]

        valid_window_start = t_init + forecast_bin_start * day
        valid_window_end = t_init + forecast_bin_end * day
        is_obs_onset_in_whole_window = (t_init + day <= gt_grd) & (gt_grd <= valid_window_end)

        has_model_onset = ~np.isnat(model_onset)
        # A forecast onset on or before the init (back-dated by the observed history) is
        # judged as if it fell on the first lead: it can be a hit or a false alarm in a
        # window starting at lead 1 and is ignored, like any early onset, in later windows.
        model_onset_in_window = np.where(has_model_onset & (model_onset <= t_init), t_init + day, model_onset)

        in_valid_window = (has_model_onset & (valid_window_start <= model_onset_in_window)
                           & (model_onset_in_window <= valid_window_end))
        with np.errstate(invalid="ignore"):
            abs_diff_days = np.abs((model_onset - gt_grd) // day)
        true_positive = in_valid_window & (abs_diff_days <= tolerance_days)
        false_positive = in_valid_window & ~true_positive
        past_window = ~has_model_onset | (~in_valid_window & (model_onset_in_window > valid_window_end))
        false_negative = past_window & is_obs_onset_in_whole_window
        true_negative = past_window & ~is_obs_onset_in_whole_window

        def count(mask):
            """Count the cases in each group where the mask holds."""
            return np.bincount(code, weights=mask, minlength=n_points).astype(np.int64)

        def error_sum(mask):
            """Sum the absolute onset error in days over the masked cases in each group."""
            return np.bincount(code, weights=np.where(mask, abs_diff_days, 0), minlength=n_points)

        tp, fp = count(true_positive), count(false_positive)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = np.where(tp + fp > 0, (error_sum(true_positive) + error_sum(false_positive)) / (tp + fp), np.nan)
            mae_tp_only = np.where(tp > 0, error_sum(true_positive) / tp, np.nan)
        back_dated = has_model_onset & (model_onset <= t_init)
        metrics_df = pd.DataFrame({
            **{point: unique_locations[point].to_numpy() for point in points},
            'total_forecasts': np.bincount(code, minlength=n_points),
            'true_positive': tp,
            'true_negative': count(true_negative),
            'false_positive': fp,
            'false_negative': count(false_negative),
            'num_onset': count(is_obs_onset_in_whole_window),
            'num_no_onset': count(~is_obs_onset_in_whole_window),
            'mae_combined': mae,
            'mae_tp_only': mae_tp_only,
            'num_tp_errors': tp,
            'num_fp_errors': fp,
            'tolerance_days': tolerance_days,
            'verification_window': [verification_window] * n_points,
            'forecast_days': forecast_bin_end,
        })
        if report_back_dated:
            metrics_df['n_back_dated'] = count(back_dated) if forecast_bin_start == 1 else np.zeros(n_points, np.int64)
    else:
        metrics_df = pd.DataFrame(columns=columns[:len(points) + 9] + (['n_back_dated'] if report_back_dated else []))

    summary_stats = {
        'total_grid_points': len(metrics_df),
        'total_forecasts': metrics_df['total_forecasts'].sum(),
        'overall_true_positive': metrics_df['true_positive'].sum(),
        'overall_true_negative': metrics_df['true_negative'].sum(),
        'overall_false_positive': metrics_df['false_positive'].sum(),
        'overall_false_negative': metrics_df['false_negative'].sum(),
        'overall_num_onset': metrics_df['num_onset'].sum(),
        'overall_num_no_onset': metrics_df['num_no_onset'].sum(),
        'overall_mae_combined': metrics_df['mae_combined'].mean(),
        'overall_mae_tp_only': metrics_df['mae_tp_only'].mean(),
        'tolerance_days': tolerance_days,
        'verification_window': verification_window,
        'forecast_days': forecast_bin_end
    }

    return metrics_df, summary_stats


_ONSET_TABLES: dict = {}
"""The onset tables of the source being scored: {family key: {year: (onset table, observed onsets)}} (see onset_tables)."""


def reset_onset_tables() -> None:
    """Forget the shared onset tables (a new run starts)."""
    _ONSET_TABLES.clear()


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
    thresh_da = io_input.thresholds_of(rc, model, select)
    out = {}
    for year in years:
        log.debug(f"\n{'-'*50}")
        log.debug(f"Processing year {year}")

        # obs onset
        imd = io_input.observations_of(year, rc, select)
        onset_da = detect_observed(imd, thresh_da, year, rc, select)
        history = io_input.history_of(year, imd, rc, select)
        # extract forecast at approporiate init dates
        p_model = None
        if ref_cond:
            pass
        elif use_ensemble:
            log.debug("Extracting ensemble forecast data ...")
            p_model = io_input.forecast_of(year, rc, model, ensemble=True, select=select)
        elif not ref_clim and not ens:
            log.debug("Extracting model forecast data ...")
            p_model = io_input.forecast_of(year, rc, model, ensemble=False, select=select)

        if p_model is not None and (use_ensemble or not ref_clim):
            check_alignment(imd, p_model, rc, case)

        # detect onset dates
        if ref_cond:  # conditional climatology as forecast
            log.debug("Computing onset for conditional climatology as forecast ...")
            onset_df = compute_conditional_climatology_as_forecast(
                extra["clim_onset"], year, initialization_dates(year, ev), onset_da,
                max_forecast_day=ev.max_forecast_day, climatology_cv=rc.reference.climatology_cv)
        elif use_ensemble:  # ensemble forecast onset
            log.debug("Computing onset for ensemble forecast ...")
            _, onset_df = detect_ensemble(p_model, thresh_da, onset_da, rc, case, select, history=history)
        elif not ref_clim:  # deterministic model onset
            log.debug("Computing onset for model forecast ...")
            onset_df = detect_deterministic(p_model, thresh_da, onset_da, rc, case, select, history=history)
        else:  # climatology as forecast
            log.debug("Computing onset for climatology as forecast ...")
            onset_df = compute_climatology_as_forecast(
                extra["climatological_onset_doy"], year, initialization_dates(year, ev), onset_da,
                max_forecast_day=ev.max_forecast_day)
        del p_model, imd, history
        out[year] = (onset_df, onset_da)
    return out


def onset_tables(rc, case, select=Selection(), *, ens=False, years=None):
    """The onset tables of the years (default the run's), ({year: (onset_df, onset_da)}, extra), detected once for all
    verification windows.

    The first case of a source (model, climatology, ...) detects every year, in worker processes unless
    ``parallel`` is off; the cases of its other verification windows get the same tables back (they differ only in the
    metrics). extra holds the climatological onset (a climatology source) and the climatology years' onsets
    (conditional climatology). Only the source being scored is kept: a new source drops the previous one's tables.
    """
    years = rc.evaluation.years if years is None else years
    thresh_da = io_input.thresholds_of(rc, case.model, select)
    key = _family_key(rc, case, ens, thresh_da, years, select)
    if key not in _ONSET_TABLES:
        _ONSET_TABLES.clear()
        extra = {}
        if case.model.name == "climatology":
            log.debug("Generating climatology onset doy for each year as onset_da_dict")
            extra["climatological_onset_doy"] = climatological_onset(rc, case, select)
        elif case.model.name == CONDITIONAL_CLIMATOLOGY:
            log.debug("Computing observed onsets of the climatology years for the conditional climatology")
            extra["clim_onset"] = climatological_onset_dataset(rc, case, select)
        chunks = map_year_args(_tables_of_years, years, (rc, case, select, ens, extra), rc.parallel, skip=None,
                               prewarm=lambda: io_input.warm_up_of(rc, case.model, select))
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

