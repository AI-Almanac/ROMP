"""The per-model pair tables of a run: read the years, detect onsets, and pair them into day bins.

The pure pairing (`forecast_pairs`, `LeadBins`) and the scores are in hcwf.metrics; this module reads and detects.
"""
from __future__ import annotations

import logging
from typing import Any, NamedTuple

import pandas as pd

from hcwf.config.model import Case, Selection
from hcwf.io import forecasts as forecasts_io
from hcwf.io import observations as observations_io
from hcwf.io import reference as reference_io
from hcwf.io import thresholds as thresholds_io
from hcwf.io import warmup as warmup_io
from hcwf.io.forecasts import initialization_dates
from hcwf.metrics.bins import create_climatological_forecast_obs_pairs, forecast_pairs
from hcwf.onset.detect import check_alignment, detect_ensemble, detect_observed
from hcwf.parallel import NO_DATA, Pending, concat_year_args, spec_workers, start_year_args
from hcwf.references.climatology import climatological_onset_dataset
from hcwf.references.conditional import CLIMATOLOGY_REFERENCES, conditional_pairs, is_conditional_climatology
from hcwf.scoring.onset_tables import reports_back_dated, share_ensemble_tables

log = logging.getLogger(__name__)


def _case_of(rc, model):
    """The case that reads `model` in the run's first window (the pair tables do not depend on the window)."""
    return Case(model, rc.evaluation.windows[0])


def forecast_obs_pairs(rc, model, years, select=Selection()):
    """(pairs, tables) of the ensemble forecast `model` (ModelSpec) over `years`, in this process: the forecast-observation
    pairs of the years concatenated, and {year: (ensemble onset table, observed onsets)} for them.

    The run `rc` carries the observations, onset definition, day bins and everything else the pairs depend on. Each year
    is read and detected once: the ensemble onset table of its detection is what the spatial metrics are made from
    (see onset_tables.share_ensemble_tables). A year whose model or observation file is missing is skipped;
    ValueError("No data ...") when no year yields data.
    """
    case = _case_of(rc, model)
    log.debug(f"Processing years: {list(years)}")

    thresh_slice = thresholds_io.thresholds_of(rc, model, select)

    # Initialize list to store all forecast-observation pairs
    all_forecast_obs_pairs = []
    tables = {}

    # Process each year
    for year in years:
        log.debug(f"\n{'-'*50}")
        log.debug(f"Processing year {year}")
        # let go of the previous year's data before this year's is read (peak memory: one year, not two)
        rainfall_ds = read_forecast = onset_all_members = ensemble_table = onset_da = None

        try:
            log.debug("Loading observational rainfall data...")
            rainfall_ds = observations_io.observations_of(year, rc, select)

            log.debug("Detecting observed onset...")
            onset_da = detect_observed(rainfall_ds, thresh_slice, year, rc, select)
            log.debug(f"Found onset in {(~pd.isna(onset_da.values)).sum()} out of {onset_da.size} grid points")

            def read_forecast(observed=rainfall_ds):
                """The year's forecast of the cells the masks keep, checked against the observations (once per run)."""
                log.debug("Loading S2S model data...")
                cube = forecasts_io.forecast_of(year, rc, model, ensemble=True, select=select, cells=True)
                check_alignment(observed, cube, rc, case)
                return cube

            log.debug("Computing onset for all ensemble members...")
            onset_all_members, ensemble_table = detect_ensemble(
                read_forecast, thresh_slice, onset_da, rc, case, select,
                history=observations_io.history_of(year, rainfall_ds, rc, select))
            log.debug(f"Found onset in {onset_all_members['onset_day'].notna().sum()} member cases")

            log.debug("Creating forecast-observation pairs...")
            forecast_obs = forecast_pairs(onset_all_members, onset_da, rc.evaluation.day_bins,
                                          back_dated=reports_back_dated(rc.onset, model.name))

            # Add to master list
            all_forecast_obs_pairs.append(forecast_obs)
            tables[year] = (ensemble_table, onset_da)
            del rainfall_ds, read_forecast, onset_all_members, ensemble_table, onset_da

            log.debug(f"Year {year} completed: {len(forecast_obs)} forecast-observation pairs")

        except FileNotFoundError as e:
            # Reforecast archives commonly have year gaps; skip years whose
            # model or observation file is absent instead of aborting the run.
            log.debug(f"Skipping year {year} (missing input file): {e}")
            continue

    # Combine all years
    log.debug(f"\n{'-'*50}")
    log.debug("Combining all years")

    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    # Print final summary statistics
    log.debug(f"\nFinal Summary Statistics:")
    log.debug(f"Years processed: {list(years)}")
    return combined_forecast_obs, tables


def _pairs_of_years(years, rc, model, select):
    """forecast_obs_pairs with the years first (the form `parallel.start_year_args` calls)."""
    return forecast_obs_pairs(rc, model, years, select)


def start_pairs(rc, model, select=Selection()):
    """Start the forecast-observation pairs of `model` over the run's years; the Pending's result() waits for them.

    The thresholds and masks every year needs are read here first (warmup_io.warm_up_of); with workers, the years run in
    them while this process goes on, and each worker gets only `(rc, model, select)`. The ensemble onset tables of the
    years that come back with the pairs are handed to the spatial metrics (share_ensemble_tables).
    """
    years = list(rc.evaluation.years)
    pending = start_year_args(_pairs_of_years, years, (rc, model, select), rc.parallel,
                              prewarm=lambda: warmup_io.warm_up_of(rc, model, select),
                              item_bytes=forecasts_io.forecast_year_bytes(years[0], rc, model, ensemble=True,
                                                                          select=select) if len(years) > 1 else None)

    def join():
        """The year pairs, concatenated in year order; a year without data is left out, none at all is an error."""
        parts = [part for part in pending.result() if part is not None]
        if not parts:
            raise ValueError(f"{NO_DATA} for any year")
        tables = {}
        for _, year_tables in parts:
            tables.update(year_tables)
        share_ensemble_tables(rc, model, tables, select)
        return pd.concat([pairs for pairs, _ in parts], ignore_index=True)

    return Pending(join)


def climatological_pairs(clim_onset, rc, *, years, day_bins):
    """Climatological forecast-observation pairs of all target years in years, concatenated.

    clim_onset is the [year, lat, lon] array of the climatology years' onsets; the run `rc` carries the init dates.
    A year that fails is logged and left out; ValueError when none gives pairs.
    """
    clim_onset_slice = clim_onset #!!!!! region subset

    all_forecast_obs_pairs = []

    for target_year in years:
        log.debug(f"\n{'-'*50}")
        log.debug(f"Processing target year {target_year}")

        try:
            # Get initialization dates for this year
            init_dates = initialization_dates(target_year, rc.evaluation)

            # Create forecast-observation pairs for this year
            forecast_obs = create_climatological_forecast_obs_pairs(
                clim_onset_slice,
                target_year,
                init_dates,
                day_bins=day_bins,
                max_forecast_day=rc.evaluation.max_forecast_day
            )

            if len(forecast_obs) > 0:
                all_forecast_obs_pairs.append(forecast_obs)
                log.debug(f"Target year {target_year} completed: {len(forecast_obs)} pairs")
            else:
                log.debug(f"No pairs generated for target year {target_year}")

        except Exception as e:
            log.error(f"Error processing target year {target_year}: {e}")
            continue

    # Combine all years
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any target year")

    combined_forecast_obs = pd.concat(all_forecast_obs_pairs, ignore_index=True)

    log.debug(f"\n{'='*50}")
    log.debug("CLIMATOLOGICAL FORECAST SUMMARY")
    log.debug(f"{'='*50}")
    log.debug(f"Target years processed: {list(years)}")
    log.debug(f"Total forecast-observation pairs: {len(combined_forecast_obs)}")
    log.debug(f"Probability range: {combined_forecast_obs['predicted_prob'].min():.3f} - {combined_forecast_obs['predicted_prob'].max():.3f}")
    log.debug(f"Overall observed onset rate: {combined_forecast_obs['observed_onset'].mean():.3f}")

    return combined_forecast_obs


class PreparedPairs(NamedTuple):
    """The pair tables a model's scores are made from, built once per model (`prepare_pairs`)."""

    forecast: pd.DataFrame
    """The model's forecast-observation pairs, every day bin."""
    reference: pd.DataFrame | None
    """The reference model's pairs, from the reference model's own data; None for a climatology reference."""
    clim_onset: Any
    """The climatology years' onsets (a climatology reference), else None."""
    climatology: pd.DataFrame | None
    """The climatology reference's pairs (conditional or not); None for a model reference."""


def _climatology_years(years, clim_onset, rc, case, select, day_bins):
    """The climatology reference's pair tables of `years` (the form `parallel.concat_year_args` calls)."""
    if is_conditional_climatology(rc.reference.model):
        return conditional_pairs(clim_onset, rc, case, select, years=years, day_bins=day_bins)
    return climatological_pairs(clim_onset, rc, years=years, day_bins=day_bins)


def _climatology_pairs(clim_onset, rc, case, select):
    """Pair tables of the climatology reference (conditional or not) for all years, in worker processes unless parallel is off."""
    return concat_year_args(_climatology_years, rc.evaluation.years,
                            (clim_onset, rc, case, select, rc.evaluation.day_bins), rc.parallel)


def _reference_model(rc, model):
    """The reference as a source to read when `model` is the forecast model it is scored against (its own settings,
    written once in RunConfig.reference_cases)."""
    return rc.reference_cases(for_model=model)[0].model


def prepare_pairs(rc, model, select=Selection()):
    """Build the forecast-observation pair tables (and the reference's) of `model` (ModelSpec) shared across its cases.

    Returns PreparedPairs: the model's pairs, the reference model's own pairs (None for climatology and
    conditional_climatology), the climatological onsets and the climatology pairs; create_score_results reuses them.

    With worker processes the model's years are sent off first and the climatological onsets are computed here
    while they run; the climatology pairs then go to the same workers. Without them the years run inside
    `start_pairs`, so the reference years' rain (held from the thresholds to the climatological onsets) is used and
    freed before they are read, not kept alongside them.
    """
    case = Case(model, rc.evaluation.windows[0])
    climatology = rc.reference.model in CLIMATOLOGY_REFERENCES
    serial = spec_workers(rc.parallel, len(rc.evaluation.years)) <= 1

    clim_onset = None
    climatology_obs_df_all = None
    ref_pending = None

    if serial and climatology:
        clim_onset = climatological_onset_dataset(rc, case, select)
        reference_io.release_reference()
    elif serial:
        thresholds_io.thresholds_of(rc, model, select)
        reference_io.release_reference()

    pending = start_pairs(rc, model, select)

    if not climatology:
        ref_pending = start_pairs(rc, _reference_model(rc, model), select)
        reference_io.release_reference()
    elif clim_onset is None:
        clim_onset = climatological_onset_dataset(rc, case, select)
        reference_io.release_reference()

    forecast_obs_df_all = pending.result()
    ref_obs_df_all = ref_pending.result() if ref_pending is not None else None

    if climatology:
        climatology_obs_df_all = _climatology_pairs(clim_onset, rc, case, select)

    return PreparedPairs(forecast_obs_df_all, ref_obs_df_all, clim_onset, climatology_obs_df_all)
