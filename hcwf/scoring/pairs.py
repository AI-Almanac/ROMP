"""The per-model pairs (PairArrays) of a run: read the years, detect onsets, and pair them into day bins.

The pure pairing (`hcwf.metrics.bins.forecast_pairs`, `hcwf.metrics.bins.LeadBins`) and the scores are in
`hcwf.metrics`; this module reads and detects. `prepare_pairs` is the entry point: `hcwf.scoring.probabilistic` calls
it once per model and `hcwf.metrics.skill` scores what it returns.
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
from hcwf.metrics.bins import PairArrays, create_climatological_forecast_obs_pairs, forecast_pairs
from hcwf.onset.detect import check_alignment, detect_ensemble, detect_observed
from hcwf.parallel import NO_DATA, Pending, concat_year_args, spec_workers, start_year_args
from hcwf.references.climatology import climatological_onset_dataset
from hcwf.references.conditional import CLIMATOLOGY_REFERENCES, conditional_pairs, is_conditional_climatology
from hcwf.scoring.onset_tables import reports_back_dated, share_ensemble_tables

log = logging.getLogger(__name__)


def _case_of(rc, model):
    """Return the case that reads `model` in the run's first window (the pairs do not depend on the window).

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The source to read.

    Returns
    -------
    hcwf.config.model.Case
        The model with the first verification window.
    """
    return Case(model, rc.evaluation.windows[0])


def forecast_obs_pairs(rc, model, years, select=Selection()):
    """Return the forecast-observation pairs of an ensemble forecast over `years`, and the onset tables made on the way.

    Runs in this process. The run `rc` carries the observations, onset definition, day bins and everything else the pairs
    depend on. Each year is read and detected once: the ensemble onset table of its detection is what the spatial
    metrics are made from (see `hcwf.scoring.onset_tables.share_ensemble_tables`). A year whose model or observation file
    is missing is skipped.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The ensemble forecast source.
    years : iterable of int
        The years to process.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.

    Returns
    -------
    pairs : hcwf.metrics.bins.PairArrays
        The forecast-observation pairs of the years, concatenated.
    tables : dict of int to tuple
        Maps each processed year to ``(ensemble onset table, observed onsets)``.

    Raises
    ------
    ValueError
        No year yields data.
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
                """Read the year's forecast of the cells the masks keep and check it against the observations."""
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

            log.debug(f"Year {year} completed: {forecast_obs.n_cases} forecast cases")

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

    combined_forecast_obs = PairArrays.concat(all_forecast_obs_pairs)

    # Print final summary statistics
    log.debug("\nFinal Summary Statistics:")
    log.debug(f"Years processed: {list(years)}")
    return combined_forecast_obs, tables


def _pairs_of_years(years, rc, model, select):
    """Call `forecast_obs_pairs` with the years first (the form `hcwf.parallel.start_year_args` calls).

    Parameters
    ----------
    years : iterable of int
        The years to process.
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The ensemble forecast source.
    select : hcwf.config.model.Selection
        The selection of the cells.

    Returns
    -------
    tuple
        The ``(pairs, tables)`` of `forecast_obs_pairs`.
    """
    return forecast_obs_pairs(rc, model, years, select)


def start_pairs(rc, model, select=Selection()):
    """Start the forecast-observation pairs of `model` over the run's years.

    The thresholds and masks every year needs are read here first (`hcwf.io.warmup.warm_up_of`); with workers, the
    years run in them while this process goes on, and each worker gets only ``(rc, model, select)``. The ensemble onset
    tables of the years that come back with the pairs are handed to the spatial metrics (`share_ensemble_tables`).

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The ensemble forecast source.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.

    Returns
    -------
    hcwf.parallel.Pending
        Its ``result()`` waits for the work and returns the `hcwf.metrics.bins.PairArrays` of all years (joined in
        year order; a year without data is left out).
    """
    years = list(rc.evaluation.years)
    pending = start_year_args(_pairs_of_years, years, (rc, model, select), rc.parallel,
                              prewarm=lambda: warmup_io.warm_up_of(rc, model, select),
                              item_bytes=forecasts_io.forecast_year_bytes(years[0], rc, model, ensemble=True,
                                                                          select=select) if len(years) > 1 else None)

    def join():
        """Concatenate the year pairs in year order, hand the onset tables on, and return the pairs.

        A year without data is left out; no data at all raises `ValueError`.
        """
        parts = [part for part in pending.result() if part is not None]
        if not parts:
            raise ValueError(f"{NO_DATA} for any year")
        tables = {}
        for _, year_tables in parts:
            tables.update(year_tables)
        share_ensemble_tables(rc, model, tables, select)
        return PairArrays.concat([pairs for pairs, _ in parts])

    return Pending(join)


def climatological_pairs(clim_onset, rc, *, years, day_bins):
    """Return the climatological forecast-observation pairs of all target years, concatenated.

    The run `rc` carries the init dates. A year that fails is logged and left out.

    Parameters
    ----------
    clim_onset : xarray.DataArray
        The onsets of the climatology years, dims (year, lat, lon) or (year, unit).
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    years : iterable of int
        The target years.
    day_bins : sequence of (int, int)
        The ``(first day, last day)`` bins within the forecast window.

    Returns
    -------
    hcwf.metrics.bins.PairArrays
        The pairs of the years that gave cases, concatenated.

    Raises
    ------
    ValueError
        No year gives pairs.
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

            if forecast_obs.n_cases > 0:
                all_forecast_obs_pairs.append(forecast_obs)
                log.debug(f"Target year {target_year} completed: {forecast_obs.n_cases} cases")
            else:
                log.debug(f"No pairs generated for target year {target_year}")

        except Exception as e:
            log.error(f"Error processing target year {target_year}: {e}")
            continue

    # Combine all years
    if not all_forecast_obs_pairs:
        raise ValueError("No data was successfully processed for any target year")

    combined_forecast_obs = PairArrays.concat(all_forecast_obs_pairs)

    log.debug(f"Climatological forecast cases of target years {list(years)}: {combined_forecast_obs.n_cases}")

    return combined_forecast_obs


class PreparedPairs(NamedTuple):
    """The pairs a model's scores are made from, built once per model (`prepare_pairs`)."""

    forecast: PairArrays
    """The model's forecast-observation pairs, every day bin."""
    reference: PairArrays | None
    """The reference model's pairs, from the reference model's own data; None for a climatology reference."""
    clim_onset: Any
    """The climatology years' onsets (a climatology reference), else None."""
    climatology: PairArrays | None
    """The climatology reference's pairs (conditional or not); None for a model reference."""


def _climatology_years(years, clim_onset, rc, case, select, day_bins):
    """Return the climatology reference's pairs of `years` (the form `hcwf.parallel.concat_year_args` calls).

    Parameters
    ----------
    years : iterable of int
        The target years.
    clim_onset : xarray.DataArray
        The onsets of the climatology years.
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case that carries the reference.
    select : hcwf.config.model.Selection
        The selection of the cells.
    day_bins : sequence of (int, int)
        The ``(first day, last day)`` bins within the forecast window.

    Returns
    -------
    hcwf.metrics.bins.PairArrays
        The conditional or unconditional climatological pairs, as the run's reference asks.
    """
    if is_conditional_climatology(rc.reference.model):
        return conditional_pairs(clim_onset, rc, case, select, years=years, day_bins=day_bins)
    return climatological_pairs(clim_onset, rc, years=years, day_bins=day_bins)


def _climatology_pairs(clim_onset, rc, case, select):
    """Return the pairs of the climatology reference (conditional or not) for all years.

    The years run in worker processes unless ``parallel`` is off.

    Parameters
    ----------
    clim_onset : xarray.DataArray
        The onsets of the climatology years.
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    case : hcwf.config.model.Case
        The case that carries the reference.
    select : hcwf.config.model.Selection
        The selection of the cells.

    Returns
    -------
    hcwf.metrics.bins.PairArrays
        The pairs of all years, concatenated.
    """
    return concat_year_args(_climatology_years, rc.evaluation.years,
                            (clim_onset, rc, case, select, rc.evaluation.day_bins), rc.parallel,
                            combine=PairArrays.concat)


def _reference_model(rc, model):
    """Return the reference as a source to read when `model` is the forecast model it is scored against.

    The reference has its own settings, written once in `hcwf.config.model.RunConfig.reference_cases`.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The forecast model.

    Returns
    -------
    hcwf.config.model.ModelSpec
        The reference's source.
    """
    return rc.reference_cases(for_model=model)[0].model


def prepare_pairs(rc, model, select=Selection()):
    """Build the forecast-observation pairs of `model` and of its reference, shared across its cases.

    With worker processes the model's years are sent off first and the climatological onsets are computed here
    while they run; the climatology pairs then go to the same workers. Without them the years run inside
    `start_pairs`, so the reference years' rain (held from the thresholds to the climatological onsets) is used and
    freed before they are read, not kept alongside them.

    Parameters
    ----------
    rc : hcwf.config.model.RunConfig
        The resolved run config.
    model : hcwf.config.model.ModelSpec
        The forecast model.
    select : hcwf.config.model.Selection, default Selection()
        The selection of the cells.

    Returns
    -------
    PreparedPairs
        The model's pairs, the reference model's own pairs (None for ``climatology`` and ``conditional_climatology``),
        the climatological onsets and the climatology pairs; `hcwf.metrics.skill.create_score_results` reuses them.
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
