"""Process- and thread-pool wrappers that split years into chunks and run the onset, pair and metric computations in parallel."""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from momp.stats.bins import multi_year_climatological_forecast_obs_pairs
from momp.stats.bins import multi_year_forecast_obs_pairs
from momp.stats.climatology import compute_climatological_onset_dataset
from momp.stats.climatology import compute_climatological_onset
from momp.stats.benchmark import compute_metrics_multiple_years
from momp.stats.benchmark import ens_compute_metrics_multiple_years
import xarray as xr


def func_wrapper(chunk_and_args, target_func=None):
    """
    wrapper function "fixes" the argument order problem
    pickable outside of main parallel fucntion
    Unpacks the chunk and the shared arguments to call the main function.
    """

    if target_func == "clim":
        chunk, clim_onset, kwargs = chunk_and_args
    elif target_func == "forecast":
        chunk, kwargs = chunk_and_args

    # Create the local copy to override 'years' without affecting others
    local_kwargs = kwargs.copy()
    local_kwargs['years'] = chunk

    # Explicitly call the function with the chunk assigned to 'years'
    if target_func == "clim":
        return multi_year_climatological_forecast_obs_pairs(
        clim_onset,
        **local_kwargs
        )

    if target_func == "forecast":
        return multi_year_forecast_obs_pairs(
        **local_kwargs
        )


def parallel_climatological_forecast_obs_pairs(clim_onset, **kwargs):
    """
    parallel version of multi_year_climatological...
    """
    # 1. get list of years for chunking
    all_years = kwargs.get('years')

    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    # 2. Determine the optimal number of workers
    # take the smaller of: (Total years to process) OR (Total CPU cores)
    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count)

    print(f"\n\n\ncpu_count is {os.cpu_count()}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...i\n\n\n")

    # 3. Split list of years into chunks
    # creates a list of arrays, e.g., [ [2010, 2011], [2012, 2013] ]
    year_chunks = np.array_split(all_years, n_workers)

    # 4. Prepare a list of tuples (chunk, clim_onset, kwargs) for each worker
    # We have to bundle them because executor.map only takes one argument
    tasks = [(chunk, clim_onset, kwargs) for chunk in year_chunks]

    # 5. Execute in parallel
    print(f"Executing on {n_workers} cores...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # executor.map passes one chunk from 'year_chunks' to each process
        func_wrapper_clim = partial(func_wrapper, target_func='clim')
        list_of_results = list(executor.map(func_wrapper_clim, tasks))

    # 5. Stitch the results back together
    # Each 'result' in 'list_of_results' is the DataFrame from function returns
    combined_forecast_obs = pd.concat(list_of_results, ignore_index=True)

    return combined_forecast_obs


def parallel_forecast_obs_pairs(**kwargs):
    """
    parallel version of multi_year_forecast...
    """
    all_years = kwargs.get('years')

    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count)

    print(f"\n\n\ncpu_count is {os.cpu_count()}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...i\n\n\n")

    year_chunks = np.array_split(all_years, n_workers)

    tasks = [(chunk, kwargs) for chunk in year_chunks]

    print(f"Executing on {n_workers} cores...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        func_wrapper_forecast = partial(func_wrapper, target_func='forecast')
        list_of_results = list(executor.map(func_wrapper_forecast, tasks))

    combined_forecast_obs = pd.concat(list_of_results, ignore_index=True)

    return combined_forecast_obs


def func_wrapper_climatology(chunk_and_args, dataset=False):
    """
    wrapper function "fixes" the argument order problem
    pickable outside of main parallel fucntion
    Unpacks the chunk and the shared arguments to call the main function.
    """

    chunk, kwargs = chunk_and_args

    local_kwargs = kwargs.copy()
    local_kwargs['years_clim'] = chunk

    if dataset:
        return compute_climatological_onset_dataset(**local_kwargs)
    else:
        return compute_climatological_onset(**local_kwargs)


def parallel_climatological_onset(**kwargs):
    """Compute climatological onset day-of-year, averaging over year chunks computed in threads unless grid_point is set or only one year is given."""


    grid_point = kwargs.get('grid_point', False)
    years_clim = kwargs.get('years_clim')

    # Logic: Only parallelize if NOT grid_point AND we have multiple years
    if not grid_point and len(years_clim) > 1:
        n_workers = min(os.cpu_count() or 1, len(years_clim))
        year_chunks = np.array_split(years_clim, n_workers)

        print(f"\n\n\ncpu_count is {os.cpu_count()}")
        print(f"task count is {len(years_clim)}")
        print(f"Running on {n_workers} cores for {len(years_clim)} years...i\n\n\n")

        tasks = [(chunk, kwargs) for chunk in year_chunks]

        print(f"Running in PARALLEL ({n_workers} cores) for non-grid-point task...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func_wrapper_climatology, tasks))

        # Stitch results (only onset is returned when grid_point is False)
        climatological_onset_doy = xr.concat(results, dim='chunk_idx').mean(dim='chunk_idx')
        climatological_onset_doy = np.round(climatological_onset_doy)
        return climatological_onset_doy

    else:
        # Single-process execution
        print("Running in SINGLE-PROCESS mode (Grid Point = True or single year)...")
        return compute_climatological_onset(**kwargs)


def parallel_climatological_onset_dataset(**kwargs):
    """Compute per-year climatological onset in threads over chunks of years_clim and concatenate the results along a 'year' dimension."""


    years_clim = kwargs.get('years_clim')

    n_workers = min(os.cpu_count() or 1, len(years_clim))
    year_chunks = np.array_split(years_clim, n_workers)

    print(f"\n\n\ncpu_count is {os.cpu_count()}")
    print(f"task count is {len(years_clim)}")
    print(f"Running on {n_workers} cores for {len(years_clim)} years...i\n\n\n")

    tasks = [(chunk, kwargs) for chunk in year_chunks]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        func_wrapper_forecast = partial(func_wrapper_climatology, dataset=True)
        list_of_results = list(executor.map(func_wrapper_forecast, tasks))

    # Since each worker returns an xr.DataArray with a 'year' dimension,
    # Use xarray's native concat to glue them back together.
    combined_da = xr.concat(list_of_results, dim='year')

    return combined_da


def func_wrapper_spatial(chunk_and_args, target_func=None):
    """
    wrapper function "fixes" the argument order problem
    pickable outside of main parallel fucntion
    Unpacks the chunk and the shared arguments to call the main function.
    """
    chunk, kwargs = chunk_and_args

    local_kwargs = kwargs.copy()
    local_kwargs['years'] = chunk

    if target_func == "ens":
        return ens_compute_metrics_multiple_years(**local_kwargs)
    else:
        return compute_metrics_multiple_years(**local_kwargs)


def parallel_metrics_multiple_years(ens=False, **kwargs):
    """
    parallel version of compute_metrics_multiple_years...
    """
    all_years = kwargs.get('years')

    if all_years is None:
        raise ValueError("The 'years' argument must be provided in kwargs.")

    cpu_count = os.cpu_count() or 1
    task_count = len(all_years)
    n_workers = min(cpu_count, task_count)

    print(f"\n\n\ncpu_count is {os.cpu_count()}")
    print(f"task count is {task_count}")
    print(f"Running on {n_workers} cores for {task_count} years...i\n\n\n")

    year_chunks = np.array_split(all_years, n_workers)

    tasks = [(chunk, kwargs) for chunk in year_chunks]

    print(f"Executing on {n_workers} cores...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        if ens:
            func_wrapper_ens = partial(func_wrapper_spatial, target_func='ens')
            list_of_results = list(executor.map(func_wrapper_ens, tasks))
        else:
            func_wrapper_det = partial(func_wrapper_spatial, target_func=None)
            list_of_results = list(executor.map(func_wrapper_det, tasks))

    final_metrics_df_dict = {}
    final_onset_da_dict = {}

    for metrics_part, onset_part in list_of_results:
        final_metrics_df_dict.update(metrics_part)
        final_onset_da_dict.update(onset_part)

    return final_metrics_df_dict, final_onset_da_dict
