"""Frozen reference copies of ROMP's original onset detection (main @ f50d4cf).

Verbatim from momp/stats/detect.py before onset detection was routed through
momp.stats.onset_rule. Used only by tests to prove the refactor preserves
behaviour. Do not "fix" anything here; discrepancies are documented in
test_onset_rule_legacy_equivalence.py.
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


def find_first_true(arr):
    """
    Find first occurrence of onset condition for each grid point.

    Works if arr contains floats, integers, or NaNs.
    Returns the index of the first True value, or -1 if none.
    """
    # Convert to boolean, treat NaN as False
    arr_bool = np.asarray(arr, dtype=float)  # ensure numeric
    arr_bool = np.nan_to_num(arr_bool, nan=0.0)  # NaN -> 0
    arr_bool = arr_bool.astype(bool)  # convert to boolean

    if arr_bool.any():
        return int(np.argmax(arr_bool))  # first True index
    else:
        return -1


def detect_onset(day, forecast_series, thresh, *, wet_init, wet_spell, dry_spell, dry_threshold, dry_extent, **kwargs):
    """
    detect onset for model forecast
    ---
    forecast_series: grid point xarray time series
    """

    #dry_threshold = wet_init # default

    # !!! day start from index 1 in data as step
    start_idx = day - 1

    if dry_extent <= wet_spell:
        end_idx = start_idx + wet_spell

        if end_idx <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]
    
            # Check basic onset condition
            #if window_series[0] > wet_init and np.nansum(window_series) > thresh:
            if np.all(window_series >=  wet_init) and np.nansum(window_series) > thresh:
                return True


            # check if followed by dry spell
    else:
        end_idx = start_idx + wet_spell
        end_idx_dry = start_idx + dry_extent

        if end_idx_dry <= len(forecast_series):
            window_series = forecast_series[start_idx:end_idx]
            dry_series = forecast_series[start_idx:end_idx_dry]

            if np.all(window_series >=  wet_init) and np.nansum(window_series) > thresh:

                dry_bool = dry_series < wet_init
                consec_dry = np.convolve(dry_bool, np.ones(dry_spell, dtype=int), 'valid')
                has_dry_spell = np.any(consec_dry == dry_spell)

                if not has_dry_spell:
                    return True


def detect_observed_onset(rain_slice, thresh_slice, year, *, wet_init, wet_spell, 
                          dry_spell, dry_threshold, dry_extent, start_date, end_date, fallback_date, mok, 
                          extend_end_day=47, **kwargs):
    """Detect observed onset dates for a given year."""

    #window = 5 # 5-day wet spell window

    #mok = kwargs["mok"]
    #wet_spell = kwargs["wet_spell"]
    #wet_init = kwargs["wet_init"]
    #dry_spell = kwargs["dry_spell"]
    #dry_extent = kwargs["dry_extent"]
    #dry_threshold = kwargs["dry_threshold"]
    #start_MMDD = kwargs["start_date"][1:]
    #fallback_MMDD = kwargs["fallback_date"]

    start_MMDD = start_date[1:]
    end_MMDD = end_date[1:]
    fallback_MMDD = fallback_date

    end_date = datetime(year, *end_MMDD)
    if extend_end_day:
        end_date = end_date + timedelta(days=extend_end_day)

    # Set start date based on mok flag
    if mok:
#        print("YESYESYES")
        start_date = datetime(year, *mok)  # MOK date: June 2nd
        date_label = f"{mok[0]:02d}-{mok[1]:02d}"

    else:
        start_date = datetime(year, *start_MMDD)  # default start_date 
        date_label = f"{start_MMDD[0]:02d}-{start_MMDD[1]:02d}"

#    start_date = datetime(year, *start_MMDD)  # default start_date 
#    date_label = f"{start_MMDD[0]:02d}-{start_MMDD[1]:02d}"

    # Find start date index
    time_dates = pd.to_datetime(rain_slice.time.values)
    #start_idx_candidates = np.where(time_dates > start_date)[0]
    start_idx_candidates = np.where(time_dates >= start_date)[0]

#    print("time_dates ", time_dates)
#    print("start_dates ", start_date)
#    print("start_idx_candidates = ", start_idx_candidates)

    if len(start_idx_candidates) == 0 and fallback_date:
        print(f"Warning: {date_label} not found in data for year {year}")
        fallback_date = datetime(year, *fallback_MMDD)
        start_idx = np.where(time_dates >= fallback_date)[0][0]
        print(f"Using fallback date: April 1st")
    else:
        start_idx = start_idx_candidates[0]
        #print(f"Using {date_label} as start date for onset detection")

    # Subset rain_slice from start date onward
    #rain_subset = rain_slice.isel(time=slice(start_idx, None))#.sel(time=slice(None,end_date))
    rain_subset = rain_slice.isel(time=slice(start_idx, None)).sel(time=slice(None,end_date))
#    print("XXX", rain_subset.time)
#    import sys
#    sys.exit()

    # Create rolling 5-day sums
    rolling_sum = rain_subset.rolling(time=wet_spell, min_periods=wet_spell, center=False).sum()
    rolling_sum_aligned = rolling_sum.shift(time=-(wet_spell-1))

    # Create onset condition
    wet_day_condition = rain_subset >= wet_init
    wet_day_spell = (wet_day_condition.rolling(time=wet_spell,
                                                       min_periods=wet_spell, center=False).reduce(np.all))
    
    #wet_day_spell = (wet_day_condition.rolling(time=wet_spell, min_periods=wet_spell)
    #                 .sum()== wet_spell) # this method make sure return bool type, no need .fillna(False) line

    first_day_condition = wet_day_spell.shift(time=-(wet_spell-1))
    first_day_condition = first_day_condition.fillna(False).astype(bool) # convert nans to bool


    sum_condition = rolling_sum_aligned > thresh_slice

    # check false onset
    #print("first_day_condition = ", first_day_condition[100,...])
    #print("sum_condition = " , sum_condition[100,...] )

    #if dry_extent > 0:
    if dry_extent >= dry_spell and dry_extent > wet_spell:
        #dry_rolling = rain_subset.rolling(time=dry_spell, min_periods=dry_spell).sum() < dry_threshold
        #dry_rolling_start_aligned = dry_rolling.shift(time=-(dry_spell-1))  # align to start of each 10-day window
        #dry_rolling_after_onset = dry_rolling_start_aligned.shift(time=-(wet_spell))

        #dry_search_window = dry_extent - dry_spell + 1  # 30 - 10 + 1 = 21
        #dry_in_extent = dry_rolling_after_onset.rolling(time=dry_search_window, min_periods=1).reduce(np.any)
        #no_dry_after = (~dry_in_extent.astype(bool))
        ##no_dry_after = ~dry_in_extent

        dry_day_condition = rain_subset < wet_init

        #dry_day_spell = (dry_day_condition.rolling(time=dry_spell,
        #                                                   min_periods=dry_spell, center=False).reduce(np.all))

        dry_day_spell = (dry_day_condition.rolling(time=dry_spell, min_periods=dry_spell)
                         .sum() == dry_spell)

        #print("dry_day_condition = ", dry_day_condition)
        #print("dry_day_spell = ", dry_day_spell)

        #no_dry_after = ~(dry_day_spell.rolling(time=dry_extent+1, min_periods=1)
        #                 .reduce(np.any).shift(time=-dry_extent))

        has_dry_after = (dry_day_spell.rolling(time=dry_extent+1, min_periods=1)
                         .sum() > 0 ).shift(time=-dry_extent).fillna(False).astype(bool)

        #print("has_dry_after = ", has_dry_after)

        no_dry_after = xr.apply_ufunc(np.logical_not, has_dry_after)
        #no_dry_after = has_dry_after == False
        #no_dry_after = ~has_dry_after

        #print("no_dry_after =  ", no_dry_after)
        onset_condition = first_day_condition & sum_condition & no_dry_after

    else:
        onset_condition = first_day_condition & sum_condition

#    print("rain_slice = ", rain_slice[100,...] )
    #print("rain_subset = ", rain_subset[100,...] )
#    print("\nrolling_sum_aligned = ", rolling_sum_aligned[100,...])
#    print("\nfirst_day_condition = ", first_day_condition[100,...])
#    print("\nsum_condition = ", sum_condition[100,...])
#    print("find_first_true = ", find_first_true)
#    print("\nonset_condition = ", onset_condition[100,...] )
#    print("onset_condition = ", onset_condition )
#    #print(" input_core_dims = ", [['time']])
#    import sys
#    sys.exit()

    onset_indices = xr.apply_ufunc(
        find_first_true,
        onset_condition,
        input_core_dims=[['time']],
        output_dtypes=[int],
        vectorize=True
    )

    # Convert indices to actual dates
    valid_mask = onset_indices.values >= 0
    time_coords = rain_subset.time.values
    onset_dates_array = np.full(onset_indices.shape, np.datetime64('NaT'), dtype='datetime64[ns]')

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices[i, j].values)
                if 0 <= idx < len(time_coords):
                    onset_dates_array[i, j] = time_coords[idx]

    # Create final onset date DataArray
    onset_da = xr.DataArray(
        onset_dates_array,
        coords=[('lat', rain_slice.lat.values), ('lon', rain_slice.lon.values)],
        name='onset_date'
    )

#    print("onset_condition = ", onset_condition)
#    print("onset_indices = ", onset_indices)
#    print("onset_da  = ", onset_da)
    return onset_da

