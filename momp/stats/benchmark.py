"""Onset contingency metrics (TP/FP/FN/TN, MAE) for model or climatology forecasts across years."""

from momp.io.input import load_thresholds, load_history, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly, get_forecast_deterministic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_deterministic_model, compute_onset_for_all_members
from momp.stats.detect import check_alignment_once
from momp.stats.climatology import compute_climatological_onset, compute_climatology_as_forecast
from momp.utils.practical import restore_args

import numpy as np
import pandas as pd


def compute_onset_metrics_with_windows(onset_df, *, tolerance_days, verification_window, **kwargs):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows.

    Returns one row per grid point, in order of first appearance in onset_df, and summary totals.
    """
    print(f"Computing onset metrics with tolerance = {tolerance_days} days")
    print(f"Verification window starts {verification_window} days after initialization")
    print(f"Forecast window length: {verification_window[1]} days")

    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]
    columns = ['lat', 'lon', 'total_forecasts', 'true_positive', 'true_negative', 'false_positive', 'false_negative',
               'num_onset', 'num_no_onset', 'mae_combined', 'mae_tp_only', 'num_tp_errors', 'num_fp_errors',
               'tolerance_days', 'verification_window', 'forecast_days']

    if onset_df.empty:
        # Small domains (e.g. a user focus area) can have no forecast onsets in a year.
        onset_df = pd.DataFrame(columns=['lat', 'lon'])
    unique_locations = onset_df[['lat', 'lon']].drop_duplicates()

    print(f"Processing {len(unique_locations)} unique grid points...")

    if len(onset_df):
        day = np.timedelta64(1, "D")
        code = onset_df.groupby(['lat', 'lon'], sort=False).ngroup().to_numpy()
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
        metrics_df = pd.DataFrame({
            'lat': unique_locations['lat'].to_numpy(),
            'lon': unique_locations['lon'].to_numpy(),
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
    else:
        metrics_df = pd.DataFrame(columns=columns[:11])

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


def compute_metrics_multiple_years(*, obs_dir, obs_file_pattern, obs_var,
                                   thresh_file, thresh_var, wet_threshold,
                                   wet_init, wet_spell, dry_spell, dry_threshold, dry_extent,
                                   start_date, end_date, fallback_date, earliest_onset_date, years, years_clim,
                                   model_dir, model_var, ref_model, date_filter_year, init_days,
                                   unit_cvt, file_pattern, tolerance_days, verification_window, max_forecast_day,
                                   members,  onset_percentage_threshold, probabilistic, save_nc_climatology, **kwargs):

    """Compute onset metrics for multiple years."""
    kwargs = restore_args(compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresholds(**kwargs)

    ref_clim = (kwargs['model'] == 'climatology')

    # calculate obs climatology onset
    if ref_clim:
        print("Generating climatology onset doy for each year as onset_da_dict")
        climatological_onset_doy = compute_climatological_onset(**kwargs)

        # climatological onset as "obs" for climatology baseline metrics
        onset_da_dict = {year: climatological_onset_doy for year in years}

    # model onset vs obs onset
    for year in years:
        print(f"\n{'-'*50}")
        print(f"Processing year {year}")

        # obs onset
        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)
        history = load_history(year, imd, **kwargs)
        # extract forecast at approporiate init dates
        if probabilistic:
            print("Extracting ensemble forecast data ...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
        elif not ref_clim:
            print("Extracting model forecast data ...")
            p_model = get_forecast_deterministic_twice_weekly(year, **kwargs)

        if probabilistic or not ref_clim:
            check_alignment_once(imd, p_model, **kwargs)

        # detect onset dates
        if probabilistic: # emsemble forecast onset
            print("Computing onset for ensemble forecast ...")
            _, onset_df = compute_onset_for_all_members(
                p_model, thresh_da, onset_da, history=history, **kwargs
            )

        elif not ref_clim: # deterministic model onset
            print("Computing onset for model forecast ...")
            onset_df = compute_onset_for_deterministic_model(
                p_model, thresh_da, onset_da, history=history, **kwargs
            )

        elif ref_clim: # climatology as forecast
            print("Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da,
                **kwargs
            )

        # onset-obs metrics TP,TN,FP,FN for all locations and init times
        print("Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df, **kwargs
        )

        metrics_df_dict[year] = metrics_df

        if not ref_clim:
            onset_da_dict[year] = onset_da

        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
              f"FP={summary_stats['overall_false_positive']}, "
              f"FN={summary_stats['overall_false_negative']}, "
              f"TN={summary_stats['overall_true_negative']}")

    return metrics_df_dict, onset_da_dict


def ens_compute_metrics_multiple_years(*, obs_dir, obs_file_pattern, obs_var,
                                   thresh_file, thresh_var, wet_threshold,
                                   wet_init, wet_spell, dry_spell, dry_threshold, dry_extent,
                                   start_date, end_date, fallback_date, earliest_onset_date, years, years_clim,
                                   model_dir, model_var, ref_model, date_filter_year, init_days,
                                   unit_cvt, file_pattern, tolerance_days, verification_window, max_forecast_day,
                                   members,  onset_percentage_threshold, probabilistic, save_nc_climatology, **kwargs):

    """Compute onset metrics for multiple years."""

    kwargs = restore_args(ens_compute_metrics_multiple_years, kwargs, locals())

    metrics_df_dict = {}
    onset_da_dict = {}

    # load onset precip threshold, 2-D or scalar
    thresh_da = load_thresholds(**kwargs)

    ref_clim = (kwargs['model'] == 'climatology')

    # calculate obs climatology onset
    if ref_clim:
        climatological_onset_doy = compute_climatological_onset(**kwargs)

        # climatological onset as "obs" for climatology baseline metrics
        onset_da_dict = {year: climatological_onset_doy for year in years}

    for year in years:
        print(f"{'-'*50}")
        print(f"Processing year {year}")

        # obs onset
        imd = load_imd_rainfall(year, **kwargs)
        onset_da = detect_observed_onset(imd, thresh_da, year, **kwargs)
        history = load_history(year, imd, **kwargs)

        # extract forecast at approporiate init dates
        if probabilistic and not ref_clim:
            print("-------Extracting ensemble forecast data ...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            check_alignment_once(imd, p_model, **kwargs)
            print("...")

            print("-------Computing onset for ensemble forecast ...")
            _, onset_df = compute_onset_for_all_members(p_model, thresh_da, onset_da, history=history, **kwargs)
            print("...")

        elif ref_clim: # climatology as forecast
            print("-------Computing onset for climatology as forecast ...")
            init_dates = get_initialization_dates(year, **kwargs)
            onset_df = compute_climatology_as_forecast(
                climatological_onset_doy, year, init_dates, onset_da,
                **kwargs
            )

        # onset-obs metrics TP,TN,FP,FN for all locations and init times
        print("-------Computing onset-obs metrics TP,TN,FP,FN for all locations and init times ...")
        metrics_df, summary_stats = compute_onset_metrics_with_windows(
            onset_df, **kwargs
        )

        metrics_df_dict[year] = metrics_df

        if not ref_clim:
            onset_da_dict[year] = onset_da

        print(f"Year {year} completed. Grid points processed: {len(metrics_df)}")
        print(f"Summary stats: TP={summary_stats['overall_true_positive']}, "
              f"FP={summary_stats['overall_false_positive']}, "
              f"FN={summary_stats['overall_false_negative']}, "
              f"TN={summary_stats['overall_true_negative']}")

    return metrics_df_dict, onset_da_dict
