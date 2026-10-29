"""Onset contingency metrics (TP/FP/FN/TN, MAE) for model or climatology forecasts across years."""

from momp.io.input import load_thresholds, get_initialization_dates
from momp.io.input import get_forecast_probabilistic_twice_weekly, get_forecast_deterministic_twice_weekly
from momp.io.input import load_imd_rainfall
from momp.stats.detect import detect_observed_onset, compute_onset_for_deterministic_model, compute_onset_for_all_members
from momp.stats.detect import check_alignment_once
from momp.stats.climatology import compute_climatological_onset, compute_climatology_as_forecast
from momp.utils.practical import restore_args

import numpy as np
import pandas as pd


def compute_onset_metrics_with_windows(onset_df, *, tolerance_days, verification_window, **kwargs):
    """Compute contingency matrix metrics following MATLAB logic with forecast and validation windows."""
    print(f"Computing onset metrics with tolerance = {tolerance_days} days")
    print(f"Verification window starts {verification_window} days after initialization")
    print(f"Forecast window length: {verification_window[1]} days")

    forecast_bin_start = verification_window[0]
    forecast_bin_end = verification_window[1]

    results_list = []
    if onset_df.empty:
        # Small domains (e.g. a user focus area) can have no forecast onsets in a year.
        onset_df = pd.DataFrame(columns=['lat', 'lon'])
    unique_locations = onset_df[['lat', 'lon']].drop_duplicates()

    print(f"Processing {len(unique_locations)} unique grid points...")

    for idx, (_, row) in enumerate(unique_locations.iterrows()):
        lat, lon = row['lat'], row['lon']

        grid_data = onset_df[(onset_df['lat'] == lat) & (onset_df['lon'] == lon)].copy()

        grid_data['obs_onset_dt'] = pd.to_datetime(grid_data['obs_onset_date'])
        grid_data['model_onset_dt'] = pd.to_datetime(grid_data['onset_date'])
        grid_data['init_dt'] = pd.to_datetime(grid_data['init_time'])

        TP = 0
        FP = 0
        FN = 0
        TN = 0
        num_onset = 0
        num_no_onset = 0
        mae_tp = []
        mae_fp = []

        gt_grd = grid_data['obs_onset_dt'].iloc[0]

        for _, init_row in grid_data.iterrows():
            t_init = init_row['init_dt']
            model_onset = init_row['model_onset_dt']

            valid_window_start = t_init + pd.Timedelta(days=forecast_bin_start)
            valid_window_end = t_init + pd.Timedelta(days=forecast_bin_end)

            whole_forecast_window_start = t_init + pd.Timedelta(days=1)
            whole_forecast_window_end = t_init + pd.Timedelta(days=forecast_bin_end)

            is_obs_onset_in_whole_window = whole_forecast_window_start <= gt_grd <= whole_forecast_window_end

            if is_obs_onset_in_whole_window:
                num_onset += 1
            else:
                num_no_onset += 1

            has_model_onset = not pd.isna(model_onset)

            if has_model_onset:
                is_model_in_valid_window = valid_window_start <= model_onset <= valid_window_end

                if is_model_in_valid_window:
                    abs_diff_days = abs((model_onset - gt_grd).days)

                    if abs_diff_days <= tolerance_days:
                        TP += 1
                        mae_tp.append(abs_diff_days)
                    else:
                        FP += 1
                        mae_fp.append(abs_diff_days)

                else:
                    if model_onset > valid_window_end: #make sure no model onset in early bins
                        if is_obs_onset_in_whole_window:
                            FN += 1
                        else:
                            TN += 1

            else:
                if is_obs_onset_in_whole_window:
                    FN += 1
                else:
                    TN += 1

        total_forecasts = len(grid_data)

        mae_combined = mae_tp + mae_fp
        mae = np.mean(mae_combined) if len(mae_combined) > 0 else np.nan
        mae_tp_only = np.mean(mae_tp) if len(mae_tp) > 0 else np.nan

        result = {
            'lat': lat,
            'lon': lon,
            'total_forecasts': total_forecasts,
            'true_positive': TP,
            'true_negative': TN,
            'false_positive': FP,
            'false_negative': FN,
            'num_onset': num_onset,
            'num_no_onset': num_no_onset,
            'mae_combined': mae,
            'mae_tp_only': mae_tp_only,
            'num_tp_errors': len(mae_tp),
            'num_fp_errors': len(mae_fp),
            'tolerance_days': tolerance_days,
            'verification_window': verification_window,
            'forecast_days': forecast_bin_end
        }
        results_list.append(result)

    metrics_df = pd.DataFrame(results_list)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(columns=[
            'lat', 'lon', 'total_forecasts', 'true_positive', 'true_negative',
            'false_positive', 'false_negative', 'num_onset', 'num_no_onset',
            'mae_combined', 'mae_tp_only',
        ])

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
                p_model, thresh_da, onset_da, **kwargs
            )

        elif not ref_clim: # deterministic model onset
            print("Computing onset for model forecast ...")
            onset_df = compute_onset_for_deterministic_model(
                p_model, thresh_da, onset_da, **kwargs
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

        # extract forecast at approporiate init dates
        if probabilistic and not ref_clim:
            print("-------Extracting ensemble forecast data ...")
            p_model = get_forecast_probabilistic_twice_weekly(year, **kwargs)
            check_alignment_once(imd, p_model, **kwargs)
            print("...")

            print("-------Computing onset for ensemble forecast ...")
            _, onset_df = compute_onset_for_all_members(p_model, thresh_da, onset_da, **kwargs)
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
