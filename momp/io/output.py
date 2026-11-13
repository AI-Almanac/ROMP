#from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from momp.stats.bins import get_target_bins
from momp.utils.printing import tuple_to_str
from momp.stats.onset_rule import resolve_rule
#from momp.utils.printing import tuple_to_str

#def file_path(directory, filename):
#    """Join directory and filename into a full path."""
#    return Path(directory) / filename


#def save_score_results(score_results, *, model, max_forecast_day, dir_out, **kwargs):
def save_score_results(score_results, *, model, verification_window, dir_out, **kwargs):
    """Save overall and binned skill scores to CSV files"""
            
    # Save overall scores
    overall_scores = {
        'Metric': ['Fair_Brier_Score', 'Fair_Brier_Skill_Score', 'Fair_RPS', 'Fair_RPS_Skill_Score', 'AUC', 'AUC_ref'],
        'Score': [
            score_results['BS']['fair_brier_score'],
            score_results['skill_results']['fair_brier_skill_score'],
            score_results['RPS']['fair_rps'],
            score_results['skill_results']['fair_rps_skill_score'],
            score_results['AUC']['auc'],
            score_results['AUC_ref']['auc'],
        ]
    }

    overall_scores_nested = dict(zip(overall_scores['Metric'], overall_scores['Score']))

    #overall_df = pd.DataFrame(overall_scores)
    # pd.DataFrame expects lists/arrays as values, or a list of dicts for rows, not a dict of scalars
    # Wrap the dict in a list so pandas treats it as a single row, or make each value wrapped in a list []

    # create dataframe and save as .csv
    overall_df = pd.DataFrame([overall_scores_nested])

    window_str = tuple_to_str(verification_window)
    overall_filename = f'overall_skill_scores_{model}_{window_str}.csv'
                        #f'overall_skill_scores_{model}_{max_forecast_day}day.csv'
                        #{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    overall_filename = os.path.join(dir_out, overall_filename)
    overall_df.to_csv(overall_filename, index=False)
    print(f"Saved overall scores to '{overall_filename}'")
    print(overall_df)

    # Get target bins, remove bins with "Day Before" and "Day After"
    target_bins = get_target_bins(score_results['BS'], score_results['BS_ref'])
    clean_bins = [b.replace("Days ", "") for b in target_bins]

    # Save binned scores
    binned_data = {
        'Bin': target_bins,
        'clean_bins': clean_bins,
        'Fair_Brier_Skill_Score': [score_results['skill_results']['bin_fair_brier_skill_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC': [score_results['AUC']['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'AUC_ref': [score_results['AUC_ref']['bin_auc_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Forecast': [score_results['BS']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins],
        'Fair_Brier_Score_Climatology': [score_results['BS_ref']['bin_fair_brier_scores'].get(bin_name, np.nan) for bin_name in target_bins]
    }

    # create dataframe and save as .csv
    binned_df = pd.DataFrame(binned_data)

    binned_filename = f'binned_skill_scores_{model}_{window_str}.csv'
                        #f'binned_skill_scores_{model}_{max_forecast_day}day.csv'
    binned_filename = os.path.join(dir_out, binned_filename)
    binned_df.to_csv(binned_filename, index=False)
    print(f"Saved binned scores to '{binned_filename}'")
    print(binned_df)

    return binned_data, overall_scores_nested



def save_ref_score_results(results, filename):
                           #*, model, verification_window, max_forecast_day, dir_out, **kwargs):

    to_save = {
    "climatology_obs_df": results["climatology_obs_df"],
    "brier_ref": results["BS_ref"],
    "rps_ref": results["RPS_ref"],
    "auc_ref": results["AUC_ref"],
    }
    
    #filename = f'ref_scores_{model}_{tuple_to_str(verification_window)}window_{max_forecast_day}day.csv'
    #filename = os.path.join(dir_out, filename)

    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(to_save, f)
    


def load_ref_score_results(ref_score_file, results_dict):

#if ref_score_file.exists():
    with ref_score_file.open("rb") as f:
        loaded_results = pickle.load(f)

    results_dict.update(loaded_results)

    return results_dict
    


def save_metrics_to_netcdf(spatial_metrics, attrs_dict, desc_dict=None, fname='spatial_metrics',
                           allowed_attrs = ['model', 'years', 'tolerance_days', 'verification_window', 
                                            'max_forecast_day', 'mok']):
    """
    Create an xarray Dataset from spatial_metrics, attach global attributes,
    and save to a NetCDF file.

    Parameters
    ----------
    spatial_metrics : dict
        Dictionary of data variables for xr.Dataset
    attrs_dict : dict
        Global attributes (values may be str, int, float, bool, list, tuple)
    fout : str
        Output NetCDF file path
    """

    ds = xr.Dataset(spatial_metrics)

    fout = os.path.join(attrs_dict['dir_out'], "{}_{}.nc")
    fout = fout.format(fname, attrs_dict.get('case_name', 'missing_case_name'))

    #allowed_attrs = ['model', 'years', 'tolerance_days', 'verification_window', 'max_forecast_day', 'mok']

    # Normalize attributes for NetCDF compatibility
    clean_attrs = {}

    #for key, val in attrs_dict.items():
    for key in allowed_attrs:

        try:
            val = attrs_dict.get(key, None)
        except:
            val = attrs_dict.get(key, "")

        if isinstance(val, (list, tuple)):
            clean_attrs[key] = ','.join(map(str, val))
        elif isinstance(val, bool):
            clean_attrs[key] = int(val)
        elif val is None:
            clean_attrs[key] = str(val)
        else:
            clean_attrs[key] = val

    ds.attrs.update(clean_attrs)
    ds.attrs.update(onset_rule_attrs(attrs_dict))

    if desc_dict is not None:
        ds.attrs.update(desc_dict)

    ds.to_netcdf(fout)

    print(f"{fname} saved to: {fout}")



def onset_rule_attrs(cfg):
    """Identify the onset definition that produced a result, so it can be checked downstream."""
    import json
    from importlib.metadata import version

    rule = resolve_rule(
        cfg.get("onset_rule"),
        cfg.get("onset_rule_params"),
        **{k: cfg.get(k) for k in ("wet_init", "wet_spell", "dry_spell", "dry_extent")},
    )
    return {
        "onset_rule": rule.name,
        "onset_rule_params": json.dumps(rule.params(), sort_keys=True, default=str),
        "onset_rule_fingerprint": rule.fingerprint(),
        "momp_version": version("momp"),
    }



def set_nested(result_dict, keys, value):
    """build nested dictionaries on the fly based on combi, dynamic-nesting"""
    d = result_dict
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
    return result_dict



def analyze_nested_dict(d):
    """ inquire number of dimension and dimension names of a nested dict"""
    depth = dict_depth(d)
    dims = dict_dims(d)
    return depth, dims


def nested_dict_to_array(nested_dict, metric_key):
    """
    Convert a 2-level nested dict to a 2D NumPy array based on the first two levels.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary with at least two levels.
        Example: results[model][window]['mae']
    metric_key : str
        The key at the leaf level to extract (e.g., 'mae').

    Returns
    -------
    arr : np.ndarray
        2D array with shape (num_top_keys, num_second_keys)
    row_labels : list
        List of top-level keys (rows)
    col_labels : list
        List of second-level keys (columns)
    """
    # Top-level keys → rows
    row_labels = list(nested_dict.keys())

    # Second-level keys → columns (assume all rows share same keys)
    first_row = nested_dict[row_labels[0]]
    col_labels = list(first_row.keys())

    # Initialize array
    arr = np.full((len(row_labels), len(col_labels)), np.nan)

    # Fill array with metric values
    for i, rkey in enumerate(row_labels):
        for j, ckey in enumerate(col_labels):
            # Safely get the value (default to np.nan if missing)
            #arr[i, j] = nested_dict.get(rkey, {}).get(ckey, {}).get(metric_key, np.nan)
            da = nested_dict.get(rkey, {}).get(ckey, {}).get(metric_key, np.nan)
            arr[i,j] = da.mean()

    return arr, row_labels, col_labels

