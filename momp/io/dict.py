"""Helpers for extracting arrays and tables from nested result dictionaries and DataFrames."""
import numpy as np
import pandas as pd


def extract_binned_dict(dic, second_level_key, bin_key="clean_bins"):
    """
    extract a 2-D array from a 2-level nested dictionary
    given a specific 2nd-level key, e.g. "Fair_Brier_Skill_Score"
    Extract all 1st-level keys
    And get one specified 2nd-level key values, e.g., "clean_bins" [(1,5),(6,10)]
    """
    # 1st-level keys
    first_level_keys = list(dic.keys())

    # binned skill scores are saved as a list without bin keys
    # use "clean_bins" key to indicate bins
    second_level_value_as_keys = next(iter(dic.values())).get(bin_key)

    # Build 2-D array (rows = 1st-level, columns = 3rd-level)
    array_2d = np.array([
        dic[first][second_level_key]
        for first in first_level_keys
    ])
    return array_2d, first_level_keys, second_level_value_as_keys


def select_key_at_level(d, level, key):
    """
    Select a specific key at a given nesting level in a nested dict.
    and remove that level from the returned dict.

    level = 1 → top level
    level = 2 → second level
    """

    if level == 1:
        # drop this level: return the selected subtree
        return d.get(key, {})

    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub = select_key_at_level(v, level - 1, key)
            if sub:
                out[k] = sub
    return out


def extract_overall_dict(dic, second_level_key):
    """
    extract a 1-D array from a 2-level nested dictionary
    given a specific 2nd-level key,
    Extract all 1st-level keys
    And get all 2nd-level values
    """
    # 1st-level keys
    first_level_keys = list(dic.keys())

    # 1-D array of values for the chosen 2nd-level key
    array_1d = np.array([dic[first][second_level_key] for first in first_level_keys])

    return array_1d, first_level_keys


def extract_pd_bins(df, day_bins, method='merge', *kwargs):
    """
    Select rows from df whose (bin_start, bin_end) match any tuple in bins.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'bin_start' and 'bin_end' columns.
    day_bins : iterable of tuples
        Example: ((1, 5), (6, 10))
    method : {"merge", "series"}
        - "merge": fast, recommended for large DataFrames
        - "series": simple, minimal change from zip/isin logic

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame
    """

    if method == "merge":
        bins_df = pd.DataFrame(day_bins, columns=["bin_start", "bin_end"])
        return df.merge(bins_df, on=["bin_start", "bin_end"], how="inner").copy()

    elif method == "series":
        mask = pd.Series(zip(df["bin_start"], df["bin_end"])).isin(day_bins)
        return df[mask]

    else:
        raise ValueError("method must be 'merge' or 'series'")
