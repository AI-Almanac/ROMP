"""Helpers that build Case objects from the config dict and manipulate case lists."""
from dataclasses import fields
from typing import Union, Optional
from pathlib import Path

from momp.lib.convention import Setting
from momp.utils.printing import combi_to_str
from momp.utils.practical import set_dir
from dataclasses import asdict
import copy


def init_dataclass(dc, dic):
    """Instantiate dataclass dc from the entries of dic that match its annotated field names."""
    keys = dc.__annotations__.keys()
    subset = {key: dic[key] for key in keys if key in dic}
    return dc(**subset)


def modify_list_keys(dictionary):
    """Return the keys of dictionary ending in "_list", with that suffix removed."""
    modified_keys = []
    for key, value in dictionary.items():
        if key.endswith("_list"):
            modified_key = key[:-5]  # Remove the last 5 characters "_list"
            modified_keys.append(modified_key)
    return modified_keys


def case_across_list(item, list1, list2):
    """find the corresponding item in list2 for a given item in list1"""
    if item in list1:
        index = list1.index(item)
        if index < len(list2):
            return list2[index]
    return None


def iter_list(dic, ext="_list"):
    """Return the dic[field + ext] list for each field named in dic["layout"]."""
    layout_pool = []
    for field in dic["layout"]:
        lst = dic.get(field + ext)
        layout_pool.append(lst)
    return layout_pool


def years_tuple_clim(year_start: int, year_end: int) -> tuple[int, ...]:
    """
    Create a tuple of integers from year_start to year_end inclusive.
    """
    return tuple(range(year_start, year_end + 1))


def years_tuple_model(start_date: tuple[int,int,int], end_date: tuple[int,int,int]) -> tuple[int, ...]:
    """
    Create a tuple of years from start_date to end_date inclusive.
    Each date is a tuple like (year, month, day)
    """
    start_year = start_date[0]
    end_year = end_date[0]
    return tuple(range(start_year, end_year + 1))


def take_ensemble_members(
    members: Optional[Union[tuple[int, ...], list[int], str]]
) -> list[int]:
    """
    Normalize members into a tuple of ints (or None).

    Accepts:
    - list[int] or tuple[int, ...] → returned as a tuple
    - string 'start-end' → expanded to a tuple
    - None, empty or 'All' → None
    """
    # Case 0: None, empty or 'All'
    if not members or members == 'All':
        return None

    # Case 1: list or tuple of ints
    if isinstance(members, (list, tuple)):
        return tuple(members)  # normalize to tuple

    # Case 2: string range "1-5"
    if isinstance(members, str) and "-" in members:
        start, end = map(int, members.split("-", 1))
        return tuple(range(start, end + 1))

    raise TypeError(
        "members must be list[int], tuple[int, ...], 'start-end' string, or None"
    )


def make_case(dataclass, combi, dic):
    """Build one Case: combi gives the values of the layout fields (in dic["layout"] order)
    and the remaining per-model _list entries are looked up by model name.
    """
    layout = dic["layout"]

    layout_dict = {key: None for key in layout}
    layout_dict.update(zip(layout_dict.keys(), combi))

    case_keys = [
        field.name for field in fields(dataclass)
    ]  # get defined keys in dataclass

    dic_list_keys = modify_list_keys(dic)  # return keys endwith _list, removing _list
    dic_list_keys = [
        item for item in dic_list_keys if item not in list(layout_dict.keys())
    ]
    list_keys = list(set(case_keys).intersection(set(dic_list_keys)))
    list_keys.remove("tolerance_days") # tolerance_days is paired with verification_window

    case = init_dataclass(dataclass, dic)
    case.update(layout_dict)

    value_list = []

    for key in list_keys:
        value = case_across_list(case.model, dic["model_list"], dic[key + "_list"])

        if key == "model_dir":
            if not Path(value).is_absolute():
                value = set_dir(value)

        value_list.append(value)

    case_name = combi_to_str(combi)
    case.case_name = case_name.replace(" ", "_")

    if dic.get('years') == 'All' or not dic.get('years'):
        case.years = years_tuple_model(dic['start_date'], dic['end_date'])

    if dic.get('years_clim') == 'All' or not dic.get('years_clim'):
        case.years_clim = years_tuple_clim(dic['start_year_clim'], dic['end_year_clim'])

    case.members = take_ensemble_members(dic['members'])

    value_dic = dict(zip(list_keys, value_list))
    case.update(value_dic)

    case.tolerance_days = case_across_list(case.verification_window,
                                           dic["verification_window_list"], dic["tolerance_days_list"])

    return case


def ref_cfg_layout(cfg, ref_model=None, verification_window=None):
    """Copy cfg for the reference model (a single-entry model_list); return the copy and its layout lists."""

    cfg_ref = copy.copy(cfg)

    if ref_model:
        cfg_ref.ref_model = ref_model

    if cfg.ref_model in ('climatology', 'conditional_climatology'):
        cfg_ref.probabilistic = False

    cfg_ref.model_list = (cfg.ref_model,)

    if verification_window:
        cfg_ref.verification_window_list = (verification_window,)

    layout_pool = iter_list(vars(cfg_ref))

    return cfg_ref, layout_pool


def ref_model_case(case_orig, setting):
    """Copy a case with the reference-model dir, variable, pattern and unit conversion; also return the merged case and setting dict."""

    case = copy.copy(case_orig)

    if isinstance(setting, Setting):
        case_ref = {'model_dir': setting.ref_model_dir,
                    'model_var': case.ref_model_var,
                    'file_pattern': setting.ref_model_file_pattern,
                    'unit_cvt': setting.ref_model_unit_cvt
                    }
    elif isinstance(setting, dict):
        case_cfg = setting
        case_ref = {'model_dir': case_cfg['ref_model_dir'],
                    'model_var': case_cfg['ref_model_var'],
                    'file_pattern': case_cfg['ref_model_file_pattern'],
                    'unit_cvt': case_cfg['ref_model_unit_cvt']
                    }

    case.update(case_ref)

    case_cfg_ref = {**asdict(case), **asdict(setting)}

    return case, case_cfg_ref


def filter_bins_in_window(day_bins, verification_window):
    """
    Returns only the day bins that lie completely within the verification window.

    Args:
        day_bins: tuple of tuples, e.g. ((1,5), (6,10), ...)
        verification_window: tuple of (start, end), e.g. (6, 20)

    Returns:
        tuple of tuples containing only the bins inside the window
    """
    start_win, end_win = verification_window

    return tuple(
        bin_range
        for bin_range in day_bins
        if start_win <= bin_range[0] and bin_range[1] <= end_win
    )
