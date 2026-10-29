"""Shorten the lead window so that every forecast search sees the whole rule (no CENSORED results).

A candidate on lead k needs the leads up to k + R - 1, where R is how many days
from the candidate the definition reads; the last usable lead is n_leads - R + 1.
"""
from __future__ import annotations

import os

import xarray as xr

from momp.lib.onset_config import OnsetConfigError, forecast_reach, is_legacy, onset_setup
from momp.utils.practical import set_dir
from momp.utils.standard import dim_fmt_model


def usable_last_lead(n_leads: int, reach: int) -> int:
    """The last lead a candidate can be on when the rule reads `reach` days from the candidate."""
    return n_leads - reach + 1


def count_leads(dic: dict, index: int) -> int:
    """Number of forecast leads (steps of at least 1) in the first year's file of model_list[index]."""
    year = dic["start_date"][0]
    path = os.path.join(set_dir(str(dic["model_dir_list"][index])), dic["file_pattern_list"][index].format(year))
    if not os.path.exists(path):
        raise OnsetConfigError(f"cannot count the forecast leads of {dic['model_list'][index]}: {path} not found")
    dims_list = dic.get("model_dims_list")
    with xr.open_dataset(path) as ds:
        ds = dim_fmt_model(ds, dims_list[index] if dims_list else None)
        return int((ds["step"].values >= 1).sum())


def _clip(pairs, last: int):
    """Indices kept and the windows clipped to `last`; windows starting after it are dropped."""
    kept = [i for i, (a, _) in enumerate(pairs) if a <= last]
    return kept, tuple((pairs[i][0], min(pairs[i][1], last)) for i in kept)


def apply_lead_window(dic: dict, n_leads=count_leads) -> dict:
    """Clip max_forecast_day, verification_window_list, tolerance_days_list and day_bins to the usable leads.

    Definition mode only. The run uses the shortest usable window over the models; the
    printable lines (one per model) go to dic["onset_lead_lines"] and the window to dic["lead_window"].
    """
    if is_legacy(dic):
        return dic
    setup = onset_setup(dic)
    reach = forecast_reach(setup)
    wanted = dic["max_forecast_day"]
    lines, usable = [], {}
    for i, model in enumerate(dic["model_list"]):
        n = n_leads(dic, i)
        last = min(wanted, usable_last_lead(n, reach))
        if last < 1:
            raise OnsetConfigError(
                f"definition '{setup.definition.id}' reads {reach} days from the onset day but the {model} "
                f"forecasts have {n} leads, so no lead can be searched. Fix: use a longer forecast or set "
                "dry_spell_check = False")
        usable[model] = last
        note = f"shortened from max_forecast_day {wanted}" if last < wanted else "max_forecast_day"
        lines.append(f"Lead window      : 1-{last} for {model} ({setup.definition.id} reads {reach} days from the "
                     f"onset day; forecasts have {n} leads)  [{note}]")
    effective = min(usable.values())
    if len(set(usable.values())) > 1:
        lines.append(f"Lead window      : 1-{effective} for the run (the shortest of the models)")
    dic["onset_lead_lines"] = lines
    dic["lead_window"] = (1, effective)
    if effective < wanted:
        kept, windows = _clip(dic["verification_window_list"], effective)
        if not kept:
            raise OnsetConfigError(f"every verification window starts after lead {effective}, the last lead the "
                                   "definition can be searched on. Fix: change verification_window_list")
        dic["verification_window_list"] = windows
        dic["tolerance_days_list"] = tuple(dic["tolerance_days_list"][i] for i in kept)
        dic["day_bins"] = _clip(dic["day_bins"], effective)[1]
        dic["max_forecast_day"] = effective
    return dic
