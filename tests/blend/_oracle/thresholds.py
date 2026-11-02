"""TEST-ONLY ORACLE: onset_blending@1e6d696 core/onsets.py `_threshold_sums` and `_compute_thresholds` (lines 378-410), the
computed climatological threshold, with the file read replaced by the series themselves. Never import this from momp."""
import numpy as np
import pandas as pd


def threshold_sums(dates, years, rain, ids, window):
    """A file's per-unit rain total and finite days inside the threshold window."""
    first, last, start, end = window
    dates = pd.DatetimeIndex(dates)
    month_day = dates.strftime("%m-%d").to_numpy()
    inside = (years >= first) & (years <= last) & (month_day >= start) & (month_day <= end)
    if not inside.any():
        return None
    rain = rain[inside]
    finite = np.isfinite(rain)
    return pd.DataFrame({
        "id": ids.astype(str),
        "total": np.where(finite, rain, 0.0).sum(axis=0),
        "days": finite.sum(axis=0),
    })


def compute_thresholds(files, ids, days, first, last, start, end):
    """The thresholds table (id, onset_thresh) from files, a list of (dates, rain[time, unit]) one per year."""
    window = (first, last, start, end)
    parts = [threshold_sums(dates, pd.DatetimeIndex(dates).year.to_numpy(), rain, ids, window) for dates, rain in files]
    parts = [part for part in parts if part is not None]
    totals = pd.concat(parts).groupby("id", sort=True)[["total", "days"]].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = totals["total"].to_numpy() / totals["days"].to_numpy()
    return pd.DataFrame({"id": totals.index.astype(str), "onset_thresh": float(days) * mean})
