"""Kernel-density climatology of onset days: conditional and unconditional onset probabilities.

A cell's historical onset days are smoothed with a Gaussian kernel density
estimate (Scott's bandwidth). Its CDF is built the way R's ``density()`` plus
``approxfun(rule = 2)`` build one: the density on 512 grid points spanning the
data +/- 3 bandwidths, cumulated and normalised, with the end values held
outside the grid. From that CDF F, for an issue day d and lead k (1..horizon):

* unconditional: P(onset on day d + k) = F(d + k) - F(d + k - 1), and lead 0
  is P(onset already happened by d) = F(d);
* conditional: P(onset on day d + k | no onset by d) = (F(d + k) - F(d + k - 1)) / (1 - F(d)).

Every probability is floored at MIN_PROB while each row keeps its sum (the
excess comes off the entries above the floor in proportion), so no lead has
probability exactly zero. Optionally the KDE is refitted without each test
year (leave-one-year-out), so a year is never forecast from its own onset.

The numerics are a port of the onset_blending package's climatology, operation
for operation, so the two give identical probabilities for the same inputs.
Days are plain numbers on any common axis: the day of the year for HCWF's reference
forecast, the days since the season start for onset_blending (`season_days` makes either).

Around the core sits onset_blending's caller logic. `issue_grid` lists the issue dates of the
test years with each one's day of the season and its horizon, `climatology_years` picks the
training and test windows, `eligible_onsets` keeps the units with enough onset years, and
`probabilities_for_issues` computes the probabilities once per distinct (season day, horizon),
or per (year, season day, horizon) when the KDE is refitted without each test year, in tasks of
units over the run's pool (hcwf.stats.parallel). `conditional_probabilities` and
`unconditional_probabilities` are the same computation on HCWF's cells and one horizon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

MIN_PROB = 5e-7
EPS = 1e-12
MIN_ONSETS = 10


def fit_kde(onset_days):
    """A Gaussian KDE (Scott's bandwidth) of the onset days, or None with fewer than MIN_ONSETS of them.

    NaN days (years without an onset) are dropped first. A degenerate sample
    that scipy cannot fit (for example all days equal) also gives None.
    """
    x = np.asarray(onset_days, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < MIN_ONSETS:
        return None
    try:
        return gaussian_kde(x, bw_method="scott")
    except Exception:
        return None


def onset_cdf(kde):
    """The KDE's CDF as a callable, built like R's density() + approxfun(rule = 2).

    The density is evaluated on 512 points from the smallest day minus three
    bandwidths to the largest plus three, cumulated and normalised; outside
    that range the CDF holds its first and last value.
    """
    data = kde.dataset.flatten()
    bandwidth = kde.factor * np.std(data, ddof=1)
    grid = np.linspace(data.min() - 3 * bandwidth, data.max() + 3 * bandwidth, 512)
    pdf = kde.evaluate(grid)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    return interp1d(grid, cdf, bounds_error=False, fill_value=(cdf[0], cdf[-1]))


def floor_probabilities(p_raw, target_sum, lb: float = MIN_PROB, eps: float = EPS):
    """Each row of p_raw at least lb, rescaled to keep its target sum (clipped to [0, 1]).

    The excess over the target comes off what lies above the floor, in
    proportion. Rows whose target is zero become zero; rows whose target is
    too small for every entry to reach the floor become uniform.
    """
    p_raw = np.asarray(p_raw, dtype=float)
    target_sum = np.asarray(target_sum, dtype=float).copy()
    p_raw = np.maximum(p_raw, 0.0)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
    target_sum = np.where(np.isfinite(target_sum) & (target_sum >= 0.0), target_sum, 0.0)
    target_sum = np.minimum(target_sum, 1.0)

    out = np.empty_like(p_raw)
    zero = target_sum <= eps
    small = (~zero) & (target_sum < p_raw.shape[1] * lb)
    normal = ~(zero | small)
    out[zero] = 0.0
    out[small] = target_sum[small, None] / p_raw.shape[1]
    if normal.any():
        p = np.maximum(p_raw[normal], lb)
        excess = p.sum(axis=1) - target_sum[normal]
        no_reduction = excess <= eps
        adjusted = np.empty_like(p)
        adjusted[no_reduction] = p[no_reduction]
        reduce = ~no_reduction
        if reduce.any():
            p_reduce = p[reduce]
            reducible = p_reduce - lb
            reducible_sum = reducible.sum(axis=1)
            degenerate = reducible_sum <= eps
            reduced = np.empty_like(p_reduce)
            targets = target_sum[normal][reduce]
            reduced[degenerate] = targets[degenerate, None] / p_raw.shape[1]
            regular = ~degenerate
            reduced[regular] = p_reduce[regular] - (
                excess[reduce][regular, None] * reducible[regular] / reducible_sum[regular, None]
            )
            adjusted[reduce] = np.maximum(reduced, lb)
        out[normal] = adjusted
    return out


def onset_probabilities(kde, issue_days, horizon: int, horizons=None) -> tuple[np.ndarray, np.ndarray]:
    """Conditional (rows, horizon) and unconditional (rows, horizon + 1) probabilities from one KDE.

    One row per issue day. Conditional column k - 1 is P(onset on issue day +
    k | none by the issue day); unconditional column 0 is P(onset by the issue
    day) and column k is P(onset on issue day + k). Both are NaN when kde is
    None. A conditional row sums to P(onset within the horizon | none yet),
    except when the KDE has put (almost) all mass before the issue day, where
    it falls back to the unconditional shape.

    `horizons` gives each row its own horizon (at most `horizon`, which sets the width): a row's leads
    past its horizon stay NaN, and a row whose horizon is not positive is all NaN (an issue date outside
    every configured horizon).
    """
    d0 = np.asarray(issue_days, dtype=int)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    windows = np.full(len(d0), horizon) if horizons is None else np.asarray(horizons, dtype=int)
    conditional = np.full((len(d0), horizon), np.nan)
    unconditional = np.full((len(d0), horizon + 1), np.nan)
    if kde is None or len(d0) == 0:
        return conditional, unconditional
    cdf = onset_cdf(kde)
    for length in np.unique(windows[windows > 0]):
        rows = np.flatnonzero(windows == length)
        length = int(length)
        days = np.arange(1, length + 1)
        num = cdf(d0[rows, None] + days) - cdf(d0[rows, None] + days - 1)
        num = np.maximum(num, 0.0)
        num = np.where(np.isfinite(num), num, 0.0)

        base_prob = cdf(d0[rows])
        unconditional[rows, 0] = base_prob
        unconditional[rows, 1:length + 1] = floor_probabilities(num, num.sum(axis=1))

        denom = 1.0 - base_prob
        cond = np.empty_like(num)
        fallback = (~np.isfinite(denom)) | (denom <= EPS)
        if fallback.any():
            cond[fallback] = floor_probabilities(num[fallback], num[fallback].sum(axis=1))
        regular = ~fallback
        if regular.any():
            p_raw = num[regular] / denom[regular, None]
            p_raw = np.maximum(p_raw, 0.0)
            p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
            target_sum = p_raw.sum(axis=1)
            empty = (~np.isfinite(target_sum)) | (target_sum <= EPS)
            cond_regular = np.empty_like(p_raw)
            cond_regular[empty] = 0.0
            if (~empty).any():
                cond_regular[~empty] = floor_probabilities(p_raw[~empty], target_sum[~empty])
            cond[regular] = cond_regular
        conditional[rows, :length] = cond
    return conditional, unconditional


UNITS_PER_TASK = 200
"""Units per process task: enough to amortise a task, few enough to spread."""


def _unit_task(task: tuple, keys: pd.DataFrame, max_h: int, cv: bool) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities for a run of units at each key (year, d0, window): (unit, key, lead) arrays."""
    years_by_unit, days_by_unit = task
    d0 = keys["d0"].to_numpy(dtype=int)
    windows = keys["window"].to_numpy(dtype=int)
    conditional = np.full((len(days_by_unit), len(keys), max_h), np.nan)
    unconditional = np.full((len(days_by_unit), len(keys), max_h + 1), np.nan)
    key_years = keys["year"].to_numpy(dtype=int)
    for u, (unit_years, onset_days) in enumerate(zip(years_by_unit, days_by_unit)):
        if not cv:
            conditional[u], unconditional[u] = onset_probabilities(fit_kde(onset_days), d0, max_h, windows)
            continue
        for year in np.unique(key_years):
            rows = np.flatnonzero(key_years == year)
            kde = fit_kde(onset_days[unit_years != year])
            conditional[u, rows], unconditional[u, rows] = onset_probabilities(kde, d0[rows], max_h, windows[rows])
    return conditional, unconditional


def unit_probabilities(years_by_unit: Sequence, days_by_unit: Sequence, keys: pd.DataFrame, max_h: int, *,
                       cv: bool = False, workers: int = 1, units_per_task: int = UNITS_PER_TASK):
    """Conditional (unit, key, max_h) and unconditional (unit, key, max_h + 1) probabilities of every unit at every key.

    Each unit has its onset years and onset days (NaN for none; the years are only read with `cv`); `keys` is a
    table with year, d0 (issue day on the onsets' axis) and window (the key's horizon, at most max_h; not positive
    for none). With cv each key year's KDE is fitted without that year's onset. Units go in tasks of `units_per_task`
    to the run's pool when `workers` > 1 (results are in unit order whatever the workers).
    """
    from hcwf.stats import parallel

    n = len(days_by_unit)
    tasks = [(list(years_by_unit[start:start + units_per_task]), list(days_by_unit[start:start + units_per_task]))
             for start in range(0, n, units_per_task)]
    results = parallel.map_items(_unit_task, tasks, (keys, max_h, cv), workers=workers)
    if not results:
        return np.full((0, len(keys), max_h), np.nan), np.full((0, len(keys), max_h + 1), np.nan)
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def _cell_onsets(value, cv: bool):
    """(years or None, onset days) of one cell, from a Series indexed by year or a plain array of days."""
    if isinstance(value, pd.Series):
        return value.index.to_numpy(dtype=int), value.to_numpy(dtype=float)
    if cv:
        raise ValueError("cv_years needs each cell's onsets as a pandas Series indexed by year")
    return None, np.asarray(value, dtype=float)


def _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, which: int) -> xr.DataArray:
    """The conditional (which = 0) or unconditional (which = 1) array over (cell, [year,] issue, lead)."""
    issue = np.asarray(issue_days, dtype=int)
    cells = list(onset_days_by_cell)
    cv = cv_years is not None
    test_years = np.asarray(list(cv_years), dtype=int) if cv else None
    onsets = [_cell_onsets(onset_days_by_cell[cell], cv) for cell in cells]
    rows = np.tile(issue, len(test_years)) if cv else issue
    keys = pd.DataFrame({"year": np.repeat(test_years, len(issue)) if cv else np.zeros(len(rows), dtype=int),
                         "d0": rows, "window": np.full(len(rows), int(horizon))})
    out = unit_probabilities([years for years, _ in onsets], [days for _, days in onsets], keys, int(horizon),
                             cv=cv)[which]
    out = out.reshape((len(cells), len(test_years), len(issue), -1) if cv else (len(cells), len(issue), -1))
    leads = np.arange(1 - which, int(horizon) + 1)
    coords = {"cell": pd.Index(cells, dtype=object, tupleize_cols=False)}
    dims = ["cell"]
    if cv:
        coords["year"] = test_years
        dims.append("year")
    coords["issue"] = issue
    coords["lead"] = leads
    return xr.DataArray(out, dims=dims + ["issue", "lead"], coords=coords)


def conditional_probabilities(onset_days_by_cell: Mapping, issue_days: Sequence[int], horizon: int, *,
                              cv_years: Sequence[int] | None = None) -> xr.DataArray:
    """P(onset on lead k | no onset by the issue day) for leads 1..horizon, per cell, issue day and optionally test year.

    onset_days_by_cell maps each cell to its historical onset days: a pandas
    Series indexed by year, or, without cv_years, any array of days (NaN for a
    year without onset is fine). issue_days are the issue days on the same day
    axis as the onsets. The result has dims (cell, issue, lead), or (cell,
    year, issue, lead) when cv_years is given, in which case each test year's
    KDE is fitted without that year's onset. A cell with fewer than
    MIN_ONSETS onsets (after any leave-out) is NaN.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 0)


def unconditional_probabilities(onset_days_by_cell: Mapping, issue_days: Sequence[int], horizon: int, *,
                                cv_years: Sequence[int] | None = None) -> xr.DataArray:
    """P(onset already happened by the issue day) at lead 0, then P(onset on lead k) for k = 1..horizon.

    Arguments and dims as conditional_probabilities, with lead running 0..horizon.
    """
    return _probabilities(onset_days_by_cell, issue_days, horizon, cv_years, 1)


# ---------------------------------------------------------------------------
# The issue grid, the training window and the probabilities of a whole season (onset_blending's climatology)
# ---------------------------------------------------------------------------

def season_days(dates, season_start: tuple[int, int] = (1, 1), leap_shift: bool = False) -> np.ndarray:
    """Days from the season start (month, day) of each date's own year, as floats; NaN for NaT.

    The default start is 1 January. onset_blending counts from its season start, in real calendar days. With
    leap_shift, dates from 1 March of a leap year count one less (when the start is before March), so a calendar
    date has the same number every year and the KDE's day axis is not blurred by 29 February (HCWF's reference forecast).
    """
    index = pd.DatetimeIndex(np.asarray(dates).ravel())
    valid = ~index.isna()
    year = np.where(valid, index.year, 1970).astype(np.int64)
    month, day = season_start
    start = ((year - 1970).astype("datetime64[Y]").astype("datetime64[M]") + np.timedelta64(month - 1, "M")
             ).astype("datetime64[D]") + np.timedelta64(day - 1, "D")
    days = (index.values.astype("datetime64[D]") - start).astype(np.int64).astype(float)
    if leap_shift and month < 3:
        days = days - (index.is_leap_year & (index.month >= 3))
    return np.where(valid, days, np.nan)


def climatology_years(observed_years: Sequence[int], forecast_years: Sequence[int], *,
                      train_years: tuple[int, int] | None = None,
                      test_years: tuple[int, int] | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    """(train, test) year windows: the given ones, else the test window spans the forecast years and training runs
    from the first observed year to the year before the first test year (to the last observed year when that
    leaves nothing)."""
    if not len(observed_years):
        raise ValueError("Cannot build a climatology without observed years")
    if not len(forecast_years):
        raise ValueError("Cannot build a climatology without forecast years")
    test = test_years or (min(forecast_years), max(forecast_years))
    if train_years is not None:
        return tuple(train_years), tuple(test)
    first = min(observed_years)
    last = min(test[0] - 1, max(observed_years))
    if last < first:
        last = max(observed_years)
    return (first, last), tuple(test)


def eligible_onsets(onsets: pd.DataFrame, window: tuple[int, int], min_onset_years: int) -> tuple[pd.DataFrame, int]:
    """The onsets (id, year, onset_day) found in the window's years, of the units with at least min_onset_years of them.

    Returns the table and the number of eligible units; it is an error when no unit is eligible.
    """
    onsets = onsets[onsets["onset_day"].notna() & onsets["year"].between(*window)].copy()
    counts = onsets.groupby("id")["onset_day"].size()
    eligible = counts[counts >= int(min_onset_years)].index
    onsets = onsets[onsets["id"].isin(eligible)].copy()
    if onsets.empty:
        raise ValueError(
            "No units have enough observed onset years for a climatology "
            f"(min_onset_years={min_onset_years}, years {window[0]}:{window[1]})"
        )
    return onsets, len(eligible)


def issue_grid(test: tuple[int, int], season_start: str, issue_end: str, forecast_window, horizons) -> pd.DataFrame:
    """Every issue date of the test years (time, year), with its day of the season (d0) and horizon (window).

    The issue dates run from season_start to issue_end ("MM-DD") in each test year; d0 counts days since the season
    start. `horizons` is None (every date has forecast_window) or a list of {"start_md", "end_md", "forecast_window"}
    (a date has the first horizon that contains it; -1 outside every one).
    """
    times, years, d0, windows = [], [], [], []
    for year in range(int(test[0]), int(test[1]) + 1):
        first, last = pd.Timestamp(f"{year}-{season_start}"), pd.Timestamp(f"{year}-{issue_end}")
        if pd.isna(first) or pd.isna(last):
            raise ValueError(f"Bad season dates; expected 'MM-DD'. Got: {season_start} / {issue_end}")
        if last < first:
            raise ValueError(f"issue_end_md is before season_start_md for year {year}")
        days = pd.date_range(first, last, freq="D")
        times.append(days.date)
        years.append(np.full(len(days), year))
        d0.append((days - first).days.to_numpy())
        if horizons is None:
            windows.append(np.full(len(days), int(forecast_window)))
            continue
        window = np.full(len(days), -1)
        unset = np.ones(len(days), dtype=bool)
        for h in horizons:
            inside = unset & (days >= pd.Timestamp(f"{year}-{h['start_md']}")) & (days <= pd.Timestamp(f"{year}-{h['end_md']}"))
            window[inside] = int(h["forecast_window"])
            unset &= ~inside
        windows.append(window)
    return pd.DataFrame({
        "time": np.concatenate(times),
        "year": np.concatenate(years).astype(int),
        "d0": np.concatenate(d0).astype(int),
        "window": np.concatenate(windows).astype(int),
    })


def plan_issues(test: tuple[int, int], season_start: str, issue_end: str, forecast_window, horizons,
                cv_by_year: bool) -> tuple[pd.DataFrame, bool, int]:
    """The test years' issue grid, whether the probabilities vary by year, and the longest horizon.

    They vary by year when the KDE is refitted without each test year, or when a season day has different horizons in
    different years. `horizons` as for issue_grid; without it forecast_window is every date's horizon.
    """
    grid = issue_grid(test, season_start, issue_end, None if horizons else forecast_window, horizons or None)
    max_h = max(h["forecast_window"] for h in horizons) if horizons else int(forecast_window)
    by_year = cv_by_year or bool((grid.groupby("d0")["window"].nunique() > 1).any())
    return grid, by_year, max_h


def probabilities_for_issues(onsets: pd.DataFrame, grid: pd.DataFrame, by_year: bool, max_h: int, cv_by_year: bool,
                             workers: int = 1) -> xr.Dataset:
    """conditional and unconditional over (unit, [year,] season_day, lead) for every unit of the training onsets.

    `onsets` has id, year and onset_day (days on the same axis as the grid's d0, e.g. since the season start); the
    units keep their order of appearance. Probabilities are computed once per distinct (season day, horizon), or per
    (year, season day, horizon) with by_year; cv_by_year refits each unit's KDE without the key year. Conditional has
    leads 1..max_h (NaN at lead 0), unconditional 0..max_h.
    """
    columns = ["year", "d0", "window"] if by_year else ["d0", "window"]
    keys = grid[columns].drop_duplicates().reset_index(drop=True)
    if not by_year:
        keys["year"] = 0
    units = onsets["id"].unique()
    grouped = onsets.groupby("id", sort=False)
    per_unit = [grouped.get_group(u) for u in units]
    conditional, unconditional = unit_probabilities(
        [g["year"].to_numpy(dtype=int) for g in per_unit], [g["onset_day"].to_numpy(dtype=float) for g in per_unit],
        keys, max_h, cv=cv_by_year, workers=workers)

    season_day = np.unique(keys["d0"].to_numpy())
    day = np.searchsorted(season_day, keys["d0"].to_numpy())
    leads = np.arange(max_h + 1)
    if by_year:
        years = np.unique(keys["year"].to_numpy())
        year = np.searchsorted(years, keys["year"].to_numpy())
        shape = (len(units), len(years), len(season_day), len(leads))
        index = (slice(None), year, day)
        dims, coords = ("unit", "year", "season_day", "lead"), {"year": years}
    else:
        shape = (len(units), len(season_day), len(leads))
        index = (slice(None), day)
        dims, coords = ("unit", "season_day", "lead"), {}
    dense_conditional = np.full(shape, np.nan)
    dense_unconditional = np.full(shape, np.nan)
    dense_conditional[index + (slice(1, None),)] = conditional
    dense_unconditional[index] = unconditional
    return xr.Dataset(
        {"conditional": (dims, dense_conditional), "unconditional": (dims, dense_unconditional)},
        coords={"unit": np.asarray(units, dtype=object), "season_day": season_day, "lead": leads, **coords},
    )


@dataclass
class Climatology:
    """Onset probabilities by unit, season day and lead day.

    data holds conditional (leads 1..H, NaN at lead 0) and unconditional (leads 0..H) over (unit, season_day, lead),
    with a year dimension when the KDE is refitted without each test year (or a season day's horizon differs by
    year); its issue coordinate lists the test years' issue dates with their year and season day. windows holds
    conditional probabilities trained on other year windows, each over its own units.
    """

    data: xr.Dataset | None
    train_years: tuple[int, int]
    test_years: tuple[int, int]
    eligible_units: int
    windows: dict[tuple[int, int], xr.DataArray] = field(default_factory=dict)

    @property
    def by_year(self) -> bool:
        return "year" in self.data["conditional"].dims

    def release(self) -> None:
        """Drop the arrays, keeping the year windows."""
        self.data = None
        self.windows = {}

    def at(self, name: str, unit, year, season_day, leads: slice | None = None) -> np.ndarray:
        """Probabilities (rows, lead) for rows given as unit positions, years and season days; leads selects lead days
        (positions) when given."""
        array = self.data[name] if leads is None else self.data[name].isel(lead=leads)
        return _gather(array, unit, year, season_day)

    def legacy_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The (conditional, unconditional) tables the legacy pipeline wrote."""
        return (
            _legacy_table(self.data["conditional"].isel(lead=slice(1, None)), self.data, True),
            _legacy_table(self.data["unconditional"], self.data, False),
        )

    def legacy_window_table(self, window: tuple[int, int]) -> pd.DataFrame:
        """The legacy conditional table of a year window's climatology."""
        return _legacy_table(self.windows[window].isel(lead=slice(1, None)), self.data, True)


def _gather(array: xr.DataArray, unit, year, season_day) -> np.ndarray:
    """array's values at rows given as unit positions, years and season days."""
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def _legacy_table(array: xr.DataArray, data: xr.Dataset, conditional: bool) -> pd.DataFrame:
    """The legacy pipeline's table (time, year, id, model, predicted_prob_day_k) of a probability array."""
    cv = bool(data.attrs["cv_by_year"])
    model = ("clim_kde" if conditional else "clim_kde_unc") + ("_cv" if cv else "")
    units = array["unit"].values
    n_issue = data.sizes["issue"]
    unit_rows = np.repeat(np.arange(len(units)), n_issue)
    issue_rows = np.tile(np.arange(n_issue), len(units))
    years = data["issue_year"].values[issue_rows]
    columns = {
        "time": pd.Series(np.tile(pd.DatetimeIndex(data["issue"].values).date, len(units)), dtype=object),
        "year": years.astype(np.int64),
        "id": pd.Series(units[unit_rows], dtype=str),
        "model": pd.Series(np.full(len(unit_rows), model, dtype=object), dtype=str),
    }
    values = _gather(array, unit_rows, years, data["issue_season_day"].values[issue_rows])
    leads = [f"predicted_prob_day_{d}" for d in array["lead"].values]
    return pd.concat([pd.DataFrame(columns), pd.DataFrame(values, columns=leads, copy=False)], axis=1)


def build_climatology(onsets: pd.DataFrame, grid: pd.DataFrame, by_year: bool, max_h: int, *,
                      train: tuple[int, int], test: tuple[int, int], min_onset_years: int, cv_by_year: bool,
                      windows: Sequence[tuple[int, int]] = (), workers: int = 1) -> Climatology:
    """The climatologies for the issue dates of the grid: the training window's, and one per extra year window.

    `onsets` are all the observed onsets (id, year, onset_day); each window keeps the units with at least
    min_onset_years onsets in its years (eligible_onsets). The grid, by_year and max_h come from plan_issues.
    """
    train_onsets, eligible = eligible_onsets(onsets, train, min_onset_years)
    data = probabilities_for_issues(train_onsets, grid, by_year, max_h, cv_by_year, workers)
    data = data.assign_coords(
        issue=pd.to_datetime(grid["time"]).to_numpy(),
        issue_year=("issue", grid["year"].to_numpy(dtype=np.int64)),
        issue_season_day=("issue", grid["d0"].to_numpy(dtype=np.int64)),
    )
    data.attrs.update(cv_by_year=int(cv_by_year))
    by_window = {}
    for window in windows:
        window_onsets, _ = eligible_onsets(onsets, window, min_onset_years)
        by_window[window] = probabilities_for_issues(window_onsets, grid, by_year, max_h, cv_by_year, workers)[
            "conditional"]
    return Climatology(data=data, train_years=train, test_years=test, eligible_units=eligible, windows=by_window)
