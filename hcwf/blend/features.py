"""Features: onsets collected over files, and the weekly features the blend trains on, built from them.

The onsets come from `hcwf.stats.detect` (the one adapter between HCWF and the onset engine: evaluation units,
search-start variants, member onset counts); this module only collects its per-file arrays, in parallel on the run's
pool, into `ObservedOnsets` and `ForecastOnsets`, and takes the members' mean rainfall for the rain predictors
(`member_mean`, the one blend-specific piece of that read).

A sample is a unit and issue date that the climatology and every model cover (the inner join of the legacy combine),
in the climatology's order (units, then issue dates), dropping issue dates after the observed onset. Its weekly
features (climatology logits, binned forecast probabilities, rain predictors, the outcome) are computed from the
arrays directly and held as arrays (`Weekly`); the legacy table is `Weekly.frame()`. The random forest reads its
day-level inputs as arrays too (`daily_inputs`).

Weeks are `hcwf.stats.bins.LeadBins` (column slices for the weekly sums, `codes` for the outcome); rain-predictor
windows are read from the onset definition (`Definition.windows()`, via the recipe). The numerics are the legacy
connector's (week sums, winsorised logits, rolling rain sums over the week's start days, rain transforms), operation
for operation, without its day-level table.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logit

from hcwf.blend.recipe import CLIMATOLOGY_FEATURE, Recipe, probability_feature_base, window_tag
from hcwf.blend.samples import DailyInputs, ForecastOnsets, ObservedOnsets, Weekly
from hcwf.config.model import Case, Selection, Window
from hcwf.io import input as io_input
from hcwf.lib.dataset_schema import file_year
from hcwf.onset.climatology import ReferenceOnset
from hcwf.onset import climatology_kde as kde
from hcwf.onset.climatology_kde import Climatology
from hcwf.onset.thresholds import unit_thresholds
from hcwf.stats import detect
from hcwf.stats.parallel import map_items
from hcwf.utils.workers import workers_within

log = logging.getLogger(__name__)

_CHUNK_ROWS = 1_000_000


# ---------------------------------------------------------------------------------------------------------------
# Onsets collected over files
# ---------------------------------------------------------------------------------------------------------------
#
# The run (`rc`, a RunConfig) says where the files are, how they become units (`units:`), the onset rule, its thresholds
# and the reading options; the thresholds come from `hcwf.io.input.thresholds_of`. What follows only adds what blending
# needs of that: the reference onset, the fixed cutoff, a year or a file at a time (a per-call `Selection`, and the
# run's specs replaced where blending reads differently: the horizon, the missing-rain policy, the lead window).


def reference_onset(recipe: Recipe) -> ReferenceOnset | None:
    """The reference onset (`onset.reference_onset`): a month-day, or a table read from its file."""
    spec = recipe.onset.reference
    if spec is None:
        return None
    if spec.month_day is not None:
        return ReferenceOnset(month_day=spec.month_day)
    if not os.path.exists(spec.file):
        raise FileNotFoundError(f"ref_onset file not found: {spec.file}")
    return ReferenceOnset.read(spec.file, date_col=spec.date_col, year_col=spec.year_col, day_col=spec.day_col,
                               base_date=spec.base_date, unit_col=spec.unit_col)


def _season_start(setup) -> str:
    start = setup.definition.seasons[0].start
    return f"{start.month:02d}-{start.day:02d}"


def observation_years(rc) -> list[int]:
    """The years of the observation files (the season a file holds is the year in its name)."""
    paths = rc.observations.paths
    years = [file_year(p) for p in paths]
    if None in years:
        raise ValueError("an observation file name has no year (19xx or 20xx); observations are read a year at a "
                         "time: " + ", ".join(p for p, y in zip(paths, years) if y is None))
    return sorted(set(years))


def _observed_year(year, rc, select, thresholds, season_start):
    """One year's observed onsets as a file summary: {ids, years, onset_day, onset_idx, present} over (unit, year)."""
    onset = detect.detect_observed(lambda: io_input.observations_of(year, rc, select), thresholds, year, rc, select)
    ids = np.asarray(onset["unit"].values).astype(str)
    dates = onset.values.astype("datetime64[D]")
    cutoff = np.datetime64(f"{year}-{season_start}", "D")
    onset_day = np.where(np.isnat(dates), np.nan, (dates - cutoff).astype("timedelta64[D]").astype(float))
    return {"ids": ids, "years": np.array([year]), "onset_day": onset_day[:, None],
            "onset_idx": onset["onset_index"].values[:, None], "present": onset["present"].values[:, None]}


def observed(rc, recipe: Recipe, thresholds, reference, *, workers: int = 1, keep_ids=None) -> ObservedOnsets:
    """Observed onsets for every observation year, years in parallel: the search starts at max(season start,
    reference onset); each year's onsets are kept in the observed-onset cache while it is active."""
    years = observation_years(rc)
    run = replace(rc, evaluation=replace(rc.evaluation, years=(), climatology_years=tuple(years)),
                  onset_options=replace(rc.onset_options, horizon=recipe.onset.observed_horizon))
    select = Selection(reference_onset=reference, keep_unit_ids=None if keep_ids is None else sorted(keep_ids))
    parts = map_items(_observed_year, years, (run, select, thresholds, _season_start(rc.onset)),
                      workers=max(1, min(workers, len(years))))
    return ObservedOnsets.from_parts(parts, _season_start(rc.onset))


def member_mean(rain: np.ndarray) -> np.ndarray:
    """(issue, member, unit, day) rainfall's mean over members, as the legacy pipeline computed it with pandas' compiled
    grouped mean: a compensated (Kahan) sum over the members in order, in the rainfall's own dtype, skipping NaN."""
    shape = (rain.shape[0],) + rain.shape[2:]
    total = np.zeros(shape, rain.dtype)
    compensation = np.zeros(shape, rain.dtype)
    count = np.zeros(shape, np.int64)
    for member in range(rain.shape[1]):
        value = rain[:, member]
        present = ~np.isnan(value)
        y = value - compensation
        t = total + y
        c = (t - total) - y
        c[np.isnan(c)] = 0  # an infinite value
        np.copyto(compensation, c, where=present)
        np.copyto(total, t, where=present)
        count += present
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count.astype(rain.dtype)
    out[count == 0] = np.nan
    return out


def _rain_mean(values: np.ndarray) -> np.ndarray:
    """One member is its own mean, with no spread."""
    return values[:, 0] if values.shape[1] == 1 else member_mean(values)


def forecast_starts(fixed_cutoff, reference: ReferenceOnset | None, issues, ids) -> dict:
    """The search starts after the base: the fixed cutoff date, and the reference onset ("MM-DD", or dates over
    init_time and unit)."""
    starts = {"fixed_cutoff": fixed_cutoff}
    if reference is not None:
        if reference.month_day is not None:
            starts["ref"] = reference.month_day
        else:
            years = pd.DatetimeIndex(issues).year.to_numpy(dtype=int)
            dates = reference.dates(np.tile(np.asarray(ids).astype(str), len(issues)), np.repeat(years, len(ids)))
            starts["ref"] = xr.DataArray(dates.reshape(len(issues), len(ids)), dims=("init_time", "unit"))
    return starts


def _forecast_file(path, rc, case, select, spec, thresholds, fixed_cutoff, reference, threshold_name):
    """One forecast file's member onset counts per lead day, a run of issues at a time.

    Returns the data, or None when `issue_dates` is given and the file has none of them: data holds, over (issue, unit, day), count_base / count_fixed_cutoff / count_ref (uint8:
    members with their onset on each lead day, from the search starting at lead 1, at the fixed cutoff date and at
    the reference onset), plus members (per issue), onset_threshold (unit) and rain_mean (the members' mean rainfall in
    the file's dtype).
    """
    with io_input.UnitForecast(path, case.model, spec, keep_ids=select.keep_unit_ids, issue_dates=select.issue_dates,
                               select=select) as file:
        ids, days, issues = file.ids, file.days, file.issues
        if not len(issues):
            return None   # none of the issues asked for is in this file
        n_issue, n_unit, n_day = len(issues), len(ids), len(days)
        variants = ("base", "fixed_cutoff", "ref")
        counts = {v: np.zeros((n_issue, n_unit, n_day), dtype=np.uint8) for v in variants}
        members, summary = np.zeros(n_issue, dtype=np.uint8), None
        run = replace(rc, evaluation=replace(rc.evaluation, max_forecast_day=n_day))   # each file has its own leads
        for start, stop, rain in file.runs():
            values = np.asarray(rain.values)
            n_member = values.shape[1]
            found, order, _ = detect.count_members(
                rain, thresholds, run, case, select,
                search_start=forecast_starts(fixed_cutoff, reference, issues[start:stop], ids))
            found = found.sel(lead=slice(1, None))
            for v in variants:
                counts[v][start:stop] = found.isel(start=order.index(v) if v in order else 0).values.astype(np.uint8)
            members[start:stop] = n_member
            if summary is None:
                summary = np.empty((n_issue, n_unit, n_day), dtype=values.dtype)
            summary[start:stop] = _rain_mean(values)
        threshold = unit_thresholds(thresholds.get(threshold_name), ids, spec.convention)
    variables = {"members": (("issue",), members), "onset_threshold": (("unit",), threshold),
                 "year": (("issue",), pd.DatetimeIndex(issues).year.to_numpy(dtype=np.int64)),
                 **{f"count_{v}": (("issue", "unit", "day"), counts[v]) for v in variants},
                 "rain_mean": (("issue", "unit", "day"), summary)}
    return xr.Dataset(variables, coords={"issue": issues, "unit": ids.astype(object), "day": np.asarray(days)})


def forecast(rc, recipe: Recipe, model: str, thresholds, reference, *, keep_ids=None, member_ids=None,
             units=None, issue_dates=None, workers: int = 1, budget_bytes: int | None = None) -> ForecastOnsets:
    """A model's forecast onsets for every one of its files, files in parallel as far as memory allows.

    keep_ids limits the units to those (the blendable units under the inner join; their results are unchanged);
    `units` overrides the model's UnitsSpec (a forecast passes the trained blend's); member_ids limits the members
    read (a forecast reads the trained ones); issue_dates reads only the issues on those days (files without any are
    left out).
    """
    spec, setup = recipe.model(model), rc.onset
    source = rc.model(model)
    case = Case(source, rc.evaluation.windows[0] if rc.evaluation.windows else Window(1, rc.evaluation.max_forecast_day, 0))
    paths = source.paths
    units = units if units is not None else source.units
    run = replace(rc, onset_options=replace(rc.onset_options, missing=recipe.onset.forecast_missing,
                                            horizon=rc.onset_options.horizon or "truncate"))
    select = Selection(
        first_lead=spec.lead_days[0], last_lead=spec.lead_days[1], max_member=spec.max_member,
        member_ids=None if member_ids is None else [int(m) for m in member_ids],
        rain_horizon=(spec.rain_horizon.max_day, spec.rain_horizon.policy), check_units=False,
        keep_unit_ids=None if keep_ids is None else sorted(keep_ids),
        issue_dates=None if issue_dates is None else tuple(issue_dates))
    item_bytes = [io_input.unit_forecast_bytes(p, source.var) for p in paths]
    count = max(1, min(workers, len(paths)))
    count = workers_within(item_bytes, count, budget_bytes) if item_bytes else count
    name = setup.definition.thresholds[0][0] if setup.definition.thresholds else None
    results = map_items(_forecast_file, paths, (run, case, select, units, thresholds, recipe.onset.fixed_cutoff,
                                                reference, name), workers=count)
    results = [r for r in results if r is not None]
    names = [tuple(data["unit"].values) for data in results]
    if len(set(names)) > 1:
        raise ValueError(f"model {model}: its files cover different units")
    data = (xr.concat(results, dim="issue", data_vars="minimal", coords="minimal", compat="equals",
                      join="exact") if results else None)
    if data is not None:
        issues = pd.DatetimeIndex(data["issue"].values)
        repeated = sorted({d.date().isoformat() for d in issues[issues.duplicated()]})
        if repeated:
            raise ValueError(f"model {model}: several issues on the same date ({', '.join(repeated[:5])}"
                             f"{'…' if len(repeated) > 5 else ''}): one initialisation per date")
        if not issues.is_monotonic_increasing:
            data = data.sortby("issue")
    return ForecastOnsets(model=model, data=data)


# ---------------------------------------------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------------------------------------------


def climatology(recipe: Recipe, setup, observed_onsets: ObservedOnsets, forecast_years, *,
                test_years: tuple[int, int] | None = None, workers: int = 1) -> Climatology:
    """The KDE climatologies for the issue dates of the test years (`hcwf.onset.climatology_kde`): the training
    window's and one per extra year window of the recipe. test_years overrides the recipe's (a live season scores
    one year); the recipe's own worker count, if any, wins over `workers`."""
    spec = recipe.climatology
    train, test = kde.climatology_years(observed_onsets.years, forecast_years, train_years=spec.train_years,
                                        test_years=test_years or spec.test_years)
    grid, by_year, max_h = kde.plan_issues(test, _season_start(setup), spec.issue_end,
                                           recipe.climatology_forecast_window(), spec.horizon_dicts(), spec.cv_by_year)
    return kde.build_climatology(observed_onsets.onsets(), grid, by_year, max_h, train=train, test=test,
                                 min_onset_years=spec.min_onset_years, cv_by_year=spec.cv_by_year,
                                 windows=spec.windows, workers=spec.workers or workers)


@dataclass
class Rows:
    """The feature rows: unit and issue positions into each source."""

    units: np.ndarray        # unit ids, per row
    times: np.ndarray        # issue dates, datetime64[ns], per row
    years: np.ndarray        # per row
    season_day: np.ndarray   # per row
    clim_unit: np.ndarray    # position in the climatology's units
    model_unit: dict[str, np.ndarray]
    model_issue: dict[str, np.ndarray]
    true_onset: np.ndarray   # datetime64[D], NaT where there is none
    true_onset_day: np.ndarray

    def __len__(self) -> int:
        return len(self.units)


def rows(recipe: Recipe, climatology: Climatology, observed_onsets: ObservedOnsets,
         forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Rows:
    """The units and issue dates every source covers, in the climatology's order."""
    clim = climatology.data
    clim_units = clim["unit"].values.astype(str)
    unit_ok = np.ones(len(clim_units), dtype=bool)
    issue_ok = np.ones(clim.sizes["issue"], dtype=bool)
    issues = clim["issue"].values
    for onsets in forecasts.values():
        unit_ok &= np.isin(clim_units, onsets.data["unit"].values.astype(str))
        issue_ok &= np.isin(issues, onsets.data["issue"].values)
    unit_pos, issue_pos = np.flatnonzero(unit_ok), np.flatnonzero(issue_ok)
    clim_unit = np.repeat(unit_pos, len(issue_pos))
    grid = np.tile(issue_pos, len(unit_pos))
    units, times, years = clim_units[clim_unit], issues[grid], clim["issue_year"].values[grid]

    observed_units = observed_onsets.data["unit"].values.astype(str)
    order = np.argsort(observed_units)
    found = order[np.searchsorted(observed_units, units, sorter=order).clip(0, len(order) - 1)]
    has_unit = observed_units[found] == units
    observed_years = observed_onsets.data["year"].values
    year_pos = np.searchsorted(observed_years, years).clip(0, observed_onsets.data.sizes["year"] - 1)
    known = has_unit & (observed_years[year_pos] == years)
    true_onset = np.where(known, observed_onsets.onset_dates()[found, year_pos], np.datetime64("NaT"))
    true_onset_day = np.where(known, observed_onsets.data["onset_day"].values[found, year_pos], np.nan)

    if trim if trim is not None else recipe.features.trim_after_true_onset:
        day = times.astype("datetime64[D]")
        keep = np.isnat(true_onset) | (day <= true_onset)
    else:
        keep = np.ones(len(units), dtype=bool)

    model_unit, model_issue = {}, {}
    for name, onsets in forecasts.items():
        m_units = onsets.data["unit"].values.astype(str)
        m_order = np.argsort(m_units)
        model_unit[name] = m_order[np.searchsorted(m_units, units[keep], sorter=m_order)]
        model_issue[name] = np.searchsorted(onsets.data["issue"].values, times[keep])
    return Rows(units=units[keep], times=times[keep], years=years[keep],
                season_day=clim["issue_season_day"].values[grid][keep], clim_unit=clim_unit[keep],
                model_unit=model_unit, model_issue=model_issue, true_onset=true_onset[keep],
                true_onset_day=true_onset_day[keep])


def _week_sums(daily: np.ndarray, bins) -> list[np.ndarray]:
    """Weekly sums of daily values (rows, days 1..D), as the legacy connector sums them."""
    daily = np.asfortranarray(daily)
    return [daily[:, window].sum(axis=1) for window in bins.column_slices()]


def _lead_days(r: Rows) -> np.ndarray:
    lead = (r.true_onset - r.times.astype("datetime64[D]")).astype("timedelta64[D]").astype(float)
    lead[np.isnat(r.true_onset)] = np.nan
    return lead


def _grid_lat_lon(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lat/lon parsed from "<lat>_<lon>" ids (NaN otherwise), per distinct id."""
    codes, distinct = pd.factorize(units)
    text = pd.Series(distinct, dtype=str)
    parts = text.str.split("_", n=1, expand=True)
    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else pd.Series(np.nan, index=text.index)
    grid = (text.str.count("_") == 1) & lat.between(-90, 90) & lon.between(-180, 360)
    return lat.where(grid).to_numpy()[codes], lon.where(grid).to_numpy()[codes]


def weekly(recipe: Recipe, definition, climatology: Climatology, observed_onsets: ObservedOnsets,
           forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Weekly:
    """The weekly features the legacy connector built from the day-level table.

    trim overrides features.trim_after_true_onset (a forecast never trims); `definition` is the onset definition the
    rain-predictor windows are read from."""
    bins = recipe.features.bins
    r = rows(recipe, climatology, observed_onsets, forecasts, trim=trim)
    first = recipe.models[0].name
    threshold = forecasts[first].data["onset_threshold"].values[r.model_unit[first]]
    units, sample_unit = np.unique(r.units, return_inverse=True)
    lat, lon = _grid_lat_lon(units)
    unit_threshold = np.full(len(units), np.nan)
    unit_threshold[sample_unit] = threshold  # one model unit per id
    labels = recipe.outcome_labels()
    codes = bins.codes(_lead_days(r), earlier=False)
    variables: dict[str, tuple] = {
        "sample_unit": ("sample", sample_unit.astype(np.int32)), "time": ("sample", r.times),
        "year": ("sample", pd.DatetimeIndex(r.times).year.to_numpy()), "lat": ("unit", lat), "lon": ("unit", lon),
        "onset_threshold": ("unit", unit_threshold), "outcome": ("sample", codes, {"labels": labels})}
    attrs = {"rain_horizons": {}, "unavailable_rain_predictors": {}}
    features: dict[str, np.ndarray] = {}
    for start in range(0, max(len(r), 1), _CHUNK_ROWS):
        here = slice(start, min(len(r), start + _CHUNK_ROWS))
        for name, values in _row_features(recipe, definition, climatology, forecasts, r, here, threshold,
                                          attrs).items():
            if name not in features:
                features[name] = np.empty(len(r), dtype=values.dtype)
            features[name][here] = values
    for name, values in features.items():
        variables[name] = ("sample", values)
    data = xr.Dataset(variables, coords={"unit": units.astype(object)})
    return Weekly(data=data, columns=["id", "time", "year", "lat", "lon", "onset_threshold", "outcome", *features],
                  attrs=attrs)


def _row_features(recipe, definition, climatology, forecasts, r: Rows, here: slice, threshold, attrs) -> dict:
    """The weekly features of a slice of rows, in the legacy column order."""
    features, bins = recipe.features, recipe.features.bins
    day_max = features.probability_days
    unit, years, season_day = r.clim_unit[here], r.years[here], r.season_day[here]
    out = {}
    feature = CLIMATOLOGY_FEATURE
    cond = climatology.at("conditional", unit, years, season_day)
    unc = climatology.at("unconditional", unit, years, season_day)
    cond_weeks = _week_sums(cond[:, 1: day_max + 1], bins)
    unc_weeks = _week_sums(unc[:, 1: day_max + 1], bins)
    out[f"{feature}_unc_earlier"] = _logit_winsor(unc[:, 0].copy())
    for w in range(bins.n_inner):
        out[f"{feature}_week{w + 1}"] = _logit_winsor(cond_weeks[w])
        out[f"{feature}_unc_week{w + 1}"] = _logit_winsor(unc_weeks[w])
    del cond, unc
    ids = r.units[here]
    for window in recipe.climatology.windows:
        array = climatology.windows[window]
        names = array["unit"].values.astype(str)
        order = np.argsort(names)
        position = order[np.searchsorted(names, ids, sorter=order).clip(0, len(order) - 1)]
        values = _windowed_at(array, position, years, season_day)
        values[names[position] != ids] = np.nan
        for w, week in enumerate(_week_sums(values[:, 1: day_max + 1], bins), start=1):
            out[f"{feature}_{window_tag(window)}_week{w}"] = _logit_winsor(week)

    series = {"base": "base", "fixed_cutoff": "fixed_cutoff", "ref_onset": "ref"}
    variants = ["base", *recipe.probability_variants()]
    for model in recipe.models:
        data = forecasts[model.name].data
        issue, unit_m = r.model_issue[model.name][here], r.model_unit[model.name][here]
        members = data["members"].values[issue].astype(float)[:, None]
        for variant in variants:
            daily = data[f"count_{series[variant]}"].values[issue, unit_m, :day_max] / members
            prefix = probability_feature_base(model.name, variant)
            for w, week in enumerate(_week_sums(daily, bins), start=1):
                out[f"{prefix}_week{w}"] = week
    out.update(_rain_predictors(recipe, definition, forecasts, r, here, threshold[here], attrs))
    return out


def _windowed_at(array, unit, year, season_day) -> np.ndarray:
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def _rain_predictors(recipe: Recipe, definition, forecasts, r: Rows, here: slice, threshold, attrs) -> dict:
    """Weekly maxima (or minima) of rolling rain sums, as the legacy connector computes them."""
    features, bins = recipe.features, recipe.features.bins
    starts_by_week = [list(range(a, b + 1)) for a, b in bins.inner]
    out = {}
    for model in recipe.models:
        name = model.name
        if not model.rain_predictors:
            continue
        transform = rain_transform(model.rain_transform if model.rain_transform is not None
                                   else features.rain_transform)
        policy = model.rain_horizon.resolved_policy
        rain_day_max = recipe.rain_horizon_days(model)
        strict = policy != "legacy"
        rain = forecasts[name].data["rain_mean"]
        days = rain["day"].values
        need = list(range(1, rain_day_max + 1))
        missing = [d for d in need if d not in days]
        if missing:
            raise ValueError(f"{name}: rainfall lead days {missing[:5]} are missing for the rain predictors")
        positions = np.searchsorted(days, need)
        issue, unit = r.model_issue[name][here], r.model_unit[name][here]
        mat = np.asfortranarray(rain.values[issue, unit][:, positions].astype(float))
        if strict and not np.isfinite(mat).all():
            raise ValueError(f"{name}: non-finite rainfall within the strict rain horizon ({rain_day_max} days)")
        attrs["rain_horizons"][name] = {"rain_day_max": rain_day_max, "strict": strict, "policy": policy}
        parsed = [(p.agg, recipe.window_days(p, definition)) for p in model.rain_predictors]
        rolled = {w: _rolling_sums(mat, w) for w in {w for _, w in parsed}}
        cache = {}
        for agg, window in parsed:
            for wi, starts in enumerate(starts_by_week):
                if (agg, window, wi) in cache:
                    continue
                valid = _complete_starts(rain_day_max, window, starts)
                cache[(agg, window, wi)] = _over_starts(rolled[window], valid, np.fmin if agg == "min" else np.fmax)
                if not valid:
                    column = f"diff_{name}_week{wi + 1}" if agg == "diff" else f"{agg}_{name}_{window}day_week{wi + 1}"
                    attrs["unavailable_rain_predictors"][column] = {
                        "model": name, "rain_day_max": rain_day_max, "window": window, "week": wi + 1}
        for agg, window in parsed:
            for w in range(1, bins.n_inner + 1):
                values = cache[(agg, window, w - 1)]
                if agg == "diff":
                    out[f"diff_{name}_week{w}"] = transform(values) - transform(threshold)
                else:
                    out[f"{agg}_{name}_{window}day_week{w}"] = transform(values)
    return out


def _logit_winsor(p, lo: float = 0.0001, hi: float = 0.9999) -> np.ndarray:
    """The logit of a probability clipped to [lo, hi]."""
    return logit(np.clip(np.asarray(p, dtype=float), lo, hi))


def rain_transform(value):
    """The rainfall transform rain predictors apply (not onset detection or the climatology): identity, sqrt,
    fourth_root, log1p, power:<p> or a number p (x ** p). Powers and log1p clip at 0; NaN stays NaN."""
    def power(p):
        p = float(p)

        def f(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.clip(x, 0.0, None) ** p)
        return f

    def identity(x):
        return np.asarray(x, dtype=float)

    if value is None:
        return identity
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return power(value)
    name = str(value).strip().lower()
    if name in ("identity", "none", ""):
        return identity
    if name == "sqrt":
        return power(0.5)
    if name == "fourth_root":
        return power(0.25)
    if name == "log1p":
        def log1p(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.log1p(np.clip(x, 0.0, None)))
        return log1p
    if name.startswith("power:"):
        return power(name.split(":", 1)[1])
    raise ValueError(f"Unknown rain_transform {value!r}: identity, sqrt, fourth_root, log1p, 'power:<p>' or a number.")


def _rolling_sums(mat: np.ndarray, k: int) -> np.ndarray:
    """Rolling k-day sums along each row: (rows, days - k + 1)."""
    n = mat.shape[1]
    if n < k:
        return np.empty((mat.shape[0], 0))
    return np.stack([mat[:, s:s + k].sum(axis=1) for s in range(n - k + 1)], axis=1)


def _complete_starts(rain_day_max: int, window: int, starts) -> list[int]:
    """The week's start days whose whole rolling window lies within 1..rain_day_max."""
    last_start = int(rain_day_max) - int(window) + 1
    return [int(s) for s in starts if 1 <= int(s) <= last_start] if last_start >= 1 else []


def _over_starts(rolled: np.ndarray, starts, reduce) -> np.ndarray:
    """Per row, the max (np.fmax) or min (np.fmin) of the rolling sums at the start days (1-based), ignoring missing
    sums; NaN without any start."""
    ok = [s for s in starts if 1 <= s <= rolled.shape[1]]
    if rolled.shape[1] == 0 or not ok:
        return np.full(rolled.shape[0], np.nan)
    return reduce.reduce(rolled[:, [s - 1 for s in ok]], axis=1)


def daily_inputs(recipe: Recipe, climatology: Climatology, observed_onsets: ObservedOnsets,
                 forecasts: dict[str, ForecastOnsets], *, max_day: int, trim: bool | None = None) -> DailyInputs:
    """The random forest's inputs for the samples `weekly` builds (with the same trim)."""
    r = rows(recipe, climatology, observed_onsets, forecasts, trim=trim)
    clim = climatology.at("conditional", r.clim_unit, r.years, r.season_day, leads=slice(1, max_day + 1))
    variables: dict[str, tuple] = {"time": ("sample", r.times), "onset_lead": ("sample", _lead_days(r)),
                                   "clim_p_onset": (("sample", "clim_day"), clim)}
    coords = {"clim_day": np.arange(1, clim.shape[1] + 1)}
    for name, onsets in forecasts.items():
        rain = onsets.data["rain_mean"]
        variables[f"rain_{name}"] = (("sample", f"{name}_day"), rain.values[r.model_issue[name], r.model_unit[name]])
        coords[f"{name}_day"] = rain["day"].values
    return DailyInputs(xr.Dataset(variables, coords=coords))
