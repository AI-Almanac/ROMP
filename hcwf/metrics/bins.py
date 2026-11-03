"""Build forecast-observation pairs per day bin (from ensemble members or climatology) for probabilistic verification."""

from dataclasses import dataclass

import pandas as pd
import numpy as np
from hcwf.cache import memo
from hcwf.spatial import axes
import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeadBins:
    """The lead bins of a forecast: inner (first, last) day pairs, and the bins before and after them.

    Lead days count from the day after issue: lead 1 is the day after the issue date. The inner bins are
    HCWF's `day_bins` ((1, 5), (6, 10), ...); blending's week bins are `LeadBins.uniform(count, days)`. Around them
    sit the "before" bin (leads below the first day: an onset on or before the issue when the first day is 1)
    and the "after" bin (everything past the last day: blending's `later`).

    Labels come in two styles: "days" (HCWF: `Days a-b`, `Before day a`, `After day b`) and "week" (blending:
    `week1..weekN`, `earlier`, `later`; contiguous bins from lead 1 only). Bin positions run over
    labels(style, earlier=True), or over labels(style, earlier=False) when the before bin is left out.
    """

    inner: tuple

    def __post_init__(self):
        object.__setattr__(self, "inner", tuple((int(a), int(b)) for a, b in self.inner))
        if not self.inner:
            raise ValueError("lead bins need at least one inner bin")

    @classmethod
    def from_day_bins(cls, day_bins):
        """The bins of HCWF's `day_bins`, e.g. ((1, 5), (6, 10), (11, 15))."""
        return cls(tuple(day_bins))

    @classmethod
    def uniform(cls, count, days):
        """`count` bins of `days` days from lead 1 (blending's features.bins.count and days)."""
        return cls(tuple((k * days + 1, (k + 1) * days) for k in range(count)))

    @property
    def n_inner(self):
        return len(self.inner)

    @property
    def first_day(self):
        """The first lead day of the inner bins."""
        return min(self.inner)[0]

    @property
    def last_day(self):
        """The last lead day of the inner bins (HCWF's max forecast day)."""
        return max(self.inner)[1]

    @property
    def extended(self):
        """The inner bins with the open bins around them: ((-inf, first - 1), *inner, (last + 1, inf))."""
        return ((-float("inf"), self.first_day - 1),) + self.inner + ((self.last_day + 1, float("inf")),)

    @property
    def contiguous(self):
        """Whether the inner bins follow each other without a gap."""
        return all(a[1] + 1 == b[0] for a, b in zip(self.inner, self.inner[1:]))

    def label(self, k, style="days"):
        """Label of the bin at position k of `extended` (0: before, 1..n_inner: inner, n_inner + 1: after)."""
        if style == "days":
            if k == 0:
                return f"Before day {self.first_day}"
            if k == self.n_inner + 1:
                return f"After day {self.last_day}"
            return "Days %d-%d" % self.inner[k - 1]
        if style == "week":
            self._require_weeks()
            return "earlier" if k == 0 else "later" if k == self.n_inner + 1 else f"week{k}"
        raise ValueError(f"style must be 'days' or 'week', not {style!r}")

    def labels(self, style="days", *, earlier=True):
        """Labels in bin order: before (when `earlier`), the inner bins, after."""
        return [self.label(k, style) for k in range(0 if earlier else 1, self.n_inner + 2)]

    def _require_weeks(self):
        if self.first_day != 1 or not self.contiguous:
            raise ValueError(f"week labels need contiguous bins from lead 1, not {self.inner}")

    def in_bin(self, lead, k, *, nan_after=False):
        """Boolean array: which leads fall in the bin at position k of `extended`.

        A bin holds the leads from its first to its last day (the before bin everything below the first day, the
        after bin everything above the last day). NaN falls in no bin, or in the after bin with nan_after (a member
        with no onset in the window is an onset later than it).
        """
        lead = np.asarray(lead, dtype=float)
        if k == 0:
            return lead < self.first_day
        if k == self.n_inner + 1:
            return ~(lead <= self.last_day) if nan_after else lead > self.last_day
        start, end = self.inner[k - 1]
        return (start <= lead) & (lead <= end)

    def codes(self, lead, *, earlier=False):
        """Position (in labels(earlier=earlier)) of the bin of each lead; -1 for NaN and, without `earlier`, for
        leads below the first day.

        Needs contiguous bins; a lead past the last day is in the after bin. Blending's outcome codes.
        """
        if not self.contiguous:
            raise ValueError(f"codes need contiguous bins, not {self.inner}")
        lead = np.asarray(lead, dtype=float)
        edges = np.array([self.first_day - 1] + [end for _, end in sorted(self.inner)], dtype=float)
        with np.errstate(invalid="ignore"):
            codes = np.searchsorted(edges, lead, side="left") - 1 + (1 if earlier else 0)
            codes = np.where(lead <= edges[0], 0 if earlier else -1, codes)
        codes[np.isnan(lead)] = -1
        return codes.astype(np.int8)

    def column_slices(self):
        """Slices of the inner bins in an array whose columns are lead days 1, 2, ... (blending's day sums)."""
        return [slice(a - 1, b) for a, b in self.inner]

    @staticmethod
    def first_day_of(label):
        """The first lead day of a `Days a-b` label, 999 for any other label (to sort them last)."""
        if 'Days ' in label:
            try:
                return int(label.replace('Days ', '').split('-')[0])
            except ValueError:
                return 999
        return 999


# Function to create forecast-observation pairs with specified day bins for probabilistic verification


def forecast_pairs(onset_all_members, onset_da, day_bins, *, back_dated):
    """
    Create forecast-observation pairs using specified day bins, including a final bin for "after max_forecast_day".

    Parameters:
    -----------
    onset_all_members : DataFrame
        DataFrame with ensemble member onset predictions
    onset_da : xarray.DataArray
        Observed onset dates
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    back_dated : bool
        Whether the table also counts the members whose onset is dated on or before the init (reports_back_dated)

    Returns one row per forecast case (init_time, lat, lon, or init_time, unit; in sorted order) and bin; cases without an
    observed onset, or initialized on or after it, are left out.
    """

    bins = LeadBins.from_day_bins(day_bins)
    extended_bins = bins.extended

    # Unique combinations of init_time and location from the filtered forecast data
    spatial = axes.spatial_dims(onset_all_members.columns)
    forecast_groups = onset_all_members.groupby(['init_time', *spatial])

    log.debug(f"Processing {len(forecast_groups)} forecast cases with day bins: {day_bins}")
    log.debug(f"Including 'after day {bins.last_day}' bin for members without onset in forecast window")

    member_case = forecast_groups.ngroup().to_numpy()
    case_size = forecast_groups.size()
    cases = case_size.index
    total_members = case_size.to_numpy()
    init_times = cases.get_level_values('init_time')
    place = {d: cases.get_level_values(d).to_numpy(dtype=float if d != 'unit' else str) for d in spatial}

    # Observed onset of each case's location, a case without a matching grid point is left out
    where, located = axes.locate(onset_da, place)
    obs_date = np.full(len(cases), np.datetime64("NaT"), dtype="datetime64[ns]")
    obs_date[located] = onset_da.transpose(*spatial).values.astype("datetime64[ns]")[tuple(w[located] for w in where)]
    init_date = pd.DatetimeIndex(init_times).values.astype("datetime64[ns]")
    # Only use forecasts initialized before the observed onset
    kept = np.flatnonzero(located & ~np.isnat(obs_date) & (init_date < obs_date))

    member_day = onset_all_members['onset_day'].to_numpy(dtype=float)
    has_onset = ~np.isnan(member_day)
    init_kept = init_date[kept]
    # Lead day of each kept case's observed onset (the day after the init is lead 1)
    obs_lead = (obs_date[kept].astype("datetime64[D]") - init_kept.astype("datetime64[D]")).astype(np.int64)

    members_in_bin = []
    observed_onset = []
    labels = []
    for k in range(len(extended_bins)):
        labels.append(bins.label(k))
        # Members with no onset in the forecast window predict "after day X"
        member_in_bin = bins.in_bin(member_day, k, nan_after=True)
        observed = bins.in_bin(obs_lead, k)
        members_in_bin.append(
            np.bincount(member_case, weights=member_in_bin, minlength=len(cases)).astype(np.int64)[kept])
        observed_onset.append(observed.astype(np.int64))
    members_with_onset = np.stack(members_in_bin, axis=1)
    if back_dated:
        # members of the scored cases whose onset is dated on or before the init (and all members with an onset)
        in_kept = np.zeros(len(cases), dtype=bool)
        in_kept[kept] = True
        scored_member = in_kept[member_case]
        back_dated_members = np.bincount(member_case, weights=scored_member & has_onset & (member_day <= 0),
                                         minlength=len(cases)).astype(np.int64)[kept]
        onset_members = np.bincount(member_case, weights=scored_member & has_onset,
                                    minlength=len(cases)).astype(np.int64)[kept]
    else:
        back_dated_members = onset_members = None
    n_bins = len(extended_bins)

    def per_case(values):
        """Repeat each per-case value once for every bin."""
        return np.repeat(values, n_bins)

    forecast_obs_df = pd.DataFrame({
        'init_time': per_case(init_times[kept]) if len(kept) else [],
        **{d: per_case(place[d][kept]) for d in spatial},
        'bin_start': np.tile(np.array([b[0] for b in extended_bins], dtype=float), len(kept)),
        'bin_end': np.tile(np.array([b[1] for b in extended_bins], dtype=float), len(kept)),
        'bin_label': np.tile(np.array(labels, dtype=object), len(kept)),
        'predicted_prob': (members_with_onset / total_members[kept][:, None]).ravel(),
        'observed_onset': np.stack(observed_onset, axis=1).ravel(),
        'members_with_onset': members_with_onset.ravel(),
        'total_members': per_case(total_members[kept]),
        'year': per_case(pd.DatetimeIndex(init_kept).year.to_numpy().astype(np.int64)),
        'obs_onset_date': per_case(pd.DatetimeIndex(obs_date[kept]).strftime('%Y-%m-%d').to_numpy(dtype=object)),
        'bin_index': np.tile(np.arange(n_bins), len(kept)),
    })
    if back_dated_members is not None:
        forecast_obs_df['back_dated_members'] = per_case(back_dated_members)
        forecast_obs_df['onset_members'] = per_case(onset_members)

    log.debug(f"Generated {len(forecast_obs_df)} forecast-observation pairs")
    log.debug(f"Total bins per forecast: {len(extended_bins)}")
    log.debug(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    log.debug(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    log.debug(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Show distribution across bins
    log.debug(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean'
    }).round(3)
    log.debug(bin_stats)

    return forecast_obs_df


_YEAR_TEXT_BITS = 10
"""Bits of a year-set code looked up at once when writing its years as text (one table of 2**10 texts per group)."""
_YEAR_TEXT = memo("metrics.bins.year_text", maxsize=16)
"""Texts of year sets already written, {sorted years: {"groups": {(group, value): text}, "codes": {code: text}}}."""


def _year_set_texts(sorted_years, codes) -> np.ndarray:
    """Comma-separated text of the years of each code (bit n set = sorted_years[n] is in), as an object array.

    Equal to joining the years of the set bits in order, but a set is written from small per-group tables instead of
    looping over every year, and a code already written (by an earlier bin or call with these years) is reused.
    """
    held = _YEAR_TEXT.setdefault(tuple(int(y) for y in sorted_years), {"groups": {}, "codes": {}})
    texts, mask = held["codes"], (1 << _YEAR_TEXT_BITS) - 1
    n_groups = -(-len(sorted_years) // _YEAR_TEXT_BITS)

    def group_text(group, value):
        """The years of the set bits of `value` within one group of the codes."""
        key = (group, value)
        if key not in held["groups"]:
            base = group * _YEAR_TEXT_BITS
            held["groups"][key] = ",".join(str(sorted_years[base + n]) for n in range(_YEAR_TEXT_BITS)
                                           if (value >> n) & 1 and base + n < len(sorted_years))
        return held["groups"][key]

    out = np.empty(len(codes), dtype=object)
    for n, code in enumerate(codes.tolist()):
        if code not in texts:
            parts = (group_text(g, (code >> (g * _YEAR_TEXT_BITS)) & mask) for g in range(n_groups))
            texts[code] = ",".join(part for part in parts if part)
        out[n] = texts[code]
    return out


## This function creates forecast-observation pairs using climatological ensemble where each year is a member
def create_climatological_forecast_obs_pairs(clim_onset, target_year, init_dates, *,
                                             day_bins, max_forecast_day):
    """
    Create forecast-observation pairs using climatological ensemble where each year is a member.
    Uses day-of-year instead of calendar dates for onset comparison.

    Parameters:
    -----------
    clim_onset : xarray.DataArray
        3D array with dimensions [year, lat, lon] containing onset dates for all years
    target_year : int
        The year to use as "truth" for observations
    init_dates : list or pandas.DatetimeIndex
        Initialization dates for forecasts
    day_bins : list of tuples
        List of (start_day, end_day) tuples for bins within forecast window
        e.g., [(1, 5), (6, 10), (11, 15)]
    max_forecast_day : int, default=15
        Maximum forecast day

    Returns:
    --------
    DataFrame with forecast-observation pairs, rows in order of init date, lon, lat and bin
    """

    # the forecast window is the span of day_bins
    bins = LeadBins.from_day_bins(day_bins)
    max_forecast_day, min_forecast_day = bins.last_day, bins.first_day

    # Get the observed onset for the target year
    if target_year not in clim_onset.year.values:
        raise ValueError(f"Target year {target_year} not found in climatological dataset")

    obs_onset_da = clim_onset.sel(year=target_year)

    # Use ALL years as ensemble members (including target year) ### redundant
    ensemble_years = list(clim_onset.year.values)

    # Create extended bins including "before initialization" and "after max_forecast_day" bins
    extended_bins = bins.extended

    log.debug(f"Creating climatological forecasts for target year {target_year}")
    log.debug(f"Using {len(ensemble_years)} years as ensemble members: {ensemble_years}")
    log.debug(f"Processing {len(init_dates)} initialization dates")
    log.debug(f"Day bins: {day_bins}")
    log.debug(f"Extended bins include: 'Before day {min_forecast_day}' and 'After day {max_forecast_day}' ")
    log.debug(f"Using day-of-year method for onset comparison")

    # Get the actual coordinates (lat and lon, or unit ids) from the data
    spatial = axes.spatial_dims(obs_onset_da.dims)
    coords = axes.coords_of(obs_onset_da, spatial)
    n_places = int(np.prod([len(coords[d]) for d in spatial]))

    log.debug(f"Processing {n_places} unique {'/'.join(spatial)} locations")

    obs_onset = axes.snap(obs_onset_da.transpose(*spatial).values.astype("datetime64[ns]"), spatial, coords)
    has_obs = ~np.isnat(obs_onset)
    obs_onset_doy = np.where(has_obs, pd.DatetimeIndex(obs_onset.ravel()).dayofyear.to_numpy().reshape(obs_onset.shape),
                             0).astype(np.int64)
    ensemble_onset = axes.snap(clim_onset.transpose("year", *spatial).values.astype("datetime64[ns]"), spatial, coords,
                               lead=1)
    ensemble_doy = np.where(np.isnat(ensemble_onset), np.nan,
                            pd.DatetimeIndex(ensemble_onset.ravel()).dayofyear.to_numpy()
                            .reshape(ensemble_onset.shape).astype(float))

    inits = pd.DatetimeIndex([pd.to_datetime(init_date) for init_date in init_dates])
    init_doy = inits.dayofyear.to_numpy().astype(np.int64)

    # A case is a location with an observed onset and an init before it (by day of year),
    # ordered by init date, then lon, then lat (a grid's axes reversed; units by id)
    reverse = tuple(reversed(range(len(spatial))))
    cases = np.nonzero(has_obs.transpose(reverse)[None]
                       & (axes.broadcast_rows(init_doy, has_obs.ndim + 1) < obs_onset_doy.transpose(reverse)[None]))
    init_case, *backwards = cases
    where = tuple(reversed(backwards))
    case_obs_doy = obs_onset_doy[where]
    obs_days_from_init = case_obs_doy - init_doy[init_case]

    # Days from init of each year's onset at the case's location, NaN where the year has none
    member_days = ensemble_doy[(slice(None), *where)].T - init_doy[init_case][:, None]
    has_onset = ~np.isnan(member_days)

    members_in_bin = []
    contributing = []
    observed_onset = []
    labels = []
    for k in range(len(extended_bins)):
        labels.append(bins.label(k))
        members_in_bin.append(has_onset & bins.in_bin(member_days, k))
        # members whose onset is dated on or before the init also count for the Before bin's years
        contributing.append(has_onset & (member_days <= 0) if k == 0 else members_in_bin[-1])
        observed_onset.append(bins.in_bin(obs_days_from_init, k).astype(np.int64))

    members_with_onset = np.stack([in_bin.sum(axis=1) for in_bin in members_in_bin], axis=1)
    total_members_with_onset = members_with_onset.sum(axis=1)
    # Skip if no members showed onset
    kept = np.flatnonzero(total_members_with_onset > 0)
    n_bins = len(extended_bins)

    # The years of each bin's members as comma-separated text, built once per distinct set of years over all bins
    year_order = np.argsort(np.array(ensemble_years), kind="stable")
    sorted_years = [ensemble_years[n] for n in year_order]
    place_value = 1 << np.arange(len(ensemble_years), dtype=np.int64)
    in_bins = [in_bin[kept][:, year_order] for in_bin in contributing]
    sets, which = np.unique(np.stack([in_bin.astype(np.int64) @ place_value for in_bin in in_bins], axis=1),
                            return_inverse=True)
    names = _year_set_texts(sorted_years, sets)
    contributing_years = names[which.reshape(len(kept), n_bins)]
    n_contributing_years = np.stack([in_bin.sum(axis=1) for in_bin in in_bins], axis=1).astype(np.int64)

    def per_case(values):
        """Repeat each per-case value once for every bin."""
        return np.repeat(values, n_bins)

    def per_bin(values):
        """Tile the per-bin values across all kept cases."""
        return np.tile(values, len(kept))

    # Predicted probability uses only members that showed onset
    members_kept = members_with_onset[kept]
    obs_days_kept = obs_days_from_init[kept]
    obs_days_column = np.repeat(obs_days_kept[:, None], n_bins, axis=1)

    forecast_obs_df = pd.DataFrame({
        'init_time': per_case(inits.strftime('%Y-%m-%d').to_numpy(dtype=object)[init_case[kept]]),
        **{d: per_case(c) for d, c in axes.columns_of(spatial, coords, tuple(w[kept] for w in where)).items()},
        'bin_start': per_bin(np.array([b[0] for b in extended_bins], dtype=float)),
        'bin_end': per_bin(np.array([b[1] for b in extended_bins], dtype=float)),
        'bin_label': per_bin(np.array(labels, dtype=object)),
        'predicted_prob': (members_kept / total_members_with_onset[kept][:, None]).ravel(),
        'observed_onset': np.stack(observed_onset, axis=1)[kept].ravel(),
        'members_with_onset': members_kept.ravel(),
        'total_members': len(ensemble_years),
        'total_members_with_onset': per_case(total_members_with_onset[kept]),
        'contributing_years': contributing_years.ravel(),
        'n_contributing_years': n_contributing_years.ravel(),
        'year': target_year,
        'obs_onset_date': per_case(pd.DatetimeIndex(obs_onset[tuple(w[kept] for w in where)])
                                   .strftime('%Y-%m-%d').to_numpy(dtype=object)),
        'obs_onset_doy': per_case(case_obs_doy[kept]),
        'init_doy': per_case(init_doy[init_case[kept]]),
        'obs_days_from_init_doy': obs_days_column.ravel(),
        'bin_index': per_bin(np.arange(n_bins)),
        'forecast_type': 'climatological_doy',
    })

    if len(forecast_obs_df) == 0:
        log.warning("No forecast-observation pairs generated")
        return forecast_obs_df

    log.debug(f"Generated {len(forecast_obs_df)} climatological forecast-observation pairs")
    log.debug(f"Unique {'/'.join(spatial)} locations processed: {n_places}")
    log.debug(f"Total bins per forecast: {len(extended_bins)}")
    log.debug(f"Probability range: {forecast_obs_df['predicted_prob'].min():.3f} - {forecast_obs_df['predicted_prob'].max():.3f}")
    log.debug(f"Observed onset rate: {forecast_obs_df['observed_onset'].mean():.3f}")
    log.debug(f"Non-zero probabilities: {(forecast_obs_df['predicted_prob'] > 0).sum()}")

    # Verify uniqueness
    unique_locations_in_output = len(forecast_obs_df[list(spatial)].drop_duplicates())
    log.debug(f"Unique locations in output: {unique_locations_in_output}")

    # Show distribution across bins
    log.debug(f"\nDistribution across bins:")
    bin_stats = forecast_obs_df.groupby('bin_label').agg({
        'predicted_prob': ['count', 'mean'],
        'observed_onset': 'mean',
        'n_contributing_years': 'mean',
        'total_members_with_onset': 'mean'
    }).round(3)
    log.debug(bin_stats)

    return forecast_obs_df


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
