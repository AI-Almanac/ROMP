"""Build forecast-observation pairs per day bin (from ensemble members or climatology) for probabilistic verification."""

from dataclasses import dataclass, replace

import pandas as pd
import numpy as np
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


@dataclass(frozen=True)
class PairArrays:
    """The forecast-observation pairs of a model, reference or climatology as arrays: one row per case, one column per bin.

    A case is a forecast of one init and place (a grid cell, or a unit) whose observed onset lies after the init.
    `prob` and `observed` are (case, bin) matrices of the predicted probability of an onset in the bin and whether
    the observed onset fell in it (0/1); `members` is the ensemble size behind each case's probabilities (inf for a
    conditional climatology). `start`, `end` and `labels` describe the bins once (the open before and after bins have
    infinite edges). `init` and `place` ({dim: values}, as the builders order the cases) identify the cases.
    Optional per-case counts: `members_with_onset` (climatologies: members with an onset, the size behind their
    probabilities) and `back_dated` / `onset_members` (the members whose onset is dated on or before the init, and all
    members with an onset; forecast_history "observed" runs).

    The arrays are never changed after building: a verification window (`select`) and the years joined (`concat`) are
    new containers, the window's matrices views of the originals where its bins are adjacent.
    """

    prob: np.ndarray
    observed: np.ndarray
    members: np.ndarray
    start: np.ndarray
    end: np.ndarray
    labels: tuple
    init: np.ndarray
    place: dict
    members_with_onset: np.ndarray | None = None
    back_dated: np.ndarray | None = None
    onset_members: np.ndarray | None = None

    _PER_CASE = ("prob", "observed", "members", "init", "members_with_onset", "back_dated", "onset_members")

    @property
    def n_cases(self):
        return len(self.prob)

    @property
    def n_bins(self):
        return len(self.labels)

    @classmethod
    def empty(cls, bins, spatial):
        """No cases, with the bins of `bins` (LeadBins) and the place dimensions `spatial`."""
        n_bins = len(bins.extended)
        return cls(prob=np.empty((0, n_bins)), observed=np.empty((0, n_bins), dtype=np.int8),
                   members=np.empty(0, dtype=np.int32), **bin_fields(bins), init=np.empty(0, dtype="datetime64[D]"),
                   place={d: np.empty(0) for d in spatial})

    @classmethod
    def concat(cls, parts):
        """The cases of `parts` one after the other (the years of a run, in order); they must share their bins."""
        parts = list(parts)
        first = parts[0]
        for part in parts[1:]:
            if part.labels != first.labels or list(part.place) != list(first.place):
                raise ValueError("pairs with different bins or places cannot be joined")
        columns = {name: [getattr(part, name) for part in parts] for name in cls._PER_CASE}
        joined = {name: None if any(v is None for v in values) else np.concatenate(values)
                  for name, values in columns.items()}
        place = {d: np.concatenate([part.place[d] for part in parts]) for d in first.place}
        return replace(first, place=place, **joined)

    def bins_where(self, keep):
        """The pairs of the bins whose position is flagged in `keep` (bool per bin), a view where they are adjacent."""
        at = np.flatnonzero(keep)
        columns = slice(at[0], at[-1] + 1) if len(at) and at[-1] - at[0] + 1 == len(at) else at
        return replace(self, prob=self.prob[:, columns], observed=self.observed[:, columns], start=self.start[at],
                       end=self.end[at], labels=tuple(self.labels[k] for k in at))

    def select(self, day_bins):
        """The pairs of the bins whose (first day, last day) is one of `day_bins` (a verification window's bins), in
        the order they are stored."""
        wanted = {(float(a), float(b)) for a, b in day_bins}
        return self.bins_where([(s, e) in wanted for s, e in zip(self.start.tolist(), self.end.tolist())])

    def without_before(self):
        """The pairs without their "Before day N" bins."""
        return self.bins_where(["Before" not in label for label in self.labels])

    def sorted_order(self):
        """Positions of the cases ordered by init and then place (the places in the order of their dimensions)."""
        return np.lexsort((*(self.place[d] for d in reversed(list(self.place))), self.init))


def bin_fields(bins):
    """The bin edges and labels of `bins` (LeadBins) as the PairArrays fields."""
    extended = bins.extended
    return {"start": np.array([b[0] for b in extended], dtype=float), "end": np.array([b[1] for b in extended], dtype=float),
            "labels": tuple(bins.labels())}


def _log_pairs(pairs):
    """Debug lines on the size and range of the pairs."""
    if not log.isEnabledFor(logging.DEBUG) or not pairs.prob.size:
        return
    log.debug(f"Generated {pairs.n_cases} cases of {pairs.n_bins} bins each")
    log.debug(f"Probability range: {pairs.prob.min():.3f} - {pairs.prob.max():.3f}")
    log.debug(f"Observed onset rate: {pairs.observed.mean():.3f}")
    log.debug(f"Non-zero probabilities: {(pairs.prob > 0).sum()}")


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

    Returns the PairArrays of the forecast cases (init, then lat and lon or unit; in sorted order); cases without an
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

    total_kept = total_members[kept]
    prob = np.empty((len(kept), len(extended_bins)))
    observed = np.empty((len(kept), len(extended_bins)), dtype=np.int8)
    for k in range(len(extended_bins)):
        # Members with no onset in the forecast window predict "after day X"
        member_in_bin = bins.in_bin(member_day, k, nan_after=True)
        members_in_bin = np.bincount(member_case, weights=member_in_bin, minlength=len(cases)).astype(np.int64)[kept]
        prob[:, k] = members_in_bin / total_kept
        observed[:, k] = bins.in_bin(obs_lead, k)
    if back_dated:
        # members of the scored cases whose onset is dated on or before the init (and all members with an onset)
        in_kept = np.zeros(len(cases), dtype=bool)
        in_kept[kept] = True
        scored_member = in_kept[member_case]
        back_dated_members = np.bincount(member_case, weights=scored_member & has_onset & (member_day <= 0),
                                         minlength=len(cases)).astype(np.int32)[kept]
        onset_members = np.bincount(member_case, weights=scored_member & has_onset,
                                    minlength=len(cases)).astype(np.int32)[kept]
    else:
        back_dated_members = onset_members = None

    pairs = PairArrays(prob=prob, observed=observed, members=total_kept.astype(np.int32),
                       **bin_fields(bins), init=init_kept.astype("datetime64[D]"),
                       place={d: place[d][kept] for d in spatial},
                       back_dated=back_dated_members, onset_members=onset_members)
    _log_pairs(pairs)
    return pairs


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
    PairArrays of the cases (a place with an observed onset and an init before it, then a member with an onset), in
    order of init date, lon, lat
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
    log.debug("Using day-of-year method for onset comparison")

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
    for k in range(len(extended_bins)):
        members_in_bin.append(has_onset & bins.in_bin(member_days, k))

    members_with_onset = np.stack([in_bin.sum(axis=1) for in_bin in members_in_bin], axis=1)
    total_members_with_onset = members_with_onset.sum(axis=1)
    # Skip if no members showed onset
    kept = np.flatnonzero(total_members_with_onset > 0)
    obs_days_kept = obs_days_from_init[kept]
    observed = np.stack([bins.in_bin(obs_days_kept, k) for k in range(len(extended_bins))], axis=1).astype(np.int8)

    # Predicted probability uses only members that showed onset
    total_kept = total_members_with_onset[kept]
    pairs = PairArrays(prob=members_with_onset[kept] / total_kept[:, None], observed=observed,
                       members=np.full(len(kept), len(ensemble_years), dtype=np.int32),
                       **bin_fields(bins), init=inits.values.astype("datetime64[D]")[init_case[kept]],
                       place=axes.columns_of(spatial, coords, tuple(w[kept] for w in where)),
                       members_with_onset=total_kept.astype(np.int32))

    if pairs.n_cases == 0:
        log.warning("No forecast-observation pairs generated")
        return pairs

    log.debug(f"Unique {'/'.join(spatial)} locations processed: {n_places}")
    _log_pairs(pairs)
    return pairs
