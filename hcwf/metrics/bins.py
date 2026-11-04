"""Build forecast-observation pairs per day bin (from ensemble members or climatology) for probabilistic verification.

The module holds `LeadBins` (the lead bins of a forecast and their labels), `PairArrays` (the pairs of one model,
reference or climatology as arrays, one row per case and one column per bin) and the two builders of pairs:
`forecast_pairs` (from the members of a forecast onset table) and `create_climatological_forecast_obs_pairs` (from the
years of a climatology). `hcwf.scoring.pairs` prepares the pairs of a run with them and `hcwf.metrics.score` scores them.
"""

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
    HCWF's ``day_bins`` (``((1, 5), (6, 10), ...)``); blending's week bins are ``LeadBins.uniform(count, days)``. Around
    them sit the "before" bin (leads below the first day: an onset on or before the issue when the first day is 1)
    and the "after" bin (everything past the last day: blending's ``later``).

    Labels come in two styles: ``"days"`` (HCWF: ``Days a-b``, ``Before day a``, ``After day b``) and ``"week"``
    (blending: ``week1..weekN``, ``earlier``, ``later``; contiguous bins from lead 1 only). Bin positions run over
    ``labels(style, earlier=True)``, or over ``labels(style, earlier=False)`` when the before bin is left out.
    """

    inner: tuple
    """The inner bins as ``(first day, last day)`` pairs of int, both ends inclusive; at least one."""

    def __post_init__(self):
        """Normalise `inner` to a tuple of int pairs.

        Raises
        ------
        ValueError
            There is no inner bin.
        """
        object.__setattr__(self, "inner", tuple((int(a), int(b)) for a, b in self.inner))
        if not self.inner:
            raise ValueError("lead bins need at least one inner bin")

    @classmethod
    def from_day_bins(cls, day_bins):
        """Return the bins of HCWF's ``day_bins``, e.g. ``((1, 5), (6, 10), (11, 15))``.

        Parameters
        ----------
        day_bins : sequence of (int, int)
            The ``(first day, last day)`` pairs of the inner bins.

        Returns
        -------
        LeadBins
            The bins.
        """
        return cls(tuple(day_bins))

    @classmethod
    def uniform(cls, count, days):
        """Return ``count`` bins of ``days`` days each from lead 1 (blending's ``features.bins.count`` and ``days``).

        Parameters
        ----------
        count : int
            The number of bins.
        days : int
            The length of each bin in days.

        Returns
        -------
        LeadBins
            The contiguous bins ``(1, days), (days + 1, 2 * days), ...``.
        """
        return cls(tuple((k * days + 1, (k + 1) * days) for k in range(count)))

    @property
    def n_inner(self):
        """The number of inner bins."""
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
        """The inner bins with the open bins around them: ``((-inf, first - 1), *inner, (last + 1, inf))``."""
        return ((-float("inf"), self.first_day - 1),) + self.inner + ((self.last_day + 1, float("inf")),)

    @property
    def contiguous(self):
        """Whether the inner bins follow each other without a gap."""
        return all(a[1] + 1 == b[0] for a, b in zip(self.inner, self.inner[1:]))

    def label(self, k, style="days"):
        """Return the label of the bin at position ``k`` of `extended`.

        Parameters
        ----------
        k : int
            The position: 0 is the before bin, 1 to `n_inner` the inner bins, ``n_inner + 1`` the after bin.
        style : {"days", "week"}, default "days"
            The labelling style (see the class description).

        Returns
        -------
        str
            The label, e.g. ``"Days 6-10"`` or ``"week2"``.

        Raises
        ------
        ValueError
            ``style`` is neither ``"days"`` nor ``"week"``, or ``"week"`` is asked of bins that are not contiguous from
            lead 1.
        """
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
        """Return the labels in bin order: before (when ``earlier``), the inner bins, after.

        Parameters
        ----------
        style : {"days", "week"}, default "days"
            The labelling style (see the class description).
        earlier : bool, default True
            Include the label of the before bin.

        Returns
        -------
        list of str
            The labels.
        """
        return [self.label(k, style) for k in range(0 if earlier else 1, self.n_inner + 2)]

    def _require_weeks(self):
        """Check that the bins can carry week labels.

        Raises
        ------
        ValueError
            The bins do not start at lead 1 or are not contiguous.
        """
        if self.first_day != 1 or not self.contiguous:
            raise ValueError(f"week labels need contiguous bins from lead 1, not {self.inner}")

    def in_bin(self, lead, k, *, nan_after=False):
        """Return which leads fall in the bin at position ``k`` of `extended`.

        A bin holds the leads from its first to its last day (the before bin everything below the first day, the
        after bin everything above the last day). NaN falls in no bin, or in the after bin with ``nan_after`` (a member
        with no onset in the window is an onset later than it).

        Parameters
        ----------
        lead : array_like of float
            The lead days (NaN for no onset), any shape.
        k : int
            The bin position in `extended`.
        nan_after : bool, default False
            Put NaN leads in the after bin.

        Returns
        -------
        numpy.ndarray of bool
            ``True`` where the lead is in the bin; the shape of ``lead``.
        """
        lead = np.asarray(lead, dtype=float)
        if k == 0:
            return lead < self.first_day
        if k == self.n_inner + 1:
            return ~(lead <= self.last_day) if nan_after else lead > self.last_day
        start, end = self.inner[k - 1]
        return (start <= lead) & (lead <= end)

    def codes(self, lead, *, earlier=False):
        """Return the position of the bin of each lead, in ``labels(earlier=earlier)``.

        The code is -1 for NaN and, without ``earlier``, for leads below the first day. Needs contiguous bins; a lead
        past the last day is in the after bin. These are blending's outcome codes.

        Parameters
        ----------
        lead : array_like of float
            The lead days (NaN for no onset), any shape.
        earlier : bool, default False
            Count the before bin as position 0 (the codes then run over the labels with the before bin).

        Returns
        -------
        numpy.ndarray of int8
            The bin position of each lead; the shape of ``lead``.

        Raises
        ------
        ValueError
            The inner bins are not contiguous.
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
        """Return the slices of the inner bins in an array whose columns are lead days 1, 2, ... (blending's day sums).

        Returns
        -------
        list of slice
            One ``slice(first - 1, last)`` per inner bin.
        """
        return [slice(a - 1, b) for a, b in self.inner]

    @staticmethod
    def first_day_of(label):
        """Return the first lead day of a ``Days a-b`` label, 999 for any other label (to sort them last).

        Parameters
        ----------
        label : str
            A bin label.

        Returns
        -------
        int
            The first day ``a``, or 999.
        """
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
    infinite edges). `init` and `place` (``{dim: values}``, as the builders order the cases) identify the cases.
    Optional per-case counts: `members_with_onset` (climatologies: members with an onset, the size behind their
    probabilities) and `back_dated` / `onset_members` (the members whose onset is dated on or before the init, and all
    members with an onset; ``forecast_history: observed`` runs).

    The arrays are never changed after building: a verification window (`select`) and the years joined (`concat`) are
    new containers, the window's matrices views of the originals where its bins are adjacent.
    """

    prob: np.ndarray
    """Predicted probability of an onset in each bin, ``float64``, shape (case, bin)."""
    observed: np.ndarray
    """Whether the observed onset fell in each bin (0 or 1), ``int8``, shape (case, bin)."""
    members: np.ndarray
    """The ensemble size behind each case's probabilities, ``int32``, shape (case,)."""
    start: np.ndarray
    """First lead day of each bin (``-inf`` for the before bin), ``float64``, shape (bin,)."""
    end: np.ndarray
    """Last lead day of each bin (``inf`` for the after bin), ``float64``, shape (bin,)."""
    labels: tuple
    """The label of each bin, ``str``, one per bin."""
    init: np.ndarray
    """The init date of each case, ``datetime64[D]``, shape (case,)."""
    place: dict
    """The place of each case as ``{dim: values}`` (``lat`` and ``lon`` as float, or ``unit``), each shape (case,)."""
    members_with_onset: np.ndarray | None = None
    """Climatologies only: the members with an onset behind each case, ``int32``, shape (case,)."""
    back_dated: np.ndarray | None = None
    """Members whose onset is dated on or before the init, ``int32``, shape (case,); None when not tracked."""
    onset_members: np.ndarray | None = None
    """Members with an onset, ``int32``, shape (case,); None when not tracked."""

    _PER_CASE = ("prob", "observed", "members", "init", "members_with_onset", "back_dated", "onset_members")
    """The names of the fields that have one entry (row) per case, which `concat` joins."""

    @property
    def n_cases(self):
        """The number of cases (rows)."""
        return len(self.prob)

    @property
    def n_bins(self):
        """The number of bins (columns)."""
        return len(self.labels)

    @classmethod
    def empty(cls, bins, spatial):
        """Return pairs with no cases.

        Parameters
        ----------
        bins : LeadBins
            The bins whose edges and labels the pairs carry.
        spatial : sequence of str
            The place dimensions (``lat`` and ``lon``, or ``unit``).

        Returns
        -------
        PairArrays
            Pairs with zero rows.
        """
        n_bins = len(bins.extended)
        return cls(prob=np.empty((0, n_bins)), observed=np.empty((0, n_bins), dtype=np.int8),
                   members=np.empty(0, dtype=np.int32), **bin_fields(bins), init=np.empty(0, dtype="datetime64[D]"),
                   place={d: np.empty(0) for d in spatial})

    @classmethod
    def concat(cls, parts):
        """Join the cases of ``parts`` one after the other (the years of a run, in order).

        An optional per-case field is joined only when every part has it, otherwise it is None.

        Parameters
        ----------
        parts : iterable of PairArrays
            The pairs to join; at least one, all with the same bins and place dimensions.

        Returns
        -------
        PairArrays
            The joined pairs, with the bins of the first part.

        Raises
        ------
        ValueError
            The parts differ in their bin labels or place dimensions.
        """
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
        """Return the pairs of the bins flagged in ``keep``, with the matrices as views where the bins are adjacent.

        Parameters
        ----------
        keep : array_like of bool, shape (bin,)
            Which bin positions to keep.

        Returns
        -------
        PairArrays
            The pairs of the kept bins; the per-case fields other than `prob` and `observed` are shared.
        """
        at = np.flatnonzero(keep)
        columns = slice(at[0], at[-1] + 1) if len(at) and at[-1] - at[0] + 1 == len(at) else at
        return replace(self, prob=self.prob[:, columns], observed=self.observed[:, columns], start=self.start[at],
                       end=self.end[at], labels=tuple(self.labels[k] for k in at))

    def select(self, day_bins):
        """Return the pairs of the bins whose ``(first day, last day)`` is one of ``day_bins``, in the stored order.

        Parameters
        ----------
        day_bins : iterable of (int, int)
            A verification window's bins (`hcwf.config.model.Window.bins`).

        Returns
        -------
        PairArrays
            The pairs of the matching bins.
        """
        wanted = {(float(a), float(b)) for a, b in day_bins}
        return self.bins_where([(s, e) in wanted for s, e in zip(self.start.tolist(), self.end.tolist())])

    def without_before(self):
        """Return the pairs without their ``Before day N`` bins.

        Returns
        -------
        PairArrays
            The pairs of the other bins.
        """
        return self.bins_where(["Before" not in label for label in self.labels])

    def sorted_order(self):
        """Return the positions of the cases ordered by init and then place (the places in the order of their dimensions).

        Returns
        -------
        numpy.ndarray of int64, shape (case,)
            The permutation that sorts the cases.
        """
        return np.lexsort((*(self.place[d] for d in reversed(list(self.place))), self.init))


def bin_fields(bins):
    """Return the bin edges and labels of ``bins`` as the `PairArrays` fields.

    Parameters
    ----------
    bins : LeadBins
        The bins.

    Returns
    -------
    dict
        ``start`` and ``end`` (``float64`` arrays over the extended bins) and ``labels`` (a tuple of str).
    """
    extended = bins.extended
    return {"start": np.array([b[0] for b in extended], dtype=float), "end": np.array([b[1] for b in extended], dtype=float),
            "labels": tuple(bins.labels())}


def _log_pairs(pairs):
    """Log debug lines on the size and range of the pairs.

    Parameters
    ----------
    pairs : PairArrays
        The pairs; nothing is logged unless debug logging is on and there are pairs.

    Returns
    -------
    None
        Nothing is returned.
    """
    if not log.isEnabledFor(logging.DEBUG) or not pairs.prob.size:
        return
    log.debug(f"Generated {pairs.n_cases} cases of {pairs.n_bins} bins each")
    log.debug(f"Probability range: {pairs.prob.min():.3f} - {pairs.prob.max():.3f}")
    log.debug(f"Observed onset rate: {pairs.observed.mean():.3f}")
    log.debug(f"Non-zero probabilities: {(pairs.prob > 0).sum()}")


# Function to create forecast-observation pairs with specified day bins for probabilistic verification


def forecast_pairs(onset_all_members, onset_da, day_bins, *, back_dated):
    """Create forecast-observation pairs from the members of a forecast, with an open bin on each side of the day bins.

    The bins are the day bins plus a before bin and a final "after the last day" bin that takes the members with no onset
    in the forecast window. A case is one init and place; it is left out when it has no observed onset or was
    initialised on or after it.

    Parameters
    ----------
    onset_all_members : pandas.DataFrame
        One row per member of a forecast case, with ``init_time``, the spatial columns (``lat`` and ``lon``, or ``unit``)
        and ``onset_day`` (the member's onset in days from the init, NaN for none).
    onset_da : xarray.DataArray
        Observed onset dates (``datetime64``) on the spatial dims.
    day_bins : sequence of (int, int)
        The ``(first day, last day)`` bins within the forecast window, e.g. ``[(1, 5), (6, 10), (11, 15)]``.
    back_dated : bool
        Whether the table also counts the members whose onset is dated on or before the init
        (`hcwf.scoring.onset_tables.reports_back_dated`); fills `PairArrays.back_dated` and `PairArrays.onset_members`.

    Returns
    -------
    PairArrays
        The forecast cases (init, then lat and lon or unit; in sorted order).
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
                                             day_bins):
    """Create forecast-observation pairs from a climatological ensemble in which each year is a member.

    Uses day of year instead of calendar dates for the onset comparison. All years of ``clim_onset`` (the target year
    included) are the members; the target year's onset is the observation. A case is a place with an observed onset and
    an init before it, kept when at least one member has an onset; the probabilities use only the members with an onset.

    Parameters
    ----------
    clim_onset : xarray.DataArray
        Onset dates (``datetime64``) of all years, dims (year, lat, lon) or (year, unit).
    target_year : int
        The year used as the truth for the observations.
    init_dates : sequence of datetime-like
        The initialisation dates of the forecasts.
    day_bins : sequence of (int, int)
        The ``(first day, last day)`` bins within the forecast window, e.g. ``[(1, 5), (6, 10), (11, 15)]``.

    Returns
    -------
    PairArrays
        The cases in order of init date, lon, lat. `PairArrays.members` is the number of years and
        `PairArrays.members_with_onset` the members behind each case's probabilities.

    Raises
    ------
    ValueError
        ``target_year`` is not in ``clim_onset``.
    """
    # the forecast window is the span of day_bins
    bins = LeadBins.from_day_bins(day_bins)
    max_forecast_day, min_forecast_day = bins.last_day, bins.first_day   # the last day of the bins is the maximum

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
