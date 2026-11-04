"""What the blend works on: onsets per unit, and the weekly samples built from them.

`ObservedOnsets` (observed onsets over unit and year) and `ForecastOnsets` (a model's member onset counts per issue,
unit and lead day) are what `hcwf.onset.detect` returns for evaluation units, collected over files
(`hcwf.blend.features`). `Weekly` is one sample per unit and issue date with a column per feature (`Weekly.take`
gives a column's values per sample, for ``rows`` (positions) or every sample; `Weekly.frame` rebuilds the legacy
table); `DailyInputs` is what the random forest reads for each sample, in the samples' order. The blend methods
(`hcwf.blend.logit`, `hcwf.blend.random_forest`, `hcwf.blend.design`, `hcwf.blend.predictions`) read a `Weekly`
through ``take``, ``__getitem__``, ``columns``, ``attrs``, ``column_dtypes``, ``frame`` and ``outcome_codes`` only,
so a table with those (``tests/blend/_samples.py``) serves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from .predictions import outcome_codes, outcome_series

__all__ = ["DailyInputs", "ForecastOnsets", "ObservedOnsets", "Weekly"]


def _as_dates(values: np.ndarray) -> np.ndarray:
    """Return datetime64 values as date objects, NaT as NaT, converting each distinct value once.

    Parameters
    ----------
    values : numpy.ndarray of datetime64[ns]
        The timestamps, 1-D (what `pandas.factorize` accepts).

    Returns
    -------
    numpy.ndarray of object
        ``datetime.date`` objects, or ``pandas.NaT`` where the input is NaT; same shape as `values`.
    """
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    stamps = pd.DatetimeIndex(distinct)
    out = np.array([pd.NaT if pd.isna(t) else t.date() for t in stamps], dtype=object)
    return out[codes]


@dataclass
class ObservedOnsets:
    """Observed onsets over (unit, year).

    The variables of `data` are ``onset_day`` (days after the season start; NaN when there is none), ``onset_idx``
    (the day in its season's file, 1-based) and ``present`` (the unit had a series that year).
    """

    data: xr.Dataset
    """Dataset with ``float64`` ``onset_day`` and ``onset_idx`` and ``bool`` ``present``, dims ``(unit, year)``."""
    season_start: str
    """The month-day the season starts, ``"MM-DD"``."""

    @classmethod
    def from_parts(cls, parts: list[dict], season_start: str) -> "ObservedOnsets":
        """Build the onsets from per-file summaries, merging them over the union of units and years.

        Parameters
        ----------
        parts : list of dict
            One summary per observation file with ``ids`` (unit ids), ``years``, and ``onset_day``, ``onset_idx``,
            ``present`` as arrays of shape ``(unit, year)`` of that file.
        season_start : str
            The month-day the season starts, ``"MM-DD"``.

        Returns
        -------
        ObservedOnsets
            The merged onsets; units are sorted strings, years sorted integers.

        Raises
        ------
        ValueError
            Two files hold the same unit and year.
        """
        units = np.unique(np.concatenate([p["ids"] for p in parts]).astype(str))
        years = np.unique(np.concatenate([p["years"] for p in parts]).astype(np.int64))
        shape = (len(units), len(years))
        onset_day, onset_idx = np.full(shape, np.nan), np.full(shape, np.nan)
        present = np.zeros(shape, dtype=bool)
        for part in parts:
            u = np.searchsorted(units, part["ids"].astype(str))
            y = np.searchsorted(years, part["years"].astype(np.int64))
            rows, cols = np.nonzero(part["present"])
            if present[u[rows], y[cols]].any():
                raise ValueError("Two observation files hold the same unit and season")
            onset_day[u[rows], y[cols]] = part["onset_day"][rows, cols]
            onset_idx[u[rows], y[cols]] = part["onset_idx"][rows, cols]
            present[u[rows], y[cols]] = True
        data = xr.Dataset(
            {"onset_day": (("unit", "year"), onset_day), "onset_idx": (("unit", "year"), onset_idx),
             "present": (("unit", "year"), present)},
            coords={"unit": units.astype(object), "year": years})
        return cls(data=data, season_start=season_start)

    @property
    def years(self) -> list[int]:
        """The years of the onsets, in the order of the ``year`` dimension."""
        return [int(y) for y in self.data["year"].values]

    def cutoff_dates(self) -> np.ndarray:
        """Return the season start of each year.

        Returns
        -------
        numpy.ndarray of datetime64[D], shape (year,)
            The date ``<year>-<season_start>``.
        """
        return np.array([np.datetime64(f"{y}-{self.season_start}") for y in self.years], dtype="datetime64[D]")

    def onset_dates(self) -> np.ndarray:
        """Return the onset date of each unit and year.

        Returns
        -------
        numpy.ndarray of datetime64[D], shape (unit, year)
            The season start plus ``onset_day``; NaT where there is no onset.
        """
        day = self.data["onset_day"].values
        dates = self.cutoff_dates()[np.newaxis, :] + np.where(np.isnan(day), 0, day).astype("timedelta64[D]")
        return np.where(np.isnan(day), np.datetime64("NaT"), dates)

    def onsets(self) -> pd.DataFrame:
        """Return the onsets of every present unit and season, season by season, units in order.

        This is what the climatology is fitted on.

        Returns
        -------
        pandas.DataFrame
            Columns ``id`` (str), ``year`` (int64) and ``onset_day`` (float64, NaN without an onset).
        """
        y, u = np.nonzero(self.data["present"].values.T)
        return pd.DataFrame({
            "id": pd.Series(self.data["unit"].values[u], dtype=str),
            "year": self.data["year"].values[y].astype(np.int64),
            "onset_day": self.data["onset_day"].values.T[y, u]})

    def frame(self) -> pd.DataFrame:
        """Return the legacy table: one row per present unit and season.

        Returns
        -------
        pandas.DataFrame
            Columns ``id``, ``year``, ``onset_idx``, ``onset_date`` (``datetime.date`` or None), ``onset_day`` and
            ``cutoff_date`` (``datetime.date``, the season start).
        """
        onsets = self.onsets()
        y, u = np.nonzero(self.data["present"].values.T)
        dates = self.onset_dates().T[y, u]
        cutoffs = self.cutoff_dates()[y]
        return pd.DataFrame({
            "id": onsets["id"], "year": onsets["year"], "onset_idx": self.data["onset_idx"].values.T[y, u],
            "onset_date": pd.Series(_as_dates(dates.astype("datetime64[ns]")), dtype=object).where(
                ~np.isnat(dates), None),
            "onset_day": onsets["onset_day"],
            "cutoff_date": pd.Series(_as_dates(cutoffs.astype("datetime64[ns]")), dtype=object)})

    def blendable(self, min_onset_years: int) -> frozenset[str]:
        """Return the units with enough observed onsets to blend.

        The climatology keeps only these units, and the inner join drops every forecast unit it lacks.

        Parameters
        ----------
        min_onset_years : int
            The least number of seasons with an observed onset a unit needs.

        Returns
        -------
        frozenset of str
            The ids of units with at least `min_onset_years` observed onsets over every season.
        """
        counts = (~np.isnan(self.data["onset_day"].values)).sum(axis=1)
        return frozenset(self.data["unit"].values[counts >= int(min_onset_years)].astype(str))


@dataclass
class ForecastOnsets:
    """A model's forecast onsets as arrays over (issue, unit, day).

    The variables of `data` are ``count_<variant>`` (members whose onset falls on each lead day, ``uint8``; variants
    ``base``, ``fixed_cutoff``, ``ref``), ``members`` (per issue), ``rain_mean`` (the members' mean rainfall) and
    ``onset_threshold`` (per unit). Probabilities are counts over members, computed on demand.
    """

    model: str
    """The model's name."""
    data: xr.Dataset | None
    """The arrays over ``(issue, unit, day)``, or None once `release` has dropped them."""
    years: list[int] = field(default_factory=list)
    """The years of the issues; taken from `data` when left empty."""

    def __post_init__(self):
        """Fill `years` from `data` when it was not given."""
        if not self.years and self.data is not None:
            self.years = sorted({int(y) for y in self.data["year"].values})

    def probability(self, variant: str = "base") -> xr.DataArray:
        """Return the share of members with an onset on each lead day.

        Parameters
        ----------
        variant : str, default "base"
            The count variable to use: ``"base"``, ``"fixed_cutoff"`` or ``"ref"``.

        Returns
        -------
        xarray.DataArray
            ``count_<variant>`` divided by ``members`` as ``float64``, dims ``(issue, unit, day)``.
        """
        return self.data[f"count_{variant}"] / self.data["members"].astype(float)

    def release(self) -> None:
        """Drop the arrays, keeping the summary (`model` and `years`)."""
        self.data = None


_RESERVED = ("sample", "unit", "sample_unit")
"""Names the `Weekly` layout uses, which a legacy table's columns may not take."""


@dataclass
class Weekly:
    """The weekly features as arrays: one sample per unit and issue date.

    `data` holds each feature over ``sample``, with the sample's position in the ``unit`` dimension
    (``sample_unit``), its issue time and year and its outcome (positions into the outcome labels, -1 without one);
    the unit ids and the per-unit columns (``lat``, ``lon``, ``onset_threshold``) are over ``unit``. A column reads
    as an array per sample (``weekly["diff_x_week1"]``) and `Weekly.frame` builds the legacy table from them, for
    export and the code that still reads tables. `Weekly.from_frame` takes a legacy table, keeping every column as
    it is.
    """

    data: xr.Dataset
    """The features over ``sample`` and the per-unit variables over ``unit``."""
    columns: list[str]
    """The legacy table's columns, in order."""
    attrs: dict = field(default_factory=dict)
    """The legacy table's ``attrs``."""
    dtypes: dict = field(default_factory=dict)
    """Legacy dtypes by column, for tables taken by `Weekly.from_frame`; empty for a native `Weekly`."""

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            The length of the ``sample`` dimension.
        """
        return self.data.sizes["sample"]

    def __contains__(self, name) -> bool:
        """Return whether `name` is a column of the legacy table.

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        bool
            True if `name` is in `columns`.
        """
        return name in self.columns

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a column's values per sample (see `Weekly.take`).

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        numpy.ndarray
            The column's values for every sample; `Weekly.take` raises ``KeyError`` for an unknown column.
        """
        return self.take(name)

    @property
    def units(self) -> np.ndarray:
        """The unit ids, an object array over the ``unit`` dimension."""
        return self.data["unit"].values

    @property
    def sample_unit(self) -> np.ndarray:
        """The position of each sample's unit in `units`, an integer array of shape ``(sample,)``."""
        return self.data["sample_unit"].values

    @property
    def nbytes(self) -> int:
        """The size of the arrays in bytes."""
        return int(self.data.nbytes)

    def take(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        """Return a column's values per sample.

        Parameters
        ----------
        name : str
            A column of the table; ``"id"`` gives the unit id of each sample.
        rows : numpy.ndarray of int, optional
            The sample positions to take; every sample when None.

        Returns
        -------
        numpy.ndarray
            One value per row; outcome columns come back as their labels (object), per-unit columns expanded to
            samples.

        Raises
        ------
        KeyError
            `name` is not a column.
        """
        if name not in self.columns:
            raise KeyError(name)
        unit = self.sample_unit if rows is None else self.sample_unit[rows]
        if name == "id":
            return self.units[unit]
        variable = self.data[name]
        if variable.dims == ("unit",):
            return variable.values[unit]
        values = variable.values if rows is None else variable.values[rows]
        if "labels" in variable.attrs:
            return outcome_series(values, list(variable.attrs["labels"])).to_numpy()
        return values

    def column_dtypes(self) -> dict[str, np.dtype]:
        """Return each column's dtype as `Weekly.take` returns it (ids and labels are objects).

        Returns
        -------
        dict of str to numpy.dtype
            The dtype by column, in column order.
        """
        return {name: np.dtype(object) if name == "id" or "labels" in self.data[name].attrs else self.data[name].dtype
                for name in self.columns}

    def outcome_codes(self, rows: np.ndarray | None = None, labels=None) -> np.ndarray:
        """Return the samples' outcomes as positions into `labels`.

        Parameters
        ----------
        rows : numpy.ndarray of int, optional
            The sample positions to take; every sample when None.
        labels : sequence of str, optional
            The labels to give positions in; the stored labels when None.

        Returns
        -------
        numpy.ndarray of int8, shape (sample,)
            The position of each outcome in `labels`, -1 without an outcome, ``len(labels)`` for a label not in
            `labels`.
        """
        variable = self.data["outcome"]
        if "labels" not in variable.attrs:
            return outcome_codes(self.take("outcome", rows), labels)
        codes = variable.values if rows is None else variable.values[rows]
        stored = list(variable.attrs["labels"])
        if labels is None or list(labels) == stored:
            return codes.astype(np.int8)
        index = {label: k for k, label in enumerate(labels)}
        remap = np.array([index.get(label, len(labels)) for label in stored] + [-1], dtype=np.int8)
        return remap[codes]

    def frame(self, rows: np.ndarray | None = None, columns=None) -> pd.DataFrame:
        """Return the legacy table, or some rows and columns of it, on a fresh ``0..n-1`` index.

        Parameters
        ----------
        rows : numpy.ndarray of int, optional
            The sample positions to take; every sample when None.
        columns : iterable of str, optional
            The columns to keep, in table order; all when None.

        Returns
        -------
        pandas.DataFrame
            The table, with `attrs` copied to its ``attrs``.
        """
        wanted = self.columns if columns is None else [c for c in self.columns if c in set(columns)]
        out = {name: self._series(name, rows) for name in wanted}
        table = pd.DataFrame(out) if out else pd.DataFrame(
            index=pd.RangeIndex(len(self) if rows is None else len(rows)))
        table.attrs.update(self.attrs)
        return table

    def columns_of(self, rows: np.ndarray | None = None, columns=None) -> dict:
        """Return some rows and columns as a mapping of arrays, for Patsy.

        These are the table's columns without building the table; a categorical column of a table taken by
        `Weekly.from_frame` stays a `pandas.Series`, which keeps its category order.

        Parameters
        ----------
        rows : numpy.ndarray of int, optional
            The sample positions to take; every sample when None.
        columns : iterable of str, optional
            The columns to keep; all when None.

        Returns
        -------
        dict of str to numpy.ndarray or pandas.Series
            The values by column name.
        """
        wanted = self.columns if columns is None else [c for c in self.columns if c in set(columns)]
        return {name: pd.Series(self.take(name, rows), dtype=self.dtypes[name])
                if isinstance(self.dtypes.get(name), pd.CategoricalDtype) else self.take(name, rows)
                for name in wanted}

    def _series(self, name: str, rows):
        """Return one column of the legacy table as a `pandas.Series` or array.

        Parameters
        ----------
        name : str
            The column.
        rows : numpy.ndarray of int or None
            The sample positions to take; every sample when None.

        Returns
        -------
        pandas.Series or numpy.ndarray
            The column in its legacy type: a Series for stored dtypes, outcome labels, ``id`` (str) and ``time``
            (``datetime.date`` objects), the plain values otherwise.
        """
        if name in self.dtypes:
            return pd.Series(self.take(name, rows), dtype=self.dtypes[name])
        variable = None if name == "id" else self.data[name]
        if variable is not None and "labels" in variable.attrs:
            codes = variable.values if rows is None else variable.values[rows]
            return outcome_series(codes, list(variable.attrs["labels"]))
        values = self.take(name, rows)
        if name == "id":
            return pd.Series(values, dtype=str)
        if name == "time":
            return pd.Series(pd.DatetimeIndex(values).date, dtype=object)
        return values

    def to_netcdf(self, path) -> None:
        """Write the features as NetCDF, for `Weekly.open`.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write.

        Raises
        ------
        ValueError
            The `Weekly` was taken from a table with `Weekly.from_frame`, which is saved as that table.
        """
        if self.dtypes:
            raise ValueError("a Weekly taken from a table is saved as that table")
        data = self.data.copy()
        data.attrs = {"columns": json.dumps(self.columns), "attrs": json.dumps(self.attrs, default=_json_scalar)}
        for name in data.variables:
            if "labels" in data[name].attrs:
                data[name].attrs["labels"] = json.dumps(list(data[name].attrs["labels"]))
        data.to_netcdf(path)

    @classmethod
    def open(cls, path) -> "Weekly":
        """Read a `Weekly` written by `Weekly.to_netcdf`.

        Parameters
        ----------
        path : str or pathlib.Path
            The NetCDF file.

        Returns
        -------
        Weekly
            The features, loaded into memory, with the columns, ``attrs`` and outcome labels restored.
        """
        with xr.open_dataset(path) as stored:
            data = stored.load()
        columns, attrs = json.loads(data.attrs.pop("columns")), json.loads(data.attrs.pop("attrs"))
        for name in data.variables:
            if "labels" in data[name].attrs:
                data[name].attrs["labels"] = json.loads(data[name].attrs["labels"])
        data = data.assign_coords(unit=np.asarray(data["unit"].values, dtype=object))
        return cls(data=data, columns=columns, attrs=attrs)

    def with_units(self, ids: np.ndarray) -> "Weekly":
        """Return the same samples with other unit ids.

        Parameters
        ----------
        ids : numpy.ndarray of object, shape (unit,)
            The new ids, one per unit, in the order of `units`.

        Returns
        -------
        Weekly
            A copy sharing the arrays, with the ``unit`` coordinate replaced.
        """
        return Weekly(data=self.data.assign_coords(unit=np.asarray(ids, dtype=object)), columns=self.columns,
                      attrs=self.attrs, dtypes=self.dtypes)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Weekly":
        """Take a legacy table: its ids as units, every other column as it is.

        Parameters
        ----------
        frame : pandas.DataFrame
            The table, with an ``id`` column and no column named ``sample``, ``unit`` or ``sample_unit``.

        Returns
        -------
        Weekly
            The samples, remembering the table's dtypes so `Weekly.frame` gives it back.

        Raises
        ------
        ValueError
            The table has no ``id`` column, or a column clashes with the layout.
        """
        if "id" not in frame.columns:
            raise ValueError("the weekly table has no id column")
        reserved = [c for c in frame.columns if c in _RESERVED]
        if reserved:
            raise ValueError(f"weekly table columns {reserved} clash with the Weekly layout")
        codes, units = pd.factorize(frame["id"], use_na_sentinel=False)
        variables = {"sample_unit": ("sample", codes.astype(np.int32))}
        for name in frame.columns:
            if name != "id":
                variables[name] = ("sample", frame[name].to_numpy())
        data = xr.Dataset(variables, coords={"unit": np.asarray(units, dtype=object)})
        return cls(data=data, columns=[str(c) for c in frame.columns], attrs=dict(frame.attrs),
                   dtypes={name: frame[name].dtype for name in frame.columns})

    @classmethod
    def of(cls, table) -> "Weekly":
        """Return `table` as a `Weekly`.

        Parameters
        ----------
        table : Weekly or pandas.DataFrame
            A `Weekly`, or a legacy table.

        Returns
        -------
        Weekly
            `table` itself if it is one, else `Weekly.from_frame` of it.
        """
        return table if isinstance(table, cls) else cls.from_frame(table)


def _json_scalar(value):
    """Convert a NumPy scalar to a Python one, as the ``default`` hook of ``json.dumps``.

    Parameters
    ----------
    value : object
        The value ``json`` could not serialise.

    Returns
    -------
    object
        ``value.item()`` for a `numpy.generic`.

    Raises
    ------
    TypeError
        `value` is not a NumPy scalar.
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


@dataclass
class DailyInputs:
    """What the random forest reads for each weekly sample, in the samples' order.

    These are each model's ensemble-mean rainfall by lead day (``rain_<model>`` over ``<model>_day``), the conditional
    climatology's onset probability by lead day (``clim_p_onset`` over ``clim_day``), and the observed onset's lead
    day after the issue (``onset_lead``, NaN without one). ``time`` is each sample's issue date, to check the
    alignment.
    """

    data: xr.Dataset
    """The variables above, over ``sample`` and the day dimensions."""

    @property
    def model_names(self) -> list[str]:
        """The names of the models that have a ``rain_<model>`` variable."""
        return [str(name)[len("rain_"):] for name in self.data.data_vars if str(name).startswith("rain_")]

    def rain(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a model's lead days and ensemble-mean rainfall.

        Parameters
        ----------
        model : str
            The model's name.

        Returns
        -------
        days : numpy.ndarray, shape (day,)
            The lead days.
        rain : numpy.ndarray of float, shape (sample, day)
            The rainfall by sample and lead day.
        """
        variable = self.data[f"rain_{model}"]
        return variable[f"{model}_day"].values, variable.values

    @property
    def climatology(self) -> np.ndarray:
        """The conditional climatology's onset probability, shape ``(sample, clim_day)``."""
        return self.data["clim_p_onset"].values

    @property
    def onset_lead(self) -> np.ndarray:
        """The observed onset's lead day after the issue, shape ``(sample,)``; NaN without an onset."""
        return self.data["onset_lead"].values
