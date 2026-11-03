"""What the blend works on: onsets per unit, and the weekly samples built from them.

`ObservedOnsets` (observed onsets over unit and year) and `ForecastOnsets` (a model's member onset counts per issue,
unit and lead day) are what `hcwf.stats.detect` returns for evaluation units, collected over files
(`hcwf.blend.features`). `Weekly` is one sample per unit and issue date with a column per feature (`take` gives a
column's values per sample, for `rows` (positions) or every sample; `frame()` rebuilds the legacy table);
`DailyInputs` is what the random forest reads for each sample, in the samples' order. The blend methods
(`logit`, `random_forest`, `design`, `predictions`) read a `Weekly` through `take`, `__getitem__`, `columns`,
`attrs`, `column_dtypes`, `frame` and `outcome_codes` only, so a table with those (tests/blend/_samples.py) serves.
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
    """datetime64 values as date objects, NaT as NaT, each distinct value once."""
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    stamps = pd.DatetimeIndex(distinct)
    out = np.array([pd.NaT if pd.isna(t) else t.date() for t in stamps], dtype=object)
    return out[codes]


@dataclass
class ObservedOnsets:
    """Observed onsets over (unit, year): onset_day (days after the season start; NaN when there is none), onset_idx
    (the day in its season's file, 1-based) and present (the unit had a series that year)."""

    data: xr.Dataset
    season_start: str

    @classmethod
    def from_parts(cls, parts: list[dict], season_start: str) -> "ObservedOnsets":
        """From per-file summaries ({ids, years, onset_day, onset_idx, present}, arrays over (unit, year))."""
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
        return [int(y) for y in self.data["year"].values]

    def cutoff_dates(self) -> np.ndarray:
        """The season start of each year, datetime64[D]."""
        return np.array([np.datetime64(f"{y}-{self.season_start}") for y in self.years], dtype="datetime64[D]")

    def onset_dates(self) -> np.ndarray:
        """(unit, year) onset dates, datetime64[D]; NaT where there is none."""
        day = self.data["onset_day"].values
        dates = self.cutoff_dates()[np.newaxis, :] + np.where(np.isnan(day), 0, day).astype("timedelta64[D]")
        return np.where(np.isnan(day), np.datetime64("NaT"), dates)

    def onsets(self) -> pd.DataFrame:
        """(id, year, onset_day) for each present unit and season, season by season, units in order: what the
        climatology is fitted on."""
        y, u = np.nonzero(self.data["present"].values.T)
        return pd.DataFrame({
            "id": pd.Series(self.data["unit"].values[u], dtype=str),
            "year": self.data["year"].values[y].astype(np.int64),
            "onset_day": self.data["onset_day"].values.T[y, u]})

    def frame(self) -> pd.DataFrame:
        """The legacy table: one row per present unit and season."""
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
        """Units with at least min_onset_years observed onsets over every season (the climatology keeps only those,
        and the inner join drops every forecast unit it lacks)."""
        counts = (~np.isnan(self.data["onset_day"].values)).sum(axis=1)
        return frozenset(self.data["unit"].values[counts >= int(min_onset_years)].astype(str))


@dataclass
class ForecastOnsets:
    """A model's forecast onsets as arrays over (issue, unit, day): count_<variant> (members whose onset falls on each
    lead day, uint8; variants base, fixed_cutoff, ref), members (per issue), rain_mean (the members' mean rainfall)
    and onset_threshold (unit). Probabilities are counts over members, computed on demand."""

    model: str
    data: xr.Dataset | None
    years: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.years and self.data is not None:
            self.years = sorted({int(y) for y in self.data["year"].values})

    def probability(self, variant: str = "base") -> xr.DataArray:
        """The share of members with an onset on each lead day."""
        return self.data[f"count_{variant}"] / self.data["members"].astype(float)

    def release(self) -> None:
        """Drop the arrays, keeping the summary."""
        self.data = None


_RESERVED = ("sample", "unit", "sample_unit")


@dataclass
class Weekly:
    """The weekly features as arrays: one sample per unit and issue date.

    `data` holds each feature over `sample`, with the sample's position in the `unit` dimension (sample_unit), its
    issue time and year and its outcome (positions into the outcome labels, -1 without one); the unit ids and the
    per-unit columns (lat, lon, onset_threshold) are over `unit`. A column reads as an array per sample
    (`weekly["diff_x_week1"]`) and `frame()` builds the legacy table from them, for export and the code that still
    reads tables. `from_frame` takes a legacy table, keeping every column as it is."""

    data: xr.Dataset
    columns: list[str]                   # the legacy table's columns, in order
    attrs: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)   # legacy dtypes, for tables taken by from_frame

    def __len__(self) -> int:
        return self.data.sizes["sample"]

    def __contains__(self, name) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> np.ndarray:
        return self.take(name)

    @property
    def units(self) -> np.ndarray:
        return self.data["unit"].values

    @property
    def sample_unit(self) -> np.ndarray:
        return self.data["sample_unit"].values

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def take(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        """A column's values per sample, for `rows` (positions) or every sample."""
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
        """Each column's dtype as `take` returns it (labels are objects)."""
        return {name: np.dtype(object) if name == "id" or "labels" in self.data[name].attrs else self.data[name].dtype
                for name in self.columns}

    def outcome_codes(self, rows: np.ndarray | None = None, labels=None) -> np.ndarray:
        """The samples' outcomes as positions into `labels` (default: the stored labels): -1 without one,
        len(labels) for another label."""
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
        """The legacy table, or some rows (positions) and columns of it, on a fresh 0..n-1 index."""
        wanted = self.columns if columns is None else [c for c in self.columns if c in set(columns)]
        out = {name: self._series(name, rows) for name in wanted}
        table = pd.DataFrame(out) if out else pd.DataFrame(
            index=pd.RangeIndex(len(self) if rows is None else len(rows)))
        table.attrs.update(self.attrs)
        return table

    def _series(self, name: str, rows):
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
        """The features as NetCDF, for `Weekly.open` (a table taken by from_frame is saved as that table)."""
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
        with xr.open_dataset(path) as stored:
            data = stored.load()
        columns, attrs = json.loads(data.attrs.pop("columns")), json.loads(data.attrs.pop("attrs"))
        for name in data.variables:
            if "labels" in data[name].attrs:
                data[name].attrs["labels"] = json.loads(data[name].attrs["labels"])
        data = data.assign_coords(unit=np.asarray(data["unit"].values, dtype=object))
        return cls(data=data, columns=columns, attrs=attrs)

    def with_units(self, ids: np.ndarray) -> "Weekly":
        """The same samples with these unit ids (one per unit)."""
        return Weekly(data=self.data.assign_coords(unit=np.asarray(ids, dtype=object)), columns=self.columns,
                      attrs=self.attrs, dtypes=self.dtypes)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Weekly":
        """A legacy table: its ids as units, every other column as it is."""
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
        return table if isinstance(table, cls) else cls.from_frame(table)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


@dataclass
class DailyInputs:
    """What the random forest reads for each weekly sample, in the samples' order: each model's ensemble-mean rainfall
    by lead day (`rain_<model>` over `<model>_day`), the conditional climatology's onset probability by lead day
    (`clim_p_onset` over `clim_day`), and the observed onset's lead day after the issue (`onset_lead`, NaN without
    one). `time` is each sample's issue date, to check the alignment."""

    data: xr.Dataset

    @property
    def model_names(self) -> list[str]:
        return [str(name)[len("rain_"):] for name in self.data.data_vars if str(name).startswith("rain_")]

    def rain(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """(lead days, rainfall (sample, day))."""
        variable = self.data[f"rain_{model}"]
        return variable[f"{model}_day"].values, variable.values

    @property
    def climatology(self) -> np.ndarray:
        return self.data["clim_p_onset"].values

    @property
    def onset_lead(self) -> np.ndarray:
        return self.data["onset_lead"].values
