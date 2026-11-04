"""A model's probabilities of the outcome bins for samples, with the samples' keys.

`Predictions.data` is an `xarray.Dataset` over ``sample`` (and ``bin``): ``probability`` (sample, bin) over the
interval bins (``week1`` .. ``weekN``, ``later``), optionally ``earlier`` (sample: an onset before the issue, which
the RPS counts) and ``daily`` (sample, lead_day: a model's daily probabilities); the keys are coordinates over
sample: ``id``, ``year``, ``time``, ``lat``, ``lon`` (the last three when the samples have them), and ``outcome``
(positions into the bins' labels; -1 without one, ``len(labels)`` for another label). CV predictions, baselines and
external predictions are all held so, scored from the arrays (`hcwf.metrics.probabilistic`), and saved as NetCDF.
Every blending module (`hcwf.blend.logit`, `hcwf.blend.random_forest`, `hcwf.blend.mme`,
`hcwf.blend.calibration`) produces or consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["Predictions", "outcome_codes", "outcome_series"]


def outcome_codes(values, labels) -> np.ndarray:
    """Convert outcome labels to positions into `labels`.

    Parameters
    ----------
    values : sequence
        The outcome label of each sample; ``None``, ``NaN`` and ``pandas.NA`` are missing.
    labels : sequence
        The bin labels.

    Returns
    -------
    numpy.ndarray of int8, shape (sample,)
        The position of each label in `labels`, -1 where missing, ``len(labels)`` for a label that is not a bin.
    """
    index = {label: k for k, label in enumerate(labels)}
    other = len(labels)
    out = np.empty(len(values), dtype=np.int8)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            out[i] = -1
        else:
            out[i] = index.get(value, other)
    return out


def outcome_series(codes: np.ndarray, labels: list[str]) -> pd.Series:
    """Convert outcome codes back to the legacy table's outcome column.

    Parameters
    ----------
    codes : numpy.ndarray of int, shape (sample,)
        The codes of `outcome_codes`.
    labels : list of str
        The bin labels.

    Returns
    -------
    pandas.Series of object
        The label of each code; ``None`` where the code is -1 or past the labels.
    """
    values = np.array([*labels, None, None], dtype=object)[np.where(codes < 0, len(labels) + 1, codes)]
    return pd.Series(values, dtype=object).astype(str).where(pd.Series(values).notna(), None)


@dataclass
class Predictions:
    """A model's probabilities for samples, held as one `xarray.Dataset` (see the module docstring)."""

    data: xr.Dataset
    """The dataset over ``sample`` and ``bin``: ``probability``, optionally ``earlier`` and ``daily``, and the keys."""

    # -- building -------------------------------------------------------------

    @classmethod
    def of_samples(cls, weekly, rows: np.ndarray, probabilities: np.ndarray, labels, *,
                   earlier: np.ndarray | None = None, daily: np.ndarray | None = None) -> Predictions:
        """Return the predictions of some weekly samples, keyed by theirs.

        Parameters
        ----------
        weekly : hcwf.blend.samples.Weekly
            The weekly samples; the keys ``id``, ``year``, the outcome and, when it has them, ``time``, ``lat`` and
            ``lon`` are taken from it.
        rows : numpy.ndarray of int, shape (sample,)
            The positions in `weekly` of the samples predicted.
        probabilities : numpy.ndarray of float64, shape (sample, bin)
            The probability of each bin.
        labels : sequence of str
            The bin labels, in the column order of `probabilities`.
        earlier : numpy.ndarray of float64, shape (sample,), optional
            The probability of an onset before the issue.
        daily : numpy.ndarray of float64, shape (sample, lead_day), optional
            The daily probabilities; ``lead_day`` is numbered from 1.

        Returns
        -------
        Predictions
            The predictions.
        """
        rows = np.asarray(rows, dtype=np.int64)
        coords = {
            "id": ("sample", weekly.take("id", rows).astype(str)),
            "year": ("sample", weekly.take("year", rows)),
            "outcome": ("sample", weekly.outcome_codes(rows, labels)),
            "bin": list(labels),
        }
        if "time" in weekly:
            coords["time"] = ("sample", _times(weekly, rows))
        for key in ("lat", "lon"):
            if key in weekly:
                coords[key] = ("sample", np.asarray(weekly.take(key, rows), dtype=float))
        return cls._build(coords, probabilities, earlier, daily)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, labels) -> Predictions:
        """Return the predictions held in a legacy CV-prediction table.

        Parameters
        ----------
        frame : pandas.DataFrame
            One row per sample: ``id``, ``year``, ``outcome`` (labels), ``cv_<bin>`` for every bin, and optionally
            ``time``, ``lat``, ``lon`` and ``cv_earlier``.
        labels : sequence of str
            The bin labels.

        Returns
        -------
        Predictions
            The predictions (without daily probabilities).
        """
        labels = list(labels)
        coords = {
            "id": ("sample", frame["id"].astype(str).to_numpy()),
            "year": ("sample", frame["year"].to_numpy()),
            "outcome": ("sample", outcome_codes(frame["outcome"].to_numpy(dtype=object), labels)),
            "bin": labels,
        }
        if "time" in frame:
            coords["time"] = ("sample", pd.to_datetime(frame["time"]).to_numpy())
        for key in ("lat", "lon"):
            if key in frame:
                coords[key] = ("sample", frame[key].to_numpy(dtype=float))
        probabilities = frame[[f"cv_{b}" for b in labels]].to_numpy(dtype=float)
        earlier = frame["cv_earlier"].to_numpy(dtype=float) if "cv_earlier" in frame else None
        return cls._build(coords, probabilities, earlier, None)

    @classmethod
    def _build(cls, coords, probabilities, earlier, daily) -> Predictions:
        """Assemble the dataset of some predictions.

        Parameters
        ----------
        coords : dict
            The coordinates, including ``bin`` (the labels) and ``outcome``.
        probabilities : numpy.ndarray of float64, shape (sample, bin)
            The probability of each bin.
        earlier : numpy.ndarray of float64, shape (sample,) or None
            The probability of an onset before the issue.
        daily : numpy.ndarray of float64, shape (sample, lead_day) or None
            The daily probabilities.

        Returns
        -------
        Predictions
            The predictions; the bin labels are kept in the attribute ``labels`` of ``outcome``.
        """
        variables = {"probability": (("sample", "bin"), np.asarray(probabilities, dtype=float))}
        if earlier is not None:
            variables["earlier"] = ("sample", np.asarray(earlier, dtype=float))
        if daily is not None:
            variables["daily"] = (("sample", "lead_day"), np.asarray(daily, dtype=float))
            coords = {**coords, "lead_day": np.arange(1, daily.shape[1] + 1)}
        labels = list(coords["bin"])
        data = xr.Dataset(variables, coords=coords)
        data["outcome"].attrs["labels"] = labels
        return cls(data)

    @classmethod
    def concat(cls, parts: list[Predictions]) -> Predictions:
        """Concatenate predictions along the samples.

        Parameters
        ----------
        parts : list of Predictions
            The predictions to join; they must share the bins and their coordinates must match exactly.

        Returns
        -------
        Predictions
            The samples of every part, in order.
        """
        data = xr.concat([p.data for p in parts], dim="sample", data_vars="minimal", coords="different",
                         compat="equals", join="exact")
        data["outcome"].attrs["labels"] = parts[0].labels
        return cls(data)

    def take(self, positions: np.ndarray) -> Predictions:
        """Return some of the samples, in the order given.

        Parameters
        ----------
        positions : numpy.ndarray of int, shape (n,)
            The positions of the samples to take.

        Returns
        -------
        Predictions
            The selected samples.
        """
        return Predictions(self.data.isel(sample=np.asarray(positions, dtype=np.int64)))

    def with_probabilities(self, probabilities: np.ndarray, *, earlier: np.ndarray | None = None) -> Predictions:
        """Return predictions with other probabilities for the same samples.

        The daily probabilities are dropped, as are ``earlier`` if `earlier` is not given.

        Parameters
        ----------
        probabilities : numpy.ndarray of float64, shape (sample, bin)
            The new probability of each bin.
        earlier : numpy.ndarray of float64, shape (sample,), optional
            The new probability of an onset before the issue.

        Returns
        -------
        Predictions
            The new predictions, with the same keys and bins.
        """
        data = self.data.drop_vars(list(self.data.data_vars))
        data["probability"] = (("sample", "bin"), np.asarray(probabilities, dtype=float))
        if earlier is not None:
            data["earlier"] = ("sample", np.asarray(earlier, dtype=float))
        if "lead_day" in data.dims:
            data = data.drop_dims("lead_day")
        data["outcome"].attrs["labels"] = self.labels
        return Predictions(data)

    # -- reading --------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples.

        Returns
        -------
        int
            The size of the ``sample`` dimension.
        """
        return self.data.sizes["sample"]

    @property
    def labels(self) -> list[str]:
        """The bin labels, in the column order of `probabilities`."""
        return list(self.data["bin"].values)

    @property
    def probabilities(self) -> np.ndarray:
        """The probability of each bin, ``float64`` of shape ``(sample, bin)``."""
        return self.data["probability"].values

    @property
    def earlier(self) -> np.ndarray | None:
        """The probability of an onset before the issue, ``float64`` of shape ``(sample,)``; ``None`` if absent."""
        return self.data["earlier"].values if "earlier" in self.data else None

    @property
    def outcome(self) -> np.ndarray:
        """The outcome codes (see `outcome_codes`), shape ``(sample,)``."""
        return self.data["outcome"].values

    @property
    def ids(self) -> np.ndarray:
        """The sample ids, shape ``(sample,)``."""
        return self.data["id"].values

    @property
    def year(self) -> np.ndarray:
        """The year of each sample, shape ``(sample,)``."""
        return self.data["year"].values

    def frame(self) -> pd.DataFrame:
        """Return the legacy CV-prediction table.

        Returns
        -------
        pandas.DataFrame
            One row per sample: ``id``, ``time`` (dates), ``year``, ``lat``, ``lon`` (those the samples have),
            ``outcome`` (labels), ``cv_<bin>`` for every bin, and, when present, ``cv_earlier`` and ``p_day_<k>``
            (the daily probabilities of days 1 to N-1) with ``p_later`` (the last daily column).
        """
        labels = self.labels
        out = {"id": pd.Series(self.ids, dtype=str)}
        if "time" in self.data.coords:
            out["time"] = pd.Series(pd.DatetimeIndex(self.data["time"].values).date, dtype=object)
        out["year"] = self.year
        for key in ("lat", "lon"):
            if key in self.data.coords:
                out[key] = self.data[key].values
        out["outcome"] = outcome_series(self.outcome, labels)
        table = pd.DataFrame(out)
        for j, label in enumerate(labels):
            table[f"cv_{label}"] = self.probabilities[:, j]
        if self.earlier is not None:
            table["cv_earlier"] = self.earlier
        if "daily" in self.data:
            daily = self.data["daily"].values
            for d in range(daily.shape[1] - 1):
                table[f"p_day_{d + 1}"] = daily[:, d]
            table["p_later"] = daily[:, -1]
        return table

    def to_netcdf(self, path) -> None:
        """Write the predictions to a NetCDF file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write; the bin labels are stored comma-joined in the ``labels`` attribute of ``outcome``.
        """
        data = self.data.copy()
        data["outcome"].attrs["labels"] = ",".join(self.labels)
        data.to_netcdf(path)

    @classmethod
    def open(cls, path) -> Predictions:
        """Read predictions written by `Predictions.to_netcdf`.

        Parameters
        ----------
        path : str or pathlib.Path
            The NetCDF file.

        Returns
        -------
        Predictions
            The predictions, loaded in memory.
        """
        with xr.open_dataset(path) as stored:
            data = stored.load()
        labels = str(data["outcome"].attrs["labels"]).split(",")
        data["outcome"].attrs["labels"] = labels
        return cls(data)


def _times(weekly, rows) -> np.ndarray:
    """Return the issue times of some weekly samples as datetimes.

    Parameters
    ----------
    weekly : hcwf.blend.samples.Weekly
        The weekly samples, with a ``time`` column.
    rows : numpy.ndarray of int, shape (sample,)
        The positions of the samples.

    Returns
    -------
    numpy.ndarray of datetime64, shape (sample,)
        The times, converted when the column holds objects.
    """
    values = weekly.take("time", rows)
    return pd.to_datetime(pd.Series(values)).to_numpy() if values.dtype == object else values
