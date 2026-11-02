"""What the blending methods read of the weekly samples and the forest's daily inputs.

The feature builders (a later step) provide these; the methods here only need the interfaces, so they are
protocols. `WeeklySamples` is one sample per unit and issue date with a column per feature (`take` gives a
column's values per sample, for `rows` (positions) or every sample, the legacy table's columns being `columns`);
`DailyInputs` is what the random forest reads for each sample, in the samples' order.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr


@runtime_checkable
class WeeklySamples(Protocol):
    columns: list[str]
    attrs: dict

    def __len__(self) -> int: ...

    def __contains__(self, name) -> bool: ...

    def __getitem__(self, name: str) -> np.ndarray: ...

    def take(self, name: str, rows: np.ndarray | None = None) -> np.ndarray: ...

    def column_dtypes(self) -> dict[str, np.dtype]: ...

    def frame(self, rows: np.ndarray | None = None, columns=None) -> pd.DataFrame: ...

    def outcome_codes(self, rows: np.ndarray, labels) -> np.ndarray: ...


class DailyInputs(Protocol):
    """Each model's ensemble-mean rainfall by lead day (`rain_<model>` over `<model>_day` in `data`), the
    conditional climatology's onset probability by lead day, `time` (each sample's issue date) and the
    observed onset's lead day (`onset_lead`, NaN without one)."""

    data: xr.Dataset

    @property
    def model_names(self) -> list[str]: ...

    def rain(self, model: str) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def climatology(self) -> np.ndarray: ...

    @property
    def onset_lead(self) -> np.ndarray: ...
