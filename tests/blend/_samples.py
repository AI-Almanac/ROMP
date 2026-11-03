"""TEST-ONLY: a weekly table as the `WeeklySamples` the blending methods read (the real one comes with the features
step). Imports nothing of hcwf, so the differential script can use it with either package."""

import numpy as np
import pandas as pd

BINS = ["week1", "week2", "later"]


class FrameSamples:
    """WeeklySamples over a legacy weekly DataFrame (id, year, outcome label or NaN, feature columns)."""

    def __init__(self, frame: pd.DataFrame):
        self.table = frame.reset_index(drop=True)
        self.columns = [str(c) for c in frame.columns]
        self.attrs = dict(frame.attrs)

    def __len__(self):
        return len(self.table)

    def __contains__(self, name):
        return name in self.columns

    def __getitem__(self, name):
        return self.take(name)

    def take(self, name, rows=None):
        if name not in self.columns:
            raise KeyError(name)
        values = self.table[name].to_numpy()
        return values if rows is None else values[rows]

    def column_dtypes(self):
        return {c: (np.dtype(object) if c in ("id", "outcome") else self.table[c].dtype) for c in self.columns}

    def outcome_codes(self, rows, labels):
        index = {label: k for k, label in enumerate(labels)}
        return np.array([index.get(v, len(labels)) if isinstance(v, str) else -1 for v in self.take("outcome", rows)],
                        dtype=np.int8)

    def frame(self, rows=None, columns=None):
        wanted = self.columns if columns is None else [c for c in self.columns if c in set(columns)]
        table = self.table[wanted] if rows is None else self.table.iloc[rows][wanted]
        return table.reset_index(drop=True)

    def columns_of(self, rows=None, columns=None):
        return {name: self.take(name, rows) for name in (self.columns if columns is None else
                                                          [c for c in self.columns if c in set(columns)])}


def weekly_table(seed=0, years=range(2010, 2021), n_cells=4, issues=8):
    """The legacy tests' weekly table: outcomes that depend on the features, 5% unknown, one non-finite predictor."""
    rng = np.random.default_rng(seed)
    rows = [{"id": f"unit{c}", "year": year, "issue": k} for c in range(n_cells) for year in years for k in range(issues)]
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame["prob_clim_week1"] = rng.normal(-1.0, 0.5, n)
    frame["prob_clim_week2"] = rng.normal(-1.0, 0.5, n)
    frame["diff_m_week1"] = rng.normal(size=n)
    logits = np.column_stack([frame["prob_clim_week1"] + frame["diff_m_week1"], frame["prob_clim_week2"], np.zeros(n)])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    frame["outcome"] = [BINS[rng.choice(3, p=row)] for row in p]
    frame.loc[rng.random(n) < 0.05, "outcome"] = np.nan
    frame.loc[5, "diff_m_week1"] = np.nan
    return frame
