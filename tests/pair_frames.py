"""Test helpers between PairArrays and long-format pair frames (one row per case and bin).

The library holds pairs only as arrays; tests that compare with pandas reference implementations (or build pairs
by hand as a table) go through these.
"""

import re
from dataclasses import replace

import numpy as np
import pandas as pd

from hcwf.metrics.bins import PairArrays


def _edges(label):
    """(first day, last day) of a bin label: 'Days a-b', 'Before day a' (everything below a) or 'After day b'."""
    numbers = [int(n) for n in re.findall(r"\d+", label)]
    if label.startswith("Before"):
        return -np.inf, numbers[0] - 1
    if label.startswith("After"):
        return numbers[0] + 1, np.inf
    return float(numbers[0]), float(numbers[1])


def pairs_of_frame(df, spatial=("lat", "lon")):
    """The PairArrays of a long-format pair frame: cases in order of first appearance, bins in bin_index order.

    Needs init_time, the spatial columns, bin_label, bin_index, predicted_prob, observed_onset and total_members;
    total_members_with_onset and back_dated_members / onset_members become the optional per-case fields.
    """
    spatial = [d for d in spatial if d in df.columns]
    case = df.groupby(["init_time", *spatial], sort=False).ngroup().to_numpy()
    order = np.lexsort((df["bin_index"].to_numpy(), case))
    n_bins = df["bin_index"].nunique()
    ordered = df.iloc[order].reset_index(drop=True)

    def matrix(column, dtype):
        return ordered[column].to_numpy().reshape(-1, n_bins).astype(dtype)

    def per_case(column, dtype=None):
        if column not in df.columns:
            return None
        values = ordered[column].to_numpy()[::n_bins]
        return values if dtype is None else values.astype(dtype)

    labels = tuple(ordered["bin_label"].to_numpy()[:n_bins])
    edges = np.array([_edges(label) for label in labels], dtype=float)
    return PairArrays(
        prob=matrix("predicted_prob", float), observed=matrix("observed_onset", np.int8),
        members=per_case("total_members"), start=edges[:, 0], end=edges[:, 1], labels=labels,
        init=per_case("init_time", "datetime64[D]"),
        place={d: per_case(d) for d in spatial},
        members_with_onset=per_case("total_members_with_onset"),
        back_dated=per_case("back_dated_members"), onset_members=per_case("onset_members"))


def frame_of_pairs(pairs):
    """The long-format frame of PairArrays: one row per case and bin, as the pair builders once wrote them."""
    n_cases, n_bins = pairs.prob.shape
    case = np.repeat(np.arange(n_cases), n_bins)
    bin_ = np.tile(np.arange(n_bins), n_cases)
    frame = pd.DataFrame({
        "init_time": pairs.init[case].astype("datetime64[ns]"),
        **{d: values[case] for d, values in pairs.place.items()},
        "bin_start": pairs.start[bin_], "bin_end": pairs.end[bin_],
        "bin_label": np.array(pairs.labels, dtype=object)[bin_],
        "predicted_prob": pairs.prob.ravel(), "observed_onset": pairs.observed.ravel().astype(np.int64),
        "total_members": pairs.members[case], "bin_index": bin_,
    })
    if pairs.members_with_onset is not None:
        frame["total_members_with_onset"] = pairs.members_with_onset[case]
    return frame


def shuffled_cases(pairs, seed=0):
    """The same PairArrays with its cases in a random order."""
    order = np.random.default_rng(seed).permutation(pairs.n_cases)
    return replace(pairs, **{name: None if getattr(pairs, name) is None else getattr(pairs, name)[order]
                             for name in PairArrays._PER_CASE},
                   place={d: values[order] for d, values in pairs.place.items()})


def pairs_equal(a, b):
    """Whether two PairArrays hold the same cases and bins, exactly."""
    if a is None or b is None:
        return a is b
    same = (a.labels == b.labels and list(a.place) == list(b.place) and np.array_equal(a.start, b.start)
            and np.array_equal(a.end, b.end))
    for name in PairArrays._PER_CASE:
        x, y = getattr(a, name), getattr(b, name)
        same = same and (x is None) == (y is None) and (x is None or np.array_equal(x, y))
    return same and all(np.array_equal(a.place[d], b.place[d]) for d in a.place)
