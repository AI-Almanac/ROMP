"""Design matrices from formulas, with array math instead of tables.

A formula whose terms are numeric columns and products of them (``a:b``, or ``a*b`` expanded) makes every design
column a product of feature columns. `plan` takes the formula's columns, in Patsy's order, from Patsy itself;
`Plan.build` then fills the matrix a column at a time straight from the feature arrays, multiplying the factors in
Patsy's order, so the values are Patsy's to the bit without a table or Patsy's intermediate copies. Other formulas
(functions, categorical or stateful terms) have no plan and go through Patsy.

The formulas themselves come from their texts here too: `formulas` expands each enabled text (``_qx`` to one term
per week bin, ``a*b`` to every interaction), drops the ``_qx`` terms the rain horizon cannot build, and adds the
window variants; `sample_support` gives the samples every formula's predictors are finite for. The logistic blend
(`hcwf.blend.logit`) and the random forest read their predictors through these.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy


@dataclass(frozen=True, kw_only=True)
class Formula:
    """A named formula's text, and whether the evaluation uses it."""

    text: str
    """The formula text, ``lhs ~ term + term ...``; ``_qx`` and ``*`` are expanded by `expand`."""
    enabled: bool = True
    """Whether the evaluation uses the formula."""


@dataclass(frozen=True, kw_only=True)
class WindowVariant:
    """Formula variants that swap a feature for its windowed-climatology version."""

    base: str
    """The name of the formula (in the formulas' mapping) to derive the variants from."""
    windows: tuple[tuple[int, int], ...] = ()
    """The ``(start, end)`` windows; one variant, named ``<base>_<start>_<end>``, for each."""
    replace: str = "prob_clim"
    """The feature whose ``<replace>_qx`` term is swapped for ``<replace>_<start>_<end>_qx``."""
    only_if_cutoff_mode: str | None = None
    """Make the variants only when the cutoff mode equals this; ``None`` makes them always."""


def right_hand_side(formula: str) -> str:
    """Return the right-hand side of a formula.

    Parameters
    ----------
    formula : str
        A formula text, with or without ``lhs ~``.

    Returns
    -------
    str
        The text after the first ``~`` (the whole text without one), stripped.
    """
    return formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()


@dataclass(frozen=True)
class Plan:
    """A formula's design columns, each the product of named feature columns."""

    formula: str
    """The formula text the plan was made from."""
    columns: tuple[str, ...]
    """The design column names, in Patsy's order, without the intercept."""
    factors: tuple[tuple[str, ...], ...]
    """For each design column, the feature columns it multiplies, in Patsy's order."""

    @property
    def features(self) -> list[str]:
        """The feature columns the design reads, in order of first use."""
        return list(dict.fromkeys(name for names in self.factors for name in names))

    def column(self, j: int, values: Callable[[str], np.ndarray], rows: np.ndarray | None = None) -> np.ndarray:
        """Return one design column: the product of its factors, multiplied in Patsy's order.

        Parameters
        ----------
        j : int
            The position of the design column in `columns`.
        values : callable
            ``values(name)`` returns the values of the feature `name`, one per row of the source.
        rows : numpy.ndarray of int, shape (row,), optional
            The positions of the rows to take; ``None`` takes every row.

        Returns
        -------
        numpy.ndarray of float64, shape (row,)
            The column; a new array that does not share memory with what `values` returned.
        """
        out = None
        for name in self.factors[j]:
            factor = np.asarray(values(name), dtype=float)
            if rows is not None:
                factor = factor[rows]
            if out is None:
                out = factor.copy() if rows is None else factor
            else:
                out *= factor
        return out

    def build(self, values: Callable[[str], np.ndarray], rows: np.ndarray | None = None, *,
              out: np.ndarray | None = None, order: str = "F") -> np.ndarray:
        """Return the design matrix, filled a column at a time.

        Parameters
        ----------
        values : callable
            ``values(name)`` returns the values of the feature `name`, one per row of the source.
        rows : numpy.ndarray of int, shape (row,), optional
            The positions of the rows to take; ``None`` takes every row.
        out : numpy.ndarray of float64, shape (row, column), optional
            Where to write the matrix; a new array when ``None``.
        order : {"F", "C"}, default "F"
            The memory order of a new array (ignored when `out` is given).

        Returns
        -------
        numpy.ndarray of float64, shape (row, column)
            The design matrix, `out` itself when it was given.
        """
        if out is None:
            n = len(rows) if rows is not None else len(np.asarray(values(self.factors[0][0])))
            out = np.empty((n, len(self.columns)), order=order)
        for j in range(len(self.columns)):
            out[:, j] = self.column(j, values, rows)
        return out


def plan(formula: str, dtypes: Mapping[str, object]) -> Plan | None:
    """Return the design plan of a formula, or ``None`` if it needs Patsy.

    A formula has a plan when every term is a product of numeric columns of `dtypes` and Patsy's column names
    (asked of Patsy on one row) are exactly the terms'.

    Parameters
    ----------
    formula : str
        The formula text, with or without ``lhs ~``.
    dtypes : mapping of str to object
        The dtype of each available feature column.

    Returns
    -------
    Plan or None
        The plan, or ``None`` for any other formula (functions, categorical or non-numeric terms, an unparsable
        text, a term with no factors only).
    """
    rhs = right_hand_side(formula)
    try:
        terms = patsy.ModelDesc.from_formula(rhs).rhs_termlist
    except patsy.PatsyError:
        return None
    by_name = {}
    for term in terms:
        names = []
        for factor in term.factors:
            name = getattr(factor, "code", None)
            if name not in dtypes or not _numeric(dtypes[name]):
                return None
            names.append(name)
        if names:
            by_name[term.name()] = tuple(names)
    if not by_name:
        return None
    # Patsy orders the columns (it groups and sorts terms): ask it, on one row.
    used = sorted({name for names in by_name.values() for name in names})
    sample = pd.DataFrame({name: np.zeros(1, dtype=dtypes[name]) for name in used})
    columns = [c for c in patsy.dmatrix(rhs, sample, return_type="dataframe").columns if c != "Intercept"]
    if len(columns) != len(by_name) or set(columns) != set(by_name):
        return None
    return Plan(formula, tuple(columns), tuple(by_name[c] for c in columns))


def _numeric(dtype) -> bool:
    """Return whether Patsy treats a dtype as numeric (float, signed or unsigned integer; not bool, object or extension).

    Parameters
    ----------
    dtype : object
        A dtype.

    Returns
    -------
    bool
        True for a NumPy dtype of kind ``f``, ``i`` or ``u``.
    """
    return isinstance(dtype, np.dtype) and dtype.kind in "fiu"


# ---------------------------------------------------------------------------
# Formulas: the given texts, expanded and resolved against the features
# ---------------------------------------------------------------------------


def formula_columns(formula: str) -> list[str]:
    """Return the feature names of a formula's right-hand side.

    Parameters
    ----------
    formula : str
        The formula text, with or without ``lhs ~``.

    Returns
    -------
    list of str
        The names between the operators ``+ - * : ( )``, without empty ones, digits and ``1``; repeats are kept.
    """
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    names = [t.strip() for t in re.split(r"[+\-\*:\(\)]", rhs)]
    return [n for n in names if n and not n.isdigit() and n != "1"]


def _expanded_terms(formula: str, n_bins: int) -> tuple[str, list[tuple[str, set[str]]]]:
    """Expand the ``_qx`` and ``*`` of a formula into its terms.

    A term with ``_qx`` columns becomes one term per week bin (``_week1`` ... ``_weekN``), and ``a*b`` becomes every
    Wilkinson term (``a``, ``b``, ``a:b``).

    Parameters
    ----------
    formula : str
        The formula text, with or without ``lhs ~``.
    n_bins : int
        The number of week bins.

    Returns
    -------
    lhs : str
        The left-hand side, empty without ``~``.
    terms : list of tuple of (str, set of str)
        Each term and the columns it uses that were generated from a ``_qx`` column.
    """
    if "~" in formula:
        lhs, rhs = (part.strip() for part in formula.split("~", 1))
    else:
        lhs, rhs = "", formula.strip()
    by_week = []
    for raw in (t.strip() for t in rhs.split("+")):
        qx = [c for c in formula_columns(raw) if "_qx" in c]
        if qx:
            for i in range(1, int(n_bins) + 1):
                by_week.append((raw.replace("_qx", f"_week{i}"), {c.replace("_qx", f"_week{i}") for c in qx}))
        else:
            by_week.append((raw, set()))
    terms = []
    for term, generated in by_week:
        if "*" in term:
            factors = [v.strip() for v in term.split("*")]
            for size in range(1, len(factors) + 1):
                for combo in itertools.combinations(factors, size):
                    terms.append((":".join(combo), generated.intersection(combo)))
        else:
            terms.append((term, generated))
    return lhs, terms


def _joined(lhs: str, terms: list[str]) -> str:
    """Join terms into a formula text.

    Parameters
    ----------
    lhs : str
        The left-hand side; empty for none.
    terms : list of str
        The terms of the right-hand side.

    Returns
    -------
    str
        ``lhs ~ term + term ...``, or only the right-hand side when `lhs` is empty.
    """
    rhs = " + ".join(terms)
    return f"{lhs} ~ {rhs}" if lhs else rhs


def expand(formula: str, n_bins: int) -> str:
    """Return the formula with ``_qx`` and ``*`` expanded.

    Parameters
    ----------
    formula : str
        The formula text, with or without ``lhs ~``.
    n_bins : int
        The number of week bins.

    Returns
    -------
    str
        The expanded formula text.
    """
    lhs, terms = _expanded_terms(formula, n_bins)
    return _joined(lhs, [term for term, _ in terms])


def resolve(formula: str, unavailable, n_bins: int) -> tuple[str, list[str]]:
    """Expand a formula and drop the terms whose columns cannot be built under the rain horizon.

    Explicit ``_weekN`` terms are never dropped, only those generated from ``_qx``.

    Parameters
    ----------
    formula : str
        The formula text, with or without ``lhs ~``.
    unavailable : iterable of str
        The columns that cannot be built.
    n_bins : int
        The number of week bins.

    Returns
    -------
    formula : str
        The expanded formula without the dropped terms.
    dropped : list of str
        The dropped terms.

    Raises
    ------
    ValueError
        No kept term has a feature column.
    """
    lhs, terms = _expanded_terms(formula, n_bins)
    unavailable = set(unavailable)
    kept, dropped = [], []
    for term, generated in terms:
        (dropped if generated & unavailable else kept).append(term)
    if not any(formula_columns(term) for term in kept):
        raise ValueError("Formula has no constructible predictor terms after resolving connector rainfall horizons.")
    return _joined(lhs, kept), dropped


def formulas(entries: Mapping[str, Formula], window_variants: Sequence[WindowVariant], unavailable, n_bins: int,
             cutoff_mode: str) -> tuple[dict[str, str], dict[str, dict]]:
    """Return the enabled formulas, resolved, with their window variants, and how each was resolved.

    A window variant is its base formula with the ``<replace>_qx`` feature swapped for its windowed version
    ``<replace>_<start>_<end>_qx``, resolved as a formula of its own. (The legacy evaluation swapped it in the
    expanded text, where ``_qx`` no longer occurs, so every variant repeated its base formula.)

    Parameters
    ----------
    entries : mapping of str to Formula
        The named formulas; those not enabled are skipped.
    window_variants : sequence of WindowVariant
        The variants to add; one whose `WindowVariant.only_if_cutoff_mode` differs from `cutoff_mode` is skipped.
    unavailable : iterable of str
        The columns that cannot be built (see `resolve`).
    n_bins : int
        The number of week bins.
    cutoff_mode : str
        The cutoff mode of the run.

    Returns
    -------
    formulas : dict of str to str
        The resolved formula text of each enabled formula and variant (named ``<base>_<start>_<end>``).
    resolution : dict of str to dict
        For each, ``original_formula``, ``expanded_formula``, ``resolved_formula`` and ``dropped_terms``.

    Raises
    ------
    ValueError
        `entries` is empty, an enabled formula has no text, a variant's base is not an enabled formula, or a
        formula has no constructible term.
    """
    if not entries:
        raise ValueError("Spec must define models.formulas with named entries containing 'text'.")
    out, resolution = {}, {}

    def add(name, text):
        """Resolve one formula and record it and how it was resolved."""
        resolved, dropped = resolve(text, unavailable, n_bins)
        out[name] = resolved
        resolution[name] = {
            "original_formula": text,
            "expanded_formula": expand(text, n_bins),
            "resolved_formula": resolved,
            "dropped_terms": dropped,
        }

    texts = {}
    for name, entry in entries.items():
        if not entry.enabled:
            continue
        if not entry.text:
            raise ValueError(f"Formula '{name}' is enabled but has no non-empty 'text' in spec.")
        texts[name] = entry.text
        add(name, entry.text)
    for variant in window_variants:
        if variant.only_if_cutoff_mode and cutoff_mode != variant.only_if_cutoff_mode:
            continue
        if variant.base not in texts:
            raise ValueError("window_variants.base_name must match a name in models.formulas.")
        feature = re.compile(rf"\b{re.escape(variant.replace)}_qx\b")
        for start, end in variant.windows:
            add(f"{variant.base}_{start}_{end}", feature.sub(f"{variant.replace}_{start}_{end}_qx", texts[variant.base]))
    return out, resolution


def _as_float(values) -> np.ndarray:
    """Return values as floats, with ``NaN`` for what is not numeric.

    Parameters
    ----------
    values : array-like
        The values, e.g. a ``pandas.Series``.

    Returns
    -------
    numpy.ndarray of float64
        The values converted by ``pandas.to_numeric(errors="coerce")``.
    """
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


def sample_support(weekly, formulas: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Find the samples for which every formula's predictors are finite.

    A predictor that is missing, or never finite, is refused; the message names a rain-horizon cause when the
    features recorded one (``weekly.attrs["unavailable_rain_predictors"]``).

    Parameters
    ----------
    weekly : pandas.DataFrame
        The weekly features, one row per sample, with the predictor columns (and the rain-horizon record in
        ``attrs``).
    formulas : dict of str to str
        The resolved formula text of each enabled formula.

    Returns
    -------
    positions : numpy.ndarray of int64, shape (n,)
        The positions of the samples that are finite in every required column.
    diagnostics : dict
        ``policy`` (``"common_complete"``), ``required_columns``, ``rows_before``, ``rows_after``,
        ``rows_excluded`` and ``nonfinite_by_column`` (the columns with non-finite rows and their counts).

    Raises
    ------
    ValueError
        A predictor column is missing or has no finite value.
    """
    feature_map = {name: list(dict.fromkeys(formula_columns(text))) for name, text in formulas.items()}
    unavailable = weekly.attrs.get("unavailable_rain_predictors", {})
    problems = []
    for name, columns in feature_map.items():
        missing = [c for c in columns if c not in weekly.columns]
        if missing:
            problems.append(f"Formula '{name}' is missing predictor columns: {', '.join(missing)}")
        for column in columns:
            if column not in weekly.columns or np.isfinite(_as_float(weekly[column])).any():
                continue
            details = unavailable.get(column)
            if details:
                problems.append(
                    f"Formula '{name}': {column} cannot be constructed under rain_day_max={details['rain_day_max']} "
                    f"(window={details['window']}, week={details['week']}; no complete start days)."
                )
            else:
                problems.append(f"Formula '{name}': {column} has no finite values and cannot be used for fitting.")
    if problems:
        raise ValueError("Formula feature validation failed before model fitting:\n- " + "\n- ".join(problems))
    required = list(dict.fromkeys(c for columns in feature_map.values() for c in columns))
    keep = np.ones(len(weekly), dtype=bool)
    excluded = {}
    for column in required:
        finite = np.isfinite(_as_float(weekly[column]))
        excluded[column] = int((~finite).sum())
        keep &= finite
    diagnostics = {
        "policy": "common_complete",
        "required_columns": required,
        "rows_before": int(len(weekly)),
        "rows_after": int(keep.sum()),
        "rows_excluded": int((~keep).sum()),
        "nonfinite_by_column": {c: n for c, n in excluded.items() if n},
    }
    return np.flatnonzero(keep), diagnostics

