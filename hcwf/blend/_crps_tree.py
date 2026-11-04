"""The CRPS random forest over onset lead days: the trees and their split search.

From onset-forest (``github.com/AI-Almanac/onset-forest``, ``forest.py`` at 61a7f5a), itself ported from the science
team's notebook (``crps_rf_three_experiments_engineered_raw_with_without_climatology.ipynb``). The split search,
node-level feature sampling and RNG consumption order are unchanged, so a seeded fit grows onset-forest's (and the
notebook's) trees exactly. Added: a fitted forest as flat arrays (`CRPSForest.to_arrays`, `CRPSForest.from_arrays`),
so a trained blend stores it without pickles; and trees grown in worker processes rather than threads (each tree has
its own seed and bootstrap rows, so they are the same trees): the split kernel releases the GIL but the per-node
bookkeeping does not, and threads used about a fifth of the cores.

The module holds the numba kernels (`_scan_feature`, `_best_split_kernel`), the tree (`_CRPSTree`) and the forest
(`CRPSForest`), which `hcwf.blend.random_forest` fits and `hcwf.blend.trained` stores. `lead_day_histogram` uses the
lead counts of `hcwf.onset.engine`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from hcwf.onset.engine import lead_counts

__all__ = []


def _jit(**options):
    """Return a decorator that compiles with ``njit``, caching on disk when a cache location is writable.

    Numba resolves a cache locator at decoration time (the package's ``__pycache__``, then ``NUMBA_CACHE_DIR`` or
    the user cache dir) and raises if none is writable, e.g. a read-only install inside a container. Fall back to
    in-memory compilation then. The compiled functions release the GIL (``nogil=True``).

    Parameters
    ----------
    **options : dict
        Further options passed on to ``numba.njit`` (for example ``inline="always"``).

    Returns
    -------
    callable
        A decorator taking a function and returning its compiled version.
    """

    def decorate(fn):
        """Compile `fn` with the cache on, or without it when numba cannot find a cache location."""
        try:
            return njit(cache=True, nogil=True, **options)(fn)
        except RuntimeError:
            return njit(cache=False, nogil=True, **options)(fn)

    return decorate


@_jit(inline="always")
def _fen_add(bit, i, delta):
    """Add `delta` at position `i` of a Fenwick (binary indexed) tree, in place.

    Compiled with numba (``njit``, inlined).

    Parameters
    ----------
    bit : numpy.ndarray of float64, shape (m + 1,)
        The tree; index 0 is unused, positions are 1-based.
    i : int
        The 1-based position to add at.
    delta : float
        The amount to add.
    """
    n = bit.shape[0] - 1
    while i <= n:
        bit[i] += delta
        i += i & (-i)


@_jit(inline="always")
def _fen_sum(bit, i):
    """Sum the first `i` positions of a Fenwick (binary indexed) tree.

    Compiled with numba (``njit``, inlined).

    Parameters
    ----------
    bit : numpy.ndarray of float64, shape (m + 1,)
        The tree; index 0 is unused, positions are 1-based.
    i : int
        The last 1-based position to include.

    Returns
    -------
    float
        The prefix sum over positions ``1..i``.
    """
    out = 0.0
    while i > 0:
        out += bit[i]
        i -= i & (-i)
    return out


@_jit(inline="always")
def _dist_to_set(bit_count, bit_sum, total_count, total_sum, idx, val):
    """Return the sum of ``|val - y|`` over the values ``y`` held in a pair of Fenwick trees.

    Compiled with numba (``njit``, inlined).

    Parameters
    ----------
    bit_count : numpy.ndarray of float64, shape (m + 1,)
        Fenwick tree of how many values sit at each dense rank.
    bit_sum : numpy.ndarray of float64, shape (m + 1,)
        Fenwick tree of the sum of the values at each dense rank.
    total_count : int
        The number of values in the set.
    total_sum : float
        The sum of the values in the set.
    idx : int
        The 1-based dense rank of `val`.
    val : float
        The value to measure the distance from.

    Returns
    -------
    float
        The summed absolute distance from `val` to the set.
    """
    count_le = _fen_sum(bit_count, idx)
    sum_le = _fen_sum(bit_sum, idx)
    count_gt = total_count - count_le
    sum_gt = total_sum - sum_le
    return val * count_le - sum_le + sum_gt - val * count_gt


@_jit()
def _scan_feature(x_sorted, y_sorted, yi_sorted, m, n, parent_T, total_sum, min_leaf):
    """Scan one feature's sorted values for its best split by the CRPS (pairwise absolute difference) criterion.

    Compiled with numba (``njit``): the arrays must be NumPy arrays of the stated dtypes. Each distinct value of the
    feature is a possible cut; the score of a cut is the mean pairwise absolute difference of the targets on the left
    plus that on the right, kept up to date with two pairs of Fenwick trees. Cuts that leave fewer than `min_leaf` rows
    on a side are skipped; the threshold is the midpoint between the value and the next one.

    Parameters
    ----------
    x_sorted : numpy.ndarray of float64, shape (n,)
        The feature's values, ascending.
    y_sorted : numpy.ndarray of float64, shape (n,)
        The targets in the order of `x_sorted`.
    yi_sorted : numpy.ndarray of int64, shape (n,)
        The 1-based dense ranks of the targets (see `_compress_y`), in the order of `x_sorted`.
    m : int
        The number of distinct targets.
    n : int
        The number of rows.
    parent_T : float
        The sum of ``|y_i - y_j|`` over the pairs of rows before the split.
    total_sum : float
        The sum of the targets.
    min_leaf : int
        The fewest rows allowed on a side.

    Returns
    -------
    best_score : float
        The lowest score found; ``inf`` if there is no valid cut.
    best_threshold : float
        The threshold of that cut; ``NaN`` if there is none.
    found : bool
        Whether a valid cut exists.
    """
    lc = np.zeros(m + 1)
    ls = np.zeros(m + 1)
    rc = np.zeros(m + 1)
    rs = np.zeros(m + 1)

    for k in range(n):
        _fen_add(rc, yi_sorted[k], 1.0)
        _fen_add(rs, yi_sorted[k], y_sorted[k])

    n_left = 0
    n_right = n
    total_left = 0.0
    total_right = total_sum
    T_left = 0.0
    T_right = parent_T

    best_score = np.inf
    best_threshold = np.nan
    found = False

    pos = 0
    while pos < n:
        x_val = x_sorted[pos]
        next_pos = pos
        while next_pos < n and x_sorted[next_pos] == x_val:
            val = y_sorted[next_pos]
            idx = yi_sorted[next_pos]

            T_right -= _dist_to_set(rc, rs, n_right, total_right, idx, val)
            _fen_add(rc, idx, -1.0)
            _fen_add(rs, idx, -val)
            n_right -= 1
            total_right -= val

            T_left += _dist_to_set(lc, ls, n_left, total_left, idx, val)
            _fen_add(lc, idx, 1.0)
            _fen_add(ls, idx, val)
            n_left += 1
            total_left += val

            next_pos += 1

        if next_pos < n and n_left >= min_leaf and n_right >= min_leaf:
            score = T_left / n_left + T_right / n_right
            if score < best_score:
                next_x = x_sorted[next_pos]
                best_threshold = x_val + 0.5 * (next_x - x_val)
                best_score = score
                found = True

        pos = next_pos

    return best_score, best_threshold, found


@_jit()
def _best_split_kernel(X, rows, y, yi, feature_ids, parent_T, total_sum, m, min_leaf):
    """Find the best split of the rows of X over the candidate features.

    Compiled with numba (``njit``): the arrays must be NumPy arrays of the stated dtypes. A feature that is constant
    over the rows is skipped; ties keep the first feature of `feature_ids`.

    Parameters
    ----------
    X : numpy.ndarray of float64, shape (sample, feature)
        The whole feature matrix.
    rows : numpy.ndarray of int64, shape (n,)
        The rows of `X` at this node.
    y : numpy.ndarray of float64, shape (n,)
        The targets of `rows`, in that order.
    yi : numpy.ndarray of int64, shape (n,)
        The 1-based dense ranks of `y`.
    feature_ids : numpy.ndarray of int64, shape (candidate,)
        The columns of `X` to try.
    parent_T : float
        The sum of ``|y_i - y_j|`` over the pairs of `y`.
    total_sum : float
        The sum of `y`.
    m : int
        The number of distinct targets.
    min_leaf : int
        The fewest rows allowed on a side.

    Returns
    -------
    best_feature : int
        The column of the best split; -1 if none.
    best_threshold : float
        Its threshold; ``NaN`` if none.
    best_score : float
        Its score; ``inf`` if none.
    found : bool
        Whether a split exists.
    """
    n = rows.shape[0]
    best_score = np.inf
    best_feature = -1
    best_threshold = np.nan
    found = False

    for f in range(feature_ids.shape[0]):
        feature = feature_ids[f]
        col = X[rows, feature]
        order = np.argsort(col, kind="mergesort")
        x_sorted = col[order]
        if x_sorted[0] == x_sorted[-1]:
            continue
        y_sorted = y[order]
        yi_sorted = yi[order]

        sc, thr, fnd = _scan_feature(
            x_sorted, y_sorted, yi_sorted, m, n, parent_T, total_sum, min_leaf
        )
        if fnd and sc < best_score:
            best_score = sc
            best_feature = feature
            best_threshold = thr
            found = True

    return best_feature, best_threshold, best_score, found


@_jit()
def _pairwise_abs_sum(y_sorted):
    """Return ``T = sum_{i<j} |y_i - y_j|`` for ascending values.

    Compiled with numba (``njit``).

    Parameters
    ----------
    y_sorted : numpy.ndarray of float64, shape (n,)
        The values, ascending.

    Returns
    -------
    float
        The sum of the absolute differences over all pairs.
    """
    total = 0.0
    csum = 0.0
    for i in range(y_sorted.shape[0]):
        total += y_sorted[i] * i - csum
        csum += y_sorted[i]
    return total


def _compress_y(y):
    """Return the dense ranks of the targets.

    Parameters
    ----------
    y : numpy.ndarray of float64, shape (n,)
        The targets.

    Returns
    -------
    yi : numpy.ndarray of int64, shape (n,)
        The 1-based rank of each target among the distinct values.
    m : int
        The number of distinct values.
    """
    uniq, inv = np.unique(y, return_inverse=True)
    return (inv + 1).astype(np.int64), int(uniq.shape[0])


class _Node:
    """A node of a CRPS tree: a split (feature, threshold, two children) or a leaf holding its members.

    Parameters
    ----------
    feature : int, optional
        The column the node splits on; None for a leaf.
    threshold : float, optional
        The node sends a row left when its value is at most this; None for a leaf.
    left : _Node, optional
        The child for values at or below the threshold.
    right : _Node, optional
        The child for values above the threshold.
    values : numpy.ndarray of float64, shape (member,), optional
        The training targets in the leaf (its members); None for a split node.
    """

    __slots__ = ("feature", "threshold", "left", "right", "values")

    def __init__(self, feature=None, threshold=None, left=None, right=None, values=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.values = values

    @property
    def is_leaf(self):
        """Whether the node is a leaf (holds ``values``)."""
        return self.values is not None

    def leaf_values(self, x):
        """Return the members of the leaf that a row reaches.

        Parameters
        ----------
        x : numpy.ndarray of float64, shape (feature,)
            One row of features.

        Returns
        -------
        numpy.ndarray of float64, shape (member,)
            The leaf's members.
        """
        node = self
        while node.values is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.values


class _CRPSTree:
    """One tree of the forest, grown by the CRPS split criterion.

    Parameters
    ----------
    min_samples_leaf : int, default 5
        The fewest rows in a leaf; a node with fewer than twice this many is not split.
    random_state : int, optional
        The seed of the tree's generator, used to sample the candidate features at each node.
    max_depth : int, optional
        The deepest level to split; None for no limit.
    max_features : {"sqrt", None}, default "sqrt"
        The candidate features per node: ``int(sqrt(p))`` drawn without replacement, or all of them.

    Attributes
    ----------
    root_ : _Node or None
        The root of the grown tree; None before fitting.
    n_features_in_ : int or None
        The number of columns the tree was fitted on.
    rng_ : numpy.random.Generator
        The generator seeded with `random_state`.
    """

    def __init__(self, min_samples_leaf=5, random_state=None, max_depth=None, max_features="sqrt"):
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.root_ = None
        self.n_features_in_ = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        """Grow the tree on all rows of X.

        Parameters
        ----------
        X : array_like of float, shape (sample, feature)
            The features.
        y : array_like of float, shape (sample,)
            The targets.

        Returns
        -------
        _CRPSTree
            The tree itself.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        return self.fit_rows(X, y, np.arange(X.shape[0], dtype=np.int64))

    def fit_rows(self, X, y, rows):
        """Grow the tree on ``X[rows]`` and ``y[rows]``, in that order, without copying `X`.

        Nodes hold row indices into `X` and the split kernel reads each candidate column through them, so a node keeps
        O(rows) memory rather than a copy of its rows of `X`. The tree is the one ``fit(X[rows], y[rows])`` grows.

        Parameters
        ----------
        X : numpy.ndarray of float64, shape (sample, feature)
            The whole feature matrix.
        y : numpy.ndarray of float64, shape (sample,)
            The targets of every row of `X`.
        rows : array_like of int, shape (n,)
            The rows to grow on, repeats allowed (a bootstrap sample).

        Returns
        -------
        _CRPSTree
            The tree itself.
        """
        self.n_features_in_ = X.shape[1]
        self.root_ = self._grow(X, y, np.asarray(rows, dtype=np.int64), 0)
        return self

    def _grow(self, X, y_all, rows, depth):
        """Grow the subtree for some rows, recursively.

        A node becomes a leaf when it has fewer than ``2 * min_samples_leaf`` rows, is at `max_depth`, has one target
        value, has no valid split, the best split does not lower the score, or a side would be too small.

        Parameters
        ----------
        X : numpy.ndarray of float64, shape (sample, feature)
            The whole feature matrix.
        y_all : numpy.ndarray of float64, shape (sample,)
            The targets of every row of `X`.
        rows : numpy.ndarray of int64, shape (n,)
            The rows at this node.
        depth : int
            The node's depth; the root is 0.

        Returns
        -------
        _Node
            The root of the subtree.
        """
        n = rows.shape[0]
        p = X.shape[1]
        y = y_all[rows]
        ml = self.min_samples_leaf
        if n < 2 * ml:
            return _Node(values=y)
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(values=y)
        if np.all(y == y[0]):
            return _Node(values=y)

        y_sorted = np.sort(y)
        parent_T = _pairwise_abs_sum(y_sorted)
        parent_score = parent_T / n
        total_sum = float(y_sorted.sum())

        yi, m = _compress_y(y)
        feature_ids = self._candidate_features(p)

        feat, thr, score, found = _best_split_kernel(
            X, rows, y, yi, feature_ids, parent_T, total_sum, m, ml
        )

        if not found:
            return _Node(values=y)
        if score >= parent_score - 1e-12:
            return _Node(values=y)

        left_mask = X[rows, feat] <= thr
        nl = left_mask.sum()
        if nl < ml or (n - nl) < ml:
            return _Node(values=y)

        left = self._grow(X, y_all, rows[left_mask], depth + 1)
        right = self._grow(X, y_all, rows[~left_mask], depth + 1)
        return _Node(feature=int(feat), threshold=float(thr), left=left, right=right)

    def _candidate_features(self, p):
        """Sample candidate predictors independently at each node, as sklearn RF does.

        Parameters
        ----------
        p : int
            The number of features.

        Returns
        -------
        numpy.ndarray of int64, shape (candidate,)
            All columns for ``max_features=None``, else ``max(1, int(sqrt(p)))`` columns drawn without replacement
            from the tree's generator.

        Raises
        ------
        ValueError
            `max_features` is neither ``"sqrt"`` nor None.
        """
        if self.max_features is None:
            return np.arange(p, dtype=np.int64)
        if self.max_features == "sqrt":
            n_candidates = max(1, int(np.sqrt(p)))
            return self.rng_.choice(p, size=n_candidates, replace=False).astype(np.int64)
        raise ValueError('max_features must be "sqrt" or None.')

    def predict_members_one(self, x):
        """Return the members of the leaf that a row reaches.

        Parameters
        ----------
        x : numpy.ndarray of float64, shape (feature,)
            One row of features.

        Returns
        -------
        numpy.ndarray of float64, shape (member,)
            The leaf's members.
        """
        return self.root_.leaf_values(x)


def _fit_one_tree(X, y, idx, min_samples_leaf, max_depth, max_features, seed):
    """Grow one tree on a bootstrap sample; the unit of work of the worker processes.

    Parameters
    ----------
    X : numpy.ndarray of float64, shape (sample, feature)
        The whole feature matrix.
    y : numpy.ndarray of float64, shape (sample,)
        The targets.
    idx : numpy.ndarray of int, shape (n,)
        The rows to grow on.
    min_samples_leaf : int
        The fewest rows in a leaf.
    max_depth : int or None
        The deepest level to split.
    max_features : {"sqrt", None}
        The candidate features per node.
    seed : int
        The seed of the tree's generator.

    Returns
    -------
    _CRPSTree
        The fitted tree.
    """
    tree = _CRPSTree(
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        max_depth=max_depth,
        max_features=max_features,
    )
    tree.fit_rows(X, y, idx)
    return tree


def _route_to_leaves(
    node: _Node, X: np.ndarray, rows: np.ndarray
) -> Iterator[tuple[_Node, np.ndarray]]:
    """Yield every leaf that some row reaches, with those rows.

    Parameters
    ----------
    node : _Node
        The root to route from.
    X : numpy.ndarray of float64, shape (sample, feature)
        The feature matrix.
    rows : numpy.ndarray of int, shape (n,)
        The rows of `X` to route.

    Yields
    ------
    leaf : _Node
        A leaf reached by at least one row.
    leaf_rows : numpy.ndarray of int
        The rows that reach it.
    """
    stack = [(node, rows)]
    while stack:
        node, rows = stack.pop()
        if rows.size == 0:
            continue
        if node.is_leaf:
            yield node, rows
            continue
        goes_left = X[rows, node.feature] <= node.threshold
        stack.append((node.left, rows[goes_left]))
        stack.append((node.right, rows[~goes_left]))


def lead_day_histogram(members: np.ndarray, max_day: int) -> np.ndarray:
    """Count the members equal to each lead day ``1..max_day``, then the members later than `max_day`.

    These are the engine's lead counts (`hcwf.onset.engine.lead_counts`, the layout of ``onset_count``) without lead
    0, and the rest as "later".

    Parameters
    ----------
    members : numpy.ndarray of float, shape (member,)
        The lead days of the members.
    max_day : int
        The last lead day counted individually.

    Returns
    -------
    numpy.ndarray of float64, shape (max_day + 1,)
        The counts for days 1 to `max_day`, then the count of members above `max_day`.
    """
    members = np.asarray(members, dtype=np.float64)
    counts = lead_counts(members, max_day, axis=0)
    later = np.count_nonzero(members > max_day)
    return np.append(counts[1:], later).astype(np.float64)


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    """Divide each row by its sum.

    Parameters
    ----------
    counts : numpy.ndarray of float64, shape (n, k)
        The counts; no row may sum to zero.

    Returns
    -------
    numpy.ndarray of float64, shape (n, k)
        The shares, each row summing to 1.
    """
    return counts / counts.sum(axis=1, keepdims=True)


class CRPSForest:
    """Bootstrapped CRPS trees; predictions pool the leaf members of every tree.

    Parameters
    ----------
    n_estimators : int, default 100
        The number of trees.
    min_samples_leaf : int, default 5
        The fewest rows in a leaf.
    bootstrap : bool, default True
        Grow each tree on a bootstrap sample of the rows (with replacement) rather than on all rows.
    max_samples : int or float, optional
        The size of each bootstrap sample: a count in ``[1, n]`` or a fraction in ``(0, 1]``; None for all rows.
        Only valid with `bootstrap`.
    random_state : int, optional
        The seed from which every tree's seed and bootstrap rows are drawn.
    max_depth : int, optional
        The deepest level to split; None for no limit.
    n_jobs : int, default -1
        The number of worker processes (joblib; -1 for all cores).
    max_features : {"sqrt", None}, default "sqrt"
        The candidate features per node.

    Attributes
    ----------
    n_features_in_ : int
        The number of columns fitted on (set by `CRPSForest.fit`).
    n_samples_bootstrap_ : int
        The rows in each tree's sample.
    trees_ : list of _CRPSTree
        The fitted trees.
    feature_importances_ : numpy.ndarray of float64, shape (feature,)
        The share of all splits made on each feature.
    """

    def __init__(
        self,
        n_estimators=100,
        min_samples_leaf=5,
        bootstrap=True,
        max_samples=None,
        random_state=None,
        max_depth=None,
        n_jobs=-1,
        max_features="sqrt",
    ):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.max_features = max_features

    def fit(self, X, y):
        """Grow the trees, in worker processes.

        Each tree has its own seed and bootstrap rows, drawn up front in the reference order, so the result does not
        depend on `n_jobs`.

        Parameters
        ----------
        X : array_like of float, shape (sample, feature)
            The features; all finite.
        y : array_like of float, shape (sample,)
            The targets; all finite.

        Returns
        -------
        CRPSForest
            The forest itself; `ValueError` is raised for invalid inputs or parameters (see
            `CRPSForest._check_fit_inputs`).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._check_fit_inputs(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_samples_bootstrap_ = self._resolve_bootstrap_size(X.shape[0])

        seeds, idxs = self._draw_tree_seeds_and_samples(X.shape[0])
        # loky keeps its workers between fits and hands them X as a memory map.
        self.trees_ = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_fit_one_tree)(
                X, y, idxs[i], self.min_samples_leaf, self.max_depth, self.max_features, seeds[i]
            )
            for i in range(self.n_estimators)
        )
        self.feature_importances_ = self._split_count_importances()
        return self

    def _check_fit_inputs(self, X, y):
        """Check the inputs and parameters of a fit.

        Parameters
        ----------
        X : numpy.ndarray of float64
            The features; must be 2-D with one row per target and all finite.
        y : numpy.ndarray of float64, shape (sample,)
            The targets; must be finite.

        Raises
        ------
        ValueError
            `X` is not 2-D, `X` and `y` differ in length, `n_estimators` or `min_samples_leaf` is below 1,
            `max_features` is not ``"sqrt"`` or None, or `X` or `y` holds a non-finite value.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if self.max_features not in {"sqrt", None}:
            raise ValueError('max_features must be "sqrt" or None.')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("This implementation requires finite numeric X and y.")

    def _resolve_bootstrap_size(self, n_samples):
        """Return the number of rows in each tree's sample.

        Parameters
        ----------
        n_samples : int
            The number of rows of the training data.

        Returns
        -------
        int
            The resolved `max_samples` with bootstrap, else `n_samples`.

        Raises
        ------
        ValueError
            `max_samples` is set without `bootstrap`, or is invalid.
        """
        if self.bootstrap:
            return self._resolve_max_samples(n_samples)
        if self.max_samples is not None:
            raise ValueError("max_samples cannot be set if bootstrap=False.")
        return n_samples

    def _draw_tree_seeds_and_samples(self, n_samples):
        """Draw each tree's seed first, then its bootstrap indices (the reference order).

        Parameters
        ----------
        n_samples : int
            The number of rows of the training data.

        Returns
        -------
        seeds : list of int
            One seed per tree.
        idxs : list of numpy.ndarray of int
            The rows of each tree: a bootstrap draw of `n_samples_bootstrap_` rows, or all rows without bootstrap.
        """
        rng = np.random.default_rng(self.random_state)
        base = np.arange(n_samples)
        seeds = []
        idxs = []
        for _ in range(self.n_estimators):
            seeds.append(int(rng.integers(0, np.iinfo(np.int32).max)))
            if self.bootstrap:
                idxs.append(rng.integers(0, n_samples, size=self.n_samples_bootstrap_))
            else:
                idxs.append(base)
        return seeds, idxs

    def _split_count_importances(self):
        """Return the fraction of all internal-node splits assigned to each predictor.

        Returns
        -------
        numpy.ndarray of float64, shape (feature,)
            The shares, summing to 1 (all zero when no tree splits).
        """
        split_counts = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.trees_:
            stack = [tree.root_]
            while stack:
                node = stack.pop()
                if node is None or node.is_leaf:
                    continue
                split_counts[node.feature] += 1.0
                stack.append(node.left)
                stack.append(node.right)
        total_splits = split_counts.sum()
        return split_counts / total_splits if total_splits > 0 else split_counts

    def _resolve_max_samples(self, n_samples):
        """Turn `max_samples` into a number of rows.

        Parameters
        ----------
        n_samples : int
            The number of rows of the training data.

        Returns
        -------
        int
            `n_samples` for None, an integer as is, a float as ``max(1, round(n_samples * max_samples))``.

        Raises
        ------
        ValueError
            `max_samples` is a bool, an integer outside ``[1, n_samples]``, a float outside ``(0, 1]``, or another type.
        """
        ms = self.max_samples
        if ms is None:
            return n_samples
        if isinstance(ms, (bool, np.bool_)):
            raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")
        if isinstance(ms, (int, np.integer)):
            ms = int(ms)
            if not 1 <= ms <= n_samples:
                raise ValueError("max_samples int must be in [1, n_samples].")
            return ms
        if isinstance(ms, (float, np.floating)):
            ms = float(ms)
            if not 0.0 < ms <= 1.0:
                raise ValueError("max_samples float must be in (0.0, 1.0].")
            return max(1, int(round(n_samples * ms)))
        raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")

    def _as_feature_matrix(self, X):
        """Check that the forest is fitted and return `X` as a 2-D float matrix.

        Parameters
        ----------
        X : array_like of float
            The features; a 1-D input is one row.

        Returns
        -------
        numpy.ndarray of float64, shape (sample, feature)
            The matrix.

        Raises
        ------
        ValueError
            The number of columns differs from the fitted one.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return X

    def predict_members(self, X):
        """Return the pooled leaf members of every tree for each row.

        Parameters
        ----------
        X : array_like of float, shape (sample, feature)
            The features.

        Returns
        -------
        list of numpy.ndarray of float64
            For each row, the members of the leaf it reaches in every tree, concatenated over the trees.
        """
        X = self._as_feature_matrix(X)
        return [np.concatenate([t.predict_members_one(x) for t in self.trees_]) for x in X]

    def predict(self, X):
        """Return the mean of the pooled leaf members for each row.

        Parameters
        ----------
        X : array_like of float, shape (sample, feature)
            The features.

        Returns
        -------
        numpy.ndarray of float64, shape (sample,)
            The mean member.
        """
        return np.array([v.mean() for v in self.predict_members(X)], dtype=float)

    def predict_lead_day_distribution(self, X, max_day=28):
        """Return the shares of the pooled leaf members per lead day ``1..max_day``, then later than `max_day`.

        Equivalent to histogramming `CRPSForest.predict_members` row by row, but routes all rows through each tree
        at once.

        Parameters
        ----------
        X : array_like of float, shape (sample, feature)
            The features.
        max_day : int, default 28
            The last lead day counted individually.

        Returns
        -------
        numpy.ndarray of float64, shape (sample, max_day + 1)
            The shares of the members on days 1 to `max_day` and after it; each row sums to 1.
        """
        X = self._as_feature_matrix(X)
        counts = np.zeros((X.shape[0], max_day + 1), dtype=np.float64)
        all_rows = np.arange(X.shape[0])
        for tree in self.trees_:
            for leaf, rows in _route_to_leaves(tree.root_, X, all_rows):
                counts[rows] += lead_day_histogram(leaf.values, max_day)
        return _normalize_rows(counts)

    def _check_is_fitted(self):
        """Check that `CRPSForest.fit` has run.

        Raises
        ------
        RuntimeError
            The forest is not fitted.
        """
        if not hasattr(self, "trees_"):
            raise RuntimeError("Estimator is not fitted yet. Call fit(X, y) first.")

    # -- as arrays ----------------------------------------------------------

    _PARAMS = ("n_estimators", "min_samples_leaf", "bootstrap", "max_samples", "random_state",
               "max_depth", "n_jobs", "max_features")
    """The constructor parameters that `CRPSForest.to_arrays` keeps in its metadata."""

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Return the fitted trees as flat arrays: every node of every tree in preorder.

        Each node has its split (feature, threshold) and children, or its leaf members.

        Returns
        -------
        meta : dict
            The constructor parameters, ``n_features_in`` and ``n_samples_bootstrap``; plain Python values.
        arrays : dict of str to numpy.ndarray
            ``feature`` (int64, -1 for a leaf), ``threshold`` (float64), ``left`` and ``right`` (int64 node indices,
            -1 for a leaf), ``value_start`` and ``value_count`` (int64, the leaf's slice of ``values``), ``values``
            (float64, all leaf members), ``roots`` (int64, each tree's first node) and ``feature_importances``
            (float64), all over the nodes except ``values``, ``roots`` and ``feature_importances``.
            `RuntimeError` is raised if the forest is not fitted.
        """
        self._check_is_fitted()
        feature, threshold, left, right, start, count, values, roots = [], [], [], [], [], [], [], []
        n_values = 0
        for tree in self.trees_:
            roots.append(len(feature))
            stack = [(tree.root_, None, None)]  # (node, parent index, is left child)
            while stack:
                node, parent, is_left = stack.pop()
                index = len(feature)
                if parent is not None:
                    (left if is_left else right)[parent] = index
                if node.is_leaf:
                    feature.append(-1)
                    threshold.append(np.nan)
                    left.append(-1)
                    right.append(-1)
                    start.append(n_values)
                    count.append(len(node.values))
                    values.append(np.asarray(node.values, dtype=np.float64))
                    n_values += len(node.values)
                else:
                    feature.append(node.feature)
                    threshold.append(node.threshold)
                    left.append(-1)
                    right.append(-1)
                    start.append(-1)
                    count.append(0)
                    stack.append((node.right, index, False))
                    stack.append((node.left, index, True))
        arrays = {
            "feature": np.asarray(feature, dtype=np.int64),
            "threshold": np.asarray(threshold, dtype=np.float64),
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "value_start": np.asarray(start, dtype=np.int64),
            "value_count": np.asarray(count, dtype=np.int64),
            "values": np.concatenate(values) if values else np.zeros(0),
            "roots": np.asarray(roots, dtype=np.int64),
            "feature_importances": np.asarray(self.feature_importances_, dtype=np.float64),
        }
        meta = {name: getattr(self, name) for name in self._PARAMS}
        meta.update(n_features_in=int(self.n_features_in_), n_samples_bootstrap=int(self.n_samples_bootstrap_))
        return meta, arrays

    @classmethod
    def from_arrays(cls, meta: dict, arrays) -> CRPSForest:
        """Rebuild a fitted forest from `CRPSForest.to_arrays`.

        Parameters
        ----------
        meta : dict
            The metadata returned by ``to_arrays``.
        arrays : mapping of str to numpy.ndarray
            The arrays returned by ``to_arrays``.

        Returns
        -------
        CRPSForest
            The fitted forest; its predictions equal those of the forest that was saved.
        """
        forest = cls(**{name: meta[name] for name in cls._PARAMS})
        forest.n_features_in_ = int(meta["n_features_in"])
        forest.n_samples_bootstrap_ = int(meta["n_samples_bootstrap"])
        feature, threshold = arrays["feature"], arrays["threshold"]
        left, right = arrays["left"], arrays["right"]
        start, count, values = arrays["value_start"], arrays["value_count"], arrays["values"]

        def node(i):
            """Build the subtree rooted at node `i` of the flat arrays."""
            if feature[i] < 0:
                return _Node(values=values[start[i]:start[i] + count[i]].copy())
            return _Node(feature=int(feature[i]), threshold=float(threshold[i]), left=node(int(left[i])),
                         right=node(int(right[i])))

        trees = []
        for root in arrays["roots"]:
            tree = _CRPSTree(min_samples_leaf=forest.min_samples_leaf, max_depth=forest.max_depth,
                             max_features=forest.max_features)
            tree.n_features_in_ = forest.n_features_in_
            tree.root_ = node(int(root))
            trees.append(tree)
        forest.trees_ = trees
        forest.feature_importances_ = np.asarray(arrays["feature_importances"], dtype=np.float64)
        return forest
