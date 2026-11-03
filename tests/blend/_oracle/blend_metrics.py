"""TEST-ONLY ORACLE: the legacy blending metrics, copied verbatim from onset_blending@1e6d696
tests/_legacy/blend_evaluation_utils.py (compute_cell_metrics_fast, compute_yearly_metrics, fair Brier, RPS, AUC,
Pietra). The parity tests compare hcwf.metrics.probabilistic with these; nothing in hcwf imports this file.
The yearly AUC needs scikit-learn (imported inside compute_yearly_metrics), which is not a hcwf dependency.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata


def week_labels(n_bins):
    return [f"week{w}" for w in range(1, int(n_bins) + 1)]


def interval_bins(n_bins):
    """Multinomial outcome bins: week1..weekN + 'later' (N+1 bins)."""
    return week_labels(n_bins) + ["later"]


def _cv_weeks(df):
    """Sorted week numbers present as `cv_week<n>` columns in df."""
    return sorted(int(m.group(1)) for c in df.columns
                  for m in [re.match(r"^cv_week(\d+)$", c)] if m)


def restrict_to_allowed(df, allowed_cells):
    """Filter df to rows whose id is in allowed_cells."""
    if allowed_cells is None:
        return df
    return df[df["id"].isin(set(allowed_cells["id"]))]


def clean_probs5(P):
    """Sanitize 5-bin probability matrix: replace non-finite with 0, normalize rows."""
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs


def pooled_rps5(P, y_onehot):
    """Pooled RPS from 5-bin probability and one-hot matrices."""
    CP = np.cumsum(P, axis=1)
    CY = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((CP - CY) ** 2, axis=1)))


def _fast_auc(y01, score):
    """Efficient AUC via Wilcoxon rank-sum (matches R's fast_auc)."""
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    # The same average ranks as pd.Series(s).rank(method="average"), without
    # building a Series per call (this runs once per cell per metric).
    ranks = rankdata(s, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def compute_fair_brier5(cv_preds, allowed_cells, n_train=30):
    """Compute fair (finite-sample corrected) Brier score over the interval bins
    (week1..weekN + later), inferred from the cv_week* columns present."""
    bins = [f"week{w}" for w in _cv_weeks(cv_preds)] + ["later"]
    cols = [f"cv_{b}" for b in bins]
    sub = restrict_to_allowed(cv_preds, allowed_cells)
    sub = sub.dropna(subset=["outcome"] + cols)

    P = sub[cols].values
    Y = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins])
    bs_rows = np.sum((P - Y) ** 2, axis=1)
    corr_rows = np.sum(P * (1 - P), axis=1) / (n_train - 1)
    return float(np.mean(bs_rows - corr_rows))


def compute_cell_metrics_fast(df, allowed_cells=None):
    """
    Compute per-cell and pooled Brier/RPS/AUC/pietra from cv_* probability columns.

    Matches R compute_cell_metrics_fast:
    - Brier uses the interval bins week1..weekN + later (NOT 'earlier'),
      inferred from the cv_week* columns present
    - RPS prepends 'earlier' when a cv_earlier column is present
    - Pooled AUC uses the fast Wilcoxon rank-sum method
    - Per-bin Brier and AUC are computed for the 'ALL' row
    - Pietra (KS statistic between pos/neg score distributions) is included
    - n (number of observations) is included
    - Returns per-cell rows PLUS an 'ALL' pooled row
    """
    # Interval bins (week1..weekN + later), inferred from cv_week* columns
    _wk = [f"week{w}" for w in _cv_weeks(df)]
    bins5 = _wk + ["later"]
    cv_cols5 = [f"cv_{b}" for b in bins5]

    # RPS bins: prepend 'earlier' if that column is present
    bins_rps_desired = (["earlier"] if "cv_earlier" in df.columns else []) + bins5
    present_rps = [b for b in bins_rps_desired if f"cv_{b}" in df.columns]
    cv_cols_rps = [f"cv_{b}" for b in present_rps]

    sub = df.dropna(subset=["outcome"] + cv_cols5).copy()
    if allowed_cells is not None:
        sub = sub[sub["id"].isin(set(allowed_cells["id"]))]
    sub = sub.reset_index(drop=True) # --NEW

    if sub.empty:
        return pd.DataFrame()

    P5 = sub[cv_cols5].values.astype(float)
    Y5 = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins5])

    # RPS matrix (6-bin if earlier available)
    if cv_cols_rps != cv_cols5:
        P_rps = np.zeros((len(sub), len(present_rps)), dtype=float)
        for j, b in enumerate(present_rps):
            col = f"cv_{b}"
            if col in sub.columns:
                vals = sub[col].values.astype(float)
                vals[~np.isfinite(vals)] = 0.0
                P_rps[:, j] = vals
        Y_rps = np.column_stack([(sub["outcome"] == b).astype(float) for b in present_rps])
    else:
        P_rps = P5.copy()
        Y_rps = Y5.copy()

    # --- Per-cell metrics ---
    rows = []
    grouped = sub.groupby("id")
    positions = grouped.indices  # the rows' positions; sub's index is 0..n-1
    for cell_id, g_idx in grouped.groups.items():
        #g_P5 = P5[sub.index.get_indexer(g_idx)]
        #g_Y5 = Y5[sub.index.get_indexer(g_idx)]
        #g_Prps = P_rps[sub.index.get_indexer(g_idx)]
        #g_Yrps = Y_rps[sub.index.get_indexer(g_idx)]
        pos = positions[cell_id]
        g_P5 = P5[pos]
        g_Y5 = Y5[pos]
        g_Prps = P_rps[pos]
        g_Yrps = Y_rps[pos]

        brier = float(np.mean(np.sum((g_P5 - g_Y5) ** 2, axis=1)))
        rps = pooled_rps5(g_Prps, g_Yrps)
        auc = _fast_auc(g_Y5.flatten().astype(int), g_P5.flatten())
        rows.append({"id": cell_id, "brier": brier, "rps": rps, "auc": auc, "n": len(g_idx)})

    cell_metrics = pd.DataFrame(rows)

    # --- Pooled 'ALL' row ---
    # Need positional indexing since sub may have non-contiguous index
    sub_reset = sub.reset_index(drop=True)
    P5_all = sub_reset[cv_cols5].values.astype(float)
    Y5_all = np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in bins5])
    P_rps_all = sub_reset[cv_cols_rps].values.astype(float) if cv_cols_rps != cv_cols5 else P5_all.copy()
    Y_rps_all = (np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in present_rps])
                 if cv_cols_rps != cv_cols5 else Y5_all.copy())
    for j in range(P_rps_all.shape[1]):
        P_rps_all[~np.isfinite(P_rps_all[:, j]), j] = 0.0

    brier_all = float(np.sum((P5_all - Y5_all) ** 2) / len(P5_all))
    rps_all = pooled_rps5(P_rps_all, Y_rps_all)
    auc_all = _fast_auc(Y5_all.flatten().astype(int), P5_all.flatten())

    # Per-bin Brier and AUC (5 bins, matches R)
    brier_by_bin = {}
    auc_by_bin = {}
    for j, b in enumerate(bins5):
        brier_by_bin[f"brier_{b}"] = float(np.mean((P5_all[:, j] - Y5_all[:, j]) ** 2))
        auc_by_bin[f"auc_{b}"] = _fast_auc(Y5_all[:, j].astype(int), P5_all[:, j])

    # Pietra: KS statistic between positive and negative score distributions
    p_flat = P5_all.flatten()
    y_flat = Y5_all.flatten().astype(int)
    pos_scores = p_flat[y_flat == 1]
    neg_scores = p_flat[y_flat == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        pietra = float(ks_2samp(pos_scores, neg_scores).statistic)
    else:
        pietra = np.nan

    pooled_row = {
        "id": "ALL",
        "lat": np.nan,
        "lon": np.nan,
        "brier": brier_all,
        "rps": rps_all,
        "n": len(sub_reset),
        "auc": auc_all,
        "pietra": pietra,
    }
    pooled_row.update(brier_by_bin)
    pooled_row.update(auc_by_bin)

    pooled_df = pd.DataFrame([pooled_row])
    return pd.concat([cell_metrics, pooled_df], ignore_index=True)


def compute_yearly_metrics(cv_preds, model_name, method, n_bins):
    """Per-year pooled Brier/RPS/AUC rows over the interval bins."""
    bins = interval_bins(n_bins)
    cv_cols = [f"cv_{b}" for b in bins]
    sub = cv_preds.dropna(subset=["outcome"] + cv_cols)
    from sklearn.metrics import roc_auc_score
    rows = []
    for yr, g in sub.groupby("year"):
        P = g[cv_cols].values.astype(float)
        Y = np.column_stack([(g["outcome"] == b).astype(float) for b in bins])
        brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
        rps = pooled_rps5(P, Y)
        y_long = Y.flatten()
        p_long = P.flatten()
        try:
            auc = roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
        except Exception:
            auc = np.nan
        rows.append({
            "year": yr, "model": model_name, "cv_method": method,
            "brier": brier, "rps": rps, "auc": auc,
        })
    return rows
