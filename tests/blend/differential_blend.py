"""Differential check: the same synthetic inputs through onset_blending's blending methods and momp.blend's.

    python tests/blend/differential_blend.py dump momp OUT.pkl                 # in ROMP's env
    <onset_blending>/.venv/bin/python tests/blend/differential_blend.py dump onset_blending OUT.pkl
    python tests/blend/differential_blend.py compare A.pkl B.pkl

`dump` runs design, logit (final fit, global/local/cluster CV, fold models), Platt calibration, the random forest
(fit, cross-validation, serial and with tree workers) and the MME optimiser on seeded data and pickles every result
as arrays; `compare` demands them bit-identical. tests/blend/test_blend_differential.py runs all three steps when the
onset_blending environment exists.
"""

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.blend._samples import BINS, FrameSamples  # noqa: E402  (imports nothing of momp)

MODULES = {
    "onset_blending": {"design": "onset_blending.core.design", "logit": "onset_blending.blend.logit",
                       "calibration": "onset_blending.core.calibration", "mme": "onset_blending.blend.mme",
                       "random_forest": "onset_blending.blend.random_forest"},
    "momp": {name: f"momp.blend.{name}" for name in ("design", "logit", "calibration", "mme", "random_forest")},
}
FORMULA = "outcome ~ x1 + x2 + x1:x2"


def table(seed=0, n_cells=8, years=range(2001, 2009), per=6):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame([{"id": f"c{c}", "year": y, "issue": k, "cluster": c % 3}
                          for c in range(n_cells) for y in years for k in range(per)])
    frame["x1"] = rng.normal(size=len(frame))
    frame["x2"] = rng.normal(size=len(frame))
    score = np.column_stack([frame["x1"], frame["x2"], rng.normal(size=len(frame))])
    frame["outcome"] = [BINS[k] for k in (score + rng.gumbel(size=score.shape)).argmax(axis=1)]
    frame.loc[rng.random(len(frame)) < 0.05, "outcome"] = np.nan
    return frame


def model_arrays(model):
    meta, arrays = model.to_arrays()
    return {"meta": meta, "arrays": arrays}


def run(package: str) -> dict:
    m = {name: importlib.import_module(path) for name, path in MODULES[package].items()}
    out = {}
    frame = table()
    samples = FrameSamples(frame)
    n = len(frame)
    rows = np.arange(n)
    year, outcome = frame["year"].to_numpy(), frame["outcome"].to_numpy(dtype=object)
    trainable = np.isin(frame["id"], ["c0", "c1", "c2", "c4", "c6"])

    plan = m["design"].plan(FORMULA, samples.column_dtypes())
    out["design"] = {"columns": plan.columns, "factors": plan.factors, "matrix": plan.build(samples.take, rows),
                     "expand": m["design"].expand("outcome ~ a_qx * b_qx", 3),
                     "resolve": m["design"].resolve("outcome ~ a_qx + b_qx", {"a_week2": 1}, 3)}

    out["final"] = model_arrays(m["logit"].fit_samples(FORMULA, samples, np.flatnonzero(trainable & (year < 2008))))
    for workers in (1, 3):
        folds, final = m["logit"].global_cv(
            plan, samples.take, rows, trainable, year, outcome, holdout_years=[2002, 2005, 2006],
            training_years=range(2001, 2008), true_holdout_years=(2008,), required_classes=BINS,
            final_years=range(2001, 2008), workers=workers)
        out[f"global_cv_{workers}"] = {
            "folds": [(y, t, p, model_arrays(model)) for y, t, p, model in folds],
            "final": (final[0].coef_, final[0].intercept_, final[1].mean_, final[1].scale_, final[2], final[3])}
        out[f"fold_models_{workers}"] = {
            y: model_arrays(model)
            for y, model in m["logit"].fold_models(m["logit"].fold_coefficients({y: mod for y, _, _, mod in folds})).items()}
    local = frame[frame["id"].isin(["c0", "c1", "c2"])].reset_index(drop=True)
    out["local_cv"] = m["logit"].local_cv(FORMULA, FrameSamples(local), np.arange(len(local)), [2003, 2006])
    out["cluster_cv"] = m["logit"].cluster_cv(FORMULA, samples, rows, [2002, 2005], "cluster")

    probabilities = np.random.default_rng(1).dirichlet(np.ones(3), size=n)
    codes = np.where(pd.isna(frame["outcome"]), -1, frame["outcome"].map({b: k for k, b in enumerate(BINS)}).fillna(-1)).astype(int)
    out["platt"] = m["calibration"].cross_validated(probabilities, codes, year, year != 2008, [2003, 2004, 2007])
    out["platt_weights"] = m["calibration"].export_weights(probabilities, codes, BINS)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    X[rng.random((400, 6)) < 0.05] = np.nan
    X[:, 3] = np.nan
    lead = np.clip(np.round(10 + 6 * np.nan_to_num(X[:, 0]) + rng.normal(0, 2, 400)), -3, 40)
    lead[rng.random(400) < 0.1] = np.nan
    names = [f"x{i}" for i in range(6)]
    rf = m["random_forest"]
    for jobs in (1, 3):
        params = rf.Params(trees=6, min_samples_leaf=5, max_day=28, n_jobs=jobs)
        fitted = rf.fit(X, lead, names, params)
        meta, arrays = fitted.to_arrays()
        out[f"forest_{jobs}"] = {"meta": meta, "arrays": arrays, "daily": rf.predict(fitted, X),
                                 "weekly": rf.weekly_from_daily(rf.predict(fitted, X), 7)}
    fold_years = rf.folds(range(2000, 2006), [2004, 2005])
    forest_year = np.repeat(np.arange(2000, 2006), 400 // 6 + 1)[:400]
    every = np.ones(400, dtype=bool)
    out["forest_cv"] = rf.cross_validate(X, lead, names, forest_year, every, every, fold_years,
                                         rf.Params(trees=6, min_samples_leaf=5, n_jobs=1))

    sources = [np.random.default_rng(5 + i).dirichlet(np.ones(3), size=300) for i in range(3)]
    sources[1][:5] = np.nan
    Y = np.eye(3)[np.random.default_rng(9).integers(0, 3, size=300)]
    from_package = importlib.import_module("momp.metrics.probabilistic" if package == "momp" else "onset_blending.core.metrics")
    cleaned = [from_package.clean(P) for P in sources]
    out["mme"] = m["mme"].optimise_weights(cleaned, Y)
    return out


def differences(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys {sorted(map(str, a))} vs {sorted(map(str, b))}"]
        return [d for k in a for d in differences(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: lengths {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            return []
        except AssertionError as error:
            return [f"{path}: {error}"]
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        same = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=np.asarray(a).dtype.kind == "f")
        return [] if same else [f"{path}: arrays differ"]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def main(argv):
    if argv[0] == "dump":
        with open(argv[2], "wb") as handle:
            pickle.dump(run(argv[1]), handle)
        return 0
    with open(argv[1], "rb") as first, open(argv[2], "rb") as second:
        found = differences(pickle.load(first), pickle.load(second))
    print("\n".join(found) if found else "identical")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
