"""Differential check: onset_blending's `train(config).save(dir)` against hcwf's `train(*converted(config)).save(dir)`
on the same synthetic files and configs.

    python tests/blend/differential_training.py dump hcwf OUTDIR [WORKERS]                 # in HCWF's env
    <onset_blending>/.venv/bin/python tests/blend/differential_training.py dump onset_blending OUTDIR [WORKERS]
    python tests/blend/differential_training.py compare DIR_A DIR_B

`dump` writes the synthetic observations, forecasts, weights and units of tests/blend/differential_onsets.py, trains
several blending configs on them (every CV method, climatology windows and a window variant, raw and Platt-calibrated
forecasts with exported weights, an MME, external predictions, a dissemination list, the random forest, thresholds
from a table, a cell transform and a regrid) and saves each result into OUTDIR/<config>; `compare` demands every
artifact of blending's output contract identical: CSV and JSON files byte for byte, NetCDF files variable for
variable (name, dims, dtype, values, attributes). The Pietra index is scipy's ks_2samp statistic, whose last bit
differs between scipy versions: when the two dumps ran under different versions (the oracle's own environment has
1.18, HCWF's 1.17) it is compared to 1e-12 and every other number exactly. trained.blend and config.yml are each package's own
format and are left out here (tests/blend/differential_forecast.py compares what the two trained blends forecast).
tests/blend/test_blend_training_differential.py runs all three steps when the onset_blending environment
exists.
"""
import json
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import differential_onsets as base  # noqa: E402

BLENDING_ONLY = {"trained.blend", "config.yml"}
BINS = ["week1", "week2", "week3", "week4", "later"]


def write_data(root: Path) -> None:
    base.write_data(root)
    ids = [f"{a:.2f}_{b:.2f}" for a in base.LATS for b in base.LONS]
    pd.DataFrame({"id": ids[:-1]}).to_csv(root / "dissemination.csv", index=False)
    rng = np.random.default_rng(11)
    rows = [{"id": i, "year": y, "outcome": BINS[int(rng.integers(0, 5))]} for i in ids for y in (2010, 2011, 2012, 2013)
            for _ in range(3)]
    external = pd.DataFrame(rows)
    probabilities = rng.dirichlet(np.ones(5), size=len(external))
    for j, b in enumerate(BINS):
        external[f"cv_{b}"] = probabilities[:, j]
    external.to_pickle(root / "external.pkl")


def configs(root: Path) -> dict:
    """The differential_onsets configs with evaluations that exercise the whole of the evaluation and training."""
    years = {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013}
    formulas = {"b": "outcome ~ prob_clim_qx + diff_ens_qx", "c": "outcome ~ prob_clim_qx * max_ens_5day_qx",
                "d": "outcome ~ prob_clim_qx"}
    scoring = [{"model": "ens", "variant": "fixed_cutoff", "platt_weights": True},
               {"model": "ens", "variant": "base", "platt_weights": True}]
    mme = {"enabled": True, "optimize": "always", "variants": ["fixed_cutoff"],
           "sources": [{"name": "clim_raw"}, {"name": "ens", "source": "forecast", "cal_variant": "fixed_cutoff"}]}
    everything = {"years": years, "formulas": formulas, "final_formula": "b", "forecasts": scoring, "mme": mme,
                  "cv_methods": ["global", "local", "clusters"], "cluster_var": "lat"}
    out = base.configs(root)
    out["plain"]["evaluation"] = {**everything, "cutoff_mode": "fixed_cutoff",
                                  "external_predictions": [{"name": "external", "file": str(root / "external.pkl")}]}
    out["plain"]["domain"] = {**out["plain"]["domain"], "dissemination": {"file": "dissemination.csv", "id_col": "id"},
                              "dissemination_scope": "training"}
    computed = out["computed"]
    computed["climatology"] = {**computed["climatology"], "windows": [[2000, 2012]]}
    computed["evaluation"] = {**everything, "cutoff_mode": "no_ref_filter", "mme": {**mme, "enabled": False},
                              "climatology_baselines": {"windows": [[2000, 2012]]},
                              "window_variants": [{"base": "b", "windows": [[2000, 2012]]}], "cv_methods": ["global"],
                              "forecasts": [{"model": "ens", "variant": "base", "platt_weights": True}]}
    table = out["table"]
    table["evaluation"] = {**everything, "cutoff_mode": "ref_onset", "mme": {**mme, "enabled": False},
                           "forecasts": [{"model": "ens", "variant": "ref_onset", "platt_weights": True}],
                           "random_forest": {"trees": 5, "min_samples_leaf": 5, "max_day": 28},
                           "cv_methods": ["global", "local"]}
    for name in ("transform", "regrid", "minlead"):
        out[name]["evaluation"] = {"years": years, "formulas": {"b": "outcome ~ prob_clim_qx"},
                                   "cutoff_mode": "fixed_cutoff", "cv_methods": ["global"]}
    out["regrid"]["evaluation"] = {**out["regrid"]["evaluation"], "formulas": formulas, "final_formula": "d",
                                   "cv_methods": ["global", "local"], "forecasts": scoring[:1]}
    out["transform"]["onset"]["fixed_cutoff"] = "06-10"
    out["transform"]["onset"]["reference_onset"] = {"month_day": "06-01"}
    return out


def run_onset_blending(config_dict, directory: Path, workers: int) -> None:
    sys.path.insert(0, "/home/hcwf/code/onset_blending")
    from onset_blending import BlendConfig
    from onset_blending.training import Resources, train

    train(BlendConfig.from_dict(config_dict), resources=Resources(workers=workers, budget_bytes=None)).save(directory)


def run_hcwf(config_dict, directory: Path, workers: int) -> None:
    from _recipes import converted

    from hcwf.blend.training import train

    t = converted(config_dict)
    train({**t.cfg, "parallel_workers": workers}, t.recipe, t.setup).save(directory, intermediates=("weekly",))   # onset_blending's save writes them by default


def dump(package: str, target: str, workers: int = 2) -> None:
    out = Path(target)
    out.mkdir(parents=True)
    meta = {}
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        root = Path(tmp)
        write_data(root)
        for name, config in configs(root).items():
            started = time.time()
            (run_hcwf if package == "hcwf" else run_onset_blending)(config, out / name, workers)
            meta[name] = round(time.time() - started, 2)
    import scipy

    meta["_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    meta["_scipy"] = scipy.__version__
    (out / "_meta.json").write_text(json.dumps(meta))


def identical_netcdf(a: Path, b: Path) -> list[str]:
    with xr.open_dataset(a) as x, xr.open_dataset(b) as y:
        x, y = x.load(), y.load()
    problems = []
    if set(x.variables) != set(y.variables):
        return [f"variables {sorted(x.variables)} vs {sorted(y.variables)}"]
    if x.attrs != y.attrs:
        problems.append(f"global attributes {x.attrs} vs {y.attrs}")
    for name in x.variables:
        u, v = x[name], y[name]
        if u.dims != v.dims or u.dtype != v.dtype or u.attrs != v.attrs:
            problems.append(f"{name}: {u.dims} {u.dtype} {u.attrs} vs {v.dims} {v.dtype} {v.attrs}")
        elif u.dtype.kind == "M":
            same = np.array_equal(u.values.astype("datetime64[ns]"), v.values.astype("datetime64[ns]"))
            if not same:
                problems.append(f"{name}: values differ")
        elif not (np.array_equal(u.values, v.values, equal_nan=True) if u.dtype.kind in "fc"
                  else np.array_equal(u.values, v.values)):
            problems.append(f"{name}: values differ")
    return problems


def same_but_pietra(a: Path, b: Path) -> bool:
    """Whether two score CSVs agree in every column but pietra, which agrees to 1e-12 (scipy's ks_2samp differs in the
    last bit between its versions, and the Pietra index is that statistic)."""
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False
    for column in x.columns:
        if column == "pietra":
            if not np.allclose(x[column], y[column], rtol=1e-12, atol=0, equal_nan=True):
                return False
        elif not x[column].equals(y[column]):
            return False
    return True


def differences(first: Path, second: Path, tolerate_pietra: bool = False) -> list[str]:
    """What differs between two dumps (each a folder per config), blending's own files aside; tolerate_pietra
    accepts the Pietra index's last-bit differences between scipy versions."""
    found = []
    for config in sorted(p.name for p in first.iterdir() if p.is_dir()):
        a, b = first / config, second / config
        names_a = {p.name for p in a.iterdir()} - BLENDING_ONLY
        names_b = {p.name for p in b.iterdir()} - BLENDING_ONLY
        if names_a != names_b:
            found.append(f"{config}: files {sorted(names_a ^ names_b)} only in one")
        for name in sorted(names_a & names_b):
            if name.endswith(".nc"):
                found += [f"{config}/{name}: {p}" for p in identical_netcdf(a / name, b / name)]
            elif (a / name).read_bytes() != (b / name).read_bytes():
                if tolerate_pietra and name.endswith(".csv") and same_but_pietra(a / name, b / name):
                    continue
                found.append(f"{config}/{name}: contents differ")
        if not (names_a & names_b):
            found.append(f"{config}: no artifacts")
    return found


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 2)
    else:
        first, second = Path(sys.argv[2]), Path(sys.argv[3])
        x, y = (json.loads((d / "_meta.json").read_text()) for d in (first, second))
        found = differences(first, second, tolerate_pietra=x["_scipy"] != y["_scipy"])
        print("\n".join(found) or "identical")
        print(f"seconds {sum(v for k, v in x.items() if k[0] != '_'):.1f} vs "
              f"{sum(v for k, v in y.items() if k[0] != '_'):.1f}; peak RSS MB {x['_rss_mb']} vs {y['_rss_mb']}",
              file=sys.stderr)
