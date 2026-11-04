"""Differential check: onset_blending's forecast against hcwf's, on the same trained blends and live files.

    python tests/blend/differential_forecast.py run WORKDIR [ORACLE_PYTHON]          # all steps, then compare

Steps (each its own process; the oracle's in onset_blending's environment when ORACLE_PYTHON exists, else this one):

    data      the synthetic observations, forecasts and weights of tests/blend/differential_training.py and, from
              them, live forecast files: one season replayed (a held-out year), one issue date of it, one single
              initialisation (the default issue date), a file holding more (members, lead days, a larger domain),
              and files holding less or other (fewer members, shorter leads, another grid, no declared units,
              another issue hour)
    oracle    onset_blending trains every config, saves its trained.blend, and forecasts every live scenario
    old       hcwf loads that trained.blend (onset_blending's format) and forecasts the same scenarios
    new       hcwf trains the same configs, saves its own trained.blend, loads it and forecasts them
    compare   every forecast file byte for byte (CSV; the NetCDF of hcwf's must agree with its CSV), every refusal
              with the same reasons, in the three results

The oracle step is reused from the oracle cache (tests/blend/_oracle_cache.py: keyed by onset_blending's commit,
environment and this script; ``HCWF_ORACLE_REFRESH=1`` recomputes); the hcwf steps always run.

Each scenario is a forecast directory, or `refused.json` with the reasons the forecast was refused for.
tests/blend/test_blend_forecast_differential.py runs this when the onset_blending checkout exists.
"""
import json
import resource
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import _oracle_cache  # noqa: E402
import differential_training as training  # noqa: E402

ORACLE_SRC = "/home/hcwf/code/onset_blending"  # the checkout whose commit keys the stored oracle output
LIVE_YEAR = 2013
SCENARIOS = {
    # name: (files by model, issue date, fewer members)
    "season": ("2013.nc", None, False),
    "issue": ("2013.nc", "2013-06-05", False),
    "latest": ("latest.nc", None, False),
    "bigger": ("bigger.nc", None, False),
    "fewer_members_refused": ("fewer.nc", None, False),
    "fewer_members_allowed": ("fewer.nc", None, True),
    "short_leads_refused": ("short.nc", None, False),
    "other_grid_refused": ("grid.nc", None, False),
    "no_units_refused": ("nounits.nc", None, False),
    "other_issue_hour_refused": ("hour.nc", None, False),
    "outside_season_refused": ("2013.nc", "2013-09-01", False),
}
REFUSED_ONLY = ("fewer_members_refused", "short_leads_refused", "other_grid_refused", "no_units_refused",
                "other_issue_hour_refused", "outside_season_refused")


def write_live(root: Path) -> None:
    """The live files, made from the replayed year's file."""
    live = root / "live"
    live.mkdir()
    source = root / "ens" / f"{LIVE_YEAR}.nc"
    with xr.open_dataset(source) as opened:
        ds = opened.load()
    shutil.copy(source, live / f"{LIVE_YEAR}.nc")
    ds.isel(time=[6]).to_netcdf(live / "latest.nc")
    rng = np.random.default_rng(3)
    bigger = ds.reindex(number=np.arange(5), day=np.arange(1, 51), lat=[9.75, 10.0, 10.25, 10.5, 10.75],
                        lon=[69.75, 70.0, 70.25, 70.5], fill_value=0.0)
    inside = (bigger["number"] < 3) & (bigger["day"] <= 40) & bigger["lat"].isin(ds["lat"]) & bigger["lon"].isin(ds["lon"])
    extra = xr.DataArray(rng.uniform(0, 4, bigger["tp"].shape).astype("float32"), dims=bigger["tp"].dims)
    attrs = ds["tp"].attrs
    bigger["tp"] = bigger["tp"].where(inside, extra).astype("float32")
    bigger["tp"].attrs = attrs
    bigger.to_netcdf(live / "bigger.nc")
    ds.isel(number=[0, 1]).to_netcdf(live / "fewer.nc")
    ds.isel(day=slice(0, 20)).to_netcdf(live / "short.nc")
    ds.assign_coords(lat=[10.0, 10.5, 11.0]).to_netcdf(live / "grid.nc")
    bare = ds.copy(deep=True)
    bare["tp"].attrs = {}
    bare.to_netcdf(live / "nounits.nc")
    ds.assign_coords(time=pd.to_datetime(ds["time"].values) + pd.Timedelta(hours=12)).to_netcdf(live / "hour.nc")


def write_data(root: Path) -> None:
    training.write_data(root)
    write_live(root)


def scenarios(root: Path, model: str = "ens") -> dict:
    return {name: ({model: str(root / "live" / file)}, issue, fewer) for name, (file, issue, fewer)
            in SCENARIOS.items()}


# ---------------------------------------------------------------------------------------------------------------
# One step per environment
# ---------------------------------------------------------------------------------------------------------------


def forecast_all(blend, root: Path, out: Path, refused_type, meta: dict, label: str) -> None:
    """Every scenario of the live files with `blend`, into out/<scenario>."""
    for name, (files, issue, fewer) in scenarios(root).items():
        target = out / name
        started = time.time()
        try:
            forecast = blend.forecast(files, issue_date=issue, fewer_members=fewer)
            forecast.save(target)
        except refused_type as exc:
            target.mkdir(parents=True)
            (target / "refused.json").write_text(json.dumps(exc.problems, indent=1))
        meta[f"{label}.{name}"] = round(time.time() - started, 3)


def run_oracle(root: Path, out: Path) -> None:
    sys.path.insert(0, ORACLE_SRC)
    from onset_blending import BlendConfig
    from onset_blending.forecast import IncompatibleForecast
    from onset_blending.trained import TrainedBlend
    from onset_blending.training import Resources, train

    meta: dict = {}
    for name, config in training.configs(root).items():
        started = time.time()
        result = train(BlendConfig.from_dict(config), resources=Resources(workers=1, budget_bytes=None))
        folder = out / name
        folder.mkdir(parents=True)
        result.blend.save(folder / "trained.blend")
        meta[f"train.{name}"] = round(time.time() - started, 2)
        forecast_all(TrainedBlend.load(folder / "trained.blend"), root, folder / "forecasts", IncompatibleForecast,
                     meta, name)
    finish(out, meta)


def run_hcwf_old(root: Path, oracle: Path, out: Path) -> None:
    from hcwf.blend import TrainedBlend
    from hcwf.errors import IncompatibleForecast

    meta: dict = {}
    for name in training.configs(root):
        folder = out / name
        folder.mkdir(parents=True)
        forecast_all(TrainedBlend.load(oracle / name / "trained.blend"), root, folder / "forecasts",
                     IncompatibleForecast, meta, name)
    finish(out, meta)


def run_hcwf_new(root: Path, out: Path) -> None:
    from _recipes import converted

    from hcwf.blend import TrainedBlend
    from hcwf.errors import IncompatibleForecast
    from hcwf.blend.training import train

    meta: dict = {}
    for name, config in training.configs(root).items():
        started = time.time()
        loaded = converted(config)
        result = train(loaded.rc(parallel_workers=1))
        folder = out / name
        result.save(folder / "training")
        meta[f"train.{name}"] = round(time.time() - started, 2)
        forecast_all(TrainedBlend.load(folder / "training" / "trained.blend"), root, folder / "forecasts",
                     IncompatibleForecast, meta, name)
    finish(out, meta)


def finish(out: Path, meta: dict) -> None:
    meta["_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (out / "_meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------------


def scenario_differences(first: Path, second: Path) -> list[str]:
    """What differs between two scenario folders: a forecast's CSV files byte for byte, a refusal's reasons."""
    names_a, names_b = {p.name for p in first.iterdir()}, {p.name for p in second.iterdir()}
    csv_a, csv_b = {n for n in names_a if n.endswith((".csv", ".json"))}, {n for n in names_b if n.endswith((".csv", ".json"))}
    if csv_a != csv_b:
        return [f"files {sorted(csv_a ^ csv_b)} only in one"]
    return [f"{n}: contents differ" for n in sorted(csv_a) if (first / n).read_bytes() != (second / n).read_bytes()]


def netcdf_matches_csv(folder: Path) -> list[str]:
    """hcwf's NetCDF holds the CSV's probabilities."""
    problems = []
    for csv, nc in (("blended_forecast_probabilities.csv", "blended_forecast_probabilities.nc"),
                    ("daily_onset_probabilities.csv", "daily_onset_probabilities.nc")):
        if not (folder / csv).exists():
            continue
        table = pd.read_csv(folder / csv, float_precision="round_trip")
        with xr.open_dataset(folder / nc) as ds:
            ds = ds.load()
        stamp = pd.to_datetime(table["time"]).to_numpy()
        for column in [c for c in table.columns if c.startswith(("cv_", "clim_", "p_later"))]:
            got = ds[column].sel(time=xr.DataArray(stamp, dims="row"), id=xr.DataArray(table["id"].astype(str), dims="row"))
            if not np.array_equal(got.values, table[column].to_numpy(), equal_nan=True):
                problems.append(f"{folder.name}/{nc}: {column} differs from the CSV")
    return problems


def compare(oracle: Path, old: Path, new: Path) -> list[str]:
    found = []
    for config in sorted(p.name for p in oracle.iterdir() if p.is_dir()):
        for scenario in SCENARIOS:
            reference = oracle / config / "forecasts" / scenario
            for label, other in (("hcwf from onset_blending's file", old), ("hcwf's own", new)):
                folder = other / config / "forecasts" / scenario
                found += [f"{config}/{scenario} ({label}): {p}" for p in scenario_differences(reference, folder)]
            if scenario in REFUSED_ONLY and not (reference / "refused.json").exists():
                found.append(f"{config}/{scenario}: onset_blending did not refuse it")
            if scenario not in REFUSED_ONLY and (reference / "refused.json").exists():
                found.append(f"{config}/{scenario}: onset_blending refused it: {(reference / 'refused.json').read_text()}")
            if (new / config / "forecasts" / scenario / "blended_forecast_probabilities.nc").exists():
                found += netcdf_matches_csv(new / config / "forecasts" / scenario)
    return found


def run_all(work: Path, oracle_python: str | None) -> list[str]:
    python = oracle_python if oracle_python and Path(oracle_python).exists() else sys.executable
    script = str(Path(__file__).resolve())
    work.mkdir(parents=True, exist_ok=True)
    root = work / "data"
    root.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_data(root)

    def run_step(interpreter, *arguments):
        """Run one step of this script in a process and raise with its stderr on a bad exit."""
        done = subprocess.run([interpreter, script, *arguments], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"step {arguments[0]} failed:\n{done.stderr[-4000:]}")

    def produce(target):
        """Run onset_blending's step into a new folder."""
        target.mkdir(parents=True)
        run_step(python, "oracle", str(root), str(target))

    # onset_blending's step is the one reused (tests/blend/_oracle_cache.py); the hcwf steps always run, cold
    oracle, _ = _oracle_cache.reuse(
        {"driver": _oracle_cache.driver_hash(script), "script": "differential_forecast.py oracle"}, produce,
        work / "oracle", src=ORACLE_SRC, python=python, src_first=True, path0=Path(script).parent,
        label="differential_forecast oracle")
    (work / "oracle_used").write_text(str(oracle))
    for arguments in (("old", str(root), str(oracle), str(work / "old")), ("new", str(root), str(work / "new"))):
        Path(arguments[-1]).mkdir(parents=True)
        run_step(sys.executable, *arguments)
    return compare(oracle, work / "old", work / "new")


if __name__ == "__main__":
    command = sys.argv[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if command == "oracle":
            run_oracle(Path(sys.argv[2]), Path(sys.argv[3]))
        elif command == "old":
            run_hcwf_old(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        elif command == "new":
            run_hcwf_new(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            problems = run_all(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
            print("\n".join(problems) or "identical")
            timings = {}
            for label in ("oracle", "old", "new"):
                folder = (Path((Path(sys.argv[2]) / "oracle_used").read_text()) if label == "oracle"
                          else Path(sys.argv[2]) / label)
                meta = json.loads((folder / "_meta.json").read_text())
                timings[label] = (sum(v for k, v in meta.items() if k[0] != "_" and ".season" in k),
                                  meta["_rss_mb"])
            print("forecast seconds (season scenario, all configs) and peak RSS MB per step: "
                  + json.dumps(timings), file=sys.stderr)
