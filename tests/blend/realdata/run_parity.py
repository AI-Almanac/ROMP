"""Real-data end-to-end training parity: onset_blending vs hcwf on ai-almanac jobs.

    python tests/blend/realdata/run_parity.py [eth|ind ...] [--work DIR] [--skip-train]

For each job (a trained blend from ~/.local/share/ai-almanac-spark/jobs): rewrite the job's
config.yml to the local datasets, train with onset_blending (its own venv) and with hcwf
(`hcwf-run convert`, `hcwf-run -p`), time both (/usr/bin/time: wall s, peak RSS KB), compare every
artifact (CSV, NetCDF, JSON, trained.blend members), forecast both trained blends on the 2024 live
file of every model, compare the forecasts, and compare hcwf's artifacts with the job's stored
output. Skips (exit 0) when the data or either environment is missing. Run under pytest it skips
unless HCWF_REALDATA=1 (it takes minutes and gigabytes).

onset_blending's two outputs (its training and its forecast) are the same for the same inputs, so they are computed
once and reused: they live in the ``oracle`` kind of the shared hcwf cache (``$HCWF_ORACLE_CACHE_DIR``, else
``$HCWF_CACHE_DIR``, default ``~/.cache/hcwf``), keyed by onset_blending's commit (a dirty checkout is never reused),
package source and interpreter environment, the config text, the command line and the path, size and mtime of every
input file (tests/blend/_oracle_cache.py). The report says "reused <key> (computed in N s)" for a reused output.
``HCWF_ORACLE_REFRESH=1`` recomputes them; ``hcwf-run cache clean --kind oracle`` deletes them. hcwf is never
reused: its steps run cold, in a cache folder under --work that is emptied at the start of each run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _oracle_cache  # noqa: E402

DATA = Path(os.environ.get("ALMANAC_DATASETS", "/home/hcwf/code/ai-almanac/datasets"))
JOBS = Path(os.environ.get("HCWF_JOBS_DIR", Path.home() / ".local/share/ai-almanac-spark/jobs"))
BLENDING_ENV = Path.home() / ".local/share/ai-almanac-spark/blending-env"
OB_REPO = Path(os.environ.get("ONSET_BLENDING", "/home/hcwf/code/onset_blending"))
OB_PY = Path(os.environ.get("ONSET_BLENDING_PYTHON", OB_REPO / ".venv/bin/python"))
HCWF_RUN = Path(sys.executable).parent / "hcwf-run"
REGIONS = {"eth": ("273f5a57", "ethiopia"), "ind": ("8e9bfb80", "india")}
TIME = ["/usr/bin/time", "-f", "%e s %M KB"]


def job_dir(prefix):
    hits = sorted(JOBS.glob(prefix + "*")) if JOBS.is_dir() else []
    return hits[0] if hits else None


def run_timed(cmd, log: Path, cwd=None, env=None):
    """Run cmd under /usr/bin/time; return 'wall s, peak KB' (the last stderr line)."""
    p = subprocess.run(TIME + [str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed ({p.returncode}); see {log}")
    return p.stderr.strip().splitlines()[-1]


# ------------------------------------------------- onset_blending's reused outputs

def config_inputs(text: str):
    """Return the input files of a config and whether they are all known: every string value that is a path.

    Parameters
    ----------
    text : str
        The config YAML.

    Returns
    -------
    tuple of (list of str, bool)
        The strings of the config that name an absolute path (existing or not), and False when a string looks like a
        relative data path (``.nc``, ``.csv``, ...), which this function cannot resolve: reuse is then switched off.
    """
    import yaml

    found, known = [], True

    def walk(node):
        """Collect the path strings under a loaded YAML node."""
        nonlocal known
        if isinstance(node, dict):
            node = list(node.values())
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            if node.startswith("/"):
                found.append(node)
            elif node.lower().endswith((".nc", ".csv", ".zarr", ".json", ".npz", ".yml", ".yaml", ".parquet")):
                known = False

    walk(yaml.safe_load(text))
    return found, known


def oracle_run(parts: dict, command_template: list, produce, target: Path, label: str, reusable: bool = True):
    """Return onset_blending's output of a command: the stored one when its key is unchanged, else run it and store it.

    The key (tests/blend/_oracle_cache.py) is the oracle's commit, package source and environment plus ``parts`` and the
    command line with its work-folder paths replaced by names. ``HCWF_ORACLE_REFRESH=1`` recomputes.

    Parameters
    ----------
    parts : dict
        The inputs of the command: config text, identity of every input file, ...
    command_template : list
        The command line with placeholders instead of the paths in the work folder.
    produce : callable
        ``produce(target)`` runs the command and writes its output at ``target``.
    target : pathlib.Path
        Where a fresh output goes.
    label : str
        Names the entry in the line printed to stderr.
    reusable : bool, default True
        False when the inputs of the key are not all known.

    Returns
    -------
    tuple of (pathlib.Path, dict)
        The output (a stored one is read-only) and the info of `_oracle_cache.reuse`.
    """
    return _oracle_cache.reuse({"command": command_template, **parts}, produce, target, src=OB_REPO, python=OB_PY,
                               src_first=False, path0="/tmp", label=label, reusable=reusable)


def oracle_time(ran: dict, info: dict) -> str:
    """Return what to print for the time of an oracle command: this run's, or a note that it was reused."""
    return f"reused {info['fingerprint']} (computed in {info['seconds']} s)" if info["reused"] else ran["timing"]


# ----------------------------------------------------------------- comparison

def cmp_arrays(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        na, nb = np.isnan(a), np.isnan(b)
        if (na != nb).any():
            return f"NaN pattern differs at {int((na != nb).sum())} entries"
        d = np.abs(a[~na].astype(float) - b[~nb].astype(float)) if (~na).any() else np.zeros(0)
        if not d.size or d.max() == 0:
            return None
        rel = d / np.maximum(np.abs(a[~na]).astype(float), 1e-300)
        return f"max abs {d.max():.3g}, max rel {rel.max():.3g}, {int((d > 0).sum())}/{d.size} entries differ"
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        a, b = a.astype(str), b.astype(str)
    bad = int((a != b).sum())
    return None if not bad else f"{bad}/{a.size} entries differ"


def cmp_csv(pa: Path, pb: Path):
    if pa.read_bytes() == pb.read_bytes():
        return None
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    if list(a.columns) != list(b.columns):
        return f"columns {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return f"rows {len(a)} vs {len(b)}"
    out = []
    for c in a.columns:
        r = cmp_arrays(a[c].to_numpy(), b[c].to_numpy())
        if r:
            out.append(f"{c}: {r}")
    return "text differs; " + ("; ".join(out[:6]) + (f" (+{len(out) - 6} more columns)" if len(out) > 6 else "")
                               if out else "values equal (formatting only)")


def cmp_json(pa, pb):
    def walk(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                return [f"{path}: keys {sorted(set(x) ^ set(y))}"]
            return [m for k in x for m in walk(x[k], y[k], f"{path}/{k}")]
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                return [f"{path}: len {len(x)} vs {len(y)}"]
            return [m for i, (u, v) in enumerate(zip(x, y)) for m in walk(u, v, f"{path}[{i}]")]
        if isinstance(x, float) and isinstance(y, float):
            return [] if x == y or (x != x and y != y) else [f"{path}: {x!r} vs {y!r} (diff {abs(x - y):.3g})"]
        return [] if x == y else [f"{path}: {x!r} vs {y!r}"]
    ja, jb = json.loads(Path(pa).read_text()), json.loads(Path(pb).read_text())
    msgs = walk(ja, jb)
    return None if not msgs else f"{len(msgs)} differences; " + "; ".join(msgs[:4])


def cmp_nc(pa, pb):
    out = []
    with xr.open_dataset(pa) as a, xr.open_dataset(pb) as b:
        va, vb = set(a.variables), set(b.variables)
        if va != vb:
            out.append(f"variables only in A {sorted(va - vb)}, only in B {sorted(vb - va)}")
        for v in sorted(va & vb):
            if a[v].dims != b[v].dims:
                out.append(f"{v}: dims {a[v].dims} vs {b[v].dims}")
                continue
            r = cmp_arrays(a[v].values, b[v].values)
            if r:
                out.append(f"{v}: {r}")
    return None if not out else "; ".join(out)


def cmp_npz(pa, pb):
    out = []
    with np.load(pa, allow_pickle=False) as a, np.load(pb, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            out.append(f"arrays only in A {sorted(set(a.files) - set(b.files))}, only in B {sorted(set(b.files) - set(a.files))}")
        for k in sorted(set(a.files) & set(b.files)):
            r = cmp_arrays(a[k], b[k])
            if r:
                out.append(f"{k}: {r}")
    return None if not out else "; ".join(out)


def cmp_file(pa: Path, pb: Path):
    s = pa.suffix
    return {".csv": cmp_csv, ".json": cmp_json, ".nc": cmp_nc, ".npz": cmp_npz}.get(
        s, lambda x, y: None if x.read_bytes() == y.read_bytes() else "bytes differ")(pa, pb)


def unpack(blend: Path, to: Path):
    to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(blend) as z:
        z.extractall(to)
    return to


def compare_dirs(A: Path, B: Path, label, rows, mapping=lambda n: n):
    """Compare same-named files of A and B (B's name via mapping)."""
    for f in sorted(p for p in A.iterdir() if p.is_file()):
        g = B / mapping(f.name)
        if not g.exists():
            rows.append((label, f.name, "only in A"))
            continue
        rows.append((label, f.name, cmp_file(f, g) or "identical"))
    for g in sorted(p for p in B.iterdir() if p.is_file()):
        if not (A / g.name).exists():
            rows.append((label, g.name, "only in B"))


# ----------------------------------------------------------------- the driver

def model_names(blend_cfg: Path):
    import yaml
    cfg = yaml.safe_load(blend_cfg.read_text())
    return {m["name"]: Path(m["files"]["paths"][0]).parent.name for m in cfg["models"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("regions", nargs="*", default=list(REGIONS))
    ap.add_argument("--work", default=os.environ.get("PARITY_WORK", "/tmp/hcwf_realdata_parity"))
    ap.add_argument("--skip-train", action="store_true", help="reuse the outputs already in --work")
    args = ap.parse_args(argv)
    if not OB_PY.exists() or not DATA.is_dir() or not HCWF_RUN.exists():
        print("SKIP: onset_blending env, ai-almanac datasets or hcwf-run missing")
        return 0
    report = []
    for key in args.regions:
        prefix, region = REGIONS[key]
        job = job_dir(prefix)
        if job is None or not (DATA / region).is_dir():
            print(f"SKIP {key}: job or data missing")
            continue
        W = Path(args.work) / key
        W.mkdir(parents=True, exist_ok=True)
        cfgp = W / "config.yml"
        text = (job / "output/config.yml").read_text()
        text = text.replace("/datasets/", f"{DATA}/").replace("/data/blending-env/", f"{BLENDING_ENV}/")
        cfgp.write_text(text)
        rows, times = [], {}
        ob_out, run_yaml = W / "ob_out", W / "run.yaml"
        import shutil
        # hcwf always runs cold: its own cache folder, empty at the start of every run (not the shared one, which
        # holds only onset_blending's reused outputs here)
        hcwf_env = {**os.environ, "HCWF_CACHE_DIR": str(W / "hcwf_cache")}
        train_info = {"fingerprint": None}
        if not args.skip_train:
            for d in (ob_out, W / "runs", W / "hcwf_cache"):
                shutil.rmtree(d, ignore_errors=True)
            ran = {}
            inputs, known = config_inputs(text)

            def produce_train(target):
                """Train onset_blending into `target`."""
                ran["timing"] = run_timed([OB_PY, "-m", "onset_blending", "train", cfgp, "-o", target], W / "ob.log",
                                          cwd="/tmp")

            ob_out, train_info = oracle_run(
                {"config": text, "inputs": _oracle_cache.inputs_identity(inputs)},
                ["-m", "onset_blending", "train", "<config>", "-o", "<out>"], produce_train, ob_out,
                f"{key} train", known)
            times["onset_blending train"] = oracle_time(ran, train_info)
            subprocess.run([HCWF_RUN, "convert", cfgp, "-o", run_yaml, "--force"], check=True, capture_output=True,
                           env=hcwf_env)
            times["hcwf-run train"] = run_timed([HCWF_RUN, "-p", run_yaml, "--quiet"], W / "hcwf.log", cwd=W,
                                                env=hcwf_env)
        blend_dirs = sorted((W / "runs").glob("*/blend"))
        if not blend_dirs:
            raise SystemExit(f"{key}: no hcwf blend folder under {W / 'runs'}")
        mo_out = blend_dirs[0]
        # 1. onset_blending vs hcwf
        compare_dirs(ob_out, mo_out, "ob-vs-hcwf", rows)
        for sub in ("blend",):
            unpack(ob_out / "trained.blend", W / "ob_blend")
            unpack(mo_out / "trained.blend", W / "hcwf_blend")
        compare_dirs(W / "ob_blend", W / "hcwf_blend", "blend-members ob-vs-hcwf", rows)
        # 2. hcwf vs the job's stored outputs
        jo = job / "output"
        for f in sorted(jo.iterdir()):
            if f.name == "trained.blend":
                unpack(f, W / "job_blend")
                continue
            g = mo_out / f.name
            rows.append(("job-vs-hcwf", f.name, (cmp_file(f, g) or "identical") if g.exists() else "only in job"))
        compare_dirs(W / "job_blend", W / "hcwf_blend", "blend-members job-vs-hcwf", rows)
        # 3. forecasts on the 2024 live files
        names = model_names(cfgp)
        fargs = []
        for model, folder in names.items():
            fargs += ["--forecast", f"{model}={DATA / region / 'models' / folder / '2024.nc'}"]
        for sub in ("forecast_ob", "forecast_hcwf"):
            shutil.rmtree(W / sub, ignore_errors=True)
        ran = {}
        live = [Path(a.split("=", 1)[1]) for a in fargs if "=" in a]

        def produce_forecast(target):
            """Forecast the live files with onset_blending's trained blend into `target`."""
            ran["timing"] = run_timed([OB_PY, "-m", "onset_blending", "forecast", ob_out / "trained.blend", *fargs,
                                       "-o", target], W / "ob_fc.log", cwd="/tmp")

        # the forecast of a trained blend is a function of that blend (its train key) and of the live files
        forecast_ob, info = oracle_run(
            {"train": train_info["fingerprint"], "models": sorted(names), "live": _oracle_cache.inputs_identity(live)},
            ["-m", "onset_blending", "forecast", "<trained.blend>", "--forecast", "<model>=<file>", "-o", "<out>"],
            produce_forecast, W / "forecast_ob", f"{key} forecast", train_info["fingerprint"] is not None)
        times["onset_blending forecast"] = oracle_time(ran, info)
        times["hcwf-run forecast"] = run_timed(
            [HCWF_RUN, "forecast", mo_out / "trained.blend", *fargs, "-o", W / "forecast_hcwf"], W / "hcwf_fc.log",
            cwd=W, env=hcwf_env)
        compare_dirs(forecast_ob, W / "forecast_hcwf", "forecast ob-vs-hcwf", rows)
        # forecasts of the job's own trained blend with hcwf (old blend files still load)
        shutil.rmtree(W / "forecast_hcwf_job", ignore_errors=True)
        subprocess.run([HCWF_RUN, "forecast", job / "output/trained.blend", *fargs, "-o", W / "forecast_hcwf_job"],
                       check=True, capture_output=True, cwd=W, env=hcwf_env)
        compare_dirs(forecast_ob, W / "forecast_hcwf_job", "forecast ob(new)-vs-hcwf(job blend)", rows)
        report.append((key, times, rows))
    for key, times, rows in report:
        print(f"\n=== {key}")
        for k, v in times.items():
            print(f"  {k}: {v}")
        for label, name, res in rows:
            print(f"  [{label}] {name}: {res}")
    return 0


def test_realdata_parity():  # pragma: no cover - heavy
    import pytest
    if os.environ.get("HCWF_REALDATA") != "1":
        pytest.skip("set HCWF_REALDATA=1 to run the real-data parity driver")
    main(["eth"])


if __name__ == "__main__":
    raise SystemExit(main())
