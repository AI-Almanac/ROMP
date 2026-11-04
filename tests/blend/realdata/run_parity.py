"""Real-data end-to-end training parity: onset_blending vs hcwf on ai-almanac jobs.

    python tests/blend/realdata/run_parity.py [eth|ind ...] [--work DIR] [--skip-train]

For each job (a trained blend from ~/.local/share/ai-almanac-spark/jobs): rewrite the job's
config.yml to the local datasets, train with onset_blending (its own venv) and with hcwf
(`hcwf-run convert`, `hcwf-run -p`), time both (/usr/bin/time: wall s, peak RSS KB), compare every
artifact (CSV, NetCDF, JSON, trained.blend members), forecast both trained blends on the 2024 live
file of every model, compare the forecasts, and compare hcwf's artifacts with the job's stored
output. Skips (exit 0) when the data or either environment is missing. Run under pytest it skips
unless HCWF_REALDATA=1 (it takes minutes and gigabytes).
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


def run_timed(cmd, log: Path, cwd=None):
    """Run cmd under /usr/bin/time; return 'wall s, peak KB' (the last stderr line)."""
    p = subprocess.run(TIME + [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed ({p.returncode}); see {log}")
    return p.stderr.strip().splitlines()[-1]


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
        if not args.skip_train:
            import shutil
            for d in (ob_out, W / "runs"):
                shutil.rmtree(d, ignore_errors=True)
            times["onset_blending train"] = run_timed(
                [OB_PY, "-m", "onset_blending", "train", cfgp, "-o", ob_out], W / "ob.log", cwd="/tmp")
            subprocess.run([HCWF_RUN, "convert", cfgp, "-o", run_yaml, "--force"], check=True, capture_output=True)
            times["hcwf-run train"] = run_timed([HCWF_RUN, "-p", run_yaml, "--quiet"], W / "hcwf.log", cwd=W)
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
            import shutil
            shutil.rmtree(W / sub, ignore_errors=True)
        times["onset_blending forecast"] = run_timed(
            [OB_PY, "-m", "onset_blending", "forecast", ob_out / "trained.blend", *fargs, "-o", W / "forecast_ob"],
            W / "ob_fc.log", cwd="/tmp")
        times["hcwf-run forecast"] = run_timed(
            [HCWF_RUN, "forecast", mo_out / "trained.blend", *fargs, "-o", W / "forecast_hcwf"], W / "hcwf_fc.log", cwd=W)
        compare_dirs(W / "forecast_ob", W / "forecast_hcwf", "forecast ob-vs-hcwf", rows)
        # forecasts of the job's own trained blend with hcwf (old blend files still load)
        shutil.rmtree(W / "forecast_hcwf_job", ignore_errors=True)
        subprocess.run([HCWF_RUN, "forecast", job / "output/trained.blend", *fargs, "-o", W / "forecast_hcwf_job"],
                       check=True, capture_output=True, cwd=W)
        compare_dirs(W / "forecast_ob", W / "forecast_hcwf_job", "forecast ob(new)-vs-hcwf(job blend)", rows)
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
