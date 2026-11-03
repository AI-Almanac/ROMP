"""Per-run state: one registered memo mechanism, one reset, and runs in one process that do not touch each other.

The memo registry (`hcwf.cache.memo`), the run scope that clears it at the start and end of every run entry, the
snapshot the workers get, the guard against module-level caches outside the registry, and the observed-onset table
being written once per run."""
import ast
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import hcwf
from hcwf import cache
from hcwf.cache import Store, memo, reset_run_state, run_scope
from test_run_experience import CLIM_YEARS, LAT, LON, _obs_rain, raw_config, write_config, write_forecast

ROOT = Path(__file__).resolve().parents[1]


# --- the registry ---------------------------------------------------------------------------------------------

def test_a_memo_is_bounded_and_drops_the_least_recently_used():
    table = memo("test.bounded", maxsize=2)
    table["a"], table["b"] = 1, 2
    assert table.get("a") == 1          # a is now the most recent
    table["c"] = 3
    assert "b" not in table and set(table) == {"a", "c"}
    assert table.get_or_make("d", lambda: 4) == 4 and set(table) == {"c", "d"}


def test_reset_clears_every_memo_but_the_persistent_ones_and_closes_what_they_hold():
    closed = []
    run, fixed = memo("test.run", on_clear=closed.append), memo("test.fixed", persistent=True)
    run["x"], fixed["y"] = "held", "kept"
    reset_run_state()
    assert len(run) == 0 and fixed.get("y") == "kept" and closed == ["held"]
    reset_run_state(everything=True)
    assert len(fixed) == 0


def test_the_run_scope_resets_at_the_start_and_the_end_and_only_the_outermost_does():
    table, ended = memo("test.scope"), []
    table["stale"] = 1
    cache.on_run_end(lambda: ended.append(dict(table.items())))
    try:
        with run_scope():
            assert len(table) == 0                  # what an earlier run left is gone
            table["a"] = 1
            with run_scope():                       # a step inside the run keeps the run's state
                assert table.get("a") == 1
            assert table.get("a") == 1 and not ended
        assert ended[-1] == {"a": 1}                # the end hooks see the state, then it is cleared
        assert len(table) == 0
    finally:
        cache._END_HOOKS.pop()


def test_the_end_of_a_run_clears_the_state_when_it_raises():
    table = memo("test.raises")
    with pytest.raises(RuntimeError):
        with run_scope():
            table["a"] = 1
            raise RuntimeError("boom")
    assert len(table) == 0


def test_shared_memos_travel_to_the_workers_through_the_registry():
    from hcwf import parallel

    shared, private = memo("test.shared", share=True), memo("test.private")
    shared["k"], private["k"] = "v", "v"
    state = parallel.snapshot()
    assert state["memos"]["test.shared"] == {"k": "v"} and "test.private" not in state["memos"]
    reset_run_state()
    parallel.install(state, token=-1)
    assert shared.get("k") == "v" and len(private) == 0
    parallel._installed["token"] = None


def test_the_real_tables_are_registered():
    import hcwf.config.detect, hcwf.config.onset, hcwf.io.forecasts, hcwf.io.observations, hcwf.io.reference  # noqa: F401
    import hcwf.io.stores, hcwf.io.thresholds, hcwf.metrics.bins, hcwf.onset.detect, hcwf.onset.observed_cache  # noqa: F401
    import hcwf.references.climatology, hcwf.scoring.onset_tables, hcwf.spatial.masks  # noqa: F401

    names = set(cache.memos())
    assert {"io.stores.open", "io.observations.years", "io.reference.held", "io.thresholds.loaded", "io.forecasts.reported",
            "config.detect.samples", "config.onset.setups", "references.climatology.onsets", "scoring.onset_tables",
            "metrics.bins.year_text", "onset.detect.alignment_done", "onset.observed_cache.tables",
            "spatial.masks.built", "spatial.masks.records", "spatial.masks.implicit_keep"} <= names
    shared = {name for name, table in cache.memos().items() if table.share}
    assert {"io.thresholds.loaded", "spatial.masks.built", "spatial.masks.records", "spatial.masks.implicit_keep",
            "onset.observed_cache.tables"} <= shared


# --- no cache outside the registry --------------------------------------------------------------------------------

ALLOWED = {
    ("hcwf/log.py", "_state"): "the console level and the worker's year label of the logging set-up, not data",
    ("hcwf/log.py", "_worker"): "the worker's year label of the log records",
    ("hcwf/parallel.py", "_installed"): "which snapshot a worker process holds",
    ("hcwf/parallel.py", "_pool"): "the run's process pool, stopped by shutdown (an end hook)",
    ("hcwf/parallel.py", "_ATTACHED"): "shared-memory blocks a worker has attached, closed by detach",
    ("hcwf/parallel.py", "_BLAS_LIMIT"): "the BLAS thread limiter of this process",
    ("hcwf/onset/backends/gpu.py", "_ROW_CUMSUM"): "a compiled GPU kernel, built on first use",
    ("hcwf/onset/regions.py", "_index"): "the registry files of the package (read-only data of the installation)",
}
"""(file, name) of the module-level state that is not a run table, with the reason."""
MUTATORS = {"clear", "update", "setdefault", "add", "append", "pop", "popitem", "extend", "remove", "discard", "insert"}
CACHE_DECORATORS = {"lru_cache", "cache", "functools.lru_cache", "functools.cache"}


def _module_level_state(path: Path):
    """The (name, line) of module-level containers that the module mutates, `global` assignments and cache decorators."""
    tree = ast.parse(path.read_text())
    containers = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        if value is None:
            continue
        mutable = isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)) or (
            isinstance(value, ast.Call) and ast.unparse(value.func) in
            {"dict", "list", "set", "OrderedDict", "defaultdict", "deque", "Counter", "collections.OrderedDict"})
        if mutable:
            containers |= {t.id for t in targets if isinstance(t, ast.Name)}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            found.update({name: node.lineno for name in node.names})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                if ast.unparse(call) in CACHE_DECORATORS:
                    found[node.name] = node.lineno
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            for target in (node.targets if hasattr(node, "targets") else [node.target]):
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id in containers:
                    found[target.value.id] = node.lineno
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS \
                and isinstance(node.func.value, ast.Name) and node.func.value.id in containers:
            found[node.func.value.id] = node.lineno
    return found


def test_no_module_level_cache_is_left_outside_the_registry():
    """A module-level dict, list or set that the module changes, a `global` assignment or an lru_cache is run state that
    would be reset unevenly: it must be a `hcwf.cache.memo` (constants are never changed; the rest is in ALLOWED)."""
    offenders = []
    for path in sorted((ROOT / "hcwf").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel == "hcwf/cache.py":
            continue
        for name, line in _module_level_state(path).items():
            if (rel, name) not in ALLOWED:
                offenders.append(f"{rel}:{line} {name}")
    assert not offenders, "module-level state outside hcwf.cache.memo: " + ", ".join(offenders)


def test_the_guard_finds_what_it_is_meant_to_find(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(textwrap.dedent('''
        import functools
        _TABLE = {}
        _SEEN = set()
        CONSTANT = {"a": 1}
        _kernel = None
        def f(key):
            global _kernel
            _TABLE[key] = 1
            _SEEN.add(key)
        @functools.lru_cache(maxsize=None)
        def g(): ...
    '''))
    assert set(_module_level_state(bad)) == {"_TABLE", "_SEEN", "_kernel", "g"}


# --- two runs in one process ------------------------------------------------------------------------------------

_FRESH_RUN = """
import pickle, sys
import hcwf
result = hcwf.run(sys.argv[1], quiet=True, parallel=False)
with open(sys.argv[2], "wb") as handle:
    pickle.dump({"summary": result.scores["summary"], "observed": result.observed}, handle)
"""


def _write_obs_zarr(path, shift):
    """The observations as a zarr store (one store for all years); `shift` moves every onset, so a rewrite changes results."""
    frames = []
    for year in CLIM_YEARS:
        time, rain = _obs_rain(year)
        rain = np.roll(rain, shift, axis=0) if shift else rain
        frames.append(xr.Dataset({"precip": (("time", "lat", "lon"), rain, {"units": "mm"})},
                                 coords={"time": time, "lat": LAT, "lon": LON}))
    xr.concat(frames, "time").to_zarr(path, mode="w")


def _scores(result):
    return result.scores["summary"], result.observed


def test_two_runs_in_one_process_give_what_fresh_processes_give(tmp_path):
    """The same config twice with the observations rewritten in place between the runs (so every table keyed by a path or
    by the config's text could collide), then a fresh process on the final data: all three agree where they should."""
    obs, base = tmp_path / "obs.zarr", tmp_path
    _write_obs_zarr(obs, shift=0)
    write_forecast(base / "DET")
    raw = raw_config(base)
    raw["observations"]["path"] = str(obs)
    config = write_config(base, raw)

    first = hcwf.run(config, quiet=True, parallel=False)
    _write_obs_zarr(obs, shift=9)
    second = hcwf.run(config, quiet=True, parallel=False)
    fresh_file = base / "fresh.pkl"
    subprocess.run([sys.executable, "-c", _FRESH_RUN, str(config), str(fresh_file)], check=True, cwd=base)
    fresh = pickle.loads(fresh_file.read_bytes())

    assert not np.array_equal(first.observed["onset_date"].values, second.observed["onset_date"].values)
    xr.testing.assert_equal(second.observed["onset_date"], fresh["observed"]["onset_date"])
    pd.testing.assert_frame_equal(second.scores["summary"], fresh["summary"], rtol=1e-9)
    assert len(cache.memos()["io.stores.open"]) == 0 and len(cache.memos()["io.thresholds.loaded"]) == 0


def test_a_run_leaves_no_state_behind_and_a_failed_run_too(tmp_path):
    from test_run_experience import write_obs

    write_obs(tmp_path / "obs")
    write_forecast(tmp_path / "DET")
    hcwf.run(raw_config(tmp_path), quiet=True, parallel=False)
    assert all(len(table) == 0 for table in cache.memos().values() if not table.persistent)
    broken = raw_config(tmp_path)
    broken["observations"]["path"] = str(tmp_path / "nowhere")
    with pytest.raises(Exception):
        hcwf.run(broken, quiet=True, parallel=False)
    assert all(len(table) == 0 for table in cache.memos().values() if not table.persistent)


# --- the observed-onset table is written once per run ----------------------------------------------------------

def _count_puts(monkeypatch):
    puts = []
    real = Store.put

    def counting(self, parts, value):
        if self.kind == "observed_onsets":
            puts.append(len(value["years"]))
        return real(self, parts, value)

    monkeypatch.setattr(Store, "put", counting)
    return puts


@pytest.mark.parametrize("parallel", [False, True])
def test_a_multi_year_run_writes_the_observed_onset_table_once_and_a_warm_run_hits(tmp_path, monkeypatch, capsys, parallel):
    from test_run_experience import write_obs

    write_obs(tmp_path / "obs")
    write_forecast(tmp_path / "DET")
    config = write_config(tmp_path, raw_config(tmp_path))
    puts = _count_puts(monkeypatch)

    hcwf.run(config, write=True, quiet=True, parallel=parallel, **({"parallel_workers": 2} if parallel else {}))
    assert puts == [len(CLIM_YEARS)]                      # once, holding every year
    entries = list((tmp_path / "runs" / ".cache" / "observed_onsets").glob("obs-*/onsets.nc"))
    assert len(entries) == 1
    with xr.open_dataset(entries[0]) as ds:
        assert list(ds["year"].values) == list(CLIM_YEARS) and "onset_date" in ds

    puts.clear()
    capsys.readouterr()
    hcwf.run(config, write=True, overwrite=True, parallel=False)
    assert re.search(rf"Observed onsets: cache hit obs-[0-9a-f]{{12}} \({len(CLIM_YEARS)} years cached", capsys.readouterr().out)
    assert puts == []                                     # a hit writes nothing


def test_a_worker_that_has_not_imported_a_module_still_gets_its_shared_memo():
    code = ("import sys; from hcwf import cache; assert 'hcwf.io.thresholds' not in sys.modules; "
            "cache.install_shared_state({'io.thresholds.loaded': {'k': 1}}); "
            "from hcwf.io import thresholds; assert thresholds._loaded_thresholds.get('k') == 1")
    subprocess.run([sys.executable, "-c", code], check=True)
