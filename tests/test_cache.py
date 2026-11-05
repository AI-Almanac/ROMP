"""The one fingerprint store: value hashing, atomic concurrent writes, miss diagnostics, file identity."""
import logging
import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hcwf import cache
from hcwf.cache import NETCDF, NETCDF_GROUPS, PICKLE, Store, file_identity, part_hash


def test_part_hash_is_by_value_not_pointer():
    """String and object arrays hash by their strings, so equal ids hash alike across separately made arrays."""
    a = np.array(["unit_a", "unit_b"], dtype=object)
    b = np.array(["unit_" + "a", "unit_" + "b"], dtype=object)
    assert part_hash(a) == part_hash(b) == part_hash(np.array(["unit_a", "unit_b"]))
    assert part_hash(a) != part_hash(np.array(["unit_a", "unit_c"], dtype=object))
    da = xr.DataArray(a, dims=["unit"])
    assert part_hash(da) == part_hash(xr.DataArray(b.copy(), dims=["unit"]))
    assert part_hash(da) != part_hash(xr.DataArray(b.copy(), dims=["cell"]))


def test_part_hash_containers_numbers_and_order():
    assert part_hash({"a": 1, "b": [1, 2]}) == part_hash({"b": [1, 2], "a": 1})
    assert part_hash([1, 2]) != part_hash([2, 1])
    assert part_hash(None) != part_hash("None") != part_hash("")
    assert part_hash(1.0) == part_hash(np.float64(1.0))
    assert part_hash(True) != part_hash(1)
    assert part_hash(np.arange(3.0)) != part_hash(np.arange(3.0).reshape(3, 1))
    assert part_hash(Path("x")) == part_hash(Path("x"))


def test_file_identity_of_files_and_zarr_dirs(tmp_path):
    f = tmp_path / "a.nc"
    f.write_bytes(b"12345")
    path, size, mtime = file_identity(f)
    assert (path, size) == (str(f), 5) and mtime == f.stat().st_mtime_ns
    assert file_identity(tmp_path / "missing") == [str(tmp_path / "missing"), None, None]
    store = tmp_path / "obs.zarr"
    (store / "rain").mkdir(parents=True)
    (store / "zarr.json").write_text("{}")
    (store / "rain" / "zarr.json").write_text("{}")
    first = file_identity(store)
    assert first[:2] == [str(store), 2]
    assert file_identity(store) == first
    meta = store / "rain" / "zarr.json"
    os.utime(meta, ns=(meta.stat().st_atime_ns, meta.stat().st_mtime_ns + 10**9))
    assert file_identity(store) != first
    again = file_identity(store)
    (store / "extra").mkdir()
    assert file_identity(store) != again


def test_get_or_compute_computes_once_and_keeps_across_stores(tmp_path):
    calls = []
    parts = {"x": 1, "ids": np.array(["a", "b"], dtype=object)}
    for _ in range(2):
        store = Store("things", tmp_path)   # a new store each time: the second is read from disk
        value = store.get_or_compute(parts, lambda: calls.append(1) or {"v": [1, 2]})
        assert value == {"v": [1, 2]}
    assert len(calls) == 1
    entries = list((tmp_path / "things").iterdir())
    assert len(entries) == 1 and {p.name for p in entries[0].iterdir()} == {"value.pkl", "parts.json"}
    store.get_or_compute({**parts, "x": 2}, lambda: calls.append(1) or 5)
    assert len(calls) == 2


def test_in_memory_store_and_damaged_entry(tmp_path):
    store = Store("m")
    assert store.get({"a": 1}) is None
    store.put({"a": 1}, "v")
    assert store.get({"a": 1}) == "v" and store.folder is None
    disk = Store("d", tmp_path)
    disk.put({"a": 1}, "v")
    (disk.entry({"a": 1}) / "value.pkl").write_bytes(b"not a pickle")
    assert Store("d", tmp_path).get({"a": 1}) is None   # a damaged entry is a miss


def test_codecs_round_trip(tmp_path):
    da = xr.DataArray([[1.0, 2.0]], dims=["lat", "lon"], coords={"lat": [1.0], "lon": [2.0, 3.0]}, name="thr",
                      attrs={"source": "s"})
    for codec, value in ((NETCDF, da), (NETCDF, da.to_dataset(name="thr")), (NETCDF_GROUPS, {"a": da, "b": da + 1}),
                         (PICKLE, {"k": 1})):
        store = Store(f"k{codec.filename}{type(value).__name__}", tmp_path, codec=codec)
        store.put({"p": 1}, value)
        back = Store(store.kind, tmp_path, codec=codec).get({"p": 1})
        if isinstance(value, dict) and codec is NETCDF_GROUPS:
            assert set(back) == {"a", "b"}
            xr.testing.assert_identical(back["b"], value["b"])
        elif isinstance(value, xr.DataArray):
            xr.testing.assert_identical(back, value)
        elif isinstance(value, xr.Dataset):
            xr.testing.assert_identical(back, value)
        else:
            assert back == value


def test_miss_diagnostics_name_the_differing_parts(tmp_path, caplog):
    parts = {"files": "f1", "definition": "d1", "years": [2000, 2001]}
    Store("obs", tmp_path, label="Obs", prefix="obs").put(parts, 1)
    with caplog.at_level(logging.INFO, logger="hcwf.cache"):
        store = Store("obs", tmp_path, label="Obs", prefix="obs")
        assert store.get(parts) == 1
        assert store.get({**parts, "definition": "d2"}) is None
        assert store.get({**parts, "definition": "d2"}) is None   # announced once per fingerprint
        empty = Store("none", tmp_path, label="Empty")
        assert empty.get(parts) is None
    text = caplog.text
    assert text.count("cache miss") == 2 and "cache hit obs-" in text
    assert "differs in: definition" in text and "no earlier entry" in text
    store2 = Store("obs", tmp_path, prefix="obs")
    assert store2.nearest(store2.hash_parts({**parts, "files": "f2", "years": [1]}))[1] == ["files", "years"]


def test_hit_line_describes_the_value(tmp_path, caplog):
    Store("d", tmp_path, label="L").put({"a": 1}, [1, 2, 3])
    with caplog.at_level(logging.INFO, logger="hcwf.cache"):
        Store("d", tmp_path, label="L", describe=lambda v: f"{len(v)} items").get({"a": 1})
    assert "L: cache hit" in caplog.text and "(3 items, " in caplog.text


def _write_many(root, n, seed):
    store = Store("race", root, codec=NETCDF)
    for _ in range(n):
        store.put({"p": 1}, xr.DataArray(np.full((200, 200), float(seed)), dims=["y", "x"], name="v"))
        Store("race", root, codec=NETCDF).get({"p": 1}, quiet=True)


def test_concurrent_processes_never_leave_a_half_file(tmp_path):
    """Several spawned processes write and read one fingerprint; every read is whole and no temp file remains."""
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_write_many, args=(tmp_path, 15, i)) for i in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    store = Store("race", tmp_path, codec=NETCDF)
    value = store.get({"p": 1})
    assert value.shape == (200, 200) and len(np.unique(value.values)) == 1
    names = [p.name for p in store.entry({"p": 1}).iterdir()]
    assert sorted(names) == ["parts.json", "value.nc"]


def test_atomic_write_removes_its_temp_file_on_failure(tmp_path):
    def boom(temp):
        temp.write_text("half")
        raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        cache.atomic_write(tmp_path / "x" / "f.txt", boom)
    assert list((tmp_path / "x").iterdir()) == []


def test_a_read_only_root_does_not_stop_the_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    store = Store("k", blocker)   # the root is a file: the folder cannot be made
    assert store.get_or_compute({"a": 1}, lambda: 7) == 7
    assert store.get({"a": 1}) == 7


def test_blendings_threshold_key_parts_fit_the_store(tmp_path):
    """The parts of onset_blending's threshold key (sections, spec, window, file identities, version) are plain parts."""
    f = tmp_path / "obs.nc"
    f.write_bytes(b"1")
    parts = {"version": 1, "observations": {"path": "obs", "variable": "p"}, "domain": {"units": ["a", "b"]},
             "climatology": {"days": 5, "years": "2001:2008"}, "window": [1, 30],
             "observation_files": [file_identity(f)], "referenced_files": []}
    store = Store("thresholds", tmp_path, codec=NETCDF)
    base = store.fingerprint(parts)
    assert base == store.fingerprint({**parts, "window": [1, 30]})
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert store.fingerprint({**parts, "observation_files": [file_identity(f)]}) != base
    assert store.fingerprint({**parts, "climatology": {"days": 5, "years": "2001:2009"}}) != base


def test_threshold_cache_reuses_and_follows_inputs_and_files(tmp_path, monkeypatch):
    """Port of blending's reuse test: kept thresholds are reused; another spec, years or a touched file recompute."""
    from hcwf.onset import climatology
    from hcwf.onset.climatology import ThresholdCache
    from hcwf.onset.model import ThresholdSpec
    from tests.test_onset_climatology import make_rain

    rain = make_rain(lambda i, t, c: 2.0)
    spec = ThresholdSpec("mean_daily")
    data = tmp_path / "obs.nc"
    data.write_bytes(b"1")
    calls = []
    original = climatology.compute_threshold
    monkeypatch.setattr(climatology, "compute_threshold", lambda *a, **k: calls.append(1) or original(*a, **k))

    def run(years=(2001, 2003), files=None):
        return ThresholdCache(tmp_path / "c", [file_identity(data)] if files is None else files).get_or_compute(
            spec, rain, years, "t")

    first = run()
    xr.testing.assert_identical(run(), first)
    assert len(calls) == 1
    run(years=(2001, 2002))
    assert len(calls) == 2
    os.utime(data, ns=(data.stat().st_atime_ns, data.stat().st_mtime_ns + 10**9))
    run()
    assert len(calls) == 3


def test_a_threshold_cache_key_does_not_load_the_lazy_reference_rain():
    import numpy as np
    from hcwf.io.reference import LazyRain
    from hcwf.onset.climatology import ThresholdCache
    from hcwf.onset.model import DailyRain

    n = 3
    probe = DailyRain(tenths=np.zeros((2, n), np.int32), missing=np.zeros((2, n), bool), dims=("x", "time"),
                      coords={"x": np.arange(2), "time": np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")},
                      year=np.full(n, 2000), month=np.ones(n, int), day=np.arange(1, n + 1), calendar="standard",
                      day_boundary="00 UTC, labelled by start", source="obs")

    def load():
        raise AssertionError("the key must not load the reference years")

    parts = ThresholdCache(None).parts("spec", LazyRain(probe, load), (2000, 2001), "thr")
    assert parts["source"] == "obs" and "resolution" not in parts


def test_store_adds_the_code_version_of_its_kind(monkeypatch):
    """A kind in CODE_MODULES gets its code version as a part, so a changed module makes every older entry miss."""
    parts = {"x": 1}
    store = Store("thresholds")
    before = store.fingerprint(parts)
    monkeypatch.setattr(cache, "_SOURCE_HASHES", {})
    monkeypatch.setattr(cache, "_read_source", lambda name: b"changed " + name.encode())
    assert store.fingerprint(parts) != before
    other = Store("not_versioned")
    assert "code_version" not in other._hashes(parts) and "code_version" in store._hashes(parts)
    assert cache.code_version("not_versioned") is None


def test_every_store_kind_of_hcwf_has_its_code_modules():
    """Each Store kind written in the package is listed, and every module listed exists as a file."""
    import ast

    root = Path(cache.__file__).resolve().parent
    kinds = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        constants = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                     for t in n.targets if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Store" and node.args:
                arg = node.args[0]
                kind = arg.value if isinstance(arg, ast.Constant) else constants.get(getattr(arg, "id", None))
                if isinstance(kind, str):
                    kinds.add(kind)
    assert kinds >= {"observed_onsets", "thresholds", "harmonic_seasons", "model_thresholds", "detect", "alignment"}
    assert kinds - {"oracle"} <= set(cache.CODE_MODULES), kinds - set(cache.CODE_MODULES)
    assert "oracle" in cache.CODE_MODULES   # the oracle's Store is made under tests/blend
    for kind, modules in cache.CODE_MODULES.items():
        assert len(set(modules)) == len(modules), kind
        for name in modules:
            assert name.startswith("hcwf.") and cache._module_path(name).is_file(), (kind, name)


def test_the_code_version_follows_the_source_of_a_listed_module_only(monkeypatch):
    """Changing the source of a listed module changes the version of its kinds; another module changes nothing."""
    monkeypatch.setattr(cache, "_SOURCE_HASHES", {})
    texts = {}
    monkeypatch.setattr(cache, "_read_source", lambda name: texts.get(name, b"same"))
    base = {kind: cache.code_version(kind) for kind in cache.CODE_MODULES}
    monkeypatch.setattr(cache, "_SOURCE_HASHES", {})
    texts["hcwf.onset.engine"] = b"a changed engine"
    changed = {kind: cache.code_version(kind) for kind in cache.CODE_MODULES}
    for kind, modules in cache.CODE_MODULES.items():
        assert (changed[kind] != base[kind]) == ("hcwf.onset.engine" in modules), kind
    monkeypatch.setattr(cache, "_SOURCE_HASHES", {})
    texts["hcwf.onset.engine"] = b"same"
    texts["hcwf.cli.parser"] = b"not computing a result"
    assert {kind: cache.code_version(kind) for kind in cache.CODE_MODULES} == base
    assert changed["detect"] == base["detect"] and changed["observed_onsets"] != base["observed_onsets"]


def test_the_code_version_is_read_from_files_without_importing_the_modules():
    """A module's source is found by path: the version of the real files is stable and imports nothing."""
    import sys

    before = set(sys.modules)
    first = cache.code_version("observed_onsets")
    assert isinstance(first, str) and first == cache.code_version("observed_onsets")
    assert set(sys.modules) == before


# --- listing and pruning the folder ------------------------------------------------------------------------------

def _aged(store, parts, days, value=1):
    """An entry of `store` written `days` ago."""
    import time

    store.put(parts, value)
    when = time.time() - days * 86400
    for path in [store.entry(parts), *store.entry(parts).iterdir()]:
        os.utime(path, (when, when))


def test_entries_list_kind_size_and_age_and_kinds_are_enumerated(tmp_path):
    thresholds, onsets = Store("thresholds", tmp_path), Store("observed_onsets", tmp_path)
    _aged(thresholds, {"a": 1}, days=40, value=np.zeros(1000))
    _aged(thresholds, {"a": 2}, days=1)
    onsets.put({"a": 1}, "x")
    assert Store.kinds(tmp_path) == ["observed_onsets", "thresholds"] and Store.kinds(tmp_path / "none") == []
    entries = thresholds.entries()
    assert [e.kind for e in entries] == ["thresholds"] * 2 and entries[0].mtime < entries[1].mtime
    assert entries[0].size > 8000 > entries[1].size and entries[0].path == thresholds.entry({"a": 1})
    assert [s.kind for s in cache.store_kinds(tmp_path, ["thresholds"])] == ["thresholds"]
    assert Store("x", None).entries() == []


def test_prune_by_age_dry_run_and_all(tmp_path):
    store = Store("thresholds", tmp_path)
    _aged(store, {"a": 1}, days=40)
    _aged(store, {"a": 2}, days=1)
    assert [e.name for e in store.prune(older_than=cache.parse_duration("30d"), dry_run=True)] == [store.fingerprint({"a": 1})]
    assert len(store.entries()) == 2                      # a dry run deletes nothing
    gone = store.prune(older_than=cache.parse_duration("30d"))
    assert [e.name for e in gone] == [store.fingerprint({"a": 1})]
    assert [e.name for e in store.entries()] == [store.fingerprint({"a": 2})]
    assert store.get({"a": 1}, quiet=True) is None        # the in-memory copy went with it
    assert len(store.prune()) == 1 and store.entries() == []


def test_parse_duration():
    assert cache.parse_duration("90m") == 5400 and cache.parse_duration("2w") == 14 * 86400
    assert cache.parse_duration("30") == 30 * 86400 and cache.parse_duration("12H") == 43200
    with pytest.raises(ValueError, match="not a duration"):
        cache.parse_duration("soon")


def _cli(args, monkeypatch, capsys):
    from hcwf.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["cache", *args])
    return stop.value.code, capsys.readouterr()


def test_cache_info_and_clean_through_the_command_line(tmp_path, monkeypatch, capsys):
    root = tmp_path / "shared"
    _aged(Store("thresholds", root), {"a": 1}, days=40)
    _aged(Store("thresholds", root), {"a": 2}, days=1)
    _aged(Store("detect", root), {"a": 1}, days=40)
    from hcwf.cli import main

    main(["cache", "info", "--dir", str(root)])
    info = capsys.readouterr().out
    assert re.search(r"detect\s+1 entries", info) and re.search(r"thresholds\s+2 entries.*oldest 40d.*newest 1d", info)
    main(["cache", "clean", "--dir", str(root), "--older-than", "30d", "--dry-run"])
    assert "would delete 1 entries" in capsys.readouterr().out and len(Store("thresholds", root).entries()) == 2
    main(["cache", "clean", "--dir", str(root), "--kind", "detect", "--all"])
    assert "deleted 1 entries" in capsys.readouterr().out and Store("detect", root).entries() == []
    main(["cache", "clean", "--dir", str(root), "--older-than", "30d"])
    assert [e.name for e in Store("thresholds", root).entries()] == [Store("thresholds", root).fingerprint({"a": 2})]
    code, out = _cli(["clean", "--dir", str(root)], monkeypatch, capsys)
    assert code == 2 and "clean needs --all" in out.err and len(Store("thresholds", root).entries()) == 1
    main(["cache", "clean", "--dir", str(root), "--all"])
    assert Store("thresholds", root).entries() == []


def test_cache_info_defaults_to_the_run_folder_cache_and_the_shared_cache(tmp_path, monkeypatch, capsys):
    shared = tmp_path / "shared"
    monkeypatch.setenv("HCWF_CACHE_DIR", str(shared))
    monkeypatch.chdir(tmp_path)
    Store("alignment", shared).put({"a": 1}, "x")
    Store("observed_onsets", tmp_path / "runs" / ".cache").put({"a": 1}, "x")
    from hcwf.cli import main

    main(["cache", "info"])
    out = capsys.readouterr().out
    assert str(tmp_path / "runs" / ".cache") in out and str(shared) in out and "alignment" in out and "observed_onsets" in out
