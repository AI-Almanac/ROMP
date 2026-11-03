# Combined proposal: restructuring, cleanup and performance

From the four reviews in this folder (layout.md, consistency.md, perf_benchmark.md,
perf_blend.md). Every step keeps results within the agreed tolerance (floats rtol 1e-9,
discrete results exact) unless marked "changes numbers"; each is verified with the
regression slices, the full suite (`pytest -n auto`) and the blend parity tests, and
committed separately. Timings are measured on the second Spark.

## Decisions needed

1. Remove the legacy flat-key onset mode (wet_init/wet_threshold/... keys,
   `EXTEND_END_DAYS = 47`), keeping the translation only inside `hcwf-run convert`.
2. Drop conversion of old `config.in` files (an `exec`-based reader).
3. Retire the case-sensitive `India` box (`legacy_domains`): `region: India` and
   `region: india` would mean the same; the four regression slices and
   `examples/config.yaml` use `India`, so their India results change (the box widens to
   6.5-38.5 N, 66.5-100 E). New regression baseline for those slices.
4. Reliability figure at 300 dpi like every other figure (now 600; halves figure time;
   changes the PNG, not numbers).
5. Solver changes that move probabilities (warm start across CV folds, float32, looser
   tolerance: 0.03-0.19 in fitted probabilities): recommended NO.
6. Delete the pre-rename `~/.cache/momp` (86 MB, unused).
7. GCP resource names in `.github/workflows/docker-publish.yml` (`romp`): keep, or rename
   the resources in GCP first.

## Waves

P0 — quick performance wins (independent of the layout, results identical, size S)
- Worker BLAS threads: the run pool limits numba threads only, so 20 workers run 20
  OpenBLAS threads each; KDE solves 39.6 s -> 0.8 s on India (stats/parallel.py:144-171).
  India training 253 -> 216 s, Ethiopia 31 -> 21 s (measured).
- NetCDF forecast reads: select inits from whole stored time chunks instead of a list
  (HDF5 re-decompresses chunks): 8.4 -> 2.2 s per read, 12-21 s per slice (region.py:217).
- Global CV: final fit first / all fits at once when memory allows: 97.5 -> 74 s.
- Scoring after CV: rank once per model, unit codes instead of a set test over 2 M ids,
  models scored in the pool: 36 -> <10 s.
- Climatology: don't re-read observations when every onset is cached (reader callable).

R1 — bugs-adjacent cleanup and deletions (disjoint files)
- One config error hierarchy (`HCWFConfigError(ValueError)` over RecipeError,
  RegionError, DefinitionError, ...), so the CLI reports them all without tracebacks.
- Break the cli <-> driver import cycle; one CLI (`python -m hcwf.onset` folded in).
- Delete ~22 functions with no caller and the test-only ones (engine.regional, ...);
  stale `__main__` blocks; unused keys (layout, work_dir, pkg_dir, debug, years,
  years_clim, fallback_date); decisions 1-3 if approved.
- Duplicates: five file-year readers -> one; two zarr checks -> one; two rain-unit
  tables -> one; align_sample's own cache and readers -> Store and the one input path
  (also fixes its ignoring of `paths` datasets); the land mask (utils/land_mask.py ->
  utils/mask.py; verify identical cells on every slice grid) — size L, last.
- Climatology: the sibling function that logs and skips unreadable years made strict
  like the one fixed in c11e9cb.

R2 — one config path, typed
- One resolve path (today: loader, api._prepare, export._complete, driver._run; two use
  the loader's privates); validation in one place.
- Typed frozen `RunConfig` (ObsSpec, ModelSpec, Window, OutputSpec with the existing
  OnsetSetup, UnitsSpec, MaskSpec, Recipe, RunPaths) replacing the flat kwargs (147 keys,
  187 `**kwargs` sites, signatures of 31-37 parameters) and the Setting/Case `asdict`
  merges, migrated leaf-first.

R3 — package by concept (scripted moves, file splits)
- config/, runs/, io/ (splitting the 1461-line io/input.py), spatial/ (masks, region
  selection, reading onto units), onset/ (taking over stats/detect), references/,
  metrics/, scoring/ (the app/ flows), figures/, cli/, parallel.py, log.py, errors.py;
  lib/, utils/, stats/, app/, params/ disappear. Move table in layout.md.
- Lazy imports (pandas/xarray/numba/cartopy/matplotlib/regionmask/seaborn at module top
  cost 0.2-0.75 s per import, paid again by every spawned worker).

R4 — per-run state and caching
- One per-run `RunState` and one bounded, registered in-memory memo with a single reset
  (today ~20 module tables reset unevenly; two hcwf.run calls in a notebook can interfere).
- `hcwf-run cache info|clean` on a new `Store.prune`; observed-onset table written once per
  run instead of once per year (quadratic today). No diskcache/xxhash/zarr caching needed
  (measured: caches cost ~0.1 s per run).

P1 — remaining performance on the new structure (results identical)
- Ensemble forecasts read and detected once per year instead of twice (bins.py:529/542,
  benchmark.py:199/214).
- Forecast onsets on units: fast path for one-to-one units (no float64 sparse matmul),
  compact reads (only kept cells), work split by file and run of issues: India forecast
  stage 63 s / 41 GB -> ~30 s / 15 GB; live forecast 24 -> ~8 s.
- Memory: release rain after thresholds and the threshold years' reference rain in the
  blend flow; drop the forecast cube during ingest; skip masked cells (51-62 % of box
  cells); rain as int16 tenths with a missing sentinel (exact mode stays float64); unused
  count variants and lead days; smaller rolling-sum chunks; main-process obs cache.
- Worker count from per-year memory (workers_within exists); dates without string
  round trips; grid-id formatting once per run.

Projected (identical results): benchmark slice India AIFS-ENS 51 -> ~17 s, peak
2.9 -> ~1.2 GB; India blend training 253 -> ~130-140 s, main process 8.9 -> ~4.5-6 GB,
~16 GB across processes; Ethiopia training 31 -> ~17 s.
