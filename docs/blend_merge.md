# Merging onset_blending into ROMP

Status: the merge is complete. This file is the historical plan and uses the old names (`momp`, ROMP, `momp-run`); for the current package see README.md, docs/blending.md and docs/migrating_from_onset_blending.md.

Written 2026-11-02 on branch `monorepo`.

onset_blending (`/home/hcwf/code/onset_blending`, pinned oracle commit `1e6d696`) moves into
`momp`. Afterwards ROMP trains, cross-validates and applies onset blends, and benchmarks and blends
on grid cells or on political boundaries (admin units), for any ROMP region, all from one run YAML
and one onset-definition system (`momp/onset`). The onset_blending repo is then retired.

Overlap analyses (verdict per concept, file:line): scratchpad `overlap_io_domain.md`,
`overlap_onset.md`, `overlap_eval_config.md` (copied to `docs/blend_merge/` for the agents).

## Decisions (user, 2026-11-02)

1. **Exact parity with onset_blending.** Every blending option keeps its exact numbers, through
   options on ROMP's single implementation (never a second implementation). ROMP's own defaults and
   outputs stay byte-identical.
2. **Regions:** blending and benchmarking work for any ROMP region or country, one region per run
   file; a batch command runs several run files.
3. **Compatibility:** `momp-run convert` turns BlendConfig YAML and legacy spec sets into ROMP run
   YAML; existing `trained.blend` files still load and forecast. ai-almanac's output contract
   (docs/pipeline-redesign.md in onset_blending) is preserved.

## Tolerance (user, 2026-11-03)

Results only need to agree within a tolerance, with no procedural errors: floats (scores,
probabilities, thresholds, coefficients) to a relative tolerance of about 1e-9; discrete
results (onset days, statuses, counts, unit ids, scored cells) exactly; procedures (onset
rule, horizon, missing handling, thresholds, CV folds, formulas) unchanged. Code that only
reproduces onset_blending's last-bit rounding may be removed; regression and parity checks
compare with this tolerance. This supersedes "byte-identical" and "bit-identical" below.

## Rules for every change

- **One implementation per concept.** When a blending function has a ROMP counterpart, extend
  ROMP's and delete the blending one; never copy both. Where ROMP itself has two (e.g. the mean
  and median climatological onset, three dimension-alias tables, two `*_climatology` metric
  copies), collapse them as part of the same change.
- **Parity options are options of the one implementation**, e.g. `rain_resolution: exact`,
  `missing: keep`, `horizon: truncate`; defaults are ROMP's current behaviour.
- **ROMP outputs byte-identical** under default settings (existing tests, `data/phase7/baseline`).
- **Blending numbers identical** under the parity options: blending's oracles (`tests/_legacy`,
  its 191 tests) move to `tests/blend/` and run against momp; end-to-end, momp's results are
  compared artifact by artifact with onset_blending@1e6d696 on the same config and data.
- Memory-efficient by design (streaming, no needless copies); report peak RSS for real-data runs.

## Target layout

```
momp/
  spatial/      units.py (EvaluationUnits: grid | polygon | region, subsets such as
                dissemination), regrid.py (SparseTransform, observed_mean, weights from shapefiles
                and CSV, conservative grid-to-grid), ids.py (string unit ids at I/O edges)
  cache.py      one fingerprint store (observed onsets, thresholds, seasons, weights, datasets)
  onset/        + parity options: rain_resolution, missing policy, horizon, dry_day, dry-run
                placement; search_start variants; onset_count histogram; Definition.windows()
  metrics/      probabilistic.py (Brier, RPS, AUC, Pietra; per unit, pooled, per year; fair and
                bin-scope options); ROMP's score code becomes adapters
  stats/bins.py one LeadBins (week1..later labels and "Days a-b" labels)
  utils/workers.py  + memory budget, BLAS limits, SharedArrays; stats/parallel.start_items
  blend/        features, design, predictions, logit, random_forest, _crps_tree, mme,
                calibration, evaluation, training, trained, forecast, signature
  lib/convert_specs.py  BlendConfig / spec sets -> run YAML
```

Config: the existing sections absorb domain, observations, models, onset, climatology; new
top-level `units:` (polygon units, dissemination, weights) and `blend:` (years, CV, formulas,
baselines, MME, random forest, rain predictors, fixed_cutoff, reference_onset). CLI: `momp-run`
trains when `blend:` is present; `momp-run forecast trained.blend --forecast MODEL=FILE`;
`momp-run convert` accepts BlendConfig and spec sets; `--check` covers the blend.

## Waves (one Sonnet agent each; disjoint files within a wave; verify + commit per agent)

Wave 1 (foundations, parallel)
- W1a spatial units and regridding (`momp/spatial/`), oracle tests from blending test_domain.
- W1b onset engine parity (`momp/onset/` model, plan, definition, engine, ingest, backends).
- W1c infrastructure: `momp/cache.py`, `utils/workers.py`, `stats/parallel.start_items`.
- W1d metrics and bins (`momp/metrics/`, `stats/score.py`, `stats/bins.py`).

Wave 2
- W2a thresholds, KDE climatology issue grid, one reference-onset provider.
- W2b input: multi-file/glob datasets, years filter, unit dims, sparse chunk reads, units table;
  `engine.regional` re-expressed as a one-unit transform.
- W2c blend methods: design, predictions, logit, random_forest, _crps_tree, mme, calibration.
- W2d regions: domain boxes for every region, per-region `units:`, legacy region names, seasons
  crossing 1 January.

Wave 3
- W3a blend-only settings (`Recipe`, alongside ROMP's resolved config) and weekly features. Onset
  detection for blends goes through ROMP's one adapter (`momp/stats/detect.py`, the observed-onset
  cache, `momp/io/input.py`), extended with units, search-start variants, onset counts and
  blending's threshold sources; there is no separate blend onset module.
- W3b evaluation, training.
- W3c trained/forecast/signature, old `trained.blend` loading.

Wave 4
- W4a config schema, CLI, API, run folders, convert (BlendConfig, specs), registry ids for
  blending's definition readings.
- W4b benchmarking on political boundaries: scoring, exports, maps on the unit axis
  (see branch `adm3-support` for an earlier attempt and its demo outputs).

Wave 6 (user, 2026-11-03): rename everything to HCWF, clean break: package `momp` -> `hcwf`,
project ROMP -> HCWF, `momp-run` -> `hcwf-run`, `MOMP_*` -> `HCWF_*`, cache dirs, format id
`momp/trained-blend` -> `hcwf/trained-blend` (files saved under the old id still load); no aliases.

Wave 5: end-to-end parity runs (synthetic + real data, Ethiopia and India), docs, dependencies,
retire onset_blending.

Wave 7 (user, 2026-11-03; reordered: before any performance work), after the rename:
read-only review agents in parallel (package layout by concept; leftovers of the original
implementation such as flat keys under sections, the Setting dataclass, config.in, the
case-sensitive India box, print-style output, per-app modules; consistency and dead code;
a fresh duplication sweep). Each also proposes speed and memory improvements for its area,
including data operations and datatypes (smallest dtype that gives identical results, object/string
arrays to integer positions, silent copies from astype, vectorised numpy/numba instead of Python loops
and pandas groupby/apply on hot paths, in-place ops and views, memory layout for the kernels, xarray
overhead in inner loops, chunk-aligned reads, sparse unit weights, no frame-to-array round trips;
changes that would alter numbers are flagged as trade-offs for the user, not applied), and caching (size limits and eviction for the disk caches, e.g. via diskcache; zarr for large
cached arrays; faster hashing (xxhash/blake3) only if profiles show hashing cost; one bounded mechanism
for the scattered in-memory memo tables; keep momp.cache's fingerprints and miss diagnostics), backed by targeted profiles of the real-data workloads (phase7 slices on NetCDF and zarr,
Ethiopia and India training and forecasts, an admin-unit benchmark). One combined proposal
(restructuring + measured hot spots with expected gains) goes to the user for approval.

Wave 8: implementation — restructuring first, then performance fixes on the new structure
(py-spy, memray, cProfile across worker processes; before/after numbers for wall time, peak
RSS and CPU use on the fixed benchmark set; memory-efficient by design, not caps). Each
change verified (byte-identical ROMP outputs, blending parity, full suite) and committed
separately.
