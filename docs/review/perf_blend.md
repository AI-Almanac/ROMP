# Performance review: blend training and forecasting

Read-only review with profiling, at commit 6194117 (branch `monorepo`). The numbers were measured on the second
machine (20 aarch64 cores, Neoverse-V2, 121 GB, OpenBLAS 0.3.31, sklearn 1.9.1, numba 0.68.1), which was otherwise
idle. Workloads: the real ai-almanac jobs, Ethiopia "Kiremt onset blend" (273f5a57) and India "Indian Monsoon
blend" (8e9bfb80). Their configs were converted with `hcwf-run convert`, trained with `hcwf-run -p`, and forecast
on the 2024 live files. onset_blending@1e6d696 ran in its own `uv sync --frozen` venv (the venv on the machine
lacked numba).

Tools: a process-tree sampler (every 0.25 s: summed PSS, largest RSS and CPU time of all processes, plus
timestamped log lines), `py-spy --subprocesses`, memray (main process, high-water-mark records), cProfile on a
stage-by-stage in-process driver, and a CV harness that replays the global CV on the dumped design matrix. The
scripts and raw outputs are on the remote machine under `~/perf_blend/` (`mon.py`, `stages.py`, `cvexp.py`,
`onefile.py`, `agg.py`, `mem.py`, `cmp.py`, `*.log`, `*.mem`, `*.pyspy.txt`, `ind.memray`).

Rule used for "risk": **identical** means the artifacts compare equal byte for byte (verified where it says
"verified"). **Within tolerance** means only float summation order changes (rtol 1e-9 is easily met). **Changes
numbers** means a user trade-off.

## 1. Baselines

Every run here starts with a cold threshold cache (the converted run keeps it in `runs/.cache`, which a new job
folder does not have) and a warm OS page cache. The first India run on a cold page cache took **344 s**. Reading
the 41 GB AIFS_ENS archive from disk accounts for the extra ~85 s.

Columns: wall time in seconds; **cores** is the mean number of busy cores (CPU-seconds over wall); **tree PSS** is
the peak summed PSS of the main process and its workers, the real memory footprint (in GB); **max RSS** is the
largest single process (what `/usr/bin/time %M` reports).

### India (hcwf train: 253 s, 3176 CPU-s, max RSS 10.2 GB, tree PSS 40.9 GB; onset_blending: 220 s, 2795 CPU-s, 6.2 GB, 15.6 GB)

| stage | hcwf wall | cores | tree PSS | max RSS | ob wall | cores | tree PSS | max RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thresholds + observed onsets | 15.6 | 8.9 | 12.5 | 3.6 | 8.8 | 15.9 | 5.2 | 0.7 |
| aifsv2 forecast onsets (26 files) | 4.4 | 13.9 | 20.5 | 6.1 | 4.6 | 13.9 | 6.2 | 2.1 |
| aifs_ensv2 forecast onsets (26 files, 41 GB zlib) | 63.4 | 15.0 | **40.9** | 8.1 | 40.5 | 14.7 | 15.6 | 4.7 |
| climatology (4948 units) | **39.6** | 19.7 | 15.8 | 8.1 | 40.2 | 19.8 | 5.9 | 4.0 |
| weekly features (2.14 M samples) | 4.2 | 1.0 | 17.1 | 9.1 | 4.0 | 1.0 | 6.3 | 6.2 |
| evaluation + final fit | 122.4 | 9.4 | 15.5 | 6.6 | 120.9 | 9.9 | 11.6 | 3.5 |
|   of which global CV, 27 tasks on 14 workers (cProfile) | 95.4 | ~14 | | | | | | |
|   of which serial scoring in main (cProfile) | ~36 | 1 | | | | | | |
| save, exit | 2.8 | | 8.4 | 4.9 | 1.2 | | 1.7 | 1.7 |

### Ethiopia (hcwf train: 31.0 s, 464 CPU-s, 2.1 GB, tree PSS 8.7 GB; onset_blending: 28.7 s, 419 CPU-s, 1.65 GB, 4.1 GB)

| stage | hcwf wall | cores | tree PSS | ob wall | cores | tree PSS |
|---|---:|---:|---:|---:|---:|---:|
| thresholds (constant 20 mm) + observed | 2.7 | 14.8 | 4.7 | 2.4 | 14.8 | 2.4 |
| aifsv2 / graphcast forecast onsets | 2.8 / 0.7 | 16.6 | 8.7 | 2.3 / 2.2 | 16.6 | 3.8 |
| climatology (1082 units) | **9.7** | 19.5 | 8.4 | 8.1 | 18.9 | 1.6 |
| weekly features (283 k samples) | 0.6 | 1.0 | 4.8 | 0.5 | 0.8 | 1.5 |
| evaluation + final fit (19 tasks, ~3-5 s each) | 13.1 | 13.3 | 5.4 | 12.5 | 12.5 | 4.1 |

### Forecast on the 2024 live files

| job | hcwf | onset_blending |
|---|---|---|
| India | 24.3 s, 80 CPU-s, 2.4 GB (2.5 GB max RSS) | 57.7 s, 824 CPU-s, 2.7 GB tree |
| Ethiopia | 2.4 s, 14 CPU-s, 0.7 GB | 9.8 s, 164 CPU-s, 0.8 GB |

hcwf's forecast is already faster than onset_blending's. Of its 24 s on India: about 60% is the ensemble file
handled by one worker, 4 s is the climatology run serially, and about 1.5 s is CSV writing. The "known" 77 s
forecast figure no longer holds at this commit.

## 2. Ranked hot spots

### H1. Every worker of the run's pool starts 20 OpenBLAS threads (oversubscription). Size S. Measured saving: India -37 s, Ethiopia -10 s. Identical (verified)

* **Cost.** On India the climatology takes **39.6 s on 20 busy cores**. The same stage takes **0.8 s** when
  OpenBLAS is limited to one thread per process, and 2.8 s run serially in one process. On Ethiopia it drops from
  9.7 s to 1.5 s. py-spy shows 98% of the worker time in `gaussian_kde.evaluate` → `scipy.linalg.solve_triangular`
  (`hcwf/onset/climatology_kde.py:74`). That is a tiny 1-D triangular solve. In isolation, one `evaluate` takes 0.2
  ms, about 1 s for all 4948 units. In the pool it is about 40x slower, because 20 workers each spin 20 OpenBLAS
  threads.
* **Root cause.** `hcwf/stats/parallel.py:144-171` (`_pool_for`) sets only `NUMBA_NUM_THREADS` for the spawned
  workers. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are left at the machine default
  (`threadpoolctl` reports `num_threads: 20` in every process). Only the global-CV workers limit themselves
  (`logit.py:_run_in_worker`). onset_blending has the same defect, which is why its climatology also takes 40 s.
* **Fix.** In `_pool_for`, also set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `threads`
  while the workers spawn, as is already done for numba. Alternatively, call `threadpoolctl.threadpool_limits(threads)`
  in `_warm` or in an initializer. In the main process, also limit BLAS around the serial small-matrix paths: the
  forecast's climatology (`blend/forecast.py:240`, `workers=1`) burns 80 CPU-s for 24 s of wall time. With the limit
  it takes 27 CPU-s for the same wall time.
* **Measured** (whole run with `OPENBLAS_NUM_THREADS=1`):
  * India 253 s → **216 s** and 3176 → 2380 CPU-s.
  * Ethiopia 31.0 s → **20.8 s**.
  * Every artifact and every `trained.blend` member is identical to the default run's; only the manifest
    timestamp differs. hcwf then beats onset_blending on Ethiopia (20.8 s vs 28.7 s).

### H2. Global CV: 27 single-threaded L-BFGS fits run in two rounds, and the final fit is queued last. Size S-M. Identical unless a trade-off is chosen

* **Cost.** India global CV takes 95 s (cProfile; 122 s for the whole evaluation stage).
  * Each task fits 1.6 M rows × 28 columns × 5 classes. Fits converge in 166-190 L-BFGS iterations and take
    27-54 s each, 1049 CPU-s in total.
  * The time splits into `X @ W.T` (29%), `grad.T @ X` (29%), sklearn's Cython multinomial loss (27%), the
    intercept-gradient `sum` (8%) and `np.unique` on object-string labels (3%) (`logit.py:536,570`).
  * sklearn 1.9 hard-codes `n_threads = 1` in `LogisticRegression.fit`, so a single fit is single-core apart from
    BLAS.
* **Root cause.** `_workers` (`hcwf/blend/logit.py:459-471`) caps the pool at "the fewest workers that keep the
  same number of rounds": ceil(27/2) = **14 workers, which leaves 6 of 20 cores idle**.
  * The `final` task is appended last (`logit.py:388-390`), although it is the largest (1.66 M rows), so it runs
    in the second round.
  * The tree's CPU use falls from 14 cores to 1 for the last ~40 s of the CV (sampler timeline, 195-236 s).
* **Measured** in the CV harness, which replays the exact `fit_fold` and `fit_final_rows` on the dumped design. All
  three schedules give identical probabilities.

  | schedule | wall |
  |---|---:|
  | 14 workers, final last (today) | 97.5 s |
  | 14 workers, **final first** (longest task first) | 89.7 s |
  | 20 workers | 83.2 s |
  | **27 workers, all tasks at once** | **74.3 s** (tree PSS about +12 GB, still within the run's memory budget) |

* **Fix (identical).** Order the tasks longest first, so that `final` comes first. When the memory budget holds
  `n_tasks` workers, run one worker per task, without the rounding down to "same number of rounds". Expected saving:
  India -20 to -25 s; Ethiopia about -3 s.
* **Trade-offs (these change numbers; listed for the user to decide).**

  | option | effect on time | max \|ΔP\| vs today |
  |---|---|---:|
  | Warm-start folds from the final fit | Fold CPU halves (1049 → 526 CPU-s); wall 73 s at 14 workers, because the final fit must finish first | 0.031 (India), 0.014 (Ethiopia) |
  | float32 design | 52 s | **0.19** |
  | `tol=1e-4` | | **0.11** (Ethiopia) |

  The float32 design and `tol=1e-4` are not recommendable.
* **Finding: the fit depends on the path to the optimum.** The fitted probabilities move by up to 0.03 when only
  the L-BFGS starting point changes. With `C=1e9` (practically unpenalised) and `tol=1e-5`, the optimum is flat
  along some directions. "Identical results" across machines and solvers therefore hinges on a bit-identical
  optimiser path: the same BLAS kernels, the same column and class order. Tightening `tol` would make the results
  robust to the path, at a CPU cost; this is a design question for the user, not a quick win.

### H3. Serial scoring after the CV in the main process: ~36 s on India. Size M. Identical / within tolerance

* **Cost** (cProfile of `evaluate`, main process only):
  * `_Scores.add` 25.8 s over 7 models. Of that, `cell_metrics` takes 21.3 s:
    * **34,678 `scipy.stats.rankdata` calls take 12.0 s.** This is the per-unit AUC loop at
      `hcwf/metrics/probabilistic.py:241-251` plus the pooled AUCs over 10.7 M values.
    * `pietra` (KS test) takes 3.1 s.
  * `_Context.allowed_ids` takes 3.0 s: a Python `set` membership test over 2 M string ids, per model
    (`evaluation.py:334`).
  * `_calibrated` takes 7.9 s.
  * `_climatology_baselines` takes 7.2 s (mostly scoring).
* **Fix.**
  * Compute the per-unit AUCs with one `lexsort` by (unit, score) per model, with average ranks for ties per group
    in numpy. The per-group rank sums are exact in float, so results are identical or within 1e-15 relative.
  * Compute `allowed` from `sample_unit` codes (`unit_allowed[sample_unit]` is already computed in `_Context`)
    instead of string sets. This is identical.
  * Score the independent models in the pool, which already exists. This is identical.
* **Expected.** 36 s → under 10 s on India, about -2 s on Ethiopia.

### H4. Forecast-onset stage: 1.6x onset_blending's CPU and 2.6x its memory. Size M. Identical

* **Cost.**
  * India aifs_ensv2: **63.4 s and tree PSS 40.9 GB** (onset_blending: 40.5 s, 15.6 GB).
  * One file (43 issues × 26 members × 50 days × 129×135 cells, zlib) through `features.forecast` in one process
    takes 17.3 s and 1.6 GB:
    * netCDF4 decompression, 8.2 s (unavoidable);
    * **`SparseTransform.observed_mean` 3.2 s**, including 1.4 s of `ravel` copies inside scipy's sparse matmul
      (`hcwf/spatial/regrid.py:256-269`);
    * `unit_series` float64 transposes and casts, 0.7 s (`hcwf/io/input.py:1046-1083`);
    * `_load_kept` NaN fill and masking of the full grid, 0.8 s (`hcwf/utils/region.py:241-257`);
    * Kahan member mean, 0.8 s;
    * detection and counts, about 2.1 s;
    * `format_grid_ids`, 0.4 s (see H6).
* **Root causes.**
  1. On the India grid the units are the grid cells themselves (identity weights). Even so, every value goes through
     a float64 sparse matmul twice: once for W @ filled and once for W @ valid. The result is x·1.0/1.0 = x.
  2. `_load_kept` allocates the **whole lat×lon grid**, NaN-filled, for a run of 11 issues: 11×26×50×17,415 float32
     values = 1.0 GB. Only 4948 of the 17,415 cells are kept (28%). `unit_series` then makes a float64 copy for the
     kept units (0.57 GB) plus per-block temporaries. With 20 workers that adds up to about 40 GB.
     onset_blending's reader (`core/io.py:_read`) writes the kept points of each tile straight into a compact
     float32 array.
  3. The 26 files are scheduled as 26 tasks on 20 workers, which takes 2 rounds: about 23 s at the end with only
     6 cores busy (timeline 65-88 s).
  4. `features.forecast` collects every file's Dataset before `xr.concat` (`blend/features.py:217-224`), so at the
     concat both the parts and the result are held (2 × 3.25 GB in main).
* **Fix.**
  1. A fast path in `unit_series` for a one-source, weight-1.0 transform: `out = cells[:, source]`, cast once.
     This is identical: dividing by 1.0 is exact, and NaN stays NaN.
  2. A compact `_load_kept` (allocate kept cells only, as onset_blending does). Identical.
  3. Items become (file, run of issues) rather than files (104 tasks on India), so the pool stays busy to the end.
     Identical: issues are processed independently.
  4. Preallocate the (issue, unit, day) result arrays and fill them as items complete. Identical.
* **Expected.** 63 s → about 30 s, tree PSS 41 → about 15 GB. The same item split makes the India **live forecast**
  about 24 s → 8 s, because the 2024 ensemble file currently runs in a single worker.

### H5. Main-process memory: high-water mark 8.9 GB heap (10.2 GB RSS) on India. Size S each. Identical

memray (high-water-mark allocation records; the peak falls inside `weekly`):

| GB at peak | where | fix |
|---:|---|---|
| 3.25 | the held forecast arrays (`blend/features.py:223`): 3 uint8 count variants × 45 days, and `rain_mean` float64 × 45 days, per model | Keep only what the recipe reads: count variants `base` + `probability_variants()` (here `fixed_cutoff`; `ref` is unused) and `probability_days` (28) days; `rain_mean` up to `rain_horizon_days` (38). The forest and live forecast ask for their own `max_day`. **About -1.5 GB.** float32 `rain_mean` would change numbers (see section 3) |
| 1.60 | `_stream_reference` (`io/input.py:277`): 30 threshold years, float64 "exact" tenths + bool, kept in `_reference` for the whole run | The blend never reads it again (observed onsets are computed in workers). Call `io_input.release_reference()` after `load_thresholds` in `training.prepare`, as the benchmark flow does (`stats/climatology.py:106`). **-1.6 GB** |
| 1.04 | `unit_series` in `_read_obs_units` (`io/input.py:1267`): 21 observation years (2000-2020 = threshold years ∩ run years) held in main's observation cache, float64 | The blend's observed stage reads in workers, so these are dead weight. Read the threshold years with `hold=False` in the blend flow. **-1.0 GB** |
| 0.76 | `_rolling_sums` temporaries (`blend/features.py:520`), from 1 M-row chunks (`_CHUNK_ROWS`, line 45) of 38-day float64 rows | Use 256 k-row chunks. **-0.6 GB** |
| 0.53 | the weekly features themselves | (needed) |

Expected: main heap 8.9 → about 4.5 GB. Separately, between stages the idle pool workers hold about 0.4 GB each
(16 GB tree PSS during climatology, weekly and evaluation, when only main works). Each worker's observation cache
keeps the run years it read, plus freed-but-unreturned arenas. Fixes: `hold=False` for the per-year observed reads
in workers, or recycle the pool after the onset stages.

### H6. Thresholds and observed onsets: 15.6 s vs onset_blending's 8.8 s. Size S. Identical

* **`unit_thresholds` → `format_grid_ids` → `established_convention`** formats 17,415 grid ids in Python, once per
  observation year per worker. It takes **36% of `_observed_year` worker time**: 125 years × about 0.35 s = 44 CPU-s,
  plus 0.4 s per forecast file (`hcwf/stats/detect.py:57-81`, `hcwf/onset/thresholds.py:145`). Fix: resolve the
  per-unit threshold array once in main, and pass the array (or memoise it on the id tuple).
* **The cold threshold computation is serial.** `_stream_reference` reads and ingests 30 years one after the other
  (`ordered_prefetch`): 7.7 s at 1.1 cores. Reading the years in the pool, then filling the preallocated
  `DailyRain` in year order, gives the same arrays. Expected about -5 s per cold run. The cache does not help
  ai-almanac jobs, because each job has a fresh run folder.
* The 2000-2020 observation years are read twice, once for thresholds in main and once in workers. This is minor
  (about 5 CPU-s).

### H7. Minor items (S, identical)

* **Labels as object strings in every fit.** The CV passes `outcome` as object strings (`Weekly.take` →
  `outcome_series`). sklearn runs `np.unique` on objects (0.31 s per call; 3% of fit time), and `fit_fold` runs
  `isinstance` loops over 1.6 M items. Passing int8 codes renumbered in **sorted-label order** keeps the same
  class order, and so gives bit-identical L-BFGS, with the label names put back on `classes_` afterwards.
  Expected: about 1-2 s per fit.
* **pandas in the climatology unit loop.** `probabilities_for_issues` calls pandas `groupby().get_group` once per
  unit (0.44 s of the 2.8 s serial climatology; `climatology_kde.py:389-407`). Use one `argsort` by id and slices.
* **`rows()` (`blend/features.py:274-315`)** gathers unit ids as unicode arrays per sample (2.1 M rows) and runs
  `searchsorted` on strings for every model and the observed onsets. Map unit → position once per unit, then
  gather integers. This is part of the 4.2 s weekly stage, the only serial stage left before the CV.
* **The closed-form 1-D Gaussian KDE** matches `gaussian_kde.evaluate` to 4e-15 relative, but is only 1.2x faster
  than scipy once H1 is fixed. Not worth the deviation.

## 3. Data types and layout: what was checked

| data | today | verdict |
|---|---|---|
| design matrix (2.14 M × 28) | float64, Fortran-order shared memory; each fold scatters its standardised rows into a C-order X; no sklearn copy on `fit` (C-contiguous float64 is accepted as is) | keep. float32 changes CV probabilities by up to 0.19 |
| weekly features | float64 arrays (0.56 GB India) | keep (feeds the design) |
| `rain_mean` | float64 Kahan mean of float32 members, all 45 days | trim days (H5); float32 would change the rain predictors by ~1e-7, and H2's path sensitivity can amplify that |
| member counts | uint8 × 3 variants × 45 days | right type; drop unused variants and days (H5) |
| reference rain for thresholds | float64 "exact" + bool, 1.6 GB, held | release after use (H5) |
| per-file read | float32 full grid → float64 units | compact float32 kept cells; cast once (H4) |
| outcome | int codes in storage, object strings per take | int codes into sklearn (H7) |
| climatology KDE | float64, per unit | keep; fix threads (H1) |

Design matrices are built from `design.plan` arrays, not patsy. Patsy runs only on one row, to learn the column
order, and in `FittedModel.predict` for the live forecast (213 k rows, negligible). Saving `trained.blend` and the
artifacts takes about 1 s.

## 4. Expected result if H1-H6 are done

India training: 253 s → about **130-140 s** with the identical fixes. The H1 measurement gives -37 s. The other
expected savings are -22 s CV schedule, -26 s scoring, -30 s forecast onsets, -5 s thresholds, -3 s observed.
Tree PSS would fall from 41 GB to about 16 GB, and main RSS from 10.2 GB to about 6 GB. Warm-started folds (H2,
changes numbers) would take off roughly another 20 s.

Ethiopia training: 31 s → about 17 s. India live forecast: 24 s → about 8 s.

All of these, except the H2 trade-offs, keep results identical. The H3 AUC is identical or within 1e-15. The parity
driver (`tests/blend/realdata/run_parity.py`) plus a byte comparison with a reference run would confirm each step,
as was done for H1 (`~/perf_blend/cmp.py ref/ind_ob1 work/ind/runs`).
