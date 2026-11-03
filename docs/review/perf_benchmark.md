# Performance review: the benchmark pipeline (hcwf-run without blend)

Read-only review with measurements, made on this machine (20 cores aarch64, 121 GB RAM, local disk, warm page
cache) on 2026-11-03, at commit `6194117`. Line numbers refer to that commit. Another agent started editing the
working tree at 02:27; every run after that used a pinned copy of `6194117` (`git archive` into the scratchpad), so
all numbers below are for the committed code.

Scratch scripts, logs and profiles are in
`/tmp/claude-1000/-home-hcwf-code-ROMP/f3fbeea6-493c-485b-bfb6-ad02edb09411/scratchpad/perf_bench/`:
`bench.sh` / `benchh.sh` (timed runs; `benchh.sh` runs the pinned copy), `results.txt` (every timing),
`treepss.py` (peak PSS summed over the process tree), `readbench.py`, `rb2.py`, `rb3.py`, `fcread.py`
(forecast read micro-benchmarks), `obsread.py`, `agg.py` (py-spy aggregation), `mpeak.py` (memray peak
breakdown), `prof/` (py-spy raw stacks, memray captures).

## 1. Baseline (to compare an implementation against)

Wall time and peak RSS come from `/usr/bin/time -f "%e s %M KB"`. `%M` is the largest *single* process, so the
last column adds the peak PSS summed over the main process and its workers. CPU = (user+sys) / wall / 20 cores.
All runs are cold: the run folder and its `.onset_cache` are deleted first, and numba's cache is warm unless noted.

| workload | config | wall s | peak RSS (1 proc) | CPU util | tree peak PSS |
|---|---|---:|---:|---:|---:|
| NetCDF slice a, India AIFS_ENS, 2018, 26 mem | `cfg/nc_slice_a_india_aifsens.yaml` | 51.2 | 2.86 GiB | 5.5 % | (serial) |
| NetCDF slice b, India GraphCast, 2018 | `cfg/nc_slice_b_india_graphcast.yaml` | 9.0 | 1.48 GiB | 6.7 % | (serial) |
| NetCDF slice c, Ethiopia AIFS_ENS, 2018 | `cfg/nc_slice_c_eth_aifsens.yaml` | 19.8 | 1.46 GiB | 6.0 % | (serial) |
| NetCDF slice d, Ethiopia AIFS, 2018 | `cfg/nc_slice_d_eth_aifs.yaml` | 2.2 | 0.63 GiB | 11.9 % | (serial) |
| zarr a, India AIFS_ENS, 2018-2020 (3 y) | `cfg/zarr_a_india_aifsens.yaml` | 12.7 | 2.54 GiB | 25.8 % | 7.3 GiB (5 proc) |
| zarr b, India GraphCast, 2010-2020 (11 y) | `cfg/zarr_b_india_graphcast.yaml` | 10.4 | 1.94 GiB | 45.9 % | 11.6 GiB (13 proc) |
| zarr c, Ethiopia AIFS_ENS, 2010-2020 (11 y) | `cfg/zarr_c_eth_aifsens.yaml` | 8.7 | 1.59 GiB | 46.6 % | 13.2 GiB (13 proc) |
| zarr d, Ethiopia AIFS, 2000-2024 (25 y) | `cfg/zarr_d_eth_aifs.yaml` | 6.2 | 0.81 GiB | 54.6 % | 9.2 GiB (22 proc) |
| full NetCDF a, India AIFS_ENS, 2018-2020 | `cfg/full_nc_a_india_aifsens.yaml` | 55.5 | 2.24 GiB | 13.8 % | 6.1 GiB (5 proc) |
| slice c with every figure on | `cfg/fig_slice_c_eth_aifsens.yaml` | 23.6 | 1.84 GiB | 5.9 % | (serial) |
| slice c on admin units (ETH_Regions, 14 units) | `cfg/units_slice_c_eth_aifsens.yaml` | 9.6 | 0.81 GiB | 6.5 % | (serial) |
| same, first run after a code move (numba compiles) | same | 17.0 | 0.91 GiB | 5.9 % | |
| warm rerun (cache hit) zarr a / NetCDF slice c | same configs, `--overwrite` | 10.5 / 19.1 | 2.52 / 1.47 GiB | | |
| `hcwf-run -p slice_c --check` | | 1.07 | 0.52 GiB | 14 % | |
| `python -c "import hcwf.cli"` | | 0.02 | 15 MB | | |

Notes. The zarr configs in `data/phase7/profile/cfg_zarr/` cover the full periods, not the slices (only NetCDF
slices exist), so the rows compare different amounts of work. Runs repeat within about 5 % (slice c 19.8 / 19.1 s,
zarr a 12.7 / 12.0 s). The full India ensemble run (a, 3 years) takes 55 s on NetCDF and 13 s on zarr, well within
15 minutes. Cold start costs little: `hcwf.cli` imports lazily. `--check` imports xarray (171 ms incl. pandas),
regionmask (72 ms), numba (59 ms) and matplotlib (36 ms), about 0.6 s in all. A spawned worker costs 0.6 s of
imports and 256 MB PSS before it does any work.

### Where the time goes (run.log timelines + py-spy)

* **NetCDF ensemble slices: about 85 % is two reads of the same forecast file.** In slice c, py-spy puts 88 % of
  the samples in `load_blocks` (region.py:217, `DataArray.values` into netCDF4). The read runs twice: 14.2 s
  (relative) in `multi_year_forecast_obs_pairs` for the skill scores and 18.3 s in `_onset_tables_of_years` for the
  spatial metrics. The run.log shows the same: forecast load 02:19:09 to :18 (9 s), and again 02:19:19 to :27 (8 s).
  Onset detection, pairs, scores and outputs together take under 2 s.
* **Full NetCDF India (3 y):** thresholds over 30 obs years take 4 s. Then three workers spend 19 s each on the
  first forecast read and 20 s on the second. The climatological onset is computed again at the end (3 s). The
  other 17 cores stay idle (CPU 13.8 %).
* **zarr runs:** decoding runs in zarr's thread pool (33 of 64 sampled s are `as_numpy_array_wrapper`), so
  reading costs little. The duplicate read and detection cost about 3 to 4 s of 12, and the second climatology
  pass costs 2 s.
* **Figures:** 3.8 s for 14 PNGs in slice c. `plot_reliability_diagram` takes about half of it, because it
  saves at `dpi=600` (reliability.py:103) while every other figure uses 300.

## 2. Ranked hot spots

Costs are measured on the workloads above. Gains are estimates from micro-benchmarks run on the same files.
Under "Results": **identical** means bit-for-bit equal outputs. **trade-off** means the outputs change, so the
user decides. S/M/L is the size of the change.

| # | hot spot (file:line) | measured cost | root cause | fix | expected gain | results | size |
|---|---|---|---|---|---|---|---|
| 1 | NetCDF forecast read with a fancy init index: io/input.py:865 `ds.sel(init_time=matching_times)` → utils/region.py:217 `load_blocks` | slice c 8.4 s per read (raw contiguous read 2.2 s); slice a 19.8 s per read (contiguous 9.3 s); allocates 44 GB of temporaries per slice-a run (memray) | netCDF4 orthogonal indexing with an integer list becomes one HDF5 hyperslab per index run. The chunks (Ethiopia 15×9×17×17×21 float32 = 3.3 MB) are larger than HDF5's default 1 MB chunk cache, so each zlib chunk is decompressed again for every selected init | Within each stored time chunk, read the contiguous range `lo:hi` that covers the wanted inits, then take the inits in memory (see 3.1). Memory stays at one block. Alternative: raise the chunk cache to ≥512 MB with `netCDF4.set_chunk_cache` (also 2.2 s, but costs memory) | slice c −12 s (19.8 → ~8), slice a −21 s, full NetCDF India −20 s on the critical path | identical (verified `array_equal`, NaN-aware) | S |
| 2 | Each ensemble year's forecast is read and detected twice: stats/bins.py:529 + :542 (skill pairs) and stats/benchmark.py:199 + :214 (onset tables) | slice c 8.4 s, slice a 20 s, full NetCDF India ~20 s wall, zarr a ~3-4 s, units ~2 s; plus a second `compute_onset_for_all_members` (≈1 s per India year) | `prepare_score_cache` and `onset_tables` are separate passes over the years. Both call `compute_onset_for_all_members` with the same forecast, thresholds and history. The skill pass drops the ensemble table (`onset_all_members, _ =`), which is exactly what the onset tables need | One per-year job that returns both the pair table and the ensemble onset table. Put the ensemble table into `benchmark._ONSET_TABLES` under the family key the spatial pass computes. That needs the same kwargs (day_bins only affects the pairs) | halves forecast reads and detection for every probabilistic run: after #1, slice c ~8 → ~5 s, slice a ~30 → ~17 s, zarr a 12.7 → ~9 s | identical (same function, same inputs) | M |
| 3 | The climatology years' observations are read even when every onset is cached: stats/climatology.py:77 (`compute_climatological_onset`, the spatial climatology reference) and :274 (`..._dataset`, on warm runs) | India NetCDF 3.5 s, India zarr 1.1 s, Ethiopia 0.3 s per pass, on the main process's critical path; a warm run repeats both passes (warm zarr a still 10.5 s) | `_read_years` reads (and ingests) each year before `detect_observed_onset` asks `observed_cache.lookup`. The cache key does not need the rain: units already pass a reader callable (detect.py:149) | Give `detect_observed_onset` a reader callable per year (as for units). Read only on a cache miss. On a cold run the second pass then reads nothing | −3.5 s India NetCDF (−6 %), −1 to −2 s zarr; warm runs skip all climatology reads | identical | S |
| 4 | Serial runs keep the reference rain alive during the skill pairs: io/input.py:277 (`_stream_reference` buffer) held from the thresholds until metrics/skill.py:76-77 releases it, *after* `_start_pairs` (:65) has run | 596 MB of the 2.63 GB peak of slice a (memray high-water mark: reference 596 MB + forecast float32 462 MB + tenths 462 MB + flags 115 MB + detection tables) | With one worker, `start_items` runs the pairs before returning, so the climatology that frees the reference comes later | When `worker_count <= 1`, compute the climatological onset (and release the reference) before the pairs. With workers, keep the current overlap | slice a peak −0.6 GB (−23 %); India serial runs generally | identical | S |
| 5 | The float32 forecast cube stays alive while it is ingested and detected: io/input.py:908 → stats/detect.py:213 `ingest_model_forecast` → onset/ingest.py:279 | 462 MB of slice a's peak (India, 12×26×34×117×93) | The callers (`p_model` in bins.py and benchmark.py) hold the cube while ingest makes the int32 tenths and bool flags from it, so the year exists twice | Ingest block by block as the cube is read (load_blocks already reads in blocks), or hand the cube over and `del` it before detection | −0.46 GB per India year per process (main or worker) | identical | M |
| 6 | Masked cells are carried through the forecast arrays: the box is read, masked cells are set to NaN (region.py:274), and ingest and detection then run on every box cell | India: 6794 of 10881 box cells are masked (62 %); Ethiopia 1515 of 2989 (51 %). They come out as INSUFFICIENT_DATA and are dropped from the tables | The grid layout keeps (lat, lon) through ingest (detect.py:213) and the engine (`_ensemble_summary`, `detect_stitched`) | Before ingest, gather the kept cells into one "cell" axis (the layout units already use); scatter back only for the map outputs | −50 to 60 % of the forecast working set (tenths, flags, prefix tables): about −0.6 GB on India slice a, and less detection CPU | identical (masked cells are excluded today) | M |
| 7 | int32 tenths plus a separate bool missing array: onset/ingest.py:279 and io/input.py:277 | reference rain, 30 India years: 596 MB; forecast tenths 462 MB + 115 MB flags | 5 bytes per value. Daily rain in tenths fits int16 (up to 3276.6 mm/day), and the missing flag could be a sentinel | int16 tenths with sentinel −32768 = missing. If any value of a block exceeds int16, fall back to int32 for the whole array. The numba kernels take the dtype as a type parameter | reference 596 → 238 MB, forecast 577 → 231 MB per India year | identical while the guard keeps int32 for out-of-range data | L |
| 8 | Year workers are sized by core count only: stats/parallel.py:221 `worker_count(config, len(items))` gets no `item_bytes` | tree PSS 13.2 GiB (zarr c, 12 workers), 11.6 GiB (zarr b), 9.2 GiB (zarr d), against a 0.8 to 1.9 GiB single-process peak. On a 16 GB laptop zarr c would swap. 256 MB per idle worker | `workers_within(item_bytes, ...)` exists (utils/workers.py) but is not used for year tasks | Pass a per-year estimate, e.g. selected forecast bytes × 3 (cube + tenths + tables; `unit_forecast_bytes` exists for units), so the memory budget caps the workers. Fewer workers then get more numba threads each | bounded footprint (memory-efficient by design); wall time unchanged where memory is plentiful | identical | S |
| 9 | Dates are round-tripped through strings on hot paths: detect.py:416 and :427 (`strftime` per member row), bins.py:268, then benchmark.py:60-62 `pd.to_datetime` parses them back | India year: about 1.3 M member rows. strftime of 5.4 M dates takes 0.67 s and the parse back 0.32 s (measured). The object strings take about 60 B per row (~80 MB per India year); memray counts 645 k string allocations in slice a | The tables carry `obs_onset_date` and `onset_date` as text, a leftover from the CSV-shaped API | Keep `datetime64[D]` columns internally and format them only where a CSV or NetCDF is written | ~1 s and ~100 MB per India year per pass (×2 today, see #2) | identical if the writers format the same text | M |
| 10 | Figures: graphics/reliability.py:103 `savefig(dpi=600)`; every figure is rendered one after another in the main process | 3.8 s of the 23.6 s figure run in slice c. Reliability is about half of that (two PNGs of 370 to 460 KB) | 600 dpi has 4× the pixels of the 300 dpi that every other figure uses | (a) dpi 300 like the other figures. (b) Optionally render the figures in the existing worker pool once the scores are known | (a) ~−1.9 s per run; (b) another −1 to −1.5 s | (a) **trade-off**: the PNG is smaller (no numbers change); (b) identical | S / M |
| 11 | numba compiles on the first run after an install or a code move (`cache=True` keys the cache on the source path), in the main process and in every worker | +7.4 s (units slice: 17.0 s first run against 9.6 s after) | Normal JIT behaviour; nothing compiles ahead of time | Compile the kernels on a background thread at startup while the inputs are read, or in `hcwf-run --check` and after install | −7 s on first runs (training sessions, CI, a fresh docker image) | identical | S |
| 12 | Parallelism gap: a one-year run is serial (5 to 7 % of 20 cores), and NetCDF/HDF5 decoding is single-threaded, so full NetCDF India uses 2.8 of 20 cores | slice a: 40 s of 51 in one thread decoding zlib; full NetCDF India: 3 workers decode while 17 cores idle | Work is split into one item per year only. HDF5 is not thread-safe, so the NetCDF read cannot use the zarr thread pool | (a) Split a year's forecast read into chunk-range items (time chunk × member chunk) over the pool's processes, each filling part of a shared-memory buffer. (b) Point users with big archives to zarr (India a: 55 s NetCDF against 13 s zarr). `hcwf-run convert`/docs could offer a rechunk recipe | (a) slice a read ~9 s (after #1) → ~1 to 2 s; (b) 4.4× end to end | identical | L / docs |
| 13 | zarr India obs store is chunked as (45656, 26, 14): one chunk holds the whole 125-year history of a tile (`data/india/obs.zarr`) | the main process peaks at 1.6 GB heap during the reference read: 0.8 GB of zarr decode buffers + 0.48 GB block | Reading any year decodes every tile's full history. `obs_years` already groups years to decode each tile once (`_long_time_chunks`) | Data layout, no code: rechunk time to 366 to 1461 days (Ethiopia uses (10227, 49, 31), the same pattern). Or lower `_GROUP_BYTES` on small machines | −0.8 GB main-process peak; faster single-year reads | identical | S (data) |
| 14 | int64 prefix sums in the stitched kernel: onset/backends/numba_cpu.py:691 (`fpx`) | 165 MB of slice a's peak (82 MB for each int32 table at :692-693) | Cumulative tenths over at most ~80 forecast days fit int32. History tables cover years, but per series, so they also fit for real rain | int32 `fpx` with an overflow check (max tenths × width < 2³¹, else int64) | −82 MB per India year | identical with the check | S |

Checked and not hot (keep as is): the pair building and the contingency metrics are vectorised with bincount
(bins.py:176-270, benchmark.py:26-130, < 0.3 s per India year). `_to_tenths` is one parallel numba pass. Observation
NetCDF files are chunked one day per chunk and read in 0.3 s for 27 Ethiopia years. Thresholds and harmonic seasons
are cached. Detection kernels run in < 1 s per year. xarray adds little overhead in the inner loops, because the
hot loops already run on numpy arrays. Imports are lazy.

## 3. Fix notes

### 3.1 Chunk-aligned init reads (#1)

`rb3.py` measures this on Ethiopia 2018: fancy `v[idx, :, :50]` takes 8.31 s; the chunk-aligned loop takes 2.15 s
with identical values. On India it takes 27.2 s against 9.3 s (full grid, steps :34). The loop:

```python
c = var.chunking()[0]                       # stored chunk length along init_time
for k0 in range(0, n_time, c):
    m = (idx >= k0) & (idx < k0 + c)
    if m.any():
        lo, hi = idx[m].min(), idx[m].max() + 1
        out[m] = var[lo:hi, ...][idx[m] - lo]   # one contiguous hyperslab per stored chunk
```

In hcwf this belongs in `load_blocks` (or in `_open_forecast`). Keep the lazy array on the file's own init axis,
pass the wanted positions, and let each block be "the wanted inits of one stored time chunk", sub-blocked along
member chunks if a block would exceed `_BLOCK_BYTES`. The zarr path is unaffected: zarr decodes whole chunks
anyway and caches nothing between calls, so the same blocking there is at worst neutral.

### 3.2 One pass per ensemble year (#2)

`skill_score_in_bins` runs before `spatial_far_mr_mae_map`. Let `multi_year_forecast_obs_pairs` (or its
per-year worker) return `(pairs, ensemble_df, onset_da)`, and store `{year: (ensemble_df, onset_da)}` in
`benchmark._ONSET_TABLES` under `_family_key(ens=True, ...)`, built from the case kwargs without `day_bins` and
the other pair-only keys. The spatial pass then finds the tables and reads nothing. The two passes share
thresholds, history and members today, so the stored table equals the one the second pass would compute.
`tests/` should check this with a run that compares both orders.

### 3.3 Order of the gains (projected, NetCDF slice a 51 s / 2.86 GiB)

#1 → ~30 s · #2 → ~17 s · #3 → ~16 s (more on multi-year India) · #4 + #5 + #6 → peak ~1.2 GiB · #12a →
~8 s. zarr a (12.7 s): #2 + #3 → ~8 s, and #8 bounds its 7.3 GiB tree footprint. None of these changes numbers,
except #10a, which changes the reliability PNG resolution only.

## 4. Method

* Timings: `bench.sh`/`benchh.sh`, sequential, nothing else running. 20 cores, `parallel` at its default (workers
  = min(cores, years)).
* py-spy 0.4.3 `record --subprocesses -f raw` at 50 to 100 Hz, aggregated by `agg.py` (inclusive hcwf
  functions; leaf hcwf line → library call). Absolute seconds in py-spy output are inflated by thread sampling,
  so only the shares are used.
* memray 1.20 (installed with `--target` into the scratchpad, on `PYTHONPATH`): `memray run` of the main process
  (spawned workers are not followed), high-water-mark records grouped by hcwf frame (`mpeak.py`).
* Process-tree memory: `treepss.py` samples `/proc/*/smaps_rollup` Pss every 0.2 s over the run's process tree.
* Read micro-benchmarks run directly on the files (`rb2.py`, `rb3.py`) and through hcwf's own loader
  (`fcread.py`, cProfile).
