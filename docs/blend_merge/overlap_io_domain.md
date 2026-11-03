# Overlap report: data input, file discovery, domain/units, regridding, masks, regions, ids, parallelism

Paths: blending = `/home/hcwf/code/onset_blending/onset_blending/`, ROMP = `/home/hcwf/code/ROMP/momp/`.
`B:` is a blending file, `R:` is a ROMP file. Verdicts: SAME / ROMP-EXTEND / NEW.

Headline: ROMP's ingest layer (dataset description, detection, ingest into tenths, region box and masks,
zarr) is more general and should be kept as the single reader. Blending's real additions are:
(a) **evaluation units that are not grid cells**: sparse area-weight transforms onto polygons, unit ids and
dissemination subsets; (b) **sparse, chunk-aware reading** of only the cells that feed the kept units;
(c) **forecast signatures** for operational compatibility checks; (d) **memory-budgeted process pools and
shared-memory arrays**. ROMP never regrids rainfall (R:lib/check.py:266-271, R:utils/mask.py:1). For polygon
units it has only a whole-region mean (R:onset/engine.py:899-934).
Note: `R:onset/units.py` parses the onset-definition text grammar (amounts and spans). It has nothing to do with
rainfall units, so it does not overlap anything in this slice.

---------------------------------------------------------------------------------------------------------
## 1. Concept-by-concept mapping

### 1.1 File discovery (B:core/io.py:44-71, B:config.py:149-166 `FileSet`)
- **Blending:** `FileSet` takes either `folder` plus a regex `pattern` or explicit `paths`, with an optional
  `years` filter. `list_files` returns the files sorted by `file_year`, which is the first `(19|20)\d{2}` in the
  file name. A year can have any number of files: operational files hold one init each (B:forecast.py:8-10,
  :150-156 swaps `files` for the live paths). NetCDF only.
- **ROMP:** a dataset entry has `path` plus `file_pattern` with `{}` for the year
  (R:lib/dataset_schema.py:41-48, :197-199). It reads exactly one file per year, `file_pattern.format(year)`
  (R:io/input.py:481-497 obs, :702-724 forecasts), or one zarr store holding every year (R:io/input.py:47-70).
  `find_pattern` detects the pattern (R:lib/dataset_detect.py:46, :98-110).
- **Verdict: ROMP-EXTEND.** Keep ROMP's dataset entry and zarr support. Add two things:
  (1) an explicit list of `paths` (or a glob) with any number of files per year, needed for live operational
  files and replays (one init per file);
  (2) an optional `years` filter.
- **Differences:** ROMP's `YEAR_RE` needs a non-digit on both sides of the year, so a name like `20200101.nc`
  never matches. Blending matches the first 19xx or 20xx anywhere in the name. ROMP also accepts 18xx.

### 1.2 Dimension roles and aliases (B:core/io.py:93-103, :120-141, :384-394; B:config.py:240, :296-300)
- **Blending:** each dimension of the variable gets a role (time, day, number, lat, lon, adm3_name). It comes
  from the model's `dims`, else from a case-insensitive alias, else from a coordinate whose units are "days".
  Two dimensions with the same role is an error.
- **ROMP:** there are three alias tables already:
  - R:onset/ingest.py:17-22 (`_TIME/_LAT/_LON/_INIT/_LEAD/_MEMBER_DIMS`);
  - R:lib/dataset_detect.py:48-55 (`_ROLE_ALIASES`, with fuzzy "lead/step/timedelta");
  - R:utils/standard.py:4-102 (`dim_fmt*` renames).
  User overrides go in the dataset `dims` (R:lib/dataset_schema.py:207-214).
- **Verdict: SAME.** ROMP replaces it. Add the blending-only aliases: `initial_time`, `lead_day(s)`, `members`,
  `ensemble_member`, `sample`, `adm3`/`adm3_name` as a unit dimension, and "a coordinate in days is the lead".
  Merging also gives a chance to collapse ROMP's own three tables into one (in `ingest.py`).

### 1.3 Rainfall units (B:core/io.py:78-91, :144-152, :106-117; B:config.py:301-305)
- **Blending:** `MILLIMETRES_PER_UNIT` maps declared CF units to a factor (m, metres, meters, m/day, mm,
  mm/day, kg m-2, ...). If the file declares no units, the model's `unit_scale` (any float) applies. There is no
  magnitude check. `ForecastFormat.units` feeds the signature.
- **ROMP:** `_unit_scale` (R:onset/ingest.py:24-26, :58-67) accepts mm and m families and raises when units
  are undeclared. Dataset `units` can only be `mm|m` (R:lib/dataset_schema.py:46, :150).
  `dataset_detect._units` and `ingest.detect_units`/`_check_units` (R:onset/ingest.py:536-568) detect units from
  magnitudes (99.9th percentile < 0.5 means metres) and reject contradictions.
- **Verdict: ROMP-EXTEND.**
  - Add "metres" and "meters" to `_UNITS_M`.
  - Decide whether a free `unit_scale` survives. Recommendation: no. Map it to `units: m|mm`, or allow a
    numeric `unit_cvt` in the dataset entry. `Dataset.unit_cvt` already exists as a float, but `parse_dataset`
    only fills it from `units`.
  - Behaviour change: an undeclared, unconfigured file now gets units detected (or fails) instead of a silent
    factor of 1.

### 1.4 Observation reading (B:core/io.py:314-377 `read_observations`, `ObservationArray`)
- **Blending:** reads the whole file with netCDF4 and returns `rain` (time, point) in the file's dtype (NaN where
  masked), `points` as a DataFrame of lat/lon or id, `dates` as normalized datetime64 plus `shift_days`, and
  `years`. Duplicate dates and gaps within a year are errors. `adm3_name` string dimensions are supported.
- **ROMP:**
  - `load_imd_rainfall` (R:io/input.py:655-685) selects the year, cuts the region box, reads blocks, masks in
    place, scales, and caches the run's years (R:io/input.py:546-621).
  - `ingest` (R:onset/ingest.py:277-312) produces `DailyRain` as int32 tenths plus a missing mask, in any
    calendar. Time must sit at the same time of day. Gaps are filled as missing with a warning.
- **Verdict: SAME.** ROMP replaces the reader. Extend it for a string unit dimension (adm3 files) and for reading
  with no region box (see 1.10).
- **Differences that change results:** see §5: tenths rounding, negative values, gap handling, and `shift_days`
  versus the `day_boundary` label.

### 1.5 Forecast reading (B:core/io.py:183-311 `ForecastFile`, :538-596 `_forecast_layout`/`forecast_points`, :667-684 `_leading_tiles`)
- **Blending:**
  - Reads (issue, member, point, day) float arrays one run of issues at a time.
  - Reads only the storage-chunk tiles that contain a kept point, where kept = trained units, or the source cells
    that feed the kept polygon units (B:core/onsets.py:586-598).
  - Puts lead days in ascending order and checks they are consecutive (B:core/io.py:166-176).
  - Applies the scale either before or after the regrid (B:core/onsets.py:578-610).
  - Records `member_spread` (two or more non-NaN members).
- **ROMP:** `_load_forecast` (R:io/input.py:727-753) selects inits, drops step 0 (:740), cuts leads at
  `onset_steps_needed` (:743), selects members, and reads the region box via `region_select(read_by=...)`, which
  uses `load_blocks` (R:utils/region.py:167-196, chunk-aligned 64 MB blocks). It then masks and scales in place.
  `ingest_model_forecast` (R:io/input.py:175-199) and `ingest_forecast_info` (R:onset/ingest.py:605-693) handle
  accumulation, sub-daily steps, `daily_boundary`, lead labels, unit checks and tenths.
- **Verdict: ROMP-EXTEND.** Keep ROMP's path, and add a sparse-selection read: `load_blocks(da, block_dim,
  keep=(lat,lon) bool)` that skips spatial tiles with no kept cell. Without it, a global live file (blending
  allows "a larger domain such as the globe", B:trained.py:144-147) or an admin-unit run reads the full box.
  `member_spread` is NEW but minor, and could fold into the signature check.

### 1.6 Lead selection and rain horizon (B:core/io.py:397-416 `LeadSelection`, :611-664; B:config.py:256-276)
- **Blending:** reads `lead_days.min..max`. A strict horizon `rain_horizon.max_day` with policy exact or truncate
  requires days 1..N (`_check_rain_horizon`).
- **ROMP:** reads leads up to `onset_steps_needed`, which comes from the definition's reach
  (R:io/input.py:30-44). `_lead_days` requires leads 1..n after ingest (R:onset/ingest.py:315-335).
- **Verdict: ROMP-EXTEND.** Add the explicit lead window used by rain predictors (min, max, horizon policy) as
  dataset or blend settings. The onset reach stays ROMP's.

### 1.7 Time decoding (B:core/io.py:443-478 `nc_time`, `time_to_dates`)
- **Blending:** a hand-written CF parser (standard calendar only). Issue dates are `normalize()`d, so the hour is
  dropped.
- **ROMP:** xarray/cftime decoding (R:onset/ingest.py:98-135), with all calendars.
- **Verdict: SAME.** ROMP replaces it. Its `daily_boundary` can shift init labels (R:onset/ingest.py:640-645,
  :683); blending never does.

### 1.8 Forecast signature and live-file check (B:trained.py:53-103 `forecast_signature`, :106-122 `issue_dates`, :125-140 `_aggregation`, :143-178 `_signature_differences`; B:forecast.py:79-129, :143-148)
- **Blending:**
  - The signature holds: variable, units, `aggregation` (the `cell_methods` text over the lead dimension plus the
    lead coordinate's description), `issue_hours`, sizes of other dims, `lead_days`, `members` (≤ `max_member`),
    lat/lon value sets rounded to 6 digits, and `unit_ids`.
  - At training, every training file must give the same signature (B:trained.py:248-260).
  - At forecast time, a live file may hold more (leads, members, domain) but not less. A different grid step,
    units, aggregation or issue hour is refused. `fewer_members` relaxes the members rule.
- **ROMP:**
  - `resolve_entry`/`check_entry` (R:lib/dataset_detect.py:400-503, :735) resolve and check units, dims, members,
    accumulation, `lead_label` and region per dataset. The resolved values are cached under a fingerprint
    (:347-398).
  - `lib/check.py` does grid-overlap checks (R:lib/check.py:255-271).
  - ROMP has no "trained versus live" comparison.
- **Verdict: NEW**, built on ROMP. Proposed home: `momp/blend/signature.py` (or `momp/lib/signature.py`).
  - Build the signature from ROMP's resolved dataset entry (units, accumulation, `daily_boundary`, `lead_label`,
    members, dims) plus the grid axes and issue hours.
  - Replace the `cell_methods` string with ROMP's resolved accumulation and `daily_boundary`. This is a
    behaviour change: blending compares metadata text, ROMP detects accumulation from the values.
  - `issue_dates` maps to ROMP decoding plus `get_initialization_dates` (R:io/input.py:461-478); see risk 5.8.

### 1.9 Unit ids (B:core/ids.py, all of it)
- **Blending:**
  - String ids for units: admin names as given, grid cells as "<lat>_<lon>".
  - The number of decimals per coordinate is configured, "established" (reproduces existing ids, :130-148), or
    "inferred" (the fewest digits that stay collision-free within 1e-4 of the spacing, :89-118).
  - Also: normalisation (:177-196), coordinate-conflict checks (:199-210), and `with_id_column` (:218-244).
  - Ids join everything: CSV thresholds, reference onsets, dissemination lists, cell transforms, and the trained
    blend's `units`.
- **ROMP:** none. Cells are identified by float (lat, lon) coordinates. Matching across datasets uses a
  coordinate tolerance (R:onset/alignment.py:72-94, `_COORD_TOL` 1e-6).
- **Verdict: NEW.** Proposed home: `momp/spatial/ids.py`. Use it only at I/O boundaries: reading CSV unit tables
  and transforms, and writing tables. The in-memory key should be the unit index plus coordinates (see §2).

### 1.10 Area of interest and dissemination filter (B:core/domain.py:289-414 `AreaFilter`; B:config.py:170-233)
- **bbox** (lat/lon min/max, inclusive, compared on raw longitudes):
  - ROMP equivalent: `region_box` (R:utils/region.py:134-141), with longitude conformed by `lon_swap` (:79-114).
  - **SAME.** ROMP's version is better: it handles the seam and 0..360 grids.
- **GeoJSON outlines:**
  - Blending: `shapely.contains_xy` on the union of outlines, using unit lat/lon or centroids (:406-414).
  - ROMP equivalent: `mask: {file: x.geojson}` → `shape_file_geometry` plus `rasterise`
    (R:utils/mask.py:284-294, :158-202).
  - **SAME** for grid units. Difference: ROMP's `centre` rule uses `intersects_xy`, so a centre on the boundary
    is kept; blending's `contains_xy` excludes it. ROMP also offers `overlap_fraction`. For polygon units,
    blending tests unit centroids: ROMP-EXTEND (test the unit centroid or polygon, not a grid cell).
- **Dissemination units** (a CSV of unit ids; scope `inputs` restricts reading, scope `training` restricts
  training and scoring only):
  - **NEW.** It is a subset of evaluation units. Model it as a boolean over the unit axis in the unified unit
    table (§2). ROMP's cell masks are the grid special case.
- **Centroids** (unit lat/lon from a CSV, :379-404): **NEW**. They belong in the unit table.

### 1.11 Regridding and cell transforms (B:core/domain.py:417-556 `Regridder`, `regrid_weights`, `SparseTransform`, `point_sources`, `grid_axes`; B:core/reference.py:28-56, :123-129, :266-300; B:core/onsets.py:165-193)
- **What it does:**
  - A CSR matrix W of shape (target, source) built from a long table (source, target, weight).
  - `observed_mean(V)` computes (W @ filled) / (W @ observed) for V of shape (source, column) in 256-column
    chunks, giving NaN where a target has no observed source (:529-543).
  - `supported` gives each target's support under a presence mask.
  - Regrid weights are keyed by float (lat, lon) with exact equality (:437-444, :546-551). Cell transforms are
    keyed by string `source_id` via grid ids (B:core/onsets.py:165-193).
  - Building weights from `domain.regrid.shapefile` is **not implemented** (B:core/reference.py:291-300 raises
    `NotImplementedError`); weights must be supplied as CSV.
  - Transforms are stored in the trained blend (B:core/reference.py:140-161).
  - Inside blending, "regrid" and "cell_transform" are two paths over the same `SparseTransform`
    (B:core/onsets.py:73-96 and :541-546). They should become one concept: source-to-unit weights.
- **ROMP:**
  - `engine.regional` (R:onset/engine.py:899-934) averages one boolean mask into one series with cos-lat or equal
    weights, renormalised over valid cells and re-rounded to tenths. That is a single-row W.
  - `mask.align_raster_mask` (R:utils/mask.py:384-454) regrids a mask with separable area-overlap weights
    (`_overlap`, `cell_edges`).
  - `rasterise(rule="overlap_fraction")` (R:utils/mask.py:186-202) already computes cell-polygon overlap areas
    with shapely, which is exactly what polygon weights need.
- **Verdict: NEW**, in `momp/spatial/regrid.py`. It takes in:
  - `SparseTransform` and `observed_mean`;
  - one weights loader that accepts both key styles;
  - a weight builder from a shapefile, reusing the overlap code of `mask.rasterise` (planar degrees² today;
    add an option for equal-area or cos-lat cell areas);
  - grid-to-grid conservative weights from `_overlap` and `cell_edges`.
  Then reimplement `engine.regional` as a one-target transform so there is only one implementation. Its
  rounding (mean of tenths, then `rint`) must be kept or deliberately changed.

### 1.12 Masks (R:utils/mask.py (land_mask.py is merged in), R:utils/region.py:144-285)
- **ROMP:**
  - Binary keep-masks on each dataset's own grid: land (Natural Earth or regionmask), country (Natural Earth
    `admin_0`, matched on NAME, NAME_LONG or ADM0_A3), file (NetCDF 0/1 or shp/geojson, whose features are
    **all unioned**, :284-294), and condition.
  - Masks intersect, are recorded, and produce a PNG.
  - Legacy implicit masks: `land_only` plus `shp_only` (country), which are True by default
    (R:utils/region.py:216-285, now `default_mask` in R:utils/mask.py, a preset of the same `build_mask`).
- **Blending:** no masks. Cells come from what the observation file covers, then `AreaFilter`, then
  dissemination.
- **Verdict:** keep ROMP's masks as the way to pick grid units. They do not support admin-unit aggregation:
  every feature is unioned into one geometry, there is no per-feature id, and the result is boolean, not
  weighted. Per-unit aggregation is NEW (1.11).

### 1.13 Regions (R:onset/regions.py, R:onset/regions/*.yaml, R:params/region_def.py, R:lib/dataset_detect.py:243-265)
- See §3. **Blending has no named regions.** Its domain is per config. **Verdict: ROMP-EXTEND** (§3).

### 1.14 Parallelism (B:core/parallel.py)
- **`map_ordered(fn, items, context, workers)`** (:12-24): a spawn process pool over arbitrary items (files),
  with results in order and queued items cancelled on failure.
  - ROMP equivalents: `start_years`/`map_years`/`concat_years` (R:stats/parallel.py:213-279) are year-keyed,
    take kwargs dicts, keep one persistent pool per run (:149-179), snapshot and install read-once caches
    (:54-80), forward logs, and set numba threads per worker. Also `utils/workers.ordered_prefetch` (threads).
  - **ROMP-EXTEND:** generalise `start_years` into `start_items(func, items, ...)` and have `start_years` call it,
    so blending's per-file, per-climatology-cell and per-fold tasks use the run's single pool.
- **`workers_within(item_bytes, workers, budget)`** (:27-32) and **`default_resources()`** (:46-62): workers by
  memory budget, defaulting to half of `MemAvailable`.
  - ROMP: `worker_count` (R:utils/workers.py:23-31) only counts cores (affinity aware), plus a fixed
    `_GROUP_BYTES` (R:io/input.py:505).
  - **ROMP-EXTEND:** add the budget to `utils/workers.py`. Merge `default_resources` with `cores()`; blending uses
    `os.cpu_count()`, which ignores affinity.
- **`limit_blas_threads`** (:38-43, threadpoolctl): **NEW** in `utils/workers.py`, as the BLAS counterpart of the
  `NUMBA_NUM_THREADS` setting (R:stats/parallel.py:158-160).
- **`SharedArrays`/`attach`/`detach`** (:65-133, POSIX shared memory for spawned workers): **NEW** in
  `momp/utils/workers.py` or `momp/stats/parallel.py`. ROMP's snapshot only sends small arrays.

---------------------------------------------------------------------------------------------------------
## 2. Data model: cells and unit ids versus (lat, lon) grids with masks

**Blending:**
- Everything after reading is "units": 1-D, keyed by a string id.
  - Grid points get "<lat>_<lon>" ids.
  - Polygons (adm3) get their names, either from pre-regridded NetCDF with an `adm3_name` dimension
    (B:core/io.py:424-440) or from in-memory sparse regridding with supplied weights.
- Unit series are (time, unit) or (issue, member, unit, day) arrays. Tables are long, keyed by `id`.
- Thresholds and reference onsets are joined by id (B:core/reference.py).
- The trained blend stores `units` as id strings (B:trained.py:228, :268).

**ROMP:**
- Everything is (lat, lon) xarray, cut to a region box, with NaNs outside the mask.
- `DailyRain`/`ForecastRain` are generic over spatial dims (R:onset/model.py:220-283; the engine never needs
  lat/lon except `cos_lat`, R:onset/engine.py:919).
- The I/O, masks, observed cache ((year, lat, lon), R:stats/observed_cache.py:198) and benchmark tables are
  lat/lon.
- Political boundaries: only a country keep-mask (Natural Earth `admin_0`, R:utils/mask.py:247-273;
  `default_mask` in the same module) and an arbitrary shp/geojson unioned into one keep-mask. There is **no
  admin-unit aggregation**. The only aggregation is a single region mean (`engine.regional`).

**Recommendation: one representation, an `EvaluationUnits` table plus one transform per source grid**
(`momp/spatial/units.py`):
- `EvaluationUnits`:
  - `id` (str, ordered and unique), `lat`, `lon` (centroid), `kind` ∈ {grid, polygon, region};
  - optional `geometry`, `name`, `parent`;
  - `subsets`: named boolean columns such as `dissemination`, `training`.
- `transform(source_lat, source_lon) -> SparseTransform`, compiled once per source grid and cached (the
  `_KEEP_CACHE` pattern). The cases:
  - **grid** (the default): units = the observation grid's cells kept by ROMP's mask (`build_mask`). The
    transform is a 0/1 selection for sources on the same grid, or conservative weights for other model grids
    (blending's `regrid` without a shapefile, B:config.py:186-191).
  - **polygon:** units = shapefile features (`unit_id_col`). Weights are cell-polygon overlap areas, optionally
    clipped to observed coverage (`clip_to_coverage`), or supplied as CSV.
  - **region:** one unit = a mask with cos-lat weights. This replaces `engine.regional`.
- Rain flows file → (source cells) → `observed_mean` → (unit) in float mm → `ingest` to tenths with spatial dim
  `("unit",)`. The engine already accepts any spatial dims, so onset detection runs unchanged, and ROMP
  benchmarking and blending both consume per-unit onsets.
- Grid-kind outputs are unstacked back to (lat, lon) for ROMP's maps and NetCDF. Polygon-kind outputs are tables
  or GeoJSON keyed by `id`.
- String ids (`ids.py`) are used only to read and write CSVs, transforms and dissemination lists, and in the
  trained artefact.
- The in-memory join key is the unit position. Coordinates are matched with ROMP's tolerance, not by exact float
  equality (blending's `point_sources`).

---------------------------------------------------------------------------------------------------------
## 3. Multiple regions

**ROMP:**
- `region:` names a registry entry (R:onset/regions.py:31-48, :178-185) with: `domain` lat/lon box,
  `observations.dataset` plus `day_boundary`, `climatology.years`, default and also definitions, and `part_of`
  inheritance (:188-217).
- `params/region_def.domain` returns the box (R:params/region_def.py:9-56).
- `region_select` cuts that box and, by default, applies land and country masks named after the region.
- `detect_region` guesses the region from a file's extent (R:lib/dataset_detect.py:243-265).
- Only `ethiopia` and `india` have a `domain` today. africa, east_africa, west_africa, sahel, kenya, amazon and
  south_asia lack one, so they cannot be selected as a box yet.

**Blending:**
- One domain per config: `area_of_interest` (a bbox or GeoJSON), `dissemination`, `regrid`, `grid_ids`
  (B:config.py:223-233).
- Season dates are calendar month-days in one year: `season_start` 05-01 and `fixed_cutoff` 06-02
  (B:config.py:427-432), `issue_end` 07-31 (B:config.py:445).
- The season is the file's or the date's calendar year (B:core/io.py:47-50, :329-330, B:core/onsets.py:122-128),
  and the issue window is `start ≤ issue ≤ end` in the same year (B:forecast.py:112-116).
- Thresholds and reference-onset files are region-specific inputs.

**Needed so blending runs for any ROMP region:**
1. A blend config takes `region: <id>`. Its box replaces `area_of_interest.bbox`; observations and their
   `day_boundary` come from the region or dataset entry (replacing `date_marks`); climatology years come from
   the region; the default onset definition comes from `definitions_for`.
2. Add `domain:` to every region YAML that blending or benchmarking will use.
3. Add an optional `units:` block to the region entry or the run config (shapefile plus `unit_id_col`, or a
   weights CSV) so admin units are per region rather than per config.
4. Seasons that cross 1 January (amazon, southern Africa): blending's season-as-calendar-year logic must use
   ROMP's season model instead (`calendar.fixed_bounds` handles year-crossing ends,
   R:onset/calendar.py:29-42). `season_start`, `fixed_cutoff` and `issue_end` should come from the onset
   definition and season, not from free month-days.
5. Longitude seam and sign (west_africa straddles 0°, amazon is negative): use ROMP's `lon_swap` before any
   weight matching, and build or match weights after the swap.
6. Legacy box conflict: `region: India` (capital I) uses the legacy box 6.46–35.51 N, 68.11–91.4 E
   (R:params/region_def.py:30-34). `region: india` uses the registry box 6.5–38.5 N, 66.5–100 E (india.yaml).
   Retire the legacy names, or make blending always use the registry box.

---------------------------------------------------------------------------------------------------------
## 4. Dependencies

The blending core needs (B pyproject `dependencies`), and ROMP lacks (R:pyproject.toml `dependencies`):
- **scikit-learn** (LogisticRegression, B:blend/logit.py; and the forest) — missing in ROMP.
- **patsy** (formulas) — missing.
- **threadpoolctl** (`limit_blas_threads`, B:core/parallel.py:41) — missing.
- **joblib** — declared. In my slice it is only referenced by `blend/*`/`calibration` (grep hits in
  `blend/logit.py`, `core/calibration.py`, `blend/_crps_tree.py`). Verify whether it is a real import or comes
  through sklearn.
- **packaging** — declared. Verify use; it may be droppable.
- netCDF4 (`netcdf4`), numpy, pandas, scipy, xarray, pyyaml and numba are present in ROMP. Versions: blending
  wants numba≥0.60, ROMP pins ≥0.68.1 (fine). Blending relies on pandas≥2 copy-on-write detection
  (B:core/ids.py:215).
- **shapely**, **geopandas**, **pyproj** (blending's `regrid` extra; `shapely` is also imported by
  B:core/domain.py:407): ROMP has geopandas (which brings shapely and pyproj) and cartopy. OK.
- **scipy.sparse** (CSR in `SparseTransform`): already a ROMP dependency.
- Blending does not need zarr, regionmask or cartopy, but gains them for free.

---------------------------------------------------------------------------------------------------------
## 5. Risks: results that change if blending switches to ROMP's implementation

1. **Tenths-of-mm quantisation.** ROMP converts all rain to int32 tenths with round-half-even
   (R:onset/ingest.py:146-160, `np.rint`). Blending detects onsets on float mm (float64 for observations,
   float32 for forecasts, B:core/onsets.py:59-104, :636-660). Wet-day tests such as `gte 1.0` mm flip for
   0.95–0.999 mm. Window sums compared with thresholds shift by up to ±0.05 mm per day. Expect small onset-date
   and probability differences everywhere. Regridded unit means must be computed in float **before** ingest
   rounding, otherwise rounding happens twice (cells, then the mean).
2. **Negative values.** ROMP clips negative forecast values to 0 (`clip_negative=True`, R:onset/ingest.py:686).
   For observations it sets values below -0.05 mm to missing (R:onset/ingest.py:151). Blending keeps negatives
   (for example tiny negative differences in ECMWF tp), which lower window sums.
3. **Missing values and gaps.** Blending refuses gaps within a year (B:core/io.py:363-372). It has
   `missing_rain: keep|drop|zero` (B:config.py:241-243), and with `keep` a NaN propagates through the float
   onset search. ROMP fills gaps as missing (R:onset/ingest.py:294-299), and the engine treats missing as
   INSUFFICIENT_DATA statuses. `zero` and `drop` have no ROMP equivalent. They need a mapping to engine options
   or a pre-ingest fill.
4. **Day labels.** For observations labelled by start (CHIRPS), blending shifts observation dates +1 day
   (`date_marks: start`, B:config.py:245-252, B:core/io.py:376) and keeps forecast lead d at issue+d. ROMP keeps
   observation dates and labels forecast lead k as init+k−1 (`lead_label: start`, R:onset/engine.py:453-454),
   with the lead label derived from the observation boundary (R:lib/onset_config.py:416-423). The obs/forecast
   pairing is identical, but **every date is one day earlier in ROMP**. As a result:
   - onset dates and `onset_day` (B:core/onsets.py:138-139) are off by one;
   - `season_start` and `fixed_cutoff` apply one day differently (B:core/onsets.py:126-132, :503-525);
   - reference-onset comparisons shift.
   For end-labelled IMD (`03 UTC, labelled by end`) both conventions agree.
5. **Daily boundary.** ROMP can recut sub-daily forecasts at, say, 03 UTC (`daily_boundary`), moving init labels
   (R:onset/ingest.py:640-645, :683). Blending files are assumed to be daily already. Do not enable
   `daily_boundary` for migrated blends unless the training files are rebuilt.
6. **Accumulation.** ROMP's `auto` detection (R:onset/ingest.py:429-463) may difference or reject data that
   blending reads as daily totals. Blending only records `cell_methods` text (B:trained.py:125-140). Pin
   `accumulation: daily` for migrated datasets. `ingest_model_forecast` already defaults to daily when the key is
   absent (R:io/input.py:184).
7. **Units.** Undeclared units: blending applies `unit_scale` (default 1.0). ROMP raises, or detects from
   magnitude (dataset_detect). Declared "metres" or "meters" fail in ROMP today.
8. **Issue (init) set.** ROMP keeps only inits whose weekday is in `init_days` between `start_date` and
   `end_date` (R:io/input.py:461-478, :719-723). Blending uses every issue in the file within the season to
   `issue_end` (B:forecast.py:111-116, B:core/onsets.py). Different issue sets change the training samples. The
   blend must pass "all inits" or its own window.
9. **Leads.** ROMP truncates leads at the definition's reach (R:io/input.py:743-745). Blending reads up to
   `lead_days.max`, default 45 (B:config.py:256-258), for rain predictors and probability days. ROMP's reach must
   not cut the 28 probability days plus the follow window, or the rain horizon. ROMP also requires leads exactly
   1..n (R:onset/ingest.py:330-334); blending accepts a window starting at `min_day`.
10. **Latitude order and longitude convention.** ROMP flips latitude to ascending (R:utils/region.py:117-122) and
    may swap longitude to 0..360 or ±180 (:98-110). Blending keeps file order, and its regrid weights match
    sources by exact float (lat, lon) (B:core/domain.py:437-444, :546-551). After ROMP's swap, longitudes such as
    280.0 against -80.0, or float noise from `% 360`, will silently fail to match: those sources get -1 and the
    unit becomes NaN or unsupported. Match after normalising, with a tolerance.
11. **Implicit land and country masks.** ROMP `region_select` defaults to `land_only=True, shp_only=True`
    (R:utils/region.py:266-285). A blend run through it would drop coastal and border cells that blending kept
    (or that feed admin-unit weights), so unit means change. Polygon-unit runs must turn the implicit masks off
    for the source read and let the weights decide.
12. **Point-in-polygon edge rule.** Blending uses `contains_xy`, so a point on the boundary is out
    (B:core/domain.py:414). ROMP's `centre` rule uses `intersects_xy`, so it is in (R:utils/mask.py:184-187).
    Grid cells centred exactly on outline vertices or edges (common with 0.25° grids and simplified outlines)
    flip.
13. **Unit-mean weights.** Blending weights come from CSV (area overlap, renormalised over observed sources each
    day, B:core/domain.py:529-543). `engine.regional` uses cos-lat or equal weights and rounds the mean of tenths
    (R:onset/engine.py:925-933). Unifying on one transform changes one side's numbers unless the weights and
    rounding are specified.
14. **Member mean.** Blending reproduces a pandas Kahan member mean in the native dtype for `rain_mean`
    (B:core/onsets.py:613-637). Any ROMP-side member reduction on tenths would differ in the last digits, which
    feeds rain features.
15. **Year attribution.** Blending's `file_year` is the first 19xx or 20xx in the name (B:core/io.py:44-50).
    ROMP's year comes from `file_pattern` or the time coordinate. Files whose names contain other 19xx or 20xx
    numbers (resolutions, versions) are assigned differently.
16. **Grid-step signature.** Blending refuses a live grid with a different median step (B:trained.py:169-178). If
    ROMP's box cut or swap changes the read axes, the signature must be computed on the same post-swap axes at
    training and at forecast time, otherwise the check gives false refusals.
