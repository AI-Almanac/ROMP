# Onset in ROMP

ROMP detects onset with the `momp.onset` component (contract: `docs/onset_spec.md`). `momp/stats/detect.py` is a thin adapter: it runs one definition on the observations and on every forecast and shapes the results for the metrics code.

## Quick start

For a known region you need no onset settings at all. With `region: Ethiopia` in your run YAML (`examples/config.yaml`), ROMP uses ICPAC's definition, CHIRPS's day boundary and the region's threshold years, and prints each choice and where it came from at the start of the run.

```
momp-run -p run.yaml
```

To choose yourself, set `onset:` in the run file:

```yaml
region: Ethiopia
onset:
  definition: icpac            # registry id, or the path of a YAML file
  override: {wet_day: at least 2 mm}   # optional; merged like `extends`, same strict errors
```

or a complete definition inline under `onset: {rule: {...}}`. The run prints the definition's description, `icpac v1, overridden: wet_day` and the hash, and writes `resolved_definition.yaml` and `resolved_config.yaml` to `dir_out`.

Look definitions up with the CLI:

```
python -m momp.onset list                 # definitions
python -m momp.onset describe icpac       # the rule in words, hash, status, notes
python -m momp.onset regions              # regions with their defaults
python -m momp.onset schematic icpac --png icpac.png
```

A rule that differs a little goes in `onset.override`, or in a shared YAML file with `extends: <id>` (set `definition` to its path). A region you define yourself in `params/region_def.py` has no defaults: set `onset_definition` and `obs_day_boundary`.

## Onset keys (the `onset:` section; flat names in brackets)

Every key can stay unset. Names below are the flat keys; in YAML write them inside `onset:` without the `onset_` prefix (`definition`, `override`, `rule`, `season`, `backend`, `percentage_threshold`, `lead_label`), `obs_day_boundary` is `observations.day_boundary`, `model_lead_label` is `onset.lead_label`. Order of use: the config value, then the region's default, then an error naming the key. Region names are case-insensitive and may be aliases (`India`, `ethiopia`); a country inherits the defaults of the regions that contain it.

| Key | Default | Meaning |
|---|---|---|
| `onset_definition` | region default | Registry id or YAML path. |
| `onset_override` / `onset_rule` | none | Changes merged into the definition, or a complete inline definition (not both with `onset_definition` for `rule`). |
| `onset_season` | first | Season name for definitions with several; `"1"` or `"2"` for derived seasons (`dunning`). |
| `threshold_years` | region default | `(first, last)` years of observations that thresholds are computed from. |
| `thresholds` | `None` | `{name: "path.nc:variable"}`: precomputed thresholds instead of computing them. |
| `threshold_source` | `"observations"` | `"observations"` or `"model"` (see below). |
| `obs_day_boundary` | region default | E.g. `"00 UTC, labelled by start"`. |
| `model_lead_label` | from the boundary | `"start"`: lead k is dated init + k - 1. `"end"`: init + k. |
| `check_alignment` | `True` | Check for day shifts at the start of a run. |
| `dry_spell_check` | `True` | `False` drops the definition's `reject if` vetoes. |
| `forecast_history` | `"observed"` | `"observed"` or `"none"` (see below). |
| `onset_backend` | `"numba"` | `"numba"`, `"reference"` or `"gpu"`; all give identical results. Falls back to `"reference"` with a warning if numba or a GPU is missing. |
| `onset_percentage_threshold` | `0.5` | Share of ensemble members that must have an onset for the ensemble to have one. |
| `explain_cells` | `None` | List of `(lat, lon)`. For each cell and year, writes an anatomy PNG and a `.txt` explain log of the observed onset to `dir_fig/explain/`. |

Thresholds (for example the local mean 5-day total) are computed from the observations of `threshold_years` and cached in `dir_out/.onset_cache`. Derived seasons (`dunning`) also compute a harmonic season analysis from those years.

`onset_definition` together with any old flat key (`wet_init`, `wet_threshold`, `wet_spell`, `dry_threshold`, `dry_spell`, `dry_extent`, `thresh_file`, `thresh_var`, `earliest_onset_date`) is an error. Put the change in a definition with `extends:`.

Results carry an `onset_status` column (`ONSET`, `NOT_FOUND`, `CENSORED`, `INSUFFICIENT_DATA`, and `PRIOR` in forecast history, below) and the run prints status counts. `CENSORED` cannot occur: the lead window is shortened so that the whole rule fits.

## Lead window

A candidate on lead k needs leads up to k + R - 1, where R is how many days from the onset day the definition reads (ICPAC 22; Moron-Robertson 35; without vetoes, the window length). At the start of a run ROMP counts the leads in each model's first-year file and searches candidates on leads up to n_leads - R + 1.

If that is below `max_forecast_day`, then `max_forecast_day`, `verification_window_list`, `tolerance_days_list` and `day_bins` are clipped (windows and bins that start later are dropped), and ROMP prints, for example:

```
Lead window      : 1-29 for AIFS (icpac reads 22 days from the onset day; forecasts have 50 leads)  [shortened from max_forecast_day 30]
```

The run uses the shortest window over its models; it is stored as the `lead_window` attribute of the spatial-metric NetCDFs. If no lead is usable ROMP stops and suggests `dry_spell_check = False`, which shrinks R. Legacy mode keeps the "Not enough forecast time steps" error.

## Dry-spell check

`dry_spell_check = False` drops the `reject if` vetoes, such as Moron-Robertson's 10-day dry window or ICPAC's 7-day dry run. It suits demos and forecasts too short for the veto window. The run prints `Dry-spell check  : off`, and the title, notes, YAML, sentence and hash in the outputs are those of the variant. A definition without vetoes (or a curve rule) is left alone (`none in this definition`). In legacy mode it has the effect of `dry_extent = 0`.

In code: `momp.onset.definition.without_dry_spell_check(defn)`. On the CLI: `python -m momp.onset describe <id> --no-dry-spell-check`. Generated configs: `ROMP_DRY_SPELL_CHECK`.

## Forecast history

`forecast_history = "observed"` (the default for definitions) searches each forecast together with the observed rain up to the day before its first lead. A wet spell that began in the observations can be completed by the forecast, and a season that started before the init is searched from its start. Nothing observed on or after lead 1's date is read. ROMP also loads the previous year's observations when the season wraps over New Year or the rule looks back past 1 January.

`"none"` searches the forecast leads alone, as earlier runs did. Legacy configs always use it.

* `PRIOR`: the observations alone already confirm the onset. Nothing is scored for those inits: inits on or after the cell's observed onset produce no rows (a `PRIOR` result in a scored row is counted, warned about and reported as no onset).
* A forecast may complete an onset that began before the init. The row then has `onset_day <= 0` (days after the init; the date lies in the observed history) and status `ONSET`. These are *back-dated* onsets.

### Scoring back-dated onsets

* **Deterministic and ensemble-mean tables** (`compute_onset_metrics_with_windows`): the onset counts as falling on the first lead for window membership. In a window starting at lead 1 it is a hit if its real date is within the tolerance of the observed onset, otherwise a false alarm. In windows starting later it is ignored, like any forecast onset before the window. MAE uses the real date. (Earlier versions dropped such onsets from every window; this slightly changes `"none"` runs with lead label "start".)
* **Ensemble probability bins**: members with `onset_day < 1` fall in the "Before day 1" bin, where the observed onset (always after the init in scored rows) is 0, so the Brier score penalises them.

How many were back-dated is reported (definition mode with `"observed"` only):

* Spatial-metric NetCDFs: the per-cell variable `n_back_dated` counts scored init cases with a back-dated forecast onset, summed over years with an observed onset. The global attribute `n_back_dated_total` sums it over the grid. Only windows starting at lead 1 count; others hold 0. For ensembles it is the ensemble-mean onset. The climatology baseline has none.
* `overall_skill_scores_<model>_<window>.csv` has a column `back_dated_member_share`: the share of scored member onsets (members with an onset, in cases initialised before the observed onset) dated on or before the init. NaN when no member found an onset.

## Threshold source

`threshold_source = "observations"` computes thresholds from the reference observations. `"model"` computes them from the model's own forecasts: the mean over leads 1 to `max_forecast_day`, over forecast days whose date lies in the threshold's `during` period in `threshold_years`, using ROMP's forecast loader (init days, unit conversion). Each verification window gets its own threshold, pooled from the forecast days at leads inside that window.

Only mean-based thresholds (`mean total over`, `mean daily rain`) support `"model"`; others are an error naming them. Results are cached in `dir_out/.onset_cache` by model, years, lead window and definition hash. Legacy configs use observations only.

## Alignment check

Once per model at the start of a run, ROMP compares the first year's forecasts (leads 1-3) with the observations at lags -3 to +3 and prints the result. If the forecasts look shifted by a day it warns; the usual cause is a wrong `model_lead_label` or `obs_day_boundary`. The check never stops a run; `check_alignment = False` skips it. The same check is available as `python -m momp.onset check-alignment` (see `--help`).

Finding on AIFS: CHIRPS v3 (00 UTC day, labelled by start) lines up with `"start"` (lead k is init + k - 1); IMD (day ending at 03 UTC, labelled by end) lines up with `"end"`. ROMP used to assume `"end"` always, so Ethiopia forecast onsets were one day late against CHIRPS.

## Exporting onset dates

`momp-run export -p run.yaml [--sources observed,AIFS,climatology,conditional_climatology] [--out DIR]` runs only the detection (no scoring) and writes one NetCDF per source; `output.save_onsets: true` in the run file writes them to `dir_out/onsets/` before a normal run scores (the detection runs twice). In Python: `momp.export.export_onsets(cfg, sources=None, out=None)` returns `{source: path}`; `momp.export.open_onsets(path)` opens a file (see below). Without `--sources` you get observed, every model and the climatology reference the run uses.

The values are the ones ROMP scores: the export calls the scoring's own functions with the keyword arguments the scoring builds (masks, shortened lead window, `forecast_history`, thresholds, definition). Unlike the scored tables, which keep only cases with an observed onset and an init before it, the files hold every init and cell; the flag `scored` (1 for those cases) marks what ROMP scores. Dates are CF times (int32 days since 1970-01-01, with the data's calendar; cftime for non-standard calendars); no onset is the fill value (NaT) and NaN in the float variables. Onset date, day of year and lead are set only where `status` is `ONSET`: a `PRIOR` search is reported as no onset, as scored. Files are zlib-compressed NetCDF4, CF-1.8, with global attributes holding the definition (id, version, hash, sentence, YAML, overrides), thresholds (source, years, files, values hash), mask summary, forecast history, lead label and window, day boundaries, dataset paths, ROMP version, creation time and the resolved run config (`resolved_config_yaml`).

| File | Variable | Dims | Meaning |
|---|---|---|---|
| `onsets_observed.nc` | `onset_date`, `onset_doy`, `status` | `year, season, lat, lon` | Observed onset (date; day of the calendar year; `status` int8 with `flag_values`/`flag_meanings` ONSET, NOT_FOUND, CENSORED, PRIOR, INSUFFICIENT_DATA). |
| `onsets_<model>.nc` | `onset_date`, `onset_doy`, `onset_lead`, `onset_day`, `status` | `init_time, [member], season, lat, lon` | Forecast onset. `onset_lead` is the engine's 1-based lead (NaN none; 0 or below: back-dated into the observed history); `onset_day` is days from the init date to the onset date, as the scored tables' `onset_day`. |
| | `scored` | `init_time, lat, lon` | 1 where ROMP scores the case. |
| | `members_by_status` | `status_code, init_time, season, lat, lon` | Ensembles: members per status. |
| | `onset_probability` | `init_time, season, lead, lat, lon` | Ensembles: share of members with onset on or before each lead (lead 0 to the lead window; `lead` has units days). |
| | `ensemble_onset_date`, `ensemble_onset_doy`, `ensemble_onset_day`, `ensemble_status` | `init_time, season, lat, lon` | Ensembles: the ensemble onset ROMP scores (mean member day rounded half to even, needs `onset_percentage_threshold` of the members; this is not a median). |
| `onsets_climatology.nc` | `climatological_onset_doy`, `climatological_onset_date` | `season, lat, lon` | Rounded mean day of the climatology years' onsets; the date places it in the first evaluated year (nominal). |
| | `onset_date`, `onset_doy`, `onset_lead`, `onset_day`, `status`, `scored` | `init_time, season, lat, lon` | The climatology as a forecast, as scored: its date if within the lead window, else NOT_FOUND. Lead k is init + k. |
| `onsets_conditional_climatology.nc` | `onset_date`, `onset_doy`, `onset_lead`, `onset_day`, `status`, `scored` | `init_time, season, lat, lon` | The conditional median onset (first lead whose conditional CDF reaches 0.5). |
| | `onset_probability` | `init_time, season, lead, lat, lon` | The conditional CDF of the onset over leads 0 to the lead window (NaN for cells without a KDE). `climatology_cv` records the mode the scoring used. |

xarray cannot decode masked dates on a non-standard calendar and decodes `units: days` as a timedelta, so open files with `momp.export.open_onsets(path)`, which gives cftime objects (None for no onset) and plain lead numbers.

## Looking inside a result

* `explain_cells` (above) writes one anatomy plot and explain log per cell and year during a run.
* `python -m momp.onset explain ...` does the same for one cell and year from the command line.
* In Python, `momp.onset.viz` has `anatomy_plot`, `schematic`, `compare_definitions`, `forecast_vs_truth` and `drill_down`; `momp.onset.explain` has `trace_observed`, `trace_forecast` and `explain`.

## Region masks

The `mask:` section (`momp/utils/mask.py`, schema in `momp/lib/config_schema.py`) chooses the cells that are evaluated. `Config(mask=Mask(...))` in Python is the same thing.

| key | meaning |
|---|---|
| `land` | land cells only (cartopy Natural Earth land polygons) |
| `country` | a name or list; matched on Natural Earth `NAME`, `NAME_LONG` or `ADM0_A3`, any case; an unknown name lists the closest ones |
| `file` | a 0/1 NetCDF (`variable` auto-detected if the file has only one, `keep_value` default 1), or a shapefile / GeoJSON (any CRS) |
| `condition` | `{file, variable, above, below}`: a field thresholded on its own grid (`above` and `below` are strict) |
| `cell_rule` | `centre` (default): the cell centre is inside the shape; `overlap_fraction`: the share of the cell's area (planar, degrees squared) inside is at least `overlap_fraction` (default 0.5) |
| `resolution` | Natural Earth `10m`, `50m` (default) or `110m` |

All given parts intersect. Shapes (land, countries, a shapefile) are intersected as geometry and rasterised once; mask files and conditions are aligned to the data grid and combined with the shape cells.

**Alignment.** The final mask is built separately for the observations and for each model, on that dataset's own grid, and rainfall is never regridded. For a mask file ROMP detects the latitude and longitude names, flips a north-to-south latitude, wraps the longitudes to the data's convention (0-360 or -180-180) and compares resolution and cell centres. If every data cell centre matches a mask cell centre within a hundredth of a cell and the cell sizes agree, the mask is used directly. Otherwise the mask is regridded onto the data grid: a finer mask keeps a data cell when the area share covered by kept mask cells is at least `overlap_fraction`; a coarser (or equally fine but offset) mask gives each data cell the value of the nearest mask cell. A mask that does not cover the data extent stops the run with both extents in the message. A shape cell on the antimeridian is tested by its centre longitude wrapped into -180..180.

**What is printed and recorded.** Per dataset the run prints the sources, the rule, every alignment action (flip, longitude shift, regrid method with both resolutions) and the kept cell count, and saves `mask_<dataset>.png` (kept cells) in `output.figure_dir`. `resolved_config.yaml` carries the `mask:` section and, as comments at the top, the sources with SHA-256 content hashes, the rule, the actions and the kept counts per dataset; the spatial-metrics and climatology NetCDF files get `mask_cell_rule`, `mask_sources` and `mask_kept_cells_<dataset>` attributes. A mask that keeps no cell is an error.

**Natural Earth data** is fetched by cartopy on first use. Offline, ROMP stops with a message saying so; run `python scripts/prefetch_cartopy_data.py` once while online (the Docker image does this at build time) or set `CARTOPY_DATA_DIR`.

**Old keys.** Without `mask:`, the flat `nc_mask` (and the implicit land and country masks of `region_select`) behave exactly as before. `mask` and `nc_mask` together are a config error; `mask: {file: x.nc}` replaces `nc_mask: x.nc` and does not apply the implicit land and country masks.

## Legacy mode

A YAML config with the old flat keys at the top level and no `onset.definition` runs as before and raises a `FutureWarning`; `momp-run convert old.in` produces such a YAML from an old config.in. `scripts/generate_config.py` writes such configs (YAML) unless an onset environment variable is set. The keys are translated by `momp.onset.legacy.from_flat_config`. Lead k of a forecast is dated init + k (`"end"`). Day boundaries are only needed for the alignment check, which is skipped with a note when they are unknown.

Translation: a wet day has at least `wet_init` mm. The search runs from `earliest_onset_date` (or the month and day of `start_date`) to the month and day of `end_date` plus 47 days minus `wet_spell - 1`. A candidate needs all `wet_spell` days wet and a total above the threshold. With `dry_extent > wet_spell` and `dry_extent >= dry_spell`, a dry run of `dry_spell` days within days 2 to `dry_extent + 1` after the candidate cancels it.

Deliberate differences from the old code:

* The forecast dry window is one day longer: the observed window (days 2 to `dry_extent + 1`) is now used for both.
* Missing (NaN) days in a window stop the search with `INSUFFICIENT_DATA` instead of being skipped.
* Input is rounded to 0.1 mm; `wet_init` and `wet_threshold` must be whole tenths of a mm.
* Negative forecast values are set to 0 (the old code summed them as negative rain).
* `dry_threshold` was never used and still is not; `fallback_date` is no longer used.
* A veto window may read observations beyond `end_date + 47` days (the old code cut the data there and treated unseen days as having no dry spell).
* A series that starts after the search start gives `INSUFFICIENT_DATA`.
* Forecast candidates are limited to the season.
