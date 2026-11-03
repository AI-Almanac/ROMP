# R3a: move table (old -> new)

Written against the tree after R1/R2 (commit 2abd5fe). Reconciles docs/review/layout.md section 2.1, which predates
R1/R2. Every move is `git mv` plus import rewrites; no shims. Nothing is left in `lib/`, `utils/`, `stats/`, `app/`,
`params/`, `graphics/`.

| old | new | note |
|---|---|---|
| cli.py | cli/__init__.py | `hcwf.cli:main` unchanged; `cli/__main__.py` added (`python -m hcwf.cli`) |
| driver.py | cli/driver.py | argv, batch, exit codes and `run_hcwf` are all CLI; one file, not split |
| lib/parser.py + lib/loader.py | cli/parser.py | M: loader was one 1 KB function (`_get_config_path_pre_parse`) |
| onset/cli.py | cli/onset.py | `python -m hcwf.onset` stays (3-line `onset/__main__.py`) |
| export.py | runs/export.py | whole file in R3a; writers split off in R3b (see below) |
| lib/run_folder.py | runs/folder.py | |
| lib/check.py | runs/check.py | the `--check` report |
| lib/resolved.py | runs/resolved.py | resolved_config.yaml text |
| lib/assertion.py | errors.py | `HCWFConfigError`; the subclasses stay next to their domain (see below) |
| lib/config_schema.py | config/schema.py | |
| lib/config_yaml.py | config/mapping.py | YAML/mapping -> RunConfig (kept apart from build.py: two different jobs, 400 vs 300 lines; `mapping.py` not `yaml.py` to avoid shadowing) |
| lib/dataset_schema.py | config/datasets.py | dataset description is configuration |
| lib/dataset_detect.py | config/detect.py | detection only exists to fill the config |
| lib/onset_config.py | config/onset.py | OnsetSetup |
| lib/lead_window.py | config/leads.py | |
| lib/init_config.py | config/init.py | |
| lib/convert.py, lib/convert_specs.py | config/convert.py, config/convert_specs.py | |
| lib/align_sample.py | onset/sample_alignment.py | |
| lib/explain_cells.py | onset/explain_cells.py | |
| stats/detect.py | onset/detect.py | run-level detection adapter |
| stats/observed_cache.py | onset/observed_cache.py | |
| stats/climatology.py | references/climatology.py | |
| stats/conditional_climatology.py | references/conditional.py | |
| onset/climatology_kde.py | references/kde.py | `onset/climatology.py` (thresholds) stays in onset/ |
| stats/benchmark.py | metrics/contingency.py (`onset_metrics`) + scoring/onset_tables.py (rest) | S, see below |
| stats/bins.py | metrics/bins.py (pure pairing) + scoring/pairs.py (readers) | S, see below |
| stats/score.py | metrics/score.py | pure |
| metrics/skill.py | metrics/skill.py (`create_score_results`) + scoring/pairs.py (`prepare_pairs` and its helpers) | S |
| stats/parallel.py + utils/workers.py | parallel.py | M (no name collisions) |
| utils/log.py | log.py | + `print_hcwf_banner` from printing.py |
| utils/printing.py | naming.py (tuple-to-string helpers) + log.py (banner) | S |
| utils/practical.py | io/paths.py | `set_dir` |
| utils/standard.py | io/dims.py | |
| utils/mask.py | spatial/masks.py | |
| utils/region.py | spatial/select.py | |
| params/region_def.py | spatial/region_def.py | `domain` wrapper and CMZ polygon; folded into onset/regions.py in R3b (see below) |
| utils/visual.py | figures/visual.py | |
| graphics/* | figures/* | |
| app/spatial_far_mr_mae.py | scoring/deterministic.py | |
| app/bin_skill_score.py | scoring/probabilistic.py | named after the forecast type, pairs with deterministic.py |
| app/onset_spatial.py + app/onset_time_series.py | figures/notebook.py | M |

Gone before R3a (not in the table): lib/convention.py, lib/control.py, utils/land_mask.py, onset/legacy.py.

Unchanged: api.py, cache.py, blend/, io/input.py (R3b), io/output.py (R3b), io/dict.py (R3b), spatial/{axes,ids,regrid,units}.py,
metrics/{probabilistic,error}.py, onset/{alignment,backends,calendar,definition,describe,engine,explain,ingest,model,
plan,regions,thresholds,units,viz,climatology}, config/{__init__,build,model,validate}.py. The onset registry and
regions YAML stay in onset/ (data of the onset component), so `pyproject.toml` package-data is unchanged.

## metrics vs scoring rule
`hcwf/metrics` is pure functions on arrays and tables (no reading, no detection, no RunConfig, no writing).
`hcwf/scoring` holds the per-case run flows (load, detect, build pairs, call metrics, hand over to writers/figures).
Splits made along it:
- metrics/skill.py: `create_score_results` stays; `PreparedPairs`, `prepare_pairs`, `_climatology_years`,
  `_climatology_pairs`, `_reference_model` -> scoring/pairs.py.
- stats/bins.py: `LeadBins`, `extract_day_range`, `get_target_bins`, `forecast_pairs`, `_year_set_texts`,
  `create_climatological_forecast_obs_pairs` -> metrics/bins.py; `reports_back_dated`, `_case_of`,
  `forecast_obs_pairs`, `_pairs_of_years`, `start_pairs`, `climatological_pairs` -> scoring/pairs.py.
- stats/benchmark.py: `onset_metrics` -> metrics/contingency.py; `_ONSET_TABLES`, `reset_onset_tables`,
  `_family_key`, `_tables_of_years`, `onset_tables`, `_metrics_of_tables`, `compute_metrics` -> scoring/onset_tables.py.

Known leftovers (not R3a): metrics/bins.py is still imported by io/output.py and figures/heatmap.py (for
`get_target_bins`, pure); error subclasses are not yet gathered in errors.py; io/dict.py has two helpers
destined for metrics; 6 undefined names (dead code) in io/output.py and spatial/select.py predate R3a.


# R3b: move table (R3a tree -> now)

Same rules: `git mv`/split plus import rewrites, no shims, no behaviour change. The one place a re-export would have been
convenient (`hcwf.io.input` as the single public reader module) was rejected: the typed API names are unchanged, only their
module is, and every caller imports the module that owns the name.

## io/input.py (1521 lines) -> io/ and spatial/
| name(s) | new module | note |
|---|---|---|
| `open_store`, `year_files`, `observation_source` (was `_obs_source`), `forecast_sources` (was `_forecast_sources`), `open_forecast` (was `_open_forecast`), `fix_dims_for` | io/stores.py | where the data is and how a file or store is opened; nothing decoded |
| `load_observations`, `read_obs_years`, `observed_history`, `observations_of`, `observation_years_of`, `history_of`, the held-years cache (`forget_years`), `ObsRead` | io/observations.py | `_read_obs_year`, `_obs_key` ... stay private |
| `release_reference`, `reference_held`, `reference_rain_year`, `reference_rain`, `LazyRain`, `clear_obs_cache` | io/reference.py | the reference (threshold and climatology) years ingested once; `clear_obs_cache` = `forget_years` + `release_reference` (it lives here so that observations.py does not import reference.py) |
| `load_forecast`, `forecast_of`, `ingest_forecast`, `initialization_dates`, `steps_needed`, `select_leads`, `select_members`, `UnitForecast`, `unit_forecast_bytes`, `ForecastRead` | io/forecasts.py | |
| `load_thresholds`, `load_onset_thresholds`, `thresholds_of` and the model-threshold helpers | io/thresholds.py | |
| `warm_up`, `warm_up_of` | io/warmup.py | thresholds and cell masks built once before the years go to workers |
| `unit_transform`, `feeding_cells`, `unit_series`, `read_on_units`, `read_unit_dim`, `read_units`, `select_units`, `declared_scale`, `unit_table`, `units_spec`, `units_files`, `point_dims`, `SOURCE_POLICIES` | spatial/onto_units.py | cells -> unit series; `unit_table` etc. are the run's evaluation units |
| `in_memory`, `scale_in_place` (were `_in_memory`, `_scale_in_place`) | spatial/select.py | next to `load_blocks` |
| `NO_SELECTION` (was `_NO_SELECTION` in input.py) | config/model.py | next to `Selection` |

Names that crossed a module boundary lost their leading underscore (`ObsRead`, `ForecastRead`, `obs_years_of`,
`observation_year`, `observation_year_once`, `years_held`, `mask_text`, `LazyRain`). `is_zarr` was never io.input's: callers
import it from config.datasets.

## writers: io/output.py + runs/export.py
There was one real overlap (NetCDF attribute cleaning, written twice with the same rules; `with_unit_labels`) so the two
writers are one module, `io/output.py`: score CSVs, spatial-metrics and climatological-onset NetCDF, per-unit tables, and the CF
onset-date files (`write_netcdf`, `open_onsets`, `date_var`, `float_var`, `status_var`, `coord_attrs`, `days_of`, `to_dates`,
`day_of_year`, `standard_calendar`, `lead_comment`, `DATE_UNITS`, `DATE_FILL`, `EPOCH`). `netcdf_attrs(attrs, none="drop"|"text")`
replaces `export._clean` and the inline loop of `save_metrics_to_netcdf`.
- runs/export.py keeps the collection only: `collect_*`, `_provenance`, `export_onsets`, `default_sources`, `onsets_folder`.
- The pure half of the old io/output.py -> metrics/tables.py: `score_tables`, `back_dated_member_share`, `add_back_dated_map`.
- `open_onsets` is now `hcwf.io.output.open_onsets` (docs updated).

## the rest
- io/dict.py deleted. `extract_pd_bins` -> metrics/bins.py (it pairs bins); `extract_binned_dict` -> figures/panel_portrait_skill.py,
  `extract_overall_dict` -> figures/panel_bar_skill.py, `nested_dict_to_array` (from io/output.py) -> figures/panel_portrait_error.py
  (each is read by that one panel); `select_key_at_level` -> scoring/probabilistic.py; `set_nested` (from io/output.py) ->
  scoring/deterministic.py.
- spatial/region_def.py deleted: `domain` was a one-line wrapper of `onset.regions.box` (callers use `box`);
  `polygon_boundary` (India CMZ outline for 1/2/4 degree grids) -> onset/regions.py, unchanged. Moving the coordinates into the
  region YAML was not done: it needs a schema key and a loader for three grids and would change a data format, not only layout.
- errors: all 14 error classes are in errors.py (see its docstring for the hierarchy). errors.py has no imports, so any module can
  import it without a cycle, which is why they could all move; leaving each next to its module would have kept `config.schema`
  importable only through half of the config package. `spatial/__init__` no longer re-exports `AreaError`/`UnitsSpecError`.
  `IncompatibleForecast` (ValueError) and `RunExistsError` (FileExistsError) keep their bases.
- `get_target_bins` -> metrics/score.py: it reads score-result dicts (`bin_fair_brier_scores`), not pairs, so it is not
  bins.py; the three users (metrics/score, metrics/tables, figures/heatmap) now import it from metrics.score. `extract_day_range`
  (one caller) was dropped for `LeadBins.first_day_of`.
- dead code removed: `analyze_nested_dict` (`dict_depth`/`dict_dims` never existed) in io/output.py; `swap_lon_axis` in
  spatial/select.py (a half port of xcdat calling three undefined helpers). `lon_swap`'s branch for a box across the data's
  longitude seam called it and always raised NameError; it now raises a ValueError that says so.
