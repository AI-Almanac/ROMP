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
| export.py | runs/export.py | whole file; the split into io/onsets + scoring/onsets is later (R3b) |
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
| params/region_def.py | spatial/region_def.py | `domain` wrapper and CMZ polygon; folding it into onset/regions.py is later |
| utils/visual.py | figures/visual.py | |
| graphics/* | figures/* | |
| app/spatial_far_mr_mae.py | scoring/deterministic.py | |
| app/bin_skill_score.py | scoring/probabilistic.py | named after the forecast type, pairs with deterministic.py |
| app/onset_spatial.py + app/onset_time_series.py | figures/notebook.py | M |

Gone before R3a (not in the table): lib/convention.py, lib/control.py, utils/land_mask.py, onset/legacy.py.

Unchanged: api.py, cache.py, blend/, io/input.py (R3b), io/output.py, io/dict.py, spatial/{axes,ids,regrid,units}.py,
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
