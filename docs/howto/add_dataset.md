# Add a dataset (NetCDF or zarr)

Observations, forecast models and the reference model are datasets. Describe each inline in the run file, with only what
cannot be detected:

```yaml
observations: {path: <obs folder>, variable: <rain variable>, units: mm}
models:
  - {name: <model>, path: <forecast folder or zarr store>, variable: <rain variable>, units: m, lead_label: end}
```

* `path` is a folder, a file, a glob or a zarr store; `paths` lists several files (names or globs, any number per year;
  `years` keeps those of some years; `unit_cvt` is the multiplier to mm of files that declare no units). Paths may use `~`
  and `${ENV_VAR}`.
* `variable`, `units` (`mm` or `m`), `dims`, `day_boundary` (observations), `lead_label`, `accumulation` and
  `daily_boundary` (forecasts) are detected from the first file when left out, and reported with the evidence. A key that
  cannot be detected reliably, or that the files contradict, is an error naming it.
* Observations and forecasts may be yearly NetCDF files or zarr stores holding all years.
* Entries can live in your own files, listed under `dataset_files:` and referenced with `dataset: <id>` (inline keys win).

Look at what was detected before running:

```bash
hcwf-run datasets -p run.yaml          # what each dataset resolves to, with detections, warnings, problems
hcwf-run init --dataset-template obs   # a blank dataset entry (or `forecast`)
```

How the files are read (chunks, alignment, zarr versus NetCDF) is explained in {doc}`../reading_data`. The keys of a
dataset are listed in the {doc}`../reference/config_keys` and in {py:class}`hcwf.config.Dataset`.
