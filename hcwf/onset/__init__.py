"""Precipitation-only onset detection: definitions, ingestion, climatology and the detection backends.

The package follows ``docs/onset_spec.md``. `hcwf.onset.model` holds the types, `hcwf.onset.definition` and
`hcwf.onset.units` parse definitions into them, `hcwf.onset.ingest` turns rainfall into `hcwf.onset.model.DailyRain`
and `hcwf.onset.model.ForecastRain`, `hcwf.onset.engine` runs a definition on them through a backend of
`hcwf.onset.backends`, and `hcwf.onset.detect` drives observed and forecast detection.
"""
