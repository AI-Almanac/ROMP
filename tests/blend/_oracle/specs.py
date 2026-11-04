# TEST-ONLY ORACLE: copied unchanged from onset_blending@1e6d696 tests/_specs.py (needs the onset_blending checkout on sys.path).
# Never import this from hcwf.
"""The legacy stage specs a BlendConfig stands for: what the legacy
functions (tests/_legacy) read, so the tests can run them as the oracle on
the same config. The package itself reads the config."""

from __future__ import annotations

from onset_blending import config as cfg
from onset_blending.config import BlendConfig

# Dimension names the readers know, mapped to theirs; the config's renames win.
_OBSERVATION_DIMS = {
    "TIME": "time", "time": "time",
    "LATITUDE": "lat", "latitude": "lat", "lat": "lat",
    "LONGITUDE": "lon", "longitude": "lon", "lon": "lon",
    "ADM3_NAME": "adm3_name", "adm3_name": "adm3_name", "adm3": "adm3_name",
}
_FORECAST_DIMS = {
    "time": "time", "day": "day", "lat": "lat", "lon": "lon",
    "adm3_name": "adm3_name", "adm3": "adm3_name", "number": "number", "sample": "number",
}
# The forecast probability series the connect stage bins, by config variant.
_CONNECT_VARIANT = {"fixed_cutoff": "fixed_cutoff", "ref_onset": "ref"}
CLIMATOLOGY = "clim"
UNCONDITIONAL_CLIMATOLOGY = "clim_unc"


def _thresholds(config: BlendConfig) -> dict:
    threshold = config.onset.threshold
    if threshold.mm is not None:
        return {"constant": float(threshold.mm)}
    out = {
        "file": str(config.path(threshold.file)),
        "thresh_col": threshold.value_col,
        "lat_col": threshold.lat_col,
        "lon_col": threshold.lon_col,
        "lat_var": threshold.lat_var,
        "lon_var": threshold.lon_var,
        "thresh_var": threshold.value_var,
        "mat_lat_name": threshold.mat_lat_name,
        "mat_lon_name": threshold.mat_lon_name,
        "mat_thresh_name": threshold.mat_value_name,
    }
    if threshold.id_col:
        out["id_col"] = threshold.id_col
    if threshold.scale is not None:
        scale = threshold.scale
        out["rule"] = {"type": "scale", "factor": scale.factor, "offset": scale.offset,
                       "min": scale.min, "max": scale.max}
    return out


def _reference_onset(config: BlendConfig) -> dict | None:
    reference = config.onset.reference_onset
    if reference is None:
        return None
    if reference.month_day is not None:
        return {"constant_month_day": reference.month_day}
    out = {"file": str(config.path(reference.file))}
    for key in ("date_col", "year_col", "day_col", "base_date", "unit_col"):
        if getattr(reference, key) is not None:
            out[key] = getattr(reference, key)
    return out


def _geometry(config: BlendConfig) -> dict:
    ids = config.domain.grid_ids
    if ids.decimal_digits is None:
        return {}
    return {"geometry": {"grid_id_decimal_digits": ids.decimal_digits, "grid_id_format": ids.format}}


def _common(config: BlendConfig) -> dict:
    out = {"thresholds": _thresholds(config), **_geometry(config)}
    reference = _reference_onset(config)
    if reference is not None:
        out["ref_onset"] = reference
    return out


def _cell_transform(config: BlendConfig, path: str | None) -> dict:
    if path is None:
        return {"cell_transform_enabled": False}
    return {"cell_transform_enabled": True, "cell_transform_file": str(config.path(path))}


def _cutoff_variant(config: BlendConfig) -> str:
    return cfg.cutoff_variant(config)


def cv_spec(config: BlendConfig) -> dict:
    evaluation = config.evaluation
    years = evaluation.years
    feature = cfg.CLIMATOLOGY_FEATURE
    baselines = evaluation.climatology_baselines

    clim_logits = [{
        "name": "unc_clim_raw",
        "base_col_prefix": f"{feature}_unc",
        "earlier_col": f"{feature}_unc_earlier",
        "earlier_is_logit": True,
    }]
    # Today's evaluation reads clim_raw as start year 1900, and windowed
    # climatologies as start years sharing one end year per entry.
    starts_by_end: dict[int, list[int]] = {}
    if baselines.conditional:
        starts_by_end[max([*years.training, *years.cv_holdout])] = [1900]
    for start, end in baselines.windows:
        starts_by_end.setdefault(end, []).append(start)
    clim_logits += [
        {"name": "clim_raw", "base_col_prefix": feature, "window_start_years": starts, "window_end_year": end}
        for end, starts in starts_by_end.items()
    ]

    window_variants = []
    for variant in evaluation.window_variants:
        by_end: dict[int, list[int]] = {}
        for start, end in variant.windows:
            by_end.setdefault(end, []).append(start)
        for end, starts in by_end.items():
            entry = {
                "enabled": True,
                "base_name": variant.base,
                "start_years": starts,
                "end_year": end,
                "replace": {"from": f"{variant.replace}_qx", "to": f"{variant.replace}{{window}}_qx"},
            }
            if variant.only_if_cutoff_mode:
                entry["only_if_cutoff_mode"] = variant.only_if_cutoff_mode
            window_variants.append(entry)

    mme = evaluation.mme
    if mme.enabled and mme.optimize == "always":
        raise NotImplementedError("the legacy cv spec cannot express evaluation.mme.optimize 'always'")
    spec = {
        "n_bins": config.features.bins.count,
        "run": {
            "cutoff_mode": evaluation.cutoff_mode,
            "training_years": list(years.training),
            "cv_holdout_years": list(years.cv_holdout),
            "true_holdout_years": list(years.true_holdout),
        },
        **(
            {"cell": {"dissemination": str(config.path(config.domain.dissemination.file)),
                      **({"id_col": config.domain.dissemination.id_col} if config.domain.dissemination.id_col else {})}}
            if config.domain.dissemination is not None
            else {}
        ),
        "cv": {
            "methods": ["cluster" if m == "clusters" else m for m in evaluation.cv_methods],
            "cluster_var": evaluation.cluster_var,
        },
        "models": {
            "formulas": {
                name: {"enabled": formula.enabled, "text": formula.text}
                for name, formula in evaluation.formulas.items()
            },
            "window_variants": window_variants or {"enabled": False},
        },
        "mme": {
            "enabled": mme.enabled,
            "variants": list(mme.variants),
            "blend_models": [
                {"name": s.name, "source": s.source, "cal_variant": s.cal_variant} for s in mme.sources
            ],
            "optimize_if_holdout_is_full_2000_2024": mme.optimize != "never",
        },
        "extras": {
            "clim_logits": clim_logits,
            "forecasts": [
                {
                    "name": s.model,
                    "variant": s.variant or _cutoff_variant(config),
                    "raw": s.raw,
                    "calibrated": s.calibrated,
                    "export_platt_weights": s.platt_weights,
                }
                for s in config.forecast_evaluations()
            ],
            "forecast_variants": {v: cfg.VARIANT_SUFFIX[v] for v in ("base", "fixed_cutoff", "ref_onset")},
            **(
                {"external_predictions": [
                    {"name": e.name, "file": e.file, "method": e.method}
                    for e in evaluation.external_predictions
                ]}
                if evaluation.external_predictions
                else {}
            ),
        },
        **_geometry(config),
    }
    return spec
