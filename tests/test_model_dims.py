"""dim_fmt_model maps a forecast file's dims onto init_time/step/member."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from momp.utils.standard import dim_fmt_model, dim_fmt_model_ensemble


def _forecast(coord_order, member=None):
    """A tiny forecast shaped like the AIFS/GraphCast 0.25° archives."""
    coords = {
        "time": pd.to_datetime(["2000-05-01", "2000-05-05"]),
        "prediction_timedelta_daily": pd.to_timedelta([1, 2, 3], unit="D"),
        "lat": [10.0, 9.75],
        "lon": [70.0, 70.25],
    }
    dims = ["time", "prediction_timedelta_daily", "lat", "lon"]
    if member:
        coords[member] = [0, 1]
        dims.insert(1, member)
    data = np.zeros([len(coords[dim]) for dim in dims])
    ds = xr.Dataset({"tp": (dims, data)})
    return ds.assign_coords({name: coords[name] for name in coord_order + ([member] if member else [])})


@pytest.mark.parametrize(
    "coord_order",
    [
        ["time", "lon", "lat", "prediction_timedelta_daily"],
        ["prediction_timedelta_daily", "lat", "time", "lon"],
    ],
)
def test_lead_time_named_with_time_is_not_taken_for_the_start_date(coord_order):
    ds = dim_fmt_model(_forecast(coord_order))

    assert pd.api.types.is_datetime64_any_dtype(ds["init_time"])
    assert ds["step"].values.tolist() == [1, 2, 3]


def test_explicit_dims_are_used_instead_of_name_inference():
    ds = _forecast(["time", "prediction_timedelta_daily", "lat", "lon"]).rename(
        {"time": "issued", "prediction_timedelta_daily": "lead"}
    )

    ds = dim_fmt_model(ds, {"init_time": "issued", "step": "lead"})

    assert set(ds.tp.dims) == {"init_time", "step", "lat", "lon"}
    assert ds["step"].values.tolist() == [1, 2, 3]


def test_explicit_member_dim_is_renamed_for_ensembles():
    ds = _forecast(["time", "prediction_timedelta_daily", "lat", "lon"], member="realization")

    ds = dim_fmt_model_ensemble(
        ds, {"init_time": "time", "step": "prediction_timedelta_daily", "member": "realization"}
    )

    assert "member" in ds.tp.dims


def test_explicit_dims_missing_from_the_file_raise():
    ds = _forecast(["time", "prediction_timedelta_daily", "lat", "lon"])

    with pytest.raises(ValueError, match="lead"):
        dim_fmt_model(ds, {"step": "lead"})
