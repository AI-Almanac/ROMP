"""The module-level helpers of hcwf.config.convert_specs that read one legacy spec or climatology run."""
from hcwf.config import convert_specs as C


def test_the_grid_id_convention_of_a_spec_is_read_at_the_top_or_under_geometry():
    assert C._grid_id_convention(None) is None
    assert C._grid_id_convention(C._Spec("a", {}, [])) is None
    top = C._Spec("a", {"grid_id_decimal_digits": 2}, [])
    assert C._grid_id_convention(top) == (2, "trimmed")
    assert "grid_id_decimal_digits" in top.read
    nested = C._Spec("regrid", {"geometry": {"grid_id_decimal_digits": 3, "grid_id_format": "FIXED"}}, [])
    assert C._grid_id_convention(nested) == (3, "fixed")


def test_the_training_years_of_a_climatology_run_are_its_two_limits():
    assert C._train_years({"train_year_min": 2000, "train_year_max": 2010}) == (2000, 2010)
    assert C._train_years({}) == (None, None)


def test_the_shared_settings_of_a_climatology_run_have_their_defaults():
    run = {"season_start_md": "05-01", "issue_end_md": "07-01", "forecast_window": 28, "test_year_min": 2011,
           "horizons": [{"start_md": "05-01", "end_md": "05-31", "forecast_window": "21"}]}
    assert C._climatology_settings(run) == {
        "onset_col": "onset_day", "season_start": "05-01", "issue_end": "07-01", "forecast_window": 28,
        "horizons": [{"start": "05-01", "end": "05-31", "forecast_window": 21}], "cv_by_year": True,
        "min_onset_years": 10, "test_years": (2011, None)}
    assert C._climatology_settings({})["horizons"] is None
