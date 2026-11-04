"""The stacking of the per-case score tables of the skill pass (`hcwf.scoring.probabilistic._stacked`)."""
import pandas as pd

from hcwf.scoring.probabilistic import _stacked


def test_stacked_puts_model_and_window_first_and_renumbers_rows():
    one = pd.DataFrame({"score": [0.5], "model": ["AIFS"], "window": ["1-15"]})
    two = pd.DataFrame({"model": ["ECMWF"], "score": [0.25], "window": ["1-15"]})

    table = _stacked([one, two])

    assert list(table.columns) == ["model", "window", "score"]
    assert list(table.index) == [0, 1]
    assert table["model"].tolist() == ["AIFS", "ECMWF"]
    assert table["score"].tolist() == [0.5, 0.25]
