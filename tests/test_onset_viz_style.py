"""momp.onset.viz.style: every status colour is legible (3:1) against the surface of its mode."""
import pytest

from momp.onset.model import Status
from momp.onset.viz import style


def luminance(hex_colour: str) -> float:
    """WCAG relative luminance of "#rrggbb"."""
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two colours."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("status", list(Status))
def test_status_colours_clear_three_to_one(status, mode):
    """Each status colour has at least 3:1 contrast against the surface of the mode it is drawn on."""
    colour = style.status_style(status, mode)[0]
    assert contrast(colour, style.palette(mode).surface) >= 3.0, (status, mode, colour)


def test_prior_follows_the_mode():
    """PRIOR keeps its light colour in light mode and takes the secondary ink in dark mode; others do not change except CENSORED amber, darkened in light mode."""
    assert style.status_style(Status.PRIOR)[0] == "#52514e"
    assert style.status_style(Status.PRIOR, "light")[0] == "#52514e"
    assert style.status_style(Status.PRIOR, "dark")[0] == "#c3c2b7" == style.DARK.ink_secondary
    for status in Status:
        if status not in (Status.PRIOR, Status.CENSORED):
            assert style.status_style(status, "dark") == style.status_style(status, "light")


def test_palette_knows_its_mode():
    """The palettes name their mode so figure code can pass it on."""
    assert style.palette("light").mode == "light"
    assert style.palette("dark").mode == "dark"
