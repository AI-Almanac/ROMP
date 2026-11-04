"""Colours, status markings and matplotlib settings shared by every `hcwf.onset` figure.

Values come from the validated reference palette of the data-visualisation guidelines (categorical slots 1-3 pass
every colour-vision check in both modes; aqua is below 3:1 contrast on the light surface, so it always ships with a
visible label). Roles, not hues, are what figure code should use. `Palette` holds the roles of one mode, ``LIGHT`` and
``DARK`` are the two palettes, `status_style` gives the colour, icon and label of a `hcwf.onset.model.Status` and
`figure_style` applies the matplotlib settings. The other modules of `hcwf.onset.viz` build on this one.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl

from hcwf.onset.model import Status


@dataclass(frozen=True)
class Palette:
    """The colour roles of one mode (light or dark)."""

    surface: str
    """Background colour of the figure and axes, as a hex string."""
    ink: str
    """Primary text and mark colour."""
    ink_secondary: str
    """Axis label and tick label colour."""
    ink_muted: str
    """Tick mark colour."""
    grid: str
    """Grid line colour."""
    axis: str
    """Axis line colour."""
    wash: str
    """Fill colour of neutral background bands."""
    observed: str
    """Colour of observed data."""
    dry_spell: str
    """Colour of dry-spell marks."""
    forecast: str
    """Colour of forecast data."""
    mode: str = "light"
    """The mode name, ``"light"`` or ``"dark"``."""


LIGHT = Palette(surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781", grid="#e1e0d9",
                axis="#c3c2b7", wash="#f0efec", observed="#2a78d6", dry_spell="#eb6834", forecast="#1baf7a")
DARK = Palette(surface="#1a1a19", ink="#f0efec", ink_secondary="#c3c2b7", ink_muted="#898781", grid="#2c2c2a",
               axis="#383835", wash="#383835", observed="#3987e5", dry_spell="#d95926", forecast="#199e70", mode="dark")

STATUS_STYLE = {
    Status.ONSET: ("#0ca30c", "✓", "onset"),
    Status.NOT_FOUND: ("#898781", "–", "no onset"),
    Status.CENSORED: ("#fab219", "…", "past the forecast"),
    Status.PRIOR: ("#52514e", "◂", "onset before the forecast"),
    Status.INSUFFICIENT_DATA: ("#d03b3b", "!", "missing data"),
}
"""The ``(colour, icon, label)`` of each `~hcwf.onset.model.Status`. Status colours are reserved for statuses and always travel with an
icon and a label."""

MODE_STATUS_COLOUR = {("light", Status.CENSORED): "#b87a00"}
"""Per-mode colours, keyed by ``(mode, status)``, where the shared one is below 3:1 on that mode's surface (amber on
light)."""

WASH_ALPHA = 0.10
"""Opacity of area fills: they are washes, the role colour at about 10 %."""
LINE_WIDTH = 2.0
"""Line width in points."""
HAIRLINE = 1.0
"""Width in points of axes, grid and tick lines."""
MARKER_SIZE = 8.0
"""Marker size in points."""
BAR_MAX_WIDTH_DAYS = 0.8
"""Largest bar width of a daily bar chart, in days."""


def palette(mode: str = "light") -> Palette:
    """Return the colour roles of a mode.

    Parameters
    ----------
    mode : {"light", "dark"}, default "light"
        The mode.

    Returns
    -------
    Palette
        ``LIGHT`` or ``DARK``.

    Raises
    ------
    ValueError
        ``mode`` is neither ``"light"`` nor ``"dark"``.
    """
    if mode not in ("light", "dark"):
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    return LIGHT if mode == "light" else DARK


def status_style(status, mode: str = "light") -> tuple[str, str, str]:
    """Return the colour, icon and label of a status in a mode; the icon and label must be shown with the colour.

    Every status colour clears 3:1 on its mode's surface. Two differ by mode: ``PRIOR``, whose light ink is nearly
    invisible on the dark surface, takes the palette's secondary ink in dark mode, and ``CENSORED`` amber is
    darkened in light mode.

    Parameters
    ----------
    status : hcwf.onset.model.Status or int
        The status, or its integer value.
    mode : {"light", "dark"}, default "light"
        The mode.

    Returns
    -------
    tuple of (str, str, str)
        The hex colour, the icon character and the label.
    """
    p = palette(mode)
    status = Status(int(status))
    colour, icon, label = STATUS_STYLE[status]
    if status == Status.PRIOR and mode == "dark":
        colour = p.ink_secondary
    colour = MODE_STATUS_COLOUR.get((mode, status), colour)
    return colour, icon, label


@contextmanager
def figure_style(mode: str = "light") -> Iterator[Palette]:
    """Apply the matplotlib settings of `hcwf.onset` figures.

    The settings are system sans, recessive hairline axes and grid, and no top or right spines.

    Use as a context manager around the figure's construction; the settings are those of `matplotlib.rc_context`.

    Parameters
    ----------
    mode : {"light", "dark"}, default "light"
        The mode.

    Yields
    ------
    Palette
        The colour roles of the mode.
    """
    p = palette(mode)
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "figure.facecolor": p.surface,
        "axes.facecolor": p.surface,
        "savefig.facecolor": p.surface,
        "axes.edgecolor": p.axis,
        "axes.linewidth": HAIRLINE,
        "axes.labelcolor": p.ink_secondary,
        "axes.titlecolor": p.ink,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": p.grid,
        "grid.linewidth": HAIRLINE,
        "grid.linestyle": "-",
        "xtick.color": p.ink_muted,
        "ytick.color": p.ink_muted,
        "xtick.labelcolor": p.ink_secondary,
        "ytick.labelcolor": p.ink_secondary,
        "xtick.major.width": HAIRLINE,
        "ytick.major.width": HAIRLINE,
        "text.color": p.ink,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": LINE_WIDTH,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
    with mpl.rc_context(rc):
        yield p
