"""Colours, status markings and matplotlib settings shared by every momp.onset figure.

Values come from the validated reference palette of the data-visualisation
guidelines (categorical slots 1-3 pass every colour-vision check in both modes;
aqua is below 3:1 contrast on the light surface, so it always ships with a
visible label). Roles, not hues, are what figure code should use.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib as mpl

from momp.onset.model import Status


@dataclass(frozen=True)
class Palette:
    """The colour roles of one mode (light or dark)."""

    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    wash: str
    observed: str
    dry_spell: str
    forecast: str


LIGHT = Palette(surface="#fcfcfb", ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781", grid="#e1e0d9",
                axis="#c3c2b7", wash="#f0efec", observed="#2a78d6", dry_spell="#eb6834", forecast="#1baf7a")
DARK = Palette(surface="#1a1a19", ink="#f0efec", ink_secondary="#c3c2b7", ink_muted="#898781", grid="#2c2c2a",
               axis="#383835", wash="#383835", observed="#3987e5", dry_spell="#d95926", forecast="#199e70")

#: Status colours are reserved for statuses and always travel with an icon and a label.
STATUS_STYLE = {
    Status.ONSET: ("#0ca30c", "✓", "onset"),
    Status.NOT_FOUND: ("#898781", "–", "no onset"),
    Status.CENSORED: ("#fab219", "…", "past the forecast"),
    Status.PRIOR: ("#52514e", "◂", "onset before the forecast"),
    Status.INSUFFICIENT_DATA: ("#d03b3b", "!", "missing data"),
}

#: Area fills are washes: the role colour at about 10 % opacity.
WASH_ALPHA = 0.10
LINE_WIDTH = 2.0
HAIRLINE = 1.0
MARKER_SIZE = 8.0
BAR_MAX_WIDTH_DAYS = 0.8


def palette(mode: str = "light") -> Palette:
    """The colour roles for "light" or "dark"."""
    if mode not in ("light", "dark"):
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    return LIGHT if mode == "light" else DARK


def status_style(status) -> tuple[str, str, str]:
    """(colour, icon, label) for a Status; the icon and label must be shown with the colour."""
    return STATUS_STYLE[Status(int(status))]


@contextmanager
def figure_style(mode: str = "light"):
    """Matplotlib settings for momp.onset figures: system sans, recessive hairline axes and grid, no top/right spines."""
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
