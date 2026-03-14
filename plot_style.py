"""Shared plotting style helpers for project notebooks."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cycler import cycler

MIN_PLOT_DATE = datetime(1999, 1, 1)
LABEL_START_DATE = datetime(2000, 1, 1)
LABEL_EVERY_YEARS = 2

COLORS = {
    "primary": "#0F4C5C",
    "secondary": "#1F3A5F",
    "accent": "#B23A48",
    "neutral": "#4F5D75",
    "blue": "#2C6EAA",
    "blue_light": "#A9CCE3",
    "orange": "#C97A1E",
    "red": "#A63446",
    "purple": "#5B4B8A",
    "green": "#2E8B57",
    "brown": "#8C5A2B",
    "reference": "#2B3A42",
    "white": "#FFFFFF",
}


def set_global_plot_style() -> None:
    """Set notebook-wide matplotlib defaults."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FAFBFC"
    plt.rcParams["axes.edgecolor"] = "#B8C2CC"
    plt.rcParams["axes.labelcolor"] = "#243447"
    plt.rcParams["xtick.color"] = "#2B3A42"
    plt.rcParams["ytick.color"] = "#2B3A42"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.xmargin"] = 0.0
    plt.rcParams["axes.autolimit_mode"] = "data"
    plt.rcParams["grid.color"] = "#D5DDE5"
    plt.rcParams["grid.alpha"] = 0.55
    plt.rcParams["grid.linewidth"] = 0.7
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["lines.linewidth"] = 1.0
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["patch.linewidth"] = 0.4
    plt.rcParams["axes.prop_cycle"] = cycler(color=[
        COLORS["blue"], COLORS["primary"], COLORS["orange"], COLORS["purple"], COLORS["accent"], COLORS["neutral"]
    ])
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.facecolor"] = (1.0, 1.0, 1.0, 0.82)
    plt.rcParams["legend.edgecolor"] = "#C4CDD7"
    plt.rcParams["legend.framealpha"] = 0.92


def style_axes(
    ax,
    *,
    grid_axis: str = "y",
    grid_alpha: float = 0.25,
    label_x: float = -0.04,
    label_y: float = 0.5,
    label_pad: int = 1,
) -> None:
    """Apply common axis formatting."""
    ax.set_xmargin(0)
    ax.margins(x=0)
    ax.grid(axis=grid_axis, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C2CC")
    ax.spines["bottom"].set_color("#B8C2CC")
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_label_coords(label_x, label_y)
    ax.yaxis.labelpad = label_pad


def style_time_axis(
    ax,
    *,
    x_min,
    x_max,
    x_ticks: Iterable,
    date_fmt: str = "%Y",
) -> None:
    """Apply shared date-axis formatting.

    Axis can start at 1999-01-01, but labels start at 2000-01-01 and then every 2 years.
    """
    min_num = mdates.date2num(MIN_PLOT_DATE)
    x_min_num = max(mdates.date2num(x_min), min_num)
    x_max_num = mdates.date2num(x_max)
    if x_max_num < min_num:
        x_max_num = min_num

    x_max_dt = mdates.num2date(x_max_num).replace(tzinfo=None)
    max_year = x_max_dt.year
    ticks = [
        datetime(y, 1, 1)
        for y in range(LABEL_START_DATE.year, max_year + 1, LABEL_EVERY_YEARS)
    ]
    ticks = [t for t in ticks if mdates.date2num(t) <= x_max_num]
    if not ticks:
        ticks = [LABEL_START_DATE]

    ax.set_xlim(mdates.num2date(x_min_num), mdates.num2date(x_max_num))
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))


def style_legend(ax, *, loc: str = "best", frameon: bool = True, title: str | None = None) -> None:
    """Apply common legend style with a semi-transparent card background."""
    ax.legend(
        loc=loc,
        frameon=True,
        facecolor=(1.0, 1.0, 1.0, 0.82),
        edgecolor="#C4CDD7",
        framealpha=0.92,
        fancybox=True,
        title=title,
        borderpad=0.55,
    )
