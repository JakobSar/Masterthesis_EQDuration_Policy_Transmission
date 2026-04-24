"""Shared plotting style helpers for project notebooks."""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from cycler import cycler

MIN_PLOT_DATE = datetime(1998, 1, 1)
LABEL_START_DATE = datetime(1999, 1, 1)
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


_TEX_FONTS_REGISTERED = False


def _register_tex_fonts() -> None:
    """Register TeX Gyre Termes and Nimbus Roman from a local TeXLive install.

    The LaTeX thesis uses ``\\usepackage{mathptmx}``, which typesets the body
    text in URW Nimbus Roman No. 9 L (the free PostScript-Times clone). macOS's
    "Times New Roman" renders noticeably thinner at typical figure DPI, so we
    point matplotlib at the exact same OpenType files that TeXLive ships —
    TeX Gyre Termes (Nimbus with extended glyph coverage) and Nimbus Roman
    itself — whenever they are available on disk. Silent no-op if they aren't.
    """
    global _TEX_FONTS_REGISTERED
    if _TEX_FONTS_REGISTERED:
        return

    # Cover TeXLive (any year), MacTeX per-user and system installs, and Linux distros.
    candidate_patterns = [
        "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
        "/usr/local/texlive/*/texmf-dist/fonts/opentype/urw/NimbusRoman-*.otf",
        "~/Library/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
        "~/Library/texlive/*/texmf-dist/fonts/opentype/urw/NimbusRoman-*.otf",
        "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
        "/Library/TeX/Root/texmf-dist/fonts/opentype/urw/NimbusRoman-*.otf",
        "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
        "/usr/share/fonts/opentype/urw-base35/NimbusRoman-*.otf",
    ]
    for pattern in candidate_patterns:
        for path in glob.glob(os.path.expanduser(pattern)):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                # Never let font registration break plotting.
                pass
    _TEX_FONTS_REGISTERED = True


def set_global_plot_style() -> None:
    """Set notebook-wide matplotlib defaults."""
    _register_tex_fonts()
    plt.rcParams["font.family"] = "serif"
    # Prefer TeX Gyre Termes / Nimbus Roman — matches the thesis body font set
    # via \usepackage{mathptmx}. Times New Roman / Times / DejaVu Serif act as
    # fallbacks if neither TeXLive font is installed on the system.
    plt.rcParams["font.serif"] = [
        "TeX Gyre Termes",
        "Nimbus Roman",
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FAFBFC"
    # Plot chrome (spines, tick marks, grid) kept light so the black text
    # dominates — matches the AER/JFE/QJE figure look.
    plt.rcParams["axes.edgecolor"] = "#C4CDD7"
    plt.rcParams["axes.labelcolor"] = "#000000"
    plt.rcParams["axes.titlecolor"] = "#000000"
    plt.rcParams["xtick.color"] = "#C4CDD7"        # tick marks match spines
    plt.rcParams["ytick.color"] = "#C4CDD7"        # tick marks match spines
    # Tick *labels* in black (same ink as body text). Older matplotlib (<3.4)
    # has no separate labelcolor key; in that case fall back to all-black ticks.
    try:
        plt.rcParams["xtick.labelcolor"] = "#000000"
        plt.rcParams["ytick.labelcolor"] = "#000000"
    except KeyError:
        plt.rcParams["xtick.color"] = "#000000"
        plt.rcParams["ytick.color"] = "#000000"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.xmargin"] = 0.0
    plt.rcParams["axes.autolimit_mode"] = "data"
    plt.rcParams["grid.color"] = "#E0E5EB"
    plt.rcParams["grid.alpha"] = 0.55
    plt.rcParams["grid.linewidth"] = 0.7
    plt.rcParams["axes.titlepad"] = 10
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["legend.title_fontsize"] = 14
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
) -> None:
    """Apply common axis formatting."""
    ax.set_xmargin(0)
    ax.margins(x=0)
    ax.grid(axis=grid_axis, alpha=grid_alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C4CDD7")
    ax.spines["bottom"].set_color("#C4CDD7")
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.labelpad = 10


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
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")


def save_figure(fig, path, *, dpi: int = 300) -> None:
    """Save *fig* to *path* (TeX Gyre Termes, i.e. the thesis body font) and a _beamer variant with Alegreya Sans.

    *path* can be a str or Path; the extension is preserved as-is (defaults to .png if none).
    The _beamer file is written next to the original with ``_beamer`` appended before the suffix.

    The main (non-beamer) variant is saved WITHOUT axis titles (the LaTeX document
    adds the title). The beamer variant keeps the title and lifts it further above
    the axes for better readability on slides.
    """
    from pathlib import Path

    path = Path(path)

    # --- Main variant: strip suptitle only, keep subplot titles, save, then restore ---
    _saved_suptitle = None
    if getattr(fig, "_suptitle", None) is not None:
        _saved_suptitle = fig._suptitle.get_text()
        fig.suptitle("")
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        if _saved_suptitle is not None:
            fig.suptitle(_saved_suptitle)

    beamer_path = path.with_stem(path.stem + "_beamer")
    orig_family = plt.rcParams["font.family"]
    orig_serif = list(plt.rcParams["font.serif"])
    # Track original font sizes of star annotations so we can restore them
    _star_orig_sizes = {}
    _STAR_SCALE = 1.4  # scale factor for significance stars in beamer variant
    _BEAMER_TITLE_PAD = 22  # extra space between axis title and plot for beamer
    _orig_title_pads = {}
    try:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Alegreya Sans", "DejaVu Sans", "Arial"]
        for ax in fig.get_axes():
            _orig_title_pads[id(ax)] = ax.title.get_position()
            # Lift the title further above the axes
            ax.set_title(ax.get_title(), pad=_BEAMER_TITLE_PAD)
        for txt in fig.texts:
            txt.set_fontfamily("sans-serif")
        for ax in fig.get_axes():
            for txt in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
                + ax.texts
            ):
                txt.set_fontfamily("sans-serif")
            # Enlarge significance star annotations for beamer readability
            for txt in ax.texts:
                if txt.get_text().strip().replace("*", "") == "":
                    orig_size = txt.get_fontsize()
                    _star_orig_sizes[id(txt)] = orig_size
                    txt.set_fontsize(orig_size * _STAR_SCALE)
            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontfamily("sans-serif")
                if leg.get_title():
                    leg.get_title().set_fontfamily("sans-serif")
        fig.savefig(beamer_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.rcParams["font.family"] = orig_family
        plt.rcParams["font.serif"] = orig_serif
        # Restore original title pad
        for ax in fig.get_axes():
            ax.set_title(ax.get_title(), pad=plt.rcParams["axes.titlepad"])
        for txt in fig.texts:
            txt.set_fontfamily("serif")
        for ax in fig.get_axes():
            for txt in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
                + ax.texts
            ):
                txt.set_fontfamily("serif")
            # Restore original star sizes
            for txt in ax.texts:
                if id(txt) in _star_orig_sizes:
                    txt.set_fontsize(_star_orig_sizes[id(txt)])
            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontfamily("serif")
                if leg.get_title():
                    leg.get_title().set_fontfamily("serif")


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
