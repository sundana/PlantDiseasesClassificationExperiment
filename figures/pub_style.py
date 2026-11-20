"""Shared publication style for the manuscript figures."""

import matplotlib as mpl

# Colorblind-safe palette (Okabe-Ito derived)
SCRATCH = "#0072B2"          # blue
NOBG = "#009E73"             # green
PRETRAINED = "#56B4E9"       # sky blue
CROP_SCRATCH = "#E69F00"     # orange
CROP_PRETRAINED = "#D55E00"  # vermillion
LAB = "#0072B2"
FIELD = "#D55E00"
POS = "#009E73"
NEG = "#D55E00"
GREY = "#6B7280"
LIGHT = "#D1D5DB"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
