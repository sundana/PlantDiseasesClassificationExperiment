"""Figure: overview of the Shortcut Reliance Audit (protocol flowchart).

A single-column vertical flow:

    data -> models -> four diagnostic probes -> positive control -> report

The four probes and the positive control are enclosed in a dashed boundary
labelled "Shortcut Reliance Audit".  Output: figures/fig_paa_protocol.pdf/.png
and a copy in paper/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import pub_style as st

st.apply()

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent / "paper"


def box(ax, x, y, w, h, text, facecolor, edgecolor="#4B5563", fontsize=7.2, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", zorder=3,
    )
    return patch


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=0.9, color="#374151"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=8,
            linewidth=lw, color=color, zorder=1,
            shrinkA=0, shrinkB=0,
        )
    )


fig, ax = plt.subplots(figsize=(5.0, 4.4))
ax.set_xlim(0, 10.15)
ax.set_ylim(0, 10.6)
ax.axis("off")

# ---------------------------------------------------------------------------
# Row 1: data
# ---------------------------------------------------------------------------
box(ax, 1.1, 9.0, 3.6, 1.0,
    "PlantVillage (laboratory)\n29 classes, 41,879 images",
    facecolor="#EAF2F8", fontsize=7.0)
box(ax, 5.3, 9.0, 3.6, 1.0,
    "PlantDoc (field)\n20 mapped classes, 1,614 / 169",
    facecolor="#FDF2E9", fontsize=7.0)

# ---------------------------------------------------------------------------
# Row 2: models
# ---------------------------------------------------------------------------
box(ax, 1.1, 7.35, 7.8, 0.95,
    "Lightweight classifiers\nMobileNetV3 Large/Small · SE / CBAM / none · scratch vs ImageNet",
    facecolor="#F3F4F6", fontsize=7.2, bold=True)

# ---------------------------------------------------------------------------
# Row 3: four diagnostic probes
# ---------------------------------------------------------------------------
probe_y, probe_h, probe_w = 5.15, 1.25, 2.05
probe_x = [0.35, 2.65, 4.95, 7.25]
probes = [
    ("Frame probe\nouter 20% border\n→ FRI", "#E8F4F8"),
    ("Background ablation\n29 vs 28 classes\n→ BRI", "#E8F6F1"),
    ("Leaf-crop test\nfull image vs crop\n→ BRI", "#FFF4E5"),
    ("Attention test\nSE vs CBAM\nequal $r$ → TOST", "#F5EEF8"),
]
for x, (label, color) in zip(probe_x, probes):
    box(ax, x, probe_y, probe_w, probe_h, label, facecolor=color, fontsize=6.6)

# ---------------------------------------------------------------------------
# Row 4: positive control
# ---------------------------------------------------------------------------
box(ax, 1.1, 3.25, 7.8, 0.95,
    "Positive control: fine-tune on PlantDoc train\n→ evaluate on the 169-image test split → adaptation gain",
    facecolor="#EAF7EE", fontsize=7.0, bold=True)

# ---------------------------------------------------------------------------
# Row 5: report
# ---------------------------------------------------------------------------
box(ax, 1.1, 1.2, 7.8, 0.95,
    "Report: FRI · BRI · macro field F1 · seed variability · adaptation gain",
    facecolor="#F3F4F6", fontsize=7.0, bold=True)

# ---------------------------------------------------------------------------
# Arrows
# ---------------------------------------------------------------------------
# data -> models
arrow(ax, 2.9, 9.0, 2.9, 8.30)
arrow(ax, 7.1, 9.0, 7.1, 8.30)

# models -> probe bus
ax.plot([5.0, 5.0], [7.35, 6.85], color="#374151", lw=0.9, zorder=1)
ax.plot([probe_x[0] + probe_w / 2, probe_x[-1] + probe_w / 2], [6.85, 6.85],
        color="#374151", lw=0.9, zorder=1)
for x in probe_x:
    arrow(ax, x + probe_w / 2, 6.85, x + probe_w / 2, probe_y + probe_h)

# probe bus -> positive control
ax.plot([probe_x[0] + probe_w / 2, probe_x[-1] + probe_w / 2], [4.95, 4.95],
        color="#374151", lw=0.9, zorder=1)
arrow(ax, 5.0, 4.95, 5.0, 4.20)

# positive control -> report
arrow(ax, 5.0, 3.25, 5.0, 2.15)

# ---------------------------------------------------------------------------
# Dashed boundary around the audit
# ---------------------------------------------------------------------------
boundary = FancyBboxPatch(
    (-0.05, 2.95), 10.0, 3.75,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    linewidth=1.0, linestyle=(0, (4, 3)),
    edgecolor="#6B7280", facecolor="none", zorder=0,
)
ax.add_patch(boundary)
ax.text(0.12, 6.76, "Shortcut Reliance Audit", fontsize=7.4, fontweight="bold",
        color="#374151", ha="left", va="bottom", zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

fig.savefig(OUT / "fig_paa_protocol.pdf")
fig.savefig(OUT / "fig_paa_protocol.png")
plt.close(fig)

for ext in ("pdf", "png"):
    shutil.copyfile(OUT / f"fig_paa_protocol.{ext}", PAPER / f"fig_paa_protocol.{ext}")
print("wrote figures/fig_paa_protocol.{pdf,png} and paper/fig_paa_protocol.{pdf,png}")
