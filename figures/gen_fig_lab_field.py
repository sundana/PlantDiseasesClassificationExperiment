"""Figure: laboratory vs zero-shot field accuracy (two seeds).

Two panels avoid compressing the 6-9% field range onto a 0-100 axis.
Output: figures/fig_lab_field.pdf and .png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pub_style as st

st.apply()

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = Path(__file__).resolve().parent

MODELS = [
    ("mobilenetv3_large", "Large SE r=4"),
    ("proposed_large_32", "Large CBAM r=32"),
    ("mobilenetv3_large_none", "Large none"),
]
CONDITIONS = [("scratch", "scratch", st.SCRATCH, "o"),
              ("imagenet", "pretrained", st.PRETRAINED, "s")]


def load(model, condition, seed):
    with open(METRICS / f"{model}__{condition}__s{seed}.json", encoding="utf-8") as fh:
        return json.load(fh)


rows = []
for condition, short, color, marker in CONDITIONS:
    for model, label in MODELS:
        payloads = [load(model, condition, seed) for seed in ("42", "43")]
        lab = np.array([p["lab"]["accuracy"] for p in payloads])
        field = np.array([p["field"]["accuracy"] for p in payloads])
        rows.append({
            "label": f"{label.replace('Large ', '')} ({short})",
            "lab_mean": lab.mean(), "lab_sd": lab.std(ddof=1),
            "field_mean": field.mean(), "field_sd": field.std(ddof=1),
            "color": color, "marker": marker,
        })

fig, axes = plt.subplots(
    1, 2, figsize=(5.0, 3.0), sharey=True,
    gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.08},
)
y = np.arange(len(rows))[::-1]

for row, yi in zip(rows, y):
    axes[0].errorbar(row["lab_mean"], yi, xerr=row["lab_sd"], fmt=row["marker"],
                     color=row["color"], ecolor=row["color"], elinewidth=0.9,
                     capsize=2, capthick=0.8, ms=4.2, zorder=3)
    axes[1].errorbar(row["field_mean"], yi, xerr=row["field_sd"], fmt=row["marker"],
                     color=row["color"], ecolor=row["color"], elinewidth=0.9,
                     capsize=2, capthick=0.8, ms=4.2, zorder=3)
    axes[1].text(row["field_mean"] + row["field_sd"] + 0.35, yi,
                 f"{row['field_mean']:.1f}", va="center", fontsize=7.2,
                 color=row["color"])

axes[0].set_xlim(95.0, 100.1)
axes[1].set_xlim(0.0, 14.0)
axes[0].set_title("(a) Laboratory (PlantVillage)")
axes[1].set_title("(b) Field (PlantDoc, zero-shot)")
axes[0].set_xlabel("Accuracy (%)")
axes[1].set_xlabel("Accuracy (%)")
axes[0].set_yticks(y)
axes[0].set_yticklabels([r["label"] for r in rows], fontsize=8.2)
axes[1].axvline(5.0, color=st.GREY, ls="--", lw=0.8, zorder=1)
axes[1].text(5.15, y[-1] - 0.55, "chance", fontsize=7.2, color=st.GREY)
axes[0].axhline(2.5, color=st.LIGHT, lw=0.8)
axes[1].axhline(2.5, color=st.LIGHT, lw=0.8)
axes[0].set_ylim(-0.7, len(rows) - 0.3)

handles = [
    plt.Line2D([], [], marker="o", ls="", color=st.SCRATCH, label="scratch"),
    plt.Line2D([], [], marker="s", ls="", color=st.PRETRAINED, label="pretrained"),
]
axes[0].legend(handles=handles, loc="lower left", frameon=False, ncol=2,
               handletextpad=0.2, columnspacing=0.8)

fig.savefig(OUT / "fig_lab_field.pdf")
fig.savefig(OUT / "fig_lab_field.png")
print("wrote figures/fig_lab_field.{pdf,png}")
