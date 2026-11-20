"""Figure: field accuracy under the audit interventions (two seeds).

Three small multiples with horizontal bars; horizontal layout keeps the five
condition labels readable. Output: figures/fig_interventions.pdf and .png
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
SEEDS = ("42", "43")

CONDITIONS = [
    ("scratch", "Scratch (full)", st.SCRATCH),
    ("nobg", "No background", st.NOBG),
    ("imagenet", "Pretrained (full)", st.PRETRAINED),
    ("crop_scratch", "Scratch+crop", st.CROP_SCRATCH),
    ("crop_pretrained", "Pretrained+crop", st.CROP_PRETRAINED),
]


def run_metric(model, condition, seed):
    with open(METRICS / f"{model}__{condition}__s{seed}.json", encoding="utf-8") as fh:
        return json.load(fh)["field"]["accuracy"]


def crop_metric(model, condition, seed):
    with open(METRICS / f"crop_{model}__{condition}__s{seed}.json", encoding="utf-8") as fh:
        return json.load(fh)["annotated_subset"]["crop"]["accuracy"]


fig, axes = plt.subplots(1, 3, figsize=(5.0, 2.7), sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0.10})
y = np.arange(len(CONDITIONS))[::-1]

for ax, (model, label) in zip(axes, MODELS):
    means, sds = [], []
    for condition, _, _ in CONDITIONS:
        if condition == "crop_scratch":
            values = [crop_metric(model, "scratch", seed) for seed in SEEDS]
        elif condition == "crop_pretrained":
            values = [crop_metric(model, "imagenet", seed) for seed in SEEDS]
        else:
            values = [run_metric(model, condition, seed) for seed in SEEDS]
        means.append(np.mean(values))
        sds.append(np.std(values, ddof=1))

    ax.barh(y, means, xerr=sds, height=0.66,
            color=[c for _, _, c in CONDITIONS],
            edgecolor="white", linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8})
    for yi, mean, sd in zip(y, means, sds):
        ax.text(mean + sd + 0.35, yi, f"{mean:.1f}", va="center", fontsize=7.0)
    ax.axvline(5.0, color=st.GREY, ls="--", lw=0.8, zorder=1)
    ax.set_title(label, fontsize=8.8)
    ax.set_xlim(0, 16)
    ax.grid(axis="y", visible=False)

axes[0].set_yticks(y)
axes[0].set_yticklabels([label for _, label, _ in CONDITIONS], fontsize=7.8)
for ax in axes[1:]:
    ax.tick_params(axis="y", labelleft=False, length=0)

axes[1].set_xlabel("PlantDoc accuracy (%)")
axes[0].set_ylim(-0.65, 4.55)
axes[0].text(5.25, -0.55, "chance", fontsize=7.0, color=st.GREY)

fig.savefig(OUT / "fig_interventions.pdf")
fig.savefig(OUT / "fig_interventions.png")
print("wrote figures/fig_interventions.{pdf,png}")
