"""PAA figures: test-split audit results and the fine-tune positive control.

Reads results/paa_tables.json (produced by scripts/report_paa_tables.py) and
the per-image prediction files, and writes four vector figures:

  fig_paa_lab_field.pdf        laboratory vs official-test field accuracy
  fig_paa_interventions.pdf    background ablation, crop and border results
  fig_paa_adaptation.pdf       zero-shot -> fine-tuned test accuracy
  fig_paa_crop_per_class.pdf   copy of the existing per-class crop figure

Run from the repository root:
    .venv/bin/python figures/gen_fig_paa.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pub_style as st

st.apply()

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent / "paper"
TABLES = ROOT / "results" / "paa_tables.json"
PRED = ROOT / "results" / "predictions"

MODELS = [
    ("mobilenetv3_large", "Large SE r=4"),
    ("proposed_large_32", "Large CBAM r=32"),
    ("mobilenetv3_large_none", "Large none"),
]
CONDITIONS = [("scratch", "scratch", st.SCRATCH, "o"),
              ("imagenet", "pretrained", st.PRETRAINED, "s")]


def load_tables():
    with open(TABLES, encoding="utf-8") as fh:
        return json.load(fh)


def lab_accuracy(model, init, seed):
    path = ROOT / "results" / "metrics" / f"{model}__{init}__s{seed}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["lab"]["accuracy"]


def test_accuracy(model, init, seed):
    path = PRED / f"{model}__{init}__s{seed}_plantdoc.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row["image"].startswith("test/")]
    return 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)


def fig_lab_field(tables):
    rows = []
    for init, short, color, marker in CONDITIONS:
        for model, label in MODELS:
            lab = np.array([lab_accuracy(model, init, seed) for seed in ("42", "43")])
            field = np.array([test_accuracy(model, init, seed) for seed in ("42", "43")])
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
    axes[1].set_title("(b) Field (PlantDoc test)")
    axes[0].set_xlabel("Accuracy (%)")
    axes[1].set_xlabel("Accuracy (%)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r["label"] for r in rows], fontsize=8.2)
    axes[1].axvline(3.45, color=st.GREY, ls="--", lw=0.8, zorder=1)
    axes[1].text(3.6, y[-1] - 0.55, "chance", fontsize=7.2, color=st.GREY)
    axes[0].axhline(2.5, color=st.LIGHT, lw=0.8)
    axes[1].axhline(2.5, color=st.LIGHT, lw=0.8)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=st.SCRATCH, label="scratch"),
        plt.Line2D([], [], marker="s", ls="", color=st.PRETRAINED, label="pretrained"),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, ncol=2,
                   handletextpad=0.2, columnspacing=0.8)

    fig.savefig(OUT / "fig_paa_lab_field.pdf")
    fig.savefig(OUT / "fig_paa_lab_field.png")
    plt.close(fig)


def fig_interventions(tables):
    core = {(row["model"], row["init"]): row for row in tables["core"] if row["split"] == "test"}
    nobg = {row["model"]: row for row in tables["nobg"] if row["split"] == "test"}
    crop = {(row["model"], row["init"]): row for row in tables["crop"] if row["split"] == "test"}

    labels = ["Large SE r=4", "Large CBAM r=32", "Large none"]
    conditions = [
        ("Scratch (full)", st.SCRATCH),
        ("No background", st.NOBG),
        ("Pretrained (full)", st.PRETRAINED),
        ("Scratch + crop", st.CROP_SCRATCH),
        ("Pretrained + crop", st.CROP_PRETRAINED),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5.0, 2.5), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.10})
    y = np.arange(len(conditions))[::-1]
    for ax, label in zip(axes, labels):
        means = [
            core[(label, "scratch")]["accuracy"]["mean"],
            nobg[label]["accuracy"]["mean"],
            core[(label, "pretrained")]["accuracy"]["mean"],
            crop[(label, "scratch")]["crop"]["mean"],
            crop[(label, "pretrained")]["crop"]["mean"],
        ]
        sds = [
            core[(label, "scratch")]["accuracy"]["sd"],
            nobg[label]["accuracy"]["sd"],
            core[(label, "pretrained")]["accuracy"]["sd"],
            crop[(label, "scratch")]["crop"]["sd"],
            crop[(label, "pretrained")]["crop"]["sd"],
        ]
        ax.barh(y, means, xerr=sds, height=0.66,
                color=[c for _, c in conditions],
                edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8})
        for yi, mean, sd in zip(y, means, sds):
            ax.text(mean + sd + 0.35, yi, f"{mean:.1f}", va="center", fontsize=6.8)
        ax.axvline(3.45, color=st.GREY, ls="--", lw=0.8, zorder=1)
        ax.set_title(label, fontsize=8.6)
        ax.set_xlim(0, 19)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label for label, _ in conditions], fontsize=7.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)
    axes[1].set_xlabel("PlantDoc test accuracy (%)")
    axes[0].set_ylim(-0.65, 4.55)
    axes[0].text(3.6, -0.55, "chance", fontsize=6.8, color=st.GREY)

    fig.savefig(OUT / "fig_paa_interventions.pdf")
    fig.savefig(OUT / "fig_paa_interventions.png")
    plt.close(fig)


def fig_adaptation(tables):
    runs = []
    for row in tables["finetune"]:
        model = row["model"]
        seed = row["seed"]
        zero = test_accuracy(model, "imagenet", seed)
        runs.append({
            "model": model,
            "seed": seed,
            "zero": zero,
            "fine": row["field_accuracy"],
        })
    color = {
        "mobilenetv3_large": st.SCRATCH,
        "proposed_large_32": st.PRETRAINED,
        "mobilenetv3_large_none": st.NOBG,
    }
    marker = {"42": "o", "43": "s"}

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for run in runs:
        ax.plot([0, 1], [run["zero"], run["fine"]], marker=marker[str(run["seed"])],
                color=color[run["model"]], lw=1.0, ms=4.0, alpha=0.9)
    # mean line
    mean_zero = np.mean([run["zero"] for run in runs])
    mean_fine = np.mean([run["fine"] for run in runs])
    mean_gain = mean_fine - mean_zero
    ax.plot([0, 1], [mean_zero, mean_fine], color="black", lw=2.0, ls="--",
            marker="D", ms=3.5, label="mean")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(0, 72)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["zero-shot", "fine-tuned"])
    ax.set_ylabel("PlantDoc test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.text(0.02, mean_zero + 4.5, f"{mean_zero:.1f}", fontsize=7.0,
            ha="left", va="bottom")
    ax.text(0.86, mean_fine + 3.0, f"{mean_fine:.1f}", fontsize=7.0,
            ha="left", va="bottom")
    ax.text(0.55, (mean_zero + mean_fine) / 2 + 5.0,
            f"mean +{mean_gain:.1f} pp", fontsize=7.0, color="black",
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    handles = [
        plt.Line2D([], [], color=st.SCRATCH, label="Large SE"),
        plt.Line2D([], [], color=st.PRETRAINED, label="Large CBAM"),
        plt.Line2D([], [], color=st.NOBG, label="Large none"),
        plt.Line2D([], [], color="black", ls="--", label="mean"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2,
              fontsize=6.5, handlelength=1.4, columnspacing=0.8)

    fig.savefig(OUT / "fig_paa_adaptation.pdf")
    fig.savefig(OUT / "fig_paa_adaptation.png")
    plt.close(fig)


def main():
    tables = load_tables()
    fig_lab_field(tables)
    fig_interventions(tables)
    fig_adaptation(tables)

    # The per-class crop figure is unchanged (all-image sensitivity analysis);
    # copy it under the PAA name so the submission bundle is self-contained.
    src = PAPER / "Fig7_crop_per_class.pdf"
    if src.exists():
        shutil.copyfile(src, OUT / "fig_paa_crop_per_class.pdf")
        shutil.copyfile(PAPER / "Fig7_crop_per_class.png", OUT / "fig_paa_crop_per_class.png")

    for name in ("fig_paa_lab_field", "fig_paa_interventions", "fig_paa_adaptation",
                 "fig_paa_crop_per_class"):
        for ext in ("pdf", "png"):
            path = OUT / f"{name}.{ext}"
            if path.exists():
                shutil.copyfile(path, PAPER / path.name)
    print("wrote PAA figures to figures/ and paper/")


if __name__ == "__main__":
    main()
