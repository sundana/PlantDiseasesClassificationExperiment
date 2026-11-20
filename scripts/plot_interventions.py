"""Summary figure: field accuracy under background-targeted interventions.

For each core model, shows scratch, no-background, pretrained, and cropped
accuracies against the laboratory accuracy (annotated, not plotted on the same
scale to avoid compressing the field bars).

Usage:
    python scripts/plot_interventions.py --out results/plots/interventions.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = {
    "mobilenetv3_large": "Large SE r=4",
    "proposed_large_32": "Large CBAM r=32",
    "mobilenetv3_large_none": "Large none",
}


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--out", default="results/plots/interventions.png")
    parser.add_argument("--seed", default="42")
    args = parser.parse_args()

    rows = []
    for model, label in MODELS.items():
        s = load(os.path.join(args.metrics_dir, f"{model}__scratch__s{args.seed}.json"))
        p = load(os.path.join(args.metrics_dir, f"{model}__imagenet__s{args.seed}.json"))
        n = load(os.path.join(args.metrics_dir, f"{model}__nobg__s{args.seed}.json"))
        cs = load(os.path.join(args.metrics_dir, f"crop_{model}__scratch__s{args.seed}.json"))
        cp = load(os.path.join(args.metrics_dir, f"crop_{model}__imagenet__s{args.seed}.json"))
        rows.append({
            "label": label,
            "lab": s["lab"]["accuracy"],
            "scratch": s["field"]["accuracy"],
            "nobg": n["field"]["accuracy"],
            "pretrained": p["field"]["accuracy"],
            "crop_scratch": cs["annotated_subset"]["crop"]["accuracy"],
            "crop_pretrained": cp["annotated_subset"]["crop"]["accuracy"],
        })

    series = [
        ("scratch", "scratch", "#1f6f5c"),
        ("nobg", "no background class", "#4f9d8b"),
        ("pretrained", "pretrained", "#7fb3a8"),
        ("crop_scratch", "scratch + crop", "#c98b5e"),
        ("crop_pretrained", "pretrained + crop", "#e0b089"),
    ]
    n_series = len(series)
    x = np.arange(len(rows))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for i, (key, label, color) in enumerate(series):
        values = [r[key] for r in rows]
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                    f"{value:.1f}", ha="center", va="bottom", fontsize=7.5)

    for i, row in enumerate(rows):
        ax.text(i, max(row["crop_pretrained"], row["pretrained"], row["nobg"],
                       row["scratch"], row["crop_scratch"]) + 1.4,
                f"lab {row['lab']:.1f}%", ha="center", fontsize=8.5, color="#5b6770")

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("PlantDoc accuracy (%)")
    ax.set_ylim(0, 16)
    ax.set_title("Field accuracy under background-targeted interventions "
                 "(all laboratory accuracies are 95.9–99.6%)", fontsize=10)
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", color="#e7ebee", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
