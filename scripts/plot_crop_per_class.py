"""Per-class effect of the leaf-crop intervention (E4).

Usage:
    python scripts/plot_crop_per_class.py \
        --metrics results/metrics/crop_mobilenetv3_large__scratch__s42.json \
        --out results/plots/crop_per_class_se.png
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out", default="results/plots/crop_per_class.png")
    parser.add_argument("--title", default="Leaf-crop effect by disorder")
    args = parser.parse_args()

    aggregate = {}
    for path in args.metrics:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for key, value in payload["annotated_subset"]["per_class_delta"].items():
            entry = aggregate.setdefault(key, {"delta": [], "full": [], "crop": []})
            entry["delta"].append(value["delta_pp"])
            entry["full"].append(value["recall_full"])
            entry["crop"].append(value["recall_crop"])
    rows = sorted(
        ((key, float(np.mean(value["delta"])), float(np.mean(value["full"])),
          float(np.mean(value["crop"]))) for key, value in aggregate.items()),
        key=lambda item: item[1],
    )
    names = [k.replace("___", " ").replace("_", " ") for k, _, _, _ in rows]
    deltas = np.array([d for _, d, _, _ in rows])
    full = np.array([f for _, _, f, _ in rows])
    crop = np.array([c for _, _, _, c in rows])

    fig, ax = plt.subplots(figsize=(8.2, 0.38 * len(rows) + 1.8))
    y = np.arange(len(rows))
    colors = ["#a3541b" if d < 0 else "#1f6f5c" for d in deltas]
    ax.barh(y, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="#1b1f24", lw=0.8)
    for yi, d, f, c in zip(y, deltas, full, crop):
        ax.text(d + (0.15 if d >= 0 else -0.15), yi, f"{d:+.1f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Change in recall after cropping to leaf boxes (percentage points)")
    ax.set_title(args.title, fontsize=10)
    ax.grid(axis="x", color="#e7ebee", lw=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
