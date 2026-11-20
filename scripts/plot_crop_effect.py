"""Plot the paired leaf-crop intervention from audit_crop.py metrics.

Usage:
    python scripts/plot_crop_effect.py --metrics-glob "results/metrics/crop_*.json"
"""

import argparse
import glob
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
    parser.add_argument("--metrics-glob", default="results/metrics/crop_*.json")
    parser.add_argument("--out", default="results/plots/crop_effect.png")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(args.metrics_glob)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        subset = payload.get("annotated_subset")
        if not subset:
            continue
        rows.append({
            "tag": payload.get("checkpoint", os.path.basename(path)),
            "model": payload.get("model", "?"),
            "full": subset["full"]["accuracy"],
            "crop": subset["crop"]["accuracy"],
            "delta": subset["delta_accuracy_pp"],
            "p": subset["mcnemar"]["p_value"],
            "lo": subset["bootstrap_crop_minus_full"]["ci_low_pp"],
            "hi": subset["bootstrap_crop_minus_full"]["ci_high_pp"],
        })
    if not rows:
        raise SystemExit(f"no crop metrics matched {args.metrics_glob}")

    rows.sort(key=lambda r: r["full"])
    fig, ax = plt.subplots(figsize=(7.6, 0.7 * len(rows) + 1.8))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        ax.plot([row["full"], row["crop"]], [yi, yi], color="#c9d1d6", lw=2, zorder=1)
        ax.scatter(row["full"], yi, s=50, color="#1f6f5c", zorder=2, label="full image" if yi == y[0] else None)
        ax.scatter(row["crop"], yi, s=50, marker="s", color="#c98b5e", zorder=2, label="leaf crop" if yi == y[0] else None)
        ax.text(max(row["crop"], row["full"]) + 0.15, yi,
                f"{row['delta']:+.2f} pp (p={row['p']:.3g})",
                va="center", fontsize=8.2)
    labels = [r["tag"].replace("__", " ").replace("_", " ")[:58] for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("PlantDoc accuracy (%)")
    ax.set_title("Leaf-crop intervention: removing background recovers only 1–2 pp", fontsize=10)
    ax.grid(axis="x", color="#e7ebee", lw=0.8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("tag,model,full_acc,crop_acc,delta_pp,ci_low_pp,ci_high_pp,mcnemar_p\n")
        for row in rows:
            handle.write(
                f"{row['tag']},{row['model']},{row['full']:.3f},{row['crop']:.3f},"
                f"{row['delta']:.3f},{row['lo']:.3f},{row['hi']:.3f},{row['p']:.3g}\n"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
