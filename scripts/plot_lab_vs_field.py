"""Plot laboratory vs field accuracy from metrics.json files.

Usage:
    python scripts/plot_lab_vs_field.py \
        --metrics-glob "results/metrics/*.json" \
        --out results/plots/lab_vs_field.png
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get(payload, dotted):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


PRETTY = {
    "mobilenetv3_large": "Large SE r=4",
    "proposed_large_32": "Large CBAM r=32",
    "mobilenetv3_large_none": "Large none",
    "mobilenetv3_small": "Small SE r=4",
    "mobilenetv2": "MobileNetV2",
    "shufflenetv2": "ShuffleNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-glob", default="results/metrics/*.json")
    parser.add_argument("--out", default="results/plots/lab_vs_field.png")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--include-probes", action="store_true")
    args = parser.parse_args()

    groups = defaultdict(lambda: {"lab": [], "field": [], "init": ""})
    for path in sorted(glob.glob(args.metrics_glob)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("probe") and not args.include_probes:
            continue
        tag = payload.get("run_tag", "")
        if tag.startswith("smoke") and not args.include_smoke:
            continue
        lab = _get(payload, "lab.accuracy")
        field = _get(payload, "field.accuracy")
        if lab is None:
            continue
        key = (payload.get("model", "?"), "pretrained" if payload.get("pretrained") else "scratch")
        groups[key]["lab"].append(lab)
        if field is not None:
            groups[key]["field"].append(field)

    rows = []
    for (model, init), values in groups.items():
        rows.append({
            "model": model,
            "init": init,
            "lab": float(np.mean(values["lab"])) if values["lab"] else np.nan,
            "lab_sd": float(np.std(values["lab"], ddof=1)) if len(values["lab"]) > 1 else 0.0,
            "field": float(np.mean(values["field"])) if values["field"] else np.nan,
            "field_sd": float(np.std(values["field"], ddof=1)) if len(values["field"]) > 1 else 0.0,
            "n": len(values["lab"]),
        })
    rows.sort(key=lambda r: (r["init"], -r["lab"]))
    if not rows:
        raise SystemExit("no metrics with a lab accuracy found")

    fig, ax = plt.subplots(figsize=(8.4, 0.55 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        ax.plot([row["lab"], row["field"]], [yi, yi], color="#c9d1d6", lw=2, zorder=1)
        ax.scatter(row["lab"], yi, s=46, color="#1f6f5c", zorder=2, label="laboratory" if yi == y[0] else None)
        ax.scatter(row["field"], yi, s=46, marker="s", color="#c98b5e", zorder=2, label="field" if yi == y[0] else None)
        if not np.isnan(row["field"]):
            ax.text(row["field"] + 1.2, yi, f"{row['field']:.1f}", va="center", fontsize=8, color="#a3541b")
        ax.text(row["lab"] - 1.2, yi, f"{row['lab']:.1f}", va="center", ha="right", fontsize=8, color="#1f6f5c")

    labels = [f"{PRETTY.get(r['model'], r['model'])} [{r['init'][:4]}] n={r['n']}" for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Laboratory vs zero-shot field accuracy (PlantVillage → PlantDoc)", fontsize=10)
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
        handle.write("model,init,n,lab_mean,lab_sd,field_mean,field_sd\n")
        for row in rows:
            handle.write(
                f"{row['model']},{row['init']},{row['n']},{row['lab']:.3f},{row['lab_sd']:.3f},"
                f"{row['field']:.3f},{row['field_sd']:.3f}\n"
            )
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
