"""Build the manuscript tables for the PAA revision.

Reads the audit prediction files and reports, for each condition:

* all-image metrics (the original audit convention), and
* official-test-split metrics (169 images).

Also recomputes the four equal-complexity attention pairs on the test split and
summarises the fine-tune runs when they are present.

Usage:
    .venv/bin/python scripts/report_paa_tables.py
    .venv/bin/python scripts/report_paa_tables.py --out results/paa_tables.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import classification_metrics, mcnemar_exact, paired_bootstrap_difference, tost_equivalence
from src.constants import EVAL_CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results", "predictions")
METRICS = os.path.join(ROOT, "results", "metrics")

CORE = {
    "mobilenetv3_large": "Large SE r=4",
    "proposed_large_32": "Large CBAM r=32",
    "mobilenetv3_large_none": "Large none",
}
INITS = {"scratch": "scratch", "imagenet": "pretrained"}


def read_rows(path: str) -> List[Dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def split_rows(rows: List[Dict], split: str) -> List[Dict]:
    if split == "all":
        return rows
    return [row for row in rows if row["image"].startswith("test/")]


def metrics_for(rows: List[Dict]) -> Dict:
    return classification_metrics([row["true"] for row in rows], [row["pred"] for row in rows])


def mean_sd(values: List[float]) -> Dict:
    return {
        "mean": float(statistics.mean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def core_table() -> List[Dict]:
    out = []
    for model, label in CORE.items():
        for init, init_label in INITS.items():
            for split in ("all", "test"):
                per_seed = []
                for seed in (42, 43):
                    path = os.path.join(PRED, f"{model}__{init}__s{seed}_plantdoc.csv")
                    if not os.path.exists(path):
                        continue
                    rows = split_rows(read_rows(path), split)
                    per_seed.append(metrics_for(rows))
                if not per_seed:
                    continue
                out.append(
                    {
                        "model": label,
                        "init": init_label,
                        "split": split,
                        "n_images": per_seed[0]["n"],
                        "accuracy": mean_sd([m["accuracy"] for m in per_seed]),
                        "f1": mean_sd([m["f1"] for m in per_seed]),
                        "background_rate": mean_sd([m["background_rate"] for m in per_seed]),
                        "unmapped_rate": mean_sd([m["unmapped_rate"] for m in per_seed]),
                    }
                )
    return out


def nobg_table() -> List[Dict]:
    out = []
    for model, label in CORE.items():
        for split in ("all", "test"):
            per_seed = []
            for seed in (42, 43):
                path = os.path.join(PRED, f"{model}__nobg__s{seed}_plantdoc.csv")
                if not os.path.exists(path):
                    continue
                rows = split_rows(read_rows(path), split)
                per_seed.append(metrics_for(rows))
            if not per_seed:
                continue
            out.append(
                {
                    "model": label,
                    "split": split,
                    "n_images": per_seed[0]["n"],
                    "accuracy": mean_sd([m["accuracy"] for m in per_seed]),
                    "f1": mean_sd([m["f1"] for m in per_seed]),
                }
            )
    return out


def border_table() -> List[Dict]:
    out = []
    for model, label in (("mobilenetv3_large", "Large SE"), ("mobilenetv3_small", "Small SE")):
        for split in ("all", "test"):
            per_seed = []
            for seed in (42, 43):
                path = os.path.join(PRED, f"border_{model}__s{seed}_plantdoc.csv")
                if not os.path.exists(path):
                    continue
                rows = split_rows(read_rows(path), split)
                per_seed.append(metrics_for(rows))
            if per_seed:
                out.append(
                    {
                        "model": label,
                        "split": split,
                        "n_images": per_seed[0]["n"],
                        "field_accuracy": mean_sd([m["accuracy"] for m in per_seed]),
                    }
                )
    return out


def crop_table() -> List[Dict]:
    out = []
    for model, label in CORE.items():
        for init, init_label in INITS.items():
            for split in ("all", "test"):
                per_seed = []
                for seed in (42, 43):
                    path = os.path.join(PRED, f"crop_{model}__{init}__s{seed}.csv")
                    if not os.path.exists(path):
                        continue
                    rows = split_rows(read_rows(path), split)
                    full = sum(int(row["correct_full"]) for row in rows) / len(rows) * 100
                    crop = sum(int(row["correct_crop"]) for row in rows) / len(rows) * 100
                    per_seed.append((full, crop))
                if per_seed:
                    out.append(
                        {
                            "model": label,
                            "init": init_label,
                            "split": split,
                            "n_images": len(split_rows(read_rows(path), split)),
                            "full": mean_sd([x[0] for x in per_seed]),
                            "crop": mean_sd([x[1] for x in per_seed]),
                            "delta": mean_sd([x[1] - x[0] for x in per_seed]),
                        }
                    )
    return out


def attention_table() -> List[Dict]:
    pairs = [
        ("Large r=16", "mobilenetv3_large_se_r16__scratch__s42_plantdoc.csv", "proposed_large_16__scratch__s42_plantdoc.csv"),
        ("Large r=32", "mobilenetv3_large_se_r32__scratch__s42_plantdoc.csv", "proposed_large_32__scratch__s42_plantdoc.csv"),
        ("Small r=16", "mobilenetv3_small_se_r16__scratch__s42_plantdoc.csv", "proposed_small_16__scratch__s42_plantdoc.csv"),
        ("Small r=32", "mobilenetv3_small_se_r32__scratch__s42_plantdoc.csv", "proposed_small_32__scratch__s42_plantdoc.csv"),
    ]
    out = []
    for name, file_a, file_b in pairs:
        path_a = os.path.join(PRED, file_a)
        path_b = os.path.join(PRED, file_b)
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue
        rows_a = {row["image"]: row for row in read_rows(path_a)}
        rows_b = {row["image"]: row for row in read_rows(path_b)}
        keys = [key for key in rows_a if key.startswith("test/") and key in rows_b]
        truth = [rows_a[key]["true"] for key in keys]
        pred_a = [rows_a[key]["pred"] for key in keys]
        pred_b = [rows_b[key]["pred"] for key in keys]
        acc_a = sum(p == t for p, t in zip(pred_a, truth)) / len(keys) * 100
        acc_b = sum(p == t for p, t in zip(pred_b, truth)) / len(keys) * 100
        mcnemar = mcnemar_exact(truth, pred_a, pred_b)
        bootstrap = paired_bootstrap_difference(truth, pred_a, pred_b, n_bootstrap=10000, seed=42)
        tost = tost_equivalence(truth, pred_a, pred_b, margin_pp=1.0, n_bootstrap=10000, seed=42)
        out.append(
            {
                "pair": name,
                "n_images": len(keys),
                "se_accuracy": acc_a,
                "cbam_accuracy": acc_b,
                "difference_pp": acc_a - acc_b,
                "mcnemar_p": mcnemar["p_value"],
                "ci_low_pp": bootstrap["ci_low_pp"],
                "ci_high_pp": bootstrap["ci_high_pp"],
                "tost_equivalent": tost["equivalent"],
            }
        )
    return out


def finetune_table() -> List[Dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(METRICS, "finetune_*.json"))):
        with open(path) as handle:
            data = json.load(handle)
        if data.get("run_tag", "").startswith("smoke") or "timing" in data.get("run_tag", ""):
            continue
        out.append(
            {
                "run_tag": data["run_tag"],
                "model": data["model"],
                "seed": data["seed"],
                "best_epoch": data.get("best_epoch"),
                "epochs_run": data.get("epochs_run"),
                "field_accuracy": data["field"]["accuracy"],
                "field_f1": data["field"]["f1"],
                "n_test": data.get("n_test"),
            }
        )
    return out


def print_markdown(core, nobg, border, crop, attention, finetune) -> None:
    print("\n## Core models (test split)\n")
    print("| Model | Init | n | Accuracy | Macro F1 | BG rate |")
    print("|---|---|---:|---:|---:|---:|")
    for row in core:
        if row["split"] != "test":
            continue
        print(
            f"| {row['model']} | {row['init']} | {row['n_images']} | "
            f"{row['accuracy']['mean']:.2f} ± {row['accuracy']['sd']:.2f} | "
            f"{row['f1']['mean']:.2f} ± {row['f1']['sd']:.2f} | "
            f"{row['background_rate']['mean']:.1f} ± {row['background_rate']['sd']:.1f} |"
        )

    print("\n## Background ablation (test split)\n")
    print("| Model | n | Accuracy | Macro F1 |")
    print("|---|---:|---:|---:|")
    for row in nobg:
        if row["split"] != "test":
            continue
        print(f"| {row['model']} | {row['n_images']} | {row['accuracy']['mean']:.2f} ± {row['accuracy']['sd']:.2f} | {row['f1']['mean']:.2f} ± {row['f1']['sd']:.2f} |")

    print("\n## Border probe (test split)\n")
    print("| Model | n | Field accuracy |")
    print("|---|---:|---:|")
    for row in border:
        if row["split"] != "test":
            continue
        print(f"| {row['model']} | {row['n_images']} | {row['field_accuracy']['mean']:.2f} ± {row['field_accuracy']['sd']:.2f} |")

    print("\n## Leaf crop (test split)\n")
    print("| Model | Init | n | Full | Crop | Delta |")
    print("|---|---|---:|---:|---:|---:|")
    for row in crop:
        if row["split"] != "test":
            continue
        print(
            f"| {row['model']} | {row['init']} | {row['n_images']} | "
            f"{row['full']['mean']:.2f} | {row['crop']['mean']:.2f} | "
            f"{row['delta']['mean']:+.2f} ± {row['delta']['sd']:.2f} |"
        )

    print("\n## Attention pairs (test split, seed 42)\n")
    print("| Pair | n | SE | CBAM | Diff | McNemar p | 95% CI | TOST |")
    print("|---|---:|---:|---:|---:|---:|---|---|")
    for row in attention:
        print(
            f"| {row['pair']} | {row['n_images']} | {row['se_accuracy']:.2f} | {row['cbam_accuracy']:.2f} | "
            f"{row['difference_pp']:+.2f} | {row['mcnemar_p']:.3f} | "
            f"[{row['ci_low_pp']:+.2f}, {row['ci_high_pp']:+.2f}] | {row['tost_equivalent']} |"
        )

    if finetune:
        print("\n## Fine-tune (test split)\n")
        print("| Run | Model | Seed | Best epoch | Accuracy | Macro F1 |")
        print("|---|---|---:|---:|---:|---:|")
        for row in finetune:
            print(f"| {row['run_tag']} | {row['model']} | {row['seed']} | {row['best_epoch']} | {row['field_accuracy']:.2f} | {row['field_f1']:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "paa_tables.json"))
    args = parser.parse_args()

    payload = {
        "core": core_table(),
        "nobg": nobg_table(),
        "border": border_table(),
        "crop": crop_table(),
        "attention": attention_table(),
        "finetune": finetune_table(),
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {args.out}")
    print_markdown(**payload)


if __name__ == "__main__":
    main()
