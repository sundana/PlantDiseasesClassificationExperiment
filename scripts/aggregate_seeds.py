"""Aggregate seed runs and run paired model comparisons.

Two modes:

1. ``aggregate`` (default) reads every ``metrics.json`` produced by
   ``train.py`` / ``audit_border.py`` and reports mean +/- SD per model and
   initialisation condition.

       python scripts/aggregate_seeds.py --metrics-glob "results/metrics/*.json"

2. ``compare`` aligns two prediction CSVs on the image path and reports the
   paired accuracy difference, a bootstrap CI, exact McNemar, and a TOST
   equivalence decision. Use it for SE vs CBAM at equal reduction ratio.

       python scripts/aggregate_seeds.py compare \
           --a results/predictions/V3L_SE4__scratch__s42_plantdoc.csv \
           --b results/predictions/V3L_CBAM32__scratch__s42_plantdoc.csv \
           --margin 1.0
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit import (
    mcnemar_exact,
    paired_bootstrap_difference,
    read_predictions,
    tost_equivalence,
)

METRIC_PATHS = [
    "lab.accuracy",
    "lab.precision",
    "lab.recall",
    "lab.f1",
    "lab_20class.accuracy",
    "lab_20class.f1",
    "field.accuracy",
    "field.precision",
    "field.recall",
    "field.f1",
    "field.background_rate",
    "field.unmapped_rate",
]


def _get(payload, dotted):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _group_key(payload):
    excluded = payload.get("excluded_classes") or []
    return (
        payload.get("model", "?"),
        "pretrained" if payload.get("pretrained") else "scratch",
        payload.get("num_classes", "?"),
        len(excluded),
        payload.get("probe", "train"),
    )


def cmd_aggregate(args):
    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise SystemExit(f"no metrics.json matched {args.metrics_glob}")
    groups = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("probe") and not args.include_probes:
            continue
        if str(payload.get("run_tag", "")).startswith("smoke") and not args.include_smoke:
            continue
        if "lab" not in payload or "model" not in payload:
            continue  # comparison JSONs and other non-run files
        groups[_group_key(payload)].append((path, payload))

    rows = []
    for key, items in sorted(groups.items()):
        model, init, num_classes, n_excluded, probe = key
        row = {
            "model": model,
            "init": init,
            "num_classes": num_classes,
            "n_excluded": n_excluded,
            "probe": probe,
            "n_runs": len(items),
        }
        for metric in METRIC_PATHS:
            values = [
                _get(payload, metric) for _, payload in items
                if _get(payload, metric) is not None
            ]
            if values:
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
                row[f"{metric}_n"] = len(values)
        rows.append(row)

    fieldnames = ["model", "init", "num_classes", "n_excluded", "probe", "n_runs"]
    for metric in METRIC_PATHS:
        fieldnames += [f"{metric}_mean", f"{metric}_sd", f"{metric}_n"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {args.out}\n")
    print(f"{'model':<28}{'init':<11}{'K':>3}{'n':>4}"
          f"{'lab acc':>10}{'field acc':>12}{'field F1':>11}{'bg rate':>10}")
    print("-" * 89)
    for row in rows:
        def fmt(metric):
            mean = row.get(f"{metric}_mean")
            sd = row.get(f"{metric}_sd")
            return "-" if mean is None else f"{mean:.2f}+-{sd:.2f}"
        print(
            f"{row['model']:<28}{row['init']:<11}{row['num_classes']:>3}"
            f"{row['n_runs']:>4}  {fmt('lab.accuracy'):>10}  "
            f"{fmt('field.accuracy'):>12}  {fmt('field.f1'):>11}  "
            f"{fmt('field.background_rate'):>10}"
        )


def _load_predictions(path, column):
    rows = read_predictions(path)
    if not rows:
        raise SystemExit(f"empty prediction file: {path}")
    if column not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise SystemExit(f"column {column!r} not in {path}; available: {available}")
    table = {}
    for row in rows:
        table[row["image"]] = (row["true"], row[column])
    return table


def cmd_compare(args):
    a = _load_predictions(args.a, args.column_a)
    b = _load_predictions(args.b, args.column_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two prediction files share no image paths")
    if len(shared) != len(a) or len(shared) != len(b):
        print(f"[warn] aligned on {len(shared)} images "
              f"(A has {len(a)}, B has {len(b)})")
    y_true = []
    pred_a = []
    pred_b = []
    mismatched_truth = 0
    for image in shared:
        true_a, pa = a[image]
        true_b, pb = b[image]
        if true_a != true_b:
            mismatched_truth += 1
            continue
        y_true.append(true_a)
        pred_a.append(pa)
        pred_b.append(pb)
    if mismatched_truth:
        print(f"[warn] dropped {mismatched_truth} images with conflicting true labels")
    n = len(y_true)
    if n == 0:
        raise SystemExit("no comparable images remain")

    correct_a = sum(int(t == p) for t, p in zip(y_true, pred_a))
    correct_b = sum(int(t == p) for t, p in zip(y_true, pred_b))
    acc_a = 100.0 * correct_a / n
    acc_b = 100.0 * correct_b / n
    mcnemar = mcnemar_exact(y_true, pred_a, pred_b)
    boot = paired_bootstrap_difference(
        y_true, pred_a, pred_b, n_bootstrap=args.bootstrap, seed=args.seed
    )
    tost = tost_equivalence(
        y_true, pred_a, pred_b, margin_pp=args.margin,
        n_bootstrap=args.bootstrap, seed=args.seed,
    )

    print(f"Compared {n} images")
    print(f"A ({args.name_a}): {acc_a:.2f}%")
    print(f"B ({args.name_b}): {acc_b:.2f}%")
    print(f"Difference (A-B): {acc_a - acc_b:+.2f} pp")
    print(f"Bootstrap 95% CI: [{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] pp")
    print(f"McNemar exact p: {mcnemar['p_value']:.3g} "
          f"(n01={mcnemar['n01']}, n10={mcnemar['n10']})")
    print(f"TOST margin +/-{args.margin} pp: "
          f"{'EQUIVALENT' if tost['equivalent'] else 'not shown equivalent'}")
    print(f"  TOST 90% CI: [{tost['ci90_low_pp']:+.2f}, {tost['ci90_high_pp']:+.2f}] pp")

    if args.out:
        payload = {
            "a": args.a,
            "b": args.b,
            "name_a": args.name_a,
            "name_b": args.name_b,
            "n": n,
            "accuracy_a": acc_a,
            "accuracy_b": acc_b,
            "difference_pp": acc_a - acc_b,
            "mcnemar": mcnemar,
            "bootstrap": boot,
            "tost": tost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    agg = sub.add_parser("aggregate", help="mean +/- SD across seed runs")
    agg.add_argument("--metrics-glob", default="results/metrics/*.json")
    agg.add_argument("--out", default="results/metrics_summary.csv")
    agg.add_argument("--include-probes", action="store_true",
                     help="include border-probe runs in the summary")
    agg.add_argument("--include-smoke", action="store_true",
                     help="include one-epoch smoke runs in the summary")
    agg.set_defaults(func=cmd_aggregate)

    cmp_ = sub.add_parser("compare", help="paired comparison of two prediction CSVs")
    cmp_.add_argument("--a", required=True)
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--name-a", default="A")
    cmp_.add_argument("--name-b", default="B")
    cmp_.add_argument("--column-a", default="pred")
    cmp_.add_argument("--column-b", default="pred")
    cmp_.add_argument("--margin", type=float, default=1.0,
                      help="TOST equivalence margin in percentage points")
    cmp_.add_argument("--bootstrap", type=int, default=10000)
    cmp_.add_argument("--seed", type=int, default=42)
    cmp_.add_argument("--out", default=None)
    cmp_.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["aggregate"])
    args.func(args)


if __name__ == "__main__":
    main()
