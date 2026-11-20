"""Figure: per-class leaf-crop effect (mean of two seeds).

Output: figures/fig_crop_per_class.pdf and .png
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import pub_style as st

st.apply()

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUT = Path(__file__).resolve().parent

MODEL = "mobilenetv3_large"
SEEDS = ("42", "43")

per = defaultdict(lambda: {"delta": [], "full": [], "crop": []})
for seed in SEEDS:
    with open(METRICS / f"crop_{MODEL}__scratch__s{seed}.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    for key, value in payload["annotated_subset"]["per_class_delta"].items():
        per[key]["delta"].append(value["delta_pp"])
        per[key]["full"].append(value["recall_full"])
        per[key]["crop"].append(value["recall_crop"])

rows = []
for key, value in per.items():
    name = key.replace("___", " — ").replace("_", " ")
    name = name.replace("Tomato — Tomato Yellow Leaf Curl Virus", "Tomato — yellow leaf curl virus")
    name = name.replace("Tomato — Tomato mosaic virus", "Tomato — mosaic virus")
    name = name.replace("Tomato — Spider mites Two-spotted spider mite", "Tomato — two-spotted spider mite")
    name = name.replace("Tomato — Spider mites spider mites", "Tomato — two-spotted spider mite")
    name = name.replace("Apple — Apple scab", "Apple — scab")
    rows.append((name, np.mean(value["delta"]), np.mean(value["full"]),
                 np.mean(value["crop"])))
rows.sort(key=lambda item: item[1])

names = [r[0] for r in rows]
deltas = np.array([r[1] for r in rows])
colors = [st.NEG if d < 0 else st.POS for d in deltas]

fig, ax = plt.subplots(figsize=(5.0, 4.3))
y = np.arange(len(rows))
ax.barh(y, deltas, color=colors, height=0.68, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="#1B1F24", lw=0.8)
for yi, d in zip(y, deltas):
    if abs(d) < 0.05:
        continue
    ax.text(d + (0.15 if d >= 0 else -0.15), yi, f"{d:+.1f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=7.0)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlabel("Change in recall after cropping to leaf boxes (pp)")
ax.set_xlim(-4.0, 11.0)
ax.grid(axis="y", visible=False)
ax.set_title("Leaf-crop effect by disorder (Large SE, mean of two seeds)", fontsize=9.0)

fig.savefig(OUT / "fig_crop_per_class.pdf")
fig.savefig(OUT / "fig_crop_per_class.png")
print("wrote figures/fig_crop_per_class.{pdf,png}")
