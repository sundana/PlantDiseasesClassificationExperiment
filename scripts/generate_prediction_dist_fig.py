"""Generate plantdoc_prediction_distribution.png — v3.
Fixes: unified x-scale 0-40%, proper spacing, no text collision, clean titles."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.config import CHECKPOINT_DIR, PLANTDOC_DIR, ROOT_DIR
from scripts.validate_plantdoc import PlantDocMappedDataset, PLANTDOC_PV_MAPPING, PV_CLASSES

PAPER_DIR = os.path.join(os.path.dirname(ROOT_DIR), "paper")

MODELS = [
    ("Large SE (r=4)", partial(MobileNetV3_Large, attention_type='se'),
     "MobileNetV3_Large_SE_epoch_39"),
    ("Large CBAM (r=16)", partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
     "MobileNetV3_Large_CBAM_r16_epoch_22"),
    ("Small SE (r=4)", partial(MobileNetV3_Small, attention_type='se'),
     "MobileNetV3_Small_SE_epoch_19"),
]

BG_IDX = 4
UNMAPPED = {1, 4, 6, 9, 10, 12, 13, 17, 25}
CLR_MAPPED   = "#2196F3"
CLR_UNMAPPED = "#BDBDBD"
CLR_BG       = "#E53935"


def shorten(name, n=16):
    p = name.split("___", 1)
    s = p[1].replace("_", " ") if len(p) == 2 else name.replace("_", " ")
    return s[:n]


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PlantDocMappedDataset(
        root_dir=PLANTDOC_DIR, mapping=PLANTDOC_PV_MAPPING,
        pv_classes=PV_CLASSES, transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    mapped_ix = {PV_CLASSES.index(v) for v in PLANTDOC_PV_MAPPING.values()}

    results = []
    for label, model_fn, ckpt_name in MODELS:
        print(f"Loading {label} ...")
        model = model_fn(num_classes=len(PV_CLASSES)).to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pth")
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        nsd = {}
        for k, v in sd.items():
            if ".se.se." in k:           k = k.replace(".se.se.", ".attention_module.se.")
            if ".module.channel_attention.fc1." in k: k = k.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
            if ".module.channel_attention.fc2." in k: k = k.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
            if ".module.spatial_attention.conv." in k: k = k.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
            if "bneck" in k and ".module." in k:  k = k.replace(".module.", ".attention_module.")
            nsd[k] = v
        model.load_state_dict(nsd, strict=False)

        all_preds, all_labels = [], []
        model.eval()
        with torch.inference_mode():
            for imgs, lbls in dataloader:
                out = model(imgs.to(device))
                all_preds.extend(out.argmax(dim=1).cpu().tolist())
                all_labels.extend(lbls.tolist())
        preds = np.array(all_preds); labels = np.array(all_labels)
        acc = (preds == labels).mean() * 100
        ctr = Counter(preds.tolist())
        unmapped_pct = sum(1 for p in preds if p not in mapped_ix) / len(preds) * 100
        bg_pct = sum(1 for p in preds if p == BG_IDX) / len(preds) * 100

        items = sorted(ctr.items(), key=lambda x: -x[1])[:8]
        results.append({
            "label": label, "acc": acc, "unmapped_pct": unmapped_pct, "bg_pct": bg_pct,
            "names": [shorten(PV_CLASSES[i]) for i, _ in items],
            "pcts":  [c / len(preds) * 100 for _, c in items],
            "idxs":  [i for i, _ in items],
        })
        print(f"  Acc={acc:.1f}%  Bg={bg_pct:.1f}%  Unmapped={unmapped_pct:.1f}%")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    UNIFIED_XMAX = 40

    for ax, res in zip(axes, results):
        cols = [CLR_BG if i == BG_IDX else CLR_UNMAPPED if i not in mapped_ix else CLR_MAPPED
                for i in res["idxs"]]
        y = range(len(res["names"]))
        ax.barh(y, res["pcts"], color=cols, edgecolor="white", linewidth=0.5, height=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels(res["names"], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, UNIFIED_XMAX)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(res["label"], fontsize=10.5, fontweight="bold", pad=4)
        ax.text(0.98, 0.96,
                f"Acc: {res['acc']:.1f}%\nBg: {res['bg_pct']:.1f}%\nUnm: {res['unmapped_pct']:.1f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85))

    axes[1].set_xlabel("% of predictions", fontsize=9, labelpad=2)

    fig.legend(
        handles=[Patch(facecolor=CLR_BG, label="Bg w/o leaves"),
                 Patch(facecolor=CLR_UNMAPPED, label="Unmapped"),
                 Patch(facecolor=CLR_MAPPED, label="Mapped")],
        loc="lower center", ncol=3, fontsize=8.5, frameon=True,
        fancybox=False, edgecolor="gray",
    )

    plt.subplots_adjust(wspace=0.42, bottom=0.16, left=0.04, right=0.98)
    output_path = os.path.join(PAPER_DIR, "plantdoc_prediction_distribution.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
