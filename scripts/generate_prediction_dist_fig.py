"""Generate plantdoc_prediction_distribution.png for the paper."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from src.config import CHECKPOINT_DIR, PLANTDOC_DIR, ROOT_DIR
from scripts.validate_plantdoc import PlantDocMappedDataset, PLANTDOC_PV_MAPPING, PV_CLASSES

PAPER_DIR = os.path.join(os.path.dirname(ROOT_DIR), "paper")

MODELS_CONFIG = [
    ("Large SE r=4", partial(MobileNetV3_Large, attention_type='se'),
     "MobileNetV3_Large_SE_epoch_39", "#2196F3"),
    ("Large CBAM r=16", partial(MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
     "MobileNetV3_Large_CBAM_r16_epoch_22", "#FF9800"),
    ("Small SE r=4", partial(MobileNetV3_Small, attention_type='se'),
     "MobileNetV3_Small_SE_epoch_19", "#4CAF50"),
]

BACKGROUND_CLASS_IDX = 4
BACKGROUND_CLASS_NAME = "Background\nw/o leaves"
UNMAPPED_INDICES = {1, 4, 6, 9, 10, 12, 13, 17, 25}

def shorten_class_name(name, max_len=14):
    parts = name.split("___")
    if len(parts) == 2:
        short = parts[1].replace("_", " ")[:max_len]
    else:
        short = name.replace("_", " ")[:max_len]
    return short

def collect_predictions(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PlantDocMappedDataset(
        root_dir=PLANTDOC_DIR,
        mapping=PLANTDOC_PV_MAPPING,
        pv_classes=PV_CLASSES,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Dataset size: {len(dataset)}")

    results = []
    for label, model_fn, ckpt_name, color in MODELS_CONFIG:
        print(f"\nLoading {label} from {ckpt_name}...")
        model = model_fn(num_classes=len(PV_CLASSES)).to(device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pth")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if ".se.se." in k:
                k = k.replace(".se.se.", ".attention_module.se.")
            if ".module.channel_attention.fc1." in k:
                k = k.replace(".module.channel_attention.fc1.", ".attention_module.channel_attention.se.0.")
            if ".module.channel_attention.fc2." in k:
                k = k.replace(".module.channel_attention.fc2.", ".attention_module.channel_attention.se.2.")
            if ".module.spatial_attention.conv." in k:
                k = k.replace(".module.spatial_attention.conv.", ".attention_module.spatial_attention.conv.")
            if "bneck" in k and ".module." in k:
                k = k.replace(".module.", ".attention_module.")
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)

        all_preds, all_labels = collect_predictions(model, dataloader, device)
        acc = (all_preds == all_labels).mean() * 100

        mapped_indices = {PV_CLASSES.index(pv_class) for pv_class in PLANTDOC_PV_MAPPING.values()}
        unmapped_preds = sum(1 for p in all_preds if p not in mapped_indices)
        bg_preds = sum(1 for p in all_preds if p == BACKGROUND_CLASS_IDX)

        print(f"  Accuracy: {acc:.2f}%, Unmapped: {unmapped_preds/len(all_preds)*100:.1f}%, "
              f"Background: {bg_preds/len(all_preds)*100:.1f}%")

        results.append({
            "label": label,
            "color": color,
            "preds": all_preds,
            "acc": acc,
            "unmapped_pct": unmapped_preds / len(all_preds) * 100,
            "bg_pct": bg_preds / len(all_preds) * 100,
        })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Prediction Distribution on PlantDoc (Zero-Shot Transfer)", fontsize=14, fontweight="bold", y=1.01)

    for ax, res in zip(axes, results):
        preds = res["preds"]
        counter = Counter(preds)

        classes_to_show = sorted(counter.keys())
        mapped_indices = {PV_CLASSES.index(pv_class) for pv_class in PLANTDOC_PV_MAPPING.values()}
        counts = [counter.get(i, 0) for i in classes_to_show]
        names = [shorten_class_name(PV_CLASSES[i]) for i in classes_to_show]
        pcts = [counter.get(i, 0) / len(preds) * 100 for i in classes_to_show]

        order = np.argsort(pcts)[::-1]
        classes_to_show = [classes_to_show[i] for i in order]
        names = [names[i] for i in order]
        pcts = [pcts[i] for i in order]

        n_display = 12
        display_indices = list(range(n_display))
        display_names = names[:n_display]
        display_pcts = pcts[:n_display]
        display_classes = classes_to_show[:n_display]

        colors = []
        for cidx in display_classes:
            if cidx == BACKGROUND_CLASS_IDX:
                colors.append("#E53935")
            elif cidx not in mapped_indices:
                colors.append("#BDBDBD")
            else:
                colors.append(res["color"])

        y_pos = range(len(display_names))
        bars = ax.barh(y_pos, display_pcts, color=colors, edgecolor="white", linewidth=0.5)

        for i, (pct, cidx) in enumerate(zip(display_pcts, display_classes)):
            if pct > 3:
                ax.text(pct + 0.5, i, f"{pct:.1f}%", va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(display_pcts) * 1.25)
        ax.set_xlabel("% of predictions", fontsize=9)

        unmapped_pct = res["unmapped_pct"]
        ax.set_title(
            f"{res['label']}\nAcc: {res['acc']:.1f}% | "
            f"Bg: {res['bg_pct']:.1f}% | "
            f"Unmapped: {unmapped_pct:.1f}%",
            fontsize=10, fontweight="bold",
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E53935", label="Background w/o leaves"),
            Patch(facecolor="#BDBDBD", label="Unmapped class"),
            Patch(facecolor=res["color"], label="Mapped class"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right",
                  framealpha=0.9, edgecolor="gray")

    plt.tight_layout()
    output_path = os.path.join(PAPER_DIR, "plantdoc_prediction_distribution.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
