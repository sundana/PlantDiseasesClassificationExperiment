"""Extract the 29-class PlantVillage subset from the HuggingFace parquet mirror.

Source: https://huggingface.co/datasets/Project-AgML/plant_village_classification
The dataset has 39 classes; this script keeps the 29 used in the manuscript
(28 leaf classes + Background_without_leaves).

Usage:
    python scripts/prepare_plantvillage.py \
        --parquet-glob "data/raw/plantvillage_hf/*.parquet" \
        --out data/plantvillage

Optional:
    --min-per-class 1000   augment classes below the threshold (off by default;
                           the manuscript's 46,564-image table used this scheme)
"""

import argparse
import glob
import io
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from torchvision import transforms

from src.constants import PV_CLASSES

# Order from the HuggingFace dataset card (label ids 0-38).
HF_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves", "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def augment_to_minimum(out_dir: str, counts: dict, minimum: int, seed: int = 42) -> dict:
    """Offline augmentation to a per-class minimum, matching the manuscript table."""
    rng = random.Random(seed)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    for class_name, count in list(counts.items()):
        if count >= minimum:
            continue
        class_dir = os.path.join(out_dir, class_name)
        files = sorted(os.listdir(class_dir))
        target = minimum - count
        for i in range(target):
            source = os.path.join(class_dir, rng.choice(files))
            with Image.open(source) as image:
                image = transform(image.convert("RGB"))
                image.save(os.path.join(class_dir, f"aug_{i:05d}.jpg"), quality=95)
        counts[class_name] = minimum
        print(f"  augmented {class_name}: {count} -> {minimum}")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-glob", default="data/raw/plantvillage_hf/*.parquet")
    parser.add_argument("--out", default="data/plantvillage")
    parser.add_argument("--min-per-class", type=int, default=0)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required: uv pip install pyarrow") from exc

    keep = set(PV_CLASSES)
    os.makedirs(args.out, exist_ok=True)
    counts = {name: 0 for name in PV_CLASSES}
    paths = sorted(glob.glob(args.parquet_glob))
    if not paths:
        raise SystemExit(f"no parquet files matched {args.parquet_glob}")

    for path in paths:
        parquet = pq.ParquetFile(path)
        print(f"reading {path} ({parquet.metadata.num_rows} rows)")
        for batch in parquet.iter_batches(batch_size=512, columns=["image", "label"]):
            data = batch.to_pydict()
            for image_struct, label in zip(data["image"], data["label"]):
                class_name = HF_CLASSES[int(label)]
                if class_name not in keep:
                    continue
                class_dir = os.path.join(args.out, class_name)
                os.makedirs(class_dir, exist_ok=True)
                index = counts[class_name]
                with Image.open(io.BytesIO(image_struct["bytes"])) as image:
                    image.convert("RGB").save(
                        os.path.join(class_dir, f"{index:06d}.jpg"),
                        quality=args.quality,
                    )
                counts[class_name] += 1
        print(f"  running total: {sum(counts.values())} images")

    if args.min_per_class:
        counts = augment_to_minimum(args.out, counts, args.min_per_class)

    print("\nclass counts")
    for name in PV_CLASSES:
        print(f"  {name:50s} {counts[name]}")
    print(f"total {sum(counts.values())}")


if __name__ == "__main__":
    main()
