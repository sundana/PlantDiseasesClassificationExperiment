"""Generate Grad-CAM for augment vs standard models on PlantDoc images."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
import cv2

from src.models.mobilenetv3 import MobileNetV3_Large
from src.config import CHECKPOINT_DIR, PLANTDOC_DIR, RESULTS_DIR

MODELS = [
    ("Augment", partial(MobileNetV3_Large, attention_type='se'),
     "mobilenetv3_large_augment_epoch_best.pth"),
    ("Standard", partial(MobileNetV3_Large, attention_type='se'),
     "MobileNetV3_Large_SE_epoch_39.pth"),
]

def find_last_conv2d(model):
    """Return the name of the last Conv2d layer (before global pooling)."""
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output.softmax(dim=1)[0, class_idx].item()

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pv_ds = datasets.ImageFolder(root=os.path.join(os.path.dirname(PLANTDOC_DIR), "plantvillage"))
    pv_classes = pv_ds.classes

    # Hand-pick diverse PlantDoc images
    pd_ds = datasets.ImageFolder(root=PLANTDOC_DIR)
    target_classes = {
        'Tomato leaf late blight': ('Late Blight', 'Symptoms visible'),
        'Apple Scab Leaf': ('Apple Scab', 'Lesions on leaf'),
        'Squash Powdery mildew leaf': ('Powdery Mildew', 'White coating'),
        'Tomato leaf': ('Tomato Healthy', 'No symptoms'),
        'grape leaf black rot': ('Grape Black Rot', 'Dark spots'),
        'Tomato Septoria leaf spot': ('Septoria', 'Small spots'),
    }
    samples = []
    for path, target in pd_ds.samples:
        cls_name = pd_ds.classes[target]
        if cls_name in target_classes:
            samples.append((path, cls_name))
        if len(samples) >= 6:
            break

    print(f"Sample images: {len(samples)}")

    fig, axes = plt.subplots(2, len(samples) + 1, figsize=(3.5 * (len(samples) + 1), 7))

    for row_idx, (label, model_fn, ckpt_name) in enumerate(MODELS):
        print(f"\n{label}: {ckpt_name}")
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)

        model = model_fn(num_classes=len(pv_classes)).to(device)
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
        model.eval()

        last_conv_name, last_conv_module = find_last_conv2d(model)
        print(f"  Target layer: {last_conv_name}")
        gradcam = GradCAM(model, last_conv_module)

        for col_idx, (img_path, cls_name) in enumerate(samples):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            cam, _ = gradcam(img_tensor)
            pred_idx = model(img_tensor).argmax(dim=1).item()
            pred_name = pv_classes[pred_idx].replace("___", " ").replace("_", " ")

            img_np = np.array(img.resize((224, 224)))
            cam_resized = cv2.resize(cam, (224, 224))
            heatmap = plt.cm.jet(cam_resized)[:, :, :3]
            overlay = (0.4 * img_np.astype(np.float32) / 255.0 + 0.6 * heatmap.astype(np.float32))
            overlay = np.clip(overlay, 0, 1)

            short_name = cls_name.split(" leaf")[0] if "leaf" in cls_name else cls_name.split(" ")[0]
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_title(f"{short_name}\nPred: {pred_name[:20]}", fontsize=7)
            axes[row_idx, col_idx].axis("off")

        axes[row_idx, -1].text(0.5, 0.5, f"{label}\nModel", ha="center", va="center",
                               fontsize=11, fontweight="bold")
        axes[row_idx, -1].axis("off")

    plt.suptitle("Grad-CAM: What the Model Attends To on PlantDoc Images", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    output_dir = os.path.join(RESULTS_DIR, "gradcam")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "augment_gradcam.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
