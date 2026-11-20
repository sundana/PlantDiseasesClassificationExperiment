"""Central model factory with optional ImageNet initialisation.

Two tracks are supported:

1. ``custom`` — the MobileNetV3 Large/Small implementations in this repository
   with SE, CBAM or no attention. When ``pretrained=True`` the *backbone*
   (stem, inverted-residual blocks and final 1x1 convolution) is initialised
   from torchvision's ImageNet weights. Attention modules and the classifier
   head stay randomly initialised. This is deliberate: it keeps the SE vs CBAM
   comparison free of any ImageNet advantage that one attention type might
   have over the other.
2. ``torchvision`` — MobileNetV2, ShuffleNetV2 and EfficientNet-B0, optionally
   with their official ImageNet weights.

The custom MobileNetV3 uses different module names from torchvision, so the
backbone weights are copied tensor by tensor. Any unmapped backbone tensor is
reported and, for the feature extractor, treated as an error.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch.nn as nn

from src.models.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

CUSTOM_SPECS = {
    "mobilenetv3_small": ("small", "se", 4),
    "mobilenetv3_large": ("large", "se", 4),
    "mobilenetv3_small_none": ("small", "none", 4),
    "mobilenetv3_large_none": ("large", "none", 4),
    "mobilenetv3_small_se_r16": ("small", "se", 16),
    "mobilenetv3_large_se_r16": ("large", "se", 16),
    "mobilenetv3_small_se_r32": ("small", "se", 32),
    "mobilenetv3_large_se_r32": ("large", "se", 32),
    "proposed_large_4": ("large", "cbam", 4),
    "proposed_large_16": ("large", "cbam", 16),
    "proposed_large_32": ("large", "cbam", 32),
    "proposed_small_4": ("small", "cbam", 4),
    "proposed_small_16": ("small", "cbam", 16),
    "proposed_small_32": ("small", "cbam", 32),
}

TORCHVISION_ARCHS = {
    "mobilenetv2": "mobilenet_v2",
    "shufflenetv2": "shufflenet_v2_x1_0",
    "efficientnet_b0": "efficientnet_b0",
}

CUSTOM_TO_TV_ARCH = {
    "mobilenetv3_small": "mobilenet_v3_small",
    "mobilenetv3_large": "mobilenet_v3_large",
}

ALL_MODELS = list(CUSTOM_SPECS) + list(TORCHVISION_ARCHS)


def _weights_enum(arch: str):
    """Return the best available ImageNet weights enum for an architecture."""
    from torchvision import models

    table = {
        "mobilenet_v3_large": ("MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
        "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", "IMAGENET1K_V1"),
        "mobilenet_v2": ("MobileNet_V2_Weights", "IMAGENET1K_V2"),
        "shufflenet_v2_x1_0": ("ShuffleNet_V2_X1_0_Weights", "IMAGENET1K_V1"),
        "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    }
    enum_name, member = table[arch]
    enum = getattr(models, enum_name, None)
    return getattr(enum, member, None) if enum is not None else None


def _build_torchvision(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models

    weights = _weights_enum(arch) if pretrained else None
    if pretrained and weights is None:
        raise RuntimeError(
            f"No ImageNet weights enum found for {arch} in this torchvision version. "
            "Refusing to run a 'pretrained' condition that would silently be random."
        )
    try:
        if arch == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=weights)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=weights)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif arch == "mobilenet_v2":
            model = models.mobilenet_v2(weights=weights)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        elif arch == "shufflenet_v2_x1_0":
            model = models.shufflenet_v2_x1_0(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"unknown torchvision architecture: {arch}")
    except Exception as exc:  # pragma: no cover - depends on local weight cache
        raise RuntimeError(
            f"Could not build {arch} with pretrained={pretrained}. "
            "If the machine is offline, pre-download the weights and set TORCH_HOME. "
            f"Original error: {exc}"
        ) from exc
    return model


def _bn_pairs(prefix: str, bn: nn.BatchNorm2d) -> Dict[str, "torch.Tensor"]:
    return {
        f"{prefix}.weight": bn.weight,
        f"{prefix}.bias": bn.bias,
        f"{prefix}.running_mean": bn.running_mean,
        f"{prefix}.running_var": bn.running_var,
        f"{prefix}.num_batches_tracked": bn.num_batches_tracked,
    }


def _stage_conv_bn(module: nn.Module):
    """Return (conv, bn) from a torchvision Conv2dNormActivation stage."""
    conv = next((m for m in module if isinstance(m, nn.Conv2d)), None)
    bn = next((m for m in module if isinstance(m, nn.BatchNorm2d)), None)
    if conv is None or bn is None:
        raise RuntimeError(f"stage has no conv/bn pair: {module}")
    return conv, bn


def _identity_init_expand(block: nn.Module) -> None:
    """Make the custom 1x1 expand conv an identity.

    torchvision omits the expand stage when expanded_channels == input_channels;
    the custom block always has one. Initialising it to the identity keeps the
    function identical to torchvision.
    """
    import torch

    with torch.no_grad():
        weight = block.conv1.weight
        weight.zero_()
        n = min(weight.shape[0], weight.shape[1])
        for j in range(n):
            weight[j, j, 0, 0] = 1.0
        block.bn1.weight.fill_(1.0)
        block.bn1.bias.zero_()
        block.bn1.running_mean.zero_()
        block.bn1.running_var.fill_(1.0)


def _zero_init_shortcut(block: nn.Module) -> None:
    """Disable the custom projection shortcut where torchvision has none.

    The custom block adds a 1x1 conv shortcut for every stride-1 block whose
    input and output widths differ. torchvision does not use a residual in that
    case (``use_res_connect = stride == 1 and in == out``). Zero-initialising
    the shortcut reproduces torchvision's behaviour; fine-tuning can still
    learn a non-zero residual.
    """
    import torch

    with torch.no_grad():
        block.shortcut[0].weight.zero_()
        block.shortcut[1].weight.zero_()
        block.shortcut[1].bias.zero_()
        block.shortcut[1].running_mean.zero_()
        block.shortcut[1].running_var.fill_(1.0)


def set_bn_settings(model: nn.Module, eps: float = 0.001, momentum: float = 0.01) -> None:
    """Match torchvision BatchNorm settings (used by pretrained checkpoints)."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = float(eps)
            module.momentum = float(momentum)


def _copy_backbone(custom: nn.Module, tv: nn.Module) -> Dict:
    """Copy torchvision MobileNetV3 feature weights into the custom model.

    The torchvision block layout is::

        [expand Conv2dNormActivation?]  # omitted when expanded == input
         depthwise Conv2dNormActivation
         [SElayer?]
         project Conv2dNormActivation

    Each Conv2dNormActivation is a Sequential holding Conv2d + BatchNorm2d.
    """
    pairs: Dict[str, "torch.Tensor"] = {}
    identity_blocks = []
    zero_shortcut_blocks = []

    # Match torchvision's BatchNorm settings so the copied running statistics
    # are interpreted with the same epsilon and momentum.
    set_bn_settings(custom)

    # Stem: Conv2dNormActivation -> [conv, bn, activation]
    tv_stem = tv.features[0]
    pairs["conv1.weight"] = tv_stem[0].weight
    pairs.update(_bn_pairs("bn1", tv_stem[1]))

    # Inverted residual blocks, in order.
    tv_blocks = [m for m in tv.features if m.__class__.__name__ == "InvertedResidual"]
    custom_blocks = list(custom.bneck)
    if len(tv_blocks) != len(custom_blocks):
        raise RuntimeError(
            f"block count mismatch: torchvision={len(tv_blocks)}, custom={len(custom_blocks)}"
        )

    for i, (custom_block, tv_block) in enumerate(zip(custom_blocks, tv_blocks)):
        stages = [m for m in tv_block.block if isinstance(m, nn.Sequential)]
        if len(stages) == 2:
            # No expand stage: torchvision's expanded width equals its input.
            _identity_init_expand(custom_block)
            identity_blocks.append(i)
            depth_stage, project_stage = stages
        elif len(stages) == 3:
            expand_stage, depth_stage, project_stage = stages
            conv, bn = _stage_conv_bn(expand_stage)
            pairs[f"bneck.{i}.conv1.weight"] = conv.weight
            pairs.update(_bn_pairs(f"bneck.{i}.bn1", bn))
        else:
            raise RuntimeError(
                f"unexpected torchvision block layout at index {i}: "
                f"{len(stages)} Conv2dNormActivation stages"
            )

        for name, bn_name, stage in (
            ("conv2", "bn2", depth_stage),
            ("conv3", "bn3", project_stage),
        ):
            conv, bn = _stage_conv_bn(stage)
            pairs[f"bneck.{i}.{name}.weight"] = conv.weight
            pairs.update(_bn_pairs(f"bneck.{i}.{bn_name}", bn))

        # The custom block may have a projection shortcut that torchvision
        # does not use; disable it (see _zero_init_shortcut).
        if isinstance(getattr(custom_block, "shortcut", None), nn.Sequential) and len(custom_block.shortcut) > 0:
            _zero_init_shortcut(custom_block)
            zero_shortcut_blocks.append(i)

    # Final 1x1 convolution.
    tv_final = tv.features[-1]
    pairs["conv2.weight"] = tv_final[0].weight
    pairs.update(_bn_pairs("bn2", tv_final[1]))

    target = custom.state_dict()
    state = {}
    skipped = []
    for key, tensor in pairs.items():
        if key not in target:
            skipped.append(key)
            continue
        if tuple(target[key].shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"shape mismatch for {key}: custom={tuple(target[key].shape)}, "
                f"torchvision={tuple(tensor.shape)}"
            )
        state[key] = tensor.detach().clone()

    missing, unexpected = custom.load_state_dict(state, strict=False)
    intentional = []
    for i in identity_blocks:
        intentional += [f"bneck.{i}.conv1.", f"bneck.{i}.bn1."]
    for i in zero_shortcut_blocks:
        intentional += [f"bneck.{i}.shortcut."]
    backbone_missing = [
        key for key in missing
        if key.startswith(("conv1.", "bn1.", "bneck.", "conv2.", "bn2."))
        and ".attention_module." not in key
        and not any(key.startswith(prefix) for prefix in intentional)
    ]
    if backbone_missing:
        raise RuntimeError(
            "backbone tensors were not initialised: "
            f"{backbone_missing[:5]}{'...' if len(backbone_missing) > 5 else ''}"
        )
    return {
        "init": "imagenet_backbone",
        "mapped_tensors": len(state),
        "skipped_source_keys": skipped,
        "identity_expand_blocks": identity_blocks,
        "zero_shortcut_blocks": zero_shortcut_blocks,
        "bn_eps": 0.001,
        "bn_momentum": 0.01,
        "attention_random": True,
        "head_random": True,
    }


def build_model(name: str, num_classes: int, pretrained: bool = False) -> Tuple[nn.Module, Dict]:
    """Build a model by repository name and return ``(model, info)``."""
    if name in CUSTOM_SPECS:
        size, attention, reduction = CUSTOM_SPECS[name]
        factory = MobileNetV3_Large if size == "large" else MobileNetV3_Small
        model = factory(
            num_classes=num_classes,
            attention_type=attention,
            reduction_ratio=reduction,
        )
        info = {
            "name": name,
            "family": "custom_mobilenetv3",
            "size": size,
            "attention": attention,
            "reduction_ratio": reduction,
            "pretrained": bool(pretrained),
            "init": "scratch",
        }
        if pretrained:
            arch = CUSTOM_TO_TV_ARCH[f"mobilenetv3_{size}"]
            tv = _build_torchvision(arch, num_classes=1000, pretrained=True)
            info.update(_copy_backbone(model, tv))
        return model, info

    if name in TORCHVISION_ARCHS:
        arch = TORCHVISION_ARCHS[name]
        model = _build_torchvision(arch, num_classes=num_classes, pretrained=pretrained)
        return model, {
            "name": name,
            "family": "torchvision",
            "arch": arch,
            "pretrained": bool(pretrained),
            "init": "imagenet" if pretrained else "scratch",
        }

    raise ValueError(f"unknown model name: {name}. Known: {ALL_MODELS}")
