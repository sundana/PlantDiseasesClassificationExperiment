import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from ptflops import get_model_complexity_info
import pandas as pd
from src.config import COMPLEXITY_DIR

def calculate_flops(models: list[nn.Module]) -> pd.DataFrame:
    """Calculate model FLOPs

    Args:
        models (list(nn.Module)): PyTorch model based on nn.Module

    Returns:
        pd.DataFrame: table of model complexity
    """
    model_complexity = {
        "models": [],
        "flops": [],
        "params": [],
        "memory_size_mb": []
    }
    for model_factory in models:
        model = model_factory(num_classes=29)
        try:
            flop, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
        except Exception:
            flop, params = None, sum(p.numel() for p in model.parameters())

        # Calculate memory size (MB) based on parameters (float32 = 4 bytes)
        if params is not None:
            memory_size_mb = (params * 4) / (1024 * 1024)
        else:
            memory_size_mb = None

        # Determine a descriptive name
        name = model.__class__.__name__
        if hasattr(model_factory, 'keywords'):
            att = model_factory.keywords.get('attention_type', 'none')
            red = model_factory.keywords.get('reduction_ratio', '')
            if att == 'none':
                name = f"{name}_none"
            elif att == 'se':
                name = f"{name}_SE_r{red}" if red else f"{name}_SE"
            elif att == 'cbam':
                name = f"{name}_CBAM_r{red}" if red else f"{name}_CBAM"

        model_complexity["models"].append(name)
        model_complexity["flops"].append(flop)
        model_complexity["params"].append(params)
        model_complexity["memory_size_mb"].append(memory_size_mb)

    df = pd.DataFrame(model_complexity)
    return df


if __name__ == "__main__":
    from src.models import mobilenetv3
    from src.models.baselines import get_mobilenet_v2, get_shufflenet_v2, get_efficientnet_b0
    from functools import partial
    
    # Register model
    models = [ 
        partial(mobilenetv3.MobileNetV3_Large, attention_type='se', reduction_ratio=4),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='none'),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='se', reduction_ratio=16),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='se', reduction_ratio=32),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='cbam', reduction_ratio=4),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='cbam', reduction_ratio=16),
        partial(mobilenetv3.MobileNetV3_Large, attention_type='cbam', reduction_ratio=32),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='se', reduction_ratio=4),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='none'),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='se', reduction_ratio=16),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='se', reduction_ratio=32),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='cbam', reduction_ratio=4),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='cbam', reduction_ratio=16),
        partial(mobilenetv3.MobileNetV3_Small, attention_type='cbam', reduction_ratio=32),
        get_mobilenet_v2,
        get_shufflenet_v2,
        get_efficientnet_b0,
    ]

    df = calculate_flops(models)
    save_path = os.path.join(COMPLEXITY_DIR, 'model_complexity.csv')
    df.to_csv(save_path, index=False) # Save result csv
    print(f"Saving model complexity completed to {COMPLEXITY_DIR}")
    print(df)
