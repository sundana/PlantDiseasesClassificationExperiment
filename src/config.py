import os
import random
import numpy as np
import torch

# Project Root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
PLANTVILLAGE_DIR = os.path.join(DATA_DIR, "plantvillage")
PLANTDOC_DIR = os.path.join(DATA_DIR, "plantdoc")

# Output Directories
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
TRAINING_LOG_DIR = os.path.join(RESULTS_DIR, "training")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
COMPLEXITY_DIR = os.path.join(RESULTS_DIR, "complexity")
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")

# Domain Adaptation / Cross-Validation Directories
DOMAIN_ADAPT_DIR = os.path.join(RESULTS_DIR, "domain_adaptation")
DA_CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints_da")
CROSSVAL_DIR = os.path.join(RESULTS_DIR, "cross_validation")
PLANTDOC_SPLIT_DIR = os.path.join(DATA_DIR, "plantdoc_split")

# Default Random Seed
DEFAULT_SEED = 42

# Create directories if they don't exist
for d in [RESULTS_DIR, CHECKPOINT_DIR, TRAINING_LOG_DIR, PLOT_DIR, CSV_DIR,
          EVAL_DIR, COMPLEXITY_DIR, CONFUSION_MATRIX_DIR,
          DOMAIN_ADAPT_DIR, CROSSVAL_DIR, DA_CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)


def set_seed(seed: int = DEFAULT_SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
