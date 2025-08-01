# ==================== utils.py ====================

import torch
import random
import numpy as np
import os
import shutil

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, is_best, checkpoint_dir, filename):
    """Save model checkpoint"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, f'best_{filename}')
        shutil.copyfile(filepath, best_filepath)

def load_checkpoint(filepath):
    """Load model checkpoint"""
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath, map_location='cpu')
        return checkpoint
    else:
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum