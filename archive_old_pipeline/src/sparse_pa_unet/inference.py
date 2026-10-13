import os

import numpy as np
import torch

from sparse_pa_unet.config import load_config
from sparse_pa_unet.models import MODEL_NAMES, get_model_class

CHECKPOINT_TEMPLATE = "{name}_unet_sparse{sparsity}.pth"


# Recommendation table based on the 15 full training runs (from the notebook).
RECOMMENDATION_TABLE = {
    32: {
        "best_psnr": "CBAM",
        "best_psnr_val": 19.02,
        "best_ssim": "CBAM",
        "best_ssim_val": 0.8354,
        "most_efficient": "SimCBAM",
        "efficient_note": "686 params, 94% of CBAM's SSIM gain",
    },
    64: {
        "best_psnr": "SE",
        "best_psnr_val": 21.71,
        "best_ssim": "CBAM",
        "best_ssim_val": 0.9015,
        "most_efficient": "SimCBAM",
        "efficient_note": "686 params, near-SE SSIM performance",
    },
    96: {
        "best_psnr": "CBAM",
        "best_psnr_val": 22.32,
        "best_ssim": "CBAM",
        "best_ssim_val": 0.9176,
        "most_efficient": "SimAM",
        "efficient_note": "0 params, 91.1% SSIM",
    },
}

MODEL_PARAM_COST = {
    "Baseline": 0,
    "SimAM": 0,
    "SimCBAM": 686,
    "SE": 13568,
    "CBAM": 14254,
}


def is_likely_vessel_image(gray_array):
    normalized = gray_array / 255.0
    foreground_ratio = np.mean(normalized > 0.5)
    contrast = np.std(normalized)
    reasons = []
    if foreground_ratio > 0.35:
        reasons.append(
            "too much bright area (real vessel images are mostly dark background with thin bright structures)"
        )
    if contrast < 0.08:
        reasons.append("too little contrast (image looks too flat/uniform)")
    is_valid = len(reasons) == 0
    return is_valid, reasons


def recommend_model(sparsity_level, priority="quality"):
    info = RECOMMENDATION_TABLE[sparsity_level]
    if priority == "psnr":
        return info["best_psnr"], f"Best PSNR at {sparsity_level} angles: {info['best_psnr_val']:.2f} dB"
    elif priority == "efficient":
        return info["most_efficient"], f"Most efficient choice: {info['efficient_note']}"
    else:
        return info["best_ssim"], f"Best SSIM at {sparsity_level} angles: {info['best_ssim_val']:.4f}"


def load_model(model_class, path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model_class().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def load_all_models(ckpt_dir=None, device=None):
    ckpt_dir = ckpt_dir or load_config()["checkpoints"]["root"]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_by_sparsity = {}
    for sparsity in [32, 64, 96]:
        models_by_sparsity[sparsity] = {}
        for key, display in MODEL_NAMES.items():
            path = os.path.join(ckpt_dir, CHECKPOINT_TEMPLATE.format(name=key, sparsity=sparsity))
            models_by_sparsity[sparsity][display] = load_model(get_model_class(key), path, device)
    return models_by_sparsity

