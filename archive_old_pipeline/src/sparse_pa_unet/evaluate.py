import argparse
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from sparse_pa_unet.config import get_device, load_config, resolve_data_dirs
from sparse_pa_unet.dataset import PADataset
from sparse_pa_unet.inference import load_model
from sparse_pa_unet.models import MODEL_NAMES, get_model_class


def evaluate_model(model, test_loader, device, data_range=1.0):
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for sparse, gt in test_loader:
            sparse, gt = sparse.to(device), gt.to(device)
            output = model(sparse)
            out_np, gt_np = output.cpu().numpy(), gt.cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_scores.append(psnr_fn(gt_np[i, 0], out_np[i, 0], data_range=data_range))
                ssim_scores.append(ssim_fn(gt_np[i, 0], out_np[i, 0], data_range=data_range))
    return np.mean(psnr_scores), np.mean(ssim_scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained sparse-view PA reconstruction models.")
    parser.add_argument("--sparsity", type=int, default=None, choices=[32, 64, 96], help="Angle level (default: all).")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model list, e.g. baseline,cbam (default: all).",
    )
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint dir (default: config).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Limit test samples per dataset (0 = all).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    print("Using device:", device)
    ckpt_dir = args.ckpt_dir or config["checkpoints"]["root"]

    model_keys = args.models.split(",") if args.models else list(MODEL_NAMES)
    sparsities = [args.sparsity] if args.sparsity is not None else [32, 64, 96]

    sparse_dirs, gt_dir = resolve_data_dirs(config)
    print("Dataset root:", config["data"]["root"])
    print("Checkpoint dir:", ckpt_dir)
    print()

    results = []
    for sparsity in sparsities:
        test_ds = PADataset(os.path.join(sparse_dirs[sparsity], "test"), os.path.join(gt_dir, "test"))
        if args.limit:
            indices = np.arange(min(args.limit, len(test_ds)))
            test_ds = torch.utils.data.Subset(test_ds, indices)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
        for key in model_keys:
            path = os.path.join(ckpt_dir, f"{key}_unet_sparse{sparsity}.pth")
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            model = load_model(get_model_class(key), path, device)
            p, s = evaluate_model(model, test_loader, device)
            results.append((sparsity, MODEL_NAMES[key], p, s))
            print(f"Sparse {sparsity:>3} | {MODEL_NAMES[key]:<8} | PSNR {p:7.2f} dB | SSIM {s:.4f}")

    if results:
        import pandas as pd

        df = pd.DataFrame(results, columns=["sparsity", "model", "psnr", "ssim"])
        pivot = df.pivot(index="model", columns="sparsity", values="psnr")
        print("\n=== PSNR (dB) by model & sparsity ===")
        print(pivot.to_string())
        pivot2 = df.pivot(index="model", columns="sparsity", values="ssim")
        print("\n=== SSIM by model & sparsity ===")
        print(pivot2.to_string())
        out_dir = config["evaluation"].get("output_dir", "./outputs")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "evaluation_results.csv"), index=False)
        print(f"\nSaved table to {os.path.join(out_dir, 'evaluation_results.csv')}")


if __name__ == "__main__":
    main()