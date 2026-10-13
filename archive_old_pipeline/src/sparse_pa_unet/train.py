import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torch.utils.data import DataLoader

from sparse_pa_unet.config import get_device, load_config, resolve_data_dirs
from sparse_pa_unet.dataset import PADataset
from sparse_pa_unet.models import MODEL_NAMES, get_model_class


def train_and_eval(
    model_class,
    name,
    train_loader,
    val_loader,
    test_loader,
    train_ds,
    val_ds,
    num_epochs=30,
    learning_rate=1e-3,
    save_path=None,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(in_ch=1, out_ch=1, base=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sparse, gt in train_loader:
            sparse, gt = sparse.to(device), gt.to(device)
            optimizer.zero_grad()
            output = model(sparse)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sparse.size(0)
        train_loss /= len(train_ds)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sparse, gt in val_loader:
                sparse, gt = sparse.to(device), gt.to(device)
                output = model(sparse)
                val_loss += criterion(output, gt).item() * sparse.size(0)
        val_loss /= len(val_ds)
        print(f"[{name}] Epoch {epoch + 1}/{num_epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for sparse, gt in test_loader:
            sparse, gt = sparse.to(device), gt.to(device)
            output = model(sparse)
            out_np, gt_np = output.cpu().numpy(), gt.cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_scores.append(psnr_fn(gt_np[i, 0], out_np[i, 0], data_range=1.0))
                ssim_scores.append(ssim_fn(gt_np[i, 0], out_np[i, 0], data_range=1.0))
    p, s = np.mean(psnr_scores), np.mean(ssim_scores)
    print(f"{name} - Test PSNR: {p:.2f} dB | Test SSIM: {s:.4f}\n")
    return model, p, s


def main():
    parser = argparse.ArgumentParser(description="Train a sparse-view PA reconstruction U-Net.")
    parser.add_argument("--sparsity", type=int, required=True, choices=[32, 64, 96], help="Sparse-view angle level.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODEL_NAMES),
        help="Model name: baseline, se, cbam, simam, simcbam.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: config).")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint output dir (default: config).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    print("Using device:", device)

    ckpt_dir = args.ckpt_dir or config["checkpoints"]["root"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = args.lr if args.lr is not None else config["training"]["learning_rate"]

    sparse_dirs, gt_dir = resolve_data_dirs(config)
    train_ds = PADataset(os.path.join(sparse_dirs[args.sparsity], "train"), os.path.join(gt_dir, "train"))
    val_ds = PADataset(os.path.join(sparse_dirs[args.sparsity], "validation"), os.path.join(gt_dir, "validation"))
    test_ds = PADataset(os.path.join(sparse_dirs[args.sparsity], "test"), os.path.join(gt_dir, "test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    name = MODEL_NAMES[args.model]
    save_path = os.path.join(ckpt_dir, f"{args.model}_unet_sparse{args.sparsity}.pth")
    model, p, s = train_and_eval(
        get_model_class(args.model),
        name,
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        val_ds,
        num_epochs=epochs,
        learning_rate=lr,
        save_path=save_path,
        device=device,
    )
    print(f"Done. {name} @ {args.sparsity} angles -> PSNR {p:.2f} dB, SSIM {s:.4f}")


if __name__ == "__main__":
    main()