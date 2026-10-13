import argparse
import os

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import iradon, radon

from sparse_pa_unet.config import get_device, load_config
from sparse_pa_unet.inference import (
    MODEL_PARAM_COST,
    is_likely_vessel_image,
    load_all_models,
    recommend_model,
)
from sparsity_classifier import load_sparsity_classifier

device = None
models_by_sparsity = None
sparsity_model = None


def process_image(uploaded_image, sparsity_level):
    img = uploaded_image.convert("L").resize((128, 128))
    gt_array = np.array(img).astype(np.float64)

    is_valid, reasons = is_likely_vessel_image(gt_array)
    if not is_valid:
        blank = np.zeros((128, 128))
        warning_msg = (
            "**This doesn't look like a vessel-style image.**\n\n"
            f"Reason(s): {', '.join(reasons)}.\n\n"
            "Please upload a binary/high-contrast vessel image (thin white branching structure on a dark background)."
        )
        return blank, blank, "", blank, "", blank, "", blank, "", blank, "", warning_msg

    gt_norm = gt_array / 255.0
    theta_sparse = np.linspace(0.0, 180.0, sparsity_level, endpoint=False)
    sinogram = radon(gt_array, theta=theta_sparse, circle=True)
    sparse_array = iradon(sinogram, theta=theta_sparse, circle=True)
    sparse_array = np.clip(sparse_array, 0, 255)
    sparse_norm = sparse_array / 255.0
    sparse_tensor = torch.from_numpy(sparse_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    results = {}
    for name in ["Baseline", "SE", "CBAM", "SimAM", "SimCBAM"]:
        model = models_by_sparsity[sparsity_level][name]
        with torch.no_grad():
            out = model(sparse_tensor).cpu().squeeze().numpy()
        out = np.clip(out, 0, 1)
        p = psnr(gt_norm, out, data_range=1.0)
        s = ssim(gt_norm, out, data_range=1.0)
        results[name] = (out, p, s)

    ranked = sorted(results.items(), key=lambda x: x[1][2], reverse=True)
    best_model = ranked[0][0]
    summary = f"**Best result: {best_model}** (highest SSIM at this sparsity level)\n\n"
    summary += "| Model | PSNR | SSIM |\n|---|---|---|\n"
    for name, (_, p, s) in ranked:
        marker = " (trophy)" if name == best_model else ""
        summary += f"| {name}{marker} | {p:.2f} dB | {s:.4f} |\n"

    return (
        gt_norm,
        sparse_norm,
        results["Baseline"][0], f"PSNR: {results['Baseline'][1]:.2f} dB | SSIM: {results['Baseline'][2]:.4f}",
        results["SE"][0], f"PSNR: {results['SE'][1]:.2f} dB | SSIM: {results['SE'][2]:.4f}",
        results["CBAM"][0], f"PSNR: {results['CBAM'][1]:.2f} dB | SSIM: {results['CBAM'][2]:.4f}",
        results["SimAM"][0], f"PSNR: {results['SimAM'][1]:.2f} dB | SSIM: {results['SimAM'][2]:.4f}",
        results["SimCBAM"][0], f"PSNR: {results['SimCBAM'][1]:.2f} dB | SSIM: {results['SimCBAM'][2]:.4f}",
        summary,
    )


def auto_process_image(uploaded_image, priority):
    img = uploaded_image.convert("L").resize((128, 128))
    gt_array = np.array(img).astype(np.float64)

    is_valid, reasons = is_likely_vessel_image(gt_array)
    if not is_valid:
        blank = np.zeros((128, 128))
        warning_msg = f"This doesn't look like a vessel-style image. {' '.join(reasons)}."
        return blank, blank, "", warning_msg

    gt_norm = gt_array / 255.0

    # Step 1: detect sparsity level using a default 64-angle simulation just for classification
    theta_default = np.linspace(0.0, 180.0, 64, endpoint=False)
    sinogram = radon(gt_array, theta=theta_default, circle=True)
    sparse_array_default = np.clip(iradon(sinogram, theta=theta_default, circle=True), 0, 255)

    img_tensor = torch.from_numpy(sparse_array_default / 255.0).float().unsqueeze(0).unsqueeze(0).to(device)
    sparsity_model.eval()
    with torch.no_grad():
        pred = sparsity_model(img_tensor)
        predicted_class = pred.argmax(dim=1).item()
    detected_sparsity = [32, 64, 96][predicted_class]
    confidence = torch.softmax(pred, dim=1)[0][predicted_class].item()

    # Step 2: regenerate the sparse image at the ACTUAL detected sparsity level
    theta_sparse = np.linspace(0.0, 180.0, detected_sparsity, endpoint=False)
    sinogram2 = radon(gt_array, theta=theta_sparse, circle=True)
    sparse_array = np.clip(iradon(sinogram2, theta=theta_sparse, circle=True), 0, 255)
    sparse_norm = sparse_array / 255.0
    sparse_tensor = torch.from_numpy(sparse_norm).float().unsqueeze(0).unsqueeze(0).to(device)

    # Step 3: run ALL available models for this sparsity level, LIVE, on this exact image
    live_results = {}
    for name, model in models_by_sparsity[detected_sparsity].items():
        model.eval()
        with torch.no_grad():
            out = model(sparse_tensor).cpu().squeeze().numpy()
        out = np.clip(out, 0, 1)
        p = psnr(gt_norm, out, data_range=1.0)
        s = ssim(gt_norm, out, data_range=1.0)
        param_cost = MODEL_PARAM_COST[name]
        live_results[name] = {"output": out, "psnr": p, "ssim": s, "params": param_cost}

    # Step 4: pick the TRUE best model for THIS image, based on chosen priority
    if priority == "psnr":
        best_name = max(live_results, key=lambda k: live_results[k]["psnr"])
    elif priority == "efficient":
        baseline_ssim = live_results["Baseline"]["ssim"]

        def efficiency_score(k):
            gain = live_results[k]["ssim"] - baseline_ssim
            cost = live_results[k]["params"] + 1  # epsilon to avoid /0
            return gain / cost

        best_name = max(live_results, key=efficiency_score)
    else:  # quality (SSIM)
        best_name = max(live_results, key=lambda k: live_results[k]["ssim"])

    best = live_results[best_name]

    # Build a live leaderboard for transparency
    ranked = sorted(live_results.items(), key=lambda x: x[1]["ssim"], reverse=True)
    leaderboard = "| Model | PSNR | SSIM | Params |\n|---|---|---|---|\n"
    for name, r in ranked:
        marker = " (trophy)" if name == best_name else ""
        leaderboard += f"| {name}{marker} | {r['psnr']:.2f} dB | {r['ssim']:.4f} | {r['params']} |\n"

    info = (
        f"**Detected sparsity:** {detected_sparsity} angles (confidence: {confidence:.1%})\n\n"
        f"**Selected model (computed live for THIS image):** {best_name}\n\n"
        f"**Result:** PSNR {best['psnr']:.2f} dB | SSIM {best['ssim']:.4f}\n\n"
        f"### Live leaderboard for this image\n{leaderboard}"
    )

    return sparse_norm, best["output"], "", info


def _auto_process_image_wrapper(uploaded_image, priority):
    result = auto_process_image(uploaded_image, priority)
    return result[0], result[1], result[3]


def _load_models(ckpt_dir):
    global models_by_sparsity, sparsity_model
    print("Loading models ...")
    models_by_sparsity = load_all_models(ckpt_dir, device)
    sparsity_path = os.path.join(ckpt_dir, "sparsity_classifier.pth")
    if os.path.exists(sparsity_path):
        sparsity_model = load_sparsity_classifier(sparsity_path, device)
    else:
        print("[warn] sparsity_classifier.pth not found; auto mode will be unavailable.")
    print("Models loaded.")


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Attention Mechanism Comparison") as combined_demo:
        gr.Markdown("# Attention Mechanism Comparison - Photoacoustic Sparse-View Reconstruction")
        gr.Markdown("Upload your image once - it's used by both Manual Mode and Auto Mode below.")
        img_input = gr.Image(type="pil", label="Upload a clean vessel image")

        gr.Markdown("---")
        gr.Markdown("## Manual Mode")
        gr.Markdown("Choose the sparsity level yourself and compare all 5 models side by side.")
        with gr.Row():
            sparsity_input = gr.Radio([32, 64, 96], value=32, label="Sparsity level (sensor angles)")
            run_btn = gr.Button("Run Comparison", variant="primary")
        with gr.Row():
            gt_out = gr.Image(label="Ground Truth (your upload)")
            sparse_out = gr.Image(label="Sparse Input (simulated)")

        gr.Markdown("### Model Outputs")
        with gr.Row():
            with gr.Column():
                base_out = gr.Image(label="Baseline")
                base_score = gr.Markdown()
            with gr.Column():
                se_out = gr.Image(label="SE")
                se_score = gr.Markdown()
            with gr.Column():
                cbam_out = gr.Image(label="CBAM")
                cbam_score = gr.Markdown()
            with gr.Column():
                simam_out = gr.Image(label="SimAM")
                simam_score = gr.Markdown()
            with gr.Column():
                simcbam_out = gr.Image(label="SimCBAM (ours)")
                simcbam_score = gr.Markdown()

        gr.Markdown("### Summary")
        summary_out = gr.Markdown()

        run_btn.click(
            fn=process_image,
            inputs=[img_input, sparsity_input],
            outputs=[gt_out, sparse_out, base_out, base_score, se_out, se_score, cbam_out, cbam_score,
                     simam_out, simam_score, simcbam_out, simcbam_score, summary_out],
        )

        gr.Markdown("---")
        gr.Markdown("## Auto Mode")
        gr.Markdown("Uses the SAME image uploaded above. Automatically detects sparsity and picks the best model - computed live for this image.")
        with gr.Row():
            priority_input = gr.Radio(["quality", "psnr", "efficient"], value="quality", label="Priority")
            auto_run_btn = gr.Button("Auto-Run", variant="primary")
        with gr.Row():
            auto_sparse_out = gr.Image(label="Detected Sparse Reconstruction Input")
            auto_output_out = gr.Image(label="Recommended Model's Output")
        auto_info_out = gr.Markdown()

        auto_run_btn.click(
            fn=_auto_process_image_wrapper,
            inputs=[img_input, priority_input],
            outputs=[auto_sparse_out, auto_output_out, auto_info_out],
        )

    return combined_demo


def main():
    global device
    parser = argparse.ArgumentParser(description="Launch the Gradio demo for sparse-view PA reconstruction.")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Checkpoint dir (default: config).")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    parser.add_argument("--share", action="store_true", help="Create a public share link (default: local).")
    parser.add_argument("--server-name", type=str, default=None, help="Server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=None, help="Server port.")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    print("Using device:", device)
    ckpt_dir = args.ckpt_dir or config["checkpoints"]["root"]

    if not os.path.exists(ckpt_dir):
        print(f"[error] Checkpoint dir not found: {ckpt_dir}")
        print("Run scripts/link_local_data.sh or set CKPT_ROOT.")
        return

    _load_models(ckpt_dir)
    demo = build_demo()
    demo.launch(share=args.share, server_name=args.server_name or None, server_port=args.port)


if __name__ == "__main__":
    main()