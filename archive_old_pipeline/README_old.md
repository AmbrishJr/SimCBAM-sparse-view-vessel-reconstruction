# sparse-pa-unet

Photoacoustic (PA) tomography **sparse-view reconstruction** with attention-enhanced U-Net variants.

Deep learning models remove streak artifacts from sparse-view PA scans at 3 sensor densities
(**32, 64, 96** projection angles). Five model variants are provided — Baseline, SE, CBAM, SimAM,
and the proposed **SimCBAM** hybrid — plus a CNN **sparsity classifier** that auto-detects the
sensor density of an input, and a **Gradio demo** for interactive comparison.

This package is a faithful refactor of the original research notebook: the model
architectures, training loop, hyperparameters, checkpoint naming, and evaluation logic are
preserved verbatim, so runs reproduce the Colab results exactly. Only Colab-specific paths and
shell cells were replaced with configuration-driven local paths.

## Contents

- `src/sparse_pa_unet/` — installable package
  - `models/` — `UNet`, `UNet_SE`, `UNet_CBAM`, `UNet_SimAM`, `UNet_SimCBAM`, `SparsityClassifier`
  - `dataset.py` — `PADataset` (sparse/GT `.tif` pairs) and `SparsityLevelDataset`
  - `train.py` — CLI trainer (MSE, Adam, 30 epochs by default)
  - `evaluate.py` — CLI evaluator (PSNR/SSIM tables)
  - `inference.py` — checkpoint loading, model recommendation, vessel-image heuristic
  - `gradio_demo.py` — interactive demo (manual + auto modes)
- `configs/config.yaml` — all hyperparameters and paths (env-var overridable)
- `scripts/` — helpers to link local data/checkpoints, train & evaluate all
- `app/gradio_demo.py` — standalone demo launcher
- `tests/` — data-free smoke tests

## Dataset layout

Per sparsity level the data must live at `<data_root>/Vessel_sparse{32,64,96}/` with
ground truth at `<data_root>/Vessel/`, using matching filenames across sparse/GT:

```
data/
├── Vessel/            (ground truth)
│   ├── train/         4000 x .tif
│   ├── test/          200  x .tif
│   └── validation/    200  x .tif
├── Vessel_sparse32/   (sparse-view inputs, same split layout/counts)
├── Vessel_sparse64/
└── Vessel_sparse96/
```

Checkpoints go in `<ckpt_root>/` and use the naming convention
`{baseline,se,cbam,simam,simcbam}_unet_sparse{32,64,96}.pth` plus `sparsity_classifier.pth`.

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For a Colab-like CUDA 12.x environment you can instead use `requirements-colab.txt`
(which pins `torch==2.2.1+cu121`, compatible with CUDA 12.2).

### 2. Provide data & checkpoints

If your data/checkpoints are in `~/Downloads`, link them into the project:

```bash
./scripts/link_local_data.sh
```

or point the package directly at them via env vars:

```bash
export DATA_ROOT=~/Downloads
export CKPT_ROOT=~/Downloads/all_model_checkpoints
```

### 3. Train a model

```bash
python -m sparse_pa_unet.train --sparsity 64 --model cbam --epochs 50
```

Train everything (15 runs, baseline/se/cbam/simam/simcbam at 32/64/96):

```bash
./scripts/train_all.sh
```

### 4. Evaluate

```bash
python -m sparse_pa_unet.evaluate --sparsity 96           # all models at 96 angles
python -m sparse_pa_unet.evaluate --models cbam,simcbam   # two models, all sparsities
./scripts/evaluate_all.sh
```

Tables are printed and saved to `outputs/evaluation_results.csv`.

### 5. Launch the demo

```bash
python app/gradio_demo.py                 # local browser
python app/gradio_demo.py --share         # public share link (internet tunnel)
```

- **Manual mode:** upload a clean vessel image, choose 32/64/96 angles, compare all 5 models.
- **Auto mode:** the sparsity classifier detects the sensor density, then the best model is
  chosen live for your image (priority: SSIM quality, PSNR, or efficiency).

## Reference results (from the 15 full training runs)

| Sparsity | Best PSNR | Best SSIM | Most efficient |
|----------|-----------|-----------|----------------|
| 32       | CBAM 19.02 dB | CBAM 0.8354 | SimCBAM (686 params, 94% of CBAM's SSIM gain) |
| 64       | SE 21.71 dB | CBAM 0.9015 | SimCBAM (686 params, near-SE SSIM) |
| 96       | CBAM 22.32 dB | CBAM 0.9176 | SimAM (0 params, 91.1% SSIM) |

## Notes

- All models are trained with `in_ch=1, out_ch=1, base=32`, MSELoss, Adam `lr=1e-3`, input size
  **128×128**, and PSNR/SSIM evaluated at `data_range=1.0`.
- Training defaults mirror the notebook: 30 epochs (U-Nets) and 10 epochs (sparsity classifier);
  override with `--epochs`.
- The code auto-selects CPU/CUDA. Running on CPU is supported but slower.