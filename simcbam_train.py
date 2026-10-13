#!/usr/bin/env python3
"""
SimCBAM study - v3 training / reporting script (Windows / Linux, no Colab needed)

Same experiment and protocol as the Colab notebook (30 epochs, Adam 1e-3, cosine decay,
best-validation checkpoint, MSE loss, batch 16). Result files have identical names and format,
so results made here and on Colab can be merged simply by copying files between the two folders.

USAGE (from a terminal, inside the folder that holds this file):

  1) check the environment (takes ~1-2 min, writes nothing into your real results):
       python simcbam_train.py --out runs/v3 --smoke

  2) real training (resumes automatically; run it again after any interruption):
       python simcbam_train.py --out runs/v3 --seeds 2

  3) when all seeds have finished (copy every v3_*.json and *.pth from other machines into runs/v3/results and /ckpt first):
       python simcbam_train.py --out runs/v3 --seeds 0 1 2 --report
       python simcbam_train.py --out runs/v3 --cost            (once, on ONE GPU)

  Outputs:  runs/v3/results/report.txt, summary_table.csv, paired_tests.csv (+ cost.csv)

The folder given by --out must contain gt_{train,validation,test}.npy and sparse{32,64,96}_{train,validation,test}.npy
and may contain results/ and ckpt/ with already finished runs (they are skipped).
"""
import argparse, glob, json, os, random, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

try:                                    # Windows consoles: never crash on a special character
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ANGLES = [32, 64, 96]
EPOCHS, BATCH, LR = 30, 16, 1e-3
PROTOCOL = 'v3'
OUT = None
NUM_WORKERS = 0
SMOKE = False
device = None


# =====================================================================
# Model definitions: copied verbatim from the original notebook
# =====================================================================
# --- U-Net architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConvSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)

class UNet_SE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConvSE(in_ch, base)
        self.enc2 = DoubleConvSE(base, base*2)
        self.enc3 = DoubleConvSE(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvSE(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConvSE(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConvSE(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConvSE(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DoubleConvCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.cbam(x)

class UNet_CBAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConvCBAM(in_ch, base)
        self.enc2 = DoubleConvCBAM(base, base*2)
        self.enc3 = DoubleConvCBAM(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvCBAM(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConvCBAM(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConvCBAM(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConvCBAM(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

class DoubleConvSimAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.simam = SimAM()
    def forward(self, x):
        return self.simam(self.conv(x))

class UNet_SimAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConvSimAM(in_ch, base)
        self.enc2 = DoubleConvSimAM(base, base*2)
        self.enc3 = DoubleConvSimAM(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvSimAM(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConvSimAM(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConvSimAM(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConvSimAM(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class SimCBAM(nn.Module):
    """Hybrid: SimAM (parameter-free 'channel-like' attention) + CBAM's spatial attention branch"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.simam = SimAM()
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.simam(x)
        x = self.sa(x)
        return x

class DoubleConvSimCBAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.simcbam = SimCBAM(out_ch)
    def forward(self, x):
        return self.simcbam(self.conv(x))

class UNet_SimCBAM(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = DoubleConvSimCBAM(in_ch, base), DoubleConvSimCBAM(base, base*2), DoubleConvSimCBAM(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConvSimCBAM(base*4, base*8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2), DoubleConvSimCBAM(base*8, base*4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2), DoubleConvSimCBAM(base*4, base*2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base*2, base, 2, stride=2), DoubleConvSimCBAM(base*2, base)
        self.out_conv = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


MODELS = {'Baseline': UNet, 'SE': UNet_SE, 'CBAM': UNet_CBAM, 'SimAM': UNet_SimAM, 'SimCBAM': UNet_SimCBAM}
count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)


def log(msg):
    line = f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if OUT and os.path.isdir(OUT):
        with open(os.path.join(OUT, 'train_log.txt'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def write_json(obj, path):
    tmp = path + '.tmp'                                    # atomic write: an interruption never leaves a half-written file
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'


def check_parameters():
    base = count(UNet())
    extra = {k: count(c()) - base for k, c in MODELS.items()}
    log('parameter check: ' + ', '.join(f'{k} +{v:,}' for k, v in extra.items()))
    assert list(extra.values()) == [0, 13568, 14254, 0, 686], 'parameter counts differ from the manuscript!'
    return base


class NPYDataset(Dataset):
    def __init__(self, n_angles, split, limit=None):
        self.x = np.load(f'{OUT}/sparse{n_angles}_{split}.npy', mmap_mode='r')
        self.y = np.load(f'{OUT}/gt_{split}.npy', mmap_mode='r')
        self.n = len(self.y) if limit is None else min(limit, len(self.y))
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.x[i]))[None], torch.from_numpy(np.array(self.y[i]))[None]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); P, S_ = [], []
    for x, y in loader:
        out = model(x.to(device)).clamp(0, 1).cpu().numpy(); y = y.numpy()
        for i in range(out.shape[0]):
            P.append(float(psnr_fn(y[i, 0], out[i, 0], data_range=1.0)))
            S_.append(float(ssim_fn(y[i, 0], out[i, 0], data_range=1.0)))
    return P, S_


def run_one(model_name, n_angles, seed):
    tag = f'{PROTOCOL}_{model_name}_{n_angles}_seed{seed}'
    rs, ck = f'{OUT}/results/{tag}.json', f'{OUT}/ckpt/{tag}.pth'
    if os.path.exists(rs):
        return None                                                   # finished earlier -> skip
    lim = 64 if SMOKE else None
    epochs = 2 if SMOKE else EPOCHS
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(NPYDataset(n_angles, 'train', lim), batch_size=BATCH, shuffle=True, generator=g, num_workers=NUM_WORKERS)
    va = DataLoader(NPYDataset(n_angles, 'validation', lim), batch_size=BATCH, num_workers=NUM_WORKERS)
    te = DataLoader(NPYDataset(n_angles, 'test', lim), batch_size=BATCH, num_workers=NUM_WORKERS)
    model = MODELS[model_name](in_ch=1, out_ch=1, base=32).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit, t0 = nn.MSELoss(), time.time()
    best_val, best_ep, best_state, hist = float('inf'), 0, None, []
    for ep in range(epochs):
        model.train(); tl = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * x.size(0)
        sched.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in va:
                vl += crit(model(x.to(device)), y.to(device)).item() * x.size(0)
        vl /= len(va.dataset); tl /= len(tr.dataset)
        hist.append(dict(epoch=ep + 1, train=tl, val=vl))
        if vl < best_val:
            best_val, best_ep = vl, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0 or SMOKE:
            log(f'[{tag}] epoch {ep + 1}/{epochs}  train {tl:.5f}  val {vl:.5f}  (best val {best_val:.5f} @ epoch {best_ep})')
    P_last, S_last = evaluate(model, te)
    model.load_state_dict(best_state); torch.save(best_state, ck)
    P, S_ = evaluate(model, te)
    res = dict(model=model_name, angles=n_angles, seed=seed, params=count(model), train_seconds=time.time() - t0,
               best_epoch=best_ep, best_val=best_val,
               psnr_mean=float(np.mean(P)), ssim_mean=float(np.mean(S_)),
               psnr_last_epoch=float(np.mean(P_last)), ssim_last_epoch=float(np.mean(S_last)),
               gpu=gpu_name(), torch=torch.__version__,
               history=hist, psnr_per_image=P, ssim_per_image=S_)
    write_json(res, rs)
    log(f'==> {tag}: PSNR {res["psnr_mean"]:.2f} dB | SSIM {res["ssim_mean"]:.4f}   '
        f'(last epoch would have been {res["psnr_last_epoch"]:.2f} dB / {res["ssim_last_epoch"]:.4f}; best epoch {best_ep}; {res["train_seconds"] / 60:.1f} min)')
    return res


def train_all(seeds):
    need = [f'{OUT}/gt_{s}.npy' for s in ('train', 'validation', 'test')] + \
           [f'{OUT}/sparse{n}_{s}.npy' for n in ANGLES for s in ('train', 'validation', 'test')]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit('ERROR: these data files are missing from --out:\n  ' + '\n  '.join(missing))
    os.makedirs(f'{OUT}/results', exist_ok=True); os.makedirs(f'{OUT}/ckpt', exist_ok=True)
    check_parameters()
    if SMOKE:
        seeds, angles = [0], [32]
    else:
        angles = ANGLES
    todo = [(m, n, s) for s in seeds for n in angles for m in MODELS
            if not os.path.exists(f'{OUT}/results/{PROTOCOL}_{m}_{n}_seed{s}.json')]
    log(f'{len(todo)} model(s) to train now (finished ones are skipped). GPU: {gpu_name()} | torch {torch.__version__}')
    t_start, done = time.time(), 0
    for (m, n, s) in todo:
        run_one(m, n, s); done += 1
        per = (time.time() - t_start) / done / 60
        log(f'progress: {done}/{len(todo)} done this session, {per:.1f} min per model on average, '
            f'about {per * (len(todo) - done) / 60:.1f} h left')
    log('all requested models are finished' + (' (SMOKE TEST ONLY: nothing here is a real result)' if SMOKE else ''))


# =====================================================================
# Reporting: results table, paired statistics, run provenance
# =====================================================================
def report(seeds):
    import pandas as pd
    from scipy.stats import wilcoxon
    res_dir = f'{OUT}/results'
    done = [s for s in seeds if all(os.path.exists(f'{res_dir}/{PROTOCOL}_{m}_{n}_seed{s}.json') for m in MODELS for n in ANGLES)]
    if not done:
        sys.exit('No seed has all 15 models finished yet. Train more, or copy the missing v3_*.json files into the results folder.')
    lines = []
    P = lambda *a: lines.append(' '.join(str(x) for x in a))
    P(f'REPORT generated {datetime.datetime.now():%Y-%m-%d %H:%M}   protocol {PROTOCOL}')
    P(f'requested seeds {list(seeds)}; complete seeds used: {done}')
    if done != list(seeds):
        P('NOTE: not all requested seeds are complete, so only the complete ones are used below.')
    R = [json.load(open(f, encoding='utf-8')) for f in sorted(glob.glob(f'{res_dir}/{PROTOCOL}_*_seed*.json'))]
    R = [r for r in R if r['seed'] in done]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('psnr_per_image', 'ssim_per_image', 'history')} for r in R])
    df['gpu'] = df.get('gpu', pd.Series(['unknown (Colab)'] * len(df))).fillna('unknown (Colab)')
    df['torch'] = df.get('torch', pd.Series(['unknown'] * len(df))).fillna('unknown')
    P('\n--- hardware / software that produced the runs (needed for the Methods section) ---')
    P(df.groupby(['seed', 'gpu', 'torch']).size().rename('runs').reset_index().to_string(index=False))
    P(f"mean training time per model: {df.train_seconds.mean() / 60:.1f} min")

    agg = df.groupby(['angles', 'model']).agg(PSNR=('psnr_mean', 'mean'), PSNR_sd=('psnr_mean', 'std'),
                                              SSIM=('ssim_mean', 'mean'), SSIM_sd=('ssim_mean', 'std'),
                                              PSNR_last_epoch=('psnr_last_epoch', 'mean'), best_epoch=('best_epoch', 'mean'),
                                              params=('params', 'first'), runs=('seed', 'count')).reset_index()
    base_p = df[df.model == 'Baseline'].params.iloc[0]
    agg['extra_params'] = agg.params - base_p
    order = {m: i for i, m in enumerate(MODELS)}
    agg = agg.sort_values(['angles', 'model'], key=lambda s: s.map(order) if s.name == 'model' else s)
    agg.to_csv(f'{res_dir}/summary_table.csv', index=False)
    P('\n--- results (mean over the complete seeds; sd = spread between seeds) ---')
    P(agg.round({'PSNR': 2, 'PSNR_sd': 3, 'SSIM': 4, 'SSIM_sd': 4, 'PSNR_last_epoch': 2, 'best_epoch': 1}).to_string(index=False))

    def per_image(model, n, key):
        return np.mean([json.load(open(f'{res_dir}/{PROTOCOL}_{model}_{n}_seed{sd}.json', encoding='utf-8'))[key] for sd in done], axis=0)

    def holm(p):
        p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
        for rank, i in enumerate(o):
            run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
        return adj

    rng = np.random.default_rng(0); rows = []
    for n in ANGLES:
        for key, label in [('psnr_per_image', 'PSNR'), ('ssim_per_image', 'SSIM')]:
            for ref in ['Baseline', 'CBAM']:
                for m in MODELS:
                    if m == ref: continue
                    d = per_image(m, n, key) - per_image(ref, n, key)
                    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                    try: p = wilcoxon(d).pvalue
                    except ValueError: p = 1.0
                    rows.append(dict(angles=n, metric=label, model=m, vs=ref, mean_diff=d.mean(), median_diff=np.median(d),
                                     ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5), p_raw=p))
    T = pd.DataFrame(rows)
    T['p_holm'] = T.groupby(['angles', 'metric', 'vs']).p_raw.transform(lambda s: holm(s.values))
    T.to_csv(f'{res_dir}/paired_tests.csv', index=False)
    pd.set_option('display.float_format', lambda v: f'{v:.4g}')
    pd.set_option('display.width', 220)
    P('\n--- paired tests over the 200 test images (per-image values averaged over seeds) ---')
    P(T.to_string(index=False))
    P('\n(Wilcoxon is about the median difference, the bootstrap interval about the mean; they can disagree when differences are skewed.)')
    txt = '\n'.join(lines)
    with open(f'{res_dir}/report.txt', 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    log(f'report written to {res_dir}/report.txt')


# =====================================================================
# Computational cost (run ONCE on ONE GPU, and report which GPU)
# =====================================================================
def cost():
    import pandas as pd
    from torch.utils.flop_counter import FlopCounterMode
    os.makedirs(f'{OUT}/results', exist_ok=True)
    x = torch.randn(1, 1, 128, 128, device=device); rows = []
    for name, cls in MODELS.items():
        m = cls().to(device).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x)
        flops = fc.get_total_flops()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(30): m(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(200): m(x)
            if device.type == 'cuda': torch.cuda.synchronize()
        ms = (time.time() - t0) / 200 * 1000
        mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else float('nan')
        rows.append(dict(model=name, GFLOPs=flops / 1e9, latency_ms=ms, peak_mem_MB=mem, gpu=gpu_name(), torch=torch.__version__))
        del m
    df = pd.DataFrame(rows); df.to_csv(f'{OUT}/results/cost.csv', index=False)
    print(df.round(3).to_string(index=False))
    log('cost table written to results/cost.csv  (GPU: ' + gpu_name() + ')')


def main():
    global OUT, NUM_WORKERS, SMOKE, device, PROTOCOL
    ap = argparse.ArgumentParser(description='SimCBAM v3 training / reporting')
    ap.add_argument('--out', required=True, help='folder with the .npy data (and results/, ckpt/), e.g. runs/v3')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers (keep 0 on Windows)')
    ap.add_argument('--smoke', action='store_true', help='1-2 minute environment test (tiny subset, 2 epochs, own file names)')
    ap.add_argument('--report', action='store_true', help='write summary_table.csv, paired_tests.csv and report.txt')
    ap.add_argument('--cost', action='store_true', help='measure FLOPs / latency / memory (run once, on one GPU)')
    ap.add_argument('--allow-cpu', action='store_true', help='train on the CPU (extremely slow, only for debugging)')
    a = ap.parse_args()
    OUT = a.out.replace('\\', '/').rstrip('/'); NUM_WORKERS = a.workers; SMOKE = a.smoke
    if SMOKE: PROTOCOL = 'smoke'
    if not os.path.isdir(OUT):
        sys.exit(f'ERROR: --out folder not found: {OUT}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    log(f'python {sys.version.split()[0]} | torch {torch.__version__} | device {device} | {gpu_name()}')
    if device.type == 'cuda':
        log('CUDA architectures supported by this PyTorch build: ' + ' '.join(torch.cuda.get_arch_list()))
    if a.report:
        report(a.seeds)
    elif a.cost:
        cost()
    else:
        if device.type == 'cpu' and not a.allow_cpu:
            sys.exit('ERROR: no CUDA GPU is visible to PyTorch. Install the CUDA build of PyTorch (see the instructions) '
                     'or pass --allow-cpu for a debugging run.')
        train_all(a.seeds)


if __name__ == '__main__':
    main()
