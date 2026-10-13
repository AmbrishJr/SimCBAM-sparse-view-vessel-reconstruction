#!/usr/bin/env python3
"""
Regenerate the results tables in README.md from the CSV files written by simcbam_train.py --report / --cost
(or the notebook): results/summary_table.csv, results/paired_tests.csv, results/cost.csv.

Everything between the markers <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md is replaced.
Missing CSV files produce a placeholder, so the README never shows numbers that were not produced by the code.

USAGE (from the repository root):
    python scripts/make_readme_tables.py                      # reads results/, rewrites README.md
    python scripts/make_readme_tables.py --results runs/v3/results --check   # print only, do not write
"""
import argparse, os, re, sys
import pandas as pd

START, END = '<!-- RESULTS:START -->', '<!-- RESULTS:END -->'
MODEL_ORDER = ['Baseline', 'SE', 'CBAM', 'SimAM', 'SimCBAM']
PLACEHOLDER = '_Pending: `{}` not found. Run `python simcbam_train.py --out runs/v3 {}` and copy the file into `results/`._'


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def pm(mean, sd, digits):
    if pd.isna(sd):
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


def by_model_order(df):
    return df.assign(_o=df.model.map({m: i for i, m in enumerate(MODEL_ORDER)})).sort_values(['angles', '_o']).drop(columns='_o')


def summary_section(path):
    if not os.path.exists(path):
        return PLACEHOLDER.format('summary_table.csv', '--seeds 0 1 2 --report')
    s = by_model_order(pd.read_csv(path))
    runs = sorted(s.runs.unique())
    out = pd.DataFrame({
        'Angles': s.angles,
        'Model': s.model,
        'PSNR (dB)': [pm(m, sd, 2) for m, sd in zip(s.PSNR, s.PSNR_sd)],
        'SSIM': [pm(m, sd, 4) for m, sd in zip(s.SSIM, s.SSIM_sd)],
        'Extra params vs Baseline': [f'{int(v):,}' for v in s.extra_params],
        'Mean best epoch': s.best_epoch.round(1),
    })
    note = (f'Mean ± SD across {"/".join(str(r) for r in runs)} training seed(s); each value is the mean over the 200 test images, '
            'evaluated with the best-validation-loss checkpoint and outputs clipped to [0, 1].')
    return md_table(out) + '\n\n' + note


def paired_section(path):
    if not os.path.exists(path):
        return PLACEHOLDER.format('paired_tests.csv', '--seeds 0 1 2 --report')
    t = by_model_order(pd.read_csv(path))
    t = t.sort_values(['vs', 'metric', 'angles'], kind='stable')
    digits = lambda metric: 3 if metric == 'PSNR' else 4
    out = pd.DataFrame({
        'Reference': t.vs,
        'Metric': t.metric,
        'Angles': t.angles,
        'Model': t.model,
        'Mean Δ [95% CI]': [f'{m:+.{digits(k)}f} [{lo:+.{digits(k)}f}, {hi:+.{digits(k)}f}]'
                            for m, lo, hi, k in zip(t.mean_diff, t.ci_low, t.ci_high, t.metric)],
        'Median Δ': [f'{v:+.{digits(k)}f}' for v, k in zip(t.median_diff, t.metric)],
        'p (Holm)': [f'{p:.3g}' for p in t.p_holm],
    })
    note = ('Paired over the 200 test images (per-image values averaged over seeds). Δ = model − reference. '
            'CI: bootstrap (2000 resamples) of the mean difference; p: Wilcoxon signed-rank test (median difference), '
            'Holm-corrected within each angle × metric × reference group.')
    return '<details>\n<summary>Paired comparisons (click to expand)</summary>\n\n' + md_table(out) + '\n\n' + note + '\n\n</details>'


def cost_section(path):
    if not os.path.exists(path):
        return PLACEHOLDER.format('cost.csv', '--cost')
    c = pd.read_csv(path)
    c = c.assign(_o=c.model.map({m: i for i, m in enumerate(MODEL_ORDER)})).sort_values('_o')
    out = pd.DataFrame({
        'Model': c.model,
        'GFLOPs': c.GFLOPs.round(3),
        'Latency (ms)': c.latency_ms.round(2),
        'Peak GPU memory (MB)': ['n/a' if pd.isna(v) else f'{v:.1f}' for v in c.peak_mem_MB],
    })
    hw = f" Measured on {c.gpu.iloc[0]}, torch {c.torch.iloc[0]}." if {'gpu', 'torch'} <= set(c.columns) else ''
    note = ('Single 1×1×128×128 input, batch 1, mean of 200 forward passes after 30 warm-up passes. '
            'FLOPs count convolution / linear layers only (element-wise attention operations are not counted).' + hw)
    return md_table(out) + '\n\n' + note


def build(results_dir):
    return '\n\n'.join([
        '### Reconstruction quality (test set)', summary_section(f'{results_dir}/summary_table.csv'),
        '### Statistical comparison', paired_section(f'{results_dir}/paired_tests.csv'),
        '### Computational cost', cost_section(f'{results_dir}/cost.csv'),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results', help='folder with summary_table.csv, paired_tests.csv, cost.csv')
    ap.add_argument('--readme', default='README.md')
    ap.add_argument('--check', action='store_true', help='print the generated section instead of writing README.md')
    a = ap.parse_args()
    section = build(a.results.rstrip('/'))
    if a.check:
        print(section)
        return
    text = open(a.readme, encoding='utf-8').read()
    if START not in text or END not in text:
        sys.exit(f'{a.readme} has no {START} ... {END} markers')
    text = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: f'{START}\n{section}\n{END}', text, flags=re.S)
    with open(a.readme, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'{a.readme}: results section updated from {a.results}/')


if __name__ == '__main__':
    main()
