# Archived: first pipeline (superseded, not used for the manuscript)

This folder holds the original code of this repository, kept only for the record. **Do not use it to reproduce
the manuscript.** The current pipeline is `notebooks/SimCBAM_v3.ipynb` / `simcbam_train.py` in the repository root.

## Known problems

1. **uint8 wrap-around of the network inputs.** The sparse FBP reconstructions were saved with
   `sparse.astype(np.uint8)` on a 0–255 scale. FBP produces values outside 0–255 (negative ringing and overshoot),
   and the cast *wraps* them instead of clipping (e.g. −3 → 253). This placed isolated saturated pixels in the
   inputs: about 10% of pixels at 32 angles. The corrected pipeline keeps the inputs as float32.
2. **`circle=True` information loss.** `radon` / `iradon` were called with `circle=True`, which discards everything
   outside the inscribed circle (~21% of the 128 × 128 image). The ground truth still contained vessels there, so the
   networks were asked to reconstruct content absent from their input. The corrected pipeline uses `circle=False`.
3. Single seed, last-epoch evaluation, constant learning rate. Validation loss fluctuated strongly between
   epochs, so last-epoch scores were noisy.

All figures, tables and numbers produced by this code are affected, including those in `README_old.md`, the
`evaluation_results.csv` outputs and the `RECOMMENDATION_TABLE` in `src/sparse_pa_unet/inference.py`.

## Changes since archiving

The files were moved here unchanged, with one exception: the Gradio demo and the sparsity classifier were moved to
[`../demo/`](../demo/). `SparsityClassifier` and `load_sparsity_classifier` were therefore removed from
`src/sparse_pa_unet/models/__init__.py`, `inference.py` and `tests/test_models.py`. Cells in
`notebooks/RESEARCH_INTERN_clean.ipynb` that import `SparsityClassifier` from the package will fail as a result.
