# Demo: NOT part of the reported results

The Gradio app and the CNN sparsity-level classifier in this folder are **not described in the manuscript and
were not used for any reported result**. They are kept as an unmaintained interactive demo.

Caveats:

- They are built on the **archived first pipeline** (`../archive_old_pipeline/`). That pipeline simulates inputs with
  `circle=True`, uses checkpoints trained on uint8-wrapped inputs, and imports the archived `sparse_pa_unet` package.
- The model recommendations and the numbers shown by the app come from the archived results. They do not match the
  manuscript.
- The "sparsity detection" classifier is trained on the same simulated Radon/FBP data and says nothing about real
  measurement data.

Run (needs `gradio` and the old checkpoints, which are not distributed):

```bash
pip install gradio pyyaml
python demo/app.py --ckpt-dir path/to/old_checkpoints
```
