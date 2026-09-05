"""sparse-pa-unet: photoacoustic sparse-view reconstruction with attention U-Nets."""

__version__ = "0.1.0"

from sparse_pa_unet.config import load_config

__all__ = ["load_config", "__version__"]