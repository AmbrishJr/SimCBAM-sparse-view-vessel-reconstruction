from sparse_pa_unet.models.unet import UNet
from sparse_pa_unet.models.unet_cbam import UNet_CBAM
from sparse_pa_unet.models.unet_se import UNet_SE
from sparse_pa_unet.models.unet_simam import UNet_SimAM
from sparse_pa_unet.models.unet_simcbam import UNet_SimCBAM

__all__ = [
    "UNet",
    "UNet_SE",
    "UNet_CBAM",
    "UNet_SimAM",
    "UNet_SimCBAM",
]

MODEL_REGISTRY = {
    "baseline": UNet,
    "se": UNet_SE,
    "cbam": UNet_CBAM,
    "simam": UNet_SimAM,
    "simcbam": UNet_SimCBAM,
}

# Display names used by load_all_models and recommendation tables.
MODEL_NAMES = {
    "baseline": "Baseline",
    "se": "SE",
    "cbam": "CBAM",
    "simam": "SimAM",
    "simcbam": "SimCBAM",
}


def get_model_class(name):
    return MODEL_REGISTRY[name]