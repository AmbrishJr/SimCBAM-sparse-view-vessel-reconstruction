import torch

from sparse_pa_unet.models import (
    MODEL_REGISTRY,
    SparsityClassifier,
    UNet,
    UNet_CBAM,
    UNet_SE,
    UNet_SimAM,
    UNet_SimCBAM,
)


def _make_input(batch=2):
    return torch.randn(batch, 1, 128, 128)


def test_all_models_forward():
    x = _make_input()
    for cls in [UNet, UNet_SE, UNet_CBAM, UNet_SimAM, UNet_SimCBAM]:
        m = cls(in_ch=1, out_ch=1, base=32)
        out = m(x)
        assert out.shape == x.shape, f"{cls.__name__}: expected {x.shape}, got {out.shape}"


def test_model_registry():
    x = _make_input(1)
    for name, cls in MODEL_REGISTRY.items():
        out = cls(in_ch=1, out_ch=1, base=32)(x)
        assert out.shape == x.shape
        assert list(cls().__dict__.keys())


def test_sparsity_classifier_forward():
    m = SparsityClassifier(num_classes=3)
    out = m(_make_input(2))
    assert out.shape == (2, 3), f"expected (2,3), got {out.shape}"


def test_state_dict_roundtrip():
    m = UNet(in_ch=1, out_ch=1, base=32)
    torch.save(m.state_dict(), "/tmp/test_unet.pth")
    m2 = UNet(in_ch=1, out_ch=1, base=32)
    m2.load_state_dict(torch.load("/tmp/test_unet.pth", map_location="cpu"))
    m.eval()
    m2.eval()
    with torch.no_grad():
        x = _make_input(1)
        a, b = m(x), m2(x)
    assert torch.allclose(a, b), "state_dict roundtrip should preserve outputs"