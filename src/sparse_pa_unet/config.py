import os

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")


def _resolve_env(config):
    if "DATA_ROOT" in os.environ:
        config["data"]["root"] = os.environ["DATA_ROOT"]
    if "CKPT_ROOT" in os.environ:
        config["checkpoints"]["root"] = os.environ["CKPT_ROOT"]
    return config


def resolve_data_dirs(config, root=None):
    root = root or config["data"]["root"]
    sparse_dirs = {}
    for sparsity, rel in config["data"]["sparse_dirs"].items():
        sparse_dirs[int(sparsity)] = os.path.join(root, rel)
    gt_dir = os.path.join(root, config["data"]["gt_dir"])
    return sparse_dirs, gt_dir


def get_device(config=None):
    if config is None:
        config = load_config()
    setting = config["training"].get("device", "auto")
    import torch

    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def load_config(path=None):
    path = path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return _resolve_env(config)