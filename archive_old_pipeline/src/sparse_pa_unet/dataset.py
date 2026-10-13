import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PADataset(Dataset):
    def __init__(self, sparse_dir, gt_dir):
        self.sparse_dir, self.gt_dir = sparse_dir, gt_dir
        self.files = sorted([f for f in os.listdir(sparse_dir) if f.endswith(".tif")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        sparse = np.array(Image.open(os.path.join(self.sparse_dir, fname))).astype(np.float32) / 255.0
        gt = np.array(Image.open(os.path.join(self.gt_dir, fname))).astype(np.float32) / 255.0
        return torch.from_numpy(sparse).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)


class SparsityLevelDataset(Dataset):
    def __init__(self, sparse_dirs, split="train"):
        self.samples = []
        for label, sparsity in enumerate(sorted(sparse_dirs)):
            folder = os.path.join(sparse_dirs[sparsity], split)
            for fname in os.listdir(folder):
                if fname.endswith(".tif"):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path)).astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0), label