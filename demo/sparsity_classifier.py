import os

import torch
import torch.nn as nn


class SparsityClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_sparsity_classifier(path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if path is None:
        from sparse_pa_unet.config import load_config
        path = os.path.join(load_config()["checkpoints"]["root"], "sparsity_classifier.pth")
    model = SparsityClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
