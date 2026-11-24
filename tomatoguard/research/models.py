"""Model + data helpers (needs torch)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from common import IMAGENET_MEAN, IMAGENET_STD

ARCHS = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    fn, weights = ARCHS[arch]
    m = fn(weights=weights if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def train_tf(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_tf(size=224):
    # Plain resize to size x size: this is exactly what the mobile app does.
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageCSV(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform):
        self.df, self.c2i, self.tf = df.reset_index(drop=True), class_to_idx, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.c2i[r["class"]]


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017) fitted on validation logits."""
    lg, lb = torch.tensor(logits, dtype=torch.float32), torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
