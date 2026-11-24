"""Shared, tested helpers for the Phase 2 Track A experiments (S2 split + architecture sweep).

Kept separate from analysis/dataset1_phase2_faithful.py so the *faithful reproduction*
of the user's notebook stays byte-faithful and untouched, while the S2 / architecture
runs share one audited implementation.

Every metric is computed by these functions; nothing is typed by hand.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from common import SEED  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(15), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])


class LeafDataset(torch.utils.data.Dataset):
    """Image-level dataset (S1/S2). Reads `path` + `class` columns from a split CSV."""

    def __init__(self, df: pd.DataFrame, c2i: dict, tf):
        self.df, self.c2i, self.tf = df.reset_index(drop=True), c2i, tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.c2i[r["class"]]


def seed_all(s: int = SEED) -> None:
    # NOTE: Python's `random` must be seeded too - the user's notebook omitted it, which is why
    # its run was not bit-reproducible (logged in CHANGELOG / faithful script docstring).
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    lg, lb = [], []
    for x, y in loader:
        lg.append(model(x.to(dev)).cpu().numpy())
        lb.append(y.numpy())
    return np.concatenate(lg), np.concatenate(lb)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def stratified_bootstrap_ci(y, pred, n_classes, iters=2000, seed=SEED):
    """Exactly the notebook's procedure (cell 15): per-class resampling with replacement."""
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(iters):
        idx = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum(), replace=True)
                              for k in range(n_classes) if (y == k).any()])
        accs.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], labels=np.arange(n_classes),
                            average="macro", zero_division=0))
    return (np.percentile(accs, [2.5, 97.5]).tolist(),
            np.percentile(f1s, [2.5, 97.5]).tolist())


def evaluate(model, classes, split_dir: Path, dev, out_tag: str, out_dir: Path,
             extra: dict | None = None, split_name: str = "test") -> dict:
    """Full metric suite on one split. Writes <out_dir>/<out_tag>_preds.csv.

    split_name is "test" for finished models (HARD RULE 3) or "validation" when comparing
    candidates that have not yet been selected.
    """
    c2i = {c: i for i, c in enumerate(classes)}
    df = pd.read_csv(split_dir / f"{split_name}.csv")
    loader = DataLoader(LeafDataset(df, c2i, EVAL_TF), batch_size=16, shuffle=False, num_workers=0)
    lg, y = predict(model, loader, dev)
    p = softmax(lg)
    pred = p.argmax(1)
    n = len(classes)
    yb = label_binarize(y, classes=np.arange(n))

    cm = confusion_matrix(y, pred, labels=np.arange(n))
    spec = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        tn = cm.sum() - tp - fp - fn
        spec[c] = float(tn / (tn + fp)) if (tn + fp) else None

    acc_ci, f1_ci = stratified_bootstrap_ci(y, pred, n)
    rep = classification_report(y, pred, labels=np.arange(n), target_names=classes,
                                output_dict=True, zero_division=0)
    m = {
        "accuracy": float(accuracy_score(y, pred)),
        "accuracy_95_ci": [float(x) for x in acc_ci],
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_95_ci": [float(x) for x in f1_ci],
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(yb, p, average="macro")),
        "macro_pr_auc": float(average_precision_score(yb, p, average="macro")),
        "specificity": spec,
        "n_test": int(len(y)),
        "confusion_matrix": cm.tolist(),
        "per_class_report": rep,
        "class_support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
    }
    if extra:
        m.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": df["path"], "y_true": y, "y_pred": pred, "confidence": p.max(1),
                  **{f"prob_{i}": p[:, i] for i in range(n)}}).to_csv(
        out_dir / f"{out_tag}_preds.csv", index=False)
    return m


def train_one(model, train_csv, val_csv, classes, dev, epochs: int, batch_size: int = 16,
              lr: float = 1e-4, wd: float = 1e-4, seed: int = SEED, log_prefix: str = "") -> dict:
    """The notebook's training recipe (cells 9-10), factored out.

    Returns {"model", "best_val_accuracy", "history", "best_epoch"}.
    """
    seed_all(seed)
    c2i = {c: i for i, c in enumerate(classes)}
    tr, va = pd.read_csv(train_csv), pd.read_csv(val_csv)
    counts = tr["class"].value_counts().reindex(classes).values
    w = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
    tl = DataLoader(LeafDataset(tr, c2i, TRAIN_TF), batch_size=batch_size, shuffle=True,
                    num_workers=0)
    vl = DataLoader(LeafDataset(va, c2i, EVAL_TF), batch_size=batch_size, shuffle=False,
                    num_workers=0)
    best_acc, best_sd, best_ep, hist = -1.0, None, -1, []
    for ep in range(1, epochs + 1):
        model.train()
        s, corr, tot = 0.0, 0, 0
        for x, y in tl:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            s += loss.item() * len(y)
            corr += int((out.argmax(1) == y).sum())
            tot += len(y)
        model.eval()
        vc, vn = 0, 0
        with torch.no_grad():
            for x, y in vl:
                vc += int((model(x).argmax(1) == y).sum())
                vn += len(y)
        vacc = vc / vn
        sched.step(vacc)
        hist.append({"epoch": ep, "train_loss": s / tot, "train_accuracy": corr / tot,
                     "val_accuracy": vacc, "lr": opt.param_groups[0]["lr"]})
        if vacc > best_acc:
            best_acc, best_ep = vacc, ep
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"{log_prefix}epoch {ep}/{epochs}: train_acc {corr/tot:.4f} "
              f"val_acc {vacc:.4f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
    model.load_state_dict(best_sd)
    return {"model": model, "best_val_accuracy": best_acc, "best_epoch": best_ep,
            "history": hist}
