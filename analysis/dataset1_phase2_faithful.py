"""PHASE 2b - FAITHFUL reproduction of the user's published MobileNetV3-Small baseline on S1,
plus a direct re-evaluation of their ORIGINAL checkpoint to verify the published 0.6733.

  .venv/bin/python analysis/dataset1_phase2_faithful.py --mode original
  .venv/bin/python analysis/dataset1_phase2_faithful.py --mode reproduce --tag run1
  .venv/bin/python analysis/dataset1_phase2_faithful.py --mode reproduce --tag run2

Everything matches the notebook (Tomato_Leaf_Analysis.ipynb cells 9-15):
  Resize 224x224, RandomHorizontalFlip, RandomRotation(15), ImageNet normalisation
  batch 16, num_workers 0, class-weighted CrossEntropyLoss, AdamW lr=1e-4 wd=1e-4,
  ReduceLROnPlateau(mode=max, patience=2, factor=0.5), 10 epochs, best model by VAL ACCURACY,
  2000-iteration stratified bootstrap CI with numpy default_rng(42).

ONE DELIBERATE DEVIATION (logged, HARD RULE 5 "fixed seeds"): the notebook seeded only
torch and numpy; torchvision's flip/rotation draws from Python's `random`, which the notebook
left unseeded, so its run was not bit-reproducible. This script also seeds `random`.

Writes results/phase2/faithful_<mode>_<tag>.json and ..._preds.csv
"""
from __future__ import annotations

import argparse
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
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from common import IMAGENET_MEAN, IMAGENET_STD, SEED  # noqa: E402

SPLITS = RES / "splits_S1"
CLAIMS = json.loads((ROOT / "analysis" / "reference_claims.json").read_text())
ORIG_CKPT = Path.home() / "Downloads" / "Tomato_Research_Results" / "best_mobilenetv3.pth"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TRAIN_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(15), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])


class SetDataset(torch.utils.data.Dataset):
    def __init__(self, df, c2i, tf):
        self.df, self.c2i, self.tf = df.reset_index(drop=True), c2i, tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with __import__("PIL.Image", fromlist=["Image"]).open(r["path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.c2i[r["class"]]


def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    lg, lb = [], []
    for x, y in loader:
        lg.append(model(x.to(dev)).cpu().numpy()); lb.append(y.numpy())
    return np.concatenate(lg), np.concatenate(lb)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def stratified_bootstrap_ci(y, pred, n_classes, iters=2000, seed=SEED):
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


def evaluate(model, classes, dev, tag: str) -> dict:
    c2i = {c: i for i, c in enumerate(classes)}
    test = pd.read_csv(SPLITS / "test.csv")
    loader = DataLoader(SetDataset(test, c2i, EVAL_TF), batch_size=16, shuffle=False, num_workers=0)
    lg, y = predict(model, loader, dev)
    p = softmax(lg)
    pred = p.argmax(1)
    n = len(classes)
    yb = label_binarize(y, classes=np.arange(n))

    cm = confusion_matrix(y, pred, labels=np.arange(n))
    spec = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp; tn = cm.sum() - tp - fp - fn
        spec[c] = float(tn / (tn + fp)) if (tn + fp) else None

    acc_ci, f1_ci = stratified_bootstrap_ci(y, pred, n)
    acc = accuracy_score(y, pred)
    m = {
        "accuracy": float(acc),
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
        "per_class_report": classification_report(
            y, pred, labels=np.arange(n), target_names=classes, output_dict=True,
            zero_division=0),
        "class_support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
    }
    pd.DataFrame({"path": test["path"], "y_true": y, "y_pred": pred,
                  "confidence": p.max(1), **{f"prob_{i}": p[:, i] for i in range(n)}}).to_csv(
        P2 / f"faithful_{tag}_preds.csv", index=False)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["original", "reproduce"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed for the seed-variance runs (default 42 = the notebook's seed)")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="NEGATIVE CONTROL: permute training labels; test accuracy must fall to "
                         "chance (~1/7). A high value here means the pipeline is broken/leaky.")
    a = ap.parse_args()
    tag = a.tag or a.mode
    t0 = time.time()
    seed_all(a.seed)                              # includes Python `random` (see module docstring)

    classes = json.loads((SPLITS / "classes.json").read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    tr = pd.read_csv(SPLITS / "train.csv")
    va = pd.read_csv(SPLITS / "validation.csv")
    if a.shuffle_labels:
        # NEGATIVE CONTROL: destroy the image->label relation while keeping the class
        # distribution. Test accuracy must collapse to chance (~1/7 = 0.143); if it does not,
        # the evaluation pipeline is broken or leaking.
        rng = np.random.default_rng(a.seed)
        tr = tr.copy()
        tr["class"] = tr["class"].to_numpy()[rng.permutation(len(tr))]
    dev = torch.device("cpu")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    note = ""

    if a.mode == "original":
        sd = torch.load(ORIG_CKPT, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"FATAL: original checkpoint does not match architecture: "
                             f"missing={missing} unexpected={unexpected}")
        note = "evaluated the user's own saved checkpoint best_mobilenetv3.pth"
        hist = None
    else:
        model = model.to(dev)
        counts = tr["class"].value_counts().reindex(classes).values
        w = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32)
        crit = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
        tl = DataLoader(SetDataset(tr, c2i, TRAIN_TF), batch_size=16, shuffle=True, num_workers=0)
        vl = DataLoader(SetDataset(va, c2i, EVAL_TF), batch_size=16, shuffle=False, num_workers=0)
        best_acc, hist, best_sd = -1.0, [], None
        for ep in range(1, a.epochs + 1):
            model.train()
            s, corr, tot = 0.0, 0, 0
            for x, y in tl:
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward(); opt.step()
                s += loss.item() * len(y); corr += int((out.argmax(1) == y).sum()); tot += len(y)
            model.eval()
            vl_c, vl_n = 0, 0
            with torch.no_grad():
                for x, y in vl:
                    out = model(x)
                    vl_c += int((out.argmax(1) == y).sum()); vl_n += len(y)
            vacc = vl_c / vl_n
            sched.step(vacc)
            hist.append({"epoch": ep, "train_loss": s / tot, "train_accuracy": corr / tot,
                         "val_accuracy": vacc, "lr": opt.param_groups[0]["lr"]})
            if vacc > best_acc:
                best_acc = vacc
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {ep}: train_acc {corr/tot:.4f}  val_acc {vacc:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        model.load_state_dict(best_sd)
        pd.DataFrame(hist).to_csv(P2 / f"faithful_{tag}_history.csv", index=False)
        note = (f"faithful re-training, {a.epochs} epochs, best val accuracy {best_acc:.4f}; "
                f"Python `random` seeded (the original notebook did not seed it)"
                + ("; NEGATIVE CONTROL: training labels permuted" if a.shuffle_labels else ""))

    m = evaluate(model, classes, dev, tag)

    # ---------------- comparison with the published claim ------------------------
    pub = CLAIMS["phase2_published_baseline"]["test_accuracy"]
    pub_ci = [0.6355555555555555, 0.7111111111111111]   # from final_results.json, read by the caller
    m.update({
        "mode": a.mode, "tag": tag, "note": note, "seed": a.seed, "epochs": a.epochs,
        "shuffle_labels": bool(a.shuffle_labels),
        "chance_accuracy_1_over_n_classes": round(1 / len(classes), 6),
        "torch": torch.__version__, "runtime_s": round(time.time() - t0, 1),
        "published_accuracy_claim": pub,
        "published_accuracy_95_ci": pub_ci,
        "abs_diff_vs_published": round(abs(m["accuracy"] - pub), 6),
        "published_acc_inside_this_ci": bool(m["accuracy_95_ci"][0] <= pub <= m["accuracy_95_ci"][1]),
        "this_acc_inside_published_ci": bool(pub_ci[0] <= m["accuracy"] <= pub_ci[1]),
    })
    (P2 / f"faithful_{tag}.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k not in ("per_class_report", "confusion_matrix")},
                     indent=2))


if __name__ == "__main__":
    main()
