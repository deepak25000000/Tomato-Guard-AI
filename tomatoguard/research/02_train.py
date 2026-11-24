"""Step 2 - Train a classifier on the leakage-aware splits.

  python 02_train.py --arch mobilenet_v3_small
  python 02_train.py --arch mobilenet_v3_large
  python 02_train.py --arch efficientnet_b0
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import IMAGENET_MEAN, IMAGENET_STD, SEED
from models import ARCHS, ImageCSV, build_model, eval_tf, fit_temperature, predict_logits, train_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="splits")
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    sd = Path(a.splits_dir)
    classes = json.loads((sd / "classes.json").read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    tr, va = pd.read_csv(sd / "train.csv"), pd.read_csv(sd / "validation.csv")
    out = Path(a.out or f"runs/{a.arch}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    mk = lambda d, t, sh: DataLoader(ImageCSV(d, c2i, t), batch_size=a.batch_size, shuffle=sh,
                                     num_workers=a.num_workers, pin_memory=dev.type == "cuda")
    tr_loader, va_loader = mk(tr, train_tf(), True), mk(va, eval_tf(), False)

    model = build_model(a.arch, len(classes)).to(dev)
    counts = tr["class"].value_counts().reindex(classes).values
    w = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, bad, hist = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y in tqdm(tr_loader, desc=f"epoch {ep}/{a.epochs}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        lg, lb = predict_logits(model, va_loader, dev)
        f1 = f1_score(lb, lg.argmax(1), average="macro", zero_division=0)
        acc = float((lg.argmax(1) == lb).mean())
        hist.append({"epoch": ep, "train_loss": tl / n, "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {ep}: loss {tl/n:.4f}  val acc {acc:.4f}  val macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping."); break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    lg, lb = predict_logits(model, va_loader, dev)
    temperature = fit_temperature(lg, lb)
    print(f"Best val macro-F1 {best:.4f}; fitted temperature {temperature:.3f}")
    (out / "meta.json").write_text(json.dumps({
        "arch": a.arch, "classes": classes, "input_size": 224,
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "temperature": temperature,
        "seed": a.seed, "best_val_macro_f1": best,
        "torch": torch.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
