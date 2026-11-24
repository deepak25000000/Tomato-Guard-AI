"""Step 8 - Train on the harmonised multi-dataset table.

  python 08_train_unified.py --mode lodo  --unified unified/all.csv --arch mobilenet_v3_small
      -> leave-one-dataset-out: for each dataset S, train on all OTHER datasets (every group that touches S is
         removed from training, so cross-dataset duplicates cannot leak) and test on S.
  python 08_train_unified.py --mode final --unified unified/all.csv --arch mobilenet_v3_small --out runs_unified/final
      -> train on everything (group-disjoint split); writes best.pt + meta.json usable by 04_export.py,
         plus per-dataset test metrics. Use the LODO numbers, not these, as your generalisation estimate.
Optional: --source-balance samples so no single big dataset dominates.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from common import IMAGENET_MEAN, IMAGENET_STD, SEED
from models import ARCHS, ImageCSV, build_model, eval_tf, fit_temperature, predict_logits, train_tf


def group_split(df, frac, seed):
    sg = StratifiedGroupKFold(n_splits=max(2, round(1 / frac)), shuffle=True, random_state=seed)
    a, b = next(sg.split(df, df["class"].values, df["group"]))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def fit(train, val, classes, a, out, dev):
    c2i = {c: i for i, c in enumerate(classes)}
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=dev.type == "cuda")
    if a.source_balance:
        cnt = train.groupby(["source", "class"])["path"].transform("count").values
        sampler = WeightedRandomSampler(1.0 / cnt, num_samples=len(train), replacement=True)
        tl = DataLoader(ImageCSV(train, c2i, train_tf()), sampler=sampler, **kw)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    else:
        tl = DataLoader(ImageCSV(train, c2i, train_tf()), shuffle=True, **kw)
        n = train["class"].value_counts().reindex(classes).fillna(1).values
        w = torch.tensor(len(train) / (len(classes) * n), dtype=torch.float32, device=dev)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    vl = DataLoader(ImageCSV(val, c2i, eval_tf()), **kw)
    model = build_model(a.arch, len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tqdm(tl, desc=f"epoch {ep}", leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        lg, lb = predict_logits(model, vl, dev)
        f1 = f1_score(lb, lg.argmax(1), average="macro", zero_division=0)
        print(f"  epoch {ep}: val macro-F1 {f1:.4f}")
        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    lg, lb = predict_logits(model, vl, dev)
    return model, fit_temperature(lg, lb), best


def score(model, df, classes, dev, bs=32, workers=2):
    c2i = {c: i for i, c in enumerate(classes)}
    lg, y = predict_logits(model, DataLoader(ImageCSV(df, c2i, eval_tf()), batch_size=bs, num_workers=workers), dev)
    p = lg.argmax(1)
    present = sorted(set(y))
    return y, p, {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
                  "macro_f1_present": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
                  "classes_present": [classes[i] for i in present]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lodo", "final"], required=True)
    ap.add_argument("--unified", default="unified/all.csv")
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=list(ARCHS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--source-balance", action="store_true")
    ap.add_argument("--holdout", default=None, help="comma list of datasets for lodo (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.unified).rename(columns={"label": "class"})
    classes = json.loads((Path(a.unified).parent / "classes.json").read_text())
    out = Path(a.out or f"runs_unified/{a.mode}_{a.arch}")
    out.mkdir(parents=True, exist_ok=True)

    if a.mode == "lodo":
        sources = a.holdout.split(",") if a.holdout else sorted(df["source"].unique())
        rows = []
        for s in sources:
            test = df[df["source"] == s].reset_index(drop=True)
            leaky = set(test["group"])
            pool = df[(df["source"] != s) & ~df["group"].isin(leaky)].reset_index(drop=True)
            print(f"\n=== hold out {s}: test {len(test)} imgs | train pool {len(pool)} "
                  f"(removed {int((df['source'] != s).sum() - len(pool))} images sharing a duplicate group with {s})")
            train, val = group_split(pool, 0.12, a.seed)
            model, T, best = fit(train, val, classes, a, out / s, dev)
            y, p, m = score(model, test, classes, dev)
            m.update({"held_out": s, "val_macro_f1": best, "temperature": T, "n_train": len(train)})
            rep = pd.DataFrame(classification_report(y, p, labels=sorted(set(y)),
                               target_names=[classes[i] for i in sorted(set(y))], output_dict=True, zero_division=0)).T
            rep.to_csv(out / s / "per_class.csv")
            rows.append(m)
            print(f"  {s}: accuracy {m['accuracy']:.3f}  macro-F1 {m['macro_f1_present']:.3f}")
        pd.DataFrame(rows).to_csv(out / "lodo_summary.csv", index=False)
        print("\nLODO summary:\n", pd.DataFrame(rows)[["held_out", "n", "accuracy", "macro_f1_present"]].round(3).to_string(index=False))
    else:
        rest, test = group_split(df, 0.15, a.seed)
        train, val = group_split(rest, 0.15 / 0.85, a.seed)
        for n, d in (("train", train), ("val", val), ("test", test)):
            d.to_csv(out / f"split_{n}.csv", index=False)
        model, T, best = fit(train, val, classes, a, out, dev)
        y, p, m = score(model, test, classes, dev)
        per_src = {s: score(model, g.reset_index(drop=True), classes, dev)[2]
                   for s, g in test.groupby("source")}
        (out / "meta.json").write_text(json.dumps({
            "arch": a.arch, "classes": classes, "input_size": 224, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "temperature": T, "seed": a.seed, "best_val_macro_f1": best, "torch": torch.__version__,
            "sources": sorted(df["source"].unique()), "test_overall": m, "test_per_source": per_src}, indent=2))
        print(json.dumps({"overall": m, "per_source": per_src}, indent=2))


if __name__ == "__main__":
    main()
