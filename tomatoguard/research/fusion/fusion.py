"""Phase 3 - RGB-only vs spectral-only vs feature-fusion vs distilled-RGB-student on TOBRFV-LMID.
Binary task: healthy (0) vs infected (1). Plant-disjoint splits come from caep/split_by_plant.py.

  python fusion.py --mode rgb      --manifest ../plant_splits/manifest_split_seed42.csv --out runs/rgb
  python fusion.py --mode spectral --manifest ... --out runs/spectral
  python fusion.py --mode fusion   --manifest ... --out runs/fusion
  python fusion.py --mode rgb --teacher runs/fusion/best.pt --manifest ... --out runs/rgb_distilled   # distillation

Each run writes preds_test.csv (for caep/paired_stats.py) and metrics.json with image-level,
plant-level and per-angle / per-batch results.

STATUS: written but NOT executed here (no GPU / no dataset in my sandbox). Fusion = concatenation of two
MobileNetV3-Small encoders; cross-attention fusion is a natural extension (see README).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224          # module global; overridden by --image-size (see CHANGELOG "Edits to existing scripts")


class Spec(Dataset):
    def __init__(self, df, train):
        self.df, self.train = df.reset_index(drop=True), train

    def __len__(self):
        return len(self.df)

    def _img(self, path, gray):
        im = Image.open(path)
        im = im.convert("L" if gray else "RGB").resize((SIZE, SIZE))
        return im

    def __getitem__(self, i):
        r = self.df.iloc[i]
        rgb, n8, n10 = self._img(r["rgb"], False), self._img(r["nir800"], True), self._img(r["nir1000"], True)
        if self.train:  # identical geometric aug on all streams; NO photometric aug (keeps radiometry)
            if random.random() < .5:
                rgb, n8, n10 = ImageOps.mirror(rgb), ImageOps.mirror(n8), ImageOps.mirror(n10)
            if random.random() < .5:
                rgb, n8, n10 = ImageOps.flip(rgb), ImageOps.flip(n8), ImageOps.flip(n10)
        a = (np.asarray(rgb, np.float32) / 255 - MEAN) / STD
        g8, g10 = np.asarray(n8, np.float32) / 255, np.asarray(n10, np.float32) / 255
        spec = (np.stack([g8, g10, (g8 + g10) / 2]) - 0.5) / 0.5
        return (torch.from_numpy(a.transpose(2, 0, 1)), torch.from_numpy(spec.astype(np.float32)),
                int(r["label"]), i)


def encoder():
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())  # 576-d


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode in ("rgb", "fusion"):
            self.enc_rgb = encoder()
        if mode in ("spectral", "fusion"):
            self.enc_spec = encoder()
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(576 * (2 if mode == "fusion" else 1), 2))

    def forward(self, rgb, spec):
        f = []
        if self.mode in ("rgb", "fusion"):
            f.append(self.enc_rgb(rgb))
        if self.mode in ("spectral", "fusion"):
            f.append(self.enc_spec(spec))
        return self.head(torch.cat(f, 1))


@torch.no_grad()
def infer(model, loader, dev):
    model.eval()
    out = []
    for rgb, spec, y, idx in loader:
        out.append(F.softmax(model(rgb.to(dev), spec.to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(df, p):
    d = df.assign(p=p)
    res = {"image_acc": accuracy_score(d["label"], d["p"] >= .5), "image_f1": f1_score(d["label"], d["p"] >= .5),
           "image_auc": roc_auc_score(d["label"], d["p"]) if d["label"].nunique() == 2 else None}
    pl = d.groupby("plant_key").agg(y=("label", "first"), p=("p", "mean"))
    res.update({"plant_acc": accuracy_score(pl["y"], pl["p"] >= .5), "n_plants": int(len(pl)),
                "plant_auc": roc_auc_score(pl["y"], pl["p"]) if pl["y"].nunique() == 2 else None})
    for col in ("angle", "batch"):
        res[f"image_acc_by_{col}"] = {str(k): float(accuracy_score(g["label"], g["p"] >= .5))
                                      for k, g in d.groupby(col)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rgb", "spectral", "fusion"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=None, help="path to a fusion best.pt -> distil into --mode student")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224,
                    help="input resolution (CPU-only runs use 160; logged in CHANGELOG)")
    a = ap.parse_args()
    global SIZE
    SIZE = a.image_size
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.manifest)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}
    mk = lambda s, tr: DataLoader(Spec(parts[s], tr), batch_size=a.batch_size, shuffle=tr,
                                  num_workers=a.num_workers)
    tl, vl, sl = mk("train", True), mk("validation", False), mk("test", False)

    model = Net(a.mode).to(dev)
    teacher = None
    if a.teacher:
        teacher = Net("fusion").to(dev)
        teacher.load_state_dict(torch.load(a.teacher, map_location=dev)); teacher.eval()
    cnt = parts["train"]["label"].value_counts().sort_index().values
    w = torch.tensor(len(parts["train"]) / (2 * cnt), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        model.train()
        for rgb, spec, y, _ in tqdm(tl, desc=f"epoch {ep}", leave=False):
            rgb, spec, y = rgb.to(dev), spec.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(rgb, spec)
            loss = F.cross_entropy(logits, y, weight=w)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(rgb, spec)
                kd = F.kl_div(F.log_softmax(logits / a.temp, 1), F.softmax(t_logits / a.temp, 1),
                              reduction="batchmean") * a.temp ** 2
                loss = a.alpha * loss + (1 - a.alpha) * kd
            loss.backward(); opt.step()
        sched.step()
        pv = infer(model, vl, dev)
        auc = roc_auc_score(parts["validation"]["label"], pv)
        print(f"epoch {ep}: val image AUC {auc:.4f}")
        if auc > best:
            best, bad = auc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    p = infer(model, sl, dev)
    res = metrics(parts["test"], p)
    res.update({"mode": a.mode, "distilled_from": a.teacher, "best_val_auc": best, "seed": a.seed,
                "params": sum(x.numel() for x in model.parameters())})
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"plant_key": parts["test"]["plant_key"], "y_true": parts["test"]["label"], "p_infected": p,
                  "day": parts["test"]["day"], "angle": parts["test"]["angle"], "batch": parts["test"]["batch"]}
                 ).to_csv(out / "preds_test.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
