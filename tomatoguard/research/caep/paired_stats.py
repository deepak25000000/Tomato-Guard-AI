"""CAEP step 3 - paired, plant-level comparison of two models (A vs B).

Each prediction CSV needs: plant_key, y_true, p_infected  (image rows are averaged per plant).
  python paired_stats.py --a runs/rgb/preds_test.csv --b runs/fusion/preds_test.csv
Reports plant-level accuracy/AUC of each model, the paired-bootstrap 95% CI of (B - A) and an exact McNemar test.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def plant_level(path):
    d = pd.read_csv(path).groupby("plant_key").agg(y=("y_true", "first"), p=("p_infected", "mean"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    A, B = plant_level(args.a), plant_level(args.b)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    y = A["y"].values
    pa, pb = A["p"].values, B["p"].values
    ca, cb = (pa >= .5) == y, (pb >= .5) == y
    print(f"{len(idx)} test plants ({int(y.sum())} infected)")
    print(f"A: acc {ca.mean():.3f}  AUC {roc_auc_score(y, pa):.3f}")
    print(f"B: acc {cb.mean():.3f}  AUC {roc_auc_score(y, pb):.3f}")
    rng = np.random.default_rng(args.seed)
    d_acc, d_auc = [], []
    for _ in range(args.boot):
        s = rng.integers(0, len(y), len(y))
        d_acc.append(cb[s].mean() - ca[s].mean())
        if len(set(y[s])) == 2:
            d_auc.append(roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s]))
    lo, hi = np.percentile(d_acc, [2.5, 97.5]); print(f"delta acc (B-A): {cb.mean()-ca.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = np.percentile(d_auc, [2.5, 97.5]); print(f"delta AUC (B-A): {roc_auc_score(y,pb)-roc_auc_score(y,pa):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    print(f"McNemar exact: A-only-correct={b01}  B-only-correct={b10}  p={p:.4f}")
    print("A CI that includes 0 means no demonstrated difference - report that honestly.")


if __name__ == "__main__":
    main()
