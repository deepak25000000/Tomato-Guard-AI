"""Phase 1.2b - ONE-SHOT evaluation of the frozen leaf model on the S2 TEST split.

  .venv/bin/python build/scripts/eval_test_once.py

Touches the test split exactly once (HARD RULE 2), after the threshold has been chosen
on validation (build/threshold.json must exist). Reports overall accuracy/AUC,
coverage and selective accuracy at the frozen threshold, all with percentile bootstrap
CIs (n_draws/seed/ci from build/config.json). Output: build/test_eval.json.
Restart-proof: skips if build/test_eval.json exists (the one touch is enforced).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import LeafDataset, EVAL_TF, predict  # noqa: E402
from models import build_model  # noqa: E402

CKPT = ROOT / "build" / "checkpoint" / "mobilenet_v3_large_seed42.pt"
S2 = ROOT / "results" / "splits_S2"
OUT = ROOT / "build" / "test_eval.json"
CFG = json.loads((ROOT / "build" / "config.json").read_text())


def boot_ci(values_fn, n: int, draws: int, seed: int, alpha: float) -> list[float]:
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        stats[i] = values_fn(idx)
    # FIXED 2026-11-23: was [(1-alpha)/2, 1-(1-alpha)/2] = [0.475, 0.525] (a ~5%
    # interval, far too narrow). Percentile CI endpoints are [alpha/2, 1-alpha/2].
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def main() -> int:
    if OUT.exists():
        print(f"[skip] {OUT} exists (test already touched once)")
        return 0
    thr = json.loads((ROOT / "build" / "threshold.json").read_text())
    if not thr.get("rule_met"):
        print("FATAL: threshold rule not met on validation - spec says stop", file=sys.stderr)
        return 2
    t0 = time.time()
    threshold = float(thr["threshold"])
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    model = build_model(ck["arch"], len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    df = pd.read_csv(S2 / "test.csv")
    loader = torch.utils.data.DataLoader(
        LeafDataset(df, c2i, EVAL_TF), batch_size=16, shuffle=False, num_workers=0)
    lg, y = predict(model, loader, torch.device("cpu"))
    z = lg if isinstance(lg, np.ndarray) else np.asarray(lg)
    e = np.exp(z - z.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    pred = p.argmax(1)
    conf = p.max(1)
    n = len(y)
    draws = int(CFG["bootstrap"]["n_draws"])
    bseed = int(CFG["bootstrap"]["seed"])
    alpha = 1 - float(CFG["bootstrap"]["ci"])

    acc = float((pred == y).mean())
    acc_ci = boot_ci(lambda i: (pred[i] == y[i]).mean(), n, draws, bseed, alpha)
    try:
        yb = np.eye(len(classes))[y]
        auc = float(roc_auc_score(yb, p, average="macro"))
        auc_ci = boot_ci(lambda i: roc_auc_score(yb[i], p[i], average="macro"),
                         n, draws, bseed + 1, alpha)
    except ValueError:
        auc, auc_ci = None, None

    covered = conf >= threshold
    k = int(covered.sum())
    cov = float(k / n)
    sel = float((pred[covered] == y[covered]).mean()) if k else None
    cov_ci = boot_ci(lambda i: covered[i].mean(), n, draws, bseed + 2, alpha)
    if k:
        # bootstrap of selective accuracy over the covered subset (resample within subset)
        rng = np.random.default_rng(bseed + 3)
        pc, yc = pred[covered], y[covered]
        sel_stats = np.empty(draws)
        for i in range(draws):
            idx = rng.integers(0, k, k)
            sel_stats[i] = (pc[idx] == yc[idx]).mean()
        # FIXED 2026-11-23: same quantile bug as boot_ci above - [alpha/2, 1-alpha/2].
        sel_ci = [float(x) for x in np.quantile(sel_stats, [alpha / 2, 1 - alpha / 2])]
    else:
        sel_ci = None

    report = {
        "split": "S2 test (touched once, after validation-only threshold selection)",
        "n_test": int(n),
        "classes": classes,
        "threshold": threshold,
        "threshold_source": "build/threshold.json (S2 validation risk-coverage rule)",
        "accuracy": acc,
        "accuracy_95_ci": acc_ci,
        "macro_auc": auc,
        "macro_auc_95_ci": auc_ci,
        "coverage": cov,
        "coverage_95_ci": cov_ci,
        "n_covered": k,
        "n_uncertain": int(n - k),
        "selective_accuracy": sel,
        "selective_accuracy_95_ci": sel_ci,
        "model": ck["arch"],
        "seed": ck["seed"],
        "best_epoch": ck["best_epoch"],
        "best_val_accuracy": ck["best_val_accuracy"],
        "bootstrap": CFG["bootstrap"],
        "runtime_s": round(time.time() - t0, 1),
        "note": "raw softmax = model score; uncalibrated (temperature scaling worsened ECE)",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
