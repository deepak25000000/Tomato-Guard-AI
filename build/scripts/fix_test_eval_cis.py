"""Correct the bootstrap CI quantiles in build/test_eval.json (bug fix, 2026-11-23).

Bug: build/scripts/eval_test_once.py computed percentile endpoints as
[(1-alpha)/2, 1-(1-alpha)/2] = [0.475, 0.525] instead of [alpha/2, 1-alpha/2]
= [0.025, 0.975], so every "95% CI" in build/test_eval.json was a ~5% interval
far narrower than the true uncertainty. Point estimates were unaffected
(verified below against the saved predictions).

This script:
  * reads build/test_eval.json and the SAVED test predictions
    results/phase2/arch3_mobilenet_v3_large_seed42_preds.csv (no test image is
    read, no model inference runs, no threshold/selection is changed - the
    frozen model and threshold stay exactly as frozen),
  * asserts every point estimate in test_eval.json still matches the saved
    predictions exactly,
  * recomputes the four bootstrap CIs with the FIXED quantiles, same draws
    (5000) and seeds (42/43/44/45) as the original one-shot evaluation,
  * rewrites build/test_eval.json in place with a ci_correction audit block.

HARD RULE 2 (test set touched once) is preserved: this is a statistic-only
recomputation from an existing prediction file, not a new evaluation.

  .venv/bin/python build/scripts/fix_test_eval_cis.py

Idempotent: skips if the ci_correction marker is already present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TEST_EVAL = ROOT / "build" / "test_eval.json"
PREDS = ROOT / "results" / "phase2" / "arch3_mobilenet_v3_large_seed42_preds.csv"
CFG = json.loads((ROOT / "build" / "config.json").read_text())


def boot_ci(values_fn, n: int, draws: int, seed: int, alpha: float) -> list[float]:
    """Identical procedure to eval_test_once.boot_ci AFTER the quantile fix."""
    rng = np.random.default_rng(seed)
    stats = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        stats[i] = values_fn(idx)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def main() -> int:
    rep = json.loads(TEST_EVAL.read_text())
    if "ci_correction" in rep:
        print("[skip] ci_correction already applied")
        return 0

    df = pd.read_csv(PREDS)
    n = int(rep["n_test"])
    threshold = float(rep["threshold"])
    draws = int(CFG["bootstrap"]["n_draws"])
    bseed = int(CFG["bootstrap"]["seed"])
    alpha = 1 - float(CFG["bootstrap"]["ci"])

    # ---- verify the saved predictions ARE the ones test_eval.json reports ----
    y = df["y_true"].to_numpy()
    pred = df["y_pred"].to_numpy()
    conf = df["confidence"].to_numpy()
    probs = df[[f"prob_{i}" for i in range(7)]].to_numpy()
    assert len(df) == n, f"n mismatch: preds {len(df)} vs test_eval {n}"
    acc = float((pred == y).mean())
    assert acc == rep["accuracy"], f"accuracy mismatch {acc} vs {rep['accuracy']}"
    covered = conf >= threshold
    k = int(covered.sum())
    assert k == rep["n_covered"], f"n_covered mismatch {k} vs {rep['n_covered']}"
    assert int(n - k) == rep["n_uncertain"], "n_uncertain mismatch"
    cov = float(k / n)
    assert cov == rep["coverage"], f"coverage mismatch {cov} vs {rep['coverage']}"
    sel = float((pred[covered] == y[covered]).mean())
    assert sel == rep["selective_accuracy"], f"selective mismatch {sel}"
    yb = np.eye(7)[y]
    auc = float(roc_auc_score(yb, probs, average="macro"))
    assert abs(auc - rep["macro_auc"]) < 1e-12, f"macro_auc mismatch {auc}"
    print(f"[ok] point estimates verified against saved predictions (n={n})")

    # ---- recompute ONLY the CIs with the fixed quantiles ---------------------
    t0 = time.time()
    acc_ci = boot_ci(lambda i: (pred[i] == y[i]).mean(), n, draws, bseed, alpha)
    auc_ci = boot_ci(lambda i: roc_auc_score(yb[i], probs[i], average="macro"),
                     n, draws, bseed + 1, alpha)
    cov_ci = boot_ci(lambda i: covered[i].mean(), n, draws, bseed + 2, alpha)
    rng = np.random.default_rng(bseed + 3)
    pc, yc = pred[covered], y[covered]
    k2 = int(covered.sum())
    sel_stats = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k2, k2)
        sel_stats[i] = (pc[idx] == yc[idx]).mean()
    sel_ci = [float(x) for x in np.quantile(sel_stats, [alpha / 2, 1 - alpha / 2])]

    old = {
        "accuracy_95_ci": rep["accuracy_95_ci"],
        "macro_auc_95_ci": rep["macro_auc_95_ci"],
        "coverage_95_ci": rep["coverage_95_ci"],
        "selective_accuracy_95_ci": rep["selective_accuracy_95_ci"],
    }
    rep["accuracy_95_ci"] = acc_ci
    rep["macro_auc_95_ci"] = auc_ci
    rep["coverage_95_ci"] = cov_ci
    rep["selective_accuracy_95_ci"] = sel_ci
    rep["ci_correction"] = {
        "date": "2026-11-23",
        "bug": "eval_test_once.py used quantiles [(1-alpha)/2, 1-(1-alpha)/2] "
               "= [0.475, 0.525] (a ~5% interval) instead of [alpha/2, 1-alpha/2] "
               "= [0.025, 0.975]; fixed in build/scripts/eval_test_once.py",
        "method": "bootstrap recomputed from saved predictions "
                  "results/phase2/arch3_mobilenet_v3_large_seed42_preds.csv "
                  "(same draws/seed as the one-shot evaluation; no test image read, "
                  "no re-inference, point estimates unchanged - verified exact)",
        "previous_ci_too_narrow": old,
        "runtime_s": round(time.time() - t0, 1),
    }
    TEST_EVAL.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in
                      ["accuracy_95_ci", "macro_auc_95_ci", "coverage_95_ci",
                       "selective_accuracy_95_ci", "ci_correction"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
