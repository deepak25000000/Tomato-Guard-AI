"""GROUP A / item A1 - accuracy of the PUBLISHED checkpoint on S1 test images
WITH vs WITHOUT a near-duplicate in train, at t = 0, 2, 3, 4, 6.

  .venv/bin/python analysis/dataset1_phase2_leakage_stratified.py

Inputs (both already produced by earlier phases - nothing is typed by hand):
  * results/phase2/faithful_original_preds.csv  - per-image probs of the user's
    original checkpoint (best_mobilenetv3.pth) on the S1 test set (450 images).
  * results/phase1/test_nearest_train.csv       - per test image, the minimum dHash
    Hamming distance (over 4 flip variants) to any TRAIN image.
  * results/phase1/leakage_by_threshold.csv     - Phase 1 counts; used to CROSS-CHECK
    that our WITH-group sizes match the audited counts exactly (HARD RULE 1/4).

For each threshold: n, accuracy with bootstrap 95% CI (seeded), and a two-sided
permutation test of (acc_with - acc_without) with group sizes held fixed (seeded,
plus-one correction). Writes results/phase2/leakage_stratified_accuracy.{json,csv}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P1, P2 = RES / "phase1", RES / "phase2"
SEED = 42
B_BOOT, B_PERM = 5000, 10000
THRESHOLDS = [0, 2, 3, 4, 6]


def bootstrap_ci(correct: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(correct)
    means = np.empty(B_BOOT)
    for b in range(B_BOOT):
        means[b] = correct[rng.integers(0, n, n)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    t0 = time.time()
    preds = pd.read_csv(P2 / "faithful_original_preds.csv")
    near = pd.read_csv(P1 / "test_nearest_train.csv")
    # The file covers BOTH held-out splits (899 rows: 450 test + 449 validation);
    # keep only the test rows this analysis is about.
    if "split" in near.columns:
        near = near[near["split"].astype(str).str.lower() == "test"]
    audited = pd.read_csv(P1 / "leakage_by_threshold.csv").set_index("threshold")

    if len(preds) != 450 or len(near) != 450:
        raise SystemExit(f"FATAL: expected 450 test rows, got preds={len(preds)} "
                         f"near={len(near)} after split filter (HARD RULE 6)")
    df = preds.merge(near, left_on="path", right_on="query_path", how="inner",
                     validate="one_to_one")
    if len(df) != 450:
        raise SystemExit(f"FATAL: join only matched {len(df)}/450 rows (HARD RULE 6)")

    y = df["y_true"].to_numpy()
    pred = df["y_pred"].to_numpy()
    correct = (y == pred).astype(float)
    d = df["hamming_min_over_flips"].to_numpy()

    rows, per_t = [], {}
    rng = np.random.default_rng(SEED)
    for t in THRESHOLDS:
        with_mask = d <= t
        n_w, n_nw = int(with_mask.sum()), int((~with_mask).sum())
        # cross-check against Phase 1's audited count (HARD RULE 1)
        aud_n = int(audited.loc[t, "n_test_near_dup_in_train"])
        if n_w != aud_n:
            raise SystemExit(f"FATAL: t={t} with-group={n_w} != Phase-1 audited {aud_n}")
        cw, cnw = correct[with_mask], correct[~with_mask]
        acc_w, acc_nw = float(cw.mean()), float(cnw.mean())
        ci_w, ci_nw = bootstrap_ci(cw, rng), bootstrap_ci(cnw, rng)
        obs = acc_w - acc_nw
        # permutation: shuffle group membership, sizes fixed, two-sided
        count = 0
        for _ in range(B_PERM):
            perm = rng.permutation(len(correct))
            diff = correct[perm[:n_w]].mean() - correct[perm[n_w:]].mean()
            if abs(diff) >= abs(obs):
                count += 1
        p = (1 + count) / (1 + B_PERM)
        rec = {"threshold": t, "n_with_near_dup_in_train": n_w,
               "n_without": n_nw,
               "accuracy_with": round(acc_w, 6),
               "accuracy_with_ci95": [round(x, 6) for x in ci_w],
               "accuracy_without": round(acc_nw, 6),
               "accuracy_without_ci95": [round(x, 6) for x in ci_nw],
               "difference_with_minus_without": round(obs, 6),
               "permutation_p_two_sided": round(p, 6),
               "n_permutations": B_PERM, "n_bootstrap": B_BOOT,
               "phase1_audited_n_with": aud_n,
               "crosscheck_matches_phase1": True}
        rows.append(rec)
        per_t[str(t)] = rec
        print(f"t={t}: with n={n_w} acc={acc_w:.4f} CI[{ci_w[0]:.4f},{ci_w[1]:.4f}] | "
              f"without n={n_nw} acc={acc_nw:.4f} CI[{ci_nw[0]:.4f},{ci_nw[1]:.4f}] | "
              f"diff={obs:+.4f} perm_p={p:.4f}", flush=True)

    out = {"phase": 2, "analysis": "leakage-stratified accuracy of the published checkpoint",
           "checkpoint": "user's original best_mobilenetv3.pth (S1-trained)",
           "split": "S1 test (n=450)",
           "distance_metric": "dHash Hamming distance over 4 flips to nearest TRAIN image "
                              "(results/phase1/test_nearest_train.csv)",
           "seed": SEED, "n_bootstrap": B_BOOT, "n_permutations": B_PERM,
           "ci_method": "percentile bootstrap within each group",
           "permutation_test": "two-sided, group sizes held fixed, plus-one correction",
           "thresholds": per_t,
           "runtime_s": round(time.time() - t0, 1)}
    (P2 / "leakage_stratified_accuracy.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(P2 / "leakage_stratified_accuracy.csv", index=False)
    print(f"wrote {P2/'leakage_stratified_accuracy.json'} and .csv "
          f"({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
