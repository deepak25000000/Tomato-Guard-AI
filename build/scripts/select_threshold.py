"""Phase 1.2a - uncertain threshold from the RISK-COVERAGE curve on S2 VALIDATION only.

  .venv/bin/python build/scripts/select_threshold.py

Pre-registered rule (build/config.json, written BEFORE this script ran):
  choose the LOWEST confidence threshold whose validation selective accuracy >= 0.80.

Inputs (read-only): build/freeze/regen_val_preds.csv (probabilities of the FROZEN
model on S2 validation - the model that will be exported). Never reads the test split
(HARD RULE 2). Output: build/threshold.json (new file).
If the rule cannot be met at any threshold, the report says so and exits 2 (spec stop).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "build" / "freeze" / "regen_val_preds.csv"
CFG = json.loads((ROOT / "build" / "config.json").read_text())
OUT = ROOT / "build" / "threshold.json"
MIN_SEL_ACC = float(CFG["uncertain_threshold_rule"]["min_selective_accuracy"])


def main() -> int:
    if not VAL.exists():
        print(f"FATAL: {VAL} missing - run freeze_leaf_model.py first", file=sys.stderr)
        return 3
    df = pd.read_csv(VAL)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    p = df[prob_cols].values
    conf = p.max(1)
    pred = p.argmax(1)
    y = df["y_true"].values
    n = len(y)

    # risk-coverage: sort by confidence descending; threshold = confidence of the
    # k-th covered sample; coverage = k/n; selective accuracy = correct among top k.
    order = np.argsort(-conf, kind="stable")
    conf_s = conf[order]
    correct_s = (pred[order] == y[order]).astype(np.int64)
    cum_correct = np.cumsum(correct_s)
    ks = np.arange(1, n + 1)
    sel_acc = cum_correct / ks
    coverage = ks / n

    # lowest threshold with selective accuracy >= MIN: the largest k that satisfies
    # it when scanning thresholds ascending = smallest confidence among qualifying k.
    qualifying = np.where(sel_acc >= MIN_SEL_ACC)[0]  # indices into ks
    if len(qualifying) == 0:
        report = {
            "rule": CFG["uncertain_threshold_rule"]["description"],
            "min_selective_accuracy": MIN_SEL_ACC,
            "split": "S2 validation",
            "n": int(n),
            "rule_met": False,
            "max_validation_selective_accuracy": float(sel_acc.max()),
            "max_selective_accuracy_coverage": float(coverage[int(sel_acc.argmax())]),
            "note": "rule cannot be met on validation - per spec, stop and report",
        }
        OUT.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2
    # among qualifying k, choose the one with the LOWEST threshold = highest coverage
    k_best = int(qualifying[-1]) + 1  # 1-based
    threshold = float(conf_s[k_best - 1])
    # ties: include all samples with conf >= threshold (>= keeps k_best consistent
    # only if no ties below; recompute exactly with >=)
    covered = conf >= threshold
    k_exact = int(covered.sum())
    sel_exact = float((pred[covered] == y[covered]).mean())
    cov_exact = float(k_exact / n)

    # per-class coverage sanity (EB/LMF confusion is the known weak spot)
    report = {
        "rule": CFG["uncertain_threshold_rule"]["description"],
        "min_selective_accuracy": MIN_SEL_ACC,
        "split": "S2 validation (selection data only)",
        "n": int(n),
        "rule_met": True,
        "threshold": threshold,
        "confidence_definition": CFG["uncertain_threshold_rule"]["confidence_definition"],
        "validation_coverage": cov_exact,
        "validation_selective_accuracy": sel_exact,
        "n_covered": k_exact,
        "n_uncertain": int(n - k_exact),
        "validation_overall_accuracy": float((pred == y).mean()),
        "curve_points": int(n),
        "source_file": str(VAL.relative_to(ROOT)),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
