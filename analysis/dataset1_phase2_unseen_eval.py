"""PHASE 2 (diagnostic) - evaluate the user's ORIGINAL S1-trained checkpoint on the subset of the
S2 test set that S1 training and validation NEVER saw.

Why this exists: results/phase2/s2_origckpt.json (0.7313-style numbers) is CONTAMINATED - 72.2% of
the S2 test images were in S1's training set, because S1 and S2 are two different partitions of the
same 2995 images. Quoting it as generalisation would be wrong (HARD RULE 8: no unearned claims).

This script computes the honest, tiny alternative:
    n = |S2_test \\ (S1_train U S1_validation)|
and reports accuracy on exactly those images, with an explicit wide-CI caveat.

  .venv/bin/python analysis/dataset1_phase2_unseen_eval.py

Writes results/phase2/faithful_origckpt_on_unseen.json + _preds.csv + overlap_accounting.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
sys.path.insert(0, str(ROOT / "analysis"))

from trackA_lib import evaluate  # noqa: E402

ORIG_CKPT = Path.home() / "Downloads" / "Tomato_Research_Results" / "best_mobilenetv3.pth"
ARCH = "mobilenet_v3_small"


def main() -> None:
    t0 = time.time()
    s1 = {k: set(pd.read_csv(RES / "splits_S1" / f"{k}.csv").path)
          for k in ("train", "validation", "test")}
    s2 = {k: set(pd.read_csv(RES / "splits_S2" / f"{k}.csv").path)
          for k in ("train", "validation", "test")}
    seen = s1["train"] | s1["validation"]
    s2test = pd.read_csv(RES / "splits_S2" / "test.csv")
    unseen_mask = ~s2test.path.isin(seen)
    unseen = s2test[unseen_mask].copy()

    accounting = {
        "s2_test_total": int(len(s2test)),
        "s2_test_seen_by_s1_train_or_val": int((~unseen_mask).sum()),
        "s2_test_seen_by_s1_train": int(len(s2["test"] & s1["train"])),
        "s2_test_seen_by_s1_validation": int(len(s2["test"] & s1["validation"])),
        "s2_test_unseen_by_s1": int(len(unseen)),
        "pct_s2_test_contaminated": round(100 * (~unseen_mask).sum() / len(s2test), 2),
        "s1_test_in_s2_train": int(len(s1["test"] & s2["train"])),
        "s1_test_in_s2_train_or_val": int(len(s1["test"] & (s2["train"] | s2["validation"]))),
        "interpretation": (
            "S1 and S2 are overlapping partitions of the SAME 2995 images. A model trained on one "
            "split's train set has already seen most of the other split's test set, so cross-split "
            "evaluation is invalid. Only the n 'unseen' images below are a (small, wide-CI) "
            "honest probe of the original checkpoint's generalisation to a group-disjoint test."),
    }
    (P2 / "overlap_accounting.json").write_text(json.dumps(accounting, indent=2))
    print(json.dumps(accounting, indent=2))

    if len(unseen) == 0:
        raise SystemExit("no unseen images - nothing to evaluate")

    # materialise the filtered split so trackA_lib.evaluate can read it
    split_dir = P2 / "splits_S2_unseen"
    split_dir.mkdir(exist_ok=True)
    unseen[["path", "class"]].to_csv(split_dir / "test.csv", index=False)
    classes = json.loads((RES / "splits_S2" / "classes.json").read_text())
    (split_dir / "classes.json").write_text(json.dumps(classes, indent=2))

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    sd = torch.load(ORIG_CKPT, map_location="cpu", weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit(f"FATAL: checkpoint mismatch {miss} {unexp}")

    m = evaluate(model, classes, split_dir, torch.device("cpu"),
                 "origckpt_on_unseen", P2,
                 extra={"note": "user's original S1-trained checkpoint, evaluated ONLY on the "
                                "S2 test images that S1 never trained/validated on",
                        "n_unseen": int(len(unseen)),
                        "caveat": "small n -> wide bootstrap CI; do not quote without the n",
                        "split": "S2_test_unseen_by_S1",
                        "runtime_s": round(time.time() - t0, 1)})
    (P2 / "faithful_origckpt_on_unseen.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("per_class_report", "confusion_matrix", "specificity")},
                     indent=2))


if __name__ == "__main__":
    main()
