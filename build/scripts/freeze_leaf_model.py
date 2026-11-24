"""Phase 1.1 - freeze the leaf model: retrain mobilenet_v3_large seed 42 on S2 WITH
checkpoint saving (the archived arch3 script saves none).

  .venv/bin/python build/scripts/freeze_leaf_model.py

Identical recipe to the archived run (same seed_all / train_one / build_model calls,
epochs=10, batch 16, AdamW 1e-4, ImageNet-pretrained backbone, best epoch by
VALIDATION accuracy). Writes ONLY under build/ (HARD RULE 3): never touches results/.
Restart-proof: skips completely if the checkpoint already exists.

Cross-check (HARD RULE 1 / risk R1): regenerated validation probabilities are joined
against the archived read-only results/phase2/arch3_mobilenet_v3_large_seed42_val_preds.csv
and max|diff| is recorded in build/freeze_report.json. Test split is NOT touched here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import evaluate, seed_all, train_one  # noqa: E402
from models import build_model  # noqa: E402

S2 = ROOT / "results" / "splits_S2"
ARCHIVED_VAL = ROOT / "results" / "phase2" / "arch3_mobilenet_v3_large_seed42_val_preds.csv"
CKPT = ROOT / "build" / "checkpoint" / "mobilenet_v3_large_seed42.pt"
OUT_DIR = ROOT / "build" / "freeze"
SEED = 42
ARCH = "mobilenet_v3_large"


def main() -> None:
    t0 = time.time()
    if CKPT.exists():
        rep = json.loads((ROOT / "build" / "freeze_report.json").read_text())
        print(f"[skip] checkpoint exists ({CKPT}); report: "
              f"val_acc={rep['best_val_accuracy']} max|diff|={rep['archived_val_max_abs_diff']}",
              flush=True)
        return
    classes = json.loads((S2 / "classes.json").read_text())
    dev = torch.device("cpu")
    seed_all(SEED)
    model = build_model(ARCH, len(classes), pretrained=True).to(dev)
    out = train_one(model, S2 / "train.csv", S2 / "validation.csv", classes, dev,
                    epochs=10, seed=SEED, log_prefix=f"[freeze:{SEED}] ")
    # validation metrics + regenerated probabilities (build/ only, new tag)
    val = evaluate(out["model"], classes, S2, dev, "regen_val", OUT_DIR,
                   split_name="validation",
                   extra={"arch": ARCH, "split": "S2", "seed": SEED,
                          "split_scored": "validation"})
    # save the frozen checkpoint (best epoch by validation accuracy)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": out["model"].state_dict(),
        "arch": ARCH, "seed": SEED, "classes": classes, "epochs": 10,
        "best_val_accuracy": out["best_val_accuracy"], "best_epoch": out["best_epoch"],
        "torch": torch.__version__,
        "recipe": "trackA_lib.train_one defaults (batch 16, AdamW lr=1e-4 wd=1e-4, "
                  "ReduceLROnPlateau, TRAIN_TF/EVAL_TF at 224px)",
    }, CKPT)
    # cross-check against the archived validation probabilities (read-only)
    new = pd.read_csv(OUT_DIR / "regen_val_preds.csv")
    old = pd.read_csv(ARCHIVED_VAL)
    merged = old.merge(new, on="path", suffixes=("_old", "_new"))
    prob_cols_old = [c for c in old.columns if c.startswith("prob_")]
    max_diff = float(max(np.abs(merged[f"{c}_old"] - merged[f"{c}_new"]).max()
                         for c in prob_cols_old))
    same_pred = bool((merged["y_pred_old"] == merged["y_pred_new"]).all())
    n_matched = int(len(merged))
    report = {
        "arch": ARCH, "seed": SEED, "split": "S2", "epochs": 10,
        "best_val_accuracy": out["best_val_accuracy"],
        "best_epoch": out["best_epoch"],
        "val_accuracy_from_probs": val["accuracy"],
        "val_n": val["n_test"],
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "archived_val_source": str(ARCHIVED_VAL.relative_to(ROOT)),
        "archived_val_rows_joined": n_matched,
        "archived_val_max_abs_diff": max_diff,
        "archived_val_all_preds_equal": same_pred,
        "runtime_s": round(time.time() - t0, 1),
        "torch": torch.__version__,
    }
    (ROOT / "build" / "freeze_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
