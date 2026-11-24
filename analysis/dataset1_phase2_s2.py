"""PHASE 2c/2d (part 1) - Track A experiments on the LEAKAGE-SAFE split S2.

  .venv/bin/python analysis/dataset1_phase2_s2.py --seeds 42 43 44

Trains MobileNetV3-Small on results/splits_S2 for each seed (the user's notebook recipe:
10 epochs, batch 16, weighted CE, AdamW 1e-4, best model by VALIDATION accuracy) and
evaluates each run ONCE on the S2 test set.

Also evaluates the user's ORIGINAL S1-trained checkpoint on the S2 test set, to measure how
much of the published 0.6733 survives a group-disjoint test set.

HARD RULE 3: model selection (best epoch) uses validation only; the test set is touched once
per finished model. HARD RULE 4: the S2 split already passed group/sha256 disjointness in
dataset1_phase2_prepare_splits.py.

Writes results/phase2/s2_seed<seed>.json (+ _preds.csv), s2_origckpt.json, s2_seed_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import evaluate, seed_all, train_one  # noqa: E402
from models import build_model  # noqa: E402

ORIG_CKPT = Path.home() / "Downloads" / "Tomato_Research_Results" / "best_mobilenetv3.pth"
S2 = RES / "splits_S2"
ARCH = "mobilenet_v3_small"


def load_classes() -> list[str]:
    return json.loads((S2 / "classes.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--skip-origckpt", action="store_true")
    a = ap.parse_args()
    classes = load_classes()
    dev = torch.device("cpu")
    rows = []

    if not a.skip_origckpt:
        print("=== evaluating the user's original S1-trained checkpoint on the S2 test set ===",
              flush=True)
        m0 = build_model(ARCH, len(classes), pretrained=False)
        sd = torch.load(ORIG_CKPT, map_location="cpu", weights_only=True)
        miss, unexp = m0.load_state_dict(sd, strict=False)
        if miss or unexp:
            raise SystemExit(f"FATAL: checkpoint mismatch missing={miss} unexpected={unexp}")
        m = evaluate(m0, classes, S2, dev, "s2_origckpt", P2,
                     extra={"arch": ARCH, "split": "S2", "seed": None,
                            "trained_on_split": "S1", "note": "user's original checkpoint"})
        (P2 / "s2_origckpt.json").write_text(json.dumps(m, indent=2))
        print(json.dumps({k: v for k, v in m.items()
                          if k not in ("per_class_report", "confusion_matrix",
                                       "specificity")}, indent=2), flush=True)

    for seed in a.seeds:
        print(f"\n=== S2 run: arch={ARCH} seed={seed} epochs={a.epochs} ===", flush=True)
        t0 = time.time()
        seed_all(seed)
        model = build_model(ARCH, len(classes), pretrained=True).to(dev)
        out = train_one(model, S2 / "train.csv", S2 / "validation.csv", classes, dev,
                        epochs=a.epochs, seed=seed, log_prefix=f"[s2:{seed}] ")
        m = evaluate(out["model"], classes, S2, dev, f"s2_seed{seed}", P2,
                     extra={"arch": ARCH, "split": "S2", "seed": seed,
                            "best_val_accuracy": out["best_val_accuracy"],
                            "best_epoch": out["best_epoch"],
                            "epochs": a.epochs, "runtime_s": round(time.time() - t0, 1),
                            "torch": torch.__version__})
        (P2 / f"s2_seed{seed}.json").write_text(json.dumps(m, indent=2))
        pd.DataFrame(out["history"]).to_csv(P2 / f"s2_seed{seed}_history.csv", index=False)
        rows.append({"seed": seed, "arch": ARCH, "accuracy": m["accuracy"],
                     "accuracy_ci_lo": m["accuracy_95_ci"][0],
                     "accuracy_ci_hi": m["accuracy_95_ci"][1],
                     "macro_f1": m["macro_f1"], "macro_f1_ci_lo": m["macro_f1_95_ci"][0],
                     "macro_f1_ci_hi": m["macro_f1_95_ci"][1],
                     "best_val_accuracy": out["best_val_accuracy"],
                     "best_epoch": out["best_epoch"], "runtime_s": m["runtime_s"]})
        print(json.dumps({k: v for k, v in m.items()
                          if k not in ("per_class_report", "confusion_matrix",
                                       "specificity")}, indent=2), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(P2 / "s2_seed_summary.csv", index=False)
    if len(df) > 1:
        agg = {"n_seeds": len(df),
               "accuracy_mean": float(df.accuracy.mean()),
               "accuracy_std": float(df.accuracy.std(ddof=1)) if len(df) > 1 else None,
               "accuracy_min": float(df.accuracy.min()), "accuracy_max": float(df.accuracy.max()),
               "macro_f1_mean": float(df.macro_f1.mean()),
               "macro_f1_std": float(df.macro_f1.std(ddof=1)) if len(df) > 1 else None,
               "per_seed": df.to_dict(orient="records")}
        (P2 / "s2_seed_aggregate.json").write_text(json.dumps(agg, indent=2))
        print("\n=== S2 seed aggregate ===")
        print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
