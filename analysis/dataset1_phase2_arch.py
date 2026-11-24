"""PHASE 2c - Track A ARCHITECTURE SWEEP on the leakage-safe split S2.

  .venv/bin/python analysis/dataset1_phase2_arch.py

HARD RULE 3 COMPLIANCE (this is the important part):
  * every candidate is trained on S2 train and scored on S2 VALIDATION only;
  * the S2 TEST set is touched exactly ONCE, for the validation-selected winner;
  * therefore no test number for an unselected architecture is ever produced.

Candidate architectures are the ones the user's paper/README mentions, all built by the
project's own tomatoguard/research/models.py (ImageNet-pretrained, same recipe as the
notebook: 10 epochs, batch 16, weighted CE, AdamW 1e-4, best epoch by validation accuracy).

Selection rule (documented deviation: the notebook picks the best EPOCH by validation
accuracy; across architectures we pick by validation MACRO-F1 because the 7 classes are
imbalanced 112-884 - both metrics are recorded).

Writes results/phase2/arch_<name>_val.json for each candidate, arch_selection.json,
arch_winner_test.json, arch_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import evaluate, seed_all, train_one  # noqa: E402
from models import build_model  # noqa: E402

S2 = RES / "splits_S2"
# Exactly the three architectures the project's own tomatoguard/research/models.py ARCHS dict
# supports (and the ones the user's protocol names: v3_small baseline, v3_large, effnet_b0).
# Adding resnet/convnext would require editing an existing script - not done (HARD RULE 7).
CANDIDATES = ["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"]
SEED = 42
ROUGE = ("per_class_report", "confusion_matrix", "specificity", "class_support")


def slim(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ROUGE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    classes = json.loads((S2 / "classes.json").read_text())
    dev = torch.device("cpu")
    results, best = [], None

    for arch in a.candidates:
        print(f"\n=== candidate {arch} (S2, seed {a.seed}) ===", flush=True)
        t0 = time.time()
        seed_all(a.seed)
        model = build_model(arch, len(classes), pretrained=True).to(dev)
        out = train_one(model, S2 / "train.csv", S2 / "validation.csv", classes, dev,
                        epochs=a.epochs, seed=a.seed, log_prefix=f"[{arch}] ")
        # score on VALIDATION only - the test set is not read here (HARD RULE 3)
        m = evaluate(out["model"], classes, S2, dev, f"arch_{arch}_val", P2,
                     extra={"arch": arch, "split": "S2", "split_scored": "validation",
                            "seed": a.seed, "epochs": a.epochs,
                            "best_val_accuracy": out["best_val_accuracy"],
                            "best_epoch": out["best_epoch"],
                            "runtime_s": round(time.time() - t0, 1)},
                     split_name="validation")
        (P2 / f"arch_{arch}_val.json").write_text(json.dumps(m, indent=2))
        pd.DataFrame(out["history"]).to_csv(P2 / f"arch_{arch}_history.csv", index=False)
        results.append({"arch": arch, "val_accuracy": m["accuracy"],
                        "val_accuracy_ci": m["accuracy_95_ci"], "val_macro_f1": m["macro_f1"],
                        "val_macro_f1_ci": m["macro_f1_95_ci"], "val_macro_roc_auc": m["macro_roc_auc"],
                        "val_macro_pr_auc": m["macro_pr_auc"],
                        "best_val_accuracy_in_training": out["best_val_accuracy"],
                        "best_epoch": out["best_epoch"], "runtime_s": m["runtime_s"],
                        "n_parameters": sum(p.numel() for p in out["model"].parameters())})
        print(f"  -> {arch}: val_acc {m['accuracy']:.4f}  val_macro_f1 {m['macro_f1']:.4f}  "
              f"({m['runtime_s']}s)", flush=True)

        # keep only the current best model's weights (RAM is tight: 7 GB total)
        key = (m["macro_f1"], m["accuracy"])
        if best is None or key > best["key"]:
            best = {"key": key, "arch": arch,
                    "sd": {k: v.detach().clone() for k, v in out["model"].state_dict().items()}}

    sel = {"selection_criterion": "highest S2 VALIDATION macro-F1 (ties broken by validation "
                                 "accuracy); the S2 test set is not consulted",
           "candidates": results, "selected_architecture": best["arch"],
           "n_candidates": len(results), "seed": a.seed, "epochs": a.epochs}
    (P2 / "arch_selection.json").write_text(json.dumps(sel, indent=2))
    pd.DataFrame([{k: v for k, v in r.items() if k != "val_accuracy_ci" and k != "val_macro_f1_ci"}
                  for r in results]).to_csv(P2 / "arch_summary.csv", index=False)
    print("\n=== selection ===")
    print(json.dumps(sel, indent=2))

    # ---- the ONE test-set evaluation for the sweep -----------------------------
    print(f"\n=== winner {best['arch']} -> S2 test (single test evaluation) ===", flush=True)
    seed_all(a.seed)
    wmodel = build_model(best["arch"], len(classes), pretrained=False).to(dev)
    wmodel.load_state_dict(best["sd"])
    m = evaluate(wmodel, classes, S2, dev, "arch_winner_test", P2,
                 extra={"arch": best["arch"], "split": "S2", "split_scored": "test",
                        "seed": a.seed, "epochs": a.epochs,
                        "selection_criterion": sel["selection_criterion"],
                        "note": "single test evaluation of the validation-selected winner"})
    (P2 / "arch_winner_test.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(slim(m), indent=2))


if __name__ == "__main__":
    main()
