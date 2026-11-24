"""GROUP A / item A4 - mobilenet_v3_large and efficientnet_b0, 3 seeds each, on S2.

  .venv/bin/python analysis/dataset1_phase2_arch_multiseed.py

Recipe identical to the audited S2 runs (trackA_lib: 10 epochs, batch 16, AdamW 1e-4,
ImageNet-pretrained backbone, best epoch by VALIDATION accuracy). For each run:
  * full metrics on VALIDATION (selection data, HARD RULE 3)
  * full metrics on TEST exactly once, for the finished model (HARD RULE 3)

After all 6 runs, selection across ALL three candidates (small's numbers come from the
existing s2_seed{42,43,44}.json - never re-scored) uses ONLY validation data:
  criterion = mean best-validation accuracy over the 3 seeds (pre-declared here and in
  run_log.md); test numbers are reported descriptively and never used to select.
Writes results/phase2/arch3_{arch}_seed{seed}.json, arch3_summary.csv,
arch3_selection.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import evaluate, seed_all, train_one  # noqa: E402
from models import build_model  # noqa: E402

S2 = RES / "splits_S2"
ARCHS = ["mobilenet_v3_large", "efficientnet_b0"]
SEEDS = [42, 43, 44]
CRITERION = ("mean best-validation accuracy over 3 seeds; validation only - "
             "test metrics are reported but never used for selection (HARD RULE 3)")


def main() -> None:
    t0 = time.time()
    classes = json.loads((S2 / "classes.json").read_text())
    dev = torch.device("cpu")
    rows = []
    for arch in ARCHS:
        for seed in SEEDS:
            print(f"\n=== arch3: {arch} seed={seed} ===", flush=True)
            jpath = P2 / f"arch3_{arch}_seed{seed}.json"
            if jpath.exists():     # resume after host restart (CHANGELOG): run finished
                j = json.loads(jpath.read_text())
                rec = {k: j[k] for k in ("arch", "seed", "epochs", "best_val_accuracy",
                                         "best_epoch", "val_macro_f1", "test_accuracy",
                                         "test_macro_f1", "n_test", "runtime_s")}
                rows.append(rec)
                print(f"[resume] {arch} seed {seed} already complete - skipped",
                      flush=True)
                continue
            t1 = time.time()
            seed_all(seed)
            model = build_model(arch, len(classes), pretrained=True).to(dev)
            out = train_one(model, S2 / "train.csv", S2 / "validation.csv", classes, dev,
                            epochs=10, seed=seed, log_prefix=f"[{arch}:{seed}] ")
            val = evaluate(out["model"], classes, S2, dev,
                           f"arch3_{arch}_seed{seed}_val", P2,
                           split_name="validation",
                           extra={"arch": arch, "split": "S2", "seed": seed,
                                  "split_scored": "validation"})
            test = evaluate(out["model"], classes, S2, dev,
                            f"arch3_{arch}_seed{seed}", P2,
                            extra={"arch": arch, "split": "S2", "seed": seed,
                                   "split_scored": "test (finished model, one touch)",
                                   "best_val_accuracy": out["best_val_accuracy"],
                                   "best_epoch": out["best_epoch"], "epochs": 10,
                                   "runtime_s": round(time.time() - t1, 1),
                                   "torch": torch.__version__})
            rec = {"arch": arch, "seed": seed, "epochs": 10,
                   "best_val_accuracy": out["best_val_accuracy"],
                   "best_epoch": out["best_epoch"],
                   "val_macro_f1": val["macro_f1"],
                   "test_accuracy": test["accuracy"],
                   "test_macro_f1": test["macro_f1"],
                   "n_test": test["n_test"],
                   "runtime_s": test["runtime_s"]}
            (P2 / f"arch3_{arch}_seed{seed}.json").write_text(
                json.dumps({"validation": val, "test": test, **rec}, indent=2))
            pd.DataFrame(out["history"]).to_csv(
                P2 / f"arch3_{arch}_seed{seed}_history.csv", index=False)
            rows.append(rec)
            print(json.dumps(rec, indent=2), flush=True)

    # ---------------- selection on validation only ------------------------------
    small = [json.loads((P2 / f"s2_seed{s}.json").read_text()) for s in SEEDS]
    cand = {}
    cand["mobilenet_v3_small"] = {
        "source": "existing results/phase2/s2_seed{42,43,44}.json (never re-scored)",
        "best_val_accuracy_per_seed": [d["best_val_accuracy"] for d in small],
        "test_accuracy_per_seed": [d["accuracy"] for d in small],
        "test_macro_f1_per_seed": [d["macro_f1"] for d in small],
        "val_macro_f1_per_seed": None}
    for arch in ARCHS:
        js = [json.loads((P2 / f"arch3_{arch}_seed{s}.json").read_text()) for s in SEEDS]
        cand[arch] = {
            "source": f"results/phase2/arch3_{arch}_seed{{42,43,44}}.json",
            "best_val_accuracy_per_seed": [j["best_val_accuracy"] for j in js],
            "test_accuracy_per_seed": [j["test_accuracy"] for j in js],
            "test_macro_f1_per_seed": [j["test_macro_f1"] for j in js],
            "val_macro_f1_per_seed": [j["val_macro_f1"] for j in js]}
    summ = {}
    for arch, c in cand.items():
        va, ta, tf = c["best_val_accuracy_per_seed"], c["test_accuracy_per_seed"], \
            c["test_macro_f1_per_seed"]
        summ[arch] = {
            "val_acc_mean": round(float(pd.Series(va).mean()), 6),
            "val_acc_sd": round(float(pd.Series(va).std(ddof=1)), 6),
            "test_acc_mean": round(float(pd.Series(ta).mean()), 6),
            "test_acc_sd": round(float(pd.Series(ta).std(ddof=1)), 6),
            "test_macro_f1_mean": round(float(pd.Series(tf).mean()), 6),
            "test_macro_f1_sd": round(float(pd.Series(tf).std(ddof=1)), 6),
            "per_seed": c}
    winner = max(summ, key=lambda k: summ[k]["val_acc_mean"])
    sel = {"phase": 2, "analysis": "multi-seed architecture comparison on S2",
           "criterion": CRITERION, "selection_uses_test_data": False,
           "n_seeds_per_arch": len(SEEDS), "epochs": 10,
           "candidates": summ, "selected_architecture": winner,
           "note": "selected on mean validation accuracy across seeds; the test columns "
                   "are each finished model's single test evaluation (HARD RULE 3)",
           "runtime_s": round(time.time() - t0, 1)}
    (P2 / "arch3_selection.json").write_text(json.dumps(sel, indent=2))
    pd.DataFrame(rows).to_csv(P2 / "arch3_summary.csv", index=False)
    print("\n" + json.dumps({k: v for k, v in sel.items() if k != "candidates"},
                            indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                      for k, v in summ.items()}, indent=2))


if __name__ == "__main__":
    main()
