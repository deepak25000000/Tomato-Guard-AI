"""GROUP A / item A5 (part 1) - re-train the S1 recipe EXCLUDING the 48 high-confidence
cross-label images, 3 seeds, and report the change versus the S1 baseline.

  .venv/bin/python analysis/dataset1_phase2_excl48.py

The 48 images are the distinct images involved in the NCC-high-confidence (>=0.998)
cross-label near-duplicate pairs from Phase 1 - the exact set counted in
results/phase1/label_conflict_summary.json as
n_distinct_images_involved_ncc_high_conf (asserted below, HARD RULE 1).

Raw data is never touched (HARD RULE 2): the exclusion happens only in the split
CSVs written to results/splits_S1_excl48/. Recipe = trackA_lib (the notebook recipe:
mobilenet_v3_small, 10 epochs, batch 16, AdamW 1e-4, best epoch by validation
accuracy); each finished model is evaluated on test ONCE (HARD RULE 3).
Writes results/splits_S1_excl48/* and results/phase2/excl48_seed{42,43,44}.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P1, P2 = RES / "phase1", RES / "phase2"
S1 = RES / "splits_S1"
OUT = RES / "splits_S1_excl48"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from trackA_lib import evaluate, seed_all, train_one  # noqa: E402
from models import build_model  # noqa: E402

ARCH = "mobilenet_v3_small"
SEEDS = [42, 43, 44]
METHOD = "ncc>=0.998"


def excluded_paths() -> set[str]:
    d = pd.read_csv(P1 / "cross_label_near_duplicates.csv")
    ncc = d[d["method"] == METHOD]
    paths = set(ncc["path_i"]) | set(ncc["path_j"])
    claim = json.loads((P1 / "label_conflict_summary.json").read_text())[
        "n_distinct_images_involved_ncc_high_conf"]
    if len(paths) != claim:
        raise SystemExit(f"FATAL: distinct NCC-high-conf images {len(paths)} != audited "
                         f"{claim} (HARD RULE 1/6)")
    return paths


def main() -> None:
    t0 = time.time()
    drop = excluded_paths()
    print(f"excluding {len(drop)} images (method={METHOD})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    sizes, dropped = {}, {}
    for name in ("train", "validation", "test"):
        src = pd.read_csv(S1 / f"{name}.csv")
        keep = ~src["path"].isin(drop)
        dropped[name] = int((~keep).sum())
        out = src[keep].reset_index(drop=True)
        out.to_csv(OUT / f"{name}.csv", index=False)
        sizes[name] = {"before": int(len(src)), "after": int(len(out))}
    shutil.copy(S1 / "classes.json", OUT / "classes.json")
    if sum(dropped.values()) != len(drop):
        raise SystemExit(f"FATAL: dropped {sum(dropped.values())} rows but {len(drop)} "
                         f"unique images - a split overlap exists (HARD RULE 6)")
    report = {"n_excluded": len(drop), "method": METHOD,
              "source": "results/phase1/cross_label_near_duplicates.csv",
              "counts_per_split": dropped, "sizes": sizes,
              "assertions": {"all_48_removed_exactly_once": "PASS",
                             "raw_data_untouched": "PASS (only split csvs written "
                                                   "under results/)"}}
    if any(v["after"] == 0 for v in sizes.values()):
        raise SystemExit("FATAL: an exclusion emptied a split (HARD RULE 6)")
    left = set()
    for name in ("train", "validation", "test"):
        left |= set(pd.read_csv(OUT / f"{name}.csv")["path"]) & drop
    if left:
        raise SystemExit(f"FATAL: {len(left)} excluded images still present")
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    classes = json.loads((OUT / "classes.json").read_text())
    dev = torch.device("cpu")
    for seed in SEEDS:
        print(f"\n=== S1_excl48 seed={seed} ===", flush=True)
        jp = P2 / f"excl48_seed{seed}.json"
        if jp.exists():     # resume after host restart (CHANGELOG): run fully completed
            print(f"[resume] excl48 seed {seed} already complete - skipped", flush=True)
            continue
        t1 = time.time()
        seed_all(seed)
        model = build_model(ARCH, len(classes), pretrained=True).to(dev)
        out = train_one(model, OUT / "train.csv", OUT / "validation.csv", classes, dev,
                        epochs=10, seed=seed, log_prefix=f"[excl48:{seed}] ")
        m = evaluate(out["model"], classes, OUT, dev, f"excl48_seed{seed}", P2,
                     extra={"arch": ARCH, "split": "S1_excl48", "seed": seed,
                            "n_excluded": len(drop),
                            "best_val_accuracy": out["best_val_accuracy"],
                            "best_epoch": out["best_epoch"], "epochs": 10,
                            "runtime_s": round(time.time() - t1, 1),
                            "torch": torch.__version__})
        (P2 / f"excl48_seed{seed}.json").write_text(json.dumps(m, indent=2))
        pd.DataFrame(out["history"]).to_csv(P2 / f"excl48_seed{seed}_history.csv",
                                            index=False)
        print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "n_test")},
                         indent=2), flush=True)
    print(f"done in {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()
