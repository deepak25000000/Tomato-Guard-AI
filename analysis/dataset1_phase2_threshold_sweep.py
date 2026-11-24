"""GROUP A / item A3 - S2 re-built at a LOOSER near-dup threshold and re-run, 3 seeds.

  .venv/bin/python analysis/dataset1_phase2_threshold_sweep.py --threshold 4
  .venv/bin/python analysis/dataset1_phase2_threshold_sweep.py --threshold 6

Steps (all recorded):
  1. Build the split with the PROJECT's own script: research/01_make_splits.py
     (same data-dir, fracs (0.15/0.15) and seed 42 as the audited t=3 split;
     only --near-dup-threshold changes). Raw data untouched (HARD RULE 2).
  2. HARD RULE 4: assert group + sha256 disjointness across train/validation/test
     (via research/common.assert_disjoint, the same function the t=3 split used).
     Failure stops the run before any training.
  3. Train mobilenet_v3_small, 3 seeds (42/43/44), the notebook recipe
     (trackA_lib.train_one: 10 epochs, batch 16, AdamW 1e-4, best epoch by
     VALIDATION accuracy); each finished model is evaluated on test ONCE
     (HARD RULE 3).

Writes results/phase2/thresh{T}_seed{seed}.json (+ _preds.csv, _history.csv),
thresh{T}_split_assertions.json, and upserts threshold_sweep_summary.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
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
from common import assert_disjoint  # noqa: E402

DATA = ROOT / "Tomato Leaf Disease Classification Dataset" / "Tomato Dataset"
RESEARCH = ROOT / "tomatoguard" / "research"
PY = str(ROOT / ".venv" / "bin" / "python")
ARCH = "mobilenet_v3_small"
SEEDS = [42, 43, 44]


def build_splits(t: int, out: Path) -> dict:
    cmd = [PY, "01_make_splits.py", "--data-dir", str(DATA), "--out-dir", str(out),
           "--near-dup-threshold", str(t), "--test-frac", "0.15", "--val-frac", "0.15",
           "--seed", "42"]
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    (P2 / f"thresh{t}_make_splits_stdout.txt").write_text(
        "$ " + " ".join(cmd) + "\n" + r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: 01_make_splits.py t={t} exited {r.returncode} "
                         f"(see thresh{t}_make_splits_stdout.txt)")
    return json.loads((out / "split_report.json").read_text())


def assert_split(split_dir: Path, t: int) -> dict:
    """HARD RULE 4 - same assertions the audited t=3 split passed."""
    frames = {}
    for name in ("train", "validation", "test"):
        frames[name] = pd.read_csv(split_dir / f"{name}.csv")
    # 01_make_splits.py already writes sha256 + group into each split CSV; verify
    # consistency with groups.csv without merging (a merge would suffix the
    # duplicate columns to group_x/group_y).
    groups = pd.read_csv(split_dir / "groups.csv")   # path, class, sha256, group
    rep: dict = {"threshold": t, "assertions": {}}
    # every image in exactly one split
    allp = pd.concat([f["path"] for f in frames.values()])
    rep["assertions"]["image_once"] = (
        "PASS" if allp.duplicated().sum() == 0 and len(allp) == len(groups)
        else f"FAIL: dups={int(allp.duplicated().sum())} n={len(allp)}/{len(groups)}")
    for name in frames:
        if not {"sha256", "group"} <= set(frames[name].columns):
            raise SystemExit(f"FATAL: t={t} {name}: split csv lacks sha256/group columns")
        if frames[name]["group"].isna().any() or frames[name]["sha256"].isna().any():
            raise SystemExit(f"FATAL: t={t} {name}: missing group/sha256 values")
        if set(frames[name]["path"]) - set(groups["path"]):
            raise SystemExit(f"FATAL: t={t} {name}: paths missing from groups.csv")
    for col in ("group", "sha256"):
        try:
            assert_disjoint(frames, col)
            rep["assertions"][f"{col}_disjoint"] = "PASS"
        except AssertionError as e:
            rep["assertions"][f"{col}_disjoint"] = f"FAIL: {e}"
    rep["sizes"] = {k: len(v) for k, v in frames.items()}
    failed = {k: v for k, v in rep["assertions"].items() if v != "PASS"}
    (P2 / f"thresh{t}_split_assertions.json").write_text(json.dumps(rep, indent=2))
    if failed:
        raise SystemExit(f"FATAL: leakage assertion(s) failed for t={t}: {failed} "
                         f"(HARD RULE 4)")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, required=True, choices=[4, 6])
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    split_dir = RES / f"splits_S2_t{a.threshold}"

    report = build_splits(a.threshold, split_dir)
    print(f"[t={a.threshold}] split built: {report}", flush=True)
    rep = assert_split(split_dir, a.threshold)
    print(f"[t={a.threshold}] leakage assertions: {rep['assertions']}", flush=True)

    classes = json.loads((split_dir / "classes.json").read_text())
    dev = torch.device("cpu")
    rows = []
    for seed in SEEDS:
        print(f"\n=== t={a.threshold} seed={seed} arch={ARCH} ===", flush=True)
        t1 = time.time()
        seed_all(seed)
        model = build_model(ARCH, len(classes), pretrained=True).to(dev)
        out = train_one(model, split_dir / "train.csv", split_dir / "validation.csv",
                        classes, dev, epochs=a.epochs, seed=seed,
                        log_prefix=f"[t{a.threshold}:{seed}] ")
        m = evaluate(out["model"], classes, split_dir, dev,
                     f"thresh{a.threshold}_seed{seed}", P2,
                     extra={"arch": ARCH, "split": f"S2_t{a.threshold}", "seed": seed,
                            "near_dup_threshold": a.threshold,
                            "best_val_accuracy": out["best_val_accuracy"],
                            "best_epoch": out["best_epoch"], "epochs": a.epochs,
                            "runtime_s": round(time.time() - t1, 1),
                            "torch": torch.__version__})
        (P2 / f"thresh{a.threshold}_seed{seed}.json").write_text(json.dumps(m, indent=2))
        pd.DataFrame(out["history"]).to_csv(
            P2 / f"thresh{a.threshold}_seed{seed}_history.csv", index=False)
        rows.append({"threshold": a.threshold, "seed": seed, "arch": ARCH,
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "best_val_accuracy": out["best_val_accuracy"],
                     "best_epoch": out["best_epoch"], "runtime_s": m["runtime_s"]})
        print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "n_test")}, indent=2),
              flush=True)

    # upsert this threshold's rows into the combined sweep csv
    csv_path = P2 / "threshold_sweep_summary.csv"
    old = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    old = old[old["threshold"] != a.threshold] if len(old) else old
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).sort_values(
        ["threshold", "seed"]).to_csv(csv_path, index=False)
    agg = {"threshold": a.threshold, "n_seeds": len(rows),
           "accuracy_mean": round(float(pd.DataFrame(rows)["accuracy"].mean()), 6),
           "accuracy_sd": round(float(pd.DataFrame(rows)["accuracy"].std(ddof=1)), 6),
           "macro_f1_mean": round(float(pd.DataFrame(rows)["macro_f1"].mean()), 6),
           "macro_f1_sd": round(float(pd.DataFrame(rows)["macro_f1"].std(ddof=1)), 6),
           "split_sizes": rep["sizes"], "split_assertions": rep["assertions"],
           "split_report": report, "runtime_s": round(time.time() - t0, 1)}
    (P2 / f"thresh{a.threshold}_aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n" + json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
