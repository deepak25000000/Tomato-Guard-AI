"""PHASE 4 - Track B (TOBRFV-LMID) experiments: RGB vs spectral vs feature fusion,
plus a label-shuffle negative control, on the PLANT-DISJOINT splits built in Phase 3.

  .venv/bin/python analysis/dataset2_phase4_experiments.py

Everything is delegated to the PROJECT'S OWN scripts (unedited except for the logged
`--image-size` flag, see CHANGELOG):
  * splits          -> caep/split_by_plant.py      (produced in Phase 3 stage E)
  * leakage asserts -> caep/check_leakage.py       (produced in Phase 3 stage E)
  * training        -> fusion/fusion.py            (this script only calls its CLI)
  * comparisons     -> caep/paired_stats.py        (stdout parsed into json here)

HARD RULE 3: fusion.py picks the best epoch on VALIDATION AUC and touches the test set once,
at the end, for the finished model. This script never looks at test numbers to choose anything.

HARD RULE 4: this script REFUSES to run unless Phase 3's split_leakage_assertions.json shows
every assertion PASS.

Settings (Phase 0 "minimal" CPU scenario, results/phase0_runtime_proposal.json):
  epochs 8, image-size 160, 3 split seeds x 3 modes = 9 runs + 1 negative control.
The 160 px choice is a compute-driven reduction, declared in CHANGELOG; the default 224 is
unchanged in fusion.py itself.

Outputs under results/phase4/  (nothing here is typed by hand).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P3 = RES / "phase3"
P4 = RES / "phase4"
P4.mkdir(parents=True, exist_ok=True)
RESEARCH = ROOT / "tomatoguard" / "research"
PY = str(ROOT / ".venv" / "bin" / "python")
FUSION = RESEARCH / "fusion" / "fusion.py"
PAIRED = RESEARCH / "caep" / "paired_stats.py"

MODES = ["rgb", "spectral", "fusion"]
SEEDS = [42, 43, 44]
EPOCHS = 8
IMAGE_SIZE = 160
BATCH = 32
WORKERS = 0          # RAM is 7 GB and shared with the desktop; workers=0 is deterministic too


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ preconditions
def check_preconditions() -> None:
    ap = P3 / "split_leakage_assertions.json"
    if not ap.exists():
        raise SystemExit(f"FATAL: {ap} missing - run Phase 3 first (HARD RULE 4)")
    rep = json.loads(ap.read_text())
    failed = {k: v for k, v in rep.get("assertions", {}).items() if str(v).startswith("FAIL")}
    if failed:
        raise SystemExit(f"FATAL: Phase 3 leakage assertions failed: {failed} (HARD RULE 4)")
    log(f"[0] Phase 3 leakage assertions all PASS "
        f"({len(rep.get('assertions', {}))} checks) - proceeding")
    for s in SEEDS:
        mp = P3 / "plant_splits" / f"manifest_split_seed{s}.csv"
        if not mp.exists():
            raise SystemExit(f"FATAL: missing {mp} (HARD RULE 6)")
        d = pd.read_csv(mp)
        need = {"rgb", "nir800", "nir1000", "label", "plant_key", "split", "angle", "batch", "day"}
        miss = need - set(d.columns)
        if miss:
            raise SystemExit(f"FATAL: {mp} lacks columns {sorted(miss)} (HARD RULE 6)")
        if d["split"].isna().any():
            raise SystemExit(f"FATAL: {mp} has unassigned rows (HARD RULE 6)")


# ------------------------------------------------------------------ negative control
def make_shuffled_manifest(seed: int = 42) -> Path:
    """Permute the LABELS of the TRAIN rows only. Validation and test keep their true labels,
    so nothing can be learned: a healthy pipeline must end at chance (AUC ~ 0.5, acc ~ 0.5).
    If it does not, the evaluation is leaking."""
    src = P3 / "plant_splits" / f"manifest_split_seed{seed}.csv"
    dst = P4 / f"manifest_shuffled_seed{seed}.csv"
    d = pd.read_csv(src)
    rng = __import__("numpy").random.default_rng(seed)
    m = d["split"] == "train"
    lab = d.loc[m, "label"].to_numpy().copy()
    d.loc[m, "label"] = rng.permutation(lab)
    d.to_csv(dst, index=False)
    log(f"[1] negative-control manifest: {int(m.sum())} train labels permuted "
        f"(val/test untouched) -> {dst}")
    return dst


# ------------------------------------------------------------------ running fusion.py
def run_mode(mode: str, manifest: Path, outdir: Path, seed: int, tag: str) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(FUSION), "--mode", mode, "--manifest", str(manifest),
           "--out", str(outdir), "--epochs", str(EPOCHS), "--image-size", str(IMAGE_SIZE),
           "--batch-size", str(BATCH), "--num-workers", str(WORKERS), "--seed", str(seed)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    rt = round(time.time() - t0, 1)
    (outdir / "stdout.txt").write_text("$ " + " ".join(cmd) + "\n" + r.stdout +
                                       "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: fusion.py --mode {mode} (seed {seed}) exited "
                         f"{r.returncode} after {rt}s:\n{(r.stdout + r.stderr)[-2500:]}\n"
                         f"(HARD RULE 6 - not guessing, not skipping)")
    mpath = outdir / "metrics.json"
    if not mpath.exists():
        raise SystemExit(f"FATAL: fusion.py --mode {mode} produced no metrics.json (HARD RULE 6)")
    m = json.loads(mpath.read_text())
    m.update({"tag": tag, "split_seed": seed, "runtime_s": rt, "command": " ".join(cmd),
              "epochs": EPOCHS, "image_size": IMAGE_SIZE, "batch_size": BATCH})
    mpath.write_text(json.dumps(m, indent=2))
    log(f"    {tag:28s} image_acc {m['image_acc']:.4f}  image_auc {m['image_auc']:.4f}  "
        f"plant_acc {m['plant_acc']:.4f}  plant_auc {m['plant_auc']:.4f}  ({rt}s)")
    return m


# ------------------------------------------------------------------ paired stats
RX = {
    "n_plants": r"(\d+) test plants \((\d+) infected\)",
    "a": r"A: acc ([\d.]+)\s+AUC ([\d.]+)",
    "b": r"B: acc ([\d.]+)\s+AUC ([\d.]+)",
    "d_acc": r"delta acc \(B-A\): ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]",
    "d_auc": r"delta AUC \(B-A\): ([+-][\d.]+)\s+95% CI \[([+-][\d.]+), ([+-][\d.]+)\]",
    "mcnemar": r"McNemar exact: A-only-correct=(\d+)\s+B-only-correct=(\d+)\s+p=([\d.]+)",
}


def paired(pa: Path, pb: Path, out: Path, label: str) -> dict:
    cmd = [PY, str(PAIRED), "--a", str(pa), "--b", str(pb), "--boot", "5000", "--seed", "42"]
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    out.write_text("$ " + " ".join(cmd) + "\n" + r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: paired_stats.py failed for {label}: "
                         f"{(r.stdout + r.stderr)[-1500:]} (HARD RULE 6)")
    res: dict = {"label": label, "a": str(pa), "b": str(pb), "raw_stdout": r.stdout}
    for k, pat in RX.items():
        m = re.search(pat, r.stdout)
        if not m:
            raise SystemExit(f"FATAL: could not parse paired_stats output for {label} "
                             f"({k}); raw:\n{r.stdout} (HARD RULE 6 - no guessing)")
        g = m.groups()
        if k == "n_plants":
            res["n_test_plants"], res["n_infected_plants"] = int(g[0]), int(g[1])
        elif k in ("a", "b"):
            res[f"{k}_plant_acc"], res[f"{k}_plant_auc"] = float(g[0]), float(g[1])
        elif k in ("d_acc", "d_auc"):
            res[f"delta_{k[2:]}"] = float(g[0])
            res[f"delta_{k[2:]}_ci"] = [float(g[1]), float(g[2])]
            res[f"delta_{k[2:]}_ci_excludes_zero"] = not (float(g[1]) <= 0 <= float(g[2]))
        else:
            res["mcnemar_a_only_correct"], res["mcnemar_b_only_correct"] = int(g[0]), int(g[1])
            res["mcnemar_p"] = float(g[2])
    log(f"    paired {label}: delta_plant_acc {res['delta_acc']:+.3f} "
        f"CI {res['delta_acc_ci']}  McNemar p={res['mcnemar_p']:.4f}")
    return res


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--skip-control", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    check_preconditions()

    results, control = [], None
    for seed in a.seeds:
        man = P3 / "plant_splits" / f"manifest_split_seed{seed}.csv"
        log(f"\n=== split seed {seed} ===")
        for mode in a.modes:
            results.append(run_mode(mode, man, P4 / f"seed{seed}" / mode, seed,
                                    f"seed{seed}_{mode}"))

    # ---- negative control (one run, seed 42, rgb) ---------------------------
    if not a.skip_control:
        log("\n=== NEGATIVE CONTROL (train labels permuted) ===")
        cm = make_shuffled_manifest(42)
        control = run_mode("rgb", cm, P4 / "control" / "rgb_shuffled", 42,
                           "control_rgb_shuffled_labels")
        control["interpretation"] = (
            "test AUC should be ~0.5 and test accuracy ~chance; a value well above chance means "
            "the pipeline is leaking (labels, groups or near-duplicates) and no result here "
            "may be trusted")
        (P4 / "control" / "control_metrics.json").write_text(json.dumps(control, indent=2))

    # ---- paired, plant-level comparisons (project's own paired_stats.py) -----
    log("\n=== paired plant-level comparisons ===")
    pairs = [("spectral", "rgb"), ("fusion", "rgb"), ("fusion", "spectral")]
    paired_out = []
    for seed in a.seeds:
        for b, aa in pairs:
            pb = P4 / f"seed{seed}" / b / "preds_test.csv"
            pa = P4 / f"seed{seed}" / aa / "preds_test.csv"
            if pb.exists() and pa.exists():
                paired_out.append(paired(pa, pb, P4 / f"seed{seed}" / f"paired_{b}_vs_{aa}.txt",
                                         f"seed{seed}_{b}_vs_{aa}"))
    (P4 / "paired_stats.json").write_text(json.dumps(paired_out, indent=2))

    # ---- aggregate -----------------------------------------------------------
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                       for r in results])
    df.to_csv(P4 / "all_runs.csv", index=False)
    agg = {}
    for mode in a.modes:
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue
        agg[mode] = {
            "n_seeds": int(len(sub)),
            "image_acc_mean": float(sub.image_acc.mean()),
            "image_acc_std": float(sub.image_acc.std(ddof=1)) if len(sub) > 1 else None,
            "image_auc_mean": float(sub.image_auc.mean()),
            "image_auc_std": float(sub.image_auc.std(ddof=1)) if len(sub) > 1 else None,
            "image_f1_mean": float(sub.image_f1.mean()),
            "plant_acc_mean": float(sub.plant_acc.mean()),
            "plant_auc_mean": float(sub.plant_auc.mean()),
            "plant_auc_std": float(sub.plant_auc.std(ddof=1)) if len(sub) > 1 else None,
            "runtime_s_total": float(sub.runtime_s.sum()),
            "per_seed": sub[["split_seed", "image_acc", "image_auc", "image_f1",
                             "plant_acc", "plant_auc", "best_val_auc", "runtime_s"]]
                        .to_dict(orient="records"),
        }
    summary = {
        "phase": 4,
        "settings": {"epochs": EPOCHS, "image_size": IMAGE_SIZE, "batch_size": BATCH,
                     "num_workers": WORKERS, "split_seeds": a.seeds, "modes": a.modes,
                     "selection": "best epoch by VALIDATION image AUC (fusion.py's own rule)",
                     "test_policy": "test evaluated once per finished model"},
        "aggregate_by_mode": agg,
        "negative_control": {k: control[k] for k in
                             ("image_acc", "image_auc", "plant_acc", "plant_auc",
                              "best_val_auc", "interpretation")} if control else "not run",
        "n_paired_comparisons": len(paired_out),
        "runtime_s_total": round(time.time() - t0, 1),
        "source_files": ["results/phase3/split_leakage_assertions.json",
                         "results/phase3/plant_splits/manifest_split_seed<seed>.csv",
                         "results/phase4/seed<seed>/<mode>/metrics.json",
                         "results/phase4/seed<seed>/paired_<b>_vs_<a>.txt"],
    }
    (P4 / "phase4_summary.json").write_text(json.dumps(summary, indent=2))
    log("\n=== phase 4 aggregate ===")
    log(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                    for k, v in agg.items()}, indent=2))
    if control:
        log("negative control:", json.dumps(summary["negative_control"], indent=2))

    # ---- run log -------------------------------------------------------------
    import datetime
    runlog = RES / "run_log.md"
    old = runlog.read_text() if runlog.exists() else "# Run log\n"
    marker = "## Phase 4 - Track B experiments"
    if marker in old:          # block-scoped: remove ONLY our previous entry (CHANGELOG)
        i = old.index(marker)
        j = old.find("\n## ", i + 1)
        head = old[:i].rstrip("\n")
        tail = "" if j == -1 else old[j + 1:]
        old = head + ("\n\n" + tail if tail else "\n")
    runlog.write_text(old + "\n".join([
        f"\n{marker}\n",
        f"- time (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "- command: `.venv/bin/python analysis/dataset2_phase4_experiments.py`",
        f"- settings: epochs={EPOCHS}, image_size={IMAGE_SIZE}, batch={BATCH}, workers={WORKERS}",
        f"- runs: {len(results)} training runs + "
        f"{'1 negative control' if control else 'no control'} + {len(paired_out)} paired tests",
        f"- runtime: {summary['runtime_s_total']} s",
        f"- artifacts: results/phase4/ (phase4_summary.json, all_runs.csv, paired_stats.json)",
        "\n"]))


if __name__ == "__main__":
    main()
