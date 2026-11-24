"""PHASE 2a - Build the two Track A splits.

  S1 = the user's ORIGINAL image-level stratified split (their notebook CSVs, stale paths remapped)
  S2 = group-disjoint split produced by the project's own research/01_make_splits.py
       at the Phase-1-chosen near-duplicate threshold

  .venv/bin/python analysis/dataset1_phase2_prepare_splits.py

Writes:
  results/splits_S1/{train,validation,test}.csv + classes.json      (S1)
  results/splits_S2/{train,validation,test}.csv + classes.json      (S2, via 01_make_splits.py)
  results/phase2/split_sizes.json
  results/phase2/split_leakage_assertions.json   (HARD RULE 4 - failure aborts)
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2 = RES / "phase2"
P2.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

from common import assert_disjoint  # noqa: E402

DATA = ROOT / "Tomato Leaf Disease Classification Dataset" / "Tomato Dataset"
ORIG = Path.home() / "Downloads" / "Tomato_Research_Results"
CLAIMS = json.loads((ROOT / "analysis" / "reference_claims.json").read_text())
PY = str(ROOT / ".venv" / "bin" / "python")
RESEARCH = ROOT / "tomatoguard" / "research"

# Phase 1 choice: largest threshold in 0..10 whose largest group stays <=5% of the dataset,
# and the first threshold that recalls 100% of the independent NCC@0.98 matches.
NEAR_DUP_THRESHOLD = json.loads((RES / "phase1" / "phase1_summary.json").read_text())[
    "d_near_duplicates"]["chosen_threshold"]


def main() -> None:
    classes = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    (RES / "splits_S1").mkdir(exist_ok=True)

    # index workspace files by (class, basename) -> guarantees a unique remap
    idx: dict = {}
    for c in classes:
        for p in (DATA / c).rglob("*"):
            if p.is_file():
                k = (c, p.name)
                if k in idx:
                    raise SystemExit(f"FATAL: duplicate basename within class: {k}")
                idx[k] = str(p)

    sizes, checks = {}, {}
    for name in ("train", "validation", "test"):
        src = pd.read_csv(ORIG / f"{name}.csv")
        paths = []
        for _, r in src.iterrows():
            p = idx.get((r["class"], Path(r["path"]).name))
            if p is None:
                raise SystemExit(f"FATAL: cannot remap {r['path']}")
            paths.append(p)
        out = pd.DataFrame({"path": paths, "class": src["class"]})
        out.to_csv(RES / "splits_S1" / f"{name}.csv", index=False)
        sizes[name] = len(out)
    (RES / "splits_S1" / "classes.json").write_text(json.dumps(classes, indent=2))

    claim = CLAIMS["phase1_original_split_sizes"]
    checks["S1_sizes_match_claim"] = {k: sizes[k] == claim[k] for k in sizes}
    if not all(checks["S1_sizes_match_claim"].values()):
        raise SystemExit(f"FATAL: S1 sizes {sizes} != claim {claim} - stopping (HARD RULE 6)")

    # S1 is the object of the leakage study, so its leakage is MEASURED (Phase 1e), not asserted.
    s1 = {k: pd.read_csv(RES / "splits_S1" / f"{k}.csv") for k in ("train", "validation", "test")}
    s1_exact = {}
    for k in ("validation", "test"):
        ov = set(s1[k]["path"]) & set(s1["train"]["path"])
        s1_exact[k] = len(ov)
    checks["S1_path_overlap_with_train"] = s1_exact     # must be 0 (same files, disjoint sets)

    # ---------------- S2 via the project's own script -----------------------------
    cmd = [PY, "01_make_splits.py", "--data-dir", str(DATA),
           "--out-dir", str(RES / "splits_S2"), "--near-dup-threshold", str(NEAR_DUP_THRESHOLD),
           "--test-frac", "0.15", "--val-frac", "0.15", "--seed", "42"]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    (P2 / "01_make_splits_stdout.txt").write_text(r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        raise SystemExit(f"FATAL: 01_make_splits.py exited {r.returncode}")
    print(r.stdout[-2500:])

    s2 = {k: pd.read_csv(RES / "splits_S2" / f"{k}.csv") for k in ("train", "validation", "test")}
    sizes2 = {k: len(v) for k, v in s2.items()}

    # ---------------- HARD RULE 4: leakage assertions (abort on failure) ----------
    report: dict = {"near_dup_threshold_used_for_S2": NEAR_DUP_THRESHOLD,
                    "S1_sizes": sizes, "S2_sizes": sizes2,
                    "checks": checks, "assertions": {}}
    try:
        assert_disjoint(s2, "group")       # near-duplicate group disjoint
        report["assertions"]["S2_group_disjoint"] = "PASS"
    except AssertionError as e:
        report["assertions"]["S2_group_disjoint"] = f"FAIL: {e}"
    try:
        assert_disjoint(s2, "sha256")      # exact hash disjoint
        report["assertions"]["S2_sha256_disjoint"] = "PASS"
    except AssertionError as e:
        report["assertions"]["S2_sha256_disjoint"] = f"FAIL: {e}"
    try:
        assert_disjoint(s1, "path")        # S1 sets are disjoint (not group-disjoint by design)
        report["assertions"]["S1_path_disjoint"] = "PASS"
    except AssertionError as e:
        report["assertions"]["S1_path_disjoint"] = f"FAIL: {e}"

    # explicit per-class distribution, so stratification is auditable
    report["class_counts"] = {
        "S1": {k: dict(Counter(s1[k]["class"])) for k in s1},
        "S2": {k: dict(Counter(s2[k]["class"])) for k in s2}}

    (P2 / "split_leakage_assertions.json").write_text(json.dumps(report, indent=2))
    (P2 / "split_sizes.json").write_text(json.dumps(
        {"S1": sizes, "S2": sizes2, "classes": classes,
         "near_dup_threshold": NEAR_DUP_THRESHOLD}, indent=2))

    failed = {k: v for k, v in report["assertions"].items() if v != "PASS"}
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(f"FATAL: leakage assertion(s) failed: {failed}  (HARD RULE 4 - run stopped)")
    print("\nAll leakage assertions PASSED.")


if __name__ == "__main__":
    main()
