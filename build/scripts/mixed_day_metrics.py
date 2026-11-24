"""Phase 2 - research-viewer data: mixed-day metrics for Track B (no training).

  .venv/bin/python build/scripts/mixed_day_metrics.py

From results/phase4/seed{42,43,44}/{rgb,spectral,fusion}/preds_test.csv (read-only),
per mode and seed:
  * image-level accuracy + AUC on mixed-label days 9-29 ONLY, and separately on
    days 1-8, with plant-cluster bootstrap CIs (5000 draws, seed 42, 95% CI),
    n images / n plants per cell,
  * label purity of each day (max(frac_pos, 1-frac_pos)),
  * how well "day number alone" predicts the label on days 1-8 (pooled AUC of the
    raw day number as score + the in-sample accuracy of a day->majority lookup).

Output: results/phase4/mixed_day_metrics.{json,csv} (NEW file names only, HARD RULE 3).
Restart-proof: skips if the JSON exists.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "phase4"
OUT_JSON = RES / "mixed_day_metrics.json"
OUT_CSV = RES / "mixed_day_metrics.csv"
SEEDS = [42, 43, 44]
MODES = ["rgb", "spectral", "fusion"]
DRAWS, BSEED = 5000, 42
MIXED_DAYS = list(range(9, 30))    # days 9-29 (interpretable stratum)
EARLY_DAYS = list(range(1, 9))     # days 1-8 (single-class days, confounded)


def cluster_bootstrap(df: pd.DataFrame, stat_fn, draws: int, seed: int):
    """Percentile bootstrap resampling PLANTS with replacement (plant-cluster)."""
    plants = df["plant_key"].unique()
    grouped = {p: df[df["plant_key"] == p] for p in plants}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        pick = rng.choice(len(plants), len(plants), replace=True)
        sub = pd.concat([grouped[plants[i]] for i in pick], ignore_index=True)
        v = stat_fn(sub)
        if v is not None and np.isfinite(v):
            stats.append(v)
    if len(stats) < draws * 0.5:
        return None, None, len(stats)
    lo, hi = np.quantile(stats, [(1 - 0.95) / 2, 1 - (1 - 0.95) / 2])
    return float(lo), float(hi), len(stats)


def acc_of(df: pd.DataFrame) -> float | None:
    return float((df["pred"] == df["y_true"]).mean()) if len(df) else None


def auc_of(df: pd.DataFrame) -> float | None:
    if df["y_true"].nunique() < 2:
        return None
    return float(roc_auc_score(df["y_true"], df["p_infected"]))


def main() -> int:
    if OUT_JSON.exists():
        print(f"[skip] {OUT_JSON} exists")
        return 0
    t0 = time.time()
    cells, purity_rows = [], []
    day_pool: dict[str, pd.DataFrame] = {}

    for seed in SEEDS:
        for mode in MODES:
            f = RES / f"seed{seed}" / mode / "preds_test.csv"
            df = pd.read_csv(f)
            df["pred"] = (df["p_infected"] >= 0.5).astype(int)
            day_pool[f"{mode}_s{seed}"] = df
            for stratum, days in (("days_9_29", MIXED_DAYS), ("days_1_8", EARLY_DAYS)):
                sub = df[df["day"].isin(days)]
                a_lo, a_hi, a_n = cluster_bootstrap(sub, acc_of, DRAWS, BSEED)
                u_lo, u_hi, u_n = cluster_bootstrap(sub, auc_of, DRAWS, BSEED + 1)
                cells.append({
                    "mode": mode, "seed": seed, "stratum": stratum,
                    "n_images": int(len(sub)), "n_plants": int(sub["plant_key"].nunique()),
                    "days": [min(days), max(days)],
                    "image_accuracy": acc_of(sub),
                    "image_accuracy_ci95": [a_lo, a_hi],
                    "image_auc": auc_of(sub),
                    "image_auc_ci95": [u_lo, u_hi],
                    "bootstrap_draws_valid_acc": a_n,
                    "bootstrap_draws_valid_auc": u_n,
                    "source": str(f.relative_to(ROOT)),
                })

    # label purity per day: split-dependent (different test plants per seed)
    base = day_pool["rgb_s42"]
    per_seed_purity = {}
    for seed in SEEDS:
        sdf = day_pool[f"rgb_s{seed}"]
        rows = []
        for day, g in sdf.groupby("day"):
            frac = float(g["y_true"].mean())
            rows.append({"day": int(day), "n_images": int(len(g)),
                         "frac_positive": frac, "purity": max(frac, 1.0 - frac),
                         "single_class": bool(frac in (0.0, 1.0))})
        per_seed_purity[str(seed)] = rows
    purity_rows = per_seed_purity["42"]  # back-compat field
    purity_consistent = False  # different plant-disjoint splits have different test plants

    # "day number alone" on days 1-8
    early = base[base["day"].isin(EARLY_DAYS)]
    day_auc = (float(roc_auc_score(early["y_true"], early["day"]))
               if early["y_true"].nunique() == 2 else None)
    lookup = early.groupby("day")["y_true"].agg(lambda s: s.mode().iloc[0])
    lookup_acc = float((early["day"].map(lookup) == early["y_true"]).mean())
    early_pure = all(p["single_class"] for s in per_seed_purity.values()
                      for p in s if p["day"] in EARLY_DAYS)

    out = {
        "phase": 2,
        "analysis": "mixed-day image-level metrics with plant-cluster bootstrap",
        "strata": {"days_9_29": "mixed-label days (interpretable)",
                   "days_1_8": "single-class days (confounded - see day purity)"},
        "bootstrap": {"draws": DRAWS, "seed": BSEED, "ci": 0.95,
                      "cluster": "plant_key (resampled with replacement)"},
        "cells": cells,
        "day_purity": purity_rows,
        "day_purity_per_seed": per_seed_purity,
        "day_purity_note": "Day purity is split-dependent (plant-disjoint splits contain different test plants). Days 1-8 are single-class in every seed.",
        "day_purity_consistent_across_files": purity_consistent,
        "all_days_1_8_single_class_all_seeds": early_pure,
        "day_number_only_on_days_1_8": {
            "pooled_auc_of_raw_day_number": day_auc,
            "in_sample_accuracy_of_day_to_majority_lookup": lookup_acc,
            "all_days_1_8_single_class": early_pure,
            "n_images": int(len(early)),
            "note": "labels on days 1-8 are deterministic per day; a predictor that sees "
                    "only the day number reaches the lookup accuracy without looking at "
                    "the leaf - so image-level accuracy on days 1-8 is explainable by "
                    "non-disease cues correlated with the capture day",
        },
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    pd.DataFrame(cells).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_JSON} and {OUT_CSV}: {len(cells)} cells, "
          f"purity consistent={purity_consistent}, runtime {out['runtime_s']}s")
    print("day-only (1-8):", json.dumps(out["day_number_only_on_days_1_8"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
