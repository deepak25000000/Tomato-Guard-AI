"""GROUP B / items B2 + B3 - Track B, IMAGE-level comparison of rgb / spectral / fusion.

  .venv/bin/python analysis/dataset2_phase4_subgroup_analysis.py

B2: image-level accuracy and AUC per mode with PLANT-CLUSTER bootstrap CIs
    (resample the 34 test plants with replacement, take all their images - the images
    of one plant are not independent, so a naive image bootstrap would be too narrow).
    Also paired plant-cluster bootstrap differences between modes within each seed
    (fusion-rgb, fusion-spectral, rgb-spectral), because every mode of a seed shares
    the same test plants.

B3: accuracy + AUC per recording day, angle and batch, per mode AND per seed,
    with n (images) and n_pos per cell. Cells with a single class get AUC = null
    (undefined), never a fabricated number.

Inputs: results/phase4/seed{42,43,44}/{rgb,spectral,fusion}/preds_test.csv
        (image-level rows: plant_key,y_true,p_infected,day,angle,batch).
Every number is read from those files (HARD RULE 1). Test data is only ever READ
here - no selection or tuning happens in this script (HARD RULE 3).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "results" / "phase4"
SEEDS = [42, 43, 44]
MODES = ["rgb", "spectral", "fusion"]
PAIRS = [("fusion", "rgb"), ("fusion", "spectral"), ("rgb", "spectral")]
B_BOOT = 5000
SEED_RNG = 20261121


def acc_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, float | None]:
    a = float((y == (p >= 0.5)).mean())
    u = np.unique(y)
    auc = float(roc_auc_score(y, p)) if len(u) == 2 else None
    return a, auc


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, plants: np.ndarray,
                      rng: np.random.Generator) -> dict:
    """Percentile 95% CI resampling PLANTS with replacement."""
    uniq = np.unique(plants)
    idx_of = {pl: np.flatnonzero(plants == pl) for pl in uniq}
    accs = np.empty(B_BOOT)
    aucs: list[float] = []
    for b in range(B_BOOT):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[d] for d in draw])
        accs[b] = (y[sel] == (p[sel] >= 0.5)).mean()
        if len(np.unique(y[sel])) == 2:
            aucs.append(roc_auc_score(y[sel], p[sel]))
    out = {"acc_ci95": [round(float(np.percentile(accs, 2.5)), 6),
                        round(float(np.percentile(accs, 97.5)), 6)],
           "n_bootstrap": B_BOOT, "bootstrap_unit": "plant_key"}
    if aucs:
        out["auc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 6),
                           round(float(np.percentile(aucs, 97.5)), 6)]
        out["n_bootstrap_valid_auc"] = len(aucs)
    else:
        out["auc_ci95"] = None
    return out


def load(seed: int, mode: str) -> pd.DataFrame:
    df = pd.read_csv(P4 / f"seed{seed}" / mode / "preds_test.csv")
    need = {"plant_key", "y_true", "p_infected", "day", "angle", "batch"}
    if not need <= set(df.columns) or len(df) == 0:
        raise SystemExit(f"FATAL: seed{seed}/{mode}/preds_test.csv missing columns "
                         f"{need - set(df.columns)} or empty (HARD RULE 6)")
    return df


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED_RNG)

    # ------------------------------------------------------------------ B2 -------
    per_seed_mode, b3_rows = {}, []
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    for seed in SEEDS:
        for mode in MODES:
            df = load(seed, mode)
            frames[(seed, mode)] = df
            y = df.y_true.to_numpy()
            p = df.p_infected.to_numpy()
            a, u = acc_auc(y, p)
            ci = cluster_bootstrap(y, p, df.plant_key.to_numpy(), rng)
            per_seed_mode[f"seed{seed}/{mode}"] = {
                "n_images": int(len(df)), "n_plants": int(df.plant_key.nunique()),
                "n_pos": int(y.sum()),
                "image_acc": round(a, 6),
                "image_auc": (round(u, 6) if u is not None else None),
                **ci,
                "source": f"results/phase4/seed{seed}/{mode}/preds_test.csv"}

            # ---- B3 subgroups -------------------------------------------------
            for col in ("day", "angle", "batch"):
                for cell, g in df.groupby(col):
                    gy, gp = g.y_true.to_numpy(), g.p_infected.to_numpy()
                    ga, gu = acc_auc(gy, gp)
                    b3_rows.append({
                        "seed": seed, "mode": mode, "factor": col,
                        "cell": str(cell), "n_images": int(len(g)),
                        "n_plants": int(g.plant_key.nunique()),
                        "n_pos": int(gy.sum()),
                        "accuracy": round(ga, 6),
                        "auc": (round(gu, 6) if gu is not None else None),
                        "auc_defined": bool(gu is not None)})

    # paired cluster bootstrap within seed (same plants for both modes) ------------
    paired = {}
    sort_key = ["plant_key", "day", "angle"]     # unique per image row (one capture
    # tuple per plant x day x angle; batch/cls are fixed within a plant)
    for seed in SEEDS:
        for b_mode, a_mode in PAIRS:          # B vs A: difference = B - A
            da = frames[(seed, a_mode)].sort_values(sort_key, kind="stable").reset_index(drop=True)
            db = frames[(seed, b_mode)].sort_values(sort_key, kind="stable").reset_index(drop=True)
            # sufficient conditions for pairing at plant-cluster level: identical row
            # count, identical plant per index, identical label per index (labels are
            # constant within a plant, so any tie-order difference inside a plant is
            # harmless for plant-cluster resampling)
            if (len(da) != len(db)
                    or not da.plant_key.equals(db.plant_key)
                    or not (da.y_true.to_numpy() == db.y_true.to_numpy()).all()):
                raise SystemExit(f"FATAL: seed{seed} {a_mode} vs {b_mode}: the two "
                                 f"modes do not cover identical image rows - cannot "
                                 f"pair (HARD RULE 6)")
            y = da.y_true.to_numpy()
            pa, pb = da.p_infected.to_numpy(), db.p_infected.to_numpy()
            plants = da.plant_key.to_numpy()
            uniq = np.unique(plants)
            idx_of = {pl: np.flatnonzero(plants == pl) for pl in uniq}
            d_acc = np.empty(B_BOOT)
            d_auc: list[float] = []
            obs_a, obs_u_a = acc_auc(y, pa)
            obs_b, obs_u_b = acc_auc(y, pb)
            for k in range(B_BOOT):
                draw = rng.choice(uniq, size=len(uniq), replace=True)
                sel = np.concatenate([idx_of[d] for d in draw])
                yb = y[sel]
                d_acc[k] = (yb == (pb[sel] >= 0.5)).mean() - \
                           (yb == (pa[sel] >= 0.5)).mean()
                if len(np.unique(yb)) == 2:
                    d_auc.append(roc_auc_score(yb, pb[sel]) - roc_auc_score(yb, pa[sel]))
            paired[f"seed{seed}/{b_mode}_minus_{a_mode}"] = {
                "n_plants": int(len(uniq)),
                "acc_difference": round(obs_b - obs_a, 6),
                "acc_diff_ci95": [round(float(np.percentile(d_acc, 2.5)), 6),
                                  round(float(np.percentile(d_acc, 97.5)), 6)],
                "acc_p_boot_le_zero": round(float((d_acc <= 0).mean()), 6),
                "auc_difference": (round(obs_u_b - obs_u_a, 6)
                                   if obs_u_a is not None and obs_u_b is not None else None),
                "auc_diff_ci95": ([round(float(np.percentile(d_auc, 2.5)), 6),
                                   round(float(np.percentile(d_auc, 97.5)), 6)]
                                  if d_auc else None),
                "auc_p_boot_le_zero": (round(float((np.array(d_auc) <= 0).mean()), 6)
                                       if d_auc else None)}

    # aggregate across seeds (point estimates only; CIs above are within-seed) -----
    agg = {}
    for mode in MODES:
        accs = [per_seed_mode[f"seed{s}/{mode}"]["image_acc"] for s in SEEDS]
        aucs = [per_seed_mode[f"seed{s}/{mode}"]["image_auc"] for s in SEEDS]
        agg[mode] = {
            "acc_mean": round(float(np.mean(accs)), 6),
            "acc_sd": round(float(np.std(accs, ddof=1)), 6),
            "auc_mean": round(float(np.mean(aucs)), 6),
            "auc_sd": round(float(np.std(aucs, ddof=1)), 6)}

    out = {"phase": 4,
           "analysis": "B2 image-level mode comparison with plant-cluster bootstrap; "
                       "B3 per day/angle/batch",
           "bootstrap": {"unit": "plant_key", "n": B_BOOT, "rng_seed": SEED_RNG,
                         "ci": "percentile 95%"},
           "per_seed_mode": per_seed_mode,
           "aggregate_point_estimates_across_seeds": agg,
           "paired_cluster_bootstrap_differences": paired,
           "note": "plant-level metrics are saturated at 1.0 (34 test plants), which "
                   "is why this image-level analysis with plant-cluster CIs exists; "
                   "read-only over finished runs (HARD RULE 3)",
           "runtime_s": round(time.time() - t0, 1)}
    (P4 / "image_level_cluster_bootstrap.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(b3_rows).to_csv(P4 / "subgroup_metrics_day_angle_batch.csv", index=False)
    (P4 / "subgroup_metrics_day_angle_batch.json").write_text(json.dumps(
        {"phase": 4, "analysis": "B3 accuracy/AUC by day, angle, batch per mode+seed",
         "auc_null_means": "single-class cell - AUC undefined, reported as null",
         "rows": len(b3_rows),
         "source_files": [f"results/phase4/seed{s}/{m}/preds_test.csv"
                          for s in SEEDS for m in MODES]}, indent=2))

    print("=== B2 aggregate (image level, mean over 3 seeds) ===")
    print(json.dumps(agg, indent=2))
    print("=== B2 per-seed CIs (first seed) ===")
    print(json.dumps({k: v for k, v in per_seed_mode.items() if k.startswith("seed42")},
                     indent=2))
    print("=== paired diffs ===")
    print(json.dumps(paired, indent=2))
    print(f"wrote {P4}/image_level_cluster_bootstrap.json and "
          f"subgroup_metrics_day_angle_batch.csv ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()
