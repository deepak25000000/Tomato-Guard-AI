"""GROUP B / item B4 - confound baselines for Track B (metadata-only and
intensity-statistics logistic regression) plus the image-level-vs-plant-disjoint
leakage demo. 5 plant-disjoint seeds.

  .venv/bin/python analysis/dataset2_phase4_confound_baselines.py

Unit of analysis: the capture tuple (one row of results/phase3/manifest_tomato.csv,
the exact Phase-4 population: 4807 tuples / 167 plants).

Feature sets (pre-declared, no tuning anywhere - fixed C=1 pipeline, HARD RULE 3):
  * metadata:      angle, batch (I/II -> 0/1), day, RGB crop width, height,
                   aspect (w/h), file size in bytes
  * intensity:     rgb_mean, rgb_red, vnir_mean, vnir_red, nir800_mean,
                   nir800_red, nir1000_mean, nir1000_red (the 4 streams joined
                   per tuple from results/phase3/intensities.csv)

Protocols (5 seeds each: 42,43,44,45,46):
  * plant_disjoint: the project's own caep/split_by_plant.py, then HARD RULE 4
                    assertion via the project's caep/check_leakage.py (must PASS;
                    failure aborts the run).
  * image_level:    group-blind random row split, stratified by label - the WRONG
                    protocol, kept deliberately as the leakage demo. Its
                    check_leakage.py run is EXPECTED to detect plant leakage; the
                    exit code and output are recorded as evidence, not asserted
                    away (this is the demonstration).

Outputs: results/phase4/confounds/
  features_manifest.csv, confound_baselines.{json,csv},
  leakage_checks/ (check_leakage stdout per split), image_level splits.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P3, P4 = RES / "phase3", RES / "phase4"
OUT = P4 / "confounds"
CHECKS = OUT / "leakage_checks"
RESEARCH = ROOT / "tomatoguard" / "research"
PY = str(ROOT / ".venv" / "bin" / "python")
SEEDS = [42, 43, 44, 45, 46]
TEST_FRAC, VAL_FRAC = 0.20, 0.15

META_COLS = ["angle", "batch_bin", "day", "rgb_width", "rgb_height", "rgb_aspect",
             "rgb_bytes"]
INT_COLS = ["rgb_mean", "rgb_red", "vnir_mean", "vnir_red",
            "nir800_mean", "nir800_red", "nir1000_mean", "nir1000_red"]
FEATURE_SETS = {"metadata": META_COLS, "intensity": INT_COLS}
STREAM_COL = {"rgb": "RGB", "vnir": "VNIR", "nir800": "VNIR_800nm",
              "nir1000": "VNIR_1000nm"}


def build_features() -> pd.DataFrame:
    man = pd.read_csv(P3 / "manifest_tomato.csv")
    inten = pd.read_csv(P3 / "intensities.csv")
    if inten["path"].duplicated().any():
        raise SystemExit("FATAL: intensities.csv has duplicate paths (HARD RULE 6)")
    by_path = inten.set_index("path")

    def col(manifest_col: str, stat: str) -> pd.Series:
        src = by_path.loc[man[manifest_col]]
        if src.index.has_duplicates:
            raise SystemExit("FATAL: manifest path not unique in intensities.csv")
        return src[stat].to_numpy()

    df = man[["plant_key", "label", "batch", "angle", "day",
              "rgb", "vnir", "nir800", "nir1000"]].copy()
    df["batch_bin"] = (df["batch"] == "II").astype(int)
    df["rgb_width"] = col("rgb", "width")
    df["rgb_height"] = col("rgb", "height")
    df["rgb_aspect"] = df.rgb_width / df.rgb_height
    df["rgb_bytes"] = col("rgb", "bytes")
    df["rgb_mean"] = col("rgb", "rgb_mean")
    df["rgb_red"] = col("rgb", "rgb_red")
    df["vnir_mean"] = col("vnir", "vnir_mean")
    df["vnir_red"] = col("vnir", "vnir_red")
    df["nir800_mean"] = col("nir800", "nir800_mean")
    df["nir800_red"] = col("nir800", "nir800_red")
    df["nir1000_mean"] = col("nir1000", "nir1000_mean")
    df["nir1000_red"] = col("nir1000", "nir1000_red")

    miss = df[META_COLS + INT_COLS].isna().sum()
    if miss.any():
        raise SystemExit(f"FATAL: NaN in features: {miss[miss > 0].to_dict()}")
    if len(df) != 4807 or df.plant_key.nunique() != 167:
        raise SystemExit(f"FATAL: population {len(df)} rows / {df.plant_key.nunique()} "
                         f"plants != audited 4807 / 167 (HARD RULE 1/6)")
    # label constant within plant (a plant cannot change class)
    per_plant = df.groupby("plant_key")["label"].nunique()
    if (per_plant > 1).any():
        raise SystemExit("FATAL: label varies within a plant (HARD RULE 6)")
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def make_plant_disjoint(df: pd.DataFrame) -> dict:
    man_path = OUT / "features_manifest.csv"
    # angle/day included because check_leakage.py is invoked with
    # --report-cols angle,batch,day (it groups by those columns after the assertion)
    df[["row_id", "plant_key", "label", "batch", "angle", "day"]].to_csv(man_path, index=False)
    out_dir = OUT / "plant_disjoint_splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "caep/split_by_plant.py", "--manifest", str(man_path),
           "--seeds", ",".join(map(str, SEEDS)), "--out-dir", str(out_dir),
           "--test-frac", str(TEST_FRAC), "--val-frac", str(VAL_FRAC)]
    r = subprocess.run(cmd, cwd=RESEARCH, capture_output=True, text=True)
    (OUT / "split_by_plant_stdout.txt").write_text(
        "$ " + " ".join(cmd) + "\n" + r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: split_by_plant.py failed: {(r.stdout + r.stderr)[-800:]}")
    checks = {}
    for seed in SEEDS:
        sp = out_dir / f"manifest_split_seed{seed}.csv"
        c = subprocess.run(
            [PY, "caep/check_leakage.py", "--manifest", str(sp),
             "--group-col", "plant_key", "--report-cols", "angle,batch,day"],
            cwd=RESEARCH, capture_output=True, text=True)
        (CHECKS / f"plant_disjoint_seed{seed}.txt").write_text(
            f"$ check_leakage.py (plant_disjoint, seed {seed})\nexit={c.returncode}\n"
            + c.stdout + c.stderr)
        checks[f"plant_disjoint_seed{seed}"] = c.returncode
        if c.returncode != 0:
            raise SystemExit(f"FATAL: plant-disjoint seed {seed} FAILED the project's "
                             f"leakage check (HARD RULE 4): {(c.stdout + c.stderr)[-600:]}")
    return {"plant_disjoint": str(out_dir), "checks": checks}


def make_image_level(df: pd.DataFrame) -> dict:
    """Group-blind row split - the deliberately wrong protocol for the demo."""
    out_dir = OUT / "image_level_splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for seed in SEEDS:
        rest, test = train_test_split(df, test_size=TEST_FRAC, stratify=df.label,
                                      random_state=seed)
        relf = VAL_FRAC / (1 - TEST_FRAC)
        train, val = train_test_split(rest, test_size=relf, stratify=rest.label,
                                      random_state=seed)
        d = df.copy()
        d["split"] = "train"
        d.loc[val.index, "split"] = "validation"
        d.loc[test.index, "split"] = "test"
        sp = out_dir / f"manifest_split_seed{seed}.csv"
        d[["row_id", "plant_key", "label", "batch", "angle", "day", "split"]].to_csv(sp, index=False)
        c = subprocess.run(
            [PY, "caep/check_leakage.py", "--manifest", str(sp),
             "--group-col", "plant_key", "--report-cols", "angle,batch,day"],
            cwd=RESEARCH, capture_output=True, text=True)
        (CHECKS / f"image_level_seed{seed}.txt").write_text(
            f"$ check_leakage.py (image_level DEMO, seed {seed})\n"
            f"exit={c.returncode}  (exit 1 here = plant leakage DETECTED = the point "
            f"of the demo)\n" + c.stdout + c.stderr)
        checks[f"image_level_seed{seed}"] = c.returncode
    return {"image_level": str(out_dir), "checks": checks}


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> dict:
    Xtr, ytr = train[feats].to_numpy(), train.label.to_numpy()
    Xte, yte = test[feats].to_numpy(), test.label.to_numpy()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, C=1.0))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    maj = int(np.bincount(ytr).argmax())
    return {
        "test_accuracy": round(float((pred == yte).mean()), 6),
        "test_auc": round(float(roc_auc_score(yte, proba)), 6),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 6),
        "majority_class_from_train": maj,
        "majority_baseline_test_accuracy": round(float((maj == yte).mean()), 6),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_plants_test": int(test.plant_key.nunique()),
        "coef_abs_max_feature": feats[int(np.abs(pipe.named_steps["lr"].coef_[0]).argmax())]}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    df = build_features()
    df.to_csv(OUT / "features_manifest.csv", index=False)

    paths = make_plant_disjoint(df)
    demo = make_image_level(df)

    rows, results = [], {"plant_disjoint": {}, "image_level": {}}
    for protocol, dirpath in (("plant_disjoint", paths["plant_disjoint"]),
                              ("image_level", demo["image_level"])):
        for seed in SEEDS:
            sp = pd.read_csv(Path(dirpath) / f"manifest_split_seed{seed}.csv")
            merged = df.merge(sp[["row_id", "split"]], on="row_id", validate="one_to_one")
            tr = merged[merged.split == "train"]
            te = merged[merged.split == "test"]
            for fs_name, feats in FEATURE_SETS.items():
                res = fit_eval(tr, te, feats)
                res.update({"protocol": protocol, "features": fs_name, "seed": seed})
                results[protocol].setdefault(fs_name, {})[str(seed)] = res
                rows.append({**res, "protocol": protocol, "features": fs_name,
                             "seed": seed})
    pd.DataFrame(rows).to_csv(OUT / "confound_baselines.csv", index=False)

    # code-computed headline: leakage inflation = image_level - plant_disjoint ----
    summary = {}
    for fs_name in FEATURE_SETS:
        pd_acc = [results["plant_disjoint"][fs_name][str(s)]["test_accuracy"]
                  for s in SEEDS]
        il_acc = [results["image_level"][fs_name][str(s)]["test_accuracy"]
                  for s in SEEDS]
        pd_auc = [results["plant_disjoint"][fs_name][str(s)]["test_auc"] for s in SEEDS]
        il_auc = [results["image_level"][fs_name][str(s)]["test_auc"] for s in SEEDS]
        summary[fs_name] = {
            "plant_disjoint_acc_mean": round(float(np.mean(pd_acc)), 6),
            "plant_disjoint_acc_sd": round(float(np.std(pd_acc, ddof=1)), 6),
            "image_level_acc_mean": round(float(np.mean(il_acc)), 6),
            "image_level_acc_sd": round(float(np.std(il_acc, ddof=1)), 6),
            "acc_inflation_image_level_minus_plant_disjoint":
                round(float(np.mean(il_acc) - np.mean(pd_acc)), 6),
            "plant_disjoint_auc_mean": round(float(np.mean(pd_auc)), 6),
            "image_level_auc_mean": round(float(np.mean(il_auc)), 6),
            "auc_inflation_image_level_minus_plant_disjoint":
                round(float(np.mean(il_auc) - np.mean(pd_auc)), 6)}

    out = {"phase": 4,
           "analysis": "B4 confound baselines + image-level vs plant-disjoint demo",
           "population": {"rows": int(len(df)), "plants": int(df.plant_key.nunique()),
                          "unit": "capture tuple", "source": "results/phase3/manifest_tomato.csv"},
           "feature_sets": FEATURE_SETS,
           "model": "StandardScaler + LogisticRegression(C=1, max_iter=2000); "
                    "no tuning, no feature selection (HARD RULE 3)",
           "seeds": SEEDS,
           "protocols": {"plant_disjoint": "project's caep/split_by_plant.py "
                                           "(test 0.20 / val 0.15)",
                         "image_level": "group-blind stratified row split (the "
                                        "leakage demo, deliberately wrong)"},
           "leakage_check_exit_codes": {**paths["checks"], **demo["checks"]},
           "leakage_check_interpretation":
               "plant_disjoint must be 0 (asserted, HARD RULE 4); image_level is "
               "expected 1 (plant leakage detected = the demonstration), recorded "
               "as evidence",
           "results": results, "summary": summary,
           "sources": ["results/phase3/manifest_tomato.csv",
                       "results/phase3/intensities.csv",
                       "tomatoguard/research/caep/split_by_plant.py",
                       "tomatoguard/research/caep/check_leakage.py"],
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "confound_baselines.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"summary": summary,
                      "leakage_exit_codes": out["leakage_check_exit_codes"]},
                     indent=2))
    print(f"wrote {OUT}/confound_baselines.json (+csv) in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
