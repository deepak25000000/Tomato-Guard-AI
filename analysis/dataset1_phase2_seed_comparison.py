"""GROUP A / item A2 - S1 seed variance BESIDE S2 seed variance (same recipe, same epochs).

  .venv/bin/python analysis/dataset1_phase2_seed_comparison.py

Reads only existing result jsons:
  S1 side: results/phase2/faithful_run1.json (seed 42), faithful_seed43.json,
           faithful_seed44.json  (+ faithful_run2.json as a determinism repeat of seed 42)
  S2 side: results/phase2/s2_seed{42,43,44}.json

Asserts both sides used the same epochs (read from the jsons) before comparing, and
records where each side's architecture is defined. Writes
results/phase2/seed_comparison_s1_vs_s2.{json,csv}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "results" / "phase2"
S1 = {"42": "faithful_run1.json", "43": "faithful_seed43.json", "44": "faithful_seed44.json"}
S2 = {"42": "s2_seed42.json", "43": "s2_seed43.json", "44": "s2_seed44.json"}


def load(rel: str) -> dict:
    return json.loads((P2 / rel).read_text())


def agg(per: dict) -> dict:
    acc = [v["accuracy"] for v in per.values()]
    f1 = [v["macro_f1"] for v in per.values()]
    return {"n_seeds": len(acc),
            "accuracy_mean": round(sum(acc) / len(acc), 6),
            "accuracy_sd": round(pd.Series(acc).std(ddof=1), 6),
            "macro_f1_mean": round(sum(f1) / len(f1), 6),
            "macro_f1_sd": round(pd.Series(f1).std(ddof=1), 6)}


def main() -> None:
    t0 = time.time()
    rows, per = [], {"S1": {}, "S2": {}}
    for side, files in (("S1", S1), ("S2", S2)):
        for seed, rel in files.items():
            d = load(rel)
            per[side][seed] = {"accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
                               "epochs": d["epochs"], "seed": d["seed"],
                               "best_val_accuracy": d.get("best_val_accuracy"),
                               "source": f"results/phase2/{rel}"}
            rows.append({"split": side, "seed": int(seed), "accuracy": d["accuracy"],
                         "macro_f1": d["macro_f1"], "epochs": d["epochs"],
                         "best_val_accuracy": d.get("best_val_accuracy"),
                         "source": f"results/phase2/{rel}"})
    ep1 = {v["epochs"] for v in per["S1"].values()}
    ep2 = {v["epochs"] for v in per["S2"].values()}
    if ep1 != ep2 or len(ep1) != 1:
        raise SystemExit(f"FATAL: epochs differ between sides: S1={ep1} S2={ep2} (HARD RULE 6)")
    a1, a2 = agg(per["S1"]), agg(per["S2"])
    run2 = load("faithful_run2.json")
    determinism = {"seed42_run1_vs_run2_accuracy": [per["S1"]["42"]["accuracy"],
                                                    run2["accuracy"]],
                   "bit_identical_accuracy": per["S1"]["42"]["accuracy"] == run2["accuracy"]}
    out = {"phase": 2, "analysis": "S1 vs S2 seed variance, same recipe",
           "recipe": {"epochs": list(ep1)[0],
                      "batch": 16, "optimizer": "AdamW lr=1e-4 wd=1e-4",
                      "model_selection": "best validation accuracy",
                      "arch": "mobilenet_v3_small (both sides)",
                      "arch_evidence": "S2: 'arch' field of s2_seed*.json; S1: "
                                       "models.mobilenet_v3_small in "
                                       "analysis/dataset1_phase2_faithful.py"},
           "S1_original_split": a1, "S2_group_disjoint": a2,
           "mean_difference_S1_minus_S2_accuracy":
               round(a1["accuracy_mean"] - a2["accuracy_mean"], 6),
           "determinism_check_seed42": determinism,
           "per_seed": per, "runtime_s": round(time.time() - t0, 1)}
    (P2 / "seed_comparison_s1_vs_s2.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(P2 / "seed_comparison_s1_vs_s2.csv", index=False)
    print(json.dumps({k: out[k] for k in ("S1_original_split", "S2_group_disjoint",
                                          "mean_difference_S1_minus_S2_accuracy",
                                          "determinism_check_seed42")}, indent=2))


if __name__ == "__main__":
    main()
