"""GROUP A / item A5 (part 2) - the missing spec items, all computed from existing files.

  .venv/bin/python analysis/dataset1_phase2_spec_extras.py

1. majority-class baseline (S1 and S2): majority class of TRAIN predicted for TEST.
2. numpy-only metric recomputation: accuracy / macro-F1 / macro OvR AUC recomputed with
   PURE NUMPY from the saved per-image probabilities and compared with the sklearn
   values in the corresponding result jsons (no sklearn imported here).
3. ECE before/after temperature scaling for the PUBLISHED checkpoint: temperature fit
   by NLL on the S1 VALIDATION split only (HARD RULE 3), ECE (15 equal-width bins)
   reported on S1 test before and after; accuracy is unchanged by construction
   (scaling does not move argmax) and that is asserted.
4. error gallery: Early Blight <-> Leaf Miner Flies confusions on S1 test
   (published checkpoint) and on S2 test (arch winner), as json lists + montage pngs.
5. excl-48 comparison: baseline S1 seeds vs excl48 seeds (both from result jsons).
6. threshold sweep table: merges t=3 (existing s2_seed*.json), t=4, t=6 rows into
   results/phase2/threshold_sweep_all.csv so the trend can be reported in one table.

Writes results/phase2/{majority_baseline,numpy_metric_check,temperature_scaling,
error_gallery_eb_lmf,excl48_comparison}.json + error_gallery_*.png +
threshold_sweep_all.csv. Every number is read from a file (HARD RULE 1).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P1, P2 = RES / "phase1", RES / "phase2"
SPLITS = {"S1": RES / "splits_S1", "S2": RES / "splits_S2"}
ORIG_CKPT = Path.home() / "Downloads" / "Tomato_Research_Results" / "best_mobilenetv3.pth"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))

EB, LMF = "Early Blight", "Leaf Miner Flies"
ECE_BINS = 15


# ---------------------------------------------------------------- 1. majority ----
def majority_baseline() -> dict:
    out = {}
    for name, d in SPLITS.items():
        tr = pd.read_csv(d / "train.csv")
        te = pd.read_csv(d / "test.csv")
        counts = tr["class"].value_counts()
        maj = counts.index[0]
        acc = float((te["class"] == maj).mean())
        out[name] = {"majority_class": str(maj), "train_counts": {str(k): int(v) for k, v in counts.items()},
                     "n_test": int(len(te)), "test_accuracy": round(acc, 6),
                     "source": [f"results/{d.name}/train.csv", f"results/{d.name}/test.csv"]}
    return {"phase": 2, "analysis": "majority-class baseline",
            "definition": "predict the most frequent TRAIN class for every test image",
            "S1_original_split": out["S1"], "S2_group_disjoint": out["S2"]}


# ---------------------------------------------------------- 2. numpy metrics ----
def np_acc(y: np.ndarray, pred: np.ndarray) -> float:
    return float((y == pred).mean())


def np_macro_f1(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    f1s = []
    for k in range(n):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def np_auc_binary(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties, pure numpy."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1      # average rank, 1-based
        i = j + 1
    n1, n2 = len(pos), len(neg)
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n2))


def np_macro_auc(y: np.ndarray, probs: np.ndarray) -> float:
    n = probs.shape[1]
    aucs = []
    for k in range(n):
        aucs.append(np_auc_binary(probs[y == k, k], probs[y != k, k]))
    return float(np.mean(aucs))


def numpy_metric_check() -> dict:
    pairs = [("faithful_original_preds.csv", "faithful_original.json"),
             ("s2_seed42_preds.csv", "s2_seed42.json"),
             ("s2_seed43_preds.csv", "s2_seed43.json"),
             ("s2_seed44_preds.csv", "s2_seed44.json"),
             ("arch_winner_test_preds.csv", "arch_winner_test.json")]
    files, max_diff = {}, 0.0
    for preds_rel, metrics_rel in pairs:
        df = pd.read_csv(P2 / preds_rel)
        m = json.loads((P2 / metrics_rel).read_text())
        y = df["y_true"].to_numpy()
        pred = df["y_pred"].to_numpy()
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        probs = df[prob_cols].to_numpy()
        n = len(prob_cols)
        recalc = {"accuracy": np_acc(y, pred), "macro_f1": np_macro_f1(y, pred, n),
                  "macro_roc_auc": np_macro_auc(y, probs)}
        stored = {k: float(m[k]) for k in recalc}
        diffs = {k: abs(recalc[k] - stored[k]) for k in recalc}
        max_diff = max(max_diff, *diffs.values())
        files[preds_rel] = {"metrics_file": f"results/phase2/{metrics_rel}",
                            "recomputed_numpy": {k: round(v, 10) for k, v in recalc.items()},
                            "stored_sklearn": stored,
                            "abs_diff": {k: round(v, 12) for k, v in diffs.items()},
                            "n_images": int(len(df))}
    return {"phase": 2, "analysis": "numpy-only recomputation of stored metrics",
            "method": "pure numpy (no sklearn): accuracy, macro-F1 from the confusion "
                      "counts, macro OvR AUC via Mann-Whitney with average ranks for ties",
            "per_file": files, "max_abs_diff": max_diff,
            "tolerance": 1e-6,
            "pass": bool(max_diff <= 1e-6)}


# ------------------------------------------------- 3. ECE + temperature scale ----
def ece(y: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    val = 0.0
    for b in range(n_bins):
        mask = (conf > edges[b]) & (conf <= edges[b + 1]) if b else \
               (conf >= edges[b]) & (conf <= edges[b + 1])
        if mask.sum():
            val += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(val)


def temperature_scaling() -> dict:
    import torch
    import torch.nn as nn
    from torchvision import models
    from trackA_lib import EVAL_TF, LeafDataset, predict, seed_all
    from torch.utils.data import DataLoader

    seed_all(42)
    classes = json.loads((SPLITS["S1"] / "classes.json").read_text())
    c2i = {c: i for i, c in enumerate(classes)}
    dev = torch.device("cpu")
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    sd = torch.load(ORIG_CKPT, map_location="cpu", weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit(f"FATAL: checkpoint mismatch missing={miss} unexpected={unexp}")
    model.eval()

    def probs_of(split: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(SPLITS["S1"] / f"{split}.csv")
        loader = DataLoader(LeafDataset(df, c2i, EVAL_TF), batch_size=16,
                            shuffle=False, num_workers=0)
        lg, y = predict(model, loader, dev)
        z = lg.numpy() if hasattr(lg, "numpy") else np.asarray(lg)  # predict may return either
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), np.asarray(y)

    p_val, y_val = probs_of("validation")
    p_te, y_te = probs_of("test")
    # sanity: test accuracy must equal the audited faithful_original.json value
    stored_acc = json.loads((P2 / "faithful_original.json").read_text())["accuracy"]
    acc_now = float((p_te.argmax(1) == y_te).mean())
    if abs(acc_now - stored_acc) > 1e-9:
        raise SystemExit(f"FATAL: re-evaluated test acc {acc_now} != stored {stored_acc}")

    from scipy.optimize import minimize_scalar
    z_val = np.log(np.clip(p_val, 1e-12, 1.0))      # logits up to an additive constant

    def nll(log_t: float) -> float:
        t = np.exp(log_t)
        lz = z_val / t
        lz -= lz.max(1, keepdims=True)
        p = np.exp(lz)
        p /= p.sum(1, keepdims=True)
        return float(-np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)).mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    t_star = float(np.exp(res.x))
    z_te = np.log(np.clip(p_te, 1e-12, 1.0)) / t_star
    z_te -= z_te.max(1, keepdims=True)
    p_te_t = np.exp(z_te)
    p_te_t /= p_te_t.sum(1, keepdims=True)

    ece_before = ece(y_te, p_te)
    ece_after = ece(y_te, p_te_t)
    acc_after = float((p_te_t.argmax(1) == y_te).mean())
    if abs(acc_after - acc_now) > 1e-12:
        raise SystemExit("FATAL: temperature scaling changed argmax (impossible) - bug")
    return {"phase": 2, "analysis": "ECE before/after temperature scaling",
            "checkpoint": "user's original best_mobilenetv3.pth (S1-trained)",
            "temperature_fit_split": "S1 validation (HARD RULE 3: test not used to fit)",
            "n_val": int(len(y_val)), "n_test": int(len(y_te)),
            "n_classes": len(classes), "ece_bins": ECE_BINS,
            "temperature": round(t_star, 6),
            "ece_before": round(ece_before, 6), "ece_after": round(ece_after, 6),
            "test_accuracy_before": round(acc_now, 6),
            "test_accuracy_after": round(acc_after, 6),
            "accuracy_unchanged_by_scaling": True,
            "method": "T minimises NLL on validation (scipy bounded scalar search over "
                      "log T in [log 0.05, log 20]); ECE = 15 equal-width confidence bins",
            "sources": ["results/phase2/faithful_original.json",
                        "results/splits_S1/validation.csv", "results/splits_S1/test.csv"]}


# ------------------------------------------------------------ 4. error gallery ----
def error_gallery() -> dict:
    from PIL import Image
    panels = {}
    for key, preds_rel, metrics_rel in (
            ("S1_test_published_checkpoint", "faithful_original_preds.csv",
             "faithful_original.json"),
            ("S2_test_arch_winner", "arch_winner_test_preds.csv",
             "arch_winner_test.json")):
        df = pd.read_csv(P2 / preds_rel)
        # map class indices via the json's class order
        m = json.loads((P2 / metrics_rel).read_text())
        classes = list(m["class_support"].keys())
        eb_i, lmf_i = classes.index(EB), classes.index(LMF)
        swaps = df[((df.y_true == eb_i) & (df.y_pred == lmf_i)) |
                   ((df.y_true == lmf_i) & (df.y_pred == eb_i))]
        rows = [{"path": r.path, "true": classes[int(r.y_true)],
                 "predicted": classes[int(r.y_pred)]}
                 for r in swaps.itertuples()]
        # montage of up to 24 thumbnails
        thumbs, CAP = [], 24
        for r in rows[:CAP]:
            im = Image.open(r["path"]).convert("RGB")
            im.thumbnail((160, 160))
            thumbs.append((im, r))
        if thumbs:
            cols = 4
            rows_n = (len(thumbs) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * 170, rows_n * 190), "white")
            for i, (im, r) in enumerate(thumbs):
                x, y = (i % cols) * 170 + 5, (i // cols) * 190 + 5
                sheet.paste(im, (x, y))
            out_png = P2 / f"error_gallery_{key}.png"
            sheet.save(out_png)
        panels[key] = {
            "metrics_file": f"results/phase2/{metrics_rel}",
            "n_eb_to_lmf": int(((df.y_true == eb_i) & (df.y_pred == lmf_i)).sum()),
            "n_lmf_to_eb": int(((df.y_true == lmf_i) & (df.y_pred == eb_i)).sum()),
            "n_total_swaps": int(len(rows)),
            "montage": f"results/phase2/error_gallery_{key}.png (first {min(CAP, len(rows))} of {len(rows)})",
            "images_in_order": rows}
    return {"phase": 2, "analysis": "Early Blight <-> Leaf Miner Flies error gallery",
            "definition": "test images whose prediction is the other class of the pair",
            "panels": panels}


# --------------------------------------------------------- 5. excl48 compare ----
def excl48_comparison() -> dict:
    def side(prefix: str, files: list[str]) -> dict:
        per = {}
        for f in files:
            d = json.loads((P2 / f).read_text())
            per[f] = {"seed": d["seed"], "accuracy": d["accuracy"],
                      "macro_f1": d["macro_f1"]}
        acc = [v["accuracy"] for v in per.values()]
        f1 = [v["macro_f1"] for v in per.values()]
        return {"per_seed": per,
                "accuracy_mean": round(float(np.mean(acc)), 6),
                "accuracy_sd": round(float(np.std(acc, ddof=1)), 6),
                "macro_f1_mean": round(float(np.mean(f1)), 6),
                "macro_f1_sd": round(float(np.std(f1, ddof=1)), 6)}
    base = side("base", ["faithful_run1.json", "faithful_seed43.json",
                         "faithful_seed44.json"])
    excl = side("excl", ["excl48_seed42.json", "excl48_seed43.json",
                         "excl48_seed44.json"])
    report = json.loads((RES / "splits_S1_excl48" / "report.json").read_text())
    return {"phase": 2, "analysis": "S1 re-trained excluding the 48 high-confidence "
                                    "cross-label images",
            "exclusion": report,
            "baseline_S1": base, "excluded_S1": excl,
            "change_accuracy_mean": round(excl["accuracy_mean"] - base["accuracy_mean"], 6),
            "change_macro_f1_mean": round(excl["macro_f1_mean"] - base["macro_f1_mean"], 6),
            "note": "same recipe, 3 seeds per side; positive change = removing the "
                    "conflicting images helped"}


# ------------------------------------------------------- 6. threshold sweep ------
def threshold_sweep_all() -> str:
    rows = []
    for t, files in ((3, [f"s2_seed{s}.json" for s in (42, 43, 44)]),
                     (4, [f"thresh4_seed{s}.json" for s in (42, 43, 44)]),
                     (6, [f"thresh6_seed{s}.json" for s in (42, 43, 44)])):
        for f in files:
            d = json.loads((P2 / f).read_text())
            rows.append({"threshold": t, "seed": d["seed"], "arch": d["arch"],
                         "accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
                         "best_val_accuracy": d["best_val_accuracy"],
                         "n_test": d["n_test"], "source": f"results/phase2/{f}"})
    df = pd.DataFrame(rows)
    df.to_csv(P2 / "threshold_sweep_all.csv", index=False)
    agg = df.groupby("threshold").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        n_seeds=("accuracy", "count")).reset_index()
    agg.round(6).to_csv(P2 / "threshold_sweep_all_aggregate.csv", index=False)

    # trend: does accuracy keep falling as grouping loosens? (computed, not asserted)
    means = {int(r.threshold): float(r.accuracy_mean) for r in agg.itertuples()}
    ref = 3
    deltas = {str(t): round(m - means[ref], 6) for t, m in sorted(means.items())}
    ordered = [means[t] for t in sorted(means)]
    monotonic = all(b <= a for a, b in zip(ordered, ordered[1:]))
    direction = ("monotonically non-increasing with looser grouping" if monotonic else
                 "NOT monotonic across the tested thresholds")
    trend = {"analysis": "near-dup threshold sweep trend (Group A3)",
             "reference_threshold": ref,
             "mean_accuracy_by_threshold": {str(t): round(m, 6)
                                            for t, m in sorted(means.items())},
             "delta_mean_acc_vs_reference": deltas,
             "monotonic_non_increasing": monotonic,
             "direction_statement": (f"mean test accuracy across 3 seeds by threshold: "
                                     + ", ".join(f"t={t}: {round(m, 6)}"
                                                 for t, m in sorted(means.items()))
                                     + f" - {direction}"),
             "n_thresholds_tested": len(means), "seeds_per_threshold": 3,
             "source": "results/phase2/threshold_sweep_all.csv"}
    (P2 / "threshold_sweep_trend.json").write_text(json.dumps(trend, indent=2))
    return "results/phase2/threshold_sweep_all.csv"


def main() -> None:
    t0 = time.time()
    out = {}
    out["majority_baseline"] = majority_baseline()
    (P2 / "majority_baseline.json").write_text(json.dumps(out["majority_baseline"],
                                                          indent=2))
    out["numpy_metric_check"] = numpy_metric_check()
    (P2 / "numpy_metric_check.json").write_text(json.dumps(out["numpy_metric_check"],
                                                           indent=2))
    if not out["numpy_metric_check"]["pass"]:
        raise SystemExit("FATAL: numpy recomputation disagrees with stored metrics "
                         f"(max abs diff {out['numpy_metric_check']['max_abs_diff']})")
    out["temperature_scaling"] = temperature_scaling()
    (P2 / "temperature_scaling.json").write_text(json.dumps(out["temperature_scaling"],
                                                            indent=2))
    out["error_gallery"] = error_gallery()
    (P2 / "error_gallery_eb_lmf.json").write_text(json.dumps(out["error_gallery"],
                                                             indent=2))
    out["excl48_comparison"] = excl48_comparison()
    (P2 / "excl48_comparison.json").write_text(json.dumps(out["excl48_comparison"],
                                                          indent=2))
    out["threshold_sweep_csv"] = threshold_sweep_all()
    out["runtime_s"] = round(time.time() - t0, 1)
    (P2 / "spec_extras_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in list(v.items())[:6]})
                      for k, v in out.items()}, indent=2)[:3000])


if __name__ == "__main__":
    main()
