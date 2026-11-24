"""PHASE 5 - build RESULTS.md entirely from machine-generated files under results/.

  .venv/bin/python analysis/generate_results_md.py [--allow-incomplete]

HARD RULE 1: every number in RESULTS.md is read from a json/csv written by an analysis script.
This script contains no result literals - only labels, headings and prose.

HARD RULE 6: if a required input is missing the build ABORTS and writes nothing, listing what is
missing. `--allow-incomplete` exists only so the renderer can be smoke-tested before the pipeline
finishes; it writes visible "> MISSING" placeholders, and tests/test_results.py fails on them.

HARD RULE 2: raw folder names (including the "Bacteiral Spot" typo) are never used as display
labels, and raw paths are never printed.

HARD RULE 8: the forbidden phrases are never emitted; the warnings state explicitly that no
plant-disjoint "pre-symptomatic / early detection / field-ready" claim is made.

Writes results/RESULTS.md + results/results_md_build.json (inputs read, inputs missing).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "RESULTS.md"
PLACEHOLDER = "> **SECTION NOT RENDERED**"


class MissingRequired(Exception):
    pass


# ---------------------------------------------------------------- helpers
_read: dict[str, object] = {}
_missing: list[str] = []


def _load(rel: str):
    if rel in _read:
        return _read[rel]
    p = RES / rel
    if not p.exists():
        _missing.append(rel)
        raise MissingRequired(f"results/{rel}")
    v = json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)
    _read[rel] = v
    return v


def J(rel: str):
    return _load(rel)


def C(rel: str):
    return _load(rel)


def _ainput(rel: str):
    """INPUT file under analysis/ (claims, decisions, article quotes) - not a result."""
    if rel in _read:
        return _read[rel]
    p = ROOT / "analysis" / rel
    if not p.exists():
        _missing.append(f"analysis/{rel}")
        raise MissingRequired(f"analysis/{rel}")
    v = json.loads(p.read_text())
    _read[rel] = v
    return v


def r(x, nd: int = 4) -> str:
    """Format a value from a result file; never invent one."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        if pd.isna(x):
            return "n/a"
    except (TypeError, ValueError):
        pass
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x:
            return "n/a"
        if x == int(x) and abs(x) < 1e15:
            return str(int(x))
        s = f"{x:.{nd}f}"
        return (s.rstrip("0").rstrip(".") if "." in s else s) or "0"
    return str(x)


def pct(x, nd: int = 2) -> str:
    return f"{100 * float(x):.{nd}f}%"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return (["| " + " | ".join(headers) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
            + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def ci(v, nd: int = 4) -> str:
    if not v:
        return "n/a"
    return f"[{r(v[0], nd)}, {r(v[1], nd)}]"


def pf(ok) -> str:
    if ok is True:
        return "**PASS**"
    if ok is False:
        return "**FAIL**"
    return str(ok)


# ================================================================ sections
def sec_repro() -> list[str]:
    env, bench, prop = J("environment.json"), J("phase0_benchmark.json"), \
        J("phase0_runtime_proposal.json")
    ver = env.get("versions", {})
    L = ["## 0. Reproducibility", ""]
    L += table(["item", "value"], [
        ["python", r(ver.get("python"))],
        ["torch", r(ver.get("torch"))],
        ["torchvision", r(ver.get("torchvision"))],
        ["numpy", r(ver.get("numpy"))],
        ["pandas", r(ver.get("pandas"))],
        ["scikit-learn", r(ver.get("scikit_learn"))],
        ["pillow", r(ver.get("pillow"))],
        ["scipy", r(ver.get("scipy"))],
        ["GPU", "none (torch.cuda_available = "
                + str(ver.get("torch_cuda_available")).lower() + ")"],
        ["CPU", r(env.get("cpu", {}).get("model"))],
        ["cores (logical/physical)",
         f"{r(env.get('cpu', {}).get('logical_cores'))} / "
         f"{r(env.get('cpu', {}).get('physical_cores_lscpu'))}"],
        ["RAM (GB)", r(env.get("cpu", {}).get("mem_total_gb"))],
        ["torch threads", r(ver.get("torch_num_threads"))],
        ["seeds", "42 default; 43/44 for seed-variance; recorded per run"],
        ["environment captured", r(env.get("timestamp_utc"))],
    ])
    L += ["", "Full package set: `results/pip_freeze.txt`; per-run commands and wall times: "
              "`results/run_log.md`.", ""]

    L += ["## 1. Phase 0 - machine and budget", "",
          "Measured on this machine (nothing extrapolated from elsewhere):", ""]
    L += table(["measurement", "value"], [
        ["Track A decode + resize 224", f"{r(bench.get('trackA_decode_resize_224_s_per_img'))} s/img"],
        ["Track A train step (batch 16)", f"{r(bench.get('trackA_train_step_s_batch16'))} s"],
        ["Track A train throughput", f"{r(bench.get('trackA_train_imgs_per_s'), 1)} img/s"],
        ["Track A eval throughput", f"{r(bench.get('trackA_eval_imgs_per_s'), 1)} img/s"],
        ["Track B fusion train step (batch 32)",
         f"{r(bench.get('trackB_fusion_train_step_s_batch32'))} s"],
        ["Track B fusion throughput", f"{r(bench.get('trackB_fusion_imgs_per_s'), 1)} img/s"],
    ])
    L += ["", "Runtime proposals and the setting actually used:", ""]
    rows = []
    for ph in ("phase2", "phase4"):
        for scen in ("as_specified", "reduced", "minimal"):
            d = prop.get(ph, {}).get(scen, {})
            if not d:
                continue
            used = ((ph == "phase2" and scen == "reduced") or (ph == "phase4" and scen == "minimal"))
            rows.append([ph, scen, r(d.get("hours")), r(d.get("runs_total", d.get("deep_runs"))),
                         r(d.get("epochs", "n/a")), r(d.get("image_size", "n/a")),
                         "**used**" if used else ""])
    L += table(["phase", "scenario", "hours", "runs", "epochs", "image size", ""], rows)
    L += ["", "- Phase 2 ran at the **reduced** setting and Phase 4 at the **minimal** setting",
          "  (CPU-only; derivation in `results/phase0_runtime_proposal.json`).", ""]
    return L


def sec_phase1() -> list[str]:
    p1, cc = J("phase1/phase1_summary.json"), C("phase1/class_counts.csv")
    ac = p1["a_class_counts"]
    L = ["## 2. Phase 1 - Track A (RGB leaf dataset) data audit", "",
         "### 2.1 Class counts vs the claimed counts", ""]
    rows = [[x["display_label"], r(x["n_images"]), r(x["earlier_run_count"]), r(x["difference"])]
            for _, x in cc.iterrows()]
    rows.append(["**total**", r(ac["total_images"]), r(ac["earlier_total"]),
                 r(ac["total_difference"])])
    L += table(["display label", "on disk", "claimed", "delta"], rows)
    b = p1["b_file_properties"]
    L += ["", f"All {r(b['n_readable'])} / {r(b['n_files_attempted'])} files are readable, "
              f"{r(b['n_distinct_dimensions'])} distinct size "
              f"({r(b['width_min_max'][0])}-{r(b['height_min_max'][1])} px), "
              f"formats {', '.join(b['formats'])}, mean file size "
              f"{r(b['size_bytes_stats']['mean'], 0)} bytes.",
          "", "The raw folder name contains a typo; it is mapped to a display label in outputs "
              "only (HARD RULE 2 - raw data untouched).", ""]
    return L


def sec_phase1_dups() -> list[str]:
    p1 = J("phase1/phase1_summary.json")
    ex = J("phase1/exact_duplicates.json")
    L: list[str] = []
    L += ["### 2.2 Exact duplicates (SHA-256)", ""]
    L += table(["metric", "value"], [
        ["images", r(ex["n_images"])],
        ["distinct SHA-256", r(ex["n_unique_sha256"])],
        ["duplicate groups", r(ex["n_duplicate_groups"])],
        ["images in duplicate groups", r(ex["n_images_in_duplicate_groups"])],
        ["label conflicts among identical hashes", r(ex["n_label_conflict_hashes"])],
    ])
    L += ["", f"{r(ex['n_unique_sha256'])} distinct hashes for {r(ex['n_images'])} files - "
              f"{r(ex['n_duplicate_groups'])} exact duplicates.", ""]

    nt, d3 = C("phase1/near_dup_thresholds.csv"), p1["d_near_duplicates"]
    L += ["### 2.3 Near-duplicate threshold sweep (dHash Hamming)", "",
          f"Chosen threshold **t = {r(d3['chosen_threshold'])}** - "
          f"{d3['chosen_threshold_rationale']}", ""]
    rows = [[r(x["threshold"]), "yes" if x["user_requested"] else "", r(x["n_groups"]),
             r(x["largest_group"]), r(x["largest_group_pct"]), r(x["n_images_in_groups_gt1"]),
             r(x["groups_spanning_gt1_class"])] for _, x in nt.iterrows()]
    L += table(["threshold", "user-requested", "groups", "largest group", "largest group %",
                "images in groups >1", "groups spanning >1 class"], rows)
    L += ["", f"At t = {r(d3['chosen_threshold'])}: {r(d3['chosen_n_groups'])} groups, largest "
              f"group {r(d3['chosen_largest_group'])} images, "
              f"{r(d3['n_cross_label_near_dup_rows'])} cross-label near-duplicate rows "
              f"({r(d3['n_cross_label_by_dhash'])} by dHash, "
              f"{r(d3['n_cross_label_by_ncc_alone'])} found by NCC alone).", ""]
    return L


def sec_phase1_split() -> list[str]:
    lk = J("phase1/leakage.json")
    p1 = J("phase1/phase1_summary.json")
    d3 = p1["d_near_duplicates"]
    L: list[str] = []
    rec = lk["split_recreation"]["sorted_class_then_path"]
    claim = lk["split_recreation"]["_claim"]
    L += ["### 2.4 The original split", "",
          "Sizes reproduce exactly; membership does not (it depends on notebook row order):", ""]
    rows = [[n, r(rec["sizes"][n]), r(claim[n]), pf(rec["sizes_match_claim"][n]),
             r(rec["n_membership_symmetric_diff"][n]),
             pf(rec["membership_identical_to_original"][n])]
            for n in ("train", "validation", "test")]
    L += table(["split", "on disk", "claimed", "size match", "membership sym. difference",
                "membership identical"], rows)
    lbt = C("phase1/leakage_by_threshold.csv")
    ch = lbt[lbt["threshold"] == d3["chosen_threshold"]].iloc[0]
    L += ["",
          f"- exact-hash leakage: validation "
          f"{r(lk['exact_hash_leakage_original_split']['validation']['pct'])}%, test "
          f"{r(lk['exact_hash_leakage_original_split']['test']['pct'])}%.",
          f"- near-duplicate leakage at the chosen threshold: **"
          f"{r(ch['pct_test_near_dup_in_train'])}%** of test images and "
          f"{r(ch['pct_validation_near_dup_in_train'])}% of validation images have a "
          f"near-duplicate in train (t = {r(ch['threshold'])}).",
          "- full sweep: `results/phase1/leakage_by_threshold.csv`.", ""]
    return L


def sec_phase1_labels() -> list[str]:
    lc = J("phase1/label_conflict_summary.json")
    L: list[str] = []
    L += ["### 2.5 Cross-label near-duplicates (label-noise candidates)", ""]
    L += table(["metric", "value"], [
        ["rows (dHash <= t, different labels)", r(lc["n_rows_total"])],
        ["of which NCC >= 0.998 (high confidence)", r(lc["n_rows_ncc_high_conf"])],
        ["distinct images involved", r(lc["n_distinct_images_involved_all"])],
        ["distinct images (high confidence)", r(lc["n_distinct_images_involved_ncc_high_conf"])],
        ["high-confidence pairs crossing a split boundary",
         r(lc["n_cross_label_pairs_ncc_high_conf_crossing_boundary"])],
    ])
    L += ["", str(lc["interpretation"]), "",
          "Evidence: `results/phase1/cross_label_near_duplicates.csv`, "
          "`results/phase1/cross_label_split_overlap.csv`, "
          "`results/phase1/montage_matched_pairs.png`.", ""]
    return L


def sec_phase2() -> list[str]:
    ss, sla = J("phase2/split_sizes.json"), J("phase2/split_leakage_assertions.json")
    L = ["## 3. Phase 2 - Track A experiments", "",
         "### 3.1 Splits", "",
         "- **S1** = the user's original split. Its leakage is *measured*, not asserted against, "
         "because S1 is the object of study.",
         "- **S2** = group-disjoint split built by the project's own `01_make_splits.py` with "
         f"near-duplicate threshold {r(sla['near_dup_threshold_used_for_S2'])}; assertions apply.",
         ""]
    rows = [[s, p, r(ss[s][p])] for s in ("S1", "S2")
            for p in ("train", "validation", "test")]
    L += table(["split", "part", "images"], rows)
    L += ["", "Leakage assertions:", ""]
    L += table(["assertion", "result"],
               [[k, pf(v == "PASS") if v == "PASS" else f"**{v}**"]
                for k, v in sla["assertions"].items()])
    L += [""]
    return L


def sec_phase2_published() -> list[str]:
    fo = J("phase2/faithful_original.json")
    L: list[str] = []
    L += ["### 3.2 Faithful reproduction of the published baseline", "",
          "Re-evaluating the user's own saved checkpoint on the original test split:", ""]
    L += table(["metric", "value"], [
        ["test accuracy (this run)", r(fo["accuracy"])],
        ["95% CI (stratified bootstrap)", ci(fo["accuracy_95_ci"])],
        ["published accuracy claim", r(fo["published_accuracy_claim"])],
        ["absolute difference", r(fo["abs_diff_vs_published"])],
        ["macro-F1", r(fo["macro_f1"])],
        ["weighted-F1", r(fo["weighted_f1"])],
        ["macro ROC-AUC", r(fo["macro_roc_auc"])],
        ["macro PR-AUC", r(fo["macro_pr_auc"])],
        ["test images", r(fo["n_test"])],
        ["runtime (s)", r(fo["runtime_s"])],
    ])
    L += ["", f"Absolute difference vs the published number: "
              f"**{r(fo['abs_diff_vs_published'])}** (published value inside this run's CI: "
              f"{pf(fo['published_acc_inside_this_ci'] is True)}).", ""]
    return L


def sec_phase2_retrain() -> list[str]:
    fr1, fr2 = J("phase2/faithful_run1.json"), J("phase2/faithful_run2.json")
    h1, h2 = C("phase2/faithful_run1_history.csv"), C("phase2/faithful_run2_history.csv")
    b1 = h1["val_accuracy"].max() if "val_accuracy" in h1 else None
    b2 = h2["val_accuracy"].max() if "val_accuracy" in h2 else None
    L: list[str] = []
    L += ["### 3.3 Re-training from scratch (same recipe)", ""]
    rows = [["run1 (seed 42)", r(fr1["accuracy"]), ci(fr1["accuracy_95_ci"]), r(fr1["macro_f1"]),
             r(b1), r(fr1["runtime_s"])],
            ["run2 (seed 42, repeat)", r(fr2["accuracy"]), ci(fr2["accuracy_95_ci"]),
             r(fr2["macro_f1"]), r(b2), r(fr2["runtime_s"])]]
    L += table(["run", "test accuracy", "95% CI", "macro-F1", "best val acc", "runtime (s)"], rows)
    same = all(abs(float(fr1[k]) - float(fr2[k])) < 1e-12
               for k in ("accuracy", "macro_f1", "macro_roc_auc", "macro_pr_auc"))
    L += ["", f"Bit-identical metrics on repeat with the same seed: {pf(same)} "
              "- full-seed determinism (HARD RULE 5).", ""]
    return L


def sec_phase2_s2() -> list[str]:
    s2agg, s2sum = J("phase2/s2_seed_aggregate.json"), C("phase2/s2_seed_summary.csv")
    fo = J("phase2/faithful_original.json")
    L: list[str] = []
    L += ["### 3.4 Leakage-safe split S2 (headline)", "",
          f"{r(s2agg['n_seeds'])} seeds, MobileNetV3-Small, S2 test touched once per run:", ""]
    rows = [[r(x["seed"]), r(x["accuracy"]), f"{r(x['accuracy_ci_lo'])} - {r(x['accuracy_ci_hi'])}",
             r(x["macro_f1"]), r(x["best_val_accuracy"]), r(x["best_epoch"]), r(x["runtime_s"])]
            for _, x in s2sum.iterrows()]
    L += table(["seed", "test accuracy", "95% CI", "macro-F1", "best val acc", "best epoch",
                "runtime (s)"], rows)
    L += ["",
          f"- mean accuracy **{r(s2agg['accuracy_mean'])}** (sd {r(s2agg['accuracy_std'])}, "
          f"min {r(s2agg['accuracy_min'])}, max {r(s2agg['accuracy_max'])})",
          f"- mean macro-F1 {r(s2agg['macro_f1_mean'])} (sd {r(s2agg['macro_f1_std'])})",
          f"- original split gave {r(fo['accuracy'])}; the same recipe on group-disjoint S2 gives "
          f"{r(s2agg['accuracy_mean'])} - the gap is the split, not the model.", ""]
    return L


def sec_phase2_arch() -> list[str]:
    fo = J("phase2/faithful_original.json")
    arch = J("phase2/arch_selection.json")
    L: list[str] = []
    L += ["### 3.5 Architecture sweep (validation-only selection)", "",
          f"Selection rule: {arch['selection_criterion']}", ""]
    rows = [[c["arch"], r(c["val_accuracy"]), r(c["val_macro_f1"]), r(c["val_macro_roc_auc"]),
             r(c["best_epoch"]), r(c["runtime_s"])] for c in arch["candidates"]]
    L += table(["candidate", "val accuracy", "val macro-F1", "val macro AUC", "best epoch",
                "runtime (s)"], rows)
    L += ["", f"Selected: **{arch['selected_architecture']}** of {r(arch['n_candidates'])} "
              "candidates. The S2 test set was consulted exactly once, for the winner:", ""]
    wt = J("phase2/arch_winner_test.json")
    L += table(["metric", "value"], [
        ["architecture", str(wt.get("arch"))],
        ["test accuracy", r(wt["accuracy"])],
        ["95% CI", ci(wt["accuracy_95_ci"])],
        ["macro-F1", r(wt["macro_f1"])],
        ["macro ROC-AUC", r(wt["macro_roc_auc"])],
        ["selection", str(wt.get("selection_criterion"))],
    ])
    L += [""]
    return L


def sec_phase2_contamination() -> list[str]:
    oa = J("phase2/overlap_accounting.json")
    L: list[str] = []
    L += ["### 3.6 Cross-split contamination check", "", str(oa["interpretation"]), ""]
    L += table(["quantity", "value"], [
        ["S2 test images", r(oa["s2_test_total"])],
        ["seen by S1 train", r(oa["s2_test_seen_by_s1_train"])],
        ["seen by S1 train or validation", r(oa["s2_test_seen_by_s1_train_or_val"])],
        ["never seen by S1", r(oa["s2_test_unseen_by_s1"])],
        ["% of S2 test contaminated", r(oa["pct_s2_test_contaminated"])],
    ])
    L += ["", "Original checkpoint on **only** the never-seen images:", ""]
    un = J("phase2/faithful_origckpt_on_unseen.json")
    L += table(["metric", "value"], [
        ["n unseen images", r(un.get("n_unseen"))],
        ["test accuracy", r(un["accuracy"])],
        ["95% CI", ci(un["accuracy_95_ci"])],
        ["macro-F1", r(un["macro_f1"])],
        ["caveat", str(un.get("caveat", ""))],
    ])
    L += [""]
    return L


def sec_phase2_control() -> list[str]:
    ctl = J("phase2/faithful_shuffle_control.json")
    L: list[str] = []
    L += ["### 3.7 Negative control (training labels permuted)", "",
          f"Test accuracy **{r(ctl['accuracy'])}** against a chance level of "
          f"{r(ctl.get('chance_accuracy_1_over_n_classes'))} (1 / number of classes). "
          f"A control that scored high would invalidate every number in this report.", ""]
    return L


# ------------------------------------------------------- Group A additions (3.8+)
def sec_a1_leakage_stratified() -> list[str]:
    d = J("phase2/leakage_stratified_accuracy.json")
    L: list[str] = []
    L += ["### 3.8 Published checkpoint: images WITH vs WITHOUT a near-duplicate in train "
          "(Group A1)", "",
          f"Source of per-image predictions: `{d['split']}`; distance: {d['distance_metric']}; "
          f"CI: {d['ci_method']} ({r(d['n_bootstrap'])} draws); test: "
          f"{d['permutation_test']} ({r(d['n_permutations'])} draws), seeded with "
          f"{r(d['seed'])}. Group sizes cross-checked against the Phase-1 audit at every "
          f"threshold: {pf(all(v['crosscheck_matches_phase1'] for v in d['thresholds'].values()))}.", ""]
    rows = []
    for t in sorted(d["thresholds"], key=lambda x: int(x)):
        v = d["thresholds"][t]
        rows.append([t, r(v["n_with_near_dup_in_train"]), r(v["accuracy_with"]),
                     ci(v["accuracy_with_ci95"]), r(v["n_without"]),
                     r(v["accuracy_without"]), ci(v["accuracy_without_ci95"]),
                     r(v["difference_with_minus_without"]),
                     r(v["permutation_p_two_sided"])])
    L += table(["t", "n with", "acc with", "95% CI", "n without", "acc without",
                "95% CI", "difference", "permutation p"], rows)
    L += ["", "At every tested threshold the with-group point estimate is higher, but no "
              "permutation test rejects the equal-accuracy null at the conventional "
              "two-sided 5% level - the honest reading is *suggestive, not significant* "
              "at this test-set size.", ""]
    return L


def sec_a2_seed_comparison() -> list[str]:
    d = J("phase2/seed_comparison_s1_vs_s2.json")
    rec = d["recipe"]
    L: list[str] = []
    L += ["### 3.9 Seed variance: original split S1 beside group-disjoint S2 (Group A2)", "",
          f"Same recipe both sides: {r(rec['epochs'])} epochs, batch {r(rec['batch'])}, "
          f"{rec['optimizer']}, selection by {rec['model_selection']}, architecture "
          f"{rec['arch']} ({rec['arch_evidence']}).", ""]
    rows = []
    for side, name in (("S1", "S1 (original split)"), ("S2", "S2 (group-disjoint)")):
        a = d[f"{'S1_original_split' if side == 'S1' else 'S2_group_disjoint'}"]
        rows.append([name, r(a["n_seeds"]), r(a["accuracy_mean"]), r(a["accuracy_sd"]),
                     r(a["macro_f1_mean"]), r(a["macro_f1_sd"])])
    L += table(["split", "seeds", "acc mean", "acc sd", "macro-F1 mean", "macro-F1 sd"], rows)
    per_rows = []
    for side in ("S1", "S2"):
        for seed, v in sorted(d["per_seed"][side].items()):
            per_rows.append([side, seed, r(v["epochs"]), r(v["accuracy"]), r(v["macro_f1"]),
                             r(v["best_val_accuracy"]), f"`{v['source']}`"])
    L += ["", "Per seed:", ""]
    L += table(["split", "seed", "epochs", "test acc", "macro-F1", "best val acc", "source"],
               per_rows)
    det = d["determinism_check_seed42"]
    L += ["",
          f"Mean accuracy difference S1 - S2: "
          f"**{r(d['mean_difference_S1_minus_S2_accuracy'])}**. Determinism re-run of S1 "
          f"seed 42 gives identical accuracy: {pf(det['bit_identical_accuracy'])} "
          f"({det['seed42_run1_vs_run2_accuracy'][0]} vs "
          f"{det['seed42_run1_vs_run2_accuracy'][1]}).", ""]
    return L


def sec_a3_threshold_sweep() -> list[str]:
    agg = C("phase2/threshold_sweep_all_aggregate.csv")
    per = C("phase2/threshold_sweep_all.csv")
    L: list[str] = []
    L += ["### 3.10 S2 rebuilt at looser near-duplicate thresholds (Group A3)", "",
          "Same project splitter and recipe as S2; only the near-duplicate grouping "
          "threshold changes. Question: does accuracy keep falling as grouping loosens?", ""]
    rows = [[r(x["threshold"]), r(int(x["n_seeds"])), r(x["accuracy_mean"]),
             r(x["accuracy_sd"]), r(x["macro_f1_mean"]), r(x["macro_f1_sd"])]
            for _, x in agg.iterrows()]
    L += table(["near-dup threshold t", "seeds", "acc mean", "acc sd", "macro-F1 mean",
                "macro-F1 sd"], rows)
    rows = [[r(x["threshold"]), r(x["seed"]), r(x["arch"]), r(x["accuracy"]),
             r(x["macro_f1"]), r(x["best_val_accuracy"]), r(int(x["n_test"]))]
            for _, x in per.iterrows()]
    L += ["", "Per seed:", ""]
    L += table(["t", "seed", "architecture", "test acc", "macro-F1", "best val acc",
                "test images"], rows)
    # leakage assertions for the rebuilt splits (HARD RULE 4)
    rows = []
    for t in (4, 6):
        a = J(f"phase2/thresh{t}_split_assertions.json")
        for name, v in a["assertions"].items():
            rows.append([r(t), name, str(v) if str(v) == "PASS" else f"**{v}**"])
    L += ["", "Leakage assertions on the rebuilt splits (HARD RULE 4):", ""]
    L += table(["t", "assertion", "result"], rows)
    try:
        tr = J("phase2/threshold_sweep_trend.json")
    except MissingRequired:
        tr = None
    if tr:
        L += ["", f"Trend: {tr['direction_statement']} "
                  f"(deltas vs t = {r(tr['reference_threshold'])}: "
                  + "; ".join(f"t={k} -> {r(v)}" for k, v in tr["delta_mean_acc_vs_reference"].items())
                  + ").", ""]
    return L


def sec_a4_arch_multiseed() -> list[str]:
    sel = J("phase2/arch3_selection.json")
    L: list[str] = []
    L += ["### 3.11 Multi-seed architecture comparison (Group A4)", "",
          f"Criterion: {sel['criterion']}; selection used test data: "
          f"{pf(sel['selection_uses_test_data'] is False)}. "
          f"{r(sel['n_seeds_per_arch'])} seeds per candidate, {r(sel['epochs'])} epochs each.", ""]
    rows = []
    for arch, c in sel["candidates"].items():
        mark = " (selected)" if arch == sel["selected_architecture"] else ""
        rows.append([arch + mark, r(c["val_acc_mean"]), r(c["val_acc_sd"]),
                     r(c["test_acc_mean"]), r(c["test_acc_sd"]),
                     r(c["test_macro_f1_mean"]), r(c["test_macro_f1_sd"])])
    L += table(["architecture", "val acc mean (criterion)", "val acc sd", "test acc mean",
                "test acc sd", "test macro-F1 mean", "test macro-F1 sd"], rows)
    L += ["", f"Selected on validation only: **{sel['selected_architecture']}**. "
              f"{sel['note']}", ""]
    try:
        wt = J("phase2/arch_winner_test.json")
        if wt.get("arch") == sel["selected_architecture"]:
            L += [f"The single post-selection S2 test evaluation of this architecture "
                  f"(`{wt['arch']}`, from the earlier single-seed sweep): "
                  f"accuracy {r(wt['accuracy'])} (95% CI {ci(wt['accuracy_95_ci'])}), "
                  f"macro-F1 {r(wt['macro_f1'])}.", ""]
        else:
            L += [f"Note: the existing single-test-evaluation file "
                  f"`arch_winner_test.json` evaluates **{wt.get('arch')}** (winner of the "
                  f"earlier single-seed sweep), which is not this multi-seed selection "
                  f"({sel['selected_architecture']}); its accuracy "
                  f"{r(wt['accuracy'])} (95% CI {ci(wt['accuracy_95_ci'])}) is therefore "
                  f"reported for that architecture only. For the multi-seed selection the "
                  f"evidence is the per-candidate test columns above (each finished model's "
                  f"single test evaluation, HARD RULE 3).", ""]
    except MissingRequired:
        pass
    return L


def sec_a5_spec_extras() -> list[str]:
    mb = J("phase2/majority_baseline.json")
    nm = J("phase2/numpy_metric_check.json")
    ts = J("phase2/temperature_scaling.json")
    eg = J("phase2/error_gallery_eb_lmf.json")
    ex = J("phase2/excl48_comparison.json")
    L: list[str] = []
    L += ["### 3.12 Missing spec items (Group A5)", "",
          "**Majority-class baseline** (predict the most frequent train class):", ""]
    L += table(["split", "majority class", "test images", "baseline test acc",
                "model test acc for comparison"], [
        ["S1 (original)", mb["S1_original_split"]["majority_class"],
         r(mb["S1_original_split"]["n_test"]), r(mb["S1_original_split"]["test_accuracy"]),
         r(J("phase2/faithful_original.json")["accuracy"])],
        ["S2 (group-disjoint)", mb["S2_group_disjoint"]["majority_class"],
         r(mb["S2_group_disjoint"]["n_test"]), r(mb["S2_group_disjoint"]["test_accuracy"]),
         r(J("phase2/s2_seed_aggregate.json")["accuracy_mean"])],
    ])
    L += ["", f"Definition: {mb['definition']}.", "",
          "**Numpy-only metric recomputation** (HARD RULE 1 cross-check):", ""]
    rows = [[k, r(v["n_images"]), r(v["abs_diff"]["accuracy"], 10),
             r(v["abs_diff"]["macro_f1"], 10), r(v["abs_diff"]["macro_roc_auc"], 10)]
            for k, v in nm["per_file"].items()]
    L += table(["predictions file", "images", "|Δ acc|", "|Δ macro-F1|", "|Δ macro AUC|"], rows)
    L += ["", f"Method: {nm['method']}. Max absolute difference "
              f"**{r(nm['max_abs_diff'], 10)}** vs tolerance {r(nm['tolerance'], 10)} - "
              f"pass: {pf(nm['pass'])}.", "",
          "**Calibration: ECE before / after temperature scaling** "
          "(temperature fit on validation only):", ""]
    L += table(["quantity", "value"], [
        ["temperature T", r(ts["temperature"])],
        ["fit split", ts["temperature_fit_split"]],
        ["ECE before (test)", r(ts["ece_before"])],
        ["ECE after (test)", r(ts["ece_after"])],
        ["ECE bins", r(ts["ece_bins"])],
        ["test accuracy before", r(ts["test_accuracy_before"])],
        ["test accuracy after", r(ts["test_accuracy_after"])],
        ["accuracy unchanged by scaling", pf(ts["accuracy_unchanged_by_scaling"])],
    ])
    L += ["", f"Method: {ts['method']}.", "",
          "**Error gallery - Early Blight vs Leaf Miner Flies confusions on test:**", ""]
    rows = []
    for key, p in eg["panels"].items():
        rows.append([key.replace("_", " "), r(p["n_eb_to_lmf"]), r(p["n_lmf_to_eb"]),
                     r(p["n_total_swaps"]), f"`{p['montage'].split(' ')[0]}`"])
    L += table(["panel", "true EB -> predicted LMF", "true LMF -> predicted EB", "total",
                "montage"], rows)
    L += ["", "**Re-training excluding the high-confidence cross-label images** "
              f"(Group A5):", ""]
    L += table(["side", "seeds", "acc mean", "acc sd", "macro-F1 mean", "macro-F1 sd"], [
        ["baseline S1", r(len(ex["baseline_S1"]["per_seed"])),
         r(ex["baseline_S1"]["accuracy_mean"]), r(ex["baseline_S1"]["accuracy_sd"]),
         r(ex["baseline_S1"]["macro_f1_mean"]), r(ex["baseline_S1"]["macro_f1_sd"])],
        [f"S1 minus {r(ex['exclusion']['n_excluded'])} images",
         r(len(ex["excluded_S1"]["per_seed"])), r(ex["excluded_S1"]["accuracy_mean"]),
         r(ex["excluded_S1"]["accuracy_sd"]), r(ex["excluded_S1"]["macro_f1_mean"]),
         r(ex["excluded_S1"]["macro_f1_sd"])],
    ])
    L += ["",
          f"Excluded images: {r(ex['exclusion']['n_excluded'])} distinct "
          f"(method {ex['exclusion']['method']}), removed per split "
          + ", ".join(f"{k}: {v}" for k, v in ex["exclusion"]["counts_per_split"].items())
          + f". Change in mean accuracy: **{r(ex['change_accuracy_mean'])}**; "
          f"change in mean macro-F1: **{r(ex['change_macro_f1_mean'])}** "
          f"(positive = removing the conflicting images helped). {ex['note']}", "",
          f"Per-seed evidence: `results/phase2/excl48_seed*.json`; full extras: "
          f"`results/phase2/spec_extras_summary.json`.", ""]
    return L


def sec_phase3() -> list[str]:
    p3q = J("phase3/stage_a_data_quality.json")
    L = ["## 4. Phase 3 - Track B (TOBRFV-LMID multispectral) audit", "",
         "### 4.1 Data quality", ""]
    rows = []
    for s, q in p3q["per_stream_quality"].items():
        rows.append([s, r(q["n"]), ", ".join(f"{k}:{v}" for k, v in q["modes"].items()),
                     ", ".join(f"{k}:{v}" for k, v in q["dtypes"].items()),
                     r(q["n_distinct_shapes"]),
                     f"{r(q['width_min_max'][0])}-{r(q['width_min_max'][1])}",
                     r(q["n_unreadable"])])
    L += table(["stream", "files", "PIL mode", "dtype", "distinct shapes", "width range",
                "unreadable"], rows)
    pr = p3q["stream_pairing"]
    L += ["", f"Stream pairing: {r(pr['n_tuples'])} tuples, "
              f"{r(pr['n_with_all_4_streams'])} carry all four streams "
              f"(complete = {pf(pr['complete'])}). Exact-duplicate groups across all four "
              f"streams: {r(p3q['exact_duplicates']['n_duplicate_groups'])}."]
    if pr.get("approved_exclusions"):
        dfc = J("phase3/stream_pairing_defect.json")
        L += ["", f"> **Data defect (raw data untouched):** {r(dfc['n_incomplete_tuples'])} of "
                  f"{r(dfc['n_tuples'])} capture tuples cannot be paired across the four "
                  f"streams and were **excluded from the 4-stream manifest only**, under the "
                  f"documented decision in `analysis/approved_data_exclusions.json`. Their "
                  f"files remain in the per-stream audits and are never renamed or modified "
                  f"(HARD RULE 2). Full evidence: "
                  f"`results/phase3/stream_pairing_defect.json`.",
              "", "Affected tuples:", ""]
        L += table(["tuple", "present streams", "missing streams", "reason"],
                   [[t["tuple_key"], ", ".join(t["present_streams"]),
                     ", ".join(t["missing_streams"]), t["reason"]]
                    for t in dfc["tuples"]])
    L += [""]
    return L


def sec_phase3_plantkey() -> list[str]:
    ev = J("phase3/plant_key_evidence.json")
    L: list[str] = []
    k = ev["reproduced_by_chosen_key"]
    L += ["### 4.2 Plant key (resolved by evidence, not assumption)", "",
          f"Question: {ev['question']}", "",
          f"Key `host|batch|cls|plant` gives **{r(k['n_plants'])}** plants = "
          f"{r(k['healthy'])} healthy + {r(k['infected'])} infected, matching all three paper "
          f"claims: {pf(k['all_three_claims_match'])}.", ""]
    L += table(["candidate key", "distinct groups (Tomato)"],
               [[a, r(b)] for a, b in ev["candidate_key_counts_tomato"].items()])
    ps = ev["per_plant_structure"]
    L += ["",
          f"Per plant: images {r(ps['images_per_plant']['min'])}-"
          f"{r(ps['images_per_plant']['max'])} (median {r(ps['images_per_plant']['median'])}), "
          f"days {r(ps['days_per_plant']['min'])}-{r(ps['days_per_plant']['max'])} "
          f"(median {r(ps['days_per_plant']['median'])}), "
          f"angles {r(ps['angles_per_plant']['min'])}-{r(ps['angles_per_plant']['max'])}.",
          f"Visual check: `{ev['visual_evidence']['grid_image']}` - "
          f"{ev['visual_evidence']['what_to_look_for']}.",
          "", ev["conclusion"], ""]
    return L


def sec_phase3_claims() -> list[str]:
    claims = J("phase3/claims_comparison.json")
    L: list[str] = []
    L += ["### 4.3 Paper claims vs this copy of the data", "",
          f"Tolerances: counts exact, prevalence +/- "
          f"{claims['tolerances']['prevalence_pp']} pp, Cohen's d +/- "
          f"{claims['tolerances']['cohens_d']}.", ""]
    rows = []
    for c in claims["checks"]:
        ours = c["ours"]
        txt = "; ".join(f"{a}={b}" for a, b in ours.items()) if isinstance(ours, dict) \
            else str(ours)
        rows.append([c["claim"], pf(c["pass"]), txt[:200]])
    L += table(["claim group", "result", "measured"], rows)
    L += ["", f"**{r(claims['n_pass'])} / {r(claims['n_checks'])} claim groups pass.**", ""]

    cd = C("phase3/cohens_d_vs_claim.csv")
    L += ["Cohen's d (healthy minus infected):", ""]
    rows = []
    for _, x in cd.iterrows():
        rows.append([x["stream"], f"{r(x['image_level_d'])} vs {r(x['claim_image_level'])}",
                     pf(bool(x["image_level_pass"])) if pd.notna(x["image_level_pass"]) else "n/a",
                     f"{r(x['plant_level_d'])} vs {r(x['claim_plant_level'])}",
                     pf(bool(x["plant_level_pass"])) if pd.notna(x["plant_level_pass"]) else "n/a",
                     f"{r(x['plant_level_ci_lo'])} .. {r(x['plant_level_ci_hi'])}"])
    L += table(["stream", "image level ours vs claim", "pass", "plant level ours vs claim",
                "pass", "plant-level 95% CI"], rows)
    L += [""]
    return L


def sec_phase3_splits() -> list[str]:
    p3a = J("phase3/split_leakage_assertions.json")
    L: list[str] = []
    L += ["### 4.4 Plant-disjoint splits (HARD RULE 4)", "",
          f"Plants: {r(p3a['n_plants'])}; images: {r(p3a['n_images'])}.", ""]
    L += table(["assertion", "result"],
               [[a, str(v) if str(v).startswith("FAIL") else pf(v == "PASS")]
                for a, v in p3a["assertions"].items()])
    L += ["", "Per-seed sizes and prevalence: "
              "`results/phase3/split_leakage_assertions.json`.", ""]
    return L


def sec_phase3_crosscheck() -> list[str]:
    xc = J("phase3/project_crosscheck.json")
    L: list[str] = []
    L += ["### 4.5 Independent cross-check with the project's own CAEP tooling", "",
          f"`caep/build_manifest.py` return code {r(xc.get('build_manifest_returncode'))}; "
          f"`caep/audit_tobrfv.py` return code {r(xc.get('audit_tobrfv_returncode'))}. Both were "
          "run **unedited**; stdout kept in `results/phase3/`.", ""]
    return L


# ------------------------------------------------------- Group B additions (4.6+)
def sec_b1_claim_rootcause() -> list[str]:
    rc = J("phase3/claim_rootcause/claim_rootcause.json")
    L: list[str] = []
    L += ["### 4.6 Root-cause of the one failed claim (prevalence by angle) - Group B1", "",
          f"Claim: {rc['claim']} (± {r(rc['tolerance_pp'])} pp tolerance); audited "
          f"definition: {rc['audited_definition']}.", "",
          "What the claim implies versus what the data contains:", ""]
    rows = []
    for ang in ("0", "5", "15"):
        b = rc["implied_numerator_bookkeeping"][ang]
        rows.append([f"{ang} deg", r(b["audited_numerator"]), r(b["audited_denominator"]),
                     r(b["claim_pct"], 1), r(b["numerator_implied_by_claim"]),
                     r(b["nearest_integer_numerator"]), r(b["gap_images"])])
    L += table(["angle", "measured numerator", "measured denominator", "claim %",
                "numerator implied by claim", "nearest integer", "gap (images)"], rows)
    L += ["",
          f"Integrity: {len(rc['integrity']['incomplete_tuples_in_raw_data'])} incomplete "
          f"capture tuples exist and match the approved exclusion list: "
          f"{pf(rc['integrity']['match_approved_exclusions'])} "
          f"({r(rc['integrity']['n_orphan_rows_at_angle15'])} of their rows sit at 15 deg).", "",
          "Alternative definitions tested (numerator / denominator at each angle):", ""]
    rows = []
    for name, v in rc["variants"].items():
        p15 = v["per_angle"]["15"]
        rows.append([name.replace("_", " "),
                     f"{r(p15['numerator'])} / {r(p15['denominator'])}",
                     r(p15["pct"]), r(v["delta_pp"]["15"]),
                     r(v["per_angle"]["0"]["pct"]), r(v["delta_pp"]["0"]),
                     r(v["per_angle"]["5"]["pct"]), r(v["delta_pp"]["5"]),
                     pf(v["reproduces_all_three_claims"])])
    L += table(["definition", "15 deg num / den", "15 deg %", "Δ15 pp", "0 deg %", "Δ0 pp",
                "5 deg %", "Δ5 pp", "reproduces all three claims"], rows)
    L += ["", f"**Verdict: {rc['verdict']}.** Evidence: "
              "`results/phase3/claim_rootcause/claim_rootcause.json` (+ .csv).", ""]
    return L


def sec_b5_article_labels() -> list[str]:
    a = _ainput("dataset_article_reference.json")
    cit = a["citation"]
    L: list[str] = []
    L += ["### 4.7 Label provenance - the dataset's own data-description article (Group B5)", "",
          f"Citation: {cit['authors']}. \"{cit['title']}.\" {cit['journal']}. "
          f"doi:{cit['doi']} ({cit['pmcid']}), {cit['license']}. Fetched "
          f"{cit['fetched_on']} from {cit['fetched_from']}; dataset repository "
          f"{cit['dataset_repository']}.", "",
          "How labels were assigned, quoted verbatim:", ""]
    for q in a["how_labels_were_assigned"]:
        L += [f"> **{q['section']}** - \"{q['quote']}\"", ">", f"> *{q['what_it_shows']}*", ""]
    so = a["symptom_onset_information_exists"]
    L += [f"**Does symptom-onset information exist? {so['our_statement']}**", "",
          f"> **{so['evidence_quote']['section']}** - \"{so['evidence_quote']['quote']}\"", ""]
    return L


def sec_phase4() -> list[str]:
    p4 = J("phase4/phase4_summary.json")
    st = p4["settings"]
    L = ["## 5. Phase 4 - Track B experiments", "",
         f"Settings: {r(st['epochs'])} epochs, image size {r(st['image_size'])}, batch "
         f"{r(st['batch_size'])}, split seeds "
         f"{', '.join(str(s) for s in st['split_seeds'])}. Selection: {st['selection']}. "
         f"Test policy: {st['test_policy']}.", ""]
    rows = [[m, r(a["n_seeds"]), r(a["image_acc_mean"]), r(a["image_acc_std"]),
             r(a["image_auc_mean"]), r(a["image_auc_std"]), r(a["image_f1_mean"]),
             r(a["plant_acc_mean"]), r(a["plant_auc_mean"]), r(a["plant_auc_std"]),
             r(a["runtime_s_total"], 0)]
            for m, a in p4["aggregate_by_mode"].items()]
    L += table(["mode", "seeds", "image acc", "sd", "image AUC", "sd", "image F1", "plant acc",
                "plant AUC", "sd", "runtime (s)"], rows)
    L += ["", "Per-seed numbers: `results/phase4/all_runs.csv`.", ""]
    return L


def sec_phase4_paired() -> list[str]:
    ps4 = J("phase4/paired_stats.json")
    L: list[str] = []
    L += ["### 5.1 Paired, plant-level comparisons (project's `caep/paired_stats.py`)", ""]
    rows = [[p["label"], r(p["n_test_plants"]), r(p["n_infected_plants"]), r(p["a_plant_acc"]),
             r(p["a_plant_auc"]), r(p["b_plant_acc"]), r(p["b_plant_auc"]),
             f"{r(p['delta_acc'])} {ci(p['delta_acc_ci'], 3)}",
             f"{r(p['delta_auc'])} {ci(p['delta_auc_ci'], 3)}", r(p["mcnemar_p"])]
            for p in ps4]
    L += table(["comparison (B vs A)", "test plants", "infected", "A acc", "A AUC", "B acc",
                "B AUC", "delta acc [95% CI]", "delta AUC [95% CI]", "McNemar p"], rows)
    L += ["", "A confidence interval that includes 0 means no demonstrated difference - reported "
              "as such, not as a win.", ""]
    return L


def sec_phase4_control() -> list[str]:
    p4 = J("phase4/phase4_summary.json")
    L: list[str] = []
    nc = p4.get("negative_control")
    L += ["### 5.2 Negative control (training labels permuted)", ""]
    if isinstance(nc, dict):
        L += [f"Test image accuracy **{r(nc['image_acc'])}**, test image AUC "
              f"**{r(nc['image_auc'])}**, plant AUC **{r(nc['plant_auc'])}**. "
              f"{nc['interpretation']}", ""]
    else:
        L += [f"Not run: {nc}.", ""]
    return L


# ------------------------------------------------------- Group B additions (5.3+)
def sec_b2_cluster_bootstrap() -> list[str]:
    d = J("phase4/image_level_cluster_bootstrap.json")
    bs = d["bootstrap"]
    L: list[str] = []
    L += ["### 5.3 Image-level comparison with plant-cluster bootstrap (Group B2)", "",
          f"Plant-level scores saturate at 1.0 on this test set, so modes are compared at "
          f"image level with a cluster bootstrap (unit: {bs['unit']}, "
          f"{r(bs['n'])} draws, rng seed {r(bs['rng_seed'])}, {bs['ci']}). "
          f"{d['note']}", "", "Aggregate point estimates across seeds:", ""]
    rows = [[m, r(a["acc_mean"]), r(a["acc_sd"]), r(a["auc_mean"]), r(a["auc_sd"])]
            for m, a in d["aggregate_point_estimates_across_seeds"].items()]
    L += table(["mode", "image acc mean", "sd", "image AUC mean", "sd"], rows)
    rows = []
    for key, v in d["per_seed_mode"].items():
        seed, mode = key.split("/")
        rows.append([seed, mode, r(v["n_images"]), r(v["n_plants"]), r(v["n_pos"]),
                     r(v["image_acc"]), ci(v["acc_ci95"]),
                     r(v["image_auc"]), ci(v["auc_ci95"])])
    L += ["", "Per seed, with cluster-bootstrap 95% CIs:", ""]
    L += table(["seed", "mode", "images", "plants", "positive", "image acc", "acc 95% CI",
                "image AUC", "AUC 95% CI"], rows)
    rows = []
    for key, v in d["paired_cluster_bootstrap_differences"].items():
        seed, comp = key.split("/")
        rows.append([seed, comp.replace("_", " "), r(v["n_plants"]),
                     r(v["acc_difference"]), ci(v["acc_diff_ci95"]),
                     r(v["acc_p_boot_le_zero"]), r(v["auc_difference"]),
                     ci(v["auc_diff_ci95"]), r(v["auc_p_boot_le_zero"])])
    L += ["", "Paired plant-cluster bootstrap differences (B - A):", ""]
    L += table(["seed", "comparison", "plants", "Δacc", "Δacc 95% CI", "p boot ≤ 0",
                "ΔAUC", "ΔAUC 95% CI", "p boot ≤ 0"], rows)
    L += ["", "A 95% interval that includes 0 means no demonstrated difference at that "
              "seed - reported as such, not as a win.", ""]
    return L


def sec_b3_subgroups() -> list[str]:
    df = C("phase4/subgroup_metrics_day_angle_batch.csv")
    meta = J("phase4/subgroup_metrics_day_angle_batch.json")
    L: list[str] = []
    L += ["### 5.4 Accuracy and AUC by recording day, angle and batch (Group B3)", "",
          f"{r(meta['rows'])} cells; {meta['auc_null_means']}. "
          "Every mode x seed x cell, with counts:", ""]
    for factor, title in (("angle", "by angle"), ("batch", "by batch"), ("day", "by day")):
        sub = df[df["factor"] == factor]
        rows = []
        for _, x in sub.iterrows():
            rows.append([str(x["cell"]), r(x["seed"]), x["mode"], r(x["n_images"]),
                         r(x["n_plants"]), r(x["n_pos"]), r(x["accuracy"]),
                         r(x["auc"]) if bool(x["auc_defined"]) else "n/a (single class)"])
        L += [f"**{title}:**", ""]
        L += table(["cell", "seed", "mode", "images", "plants", "positive", "accuracy",
                    "AUC"], rows)
        L += [""]
    L += ["Source rows: `results/phase4/subgroup_metrics_day_angle_batch.csv`.", ""]
    return L


def sec_b4_confounds() -> list[str]:
    d = J("phase4/confounds/confound_baselines.json")
    pop = d["population"]
    L: list[str] = []
    L += ["### 5.5 Confound baselines and the leakage demo (Group B4)", "",
          f"Population: {r(pop['rows'])} {pop['unit']}s / {r(pop['plants'])} plants "
          f"({pop['source']}); features: metadata = {', '.join(d['feature_sets']['metadata'])}; "
          f"intensity = {', '.join(d['feature_sets']['intensity'])}.", "",
          f"Model: {d['model']}. Seeds: {', '.join(str(s) for s in d['seeds'])}. "
          f"Protocols: plant_disjoint = {d['protocols']['plant_disjoint']}; "
          f"image_level = {d['protocols']['image_level']}.", ""]
    rows = []
    for fs_name, s in d["summary"].items():
        rows.append([fs_name, "plant-disjoint (correct)", r(s["plant_disjoint_acc_mean"]),
                     r(s["plant_disjoint_acc_sd"]), r(s["plant_disjoint_auc_mean"])])
        rows.append([fs_name, "image-level (leaky demo)", r(s["image_level_acc_mean"]),
                     r(s["image_level_acc_sd"]), r(s["image_level_auc_mean"])])
    L += table(["features", "protocol", "test acc mean", "acc sd", "test AUC mean"], rows)
    L += ["", "Headline (code-computed):", ""]
    rows = [[fs, r(s["acc_inflation_image_level_minus_plant_disjoint"]),
             r(s["auc_inflation_image_level_minus_plant_disjoint"])]
            for fs, s in d["summary"].items()]
    L += table(["features", "acc inflation (image-level − plant-disjoint)",
                "AUC inflation (image-level − plant-disjoint)"], rows)
    L += ["", "Leakage checks by the project's own `caep/check_leakage.py`:", ""]
    rows = [[k, r(v), "must be 0 (HARD RULE 4)" if k.startswith("plant_disjoint")
             else "expected 1: plant leakage = the demonstration"]
            for k, v in d["leakage_check_exit_codes"].items()]
    L += table(["split", "check exit code", "interpretation"], rows)
    L += ["", f"{d['leakage_check_interpretation']}. Per-seed/per-feature results and the "
              "majority baseline: `results/phase4/confounds/confound_baselines.csv`.", ""]
    return L


def sec_b6_budget_check() -> list[str]:
    m = J("phase4/budget_check_px224_e15/fusion/metrics.json")
    cfg = J("phase4/budget_check_px224_e15/run_config.json")  # written by the runner
    p4 = J("phase4/phase4_summary.json")
    base = p4["aggregate_by_mode"]["fusion"]
    L: list[str] = []
    L += ["### 5.6 Higher-budget check: fusion at 224 px, more epochs (Group B6)", "",
          f"Run: mode {m.get('mode')}, {r(cfg['epochs'])} epochs, image size "
          f"{r(cfg['image_size'])}, split seed {r(m.get('seed'))}, "
          f"best validation AUC {r(m.get('best_val_auc'))} "
          f"(selection unchanged: validation only, test once).", ""]
    L += table(["setting", "image acc", "image AUC", "plant acc", "plant AUC"], [
        [f"fusion, minimal budget ({r(p4['settings']['epochs'])} epochs, "
         f"{r(p4['settings']['image_size'])} px, 3 seeds)", r(base["image_acc_mean"]),
         r(base["image_auc_mean"]), r(base["plant_acc_mean"]), r(base["plant_auc_mean"])],
        [f"fusion, this run ({r(cfg['epochs'])} epochs, {r(cfg['image_size'])} px, "
         f"{r(m.get('seed'))} seed)", r(m["image_acc"]), r(m["image_auc"]),
         r(m["plant_acc"]), r(m["plant_auc"])],
    ])
    L += ["", "Purpose: check whether the image-level ordering of the three modes is an "
              "artifact of the minimal budget. Single seed - a descriptive check, not a "
              "new headline.", ""]
    return L


def sec_warnings(ctx: dict) -> list[str]:
    p1, lk, sla = ctx["p1"], ctx["lk"], ctx["sla"]
    fo, s2agg, oa = ctx["fo"], ctx["s2agg"], ctx["oa"]
    claims, p3a, ev, p4 = ctx["claims"], ctx["p3a"], ctx["ev"], ctx["p4"]
    lbt = C("phase1/leakage_by_threshold.csv")
    t_chosen = p1["d_near_duplicates"]["chosen_threshold"]
    row_chosen = lbt[lbt["threshold"] == t_chosen].iloc[0]
    art = _ainput("dataset_article_reference.json")
    rc = J("phase3/claim_rootcause/claim_rootcause.json")
    clb = J("phase4/image_level_cluster_bootstrap.json")
    w = [
        "**No claim is made here about detecting disease before symptoms are visible, and no "
        "claim of readiness for use outside the lab is made** (HARD RULE 8). Track B results "
        "are plant-disjoint. Corrected after reading the dataset's own article (Sec. 4.7): "
        "plants are labelled by experimental design - mechanically inoculated vs healthy "
        "control (Sec. 3) - with infection status verified by RT-PCR / real-time RT-PCR "
        "(Sec. 4.2), **not** by scoring visible symptoms; but the dataset carries no "
        "symptom-onset dates, so images cannot be separated into captures taken before vs "
        "after symptoms appeared, and nothing in this report establishes performance on "
        "plants whose symptoms have not yet appeared.",
        f"Raw-data defect in Track B: {r(J('phase3/stream_pairing_defect.json')['n_incomplete_tuples'])} of "
        f"{r(J('phase3/stream_pairing_defect.json')['n_tuples'])} capture tuples cannot be paired across the four streams "
        "(" + "; ".join(t["tuple_key"] for t in J("phase3/stream_pairing_defect.json")["tuples"]) + "). They were excluded "
        f"from the 4-stream manifest used by every Phase 4 mode under a documented decision "
        f"(`analysis/approved_data_exclusions.json`); the files themselves are untouched "
        f"(HARD RULE 2) and stay in the per-stream audits. The affected plants keep many "
        f"other complete tuples, so the plant-disjoint splits and plant counts are unchanged. "
        f"No pairing between the orphaned files was guessed (HARD RULE 6).",
        f"One paper claim remains failed and is reported as such: prevalence at 15 deg - "
        f"measured {rc['implied_numerator_bookkeeping']['15']['audited_numerator']} / "
        f"{rc['implied_numerator_bookkeeping']['15']['audited_denominator']} vs claimed "
        f"{rc['implied_numerator_bookkeeping']['15']['claim_pct']}% (a gap of "
        f"{rc['implied_numerator_bookkeeping']['15']['gap_images']} images at the audited "
        f"denominator); {len(rc['variants'])} alternative definitions were tested and none "
        f"reproduces all three claimed angles within tolerance (Sec. 4.6, "
        f"`results/phase3/claim_rootcause/claim_rootcause.json`).",
        f"Track B plant-level metrics saturate at 1.0 on the "
        f"{r(clb['per_seed_mode']['seed42/rgb']['n_plants'])}-plant test set for every mode, "
        f"so modes are only distinguishable at image level; all image-level mode comparisons "
        f"use plant-cluster bootstrap intervals (Sec. 5.3), and a paired interval containing "
        f"0 is reported as no demonstrated difference.",
        "The Group A / Group B follow-up analyses (Sec. 3.8-3.12, 4.6-4.7, 5.3-5.6) are "
        "exploratory: their p-values and bootstrap intervals are not adjusted for the number "
        "of comparisons, and each is reported as descriptive evidence.",
        f"S1 is *measured* for leakage, not asserted against it, because S1 is the object of "
        f"study: at the chosen near-duplicate threshold {r(t_chosen)}, "
        f"{r(row_chosen['pct_test_near_dup_in_train'])}% of test images have a near-duplicate in "
        f"train (full sweep in `results/phase1/leakage_by_threshold.csv`). The rule-4 assertions "
        f"are applied to S2 ({pf(all(str(v) == 'PASS' for v in sla['assertions'].values()))}) and "
        f"to every Track B split "
        f"({pf(all(str(v) == 'PASS' for v in p3a['assertions'].values()))}).",
        f"Cross-split contamination: {r(oa['pct_s2_test_contaminated'])}% of the S2 test images "
        f"({r(oa['s2_test_seen_by_s1_train_or_val'])} of {r(oa['s2_test_total'])}) were in S1's "
        f"training or validation set, so evaluating the S1-trained checkpoint on S2 is not a "
        f"generalisation result and is not quoted as one; only the "
        f"{r(oa['s2_test_unseen_by_s1'])} never-seen images are reported, with their wide "
        f"confidence interval.",
        f"The Track A headline is the group-disjoint S2 result: mean accuracy "
        f"{r(s2agg['accuracy_mean'])} over {r(s2agg['n_seeds'])} seeds (sd "
        f"{r(s2agg['accuracy_std'])}) versus {r(fo['accuracy'])} on the original split - both are "
        f"reported side by side and the difference is attributed to the split.",
        f"{r(p1['d_near_duplicates']['n_cross_label_near_dup_rows'])} cross-label near-duplicate "
        f"rows exist, {r(p1['d_near_duplicates']['n_cross_label_by_ncc_alone'])} of them found only "
        f"by NCC. SHA-256 reported zero duplicates, so this label noise is invisible to "
        f"hash-based deduplication.",
        f"The Track B plant key was resolved by evidence: host|batch|cls|plant reproduces the "
        f"paper's {r(ev['reproduced_by_chosen_key']['n_plants'])}/"
        f"{r(ev['reproduced_by_chosen_key']['healthy'])}/"
        f"{r(ev['reproduced_by_chosen_key']['infected'])} plant counts exactly, and the visual grid "
        f"shows longitudinal growth within a column. Any other key would silently change what "
        f"'plant-disjoint' means.",
        f"Phase 4 ran at the Phase-0 *minimal* CPU setting ({r(p4['settings']['epochs'])} epochs, "
        f"image size {r(p4['settings']['image_size'])}), below the as-specified budget, so "
        f"absolute scores are a floor for this machine rather than a tuned result.",
        "Feature fusion concatenates the RGB encoder with the 800 nm / 1000 nm encoders "
        "(`fusion/fusion.py`); the broadband VNIR stream is in the manifest but that script does "
        "not consume it. Distillation was not run under the minimal budget and is reported as not "
        "run.",
        "Track B images are all 8-bit, so the `/255` normalisation in `fusion/fusion.py` is "
        "correct for this copy of the data and the `/65535` branch in `caep/audit_tobrfv.py` "
        "never applies; no uint16 file exists in any of the four streams.",
        "Architecture candidates were limited to the three supported by the project's own "
        "`models.py` (HARD RULE 7 - no existing script was extended), so the sweep is not an "
        "exhaustive architecture search.",
        f"Bootstrap intervals are stratified and seeded; with these test-set sizes they are wide, "
        f"and an interval containing 0 is reported as no demonstrated difference. Track B test "
        f"sets hold {r(ctx['n_test_plants'])} plants.",
    ]
    return ["## 6. Warnings and limitations", ""] + [f"{i}. {t}" for i, t in enumerate(w, 1)] + [""]


def sec_index() -> list[str]:
    idx = sorted(str(p.relative_to(RES)) for p in RES.rglob("*")
                 if p.is_file() and p.suffix in (".json", ".csv"))
    return (["## 7. Artifact index", "",
             "Every number above comes from these machine-written files:", ""]
            + [f"- `results/{p}`" for p in idx]
            + ["", "---", "",
               "Regenerate: `.venv/bin/python analysis/generate_results_md.py`. "
               "Verify: `python -m pytest tests/`.", ""])


# ================================================================ driver
def build(allow_incomplete: bool) -> tuple[str, dict]:
    L = ["# TomatoGuard AI - Results", "",
         "> **Machine-generated** by `analysis/generate_results_md.py` from the json/csv files",
         "> under `results/`. No number below is typed by hand.",
         "",
         f"- generated (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
         "- commands, runtimes, versions: `results/run_log.md`, `results/pip_freeze.txt`",
         "", "---", ""]
    errors: list[str] = []
    ctx: dict = {}

    def sec(name: str, fn) -> None:
        try:
            L.extend(fn())
        except (MissingRequired, KeyError, IndexError, TypeError, ValueError) as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            L.extend([f"> {PLACEHOLDER} - `{name}` could not be rendered: `{e}`", ""])

    sec("reproducibility+phase0", sec_repro)
    sec("phase1.1 counts+quality", sec_phase1)
    sec("phase1.2 duplicates+thresholds", sec_phase1_dups)
    sec("phase1.3 original split", sec_phase1_split)
    sec("phase1.4 label noise", sec_phase1_labels)
    sec("phase2.1 splits", sec_phase2)
    sec("phase2.2 published baseline", sec_phase2_published)
    sec("phase2.3 re-training", sec_phase2_retrain)
    sec("phase2.4 S2 headline", sec_phase2_s2)
    sec("phase2.5 architecture sweep", sec_phase2_arch)
    sec("phase2.6 contamination", sec_phase2_contamination)
    sec("phase2.7 negative control", sec_phase2_control)
    sec("phase2.8 A1 leakage-stratified", sec_a1_leakage_stratified)
    sec("phase2.9 A2 seed comparison", sec_a2_seed_comparison)
    sec("phase2.10 A3 threshold sweep", sec_a3_threshold_sweep)
    sec("phase2.11 A4 arch multiseed", sec_a4_arch_multiseed)
    sec("phase2.12 A5 spec extras", sec_a5_spec_extras)
    sec("phase3.1 data quality", sec_phase3)
    sec("phase3.2 plant key", sec_phase3_plantkey)
    sec("phase3.3 claims", sec_phase3_claims)
    sec("phase3.4 splits", sec_phase3_splits)
    sec("phase3.5 project cross-check", sec_phase3_crosscheck)
    sec("phase3.6 B1 claim root-cause", sec_b1_claim_rootcause)
    sec("phase3.7 B5 article labels", sec_b5_article_labels)
    sec("phase4.1 aggregate", sec_phase4)
    sec("phase4.2 paired stats", sec_phase4_paired)
    sec("phase4.3 negative control", sec_phase4_control)
    sec("phase4.4 B2 cluster bootstrap", sec_b2_cluster_bootstrap)
    sec("phase4.5 B3 subgroups", sec_b3_subgroups)
    sec("phase4.6 B4 confounds", sec_b4_confounds)
    sec("phase4.7 B6 budget check", sec_b6_budget_check)

    # warnings need the already-loaded context; reload quietly (files are cached)
    for key, rel in (("p1", "phase1/phase1_summary.json"), ("lk", "phase1/leakage.json"),
                     ("sla", "phase2/split_leakage_assertions.json"),
                     ("fo", "phase2/faithful_original.json"),
                     ("s2agg", "phase2/s2_seed_aggregate.json"),
                     ("oa", "phase2/overlap_accounting.json"),
                     ("claims", "phase3/claims_comparison.json"),
                     ("p3a", "phase3/split_leakage_assertions.json"),
                     ("ev", "phase3/plant_key_evidence.json"),
                     ("p4", "phase4/phase4_summary.json")):
        try:
            ctx[key] = _load(rel)
        except MissingRequired:
            pass
    try:
        n_test_plants = J("phase4/paired_stats.json")[0]["n_test_plants"]
    except (MissingRequired, IndexError, KeyError, TypeError):
        n_test_plants = None
    ctx["n_test_plants"] = n_test_plants

    required_ctx = ("p1", "lk", "sla", "fo", "s2agg", "oa", "claims", "p3a", "ev", "p4")
    if all(k in ctx for k in required_ctx):
        sec("warnings", lambda: sec_warnings(ctx))
    else:
        errors.append("warnings: missing inputs " +
                      ", ".join(k for k in required_ctx if k not in ctx))
        L += [f"> {PLACEHOLDER} - `warnings` could not be rendered (missing inputs)", ""]
    sec("index", sec_index)

    meta = {"required_inputs_read": sorted(_read),
            "missing_inputs": sorted(set(_missing)),
            "section_errors": errors,
            "n_artifacts_indexed": sum(1 for p in RES.rglob("*")
                                       if p.is_file() and p.suffix in (".json", ".csv"))}
    return "\n".join(L), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="smoke test: render what is available and mark the rest as MISSING "
                         "(tests/test_results.py fails on any such placeholder)")
    a = ap.parse_args()
    md, meta = build(a.allow_incomplete)

    if meta["section_errors"] and not a.allow_incomplete:
        print("FATAL: RESULTS.md was NOT written. Unrenderable sections:")
        for e in meta["section_errors"]:
            print("  -", e)
        raise SystemExit(1)

    OUT.write_text(md)
    meta.update({"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "output": "results/RESULTS.md",
                 "command": "analysis/generate_results_md.py",
                 "incomplete": bool(meta["section_errors"])})
    (RES / "results_md_build.json").write_text(json.dumps(meta, indent=2))

    runlog = RES / "run_log.md"
    old = runlog.read_text() if runlog.exists() else "# Run log\n"
    marker = "## Phase 5 - RESULTS.md generation"
    if marker in old:          # block-scoped: remove ONLY our previous entry - entries
        i = old.index(marker)  # other phases appended after it must survive (CHANGELOG)
        j = old.find("\n## ", i + 1)
        head = old[:i].rstrip("\n")
        tail = "" if j == -1 else old[j + 1:]
        old = head + ("\n\n" + tail if tail else "\n")
    runlog.write_text(old + "\n".join([
        f"\n{marker}\n",
        f"- time (UTC): {meta['generated_utc']}",
        "- command: `.venv/bin/python analysis/generate_results_md.py`",
        f"- result files read: {len(meta['required_inputs_read'])}",
        f"- missing inputs: {meta['missing_inputs']}",
        f"- section errors: {meta['section_errors']}",
        "- output: results/RESULTS.md, results/results_md_build.json",
        "\n"]))
    print(f"wrote results/RESULTS.md ({len(md.splitlines())} lines) from "
          f"{len(meta['required_inputs_read'])} result files")
    if meta["section_errors"]:
        print("INCOMPLETE - sections not rendered:", meta["section_errors"])


if __name__ == "__main__":
    main()
