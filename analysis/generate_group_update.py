"""Separate Group A/B progress + results file (the user's "updates in another file").

  .venv/bin/python analysis/generate_group_update.py

HARD RULE 1: every number written to results/GROUP_A_B_UPDATE.md is READ from a
json/csv under results/ or from the analysis input files - none is typed here.
Steps whose outputs do not exist yet are listed as PENDING (no numbers invented,
HARD RULE 6/8); re-run this script any time - it is idempotent.

Also records, as plain status text (not results), whether the two serialized
chains are running/done by reading their logs.

Writes: results/GROUP_A_B_UPDATE.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P2, P3, P4 = RES / "phase2", RES / "phase3", RES / "phase4"


def has(rel: str) -> bool:
    return (RES / rel).exists()


def J(rel: str) -> dict:
    return json.loads((RES / rel).read_text())


def r(x, dp: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{dp}f}"


def ci(v) -> str:
    return f"[{r(v[0], 4)}, {r(v[1], 4)}]" if v else "n/a"


def table(hdr: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(hdr) + " |",
            "| " + " | ".join("---" for _ in hdr) + " |"] + \
           ["| " + " | ".join(str(c) for c in row) + " |" for row in rows] + [""]


def pending(item: str, needed: str) -> list[str]:
    return [f"- **{item}**: PENDING - waiting on `{needed}`.", ""]


def chain_status() -> list[str]:
    L = ["### Run status (text read from the chain logs)", ""]
    for name, log in (("Group A chain", "groupA_chain.log"),
                      ("Group B tail chain", "groupB_tail.log")):
        p = ROOT / "analysis" / log
        if not p.exists():
            L.append(f"- {name}: not started.")
            continue
        lines = [l for l in p.read_text().splitlines() if l.startswith("===")]
        if any("COMPLETE" in l for l in lines):
            L.append(f"- {name}: COMPLETE. " + (lines[-1] if lines else ""))
        elif lines:
            L.append(f"- {name}: running - last milestone: `{lines[-1]}`")
        else:
            L.append(f"- {name}: started (no milestone yet).")
    L.append("")
    return L


def sec_a1() -> list[str]:
    L = ["## A1 - leakage-stratified accuracy of the published checkpoint", ""]
    if not has("phase2/leakage_stratified_accuracy.json"):
        return pending("A1", "results/phase2/leakage_stratified_accuracy.json")
    d = J("phase2/leakage_stratified_accuracy.json")
    L += [f"Source `{d['split']}`, distance `{d['distance_metric']}`, "
          f"{d['ci_method']} ({d['n_bootstrap']} draws), {d['permutation_test']} "
          f"({d['n_permutations']} draws, seed {d['seed']}). Group counts match the "
          f"Phase-1 audit at every threshold: "
          f"**{all(v['crosscheck_matches_phase1'] for v in d['thresholds'].values())}**.",
          ""]
    rows = []
    for t in sorted(d["thresholds"], key=int):
        v = d["thresholds"][t]
        rows.append([t, v["n_with_near_dup_in_train"], r(v["accuracy_with"], 4),
                     ci(v["accuracy_with_ci95"]), v["n_without"],
                     r(v["accuracy_without"], 4), ci(v["accuracy_without_ci95"]),
                     r(v["difference_with_minus_without"], 4),
                     r(v["permutation_p_two_sided"], 4)])
    return L + table(["t", "n with", "acc with", "95% CI", "n without", "acc without",
                      "95% CI", "diff", "perm p"], rows)


def sec_a2() -> list[str]:
    L = ["## A2 - S1 seeds beside S2 seeds", ""]
    if not has("phase2/seed_comparison_s1_vs_s2.json"):
        return pending("A2", "results/phase2/seed_comparison_s1_vs_s2.json")
    d = J("phase2/seed_comparison_s1_vs_s2.json")
    rows = []
    for k, name in (("S1_original_split", "S1 (original)"),
                    ("S2_group_disjoint", "S2 (group-disjoint)")):
        a = d[k]
        rows.append([name, a["n_seeds"], r(a["accuracy_mean"]), r(a["accuracy_sd"]),
                     r(a["macro_f1_mean"]), r(a["macro_f1_sd"])])
    rows.append(["difference (S1 - S2)", "", r(d["mean_difference_S1_minus_S2_accuracy"]),
                 "", "", ""])
    det = d["determinism_check_seed42"]
    return L + table(["split", "seeds", "acc mean", "acc sd", "macro-F1 mean",
                      "macro-F1 sd"], rows) + \
           [f"Determinism: seed-42 rerun bit-identical accuracy "
            f"**{det['bit_identical_accuracy']}**.", ""]


def sec_a3() -> list[str]:
    L = ["## A3 - S2 rebuilt at looser near-duplicate thresholds", ""]
    agg_csv = "phase2/threshold_sweep_all_aggregate.csv"
    rows = []
    if has(agg_csv):
        import pandas as pd
        for _, x in pd.read_csv(RES / agg_csv).iterrows():
            rows.append([int(x["threshold"]), int(x["n_seeds"]), r(x["accuracy_mean"]),
                         r(x["accuracy_sd"]), r(x["macro_f1_mean"]), r(x["macro_f1_sd"])])
    else:
        # partial view from whatever per-threshold aggregates exist so far
        for t, f in ((3, "phase2/s2_seed_aggregate.json"),
                     (4, "phase2/thresh4_aggregate.json"),
                     (6, "phase2/thresh6_aggregate.json")):
            if has(f):
                a = J(f)
                # t=3's older aggregate uses *_std, the thresh aggregates use *_sd
                sd = a.get("accuracy_sd", a.get("accuracy_std"))
                f1sd = a.get("macro_f1_sd", a.get("macro_f1_std"))
                rows.append([t, a["n_seeds"], r(a["accuracy_mean"]), r(sd),
                             r(a["macro_f1_mean"]), r(f1sd)])
            else:
                rows.append([t, "-", "PENDING", "-", "-", "-"])
        L += [f"> Aggregate CSV not written yet (needs A5b); rows below are partial.", ""]
    L += table(["near-dup t", "seeds", "acc mean", "acc sd", "macro-F1 mean",
                "macro-F1 sd"], rows)
    for t in (4, 6):
        f = f"phase2/thresh{t}_split_assertions.json"
        if has(f):
            a = J(f)["assertions"]
            L.append(f"- t={t} leakage assertions: "
                     + ", ".join(f"{k}={v}" for k, v in a.items()))
    L.append("")
    return L


def sec_a4() -> list[str]:
    L = ["## A4 - multi-seed architecture comparison (validation-only selection)", ""]
    if not has("phase2/arch3_selection.json"):
        return pending("A4", "results/phase2/arch3_selection.json (chain step 3)")
    sel = J("phase2/arch3_selection.json")
    rows = [[a + (" (selected)" if a == sel["selected_architecture"] else ""),
             r(c["val_acc_mean"]), r(c["val_acc_sd"]), r(c["test_acc_mean"]),
             r(c["test_acc_sd"]), r(c["test_macro_f1_mean"]), r(c["test_macro_f1_sd"])]
            for a, c in sel["candidates"].items()]
    return L + [f"Criterion: {sel['criterion']}; selection used test data: "
                f"{sel['selection_uses_test_data']}; {sel['n_seeds_per_arch']} seeds/arch, "
                f"{sel['epochs']} epochs.", ""] + \
           table(["architecture", "val acc mean", "val acc sd", "test acc mean",
                  "test acc sd", "test macro-F1 mean", "test macro-F1 sd"], rows)


def sec_a5() -> list[str]:
    L = ["## A5 - majority baseline, calibration, metric audit, gallery, excl-48", ""]
    if not has("phase2/spec_extras_summary.json"):
        return pending("A5b (extras)", "results/phase2/spec_extras_summary.json "
                       "(needs chain step 4 = excl48 first)")
    mb, nm, ts, eg = (J(f"phase2/{n}.json") for n in
                      ("majority_baseline", "numpy_metric_check",
                       "temperature_scaling", "error_gallery_eb_lmf"))
    L += table(["split", "majority class", "test images", "baseline acc"], [
        ["S1 (original)", mb["S1_original_split"]["majority_class"],
         mb["S1_original_split"]["n_test"], r(mb["S1_original_split"]["test_accuracy"])],
        ["S2 (group-disjoint)", mb["S2_group_disjoint"]["majority_class"],
         mb["S2_group_disjoint"]["n_test"], r(mb["S2_group_disjoint"]["test_accuracy"])],
    ])
    L += [f"Numpy-only metric recheck: max |diff| **{r(nm['max_abs_diff'], 12)}** "
          f"(tolerance {r(nm['tolerance'], 12)}) -> pass **{nm['pass']}**.",
          "",
          f"Temperature scaling (fit on validation): T = {r(ts['temperature'], 4)}; "
          f"ECE {r(ts['ece_before'], 4)} -> {r(ts['ece_after'], 4)} "
          f"({ts['ece_bins']} bins); accuracy {r(ts['test_accuracy_before'], 4)} -> "
          f"{r(ts['test_accuracy_after'], 4)} (unchanged: "
          f"{ts['accuracy_unchanged_by_scaling']}).",
          ""]
    rows = [[k.replace("_", " "), p["n_eb_to_lmf"], p["n_lmf_to_eb"],
             p["n_total_swaps"]] for k, p in eg["panels"].items()]
    L += table(["panel", "true EB -> LMF", "true LMF -> EB", "total swaps"], rows)
    if has("phase2/excl48_comparison.json"):
        ex = J("phase2/excl48_comparison.json")
        L += table(["side", "acc mean", "acc sd", "macro-F1 mean", "macro-F1 sd"], [
            ["baseline S1", r(ex["baseline_S1"]["accuracy_mean"]),
             r(ex["baseline_S1"]["accuracy_sd"]), r(ex["baseline_S1"]["macro_f1_mean"]),
             r(ex["baseline_S1"]["macro_f1_sd"])],
            [f"S1 minus {ex['exclusion']['n_excluded']} images",
             r(ex["excluded_S1"]["accuracy_mean"]), r(ex["excluded_S1"]["accuracy_sd"]),
             r(ex["excluded_S1"]["macro_f1_mean"]), r(ex["excluded_S1"]["macro_f1_sd"])],
        ])
        L += [f"Change: accuracy {r(ex['change_accuracy_mean'])}, "
              f"macro-F1 {r(ex['change_macro_f1_mean'])}.", ""]
    return L


def sec_b1() -> list[str]:
    L = ["## B1 - root-cause of the prevalence-by-angle claim", ""]
    if not has("phase3/claim_rootcause/claim_rootcause.json"):
        return pending("B1", "results/phase3/claim_rootcause/claim_rootcause.json")
    rc = J("phase3/claim_rootcause/claim_rootcause.json")
    rows = []
    for ang in ("0", "5", "15"):
        b = rc["implied_numerator_bookkeeping"][ang]
        rows.append([f"{ang} deg", b["audited_numerator"], b["audited_denominator"],
                     r(b["claim_pct"], 1), r(b["numerator_implied_by_claim"], 2),
                     r(b["gap_images"], 2)])
    L += table(["angle", "measured num", "measured den", "claim %",
                "num implied by claim", "gap (images)"], rows)
    L += [f"**Verdict: {rc['verdict']}** - {len(rc['variants'])} alternative "
          f"definitions tested; none reproduces all three claimed angles "
          f"(details: results/phase3/claim_rootcause/claim_rootcause.json).", ""]
    return L


def sec_b2() -> list[str]:
    L = ["## B2 - image-level mode comparison (plant-cluster bootstrap)", ""]
    if not has("phase4/image_level_cluster_bootstrap.json"):
        return pending("B2", "results/phase4/image_level_cluster_bootstrap.json")
    d = J("phase4/image_level_cluster_bootstrap.json")
    bs = d["bootstrap"]
    rows = [[m, r(a["acc_mean"]), r(a["acc_sd"]), r(a["auc_mean"]), r(a["auc_sd"])]
            for m, a in d["aggregate_point_estimates_across_seeds"].items()]
    L += [f"Cluster bootstrap: unit `{bs['unit']}`, {bs['n']} draws, seed "
          f"{bs['rng_seed']}, {bs['ci']}.", ""]
    L += table(["mode", "image acc mean", "sd", "image AUC mean", "sd"], rows)
    rows = []
    for key, v in d["paired_cluster_bootstrap_differences"].items():
        seed, comp = key.split("/")
        rows.append([seed, comp.replace("_", " "), r(v["acc_difference"], 4),
                     ci(v["acc_diff_ci95"]), r(v["acc_p_boot_le_zero"], 4),
                     r(v["auc_difference"], 4), ci(v["auc_diff_ci95"])])
    L += table(["seed", "comparison", "d acc", "d acc 95% CI", "p boot<=0",
                "d AUC", "d AUC 95% CI"], rows)
    return L


def sec_b3() -> list[str]:
    L = ["## B3 - accuracy/AUC by day, angle, batch", ""]
    if not has("phase4/subgroup_metrics_day_angle_batch.csv"):
        return pending("B3", "results/phase4/subgroup_metrics_day_angle_batch.csv")
    import pandas as pd
    df = pd.read_csv(P4 / "subgroup_metrics_day_angle_batch.csv")
    meta = J("phase4/subgroup_metrics_day_angle_batch.json")
    L += [f"{meta['rows']} cells (mode x seed x factor level); full table: "
          f"`results/phase4/subgroup_metrics_day_angle_batch.csv`. Angle and batch "
          f"rows here (day rows are the long tail in the CSV):", ""]
    rows = []
    for fac in ("angle", "batch"):
        for _, x in df[df["factor"] == fac].iterrows():
            rows.append([fac, x["cell"], int(x["seed"]), x["mode"],
                         int(x["n_images"]), int(x["n_plants"]), int(x["n_pos"]),
                         r(x["accuracy"], 4),
                         r(x["auc"], 4) if bool(x["auc_defined"]) else "n/a"])
    return L + table(["factor", "cell", "seed", "mode", "images", "plants", "pos",
                      "acc", "AUC"], rows)


def sec_b4() -> list[str]:
    L = ["## B4 - confound baselines + image-level vs plant-disjoint leakage demo", ""]
    if not has("phase4/confounds/confound_baselines.json"):
        return pending("B4", "results/phase4/confounds/confound_baselines.json")
    d = J("phase4/confounds/confound_baselines.json")
    rows = []
    for fs, s in d["summary"].items():
        rows.append([fs, "plant-disjoint", r(s["plant_disjoint_acc_mean"]),
                     r(s["plant_disjoint_acc_sd"]), r(s["plant_disjoint_auc_mean"])])
        rows.append([fs, "image-level (leaky demo)", r(s["image_level_acc_mean"]),
                     r(s["image_level_acc_sd"]), r(s["image_level_auc_mean"])])
    L += table(["features", "protocol", "acc mean", "acc sd", "AUC mean"], rows)
    rows = [[fs, r(s["acc_inflation_image_level_minus_plant_disjoint"]),
             r(s["auc_inflation_image_level_minus_plant_disjoint"])]
            for fs, s in d["summary"].items()]
    L += table(["features", "acc inflation", "AUC inflation"], rows)
    L += ["Leakage-check exit codes: "
          + ", ".join(f"{k}={v}" for k, v in d["leakage_check_exit_codes"].items())
          + " (plant-disjoint must be 0; image-level 1 = leakage detected = the demo).",
          ""]
    return L


def sec_b5() -> list[str]:
    L = ["## B5 - label provenance (the dataset's own article)", ""]
    p = ROOT / "analysis" / "dataset_article_reference.json"
    if not p.exists():
        return pending("B5", "analysis/dataset_article_reference.json")
    a = json.loads(p.read_text())
    cit = a["citation"]
    L += [f"{cit['authors']} \"{cit['title']}\" {cit['journal']} "
          f"doi:{cit['doi']} ({cit['pmcid']}).",
          ""]
    for q in a["how_labels_were_assigned"]:
        L.append(f"- **{q['section']}** - \"{q['quote']}\"")
    L += ["",
          f"**Symptom-onset information exists? "
          f"{a['symptom_onset_information_exists']['our_statement']}**", ""]
    return L


def sec_b6() -> list[str]:
    L = ["## B6 - higher-budget check (fusion, 224 px, 15 epochs, 1 seed)", ""]
    if not has("phase4/budget_check_px224_e15/fusion/metrics.json"):
        return pending("B6", "results/phase4/budget_check_px224_e15/fusion/metrics.json")
    m = J("phase4/budget_check_px224_e15/fusion/metrics.json")
    cfg = J("phase4/budget_check_px224_e15/run_config.json")
    p4 = J("phase4/phase4_summary.json")
    base = p4["aggregate_by_mode"]["fusion"]
    return L + table(["setting", "image acc", "image AUC", "plant acc", "plant AUC"], [
        [f"fusion, minimal budget ({p4['settings']['epochs']} epochs, "
         f"{p4['settings']['image_size']} px, 3 seeds)",
         r(base["image_acc_mean"]), r(base["image_auc_mean"]),
         r(base["plant_acc_mean"]), r(base["plant_auc_mean"])],
        [f"fusion, this run ({cfg['epochs']} epochs, {cfg['image_size']} px, "
         f"seed {m.get('seed')})", r(m["image_acc"]), r(m["image_auc"]),
         r(m["plant_acc"]), r(m["plant_auc"])],
    ])


def main() -> None:
    out = ["# Group A / Group B update - machine-generated status + results", "",
           f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} by "
           f"`analysis/generate_group_update.py`. Every number below is read from "
           f"files under `results/` or from `analysis/dataset_article_reference.json`; "
           f"PENDING means the producing step has not finished yet - no number is "
           f"estimated. The authoritative report is `results/RESULTS.md`.", ""]
    out += chain_status()
    for fn in (sec_a1, sec_a2, sec_a3, sec_a4, sec_a5,
               sec_b1, sec_b2, sec_b3, sec_b4, sec_b5, sec_b6):
        out += fn()
        out += ["---", ""]
    (RES / "GROUP_A_B_UPDATE.md").write_text("\n".join(out))
    pend = sum(1 for l in out if "PENDING" in l)
    print(f"wrote results/GROUP_A_B_UPDATE.md ({len(out)} lines, {pend} pending items)")


if __name__ == "__main__":
    main()
