"""HARD RULE 5 - append results/run_log.md entries for the Group A / Group B runs.

  .venv/bin/python analysis/append_group_run_log.py

Block-scoped and idempotent (each `## <marker>` block is replaced in place, other
entries - including ones appended after it - survive; same pattern as the fixed
rewriters, see CHANGELOG). Every runtime/number in the entries is READ from the
corresponding result json (HARD RULE 1); commands are the literal invocations.
Aborts before writing anything if a source file is missing (HARD RULE 6).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P1, P2, P3, P4 = RES / "phase1", RES / "phase2", RES / "phase3", RES / "phase4"


def J(p: Path) -> dict:
    if not p.exists():
        raise SystemExit(f"FATAL: missing {p} - run that step first (HARD RULE 6)")
    return json.loads(p.read_text())


def build_entries() -> list[tuple[str, list[str]]]:
    e: list[tuple[str, list[str]]] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- Group A ---------------------------------------------------------------
    a1 = J(P2 / "leakage_stratified_accuracy.json")
    e.append(("Group A1 - leakage-stratified accuracy of the published checkpoint", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset1_phase2_leakage_stratified.py`",
        "- exit code: 0 (first attempt aborted by its own 450-row assertion because "
        "`phase1/test_nearest_train.csv` also holds validation rows - fixed to filter "
        "`split == test`; see CHANGELOG)",
        f"- result: thresholds {', '.join(sorted(a1['thresholds'], key=int))} - "
        f"with/without-near-duplicate accuracy, bootstrap CIs "
        f"({a1['n_bootstrap']} draws) and permutation p "
        f"({a1['n_permutations']} draws), all counts cross-checked against Phase 1",
        f"- runtime: {a1['runtime_s']} s",
        "- outputs: results/phase2/leakage_stratified_accuracy.{json,csv}",
    ]))

    a2 = J(P2 / "seed_comparison_s1_vs_s2.json")
    e.append(("Group A2 - S1 vs S2 seed variance (same recipe)", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset1_phase2_seed_comparison.py`",
        "- exit code: 0 (reads existing jsons only - no training)",
        f"- result: S1 {a2['S1_original_split']['n_seeds']} seeds acc "
        f"{a2['S1_original_split']['accuracy_mean']} ± {a2['S1_original_split']['accuracy_sd']} "
        f"vs S2 {a2['S2_group_disjoint']['n_seeds']} seeds acc "
        f"{a2['S2_group_disjoint']['accuracy_mean']} ± "
        f"{a2['S2_group_disjoint']['accuracy_sd']}; difference "
        f"{a2['mean_difference_S1_minus_S2_accuracy']}",
        f"- runtime: {a2['runtime_s']} s",
        "- outputs: results/phase2/seed_comparison_s1_vs_s2.{json,csv}",
    ]))

    for t in (4, 6):
        agg = J(P2 / f"thresh{t}_aggregate.json")
        asrt = J(P2 / f"thresh{t}_split_assertions.json")
        e.append((f"Group A3 - S2 rebuilt at near-dup threshold {t} (3 seeds)", [
            f"- time (UTC): {now}",
            "- command: `.venv/bin/python analysis/dataset1_phase2_threshold_sweep.py "
            f"--threshold {t}`",
            "- exit code: 0 (first attempt for t=4 aborted on a KeyError in the split "
            "assertion helper - split CSVs already carry sha256/group; fixed, see CHANGELOG)",
            f"- result: split {agg['split_sizes']} - assertions "
            f"{json.dumps(asrt['assertions'], sort_keys=True)}; mean acc "
            f"{agg['accuracy_mean']} ± {agg['accuracy_sd']} over "
            f"{agg['n_seeds']} seeds, mean macro-F1 {agg['macro_f1_mean']} ± "
            f"{agg['macro_f1_sd']}",
            f"- runtime: {agg['runtime_s']} s (split build + 3 training runs)",
            f"- outputs: results/phase2/thresh{t}_seed*.json, "
            f"results/phase2/thresh{t}_split_assertions.json, "
            f"results/splits_S2_t{t}/",
        ]))

    a4 = J(P2 / "arch3_selection.json")
    e.append(("Group A4 - multi-seed architecture comparison (mobilenet_v3_large, "
              "efficientnet_b0; 3 seeds each)", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset1_phase2_arch_multiseed.py`",
        "- exit code: 0",
        f"- result: selected `{a4['selected_architecture']}` on validation only "
        f"(criterion: {a4['criterion']}; selection used test data: "
        f"{a4['selection_uses_test_data']})",
        f"- runtime: {a4['runtime_s']} s (6 training runs)",
        "- outputs: results/phase2/arch3_*.json, results/phase2/arch3_selection.json",
    ]))

    excl = J(RES / "splits_S1_excl48" / "report.json")
    per_seed = {s: J(P2 / f"excl48_seed{s}.json") for s in (42, 43, 44)}
    rt = sum(v["runtime_s"] for v in per_seed.values())
    e.append(("Group A5a - re-train S1 excluding the 48 high-confidence cross-label images", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset1_phase2_excl48.py`",
        "- exit code: 0",
        f"- result: excluded {excl['n_excluded']} images ({excl['method']}), per split "
        + ", ".join(f"{k}={v}" for k, v in excl["counts_per_split"].items())
        + "; per-seed test accuracy "
        + ", ".join(f"seed{s}={v['accuracy']}" for s, v in per_seed.items()),
        f"- runtime: {round(rt, 1)} s (sum of the 3 recorded run runtimes)",
        "- outputs: results/splits_S1_excl48/, results/phase2/excl48_seed*.json",
    ]))

    a5 = J(P2 / "spec_extras_summary.json")
    e.append(("Group A5b - spec extras (majority baseline, numpy metric audit, "
              "temperature scaling, error gallery, threshold trend)", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset1_phase2_spec_extras.py`",
        "- exit code: 0",
        f"- result: numpy recomputation pass = "
        f"{J(P2 / 'numpy_metric_check.json')['pass']}; temperature "
        f"{J(P2 / 'temperature_scaling.json')['temperature']}; outputs "
        f"{', '.join(sorted(k for k in a5 if k != 'runtime_s'))}",
        f"- runtime: {a5['runtime_s']} s",
        "- outputs: results/phase2/{majority_baseline,numpy_metric_check,"
        "temperature_scaling,error_gallery_eb_lmf,excl48_comparison,"
        "threshold_sweep_trend}.json, results/phase2/threshold_sweep_all*.csv",
    ]))

    # ---- Group B ---------------------------------------------------------------
    b1 = J(P3 / "claim_rootcause" / "claim_rootcause.json")
    e.append(("Group B1 - root-cause of the prevalence-by-angle claim FAIL", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset2_phase3_claim_rootcause.py`",
        "- exit code: 0 (first attempt aborted on its own tuple-key column-order check "
        "- fixed, see CHANGELOG)",
        f"- result: {b1['verdict']}; {len(b1['variants'])} definitions tested; "
        f"15 deg measured {b1['implied_numerator_bookkeeping']['15']['audited_numerator']}"
        f"/{b1['implied_numerator_bookkeeping']['15']['audited_denominator']} vs claim "
        f"{b1['claim']['15']}",
        f"- runtime: {b1['runtime_s']} s",
        "- outputs: results/phase3/claim_rootcause/claim_rootcause.{json,csv}",
    ]))

    b2 = J(P4 / "image_level_cluster_bootstrap.json")
    e.append(("Group B2/B3 - image-level cluster bootstrap + per day/angle/batch", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset2_phase4_subgroup_analysis.py`",
        "- exit code: 0",
        f"- result: plant-cluster bootstrap ({b2['bootstrap']['n']} draws, unit "
        f"{b2['bootstrap']['unit']}); aggregate acc by mode "
        + ", ".join(f"{m}={a['acc_mean']}"
                    for m, a in b2["aggregate_point_estimates_across_seeds"].items())
        + f"; subgroup cells: {J(P4 / 'subgroup_metrics_day_angle_batch.json')['rows']}",
        f"- runtime: {b2['runtime_s']} s",
        "- outputs: results/phase4/image_level_cluster_bootstrap.json, "
        "results/phase4/subgroup_metrics_day_angle_batch.{json,csv}",
    ]))

    b4 = J(P4 / "confounds" / "confound_baselines.json")
    e.append(("Group B4 - confound baselines + image-level vs plant-disjoint demo", [
        f"- time (UTC): {now}",
        "- command: `.venv/bin/python analysis/dataset2_phase4_confound_baselines.py`",
        "- exit code: 0",
        f"- result: leakage check exit codes "
        f"{json.dumps(b4['leakage_check_exit_codes'], sort_keys=True)}; "
        + "; ".join(f"{k}: acc inflation {v['acc_inflation_image_level_minus_plant_disjoint']}"
                    for k, v in b4["summary"].items()),
        f"- runtime: {b4['runtime_s']} s",
        "- outputs: results/phase4/confounds/confound_baselines.{json,csv}, "
        "results/phase4/confounds/leakage_checks/",
    ]))

    art = json.loads((ROOT / "analysis" / "dataset_article_reference.json").read_text())
    cit = art["citation"]
    e.append(("Group B5 - dataset article quotes (label provenance)", [
        f"- time (UTC): {now}",
        f"- source fetched: {cit['fetched_from']} (doi:{cit['doi']}, {cit['pmcid']})",
        f"- fetched on: {cit['fetched_on']}",
        "- output: analysis/dataset_article_reference.json (INPUT file - quotes with "
        "section references; used by the report renderer for warning 1's correction)",
        "- note: no runtime - web fetch, not a computation",
    ]))

    return e


def main() -> None:
    entries = build_entries()          # all-or-nothing: missing file aborts before writing
    log = RES / "run_log.md"
    old = log.read_text()
    for marker, lines in entries:
        full = "## " + marker
        if full in old:                # block-scoped: remove ONLY our previous block
            i = old.index(full)
            j = old.find("\n## ", i + 1)
            head = old[:i].rstrip("\n")
            tail = "" if j == -1 else old[j + 1:]
            old = head + ("\n\n" + tail if tail else "\n")
        old += ("\n" if not old.endswith("\n") else "") + full + "\n" \
            + "\n".join(lines) + "\n"
    # collapse blank-line pileup from replaced blocks
    while "\n\n\n" in old:
        old = old.replace("\n\n\n", "\n\n")
    log.write_text(old)
    print(f"appended/replaced {len(entries)} run-log blocks in {log}")


if __name__ == "__main__":
    main()
