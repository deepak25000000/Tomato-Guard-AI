"""Phase 5 automated verification.

  python -m pytest tests/ -q

These tests are the machine's answer to "can I trust RESULTS.md?". They check, in order:
  1. the report exists, is complete, and was produced by the generator (no hand edits);
  2. every number printed in a RESULTS.md table can be traced back to a results json/csv
     (HARD RULE 1 - nothing typed by hand);
  3. HARD RULE 2: the raw typo'd folder name is untouched on disk and never used as a
     display label in the report;
  4. HARD RULE 4: every leakage assertion recorded by the pipeline is PASS;
  5. HARD RULE 8: the forbidden phrases do not appear anywhere in the report;
  6. the headline claims that this analysis set out to verify actually reproduce.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORT = RES / "RESULTS.md"

FORBIDDEN = ["pre-symptomatic", "early detection", "field-ready"]
TYPO = "Bacteiral Spot"
# files under results/ that could not be parsed while walking (should be empty at Phase 5,
# when no phase is writing any more; a non-empty list always names the offending file)
SOURCE_READ_ERRORS: list[str] = []


# ---------------------------------------------------------------- helpers
def load(rel: str):
    p = RES / rel
    assert p.exists(), f"missing results/{rel} - was the owning phase run?"
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


def collect_numbers(node, out: set[float]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            collect_numbers(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            collect_numbers(v, out)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        if math.isfinite(float(node)):
            out.add(float(node))
    elif isinstance(node, str):
        try:
            f = float(node)
        except ValueError:
            return
        if math.isfinite(f):
            out.add(f)


def source_numbers() -> set[float]:
    """Every numeric value stored anywhere under results/ (json values + csv cells)."""
    out: set[float] = set()
    for p in sorted(RES.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(RES)
        if p.suffix == ".json":
            # skip our own build manifest: it stores counts of files, not results
            if p.name == "results_md_build.json":
                continue
            try:
                collect_numbers(json.loads(p.read_text()), out)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                SOURCE_READ_ERRORS.append(f"{rel}: {type(e).__name__}: {e}")
                continue
        elif p.suffix == ".csv":
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                # whitespace-only CSVs are EMPTY EVIDENCE FILES by design (e.g.
                # phase1/unreadable.csv = "0 unreadable images", a single newline) and
                # carry no numbers; anything else with content is a real read failure
                if p.read_text().strip():
                    SOURCE_READ_ERRORS.append(
                        f"{rel}: EmptyDataError on non-whitespace content")
                continue
            except Exception as e:  # noqa: BLE001 - record, never swallow
                SOURCE_READ_ERRORS.append(f"{rel}: {type(e).__name__}: {e}")
                continue
            for col in df.columns:
                for v in df[col].tolist():
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)):
                        if math.isfinite(float(v)):
                            out.add(float(v))
                    elif isinstance(v, str):
                        try:
                            f = float(v)
                        except ValueError:
                            continue
                        if math.isfinite(f):
                            out.add(f)
    return out


def table_rows(md: str):
    """Yield (row_index, [cells]) for the data rows of every markdown table."""
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and \
                set(lines[i + 1].replace("|", "").replace(" ", "")) <= {"-", ""}:
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                yield i, [c.strip() for c in lines[i].strip("|").split("|")]
                i += 1
            continue
        i += 1


def cell_numbers(cell: str) -> list[tuple[str, float]]:
    """Numbers in a cell, with the decimal precision the report printed."""
    c = cell.strip().strip("*")
    if not re.fullmatch(r"[\d\.\s,\[\]\-]+", c) or not re.search(r"\d", c):
        return []
    out = []
    for tok in re.split(r"[\s,\[\]]+", c):
        if re.fullmatch(r"-?\d+\.\d+", tok):
            out.append((tok, float(tok)))
        elif re.fullmatch(r"-?\d+", tok):
            out.append((tok, float(tok)))
    return out


# ---------------------------------------------------------------- tests
def test_report_exists_and_is_complete():
    assert REPORT.exists(), "results/RESULTS.md missing - run analysis/generate_results_md.py"
    md = REPORT.read_text()
    assert "SECTION NOT RENDERED" not in md, \
        "RESULTS.md contains unrendered sections - regenerate after all phases finish"
    build = load("results_md_build.json")
    assert build["incomplete"] is False, f"report built incomplete: {build['section_errors']}"
    assert build["missing_inputs"] == [], f"missing inputs: {build['missing_inputs']}"
    assert build["section_errors"] == [], f"section errors: {build['section_errors']}"


def test_every_table_number_is_traceable_to_a_result_file():
    """HARD RULE 1: no result number may be typed by hand."""
    md = REPORT.read_text()
    src = source_numbers()
    assert not SOURCE_READ_ERRORS, \
        "result file(s) unreadable while collecting sources: " + "; ".join(SOURCE_READ_ERRORS)
    assert len(src) > 100, f"only {len(src)} numbers found under results/ - walker is broken"
    untraced: list[str] = []
    for lineno, cells in table_rows(md):
        for cell in cells:
            for text, val in cell_numbers(cell):
                d = len(text.split(".")[1]) if "." in text else 0
                if any(abs(round(y, d) - val) < 1e-9 for y in src):
                    continue
                untraced.append(f"line {lineno}: {text!r} (cell {cell!r})")
    assert not untraced, "numbers in RESULTS.md with no source in results/: " + \
                         "; ".join(untraced[:15])


def test_no_forbidden_phrases():
    """HARD RULE 8 - none of these may appear anywhere in the report."""
    md = REPORT.read_text().lower()
    hits = [p for p in FORBIDDEN if p in md]
    assert not hits, f"forbidden phrase(s) in RESULTS.md: {hits}"


def test_typo_folder_untouched_and_never_a_display_label():
    """HARD RULE 2 - raw data is not renamed, and the typo never reaches a display label."""
    typo_dirs = list((ROOT / "Tomato Leaf Disease Classification Dataset").rglob(TYPO))
    assert typo_dirs, f"raw folder {TYPO!r} was renamed or deleted - HARD RULE 2 violation"
    md = REPORT.read_text()
    assert TYPO not in md, "the raw folder typo appears in RESULTS.md as a display label"
    # the mapping must exist in the generated class-count table instead
    assert "Bacterial Spot" in md


def test_all_leakage_assertions_pass():
    """HARD RULE 4 - a failing assertion must stop the run, so none may be recorded as FAIL."""
    for rel in ("phase2/split_leakage_assertions.json", "phase3/split_leakage_assertions.json"):
        rep = load(rel)
        bad = {k: v for k, v in rep["assertions"].items()
               if not (str(v) == "PASS" or str(v).startswith("INFO"))}
        assert not bad, f"{rel} has failing assertions: {bad}"


def test_split_sizes_add_up():
    ss = load("phase2/split_sizes.json")
    for split in ("S1", "S2"):
        total = sum(ss[split][p] for p in ("train", "validation", "test"))
        assert total == 2995, f"{split} has {total} images, expected 2995"
    p3 = load("phase3/split_leakage_assertions.json")
    per_split = sum(v for k, v in p3["seeds"]["42"]["images"].items())
    assert per_split == p3["n_images"], \
        f"Track B split images {per_split} != manifest images {p3['n_images']}"


def test_published_baseline_reproduced_exactly():
    fo = load("phase2/faithful_original.json")
    assert fo["abs_diff_vs_published"] <= 1e-9, \
        f"checkpoint re-evaluation differs from the published number: {fo['abs_diff_vs_published']}"
    assert fo["published_acc_inside_this_ci"] is True


def test_negative_controls_are_at_chance():
    """A control that scores well means the pipeline leaks - then nothing may be trusted."""
    ctl = load("phase2/faithful_shuffle_control.json")
    chance = ctl["chance_accuracy_1_over_n_classes"]
    assert ctl["accuracy"] < max(0.35, 3 * chance), \
        f"label-shuffled control scored {ctl['accuracy']} (chance {chance}) - pipeline looks leaky"
    p4 = load("phase4/phase4_summary.json")
    nc = p4["negative_control"]
    assert isinstance(nc, dict), f"Phase 4 negative control not run: {nc}"
    real = max(a["image_auc_mean"] for a in p4["aggregate_by_mode"].values())
    assert 0.2 <= nc["image_auc"] <= 0.8, \
        f"shuffled-label test AUC {nc['image_auc']} is not near chance"
    assert nc["image_auc"] < real, \
        f"control AUC {nc['image_auc']} >= best real model AUC {real}"


def test_trackB_plant_key_reproduces_the_paper_counts():
    ev = load("phase3/plant_key_evidence.json")
    k = ev["reproduced_by_chosen_key"]
    assert k["all_three_claims_match"] is True, \
        f"plant key does not reproduce 167/77/90: {k}"
    assert k["n_plants"] == k["healthy"] + k["infected"]


def test_claim_comparison_is_consistent():
    c = load("phase3/claims_comparison.json")
    assert c["n_checks"] == len(c["checks"])
    assert c["n_pass"] == sum(1 for x in c["checks"] if x["pass"])
    assert 0 <= c["n_pass"] <= c["n_checks"]


def test_environment_and_run_log_are_recorded():
    env = load("environment.json")
    assert (RES / "pip_freeze.txt").exists() and (RES / "pip_freeze.txt").stat().st_size > 0
    assert (RES / "run_log.md").exists()
    log = (RES / "run_log.md").read_text()
    for marker in ("Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"):
        assert marker in log, f"run log has no entry for {marker}"
    assert env["versions"]["torch_cuda_available"] is False, \
        "environment.json no longer matches this CPU-only machine"
    assert (RES / "phase0_benchmark.json").exists()


def test_stream_pairing_defect_documented_and_only_approved_exclusions():
    """HARD RULE 6 + user decision: unpairable tuples are excluded only if approved, loudly."""
    defect = load("phase3/stream_pairing_defect.json")
    approved = json.loads((ROOT / "analysis" / "approved_data_exclusions.json").read_text())
    approved_keys = {e["tuple_key"] for e in approved["exclusions"]}
    assert defect["n_unapproved_incomplete"] == 0, "an unapproved tuple was excluded!"
    assert defect["n_incomplete_tuples"] == len(defect["tuples"]) > 0
    assert {t["tuple_key"] for t in defect["tuples"]} <= approved_keys
    for t in defect["tuples"]:
        assert t["missing_streams"] and t["present_streams"]
    sla = load("phase3/split_leakage_assertions.json")
    assert sla["assertions"]["pairing_exclusions_all_approved"] == "PASS"
    assert sla["n_excluded_pairing_defect"] == len(sla["excluded_pairing_defect"])
    assert set(sla["excluded_pairing_defect"]) <= approved_keys
    # the defect must be stated in the report itself, not buried
    md = REPORT.read_text()
    assert "stream_pairing_defect.json" in md
    assert "Data defect" in md


def test_two_scripts_per_dataset_and_changelog():
    """The protocol asks for two Python files per dataset; edits must be logged (HARD RULE 7)."""
    a = list((ROOT / "analysis").glob("dataset1_*.py"))
    b = list((ROOT / "analysis").glob("dataset2_*.py"))
    assert len(a) >= 2, f"need >=2 dataset1_*.py files, found {len(a)}"
    assert len(b) >= 2, f"need >=2 dataset2_*.py files, found {len(b)}"
    cl = (ROOT / "CHANGELOG.md").read_text()
    assert "Edits to existing scripts" in cl and "fusion.py" in cl


def test_report_states_the_interpretation_limits():
    """The warnings section must carry the caveats the protocol demands."""
    md = REPORT.read_text()
    assert "## 6. Warnings and limitations" in md
    warn = md.split("## 6. Warnings and limitations")[1].split("## 7.")[0]
    assert "S1" in warn and "measured" in warn, "S1 leakage interpretation missing"
    assert "contamination" in warn, "cross-split contamination caveat missing"
    assert "HARD RULE 8" in warn, "the no-unearned-claim statement must be explicit"


def test_no_placeholder_or_todo_text_in_report():
    md = REPORT.read_text()
    for token in ("TODO", "FIXME", "XXX", "PLACEHOLDER", "MISSING"):
        assert token not in md, f"report contains {token!r}"


# ------------------------------------------------- Group A / Group B follow-ups
def test_threshold_sweep_splits_pass_leakage_assertions():
    """HARD RULE 4 - the S2 rebuilds at near-dup thresholds 4 and 6 must assert too."""
    for t in (4, 6):
        rep = load(f"phase2/thresh{t}_split_assertions.json")
        bad = {k: v for k, v in rep["assertions"].items() if str(v) != "PASS"}
        assert not bad, f"thresh{t} has failing leakage assertions: {bad}"


def test_confounds_plant_disjoint_leakage_checks_and_demo():
    """HARD RULE 4 for the confound-baseline splits; the image-level demo must have been
    RUN (its exit codes recorded) so the report can show the leakage it demonstrates."""
    d = load("phase4/confounds/confound_baselines.json")
    codes = d["leakage_check_exit_codes"]
    pd_codes = {k: v for k, v in codes.items() if k.startswith("plant_disjoint")}
    il_codes = {k: v for k, v in codes.items() if k.startswith("image_level")}
    assert pd_codes, "no plant-disjoint leakage checks recorded"
    bad = {k: v for k, v in pd_codes.items() if v != 0}
    assert not bad, f"plant-disjoint split(s) failed check_leakage.py: {bad}"
    assert len(il_codes) == len(pd_codes), "image-level demo checks missing"
    assert all(isinstance(v, int) for v in il_codes.values())


def test_warning1_matches_the_articles_label_provenance():
    """The old warning claimed labels came from visible symptoms; the dataset's article
    says otherwise (inoculation + RT-PCR), so the report must carry the correction."""
    md = REPORT.read_text()
    assert "labelled from visible symptoms" not in md, \
        "warning 1 still carries the pre-correction wording"
    assert "mechanically inoculated" in md, "article-based label provenance missing"
    assert "RT-PCR" in md, "the RT-PCR verification quote is missing"
    assert "symptom-onset" in md, "the symptom-onset finding is missing"


def test_failed_claim_is_reported_as_unexplained_or_explained():
    """Group B1: the prevalence claim FAIL must carry a computed verdict."""
    rc = load("phase3/claim_rootcause/claim_rootcause.json")
    assert rc["verdict"], "no verdict recorded"
    assert rc["variants"], "no alternative definitions tested"
    assert rc["integrity"]["match_approved_exclusions"] is True
    md = REPORT.read_text()
    assert "Verdict" in md, "the claim root-cause verdict is not in the report"


if __name__ == "__main__":      # pragma: no cover
    pytest.main([__file__, "-q"])
