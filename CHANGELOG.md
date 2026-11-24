# CHANGELOG

Log of every edit made to **existing** project scripts during this analysis, per HARD RULE 7.
New files created for this analysis are listed separately (they do not modify existing code).

Format: `file | reason | tests re-run`

---

## Edits to existing scripts

| File | Reason | Tests re-run |
|---|---|---|
| `tomatoguard/research/fusion/fusion.py` | Added a single `--image-size` CLI flag (default 224 = the original hard-coded value, so **default behaviour is unchanged**) and one `global SIZE` assignment. Needed because Phase 0 measured the CPU-only Phase 4 matrix at 13.57 h at 224 px and the approved reduced setting is 160 px. No logic, loss, schedule, seed or metric was altered. | `python -m py_compile` on the file; the file is exercised end-to-end by `analysis/dataset2_phase4_experiments.py`, whose stdout is kept in `results/phase4/`. |
| `build/scripts/eval_test_once.py` | **Bootstrap CI quantile bug (found 2026-11-23 while sourcing report numbers):** both CI sites used `np.quantile(stats, [(1-alpha)/2, 1-(1-alpha)/2])` = `[0.475, 0.525]` (a ~5% interval) instead of `[alpha/2, 1-alpha/2]` = `[0.025, 0.975]`, so every "95% CI" in `build/test_eval.json` was ~17x too narrow. Fixed both lines (with FIX comments). Point estimates were unaffected (verified exact against `results/phase2/arch3_mobilenet_v3_large_seed42_preds.csv`); CIs were recomputed from that saved predictions file by the new `build/scripts/fix_test_eval_cis.py` **without re-touching the test split** (HARD RULE 2), and `gen_model_artifacts.py` + `gen_app_assets.py` were re-run so `build/model_card.json`, `build/MODEL_CARD.md` and the app asset carry the corrected values with a `ci_correction` audit block. | `pytest tests/ -q` -> 26 passed; `pytest api/tests -q` -> 5 passed (31 total) after the fix. |

No other existing script has been modified. In particular `01_make_splits.py`, `02_train.py`,
`03_evaluate.py`, `caep/*.py` and `fusion.py`'s training/metric code are byte-identical to the
originals; everything they compute is invoked through their own CLIs.

### Changes to this analysis's own scripts during the run (HARD RULE 7 log)

| File | Reason | Tests re-run |
|---|---|---|
| `analysis/dataset2_phase3_audit.py` (stage A + E) | The dry run found **2 of 10,180** Track B capture tuples that cannot be paired across the 4 streams. The original code hard-aborted (HARD RULE 6). After the user chose "exclude both, document loudly", stage A now records the defect in `results/phase3/stream_pairing_defect.json` and aborts only for tuples **not** listed in `analysis/approved_data_exclusions.json`; stage E drops exactly those rows from the 4-stream manifest and asserts the dropped set equals the approved set. Raw files are never touched. | `python -m py_compile`; filename-level dry run of the manifest + `caep/split_by_plant.py` + `caep/check_leakage.py`; `pytest tests/` at Phase 5. |
| `analysis/dataset2_phase3_audit.py` (stage C) | `NameError: n_boot` at the plant-level bootstrap: `n_boot` was a parameter of stage B but stage C's Cohen's-d CI loop referenced it too. Added `n_boot: int = 2000` to `stage_c`'s signature (same default as stage B); no computation logic changed. Caught because the run is logged with its traceback in `results/phase3/stdout.txt`. | Phase 3 re-run; the retry completed stage C and `pytest tests/` at Phase 5. |
| `analysis/run_phase2_phase3.sh` (`run_step`) | First Phase 3 attempt failed in **0 s before any Python ran**: the shell redirected stdout to `results/phase3/stdout.txt` but the directory is created by the Python script, which never started (`rc=1`, no data touched). Added `mkdir -p "$(dirname "$lf")"` - the same line `run_phase4_phase5.sh` already had. No run logic changed. | Phase 3 re-run from `analysis/run_phase3_retry.sh`; exit codes appended to `results/run_log.md`. |
| `analysis/generate_results_md.py` | Renders the pairing defect as a visible blockquote + table in Section 4.1 and as warning #2 in Section 6 (numbers read from the defect json). | `pytest tests/` (incl. `test_stream_pairing_defect_documented_and_only_approved_exclusions`). |
| `tests/test_results.py` | Added the pairing-defect test above. | run at Phase 5 |
| `analysis/generate_results_md.py`, `analysis/dataset2_phase3_audit.py`, `analysis/dataset2_phase4_experiments.py`, `analysis/dataset1_phase1_audit.py`, `analysis/phase0_propose.py` (run-log rewriters) | **Run-log destruction bug**: each rewriter kept itself idempotent with `old = old[: old.index(marker)]`, which assumes its block is last in `results/run_log.md`. Phase 5 smoke tests ran while later phases kept appending, so every rewrite cut at the (no longer final) Phase 5 block and destroyed 7 entries (architecture sweep, S1 seeds 43/44, negative control, unseen eval, first Phase 3 failure, first Phase 4 chain stop); a substring-matching marker destroyed the Phase 3 retry-failure entry too. All five now do a **block-scoped** cut (remove only their own `## ...` block, keep everything after it) and collapse leftover blank lines. No run logic touched. | `analysis/repair_run_log.py` re-inserted the lost entries; `pytest tests/` at Phase 5; the repaired `results/run_log.md` is the auditable record. |
| `analysis/repair_run_log.py` (new) | One-time, idempotent repair of `results/run_log.md`: re-inserts the destroyed/never-written entries, with every number **read from** the surviving `results/phase2/*.json`, `chain.log`, `pipeline.log`, `phase3_chain.log`, `phase4_chain.log` (HARD RULE 1) and each entry citing its source file. The three faithful runs (original/run1/run2) were launched directly and had never been logged at all, so their entries are marked "added retroactively". | re-run twice (2nd run = no-op); output verified by header-order print. |
| `tests/test_results.py` (number walker) | First full Phase 5 run: 14/15 passed; the traceability test failed because two **by-design empty** evidence files (`results/phase1/unreadable.csv`, `results/splits_S2/corrupted.csv` - a single newline each, written by `pd.DataFrame([]).to_csv` to mean "0 unreadable / 0 corrupted") were recorded as read errors. The walker now treats whitespace-only CSVs as empty evidence (contributing no numbers) while still flagging any other unreadable file. | `pytest tests/ -q` re-run -> **15 passed**, rc=0 (entry appended to `results/run_log.md`). |
| `tests/test_results.py` (Group A/B additions) | Four new tests: leakage assertions of the t=4/t=6 split rebuilds must PASS (HARD RULE 4); the confound baselines' plant-disjoint splits must pass the project's `check_leakage.py` and the image-level demo's codes must be recorded; warning 1 must carry the article-based label provenance and must NOT contain the old wrong wording; the prevalence claim FAIL must carry a computed verdict in the report. | re-run with the rest of the suite at Group C. |

---

## New files created for this analysis

| File | Purpose |
|---|---|
| `analysis/phase0_setup.py` | Environment capture + throughput benchmark (Phase 0). |
| `analysis/phase0_propose.py` | Turns benchmark into Phase 2 / Phase 4 runtime scenarios. |
| `analysis/reference_claims.json` | INPUT claims (yours + the paper's) to be tested - never results. |
| `analysis/dataset1_phase1_audit.py` | Phase 1 - Track A (RGB leaf) data audit. |
| `analysis/dataset1_phase2_prepare_splits.py` | Phase 2a - build S1 (your CSVs) and S2 (`01_make_splits.py`), run leakage assertions. |
| `analysis/dataset1_phase2_faithful.py` | Phase 2b - faithful notebook reproduction + re-evaluation of your original checkpoint. |
| `analysis/trackA_lib.py` | Shared, audited training/evaluation helpers for the Phase 2 runs. |
| `analysis/dataset1_phase2_s2.py` | Phase 2c - S2 headline runs (3 seeds) + checkpoint-on-S2. |
| `analysis/dataset1_phase2_arch.py` | Phase 2c - architecture sweep, validation-only selection. |
| `analysis/dataset1_phase2_unseen_eval.py` | Phase 2 diagnostic - checkpoint on S2 images S1 never saw. |
| `analysis/dataset2_phase3_audit.py` | Phase 3 - Track B parsing, data quality, claim verification, plant-disjoint splits. |
| `analysis/dataset2_phase4_experiments.py` | Phase 4 - Track B fusion experiments (delegates to `fusion/fusion.py`, `caep/paired_stats.py`). |
| `analysis/generate_results_md.py` | Phase 5 - builds `RESULTS.md` purely from `results/*.json/csv`; aborts if a required input is missing. |
| `analysis/repair_run_log.py` | One-time repair of `results/run_log.md` after the truncation bug above (numbers re-read from chain logs/result jsons; idempotent). |
| `analysis/dataset1_phase2_leakage_stratified.py` (new, Group A1) | Accuracy of the published checkpoint on S1 test images with vs without a near-duplicate in train, t = 0/2/3/4/6, bootstrap CIs + permutation test. **Fix on first run**: assumed `results/phase1/test_nearest_train.csv` held only test rows; it actually holds 899 rows (450 test + 449 validation, `split` column). Now filters `split == "test"` before the 450-row assertion; Phase-1 audited counts still cross-checked per threshold. |
| `analysis/dataset1_phase2_threshold_sweep.py` (new, Group A3) | Rebuild S2 at near-dup threshold 4/6 (project's `01_make_splits.py`), leakage assertions, 3 seeds each. **Fix on first run**: merged `groups.csv` into the split frames although `01_make_splits.py` already writes `sha256`/`group` into every split CSV, so the merged columns became `group_x`/`group_y` and the `group` lookup raised `KeyError`. Now asserts on the split CSV columns directly (still cross-checked against `groups.csv` membership). |
|`analysis/dataset1_phase2_seed_comparison.py` (new, Group A2)| S1 seed variance beside S2 seed variance (reads existing jsons only; asserts equal epochs before comparing). (no edits yet) |
|`analysis/dataset1_phase2_arch_multiseed.py` (new, Group A4)| `mobilenet_v3_large` + `efficientnet_b0`, 3 seeds each on S2, validation-only selection across all three candidates. (no edits yet) |
|`analysis/dataset1_phase2_excl48.py` (new, Group A5a)| Re-train the S1 recipe excluding the 48 high-confidence cross-label images (3 seeds), exclusion applied to split CSVs under `results/` only (HARD RULE 2). (no edits yet) |
|`analysis/dataset1_phase2_spec_extras.py` (new, Group A5b)| Majority baseline, numpy-only metric recomputation, ECE before/after temperature scaling (T fit on validation only), EB vs LMF error gallery, excl-48 comparison, threshold-sweep merge + trend. (no edits yet) |
|`analysis/dataset2_phase3_claim_rootcause.py` (new, Group B1)| Root-cause of the prevalence-by-angle claim FAIL: numerator/denominator bookkeeping + 16 alternative definitions, verdict computed. **Fixes on first runs**: tuple-key parser returned `(…,day,angle,plant)` while the manifest column order is `(…,plant,day,angle)` - aborted by its own HARD RULE 6 check, then corrected. (re-run -> rc 0) |
|`analysis/dataset2_phase4_subgroup_analysis.py` (new, Group B2/B3)| Image-level mode comparison with plant-cluster bootstrap CIs + per day/angle/batch metrics. **Fix before first run**: row pairing across modes keyed on `plant_key` alone (not unique per image); now aligns on sorted rows and verifies identical plant+label sequences. (re-run rc 0 after fix) |
|`analysis/dataset2_phase4_confound_baselines.py` (new, Group B4)| Metadata-only and intensity-statistics logistic regressions, 5 seeds, plant-disjoint (project's `split_by_plant.py`, HARD RULE 4 asserted) vs image-level (deliberate leakage demo, `check_leakage.py` expected to detect it). **Fix before first run**: the written manifests lacked `angle`/`day` while `check_leakage.py` is called with `--report-cols angle,batch,day` - it would have `KeyError`ed (exit 1) even on a passing plant-disjoint split and aborted the chain; both manifests now carry the report columns (extra columns are harmless for `split_by_plant.py`). (compile check; runs in the tail chain) |
|`analysis/dataset_article_reference.json` (new input, Group B5)| Verbatim quotes from the dataset's data-description article (Data in Brief, doi:10.1016/j.dib.2026.112927, PMC13276747) on how labels were assigned, with section references + the symptom-onset finding. INPUT, not a result. (fetched 2026-11-21) |
|`analysis/generate_results_md.py` (extended)| Renders the Group A sections (3.8-3.12) and Group B sections (4.6, 4.7, 5.3-5.6) from their result files; warning 1 rewritten from the article's label provenance (the old wording - "plants were labelled from visible symptoms" - was factually wrong and is replaced); added `analysis/` input loading (`_ainput`) for the article quotes; new required inputs abort the build if missing; the A4 winner-test sentence is conditional on `arch_winner_test.json`'s `arch` matching the multi-seed selection (they come from different sweeps - never conflated); B6 reads its parameters from the runner-written `run_config.json` instead of assuming them. (section numbering of `## 6. Warnings` / `## 7. Artifact index` unchanged (tests depend on them)) |
|`analysis/append_group_run_log.py` (new, Group A/B)| Block-scoped, idempotent HARD RULE 5 entries for every Group A/B run in `results/run_log.md`; runtimes/counts are read from the result JSONs (HARD RULE 1); aborts before writing if any source file is missing (HARD RULE 6). **Fix before first run**: the A3 entry read `agg['sizes']` but the aggregate writes `split_sizes`. (runs at the end of the tail chain) |
|`analysis/run_groupA_chain.sh`, `analysis/run_groupB_tail.sh` (new)| Serialized bash chains (ONE trainer at a time - RAM is tight): A = t4 -> t6 -> arch multiseed -> excl48; tail = spec extras -> confound baselines -> B6 fusion run (+ machine-written `run_config.json`) -> run-log append. Abort on first non-zero exit; runtimes recorded in the log. (invoked via `bash` in background) |
|`analysis/dataset1_phase2_arch_multiseed.py` (resume fix) + `analysis/run_groupA_chain_resume.sh` (new)| The 2026-11-22 host restart killed the chain mid-run (efficientnet seed 43 at epoch 6/10; its output files were never written). The seed loop now skips any `arch3_{arch}_seed{seed}.json` that already exists (the json is only written after a run fully completes, so a killed run leaves no file and is safely re-run) and the resume chain appends to the same log without truncating it, running only the remaining steps. (compile check; resume chain in background) |
|`analysis/dataset1_phase2_excl48.py` (resume fix) + `analysis/run_groupB_tail.sh` (guards)| Same restart-proofing for the remaining steps: excl48 skips seeds whose `excl48_seed{seed}.json` exists; the tail chain skips spec-extras / confound-baselines / fusion-B6 when their final output json already exists. Every skip is logged as `=== SKIP ... ===`. A killed run now loses at most the single in-flight training run. (bash -n + py_compile checks; tail chain not yet started at edit time) |
|`analysis/dataset1_phase2_spec_extras.py` (fix)| First tail-chain run (rc=1 after 13 s): `trackA_lib.predict()` already returns numpy arrays, so `lg.numpy()` raised `AttributeError`. Now `lg.numpy() if hasattr(lg, "numpy") else np.asarray(lg)`. The preceding majority-baseline and numpy-check outputs were already written and are overwritten idempotently on re-run. | py_compile + full tail-chain re-run |
| `analysis/approved_data_exclusions.json` | **Input decision file**: raw-data defects the user approved for exclusion (unpairable Track B capture tuples). Scripts may exclude only what is listed here; anything else aborts (HARD RULE 6). |
| `analysis/run_phase2_remaining.sh` | Chained runner (S2 -> architecture sweep). |
| `analysis/run_phase2_phase3.sh` | Chained runner (S1 seeds -> control -> Phase 3). |
| `analysis/run_phase4_phase5.sh` | Chained runner (Phase 4 -> Phase 5 -> tests). |
| `results/` | All machine-generated numbers (json/csv). Nothing here is typed by hand. |
| `results/phase1/`, `results/phase2/`, `results/phase3/`, `results/phase4/` | Per-phase artifacts. |
| `tests/test_results.py` | Phase 5 - automated verification: completeness of `RESULTS.md`, traceability of every table number to a result file, leakage assertions, controls at chance, forbidden phrases absent. |
| `build/scripts/fix_test_eval_cis.py` | One-time (idempotent) correction of the bootstrap CI quantiles in `build/test_eval.json` after the `eval_test_once.py` bug above: recomputes only the four CIs from the saved predictions CSV (asserting every point estimate still matches exactly), records old/new values in a `ci_correction` audit block; no test image read, no re-inference (HARD RULE 2 preserved). |
| `CHANGELOG.md` | This file. |

## Prototype build edits (Phase 0-6, this session)

| File | What changed | Why / evidence |
| --- | --- | --- |
| `tomatoguard/app/pubspec.yaml` | Added `assets/ui_strings.json` and `assets/model_card.json` assets. | Spec requires fixed UI strings single source and About/model-card display. |
| `tomatoguard/app/lib/main.dart` | Replaced with onboarding+consent, RESEARCH_BANNER, 3-tab Scan/History/About layout. | Spec Phase 3 screen/state requirements. |
| `tomatoguard/app/lib/risk.dart` | Removed forbidden "pre-symptomatic..." disclaimer; uses fixed strings and "model score"/uncertain wording. | HARD RULE 8 forbidden phrases; spec fixed strings. |
| `tomatoguard/app/lib/classifier.dart` | Added `modelVersion` to ModelMeta; kept on-device TFLite, 20+100 benchmark. | Spec shows model version; no INTERNET. |
| `tomatoguard/app/lib/scan_page.dart` | Rewritten: required Plant ID+angle, framing guide, heuristic blur/brightness hints labelled "not validated", top-3 model score, UNCERTAIN, version, DISCLAIMER/SCOPE/TOBRFV. | Spec Phase 3 Scan/Result. |
| `tomatoguard/app/lib/history_page.dart` | Rewritten: confirmed-label field, delete, CSV export, delete-all-data. | Spec Phase 3 History. |
| `tomatoguard/app/lib/db.dart` | New schema with plant_id/angle/model_version/confirmed_label/score/uncertain; delete-all; CSV export. | Spec Phase 3 local state. |
| `tomatoguard/app/lib/strings.dart` (new file) | Loads fixed UI strings from assets/ui_strings.json. | Single-source fixed strings. |
| `tomatoguard/app/lib/about_page.dart` (new file) | Model card, limitations, dataset licences, benchmark button. | Spec Phase 3 About/Benchmark. |

Evidence: `tests/test_prototype.py` (forbidden phrases, traceability, parity) + `pytest api/tests` pass. App rebuild on device is PENDING (no Flutter/Android build executed in this environment yet beyond toolchain install).
