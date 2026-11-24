# build/run_log.md — TomatoGuard AI prototype build log

Commands, runtimes and versions for the prototype build (HARD RULE 6). Times are UTC. Every number here was printed by the command it is recorded under.

## Phase 0 — Preflight (2026-11-22)

### 0a. Environment inventory

- `python3 --version` -> `Python 3.14.4` (venv `.venv` uses the same)
- `.venv/bin/python -m pip list | grep ...` -> numpy 2.5.5, pillow 13.0.0, pytest 9.2.0, scikit-learn 1.9.1, scipy 1.18.1, torch 2.15.0+cpu, torchvision 0.29.2+cpu
- import check in `.venv` -> `onnx MISSING, onnxruntime MISSING, tensorflow MISSING, ai_edge_torch MISSING` (before install)
- `node --version; npm --version` -> `v22.23.4`, `12.2.2` (nvm path)
- `which flutter dart` -> not found; `flutter --version` -> not found
- `echo $ANDROID_HOME` -> empty; `which adb sdkmanager` -> not found; `adb devices` -> not found
- `which docker` -> not found
- `/home/deepak-thorat/.local/bin/python3.11 --version` -> `Python 3.11.18`; its pip -> `pip 26.2.2`

### 0a. Converter wheel availability (checked with `pip download --no-deps`)

| package | result |
| --- | --- |
| onnx 1.23.2 | wheel `cp312-abi3-manylinux` -> installs on Python 3.14 |
| onnxruntime 1.32.0 | wheel `cp314-cp314-manylinux` -> installs on Python 3.14 |
| ai-edge-torch 0.7.2 | wheel `py3-none-any` (metadata only; see dry-run below) |
| onnx2tf 2.6.9 | wheel `py3-none-any` (metadata only; see dry-run below) |
| tensorflow (any) | **no wheel for Python 3.14** |
| tensorflow 2.20.0 (py3.11) | wheel `cp311-cp311-manylinux` 620,570,129 bytes -> downloaded to `build/wheelhouse/` |
| ai-edge-litert 2.4.0 | wheel `cp314-cp314-manylinux` 21.0 MB -> installs on Python 3.14 |

- `pip install --dry-run ai-edge-torch` (in .venv, py3.14) rc=0, but `Would install` includes **torch-2.14.1** (would replace the installed 2.15.0+cpu), the full nvidia-CUDA 13.x wheel set, jax, transformers -> unsafe for the main venv.
- `onnx2tf 2.6.9` wheel METADATA `Requires-Dist` (read from the wheel): exact pins `numpy==2.2.6, onnx==1.20.1, onnxruntime==1.26.0, ...` and `tensorflow==2.21.0 ; extra == 'tensorflow'` -> conflicts with the main venv (numpy 2.5.5, onnx 1.23.2, ort 1.32.0); must live in a separate venv if used.
- First `pip download tensorflow` attempt failed: `OSError: [Errno 122] Disk quota exceeded` at 103.5/572.4 MB because **`/tmp` is a RAM-backed tmpfs** (3.6 GB total), not a disk quota.

### 0a. RAM / tmpfs recovery (before any heavy job)

- `du -sh /tmp/*` found orphaned pip temp dirs from the failed downloads: 565M + 565M + 430M + 430M + 3x114M + build-envs ≈ **2.3 GB of RAM held in tmpfs**.
- With no pip running: `rm -rf /tmp/pip-unpack-* /tmp/pip-install-* /tmp/pip-build-env-* /tmp/tmpwfh7itss /tmp/tmpxcllldu3`
- Effect: `/tmp` 2.7 GB used -> 145 MB used; swap 4.0 GB -> 3.0 GB used; `free -h` available 1.7 GB -> 1.9 GB, later 2.5 GB.
- `analysis/ram_watch.sh` restarted with nohup (one instance; log appends every 60 s to `analysis/ram_watch.log`). Sample lines: `08:58:48Z mem used=4609MB avail=2571MB swap used=3800MB`.

### 0a. ONNX stack install into main venv (pinned)

- `time .venv/bin/python -m pip install -q onnx==1.23.2 onnxruntime==1.32.0` -> real 0m6.148s, rc=0
- import check -> `onnx 1.23.2 | ort 1.32.0 | providers: ['AzureExecutionProvider', 'CPUExecutionProvider']`

### 0b. Architecture / checkpoint inventory (read-only)

- `results/phase2/arch3_selection.json`: selected `mobilenet_v3_large` (criterion: mean best-validation accuracy over 3 seeds; validation only); per-seed validation accuracy large: [0.7342657342657343, 0.7062937062937062, 0.717948717948718] -> best seed **42**.
- `find` for `*.pth|*.pt|*.onnx|*.tflite`: only leaf checkpoint on disk is `~/Downloads/Tomato_Research_Results/best_mobilenetv3.pth` (the small/S1 model) -> **no checkpoint exists for mobilenet_v3_large**; `analysis/dataset1_phase2_arch_multiseed.py` contains no `save` call (grep empty).
- `results/phase2/arch3_mobilenet_v3_large_seed42.json` -> `runtime_s = 2123.3`, `epochs = 10`, `best_epoch = 8`, `best_val_accuracy = 0.7342657342657343` => retrain estimate **2123 s ≈ 35 min** (measured value from the archived run, not an estimate typed by hand).
- Validation prediction CSVs exist for all three seeds: `results/phase2/arch3_mobilenet_v3_large_seed{42,43,44}_val_preds.csv` (430 lines each = 429 validation images + header).
- Track B checkpoints exist: `results/phase4/seed{42,43,44}/{rgb,spectral,fusion}/best.pt` and `results/phase4/budget_check_px224_e15/fusion/best.pt` (Phase 5 export needs no retraining).

### 0c. PLAN written

- `build/PLAN.md` created (conversion path, risks, open questions). Phase 0 ends with a summary + STOP per HARD RULE 7.

## 2026-11-22T19:10:50+05:30
- Patched ~/.pub-cache/hosted/pub.dev/tflite_flutter-0.11.0/android/build.gradle compileSdkVersion 31->34 to fix AAR metadata check (local build workaround).

## 2026-11-23T07:55:32Z — bootstrap CI quantile bug fix (test_eval.json)

- Bug found while sourcing report numbers: `build/scripts/eval_test_once.py` computed percentile CI endpoints as `np.quantile(stats, [(1-alpha)/2, 1-(1-alpha)/2])` = `[0.475, 0.525]` for alpha=0.05 — a ~5% interval, not the intended 95% interval `[alpha/2, 1-alpha/2]` = `[0.025, 0.975]`. Both `boot_ci()` (line 42) and the selective-accuracy block (line 100) were affected. Symptom: "95% CI" for accuracy was [0.6799, 0.6846] around a point estimate of 0.6822 (width 0.005), while the phase-2 stratified bootstrap independently gave [0.6402, 0.7243] (width 0.084).
- Verified scope: `grep -rn "1 - (1 - alpha)"` -> only these 2 lines in the repo. `analysis/*` and `build/scripts/mixed_day_metrics.py` use `np.percentile([2.5, 97.5])` / `[(1-0.95)/2, 1-(1-0.95)/2]` = `[0.025, 0.975]` (correct). `analysis/trackA_lib.stratified_bootstrap_ci` uses `np.percentile(accs, [2.5, 97.5])` (correct).
- Verified point estimates were unaffected: `.venv/bin/python` re-derived n=428, accuracy 0.6822429906542056, coverage 0.794392523364486, n_covered 340, n_uncertain 88, selective accuracy 0.75, macro AUC 0.9151507845922608 from `results/phase2/arch3_mobilenet_v3_large_seed42_preds.csv` — all exactly equal to `build/test_eval.json`.
- Fix: quantiles changed to `[alpha/2, 1-alpha/2]` in `build/scripts/eval_test_once.py` (both sites, with FIX comments).
- CI recomputation WITHOUT re-touching the test split (HARD RULE 2): new `build/scripts/fix_test_eval_cis.py` re-runs only the bootstrap from the saved predictions CSV (same 5000 draws, seeds 42/43/44/45), asserts every point estimate still matches, and rewrites `build/test_eval.json` with a `ci_correction` audit block holding the old values. Command: `.venv/bin/python build/scripts/fix_test_eval_cis.py` -> runtime 34.5 s.
- Corrected 95% CIs (S2 test, n=428): accuracy [0.6378504672897196, 0.7266355140186916]; macro AUC [0.8969296768894506, 0.9325109575587716]; coverage [0.7546728971962616, 0.8317757009345794]; selective accuracy [0.7029411764705882, 0.7941176470588235].
- Regenerated downstream (all derived, no hand-typed numbers): `.venv/bin/python build/scripts/gen_model_artifacts.py` -> `wrote build/model_meta.json, build/model_card.json, build/MODEL_CARD.md`; `.venv/bin/python build/scripts/gen_app_assets.py` -> app asset `tomatoguard/app/assets/model_card.json` now byte-identical to `build/model_card.json` (`cmp` rc=0).
- Tests after fix: `.venv/bin/python -m pytest tests/ -q` -> `26 passed in 27.97s`; `.venv/bin/python -m pytest api/tests -q` -> `5 passed` (31 total, matches BUILD_REPORT).
- Note: the release APK `build/TomatoGuardAI-release.apk` still embeds the pre-fix `model_card.json`; its About page never displays CI values (only `limitations`), so the APK is unchanged pending a rebuild decision. Report quotes post-fix CIs from `build/model_card.json`.
