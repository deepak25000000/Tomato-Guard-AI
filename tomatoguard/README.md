# TomatoGuard AI - project

Companion code for the paper *"TomatoGuard AI: Mobile RGB Baseline, Multispectral Data Audit, and
Evaluation Protocol for Tomato Disease Detection"*. It turns the paper's stated limitations into work:

| Paper limitation | Where it is addressed |
|---|---|
| Image-level split, near-duplicates unchecked | `research/01_make_splits.py` |
| One baseline architecture only | `research/02_train.py` (3 architectures) + `05_compare.py` |
| No calibration / uncertainty | temperature scaling in `02_train.py`, ECE in `03_evaluate.py`, "uncertain" state in the app |
| No on-device benchmark / no quantised deployment | `04_export.py`, in-app Benchmark button, `benchmarks/` |
| Protocol described but not implemented | `research/caep/` (plant-disjoint split, leakage check, paired stats) |
| No fusion experiment | `research/fusion/fusion.py` (RGB / spectral / fusion / distilled student) |

## Layout
```
research/   01_make_splits.py  02_train.py  03_evaluate.py  04_export.py  05_compare.py
            caep/{build_manifest,split_by_plant,check_leakage,paired_stats}.py
            fusion/fusion.py
app/        Flutter (Android) offline app: lib/, assets/models/, pubspec.yaml
benchmarks/ bench_desktop.py, device_results_template.csv
scripts/    run_phase1.sh
docs/       MODEL_CARD.md
```

## What has and has not been tested
Tested in a sandbox with synthetic data: `01_make_splits.py` (planted flipped/blurred/exact duplicates were all
grouped correctly; leakage assertions pass), `caep/build_manifest.py`, `split_by_plant.py`, `check_leakage.py`
(deliberately leaked split is detected, exit code 1), `paired_stats.py`. All Python files compile.

**NOT executed** (needs your GPU/dataset/phone, or SDKs not available in my sandbox): `02_train.py`, `03_evaluate.py`,
`04_export.py`, `fusion.py`, the Flutter app. Expect to fix small issues on first run - paste any error back and it
can be fixed quickly. **Do not quote any number in the paper that you did not produce by running these.**

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock.txt      # record your real versions for the paper
```

## Phase 1 - model + app (do in this order)
```bash
# 1-3: splits, train 3 architectures, evaluate, compare   (GPU: ~30-60 min; CPU: several hours)
bash scripts/run_phase1.sh "/home/deepak-thorat/Downloads/Tomato Leaf Disease Classification Dataset/Tomato Dataset"

# 4: export the best run into the app
cd research && python 04_export.py --run runs/<best_run> --app-dir ../app
```
**Read the parity lines** printed by `04_export.py` (ONNX / TFLite max difference should be ~1e-4 or smaller).
If `ai-edge-torch` is missing it tells you the `onnx2tf` fallback. Optional int8: convert with
`onnx2tf -i runs/<run>/model.onnx -o tf_out -oiqt` and compare accuracy FP32 vs int8 on the test split.

### Build the app
```bash
flutter create --platforms=android --project-name tomatoguard tomatoguard_app
cp -r app/lib app/assets app/pubspec.yaml tomatoguard_app/      # overwrite generated files
cd tomatoguard_app && flutter pub get
flutter run --release            # phone connected with USB debugging
```
If Gradle complains about `minSdkVersion`, set it to 26 in `android/app/build.gradle`.
The app is fully offline. Pre-processing (resize to 224x224, ImageNet normalisation) mirrors `eval_tf()` - slight
resize-filter differences vs PIL are normal; check a few images on the phone against Python's prediction.

### On-device benchmark (closes the "no phone benchmark" limitation)
On 2-3 phones (low / mid / high end), release build: tap the speed icon -> note median & p95 -> fill
`benchmarks/device_results_template.csv`. Measure battery (%) before/after ~500 scans, and peak RAM with
Android Studio Profiler. Report phone model, Android version, threads, precision.

## Multi-dataset training (field generalisation) - see docs/DATASETS.md for links
```bash
cd research
cp datasets/datasets_local.example.json datasets/datasets_local.json     # edit paths, delete datasets you lack
python 07_harmonize.py --config datasets/datasets_local.json --out unified   # unified labels + joint duplicate grouping
python 08_train_unified.py --mode lodo  --unified unified/all.csv --source-balance   # leave-one-dataset-out (the honest number)
python 08_train_unified.py --mode final --unified unified/all.csv --source-balance --out runs_unified/final
python 04_export.py --run runs_unified/final --app-dir ../app        # app now also says "Not a tomato leaf"
```
`07_harmonize.py` was tested on synthetic multi-dataset folders (mapping, skips, cross-dataset exact + flipped duplicates).
`08_train_unified.py` is NOT executed here. Report the LODO table (per held-out dataset) as your domain-shift evidence;
review `unified/skipped.csv` before trusting any class counts.

## Phase 2 - the evaluation protocol (CAEP) on TOBRFV-LMID
```bash
cd research/caep
python build_manifest.py --root <TOBRFV-LMID tomato folder> --regex '<see docstring>' --out manifest.csv
python split_by_plant.py --manifest manifest.csv --seeds 42,43,44,45,46 --out-dir plant_splits
python check_leakage.py  --manifest plant_splits/manifest_split_seed42.csv --report-cols angle,batch,day
```
`build_manifest.py` needs a regex matching YOUR copy's file names (I could not see them) - the docstring has
an example, and the script prints sample paths to help you write it.

### Re-run the paper's TOBRFV-LMID audit (Section V-A) on your copy
```bash
cd research/caep
python build_manifest.py --root <TOBRFV-LMID tomato folder> --regex '<yours>' \
    --stream-map rgb=RGB,vnir=VNIR,nir800=800,nir1000=1000 --require rgb,nir800,nir1000 --out manifest.csv
python audit_tobrfv.py --manifest manifest.csv --streams rgb,vnir,nir800,nir1000 --out audit_out
```
It prints a checklist of the paper's published numbers (4,808 images/stream, 167 plants, 68.0/52.1/52.2 % prevalence,
plant-level d = +0.543 / +0.647 / -0.014 / -0.897 ...). If your numbers differ, find out whether the parsing or the
paper is wrong *before* the paper is finalised. Tested only on synthetic data with a known answer.
`vnir` tokens can clash with `800`/`1000` in filenames - check `build_manifest` output counts per stream.

### Cross-dataset check (domain shift) - `research/06_external_eval.py`
Evaluate the trained RGB model on any other folder-per-class tomato dataset or on your own phone photos, using a
class-map JSON. Report the accuracy gap versus in-domain as your domain-shift evidence.

## Phase 3 - fusion (laptop/server, not the phone)
```bash
cd research/fusion
M=../plant_splits/manifest_split_seed42.csv
python fusion.py --mode rgb      --manifest $M --out runs/rgb
python fusion.py --mode spectral --manifest $M --out runs/spectral
python fusion.py --mode fusion   --manifest $M --out runs/fusion
python fusion.py --mode rgb --teacher runs/fusion/best.pt --manifest $M --out runs/rgb_distilled
python ../caep/paired_stats.py --a runs/rgb/preds_test.csv --b runs/fusion/preds_test.csv
```
Repeat for several split seeds. If the confidence interval of (fusion - RGB) includes 0, report "no demonstrated
advantage" - that is a valid, honest result. Phones cannot capture 800/1000 nm, so the phone only benefits via the
distilled RGB student. Natural extensions: cross-attention fusion, size-matched baselines, quantised students.

## Claims you can and cannot make
- Can: RGB baseline accuracy on the near-duplicate-group-disjoint split; calibration; model size; latency on the phones you measured.
- Cannot: "plant-disjoint" for the 7-class RGB dataset (no plant IDs); pre-symptomatic detection; ToBRFV detection
  from the RGB model (it has no ToBRFV class); field performance; agronomic validity of the risk labels.
- Anything about fusion only after Phase 3 has produced plant-disjoint results.
