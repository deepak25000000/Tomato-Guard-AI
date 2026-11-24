# Limitations (TomatoGuard AI)

This file is a generated/human-readable companion to `build/model_card.json` and `results/RESULTS.md`. No number here is typed by hand except counts that are structural (e.g. 7 classes) which are also present in the model artifacts.

## Leaf model (Track A)
- **No ToBRFV class.** The leaf model was trained on 7 tomato leaf condition classes and has no ToBRFV class. It must not be presented as detecting ToBRFV.
- **Single public dataset, lab-style photos.** Trained on one public dataset of lab-style leaf photos (group-disjoint split S2). Accuracy on farm photos, phone photos, or other cultivars/regions is unknown.
- **Confusion.** Early Blight and Leaf Miner Flies are the dominant confusion pair on S2 test (see `build/model_card.json` known_confusions).
- **Uncalibrated model score.** Raw softmax is reported as a "model score". Temperature scaling was evaluated and made ECE worse on validation, so it is not applied. Do not read the score as a probability.
- **No field validation. No phone-camera validation.** No real-field or real-phone-camera validation has been performed in this build.
- **Distillation not run.** Knowledge distillation was not run.

## Track B research viewer (TOBRFV-LMID)
- **Labels are inoculation status (RT-PCR verified), not symptom onset.** The dataset has no symptom-onset dates; nothing here establishes performance on plants whose symptoms have not yet appeared.
- **Days 1-8 are single-class days** (confounded). Only mixed-label days 9-29 are interpretable for image-level discrimination; the viewer defaults to days 9-29 and warns on days 1-8.
- **Plant-level metrics saturated at 1.0** in the archived experiments; image-level + plant-cluster bootstrap is the reported level.
- **Research demo only.** Not for field use. Read-only replay of recorded greenhouse data.

## Prototype / deployment
- **Real-phone benchmark PENDING** unless recorded on a physical device via `benchmarks/device_results_template.csv`.
- **Flutter/Android toolchain** may be absent on the build machine; APK and on-device parity are PENDING in that case (see `build/BUILD_REPORT.md`).
- **No upload by default.** Web/PWA and mobile keep data local. API has no file-upload endpoints.
