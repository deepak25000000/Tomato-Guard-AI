# TomatoGuard AI — Research Prototype

**TomatoGuard AI: Mobile RGB Baseline, Multispectral Data Audit, and Evaluation Protocol for Tomato Disease Detection**

A B. Tech Computer Engineering project report (Vishwakarma University, Pune — November 2026).  
**Screening aid for research only — not a diagnosis.**

---

## 📱 Prototype App (Android APK)

The Flutter Android application is a fully offline, on-device screening aid that runs a
MobileNetV3-Large TFLite model on tomato leaf photos. It displays a model score (not a
calibrated probability) together with fixed disclaimers, and never uploads images anywhere.

### Download the APK

| Artifact | Link | SHA-256 |
|---|---|---|
| `TomatoGuardAI-release.apk` (prototype, ~83 MB) | [Releases](https://github.com/deepak25000000/Tomato-Guard-AI/releases) | `758f8c20ea1608cb65ab8dd24706947133cc8b3e6cc5a5b4d672a7413e1b829b` |

> **How to install:** download the APK from the Releases page, enable "Install from unknown
> sources" on your Android device, and install. No internet permission is requested in the
> release manifest — inference runs entirely on-device.

> ⚠️ **Real-phone benchmark is pending** — no physical device was available during the build.
> Latency, energy, and field performance have not been measured on real phones.

---

## 📄 Project Report

The full university project report is included in this repository:

- **`G41_Project_Report.pdf`** — complete report (chapters 1–9, references, appendices A/B),
  matching the Vishwakarma University format, with the Declaration and Certificate pages.

---

## 🗂️ Repository Structure

| Path | Contents |
|---|---|
| `build/` | Generated model artifacts: frozen checkpoint, ONNX/TFLite models, golden vectors, model card/meta, configs, build reports, APK |
| `tomatoguard/app/` | Flutter Android app (on-device TFLite), offline — **prototype** |
| `tomatoguard/research/` | Companion analysis scripts (splits, training, evaluation, export, CAEP protocol) |
| `tomatoguard/benchmarks/` | Benchmark templates (desktop + device results template) |
| `web/` | PWA (TypeScript + Vite + onnxruntime-web WASM), local-only history |
| `api/` | FastAPI read-only Track B research viewer (ONNX Runtime), no upload endpoints |
| `analysis/` | Analysis pipeline scripts (Phases 0–4) |
| `docs/` | PREPROCESSING.md, LIMITATIONS.md, PRIVACY.md |
| `tests/` | pytest suite (forbidden phrases, traceability, parity, API) |
| `results/` | Verified analysis outputs (read-only inputs to the build) — not committed |
| `G41_Project_Report.pdf` | University project report |

---

## 🔑 Key Rules (from spec)

- Every reported number is traceable to `build/model_card.json`, generated from `results/*`.
- Fixed UI strings live in `build/ui_strings.json`.
- Test split touched **once** for final evaluation; threshold selected on validation only.
- Raw data and existing `results/` files are **never modified**.
- The test set is used exactly once, after all selection decisions on validation data.
- No claim of ToBRFV detection, pre-symptomatic detection, or field readiness is made.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Analysis & training | Python 3.14.4, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu |
| Data & statistics | NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1 |
| Model export | ONNX opset 17, onnxruntime 1.32.0, TFLite (ai-edge-litert) |
| Android client | Flutter 3.50.1, Dart, TensorFlow Lite |
| Web client | TypeScript, Vite, ONNX Runtime Web (WASM) |
| Research viewer | FastAPI, Uvicorn, Python |
| Testing | pytest |

---

## 🧪 Test Results

31 tests pass, 0 skipped (prototype + results + API). See `build/BUILD_REPORT.md` for
done/pending/tolerance/deviation details.

### Known deviations (reported honestly)
- **Web browser parity FAIL** — top-1 agreement 0.84 (tolerance 0.98). Node parity PASS (1.0).
- **Track B ONNX** — 8/9 logit checks pass; rgb seed42 logit diff 1.24e-4 > 1e-4 (pred-vs-archived pass).
- **Real-phone benchmark** — PENDING (no physical device).

---

## 📊 Headline Results (frozen MobileNetV3-Large, S2 test, n = 428)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.6822 | [0.6379, 0.7266] |
| Macro F1 | 0.6333 | [0.5779, 0.6847] |
| Macro ROC-AUC | 0.9152 | [0.8969, 0.9325] |
| Coverage | 0.7944 (340/428) | [0.7547, 0.8318] |
| Selective accuracy | 0.7500 | [0.7029, 0.7942] |

Majority-class baseline (Leaf Miner Flies): 0.2967 accuracy.

---

## 👥 Team

Deepak Thorat · Adarsh Priyadarshi · Atharva Birajdar · Shivaraj Ghorpade · Yash Bandal  
Department of Computer Engineering, Vishwakarma University, Pune  
Supervisor: Dr. Harish Motekar  
November 2026

---

## 📄 License

This is an academic research prototype. See `THIRD_PARTY_LICENSES.md` for third-party
license notices.
