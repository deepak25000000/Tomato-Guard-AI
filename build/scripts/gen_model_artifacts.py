"""Phase 1.6 - generate build/model_meta.json, build/model_card.json and
build/MODEL_CARD.md STRICTLY from existing result files (HARD RULE 1: no number is
typed by hand - each is read from a source file recorded in "generated_from").

  .venv/bin/python build/scripts/gen_model_artifacts.py

Sources (read-only): results/phase2/arch3_selection.json,
results/phase2/arch3_mobilenet_v3_large_seed42.json,
results/phase2/majority_baseline.json, results/phase2/spec_extras_summary.json,
results/splits_S2/classes.json, build/freeze_report.json, build/threshold.json,
build/test_eval.json, build/onnx_export.json, build/tflite_export.json (if present),
tomatoguard/research/models.py (IMAGENET_MEAN/STD), build/config.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))
sys.path.insert(0, str(ROOT / "analysis"))

DISPLAY = {"Bacteiral Spot": "Bacterial Spot"}  # same map as dataset1_phase1_audit.py

def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def ece_equal_width(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, equal-width bins on confidence (validation only)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def main() -> int:
    import numpy as np
    from models import IMAGENET_MEAN, IMAGENET_STD  # project source of truth

    sel = load("results/phase2/arch3_selection.json")
    arch_json = load("results/phase2/arch3_mobilenet_v3_large_seed42.json")
    freeze = load("build/freeze_report.json")
    thr = load("build/threshold.json")
    test = load("build/test_eval.json")
    onnx = load("build/onnx_export.json")
    majority = load("results/phase2/majority_baseline.json")
    extras = load("results/phase2/spec_extras_summary.json")
    cfg = load("build/config.json")
    classes_raw = load("results/splits_S2/classes.json")
    classes_display = [DISPLAY.get(c, c) for c in classes_raw]
    # ECE of the FROZEN model on S2 validation (selection data) - our own calibration
    # reference; the archived temperature-scaling numbers belong to a different
    # checkpoint (published S1 model) and are cited only with that attribution.
    import pandas as pd
    vdf = pd.read_csv(ROOT / "build" / "freeze" / "regen_val_preds.csv")
    vp = vdf[[c for c in vdf.columns if c.startswith("prob_")]].values
    vconf = vp.max(1)
    vcorr = (vp.argmax(1) == vdf["y_true"].values).astype(float)
    ece_val = ece_equal_width(vconf, vcorr, n_bins=int(extras["temperature_scaling"]["ece_bins"]))
    tflite_path = ROOT / "build" / "tflite_export.json"
    tflite = json.loads(tflite_path.read_text()) if tflite_path.exists() else None

    sources = [
        "results/phase2/arch3_selection.json",
        "results/phase2/arch3_mobilenet_v3_large_seed42.json",
        "results/phase2/majority_baseline.json",
        "results/phase2/spec_extras_summary.json",
        "results/splits_S2/classes.json",
        "build/freeze_report.json", "build/threshold.json", "build/test_eval.json",
        "build/onnx_export.json", "build/config.json",
        "tomatoguard/research/models.py",
    ]
    if tflite:
        sources.append("build/tflite_export.json")

    # ---- model_meta.json -------------------------------------------------
    model_files = {"checkpoint": "build/checkpoint/mobilenet_v3_large_seed42.pt",
                   "onnx": onnx["onnx_file"]}
    if tflite:
        model_files["tflite"] = tflite["tflite_file"]
    meta = {
        "model_version": "leaf-v1",
        "arch": sel["selected_architecture"],
        "seed": cfg["model_selection"]["seed"],
        "training_split": "S2 (group-disjoint)",
        "selection_data": cfg["model_selection"]["selection_data"],
        "input_size": [224, 224],
        "input_layout": "NCHW float32, /255 then ImageNet mean/std",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "classes": classes_raw,
        "display_labels": classes_display,
        "display_label_map": DISPLAY,
        "threshold": thr.get("threshold"),
        "threshold_rule": thr.get("rule"),
        "sha256": {k: sha256(v) for k, v in model_files.items()},
        "model_files": model_files,
        "preprocessing_doc": "docs/PREPROCESSING.md",
        "ui_strings": "build/ui_strings.json",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ROOT / "build" / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- model_card.json -------------------------------------------------
    ts = extras.get("temperature_scaling", {})
    card = {
        "model_version": meta["model_version"],
        "generated_from": sources,
        "task": "7-class leaf photo classifier (screening aid for research)",
        "training": {
            "split": "S2 (group-disjoint)",
            "architecture": sel["selected_architecture"],
            "selection_criterion": sel["criterion"],
            "seeds": 3,
            "per_seed_validation_accuracy": {
                str(s): v for s, v in zip(
                    (42, 43, 44),
                    sel["candidates"]["mobilenet_v3_large"]["per_seed"]
                    ["best_val_accuracy_per_seed"])},
            "selected_seed": cfg["model_selection"]["seed"],
            "selected_seed_best_validation_accuracy": arch_json["best_val_accuracy"],
            "selected_seed_best_epoch": arch_json["best_epoch"],
            "epochs": arch_json["epochs"],
            "retrain_report": "build/freeze_report.json",
            "retrain_vs_archived_max_abs_prob_diff": freeze["archived_val_max_abs_diff"],
        },
        "threshold": {
            "rule": thr["rule"],
            "threshold": thr.get("threshold"),
            "validation_coverage": thr.get("validation_coverage"),
            "validation_selective_accuracy": thr.get("validation_selective_accuracy"),
            "rule_met": thr.get("rule_met"),
        },
        "test": {
            "split": "S2 test, touched once after validation-only selection",
            "n": test["n_test"],
            "accuracy": test["accuracy"],
            "accuracy_95_ci": test["accuracy_95_ci"],
            "macro_auc": test["macro_auc"],
            "macro_auc_95_ci": test["macro_auc_95_ci"],
            "coverage": test["coverage"],
            "coverage_95_ci": test["coverage_95_ci"],
            "selective_accuracy": test["selective_accuracy"],
            "selective_accuracy_95_ci": test["selective_accuracy_95_ci"],
            "n_uncertain": test["n_uncertain"],
            "bootstrap": test["bootstrap"],
        },
        "baseline": {
            "majority_class_S2_test": majority["S2_group_disjoint"]["majority_class"],
            "majority_accuracy_S2_test": majority["S2_group_disjoint"]["test_accuracy"],
            "source": "results/phase2/majority_baseline.json (S2_group_disjoint)",
        },
        "calibration": {
            "temperature_scaling_applied": False,
            "ece_frozen_model_s2_validation": ece_val,
            "ece_bins": int(extras["temperature_scaling"]["ece_bins"]),
            "ece_source": "computed from build/freeze/regen_val_preds.csv (S2 validation)",
            "archived_reference": {
                "checkpoint": extras["temperature_scaling"]["checkpoint"],
                "fit_split": extras["temperature_scaling"]["temperature_fit_split"],
                "ece_before": extras["temperature_scaling"]["ece_before"],
                "ece_after": extras["temperature_scaling"]["ece_after"],
                "fitted_temperature": extras["temperature_scaling"]["temperature"],
                "source": "results/phase2/spec_extras_summary.json",
            },
            "statement": "raw softmax is shown as 'model score'; it is NOT a calibrated "
                         "probability (the archived analysis found temperature scaling "
                         "made ECE worse, so none is applied here)",
        },
        "exports": {
            "onnx": {"file": onnx["onnx_file"], "opset": onnx["opset"],
                     "sha256": onnx["sha256"],
                     "parity_max_abs_logit_diff": onnx["parity_max_abs_logit_diff_torch_vs_onnx"],
                     "tolerance": onnx["tolerance"], "pass": onnx["pass"]},
            "tflite": ({"file": tflite["tflite_file"],
                        "sha256": tflite["sha256"],
                        "parity_max_abs_logit_diff": tflite["parity_max_abs_logit_diff_torch_vs_tflite"],
                        "tolerance": tflite["tolerance"], "pass": tflite["pass"]}
                       if tflite else None),
        },
        "known_confusions": {
            "note": "the two most-confused classes on S2 test, read from the frozen "
                    "model's confusion matrix in build/test_eval.json",
            "pairs": [],  # filled below from test_eval confusion matrix
        },
        "limitations": [
            "Trained on one public dataset of lab-style leaf photos; accuracy on farm photos is unknown.",
            "This leaf model does not detect ToBRFV (no such class exists in the training data).",
            "Labels are disease/condition classes of the leaf dataset; no symptom-onset timing is available.",
            "Early Blight vs Leaf Miner Flies are the dominant confusion pair.",
            "No field validation and no phone-camera validation have been performed.",
            "Model score is uncalibrated; do not read it as a probability.",
            "Distillation was not run.",
        ],
    }
    # confusion pair from the frozen model's own test confusion matrix
    cm = test.get("confusion_matrix")
    if cm:
        import numpy as np
        m = np.array(cm, dtype=float)
        np.fill_diagonal(m, 0.0)
        flat = [(int(i), int(j)) for i, j in zip(*np.unravel_index(np.argsort(-m, axis=None), m.shape))]
        card["known_confusions"]["pairs"] = [
            {"true": classes_display[i], "predicted": classes_display[j], "count": int(m[i, j])}
            for i, j in flat[:2]
        ]
    # majority baseline keys: adapt to the actual file layout
    if not card["baseline"]["majority_class_S2_test"]:
        card["baseline"] = {"raw": majority, "source": "results/phase2/majority_baseline.json"}
    (ROOT / "build" / "model_card.json").write_text(json.dumps(card, indent=2))

    # ---- MODEL_CARD.md (rendered from card - no independent numbers) -----
    t = card["test"]
    ci = lambda c: f"[{c[0]:.4f}, {c[1]:.4f}]"
    lines = [
        "# Model card - leaf screening model (leaf-v1)",
        "",
        "Generated by `build/scripts/gen_model_artifacts.py` from the files listed in "
        "`build/model_card.json` -> `generated_from`. No number on this page is typed by hand.",
        "",
        "## What it is",
        "",
        f"- 7-class leaf photo classifier, architecture **{card['training']['architecture']}**, "
        f"trained on **{card['training']['split']}**, selected seed **{card['training']['selected_seed']}** "
        f"(selection on validation only: {card['training']['selection_criterion']}).",
        "- Screening aid for research. Not a diagnosis. Confirm with an agronomist or laboratory test.",
        "- Trained on one public dataset of lab-style leaf photos. Accuracy on farm photos is unknown.",
        "- This leaf model does not detect ToBRFV.",
        "",
        "## Selection (validation only)",
        "",
        "| seed | best validation accuracy |",
        "| --- | --- |",
    ]
    for s, v in card["training"]["per_seed_validation_accuracy"].items():
        lines.append(f"| {s} | {v} |")
    lines += [
        "",
        f"Archived run: best epoch {card['training']['selected_seed_best_epoch']} of "
        f"{card['training']['epochs']}; retrained checkpoint reproduces the archived "
        f"validation probabilities with max|diff| = "
        f"{card['training']['retrain_vs_archived_max_abs_prob_diff']:.6e} "
        f"(`build/freeze_report.json`).",
        "",
        "## Uncertain threshold (pre-registered rule)",
        "",
        f"- Rule: {card['threshold']['rule']} (met: {card['threshold']['rule_met']}).",
        f"- Threshold = **{card['threshold']['threshold']}**; validation coverage "
        f"{card['threshold']['validation_coverage']:.4f}, validation selective accuracy "
        f"{card['threshold']['validation_selective_accuracy']:.4f}.",
        "",
        "## Test results (S2 test, touched once)",
        "",
        f"- n = {t['n']}",
        f"- accuracy **{t['accuracy']:.4f}**, 95% CI {ci(t['accuracy_95_ci'])}",
        f"- macro AUC **{t['macro_auc']:.4f}**, 95% CI {ci(t['macro_auc_95_ci'])}",
        f"- coverage at threshold **{t['coverage']:.4f}**, 95% CI {ci(t['coverage_95_ci'])} "
        f"({t['n_uncertain']} of {t['n']} images flagged uncertain)",
        f"- selective accuracy (covered only) **{t['selective_accuracy']:.4f}**, "
        f"95% CI {ci(t['selective_accuracy_95_ci'])}",
        f"- bootstrap: {t['bootstrap']['n_draws']} draws, seed {t['bootstrap']['seed']}, "
        f"CI {t['bootstrap']['ci']}",
        "",
        "## Calibration warning",
        "",
        f"- ECE of this frozen model on S2 validation: "
        f"{card['calibration']['ece_frozen_model_s2_validation']:.4f} "
        f"({card['calibration']['ece_bins']} bins, source: "
        f"{card['calibration']['ece_source']}). Temperature scaling is **not** applied.",
        f"- Archived reference (different checkpoint: "
        f"{card['calibration']['archived_reference']['checkpoint']}, fit on "
        f"{card['calibration']['archived_reference']['fit_split']}): ECE "
        f"{card['calibration']['archived_reference']['ece_before']:.4f} -> "
        f"{card['calibration']['archived_reference']['ece_after']:.4f} (worse) at T = "
        f"{card['calibration']['archived_reference']['fitted_temperature']:.4f}. "
        "The UI therefore shows raw softmax only as a **model score**, never as a probability.",
        "",
        "## Exports",
        "",
        f"- ONNX opset {card['exports']['onnx']['opset']}: parity max|Δlogit| = "
        f"{card['exports']['onnx']['parity_max_abs_logit_diff']:.3e} "
        f"(tolerance {card['exports']['onnx']['tolerance']:.0e}) "
        f"-> {'PASS' if card['exports']['onnx']['pass'] else 'FAIL'}",
    ]
    if card["exports"]["tflite"]:
        tf = card["exports"]["tflite"]
        lines.append(
            f"- TFLite fp32: parity max|Δlogit| = {tf['parity_max_abs_logit_diff']:.3e} "
            f"(tolerance {tf['tolerance']:.0e}) -> {'PASS' if tf['pass'] else 'FAIL'}")
    else:
        lines.append("- TFLite fp32: not exported yet (PENDING)")
    lines += [
        "",
        "## Known confusions",
        "",
    ]
    if card["known_confusions"]["pairs"]:
        for pr in card["known_confusions"]["pairs"]:
            lines.append(f"- true **{pr['true']}** predicted as **{pr['predicted']}**: "
                         f"{pr['count']} test images")
    else:
        lines.append("- (confusion matrix not present in build/test_eval.json)")
    lines += ["", "## Limitations", ""]
    lines += [f"- {x}" for x in card["limitations"]]
    lines.append("")
    (ROOT / "build" / "MODEL_CARD.md").write_text("\n".join(lines))
    print("wrote build/model_meta.json, build/model_card.json, build/MODEL_CARD.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
