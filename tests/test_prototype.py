"""Prototype tests (Phases 1-6). Phase 1 subset runs now; later phases append tests
for their artifacts (same file, guarded by artifact existence where appropriate).

Covers HARD RULE 1 (number traceability), HARD RULE 8 (forbidden phrases) and the
pre-registered threshold rule; golden-vector parity re-runs the real models.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
sys.path.insert(0, str(BUILD / "scripts"))

# spec-mandated fixed strings (must match build/ui_strings.json exactly)
FIXED = {
    "DISCLAIMER": "Screening aid for research. Not a diagnosis. Confirm with an agronomist or laboratory test.",
    "SCOPE": "Trained on one public dataset of lab-style leaf photos. Accuracy on farm photos is unknown.",
    "TOBRFV": "This leaf model does not detect ToBRFV.",
    "UNCERTAIN": "Not confident. Re-photograph one leaf, filling the frame, in good light.",
    "RESEARCH_BANNER": "Research demo on recorded greenhouse data. Not for field use.",
}
FORBIDDEN = ["diagnose", "diagnoses", "diagnosis", "early detection",
             "pre-symptomatic", "field-ready", "detects tobrfv", "detects toBRFV"]
NEGATIONS = ["not a diagnosis", "does not diagnose", "no diagnosis", "without diagnosis"]


def _load(rel: str):
    return json.loads((ROOT / rel).read_text())


def _iter_numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_numbers(v)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)


def test_ui_strings_are_the_fixed_spec_strings():
    ui = _load("build/ui_strings.json")
    for k, v in FIXED.items():
        assert ui.get(k) == v, f"ui_strings.json {k} does not match the spec string"


def test_no_forbidden_phrases_in_ui_strings_docs_and_artifacts():
    """HARD RULE 8: forbidden words only allowed in a negated disclaimer context."""
    targets = [BUILD / "ui_strings.json", BUILD / "MODEL_CARD.md",
               BUILD / "model_card.json", BUILD / "model_meta.json", BUILD / "PLAN.md"]
    targets += list((ROOT / "docs").glob("*.md")) if (ROOT / "docs").exists() else []
    for d in ("web", "api", "app"):
        p = ROOT / d
        if p.exists():
            targets += [f for f in p.rglob("*")
                        if f.is_file() and f.suffix in {".ts", ".tsx", ".html", ".json",
                                                        ".dart", ".md", ".py", ".yaml"}
                        and "node_modules" not in f.parts and ".dart_tool" not in f.parts
                        and "build" not in f.parts[len(p.parts):len(p.parts) + 1]]
    hits = []
    for f in targets:
        if not f.exists():
            continue
        text = f.read_text(errors="ignore").lower()
        for pat in FORBIDDEN:
            for m in re.finditer(re.escape(pat), text):
                ctx = text[max(0, m.start() - 60):m.end() + 60]
                if any(n in ctx for n in NEGATIONS):
                    continue
                hits.append(f"{f.relative_to(ROOT)}: '{pat}'")
    assert not hits, f"forbidden phrase(s) outside a negated disclaimer: {hits}"


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _iter_numbers_and_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and re.fullmatch(r"-?\d+", k):
                yield float(k)
            yield from _iter_numbers_and_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_numbers_and_keys(v)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for m in _NUM_RE.finditer(obj):
            try:
                yield float(m.group(0))
            except ValueError:
                pass


@pytest.mark.skipif(not (BUILD / "MODEL_CARD.md").exists(),
                    reason="Phase 1 artifacts not generated yet")
def test_every_number_in_model_card_is_traceable():
    """HARD RULE 1: every number in MODEL_CARD.md must come from model_card.json
    (which is itself generated from results/* via build/scripts/gen_model_artifacts.py)."""
    card = _load("build/model_card.json")
    allowed = set(_iter_numbers_and_keys(card))
    # derived renderings allowed: bootstrap CI 0.95 -> "95%"
    if "bootstrap" in card.get("test", {}):
        ci = card["test"]["bootstrap"].get("ci")
        if ci is not None:
            allowed.add(float(round(ci * 100)))
    md = (BUILD / "MODEL_CARD.md").read_text()
    # numeric tokens (ints/floats/sci notation); skip tokens embedded in identifiers (S2, leaf-v1)
    bad = []
    for m in re.finditer(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z])", md):
        t = m.group(1)
        try:
            v = float(t)
        except ValueError:
            continue
        if v in allowed:
            continue
        if any(f"{s:.4f}" == t or f"{s:.3e}" == t or f"{s:.0e}" == t or f"{s:.6e}" == t
               for s in allowed):
            continue
        if v.is_integer() and any(s.is_integer() and s == v for s in allowed):
            continue
        bad.append(t)
    assert not bad, f"numbers in MODEL_CARD.md not traceable to model_card.json: {bad}"


@pytest.mark.skipif(not (BUILD / "model_meta.json").exists(),
                    reason="Phase 1 artifacts not generated yet")
def test_model_meta_hashes_labels_and_typo_map():
    meta = _load("build/model_meta.json")
    for kind, rel in meta["model_files"].items():
        p = ROOT / rel
        assert p.exists(), f"model file missing: {rel}"
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == meta["sha256"][kind], f"sha256 mismatch for {kind}"
    # raw typo never a display label; fixed spelling present; map recorded
    assert "Bacteiral Spot" not in meta["display_labels"]
    assert "Bacterial Spot" in meta["display_labels"]
    assert meta["display_label_map"]["Bacteiral Spot"] == "Bacterial Spot"
    # classes keep the raw folder names (raw data untouched, HARD RULE 2)
    assert "Bacteiral Spot" in meta["classes"]


@pytest.mark.skipif(not (BUILD / "threshold.json").exists(),
                    reason="threshold not selected yet")
def test_threshold_rule_is_the_pre_registered_one():
    thr = _load("build/threshold.json")
    cfg = _load("build/config.json")
    need = cfg["uncertain_threshold_rule"]["min_selective_accuracy"]
    assert thr["rule_met"] is True
    assert thr["validation_selective_accuracy"] >= need - 1e-12
    # recompute the lowest qualifying threshold from the validation predictions
    import pandas as pd
    df = pd.read_csv(BUILD / "freeze" / "regen_val_preds.csv")
    p = df[[c for c in df.columns if c.startswith("prob_")]].values
    conf, pred, y = p.max(1), p.argmax(1), df["y_true"].values
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(pred[order] == y[order])
    ks = np.arange(1, len(y) + 1)
    sel = cum / ks
    ok = np.where(sel >= need)[0]
    assert len(ok) > 0
    # largest qualifying k -> lowest threshold with maximal coverage
    k = int(ok[-1]) + 1
    recomputed = float(np.sort(conf)[::-1][k - 1])
    covered = conf >= recomputed
    assert abs(float((pred[covered] == y[covered]).mean())
               - thr["validation_selective_accuracy"]) < 1e-9
    assert abs(recomputed - thr["threshold"]) < 1e-9


def _golden_inputs():
    man = _load("build/golden/manifest.json")
    from make_golden import preprocess_jpeg_bytes
    x = np.concatenate([preprocess_jpeg_bytes(BUILD / "golden" / v["file"])[0]
                        for v in man["vectors"]], axis=0).astype(np.float32)
    expected = np.stack([v["logits"] for v in man["vectors"]]).astype(np.float32)
    return man, x, expected


@pytest.mark.skipif(not (BUILD / "onnx_export.json").exists(),
                    reason="ONNX export not done yet")
def test_golden_parity_onnx():
    """Phase 1 parity: max |logit diff| PyTorch vs ONNX <= 1e-4 (re-run live)."""
    cfg = _load("build/config.json")
    man, x, expected = _golden_inputs()
    onnx_path = ROOT / _load("build/onnx_export.json")["onnx_file"]
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input": x})[0]
    diff = float(np.abs(got - expected).max())
    limit = cfg["onnx"]["parity_max_abs_logit_diff"]
    assert diff <= limit, f"ONNX parity {diff} > {limit}"
    # report file agrees with the live re-run (small float noise allowed)
    rep = _load("build/onnx_export.json")
    assert rep["pass"] is True
    assert abs(rep["parity_max_abs_logit_diff_torch_vs_onnx"] - diff) < 1e-5


@pytest.mark.skipif(not (BUILD / "tflite_export.json").exists(),
                    reason="TFLite export not done yet")
def test_golden_parity_tflite():
    """Phase 1 parity: max |logit diff| PyTorch vs TFLite <= 1e-3 (re-run live)."""
    cfg = _load("build/config.json")
    man, x, expected = _golden_inputs()
    tfl_path = BUILD / "models" / "leaf_mobilenet_v3_large_seed42.tflite"
    from ai_edge_litert.interpreter import Interpreter
    itp = Interpreter(model_path=str(tfl_path))
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    got = np.empty_like(expected)
    for i in range(len(x)):
        itp.set_tensor(inp["index"], x[i:i + 1])
        itp.invoke()
        got[i] = itp.get_tensor(out["index"])[0]
    diff = float(np.abs(got - expected).max())
    limit = cfg["tflite"]["parity_max_abs_logit_diff"]
    assert diff <= limit, f"TFLite parity {diff} > {limit}"
    rep = _load("build/tflite_export.json")
    assert rep["pass"] is True
