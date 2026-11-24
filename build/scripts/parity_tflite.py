"""Phase 1.5b - TFLite parity vs PyTorch (MAIN venv, ai-edge-litert interpreter).

  .venv/bin/python build/scripts/parity_tflite.py

Reads build/models/leaf_mobilenet_v3_large_seed42.tflite (produced by export_tflite.py
in the export venv) and the golden manifest; runs each golden input through the TFLite
interpreter; max |logit diff| vs the manifest's PyTorch logits must be <= 1e-3
(build/config.json). Writes build/tflite_export.json (consumed by gen_model_artifacts).
Restart-proof: skips if the report exists.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "scripts"))

TFLITE = ROOT / "build" / "models" / "leaf_mobilenet_v3_large_seed42.tflite"
GOLDEN = ROOT / "build" / "golden" / "manifest.json"
REPORT = ROOT / "build" / "tflite_export.json"
CFG = json.loads((ROOT / "build" / "config.json").read_text())


def main() -> int:
    if REPORT.exists():
        print(f"[skip] {REPORT} exists")
        return 0
    if not TFLITE.exists():
        print(f"FATAL: {TFLITE} missing - run export_tflite.py in the export venv first",
              file=sys.stderr)
        return 3
    t0 = time.time()
    from ai_edge_litert.interpreter import Interpreter

    man = json.loads(GOLDEN.read_text())
    from make_golden import preprocess_jpeg_bytes
    x = np.concatenate([preprocess_jpeg_bytes(ROOT / "build" / "golden" / v["file"])[0]
                        for v in man["vectors"]], axis=0).astype(np.float32)
    expected = np.stack([v["logits"] for v in man["vectors"]]).astype(np.float32)

    itp = Interpreter(model_path=str(TFLITE))
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    got = np.empty_like(expected)
    for i in range(len(x)):
        itp.set_tensor(inp["index"], x[i:i + 1])
        itp.invoke()
        got[i] = itp.get_tensor(out["index"])[0]

    limit = float(CFG["tflite"]["parity_max_abs_logit_diff"])
    max_diff = float(np.abs(got - expected).max())
    top1_agree = float((got.argmax(1) == expected.argmax(1)).mean())
    ok = max_diff <= limit
    report = {
        "tflite_file": str(TFLITE.relative_to(ROOT)),
        "sha256": __import__("hashlib").sha256(TFLITE.read_bytes()).hexdigest(),
        "dtype": "fp32",
        "n_golden": len(man["vectors"]),
        "parity_max_abs_logit_diff_torch_vs_tflite": max_diff,
        "tolerance": limit,
        "top1_agreement": top1_agree,
        "pass": bool(ok),
        "runtime": "ai-edge-litert interpreter (main venv)",
        "runtime_s": round(time.time() - t0, 1),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
