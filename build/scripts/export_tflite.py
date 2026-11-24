"""Phase 1.5a - convert the frozen leaf model to TFLite fp32 (converter venv).

  python3.11 export venv:  .venv_tflite/bin/python build/scripts/export_tflite.py

Runs ONLY inside the pinned Python 3.11 export venv (approved in build/PLAN.md §5.1):
main venv Python 3.14 has no TensorFlow/ai-edge-torch support.

Input: build/checkpoint/mobilenet_v3_large_seed42.pt + build/golden/manifest.json
Output: build/models/leaf_mobilenet_v3_large_seed42.tflite + build/tflite_convert.json
Numerical parity is checked separately in the MAIN venv by parity_tflite.py (litert).
Restart-proof: skips if the TFLite file exists.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))
sys.path.insert(0, str(ROOT / "build" / "scripts"))

from models import build_model  # noqa: E402

CKPT = ROOT / "build" / "checkpoint" / "mobilenet_v3_large_seed42.pt"
GOLDEN = ROOT / "build" / "golden" / "manifest.json"
OUT = ROOT / "build" / "models" / "leaf_mobilenet_v3_large_seed42.tflite"
REPORT = ROOT / "build" / "tflite_convert.json"


def main() -> int:
    if OUT.exists() and REPORT.exists():
        print(f"[skip] {OUT.name} exists")
        return 0
    t0 = time.time()
    try:
        import litert_torch as converter_mod  # preferred (ai-edge-torch renamed)
    except Exception:
        import ai_edge_torch as converter_mod  # fallback

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    sample = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    converted = converter_mod.convert(model, (sample,))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(converted, "export"):
        try:
            ret = converted.export(str(OUT))
            if isinstance(ret, (bytes, bytearray)):
                OUT.write_bytes(bytes(ret))
        except TypeError:
            ret = converted.export()
            if isinstance(ret, (bytes, bytearray)):
                OUT.write_bytes(bytes(ret))
    elif hasattr(converted, "tflite_model"):
        OUT.write_bytes(converted.tflite_model)
    elif isinstance(converted, (bytes, bytearray)):
        OUT.write_bytes(bytes(converted))
    else:
        OUT.write_bytes(bytes(converted))
    if not OUT.exists() or OUT.stat().st_size == 0:
        raise RuntimeError(f"TFLite export failed: {OUT} missing or empty")

    # sanity: venv-torch logits vs golden manifest logits (torch version drift check)
    from make_golden import preprocess_jpeg_bytes
    man = json.loads(GOLDEN.read_text())
    x = np.concatenate([preprocess_jpeg_bytes(ROOT / "build" / "golden" / v["file"])[0]
                        for v in man["vectors"]], axis=0)
    expected = np.stack([v["logits"] for v in man["vectors"]]).astype(np.float32)
    with torch.no_grad():
        venv_logits = model(torch.from_numpy(x)).numpy()
    report = {
        "tflite_file": str(OUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "dtype": "fp32",
        "converter": "ai-edge-torch",
        "converter_module": getattr(converter_mod, "__name__", "?"),
        "converter_version": getattr(converter_mod, "__version__", "?"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "venv_torch_vs_manifest_max_abs_logit_diff": float(np.abs(venv_logits - expected).max()),
        "n_golden": len(man["vectors"]),
        "runtime_s": round(time.time() - t0, 1),
        "note": "numerical parity vs PyTorch is decided in the main venv by parity_tflite.py",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
