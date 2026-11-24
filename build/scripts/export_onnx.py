"""Phase 1.3 - export the frozen leaf model to ONNX (opset 17) + parity vs PyTorch.

  .venv/bin/python build/scripts/export_onnx.py

Input: build/checkpoint/mobilenet_v3_large_seed42.pt (the frozen model).
Output: build/models/leaf_mobilenet_v3_large_seed42.onnx + build/onnx_export.json.
Parity: max |logit diff| over the 50 golden vectors must be <= 1e-4 (build/config.json);
otherwise this script exits 2 and reports the miss (spec: fix or stop).
Restart-proof: skips if the report exists.
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

from models import build_model  # noqa: E402

CKPT = ROOT / "build" / "checkpoint" / "mobilenet_v3_large_seed42.pt"
GOLDEN = ROOT / "build" / "golden" / "manifest.json"
ONNX_OUT = ROOT / "build" / "models" / "leaf_mobilenet_v3_large_seed42.onnx"
REPORT = ROOT / "build" / "onnx_export.json"
CFG = json.loads((ROOT / "build" / "config.json").read_text())


def main() -> int:
    if REPORT.exists():
        print(f"[skip] {REPORT} exists")
        return 0
    t0 = time.time()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(*CFG["onnx"]["input_shape"], dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(ONNX_OUT),
        input_names=["input"], output_names=["logits"],
        opset_version=int(CFG["onnx"]["opset"]),
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    # parity on the golden vectors (same tensors for torch and ORT)
    man = json.loads(GOLDEN.read_text())
    from make_golden import preprocess_jpeg_bytes  # shared contract implementation
    x = np.concatenate(
        [preprocess_jpeg_bytes(ROOT / "build" / "golden" / v["file"])[0]
         for v in man["vectors"]], axis=0)
    expected = np.stack([v["logits"] for v in man["vectors"]], axis=0).astype(np.float32)

    with torch.no_grad():
        t_out = model(torch.from_numpy(x)).numpy()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    o_out = sess.run(["logits"], {"input": x})[0]

    torch_vs_golden = float(np.abs(t_out - expected).max())
    torch_vs_onnx = float(np.abs(t_out - o_out).max())
    limit = float(CFG["onnx"]["parity_max_abs_logit_diff"])
    sha = hashlib.sha256(ONNX_OUT.read_bytes()).hexdigest()
    ok = torch_vs_onnx <= limit
    report = {
        "onnx_file": str(ONNX_OUT.relative_to(ROOT)),
        "sha256": sha,
        "opset": int(CFG["onnx"]["opset"]),
        "input_shape": CFG["onnx"]["input_shape"],
        "n_golden": len(man["vectors"]),
        "parity_max_abs_logit_diff_torch_vs_onnx": torch_vs_onnx,
        "tolerance": limit,
        "pass": bool(ok),
        "golden_recompute_max_abs_diff": torch_vs_golden,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "runtime_s": round(time.time() - t0, 1),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
