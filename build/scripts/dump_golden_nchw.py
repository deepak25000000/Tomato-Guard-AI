"""Dump golden inputs as preprocessed NCHW float32 arrays for Node parity tests.

  .venv/bin/python build/scripts/dump_golden_nchw.py

Reads build/golden/manifest.json, applies the documented preprocessing contract
(docs/PREPROCESSING.md) starting from the JPEG bytes, and writes
build/golden/golden_nchw.json. Deterministic; restart-proof via overwrite.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build" / "scripts"))
from make_golden import preprocess_jpeg_bytes  # noqa: E402
man = json.loads((ROOT / "build/golden/manifest.json").read_text())
vecs = []
for v in man["vectors"]:
    x, _ = preprocess_jpeg_bytes(ROOT / "build/golden" / v["file"])
    vecs.append({"file": v["file"], "input_nchw": x.reshape(-1).astype(np.float32).tolist(),
                 "logits": v["logits"]})
(ROOT / "build/golden/golden_nchw.json").write_text(json.dumps({"n": len(vecs), "vectors": vecs}))
print(f"wrote golden_nchw.json ({len(vecs)} vectors)")
