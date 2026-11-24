"""Latency benchmark on a desktop/laptop CPU (ONNX Runtime). Reproduces the paper's latency table
with proper warm-up. For PHONE numbers use the in-app Benchmark screen (see docs/README).

  python bench_desktop.py --onnx ../research/runs/mobilenet_v3_small/model.onnx --threads 1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=300)
    a = ap.parse_args()
    so = ort.SessionOptions(); so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for _ in range(a.warmup):
        sess.run(None, {"input": x})
    t = []
    for _ in range(a.runs):
        s = time.perf_counter(); sess.run(None, {"input": x}); t.append((time.perf_counter() - s) * 1000)
    res = {"onnx": a.onnx, "threads": a.threads, "runs": a.runs,
           "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)),
           "size_mb": Path(a.onnx).stat().st_size / 2**20}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
