"""Phase 5 - export Track B Net(rgb|spectral|fusion) to ONNX (opset 17) + parity.

  .venv/bin/python build/scripts/export_trackb_onnx.py

For each mode × seed in {42,43,44} reads the existing checkpoint at
results/phase4/seed{seed}/{mode}/best.pt (read-only). Builds Net(mode) from
tomatoguard/research/fusion/fusion.py, loads state_dict, exports ONNX with two
inputs (rgb 1x3x224x224, spec 1x3x224x224). Parity: max |logit diff| vs PyTorch on
all test rows for that seed must be <= 1e-4. Also verifies the softmax[:,1]
(p_infected) matches the archived preds_test.csv (<= 1e-3) - the "live ONNX must
match precomputed predictions" requirement.

Output: build/models/trackb_{mode}_seed{seed}.onnx + build/trackb_export.json.
Restart-proof per (mode,seed). No training, no raw data writes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tomatoguard" / "research" / "fusion"))

import fusion as fusion_mod  # noqa: E402
from fusion import Net, Spec  # noqa: E402

SEEDS = [42, 43, 44]
MODES = ["rgb", "spectral", "fusion"]
REPORT = ROOT / "build" / "trackb_export.json"
LIMIT_LOGIT = 1e-4
LIMIT_PRED = 1e-3


def image_size_for(seed: int, mode: str) -> int:
    mpath = ROOT / f"results/phase4/seed{seed}/{mode}/metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        if "image_size" in m:
            return int(m["image_size"])
    return 160  # Phase 4 minimal budget default


def main() -> int:
    if REPORT.exists():
        print(f"[skip] {REPORT} exists")
        return 0
    t0 = time.time()
    (ROOT / "build" / "models").mkdir(parents=True, exist_ok=True)
    results = []
    for seed in SEEDS:
        manifest = pd.read_csv(ROOT / f"results/phase3/plant_splits/manifest_split_seed{seed}.csv")
        test = manifest[manifest["split"] == "test"].reset_index(drop=True)
        for mode in MODES:
            size = image_size_for(seed, mode)
            fusion_mod.SIZE = size
            ds = Spec(test, train=False)
            rgb = torch.stack([ds[i][0] for i in range(len(ds))])
            spec_t = torch.stack([ds[i][1] for i in range(len(ds))])
            ck = ROOT / f"results/phase4/seed{seed}/{mode}/best.pt"
            out_path = ROOT / "build" / "models" / f"trackb_{mode}_seed{seed}.onnx"
            model = Net(mode)
            model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                torch_logits = model(rgb, spec_t).numpy()
                torch_p = torch.softmax(torch.from_numpy(torch_logits), 1)[:, 1].numpy()

            import onnxruntime as ort

            def _export_run(do_fold: bool):
                torch.onnx.export(
                    model, (rgb[:1], spec_t[:1]), str(out_path),
                    input_names=["rgb", "spec"], output_names=["logits"],
                    opset_version=17, dynamo=False, do_constant_folding=do_fold,
                    dynamic_axes={"rgb": {0: "n"}, "spec": {0: "n"}, "logits": {0: "n"}},
                )
                s = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
                actual_in = [i.name for i in s.get_inputs()]
                fd = {}
                if "rgb" in actual_in: fd["rgb"] = rgb.numpy()
                if "spec" in actual_in: fd["spec"] = spec_t.numpy()
                ol = s.run(["logits"], fd)[0]
                op = np.exp(ol - ol.max(1, keepdims=True))
                op = (op / op.sum(1, keepdims=True))[:, 1]
                return ol, op

            if not out_path.exists():
                ort_logits, ort_p = _export_run(True)
            else:
                s = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
                actual_in = [i.name for i in s.get_inputs()]
                fd = {}
                if "rgb" in actual_in: fd["rgb"] = rgb.numpy()
                if "spec" in actual_in: fd["spec"] = spec_t.numpy()
                ort_logits = s.run(["logits"], fd)[0]
                ort_p = np.exp(ort_logits - ort_logits.max(1, keepdims=True))
                ort_p = (ort_p / ort_p.sum(1, keepdims=True))[:, 1]

            logit_diff = float(np.abs(torch_logits - ort_logits).max())
            if logit_diff > LIMIT_LOGIT and out_path.exists():
                ort_logits, ort_p = _export_run(False)
                logit_diff = float(np.abs(torch_logits - ort_logits).max())
            archived = pd.read_csv(ROOT / f"results/phase4/seed{seed}/{mode}/preds_test.csv")
            pred_diff = float(np.abs(torch_p - archived["p_infected"].values).max())
            pred_diff_ort = float(np.abs(ort_p - archived["p_infected"].values).max())
            rec = {
                "seed": seed, "mode": mode,
                "onnx": str(out_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
                "n_test": int(len(test)),
                "max_abs_logit_diff_torch_vs_onnx": logit_diff,
                "max_abs_p_diff_torch_vs_archived": pred_diff,
                "max_abs_p_diff_onnx_vs_archived": pred_diff_ort,
                "logit_tolerance": LIMIT_LOGIT, "pred_tolerance": LIMIT_PRED,
                "pass_logit": logit_diff <= LIMIT_LOGIT,
                "pass_pred_vs_archived": pred_diff_ort <= LIMIT_PRED,
            }
            results.append(rec)
            print(f"{mode} seed{seed}: logit {logit_diff:.2e} pred_vs_arch {pred_diff_ort:.2e}",
                  "OK" if rec["pass_logit"] and rec["pass_pred_vs_archived"] else "FAIL",
                  flush=True)

    ok = all(r["pass_logit"] and r["pass_pred_vs_archived"] for r in results)
    report = {"models": results, "all_pass": ok, "runtime_s": round(time.time() - t0, 1)}
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"all_pass": ok, "runtime_s": report["runtime_s"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
