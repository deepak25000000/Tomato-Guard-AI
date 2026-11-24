"""Step 4 - Export the trained model for the app: PyTorch -> ONNX (+ parity check) -> TFLite.

  python 04_export.py --run runs/mobilenet_v3_small --app-dir ../app

TFLite conversion uses `ai-edge-torch` (Linux). If it is not installed the ONNX file is still
produced and the script tells you the fallback (onnx2tf). ALWAYS read the parity numbers it prints.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--app-dir", default=None, help="Flutter project dir; model files are copied to assets/models")
    ap.add_argument("--uncertain-threshold", type=float, default=0.60)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    classes, size = meta["classes"], meta["input_size"]

    model = build_model(meta["arch"], len(classes), pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ref = model(dummy).numpy()

    onnx_path = run / "model.onnx"
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"ONNX parity max|diff| = {np.abs(ref - onnx_out).max():.2e}  (size {onnx_path.stat().st_size/2**20:.2f} MB)")

    tflite_path = run / "tomatoguard.tflite"
    try:
        import ai_edge_torch
        ai_edge_torch.convert(model, (dummy,)).export(str(tflite_path))
        print(f"TFLite written: {tflite_path} ({tflite_path.stat().st_size/2**20:.2f} MB)")
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
            it = Interpreter(model_path=str(tflite_path)); it.allocate_tensors()
            inp, outp = it.get_input_details()[0], it.get_output_details()[0]
            x = dummy.numpy() if inp["shape"][1] == 3 else dummy.numpy().transpose(0, 2, 3, 1)
            it.set_tensor(inp["index"], x.astype(np.float32)); it.invoke()
            print(f"TFLite parity max|diff| = {np.abs(ref - it.get_tensor(outp['index'])).max():.2e}"
                  f"  input shape {inp['shape'].tolist()}")
        except Exception as e:  # noqa: BLE001
            print("Could not run TFLite parity check:", e)
    except ImportError:
        print("ai-edge-torch not installed -> skipping TFLite. Options:\n"
              "  pip install ai-edge-torch        (Linux)\n"
              f"  or: pip install onnx2tf tensorflow && onnx2tf -i {onnx_path} -o {run}/tf_out")
        tflite_path = None

    app_meta = {
        "labels": classes, "input_size": size, "mean": meta["mean"], "std": meta["std"],
        "temperature": meta["temperature"], "uncertain_threshold": a.uncertain_threshold,
        "healthy_label": next((c for c in classes if "healthy" in c.lower()), None),
        "not_tomato_label": next((c for c in classes if "not tomato" in c.lower()), None),
        "arch": meta["arch"],
    }
    (run / "model_meta.json").write_text(json.dumps(app_meta, indent=2))
    if a.app_dir:
        dst = Path(a.app_dir) / "assets" / "models"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "model_meta.json", dst / "model_meta.json")
        if tflite_path and tflite_path.exists():
            shutil.copy(tflite_path, dst / "tomatoguard.tflite")
        print("Copied to", dst)


if __name__ == "__main__":
    main()
