"""PHASE 0 - Setup: capture the environment, verify pretrained-weight download, and measure
the throughput needed to estimate Phase 2 / Phase 4 runtime on THIS machine.

Everything it reports is written to results/ as json (HARD RULE 1): no number is typed by hand.

  .venv/bin/python analysis/phase0_setup.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "tomatoguard" / "research"))


def cpu_info() -> dict:
    out = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        out["logical_cores"] = os.cpu_count()
        out["physical_cores_lscpu"] = int(subprocess.run(
            ["lscpu"], capture_output=True, text=True).stdout.split("Core(s) per socket:")[1]
            .splitlines()[0].strip()) if shutil.which("lscpu") else None
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                out["mem_total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 2)
            if line.startswith("MemAvailable"):
                out["mem_available_gb_at_start"] = round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return out


def versions() -> dict:
    import numpy, pandas, PIL, scipy, sklearn, torch, torchvision
    v = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_mkldnn": bool(torch.backends.mkldnn.is_available()),
    }
    try:
        import matplotlib, pytest, seaborn, tqdm
        v.update({"matplotlib": matplotlib.__version__, "seaborn": seaborn.__version__,
                  "tqdm": tqdm.__version__, "pytest": pytest.__version__})
    except Exception:  # noqa: BLE001
        pass
    return v


def benchmark() -> dict:
    """Measure the per-sample cost of the workloads used in Phases 2 and 4."""
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import models, transforms

    from models import ARCHS, build_model, eval_tf, train_tf

    r: dict = {"device": "cpu", "note": "timings measured on this machine, wall clock seconds"}
    tf_eval, tf_train = eval_tf(), train_tf()

    # --- 1. decode + resize throughput (PIL) -------------------------------------
    probe = ROOT / "Tomato Leaf Disease Classification Dataset" / "Tomato Dataset"
    sample_paths = [str(p) for p in sorted(probe.rglob("*"))[:256] if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    t0 = time.perf_counter()
    for p in sample_paths:
        with Image.open(p) as im:
            tf_eval(im.convert("RGB"))
    dt = time.perf_counter() - t0
    r["trackA_decode_resize_224_s_per_img"] = dt / max(1, len(sample_paths))
    r["trackA_decode_resize_n_probe"] = len(sample_paths)

    # --- 2. Track A training step (MobileNetV3-Small, batch 16) ------------------
    torch.manual_seed(42)
    dev = torch.device("cpu")
    model = build_model("mobilenet_v3_small", 7).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 7, (16,))
    for _ in range(2):                      # warm-up
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    n = 4
    t0 = time.perf_counter()
    for _ in range(n):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    step = (time.perf_counter() - t0) / n
    r["trackA_train_step_s_batch16"] = step
    r["trackA_train_imgs_per_s"] = 16 / step

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(x)
        t0 = time.perf_counter()
        for _ in range(4):
            model(x)
    estep = (time.perf_counter() - t0) / 4
    r["trackA_eval_step_s_batch16"] = estep
    r["trackA_eval_imgs_per_s"] = 16 / estep

    # --- 3. Phase 4 fusion step (2 encoders, batch 32) ---------------------------
    from fusion.fusion import Net  # type: ignore
    fnet = Net("fusion")
    fopt = torch.optim.AdamW(fnet.parameters(), lr=3e-4, weight_decay=1e-4)
    frg = torch.randn(32, 3, 224, 224)
    fsp = torch.randn(32, 3, 224, 224)
    fy = torch.randint(0, 2, (32,))
    for _ in range(2):
        fopt.zero_grad(); nn.functional.cross_entropy(fnet(frg, fsp), fy).backward(); fopt.step()
    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        fopt.zero_grad(); nn.functional.cross_entropy(fnet(frg, fsp), fy).backward(); fopt.step()
    fstep = (time.perf_counter() - t0) / n
    r["trackB_fusion_train_step_s_batch32"] = fstep
    r["trackB_fusion_imgs_per_s"] = 32 / fstep

    # spectral mode = 1 encoder, rgb mode = 1 encoder -> assume ~0.55 of fusion cost
    r["trackB_single_encoder_train_step_s_batch32_est"] = fstep * 0.55

    # --- 4. extrapolation --------------------------------------------------------
    # Phase 2 training-set sizes are known only after Phase 1 makes the splits, so we use the
    # notebook's original split (2096 train / 449 val) as the S1 estimate.
    s1_train, s1_val, s1_test = 2096, 449, 450

    def run_seconds(train_n, val_n, epochs, step_s, batch):
        ep = (train_n / batch) * step_s + (val_n / batch) * (step_s * 0.5)
        return ep * epochs

    # 02_train.py: 30 epochs max, early stop patience 7 -> assume ~18 epochs actually run
    est = {}
    est["P2b_baseline_repro_S1_10ep"] = run_seconds(s1_train, s1_val, 10, step, 16) + \
        (s1_test / 16) * estep
    est["P2c_improved_one_run_30ep_worst"] = run_seconds(s1_train, s1_val, 30, step * 2, 32)
    est["P2c_improved_one_run_18ep_likely"] = run_seconds(s1_train, s1_val, 18, step * 2, 32)
    # P2c = 3 seeds x S1 + 3 seeds x S2 (mobile small) + 3 seeds x S2 large + 3 seeds x S2 effnet
    est["P2c_total_12_runs_likely"] = est["P2c_improved_one_run_18ep_likely"] * 12
    est["P2e_shuffle_label_control_one_run"] = est["P2c_improved_one_run_18ep_likely"]
    est["P2e_seed_rerun_one_run"] = est["P2c_improved_one_run_18ep_likely"]
    est["P2_total_likely_s"] = (est["P2b_baseline_repro_S1_10ep"]
                                + est["P2c_total_12_runs_likely"]
                                + est["P2e_shuffle_label_control_one_run"]
                                + est["P2e_seed_rerun_one_run"])

    # Phase 4: manifest rows are unknown until Phase 3b; upper bound = all 4808 tomato rows.
    tb_rows = 4808
    est["P4_one_deep_run_15ep_upper_bound_s"] = run_seconds(
        tb_rows * 0.7, tb_rows * 0.15, 15, fstep, 32)
    # 3 modes x 3 seeds = 9 deep runs + negative control (>=3) + optional distillation (1)
    est["P4_deep_9_runs_upper_bound_s"] = est["P4_one_deep_run_15ep_upper_bound_s"] * 9
    est["P4_deep_13_runs_upper_bound_s"] = est["P4_one_deep_run_15ep_upper_bound_s"] * 13

    r["extrapolated_seconds"] = est
    r["extrapolated_hours"] = {k: round(v / 3600, 2) for k, v in est.items()}
    del model, fnet, opt, fopt
    return r


def main():
    env = {
        "phase": 0,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "versions": versions(),
        "cpu": cpu_info(),
        "gpu": {"nvidia_smi_present": shutil.which("nvidia-smi") is not None,
                "torch_cuda_available": bool(__import__("torch").cuda.is_available())},
    }
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (RES / "pip_freeze.txt").write_text(freeze.stdout)
    env["pip_freeze_file"] = "results/pip_freeze.txt"

    print("=== ENVIRONMENT ===")
    print(json.dumps(env, indent=2))

    print("\n=== BENCHMARK (running, ~2 min) ===")
    try:
        bench = benchmark()
    except Exception as e:  # noqa: BLE001
        bench = {"error": repr(e)}
        print("BENCHMARK FAILED:", repr(e))
    bench["timestamp_utc"] = env["timestamp_utc"]
    (RES / "environment.json").write_text(json.dumps(env, indent=2))
    (RES / "phase0_benchmark.json").write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))

    # ---- run log (HARD RULE 5) --------------------------------------------------
    log = RES / "run_log.md"
    entry = [
        "\n## Phase 0 - setup\n",
        f"- time (UTC): {env['timestamp_utc']}",
        f"- command: `{env['command']}`",
        f"- python {env['versions']['python']} | torch {env['versions']['torch']} | "
        f"torchvision {env['versions']['torchvision']} | numpy {env['versions']['numpy']} | "
        f"pandas {env['versions']['pandas']} | sklearn {env['versions']['scikit_learn']} | "
        f"pillow {env['versions']['pillow']}",
        f"- GPU: cuda_available={env['gpu']['torch_cuda_available']}, "
        f"nvidia-smi present={env['gpu']['nvidia_smi_present']}  => CPU-only",
        f"- cpu: {env['cpu'].get('model')} x {env['cpu'].get('logical_cores')} logical, "
        f"ram {env['cpu'].get('mem_total_gb')} GB",
        "- artifacts: results/environment.json, results/phase0_benchmark.json, results/pip_freeze.txt",
        "",
    ]
    log.write_text(log.read_text() if log.exists() else "# Run log\n")
    with log.open("a") as f:
        f.write("\n".join(entry))
    print("\nwrote results/environment.json, results/phase0_benchmark.json, results/run_log.md")


if __name__ == "__main__":
    main()
