"""PHASE 0 (part 2) - turn the measured benchmark into explicit runtime scenarios for
Phase 2 and Phase 4, so the reduced-settings proposal is computed, not typed by hand.

  .venv/bin/python analysis/phase0_propose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

B = json.loads((RES / "phase0_benchmark.json").read_text())
if "error" in B:
    sys.exit("benchmark failed: " + B["error"])

STEP16 = B["trackA_train_step_s_batch16"]        # s / 16 images
EVAL16 = B["trackA_eval_step_s_batch16"]
DECODE = B["trackA_decode_resize_224_s_per_img"]
FUS32 = B["trackB_fusion_train_step_s_batch32"]  # s / 32 images, 2 encoders

# S1 = the notebook's original split (sizes read from the user's own CSVs, not typed)
SPLITS = ROOT.parent / "Tomato_Research_Results"
sizes = {}
for name in ("train", "validation", "test"):
    with (SPLITS / f"{name}.csv").open() as f:
        sizes[name] = sum(1 for _ in f) - 1
S1_TRAIN, S1_VAL, S1_TEST = sizes["train"], sizes["validation"], sizes["test"]


def trackA_run(train_n, val_n, epochs, batch=32) -> float:
    """Seconds for one 02_train.py-style run (batch-32 step estimated as 2x the measured
    batch-16 step, val step assumed 0.5x of train step because it has no backward)."""
    step = STEP16 * (batch / 16)
    return epochs * ((train_n / batch) * step + (val_n / batch) * step * 0.5)


def trackB_run(rows_train, rows_val, epochs, batch=32, size=224) -> float:
    px = (size / 224) ** 2
    step = FUS32 * px
    return epochs * ((rows_train / batch) * step + (rows_val / batch) * step * 0.5)


def h(s):
    return round(s / 3600, 2)


# ---------------------------------------------------------------- Phase 2 scenarios
def p2_full():
    runs = {"P2b baseline reproduction on S1 (10 epochs, faithful)": 10}
    per = 18  # 02_train.py default 30 epochs with patience 7 -> assume ~18 actually run
    n_improved = 12   # 3 seeds x S1 + 3 x S2 (v3-small) + 3 x S2 (v3-large) + 3 x S2 (effnet-b0)
    t = trackA_run(S1_TRAIN, S1_VAL, 10) + n_improved * trackA_run(S1_TRAIN, S1_VAL, per) \
        + 2 * trackA_run(S1_TRAIN, S1_VAL, per)   # shuffle-label control + identical-seed rerun
    return t, {"baseline_repro_runs": 1, "improved_runs": n_improved,
               "sanity_runs": 2, "assumed_epochs_improved": per, "runs_total": 15}


def p2_reduced():
    # Faithful 10-epoch reproduction kept untouched; everything else at 14 epochs / patience 5,
    # architecture sweep down to 1 seed (reported as a single run, NOT mean+/-sd).
    per = 14
    detail = {"baseline_repro_runs": 1, "epochs_improved": per,
              "S1_v3small_seeds": 3, "S2_v3small_seeds": 3,
              "S2_v3large_seeds": 1, "S2_effnetb0_seeds": 1,
              "sanity_runs": 2}
    n = 3 + 3 + 1 + 1 + 2
    t = trackA_run(S1_TRAIN, S1_VAL, 10) + n * trackA_run(S1_TRAIN, S1_VAL, per)
    detail["runs_total"] = n + 1
    return t, detail


def p2_minimal():
    per = 10
    detail = {"baseline_repro_runs": 1, "epochs_improved": per,
              "S1_v3small_seeds": 3, "S2_v3small_seeds": 3,
              "S2_v3large_seeds": 1, "S2_effnetb0_seeds": 1,
              "sanity_runs": 1, "note": "shuffle-label control only; identical-seed rerun kept"}
    n = 3 + 3 + 1 + 1 + 2
    t = trackA_run(S1_TRAIN, S1_VAL, 10) + n * trackA_run(S1_TRAIN, S1_VAL, per)
    detail["runs_total"] = n + 1
    return t, detail


# ---------------------------------------------------------------- Phase 4 scenarios
TB_TRAIN, TB_VAL = int(4808 * 0.70), int(4808 * 0.15)   # split_by_plant defaults 0.2 test / 0.15 val


def p4_full():
    # 3 modes x 3 split seeds + 3 plant-shuffled negative controls + 1 distillation run
    n = 13
    t = n * trackB_run(TB_TRAIN, TB_VAL, 15)
    return t, {"deep_runs": n, "epochs": 15, "image_size": 224, "split_seeds": 3,
               "negative_control_runs": 3, "distillation_runs": 1}


def p4_reduced():
    # keep the spec minimum (3 split seeds, 3 modes), 8 epochs, drop distillation,
    # 1 negative-control run
    n = 9 + 1
    t = n * trackB_run(TB_TRAIN, TB_VAL, 8)
    return t, {"deep_runs": 9, "negative_control_runs": 1, "distillation_runs": 0,
               "epochs": 8, "image_size": 224, "split_seeds": 3}


def p4_minimal():
    # 8 epochs at 160x160 (MobileNetV3-Small has adaptive pooling, so any size works),
    # 3 modes x 3 seeds + 1 control
    n = 9 + 1
    t = n * trackB_run(TB_TRAIN, TB_VAL, 8, size=160)
    return t, {"deep_runs": 9, "negative_control_runs": 1, "distillation_runs": 0,
               "epochs": 8, "image_size": 160, "split_seeds": 3}


THRESHOLD_H = 2.0
out = {"generated_from": "results/phase0_benchmark.json",
       "threshold_hours": THRESHOLD_H,
       "s1_split_sizes_measured": {"train": S1_TRAIN, "validation": S1_VAL, "test": S1_TEST},
       "phase2": {}, "phase4": {}}
for key, fn in (("as_specified", p2_full), ("reduced", p2_reduced), ("minimal", p2_minimal)):
    t, d = fn()
    out["phase2"][key] = {"hours": h(t), "exceeds_threshold": h(t) > THRESHOLD_H, **d}
for key, fn in (("as_specified", p4_full), ("reduced", p4_reduced), ("minimal", p4_minimal)):
    t, d = fn()
    out["phase4"][key] = {"hours": h(t), "exceeds_threshold": h(t) > THRESHOLD_H, **d}

(RES / "phase0_runtime_proposal.json").write_text(json.dumps(out, indent=2))

# append a run-log entry generated from the json (never hand-typed)
log = RES / "run_log.md"
old = log.read_text() if log.exists() else "# Run log\n"
marker = "## Phase 0 - runtime proposal (awaiting approval)"
if marker in old:                       # keep the log idempotent on re-runs, but remove
    i = old.index(marker)               # ONLY our own block - entries other phases
    j = old.find("\n## ", i + 1)        # appended after it must survive (CHANGELOG)
    head = old[:i].rstrip("\n")
    tail = "" if j == -1 else old[j + 1:]
    old = head + ("\n\n" + tail if tail else "\n")
lines = [f"\n{marker}\n",
         f"- generated_from: results/phase0_benchmark.json",
         f"- split sizes measured from the user's original CSVs: {out['s1_split_sizes_measured']}",
         "\n| phase | scenario | hours | >2h |",
         "|---|---|---|---|"]
for phase in ("phase2", "phase4"):
    for k, v in out[phase].items():
        lines.append(f"| {phase} | {k} | {v['hours']} | {v['exceeds_threshold']} |")
lines.append("\nArtifacts: results/phase0_runtime_proposal.json, results/phase0_benchmark.json\n")
log.write_text(old + "\n".join(lines))

print(json.dumps(out, indent=2))
print("\nappended to results/run_log.md")
