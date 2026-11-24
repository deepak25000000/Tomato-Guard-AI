#!/usr/bin/env bash
# Group B/C tail chain - run AFTER run_groupA_chain.sh (RAM is tight: ONE python at a time).
#   1. A5b spec extras (needs the chain's thresh4/6, arch3 and excl48 outputs)
#   2. B4  confound baselines (5 seeds, plant-disjoint vs image-level demo)
#   3. B6  fusion at 224 px / 15 epochs / 1 seed (higher-budget ordering check)
#   4. run-log entries for every Group A/B run
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG="analysis/groupB_tail.log"
: > "$LOG"

run() {
  echo "=== START $* ===" >> "$LOG"
  start=$(date +%s)
  "$PY" "$@" >> "$LOG" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== DONE $* rc=$rc runtime=$((end-start))s ===" >> "$LOG"
  return $rc
}

# Restart-proof: skip any step whose FINAL output already exists (jsons are only
# written after a step fully completes, so a killed step leaves no file).
if [ -f results/phase2/spec_extras_summary.json ]; then
  echo "=== SKIP spec extras (results/phase2/spec_extras_summary.json exists) ===" >> "$LOG"
else
  run analysis/dataset1_phase2_spec_extras.py || exit 1
fi
if [ -f results/phase4/confounds/confound_baselines.json ]; then
  echo "=== SKIP confound baselines (output exists) ===" >> "$LOG"
else
  run analysis/dataset2_phase4_confound_baselines.py || exit 1
fi

if [ -f results/phase4/budget_check_px224_e15/fusion/metrics.json ]; then
  echo "=== SKIP fusion B6 (metrics.json exists) ===" >> "$LOG"
else
mkdir -p results/phase4/budget_check_px224_e15/fusion
# run config recorded BY THE RUNNER from its own command line (config = input, not a result)
"$PY" - <<'PYEOF'
import json
from datetime import datetime, timezone
cfg = {"mode": "fusion", "epochs": 15, "image_size": 224, "batch_size": 32,
       "num_workers": 0, "split_seed": 42,
       "manifest": "results/phase3/plant_splits/manifest_split_seed42.csv",
       "command": "tomatoguard/research/fusion/fusion.py --mode fusion --manifest "
                  "results/phase3/plant_splits/manifest_split_seed42.csv --out "
                  "results/phase4/budget_check_px224_e15/fusion --epochs 15 "
                  "--image-size 224 --batch-size 32 --num-workers 0 --seed 42",
       "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "selection": "best epoch by VALIDATION image AUC; test evaluated once"}
open("results/phase4/budget_check_px224_e15/run_config.json", "w").write(
    json.dumps(cfg, indent=2))
PYEOF
echo "=== START fusion B6 (224px, 15 epochs, seed 42) ===" >> "$LOG"
start=$(date +%s)
"$PY" tomatoguard/research/fusion/fusion.py \
  --mode fusion \
  --manifest results/phase3/plant_splits/manifest_split_seed42.csv \
  --out results/phase4/budget_check_px224_e15/fusion \
  --epochs 15 --image-size 224 --batch-size 32 --num-workers 0 --seed 42 \
  >> "$LOG" 2>&1
rc=$?
end=$(date +%s)
echo "=== DONE fusion B6 rc=$rc runtime=$((end-start))s ===" >> "$LOG"
[ $rc -eq 0 ] || exit 1
fi

run analysis/append_group_run_log.py || exit 1
echo "=== GROUP B TAIL COMPLETE ===" >> "$LOG"
