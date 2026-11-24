#!/usr/bin/env bash
# Group A serialized training chain (RAM is tight - ONE python at a time).
#   1. A3: S2 rebuilt at near-dup threshold 4, 3 seeds
#   2. A3: S2 rebuilt at near-dup threshold 6, 3 seeds
#   3. A4: mobilenet_v3_large + efficientnet_b0, 3 seeds each on S2
#   4. A5: re-train S1 excluding the 48 high-confidence cross-label images, 3 seeds
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG="analysis/groupA_chain.log"
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

run analysis/dataset1_phase2_threshold_sweep.py --threshold 4 || exit 1
run analysis/dataset1_phase2_threshold_sweep.py --threshold 6 || exit 1
run analysis/dataset1_phase2_arch_multiseed.py || exit 1
run analysis/dataset1_phase2_excl48.py || exit 1
echo "=== GROUP A CHAIN COMPLETE ===" >> "$LOG"
