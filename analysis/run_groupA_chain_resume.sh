#!/usr/bin/env bash
# Group A chain RESUME after the 2026-11-22 host restart.
# State at restart: threshold 4 DONE, threshold 6 DONE, arch3 large{42,43,44} +
# efficientnet{42} complete on disk, efficientnet{43} killed mid-run (no output
# files), efficientnet{44} and excl48 not started.
# Appends to the SAME log (no truncation) so the update file's status reader works.
set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG="analysis/groupA_chain.log"

run() {
  echo "=== START $* ===" >> "$LOG"
  start=$(date +%s)
  "$PY" "$@" >> "$LOG" 2>&1
  rc=$?
  end=$(date +%s)
  echo "=== DONE $* rc=$rc runtime=$((end-start))s ===" >> "$LOG"
  return $rc
}

echo "=== RESUME after host restart (t4+t6 done; arch3 partially done) ===" >> "$LOG"
run analysis/dataset1_phase2_arch_multiseed.py || exit 1
run analysis/dataset1_phase2_excl48.py || exit 1
echo "=== GROUP A CHAIN COMPLETE ===" >> "$LOG"
