#!/usr/bin/env bash
# Runs the remaining Phase 2 work in order, appending a machine-made entry to results/run_log.md.
# Started in the background by the agent; waits for the in-flight faithful run2 first.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/run_log.md

say() { echo "[$(date -u +%H:%M:%S)] $*"; }

# --- wait for the in-flight faithful run2 -------------------------------------
while pgrep -f "dataset1_phase2_faithful.py --mode reproduce --tag run2" >/dev/null 2>&1; do
  sleep 20
done
say "run2 finished; starting S2 runs"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# --- S2: 3 seeds --------------------------------------------------------------
t0=$(date +%s)
$PY analysis/dataset1_phase2_s2.py --seeds 42 43 44 > results/phase2/s2_stdout.txt 2>&1
rc=$?
t1=$(date +%s)
{
  echo ""
  echo "## Phase 2 - Track A experiments on split S2"
  echo "- time (UTC): $(ts)"
  echo "- command: \`.venv/bin/python analysis/dataset1_phase2_s2.py --seeds 42 43 44\`"
  echo "- exit code: $rc"
  echo "- runtime: $((t1 - t0)) s"
  echo "- artifacts: results/phase2/s2_seed{42,43,44}.json, s2_seed_summary.csv, s2_seed_aggregate.json, s2_origckpt.json"
} >> "$LOG"
say "S2 runs done (rc=$rc, $((t1 - t0))s)"
[ $rc -ne 0 ] && { say "S2 FAILED - stopping"; exit $rc; }

# --- architecture sweep (validation-only selection, one test evaluation) ------
t0=$(date +%s)
$PY analysis/dataset1_phase2_arch.py > results/phase2/arch_stdout.txt 2>&1
rc=$?
t1=$(date +%s)
{
  echo ""
  echo "## Phase 2 - Track A architecture sweep (S2, validation-only selection)"
  echo "- time (UTC): $(ts)"
  echo "- command: \`.venv/bin/python analysis/dataset1_phase2_arch.py\`"
  echo "- exit code: $rc"
  echo "- runtime: $((t1 - t0)) s"
  echo "- artifacts: results/phase2/arch_*_val.json, arch_selection.json, arch_winner_test.json, arch_summary.csv"
} >> "$LOG"
say "arch sweep done (rc=$rc, $((t1 - t0))s)"
exit $rc
