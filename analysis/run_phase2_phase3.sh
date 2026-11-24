#!/usr/bin/env bash
# Phase 2 remainder -> Phase 3, chained so nothing runs concurrently (RAM is 7 GB, swap full).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/run_log.md
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_step() {   # run_step <label> <logfile> <command...>
  local label="$1" lf="$2"; shift 2
  local t0 t1 rc
  t0=$(date +%s)
  mkdir -p "$(dirname "$lf")"   # phase output dirs may not exist yet (results/phase3 was missing)
  "$@" > "$lf" 2>&1
  rc=$?
  t1=$(date +%s)
  { echo ""; echo "## $label"
    echo "- time (UTC): $(ts)"
    echo "- command: \`$*\`"
    echo "- exit code: $rc"
    echo "- runtime: $((t1 - t0)) s"
    echo "- log: $lf"; } >> "$LOG"
  say "$label -> rc=$rc ($((t1 - t0))s)"
  return $rc
}

# --- wait for the in-flight Phase 2 pipeline ---------------------------------
while pgrep -f "run_phase2_remaining.sh" >/dev/null 2>&1; do sleep 20; done
say "phase2 pipeline finished"

# --- S1 seed variance (2 more seeds) + negative control + unseen diagnostic ---
run_step "Phase 2 - S1 seed 43" results/phase2/seed43.log \
  $PY analysis/dataset1_phase2_faithful.py --mode reproduce --tag seed43 --seed 43 || exit 1
run_step "Phase 2 - S1 seed 44" results/phase2/seed44.log \
  $PY analysis/dataset1_phase2_faithful.py --mode reproduce --tag seed44 --seed 44 || exit 1
run_step "Phase 2 - NEGATIVE CONTROL (shuffled labels)" results/phase2/shuffle_control.log \
  $PY analysis/dataset1_phase2_faithful.py --mode reproduce --tag shuffle_control --seed 42 \
      --shuffle-labels || exit 1
run_step "Phase 2 - original checkpoint on UNSEEN S2 test images" \
  results/phase2/unseen_eval.log \
  $PY analysis/dataset1_phase2_unseen_eval.py || exit 1

# --- Phase 3 ----------------------------------------------------------------
run_step "Phase 3 - Track B data audit" results/phase3/stdout.txt \
  $PY analysis/dataset2_phase3_audit.py || exit 1

say "PHASE 2 + PHASE 3 COMPLETE"
