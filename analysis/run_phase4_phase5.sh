#!/usr/bin/env bash
# Phase 4 (Track B experiments) -> Phase 5 (RESULTS.md + tests), chained after
# run_phase2_phase3.sh so nothing ever runs concurrently (7 GB RAM, swap full).
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
  mkdir -p "$(dirname "$lf")"
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

# --- wait for the Phase 2 + Phase 3 chain -------------------------------------
while pgrep -f "run_phase2_phase3.sh" >/dev/null 2>&1; do sleep 30; done
say "phase2+3 chain finished"

if [ ! -f results/phase3/split_leakage_assertions.json ]; then
  say "FATAL: Phase 3 did not produce its split assertions - stopping (HARD RULE 4/6)"
  exit 1
fi
if [ ! -f results/phase2/arch_selection.json ]; then
  say "FATAL: Phase 2 architecture sweep did not finish - stopping (HARD RULE 6)"
  exit 1
fi

# --- Phase 4 -----------------------------------------------------------------
run_step "Phase 4 - Track B experiments" results/phase4/stdout.txt \
  $PY analysis/dataset2_phase4_experiments.py || exit 1

# --- Phase 5 -----------------------------------------------------------------
run_step "Phase 5 - generate RESULTS.md" results/phase5_stdout.txt \
  $PY analysis/generate_results_md.py || exit 1

run_step "Phase 5 - automated tests" results/tests_stdout.txt \
  $PY -m pytest tests/ -q || exit 1

say "ALL PHASES COMPLETE"
