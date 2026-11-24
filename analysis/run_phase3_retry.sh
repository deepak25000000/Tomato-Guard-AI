#!/usr/bin/env bash
# Re-run Phase 3 (the first attempt died in 0s on a missing output directory - see CHANGELOG),
# then hand over to run_phase4_phase5.sh. Serialized: nothing else runs concurrently (7 GB RAM).
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

# nothing else may be running (HARD RULE 5: serialized phases)
if pgrep -f "dataset1_phase2|dataset2_phase|run_phase2|run_phase4" | grep -v $$ >/dev/null 2>&1; then
  say "another analysis process seems to be running - aborting to avoid concurrent loads"
  exit 1
fi

run_step "Phase 3 - Track B data audit (retry)" results/phase3/stdout.txt \
  $PY analysis/dataset2_phase3_audit.py || exit 1

say "Phase 3 done - starting Phase 4 + 5 chain"
nohup bash analysis/run_phase4_phase5.sh >> results/phase4_chain.log 2>&1 &
say "run_phase4_phase5.sh launched (pid $!)"
