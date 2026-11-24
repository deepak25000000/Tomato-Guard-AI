#!/usr/bin/env bash
# Waits for the Phase 4 -> Phase 5 -> pytest chain to finish, then reports its state.
# Exits 0 on the chain's own completion marker, 1 if the chain died early, 2 on timeout.
cd "$(dirname "$0")/.."
for i in $(seq 1 240); do   # 2 h at 30 s
  if grep -q "ALL PHASES COMPLETE" results/phase4_chain.log 2>/dev/null; then
    echo "=== CHAIN COMPLETE ==="
    tail -6 results/phase4_chain.log
    echo "--- pytest result ---"
    tail -3 results/tests_stdout.txt 2>/dev/null
    exit 0
  fi
  if ! pgrep -f "run_phase4_phase5.sh|dataset2_phase4_experiments.py|fusion/fusion.py|generate_results_md.py|pytest tests" >/dev/null 2>&1; then
    echo "=== CHAIN STOPPED WITHOUT COMPLETION MARKER ==="
    tail -10 results/phase4_chain.log 2>/dev/null
    exit 1
  fi
  sleep 30
done
echo "=== WATCHER TIMEOUT (2 h) ==="
exit 2
