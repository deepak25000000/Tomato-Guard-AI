#!/usr/bin/env bash
# Phase 1 end-to-end: splits -> train 3 architectures -> evaluate -> compare.
# Usage: bash scripts/run_phase1.sh "/path/to/Tomato Leaf Disease Classification Dataset/Tomato Dataset"
set -euo pipefail
DATA_DIR="${1:?usage: run_phase1.sh /path/to/'Tomato Dataset'}"
cd "$(dirname "$0")/../research"
python 01_make_splits.py --data-dir "$DATA_DIR"
for arch in mobilenet_v3_small mobilenet_v3_large efficientnet_b0; do
  python 02_train.py --arch "$arch"
  python 03_evaluate.py --run "runs/$arch"
done
python 05_compare.py
echo
echo "Pick the best run, then:  python 04_export.py --run runs/<best> --app-dir ../app"
