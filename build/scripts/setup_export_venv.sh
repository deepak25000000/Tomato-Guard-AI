#!/usr/bin/env bash
# Phase 1.5 preflight - the pinned Python 3.11 EXPORT-ONLY venv (approved: build/PLAN.md §5.1).
# Created because Python 3.14 has no TensorFlow wheel and ai-edge-torch would downgrade
# torch in the main venv (build/run_log.md §0a evidence).
# Installs: CPU-only torch/torchvision (no CUDA wheels), ai-edge-torch + its deps, pillow.
# Restart-proof: skips if the venv python + ai_edge_torch import already work.
# All downloads go through pip's on-disk cache; nothing is written to /tmp (tmpfs=RAM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/build/venvs/export-311"
PY311="/home/deepak-thorat/.local/bin/python3.11"
LOG="$ROOT/build/export_venv_install.log"
# /tmp is a RAM-backed tmpfs here: pip extracts wheels there, which would spike RAM.
# Force pip/venv temp extraction onto disk instead.
export TMPDIR="$ROOT/build/piptmp"
mkdir -p "$TMPDIR"

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import ai_edge_torch" 2>/dev/null; then
  echo "[skip] export venv ready: $("$VENV/bin/python" --version)"
  "$VENV/bin/python" -c "import torch, torchvision, ai_edge_torch, PIL; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"
  exit 0
fi

echo "=== creating $VENV (Python 3.11) ==="
"$PY311" -m venv "$VENV"
VP="$VENV/bin/python"
"$VP" -m pip install -q --upgrade pip

echo "=== CPU-only torch + torchvision (pinned to the resolution pip picks; logged) ==="
"$VP" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee -a "$LOG"

echo "=== ai-edge-torch + conversion deps (CPU torch index to avoid CUDA wheels) ==="
"$VP" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu ai-edge-torch pillow 2>&1 | tee -a "$LOG"

echo "=== freeze export-venv versions ==="
"$VP" -m pip freeze > "$ROOT/build/requirements_export.lock"

echo "=== verify ==="
"$VP" - <<'PY'
import torch, torchvision, ai_edge_torch, PIL, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("ai_edge_torch", getattr(ai_edge_torch, "__version__", "?"))
print("pillow", PIL.__version__)
PY
echo "EXPORT_VENV_READY"
