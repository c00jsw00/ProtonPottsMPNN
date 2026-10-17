#!/usr/bin/env bash
set -uo pipefail
cd /home/c00jsw00/ProtonPottsMPNN
PY=.venv/bin/pip
echo "[1/5] upgrading pip"
$PY install --upgrade pip wheel
echo "[2/5] installing torch (cu128, GPU)"
if ! $PY install "torch" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1 | grep -qi 'Successfully installed'; then
  echo "  (cu128 attempt done; proceeding)"
fi
echo "[3/5] installing foundry (core mpnn)"
$PY install -e ./foundry
echo "[4/5] installing biotite pandas numpy"
$PY install biotite pandas numpy
echo "[5/5] verifying imports"
.venv/bin/python -c "import torch; print('  torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
.venv/bin/python -c "import mpnn; print('  mpnn at', mpnn.__file__)"
.venv/bin/python -c "import biotite; print('  biotite ok')"
echo "=== DONE ==="
.venv/bin/python -c "
from mpnn.inference_engines.potts_mpnn_ph import PottsMPNNPHEngine, PHDesignCriteria
from mpnn.potts_inference import prepare_potts_input, load_model
print('  pH engine + potts_inference import ok')
"
echo "=== ALL DONE ==="
