#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Astribot Kinematics — Complete Environment Setup Script
#
# Creates a working Python environment with all dependencies installed via pip.
#
# Usage:
#   bash install.sh                     # Install into current Python env
#   bash install.sh --create-env        # Create new conda env 'kinematics' first
#   bash install.sh --sim               # Also install Meshcat + MuJoCo (simulation)
# -----------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Parse arguments
CREATE_ENV=0
INSTALL_SIM=0
for arg in "$@"; do
  case "$arg" in
    --create-env) CREATE_ENV=1 ;;
    --sim) INSTALL_SIM=1 ;;
    -h|--help)
      sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# -----------------------------------------------------------------------------
# 1. Environment Setup (optional)
# -----------------------------------------------------------------------------
if [[ "$CREATE_ENV" -eq 1 ]]; then
  echo "[install] Creating conda environment 'kinematics' with Python 3.10..."
  conda create -n kinematics python=3.10 -y
  echo "[install] Environment created. Activate it and re-run:"
  echo "    conda activate kinematics"
  echo "    bash install.sh"
  exit 0
fi

PYTHON_BIN="${PYTHON:-python3}"
echo "[install] Using Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
${PYTHON_BIN} -c 'import sys; assert sys.version_info >= (3, 8), "Python >= 3.8 required"'
echo "[install] Python version: $(${PYTHON_BIN} --version)"

# -----------------------------------------------------------------------------
# 2. Install Core Dependencies
# -----------------------------------------------------------------------------
echo "[install] Installing core dependencies..."
${PYTHON_BIN} -m pip install --upgrade pip
${PYTHON_BIN} -m pip install "numpy>=1.20" "pin>=3.0"  # pin = pinocchio

# -----------------------------------------------------------------------------
# 3. Install Test Dependencies
# -----------------------------------------------------------------------------
echo "[install] Installing test dependencies: pytest, h5py..."
${PYTHON_BIN} -m pip install "pytest>=7" h5py

# -----------------------------------------------------------------------------
# 3b. Optional Simulation Dependencies
# -----------------------------------------------------------------------------
if [[ "$INSTALL_SIM" -eq 1 ]]; then
  echo "[install] Installing simulation dependencies: meshcat, mujoco..."
  ${PYTHON_BIN} -m pip install "meshcat>=0.3" "mujoco>=3.0"
fi

# -----------------------------------------------------------------------------
# 4. Install astribot_kinematics Package
# -----------------------------------------------------------------------------
echo "[install] Installing astribot_kinematics..."
${PYTHON_BIN} -m pip install .

# -----------------------------------------------------------------------------
# 5. Smoke Test
# -----------------------------------------------------------------------------
echo "[install] Running smoke test..."
${PYTHON_BIN} -c "
import os
from astribot_kinematics import AstribotFK, AstribotIK
import numpy as np

fk = AstribotFK()
q = np.zeros(fk.nq)
pose = fk.eef_left(q)

meshes_dir = os.path.join(os.path.dirname(fk.urdf_path), 'meshes')
n_stl = sum(1 for _, _, fs in os.walk(meshes_dir) for f in fs if f.endswith('.STL'))

print(f'[install] OK - nq={fk.nq}, urdf={fk.urdf_path}')
print(f'[install] bundled meshes: {n_stl} STL files under {meshes_dir}')
print(f'[install] EEF_left(q=0) xyz = {pose[:3].round(4)}')
"

echo ""
echo "[install] Installation complete!"
echo ""
echo "Quick start:"
echo "    python -c \"from astribot_kinematics import AstribotFK; print(AstribotFK().eef_left([0]*18))\""
echo ""
echo "Run tests:"
echo "    pytest tests/ -q"
