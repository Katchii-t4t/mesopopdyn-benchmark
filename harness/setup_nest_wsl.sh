#!/usr/bin/env bash
# setup_nest_wsl.sh
# Run this INSIDE WSL (Ubuntu) after WSL2 is installed and you have rebooted.
#
#   wsl                       # enter Ubuntu
#   cd /mnt/c/Users/Karthik/Documents/mesopopdyn-internship/harness
#   bash setup_nest_wsl.sh
#
# It installs Miniforge (if missing), creates a 'nest' conda env with NEST +
# the Python deps, and verifies that `import nest` works.
#
# Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.
set -euo pipefail

echo "=== CogPrint/mesopopdyn NEST setup (WSL) ==="

# 1. Miniforge (conda-forge-first conda) if no conda present
if ! command -v conda >/dev/null 2>&1; then
  echo "[1/4] installing Miniforge ..."
  cd /tmp
  wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O miniforge.sh
  bash miniforge.sh -b -p "$HOME/miniforge3"
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda init bash
else
  echo "[1/4] conda already present"
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# 2. create env with NEST (conda-forge has Linux builds of nest-simulator)
if conda env list | grep -q '^nest '; then
  echo "[2/4] 'nest' env already exists"
else
  echo "[2/4] creating 'nest' env (this pulls NEST + deps) ..."
  conda create -y -n nest -c conda-forge \
      python=3.11 nest-simulator numpy scipy matplotlib numba
fi

# 3. activate
echo "[3/4] activating 'nest' ..."
conda activate nest

# 4. verify
echo "[4/4] verifying NEST ..."
python - <<'PY'
import nest
print("NEST version:", nest.__version__)
# the two models the harness needs:
for m in ("gif_pop_psc_exp", "gif_psc_exp"):
    assert m in nest.node_models, f"missing model {m}"
    print("  model OK:", m)
print("NEST is ready.")
PY

echo
echo "=== done. To use it later: ==="
echo "  conda activate nest"
echo "  cd /mnt/c/Users/Karthik/Documents/mesopopdyn-internship/harness"
echo "  python benchmark_nest.py --scaling 0.1 --T 1   # quick check"
