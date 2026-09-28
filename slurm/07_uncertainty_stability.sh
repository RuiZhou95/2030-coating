#!/bin/bash
#SBATCH --job-name=coat-uncert
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/07_uncertainty_stability_%j.out
#SBATCH --error=logs/07_uncertainty_stability_%j.err

set -euo pipefail
PROJECT_ROOT="${COATING_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -n "${COATING_CONDA_SH:-}" ]]; then
  source "$COATING_CONDA_SH"
fi
if [[ -n "${COATING_CONDA_ENV:-}" ]]; then
  conda activate "$COATING_CONDA_ENV"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"${COATING_PYTHON:-python3}" src/07_uncertainty_stability.py
