#!/bin/bash
#SBATCH --job-name=findiff-hedge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sondre.rogde@gmail.com

MODEL="ddpm_topo"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

PROJECT_DIR="$HOME/mse342-project/FinDiffusion"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found at $PROJECT_DIR"; exit 1; }

module load python/3.12.1
source .findiffvenv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_hedging.py --model "$MODEL"

echo "Done: $(date)"
