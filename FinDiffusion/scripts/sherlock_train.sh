#!/bin/bash
#SBATCH --job-name=findiff-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sondre.rogde@gmail.com

# Parse arguments (passed after script name: sbatch sherlock_train.sh --model ddpm_topo)
MODEL="ddpm"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# Navigate to project directory — adjust this path after uploading to Sherlock
PROJECT_DIR="$HOME/mse342-project/FinDiffusion"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found at $PROJECT_DIR"; exit 1; }

# Activate virtual environment
# If you use conda instead, replace with: conda activate findiff
source .findiffvenv/bin/activate

# W&B offline — avoids needing internet on compute nodes
# After the job, sync with: wandb sync wandb/run-*
export WANDB_MODE=offline

# Verify GPU is visible
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Full pipeline: train DDPM → generate synthetic data → train deep hedger
# --ddim makes generation ~20x faster (50 steps instead of 1000)
python scripts/pipeline.py \
    --model "$MODEL" \
    --config configs/default.yaml \
    --n_generate 10000 \
    --ddim --ddim_steps 50 \
    --wandb

echo "Done: $(date)"
