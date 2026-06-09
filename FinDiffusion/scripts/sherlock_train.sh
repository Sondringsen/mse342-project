#!/bin/bash
#SBATCH --job-name=findiff-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sondre.rogde@gmail.com

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

# Run training (omit --gpus: it's a no-op in the current codebase)
python scripts/train.py \
    --config configs/default.yaml \
    --wandb

echo "Done: $(date)"
