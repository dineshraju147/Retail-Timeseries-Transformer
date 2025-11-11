#!/bin/bash

#SBATCH --job-name=train-m5              # A name for your job
#SBATCH --output=slurm_outputs/train_m5_%j.log # Log file. Make sure 'slurm_outputs' directory exists.

#SBATCH --time=64:00:00                  # Time limit (12 hours)
#SBATCH --partition=gpu                  # The partition to run on
#SBATCH --gpus-per-node=tesla_v100s:1    # Request 1 Tesla V100s GPU
#SBATCH --mem=64G                        # Request 32GB of system memory
#SBATCH --cpus-per-task=4                # Request 4 CPU cores

#SBATCH --mail-user=kattungadineshraju@gmail.com # Your email
#SBATCH --mail-type=ALL                  # Send email on job start, end, or failure

# --- Your job's commands below ---

echo "========================================================="
echo "SLURM Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE"
echo "========================================================="

# --- Activate the project-specific virtual environment ---
# --- This assumes you ran 'uv venv' in the project root ---
# --- and are submitting this job from the project root ---
echo "Activating '.venv' virtual environment..."
source .venv/bin/activate

echo "Virtual environment activated."
echo "Starting Python training script for train_m5.py..."

# Run your training script
# PyTorch Lightning will automatically detect the GPU reserved by SLURM
python train_m5_enhanced.py

echo "========================================================="
echo "Python script finished with exit code $?"
echo "Job finished."
echo "========================================================="

