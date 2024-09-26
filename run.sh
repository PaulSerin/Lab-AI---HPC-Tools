#!/bin/bash
# Based on: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --mem=32G                # total memory per node
#SBATCH --gres=gpu:a100          # number of gpus per node
#SBATCH --cpus-per-task=32       # number of CPUs required per GPU
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)

# Load and activate your Python environment
source $STORE/mypython/bin/deactivate
source $STORE/mypython/bin/activate

# Verify the Python path and run the Python script
which python

# Use srun to distribute the Python script execution across the resources
srun python BASELINE.py
