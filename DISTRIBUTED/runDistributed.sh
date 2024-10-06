#!/bin/bash -l
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2               # node count
#SBATCH --ntasks-per-node=2     # total number of tasks per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a100:2             # number of gpus per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)


# Load and activate your Python environment
source $STORE/mypython/bin/deactivate
source $STORE/mypython/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export TOKENIZERS_PARALLELISM=false

# Verify the Python path and run the Python script
which python

# Use srun to distribute the Python script execution across the resources
srun python BASELINE2.py
