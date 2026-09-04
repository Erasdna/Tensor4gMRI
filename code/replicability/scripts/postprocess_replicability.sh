#!/bin/bash -l

#SBATCH --output=slurm/%x_%j.out
#SBATCH --job-name=post_processnig
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=00:10:00

SCRIPT_PATH="code/replicability/scripts"

cd $SLURM_SUBMIT_DIR
uv run --no-sync python $SCRIPT_PATH/postprocess_replicability.py $1 $2
