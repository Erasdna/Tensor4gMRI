#!/bin/bash -l

#SBATCH --output=slurm/preproc_%A.out
#SBATCH --job-name=preproc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3:00:00

# Prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

apptainer exec \
    --cleanenv \
    --bind $BASE_FOLDER \
    --env GMRITENSOR_USE_GPU=False \
    --env CPUS_PER_TASK=1 \
    --env MULTIPROCESSING_TASKS=$SLURM_CPUS_PER_TASK \
    --env OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1 \
    apptainer/app.sif \
    python code/data_preparation/run_processing.py "$@"
