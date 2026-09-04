#!/bin/bash -l

#SBATCH --output=slurm/replicability_%A_%a.out
#SBATCH --job-name=replicability
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=3:00:00

module load CUDA/12.1.1-GCC-12.3.0
module load GCCcore/12.3.0

PARAM_FILE=$1
CURRENT_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_FILE")

# Read inputs from parameter file
echo "Params: $CURRENT_LINE"
GROUP=$(echo "$CURRENT_LINE" | awk -F',' '{print $1}')
RANK=$(echo "$CURRENT_LINE" | awk -F',' '{print $2}')
METHOD=$(echo "$CURRENT_LINE" | awk -F',' '{print $3}')

# Prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=

cd $SLURM_SUBMIT_DIR

# Route PyTorch compilation cache to node local scratch ($TMPDIR)
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torch_inductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"

apptainer exec \
    --cleanenv \
    --bind $PROJECT \
    --env GMRITENSOR_USE_GPU=False \
    --env CPUS_PER_TASK=1 \
    --env MULTIPROCESSING_TASKS=$SLURM_CPUS_PER_TASK \
    --env OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1 \
    apptainer/app.sif \
    python code/replicability/run_replicability.py $2 $GROUP $RANK $METHOD $4
