#!/bin/bash -l

SCRIPT_PATH="code/replicability/scripts"
SAVE_PATH="$BASE_DATA_FOLDER/replicability"
mkdir -p $SAVE_PATH
PARAM_FILE="$SAVE_PATH/config.csv"

# Make config
uv run python "$SCRIPT_PATH/prepare_job_array.py" $PARAM_FILE
N_JOBS=$(wc -l < "$PARAM_FILE")

# Launch job array
REPLICABILITY_JOBS=$(sbatch --acount $SLURM_ACCOUNT --parsable --array=1-$N_JOBS%60 $SCRIPT_PATH/sbatch_replicability.sh $PARAM_FILE "data/data.csv" $SAVE_PATH)
sbatch --acount $SLURM_ACCOUNT --dependency=afterok:$REPLICABILITY_JOBS $SCRIPT_PATH/postprocess_replicability.sh $SAVE_PATH "data/data.csv"
