#!/bin/bash -l

source .env

uv run code/data_preparation/prepare_input_files.py \
    $RAW_TRACER_DATA_FOLDER \
    $SUBJECT_LABELS \
    "$BASE_DATA_FOLDER/run_files.csv"

echo $SBATCH
if [ "$SBATCH" = "true" ]; then
    echo $SBATCH
    sbatch --account $SLURM_ACCOUNT code/data_preparation/scripts/sbatch_preproc.sh \
        "$BASE_DATA_FOLDER/run_files.csv" \
        $SUBJECT_LABELS \
        "$BASE_DATA_FOLDER/data.csv"
else
    sh code/data_preparation/scripts/sbatch_preproc.sh \
        "$BASE_DATA_FOLDER/run_files.csv" \
        $SUBJECT_LABELS \
        "$BASE_DATA_FOLDER/data.csv"
fi
