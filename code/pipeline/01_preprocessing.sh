#!/bin/bash -l
# Images -> per-label signal (gmri preprocess, T1w: the post/baseline intensity ratio, its median
# and mean per label of the merged segmentation) in preprocess/data/roi_signal.parquet.
# Requires: 00_find_files.sh (manifest.csv).
# Usage: bash code/pipeline/01_preprocessing.sh

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri preprocess options (gMRI docs/cli.md, "gmri preprocess") ---------------
N_PROCS=${N_PROCS:-8}        # --n-procs: images read in parallel

# --- SLURM options (sbatch) -----------------------------------------------------
PREPROCESS_JOB=(--partition=normal --cpus-per-task=8 --mem-per-cpu=4G --time=03:00:00)

manifest="$RESULTS_DIR/manifest.csv"
if [ ! -f "$manifest" ]; then
    echo "No $manifest: run code/pipeline/00_find_files.sh first" >&2
    exit 1
fi
cmd=(gmri preprocess --manifest "$manifest" --output-dir "$RESULTS_DIR/preprocess"
    --input-type T1w --n-procs "$(local_procs "$N_PROCS")")
id=$(submit preprocess "${PREPROCESS_JOB[@]}" -- "${cmd[@]}")
echo "preprocess: job $id, data in $RESULTS_DIR/preprocess/data"
