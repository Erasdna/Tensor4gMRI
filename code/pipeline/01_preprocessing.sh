#!/bin/bash -l
# Images -> per-label signal (gmri preprocess, T1w: the post/baseline intensity ratio, its median
# and mean per label of the merged segmentation) in preprocess/data/roi_signal.parquet. Then a job
# writes the table that 02-04 read, roi_signal_min<MIN_VOXELS>voxels.parquet (and .csv), cleaned in
# three steps (code/preprocessing/clean_roi_signal.py): remove every subject's ROI with fewer
# than MIN_VOXELS voxels (settings.sh, default 100), then the rows with NaN values, then, for
# every subject, each label some subject has no row of left. Every label left is in every
# subject; a scan's removed NaN row is imputed by CP and does not count toward
# MAX_INVALID_FRACTION. The removed entries, with the reason, are listed in
# removed_rois_min<N>voxels.csv, the labels removed for every subject in
# removed_labels_min<N>voxels.csv. A second job writes each subject's brain volumes (ventricles,
# SAS, parenchyma, cortical grey matter; mL and fraction of the total segmented volume) from the
# uncleaned roi_signal.parquet to volumes.csv and adds them to subject_info.csv, where
# 05_subject_stats.sh tests them (code/preprocessing/subject_volumes.py); rerunning 00 rewrites
# subject_info.csv without them, until 01 (or PREPROCESS=no) runs again.
# Requires: 00_find_files.sh (manifest.csv).
# Usage: bash code/pipeline/01_preprocessing.sh
#   PREPROCESS=no MIN_VOXELS=50 bash code/pipeline/01_preprocessing.sh   # only clean again

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri preprocess options (gMRI docs/cli.md, "gmri preprocess") ---------------
N_PROCS=${N_PROCS:-8}          # --n-procs: images read in parallel
PREPROCESS=${PREPROCESS:-yes}  # no: only clean the existing roi_signal.parquet (e.g. another
                               # MIN_VOXELS)

# --- SLURM options (sbatch) -----------------------------------------------------
PREPROCESS_JOB=(--partition=normal --cpus-per-task=8 --mem-per-cpu=4G --time=03:00:00)
CLEAN_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=4G --time=00:30:00)  # also volumes

require_each PREPROCESS "$PREPROCESS" yes no
data="$RESULTS_DIR/preprocess/data"
manifest="$RESULTS_DIR/manifest.csv"
if [ "$PREPROCESS" = "yes" ] && [ ! -f "$manifest" ]; then
    echo "No $manifest: run code/pipeline/00_find_files.sh first" >&2
    exit 1
fi
if [ ! -f "$SUBJECT_INFO" ]; then
    echo "No $SUBJECT_INFO: run code/pipeline/00_find_files.sh first" >&2
    exit 1
fi
if [ "$PREPROCESS" = "no" ] && [ ! -f "$data/roi_signal.parquet" ]; then
    echo "No $data/roi_signal.parquet to clean: run with PREPROCESS=yes first" >&2
    exit 1
fi

dependency=()
if [ "$PREPROCESS" = "yes" ]; then
    cmd=(gmri preprocess --manifest "$manifest" --output-dir "$RESULTS_DIR/preprocess"
        --input-type T1w --n-procs "$(local_procs "$N_PROCS")")
    id=$(submit preprocess "${PREPROCESS_JOB[@]}" -- "${cmd[@]}")
    echo "preprocess: job $id, data in $data"
    dependency=(--dependency="afterok:$id")
fi
id=$(submit clean_roi_signal "${dependency[@]}" "${CLEAN_JOB[@]}" -- \
    python code/preprocessing/clean_roi_signal.py --input "$data/roi_signal.parquet" \
    --min-voxels "$MIN_VOXELS" --output "$(signal_table)" \
    --report "$data/removed_rois_min${MIN_VOXELS}voxels.csv" \
    --label-report "$data/removed_labels_min${MIN_VOXELS}voxels.csv")
echo "clean_roi_signal: job $id, $(signal_table)"
id=$(submit subject_volumes "${dependency[@]}" "${CLEAN_JOB[@]}" -- \
    python code/preprocessing/subject_volumes.py --input "$data/roi_signal.parquet" \
    --subject-info "$SUBJECT_INFO" --output "$RESULTS_DIR/volumes.csv")
echo "subject_volumes: job $id, $RESULTS_DIR/volumes.csv (and $SUBJECT_INFO)"
