#!/bin/bash -l
# Find each subject's scans at 2, 4, 8 and 24 h after the first one and merge its SAS and
# parenchyma segmentations (code/preprocessing/find_files.py), then write the subject table
# (subject_info.csv: Group, Sex and Stratum named) and the manifest of gmri preprocess.
# Subjects marked exclude in $SUBJECT_LABELS are left out. Runs here, not as a job: the next
# stage reads its output straight away.
# Requires: $RAW_TRACER_DATA_FOLDER, $SUBJECT_LABELS (.env).
# Usage: bash code/pipeline/00_find_files.sh

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

: "${RAW_TRACER_DATA_FOLDER:?set RAW_TRACER_DATA_FOLDER in .env}"
: "${SUBJECT_LABELS:?set SUBJECT_LABELS in .env}"

# Log and run <tool> <args...> here.
run_here() {
    log_cmd "$@"
    run_tool "$@"
}

run_here python code/preprocessing/find_files.py "$RAW_TRACER_DATA_FOLDER" "$SUBJECT_LABELS" "$RESULTS_DIR"
run_here python code/preprocessing/make_subject_info.py "$SUBJECT_LABELS" \
    "$RESULTS_DIR/files_to_process.csv" "$RESULTS_DIR"
run_here python code/preprocessing/make_manifest.py "$RESULTS_DIR/files_to_process.csv" "$RESULTS_DIR"
