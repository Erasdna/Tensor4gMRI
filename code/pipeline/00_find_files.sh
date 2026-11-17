#!/bin/bash -l
# Write the subject table of the study's subjects, those of $SUBJECT_LABELS not marked exclude
# and diagnosed REF or iNPH (subject_info.csv: Group, Sex and Stratum named; the others are listed
# as left out). Then, for these subjects only, find the scans at 2, 4, 8 and 24 h after the first
# one, merge the SAS and parenchyma segmentations and copy the baseline T1w
# (code/preprocessing/find_files.py), and write the manifest of gmri preprocess. Runs here, not as
# a job: the next stage reads its output straight away.
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

run_here python code/preprocessing/make_subject_info.py "$SUBJECT_LABELS" "$RESULTS_DIR"
run_here python code/preprocessing/find_files.py "$RAW_TRACER_DATA_FOLDER" \
    "$RESULTS_DIR/subject_info.csv" "$RESULTS_DIR"
run_here python code/preprocessing/make_manifest.py "$RESULTS_DIR/files_to_process.csv" "$RESULTS_DIR"
