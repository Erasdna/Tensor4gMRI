#!/bin/bash -l
# Fit a CP model of each group's data with random restarts, one rank at a time, into
# decompose/cp/rank_<r>_<group>/: keep the best restart (cp_rank_<r>/model.h5) and score how
# reproducibly the restarts converge (restarts/cp_rank_<r>/reproducibility.csv). A group's fit
# leaves out the subjects not in it, as if absent from the data (cp_rank_<r>/excluded_subjects.csv).
# Every group and rank is an independent chain of jobs, which ends with its figures
# (04_plot_decomp.sh; PLOT=no: none).
# DISTRIBUTE=yes: plan -> fit array -> collect, TASKS_PER_JOB restarts per array task
# (settings.sh); DISTRIBUTE=no: one process, RESTART_PROCS restarts in parallel.
# Rerun to resume (DISTRIBUTE=yes): restarts stored with the same settings are reused, those
# fitted with other settings are refitted and overwritten; OVERWRITE=yes (default) refits all.
# Requires: 00_find_files.sh (subject_info.csv), 01_preprocessing.sh (roi_signal.parquet).
# Usage: bash code/pipeline/02_decompose.sh "<groups>" ["<ranks>"]
#   groups: all REF iNPH M F
#   e.g. bash code/pipeline/02_decompose.sh "REF iNPH" "3 4"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri decompose options (gMRI docs/cli.md, "gmri decompose") ----------------
RANKS=${2:-$RANKS}                           # --ranks, or the optional 2nd argument
RESTARTS=${RESTARTS:-50}                     # --restarts: random initialisations per rank
STATISTIC=${STATISTIC:-median}               # --statistic: per-label value to decompose
MIN_TIMEPOINTS=${MIN_TIMEPOINTS:-3}          # --min-timepoints: drop subjects with fewer
                                             # (CP imputes the missing sessions of the others)
MAX_INVALID_FRACTION=${MAX_INVALID_FRACTION:-0.2}  # --max-invalid-fraction: worse scans count as missing
MAX_ITER=${MAX_ITER:-5000}                   # --max-iter: iterations per restart
TOLERANCE=${TOLERANCE:-1e-7}                 # --tolerance: reconstruction-error change
DISTRIBUTE=${DISTRIBUTE:-auto}               # yes: plan/array/collect; no: one process;
                                             # auto: yes if jobs are submitted
RESTART_PROCS=${RESTART_PROCS:-}             # --restart-procs: parallel restarts per process;
                                             # empty: the CPUs of the fit job (LOCAL_PROCS here)
OVERWRITE=${OVERWRITE:-yes}                  # yes: plan --overwrite, refit every restart
PLOT=${PLOT:-yes}                            # yes: plot each fit (04_plot_decomp.sh) once collected

# --- SLURM options (sbatch) of the jobs this stage submits ----------------------
SMALL_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=4G --time=00:30:00)   # plan, collect
FIT_JOB=(--partition=normal --cpus-per-task=4 --mem-per-cpu=2G --time=02:00:00)     # fit task (TASKS_PER_JOB restarts)
SINGLE_JOB=(--partition=normal --cpus-per-task=16 --mem-per-cpu=2G --time=06:00:00) # DISTRIBUTE=no

require_each group "${1:-}" "${SUBJECT_GROUPS[@]}"
require_ranks "$RANKS"
require_each DISTRIBUTE "$DISTRIBUTE" auto yes no
require_each OVERWRITE "$OVERWRITE" yes no
require_each PLOT "$PLOT" yes no
require_data
if [ "$PLOT" = "yes" ]; then require_reference; fi
if [ "$DISTRIBUTE" = "auto" ]; then
    DISTRIBUTE=no
    if submits; then DISTRIBUTE=yes; fi
fi

# Submit the jobs of the current group's (use_group) fit of <group> <rank>, then its figures.
decompose() {
    local out done_id
    out=$(model_dir "$1" "$2")
    local input=(--input "$(signal_table)" --ranks "$2" "${FIT_FLAGS[@]}")
    if [ "$DISTRIBUTE" = "no" ]; then
        done_id=$(submit "$(job_name decompose "$1" "$2")" "${SINGLE_JOB[@]}" -- \
            gmri decompose run --output-dir "$out" "${input[@]}")
        echo "decompose $1 rank $2: single process $done_id"
    else
        decompose_distributed "$@"
    fi
    if [ "$PLOT" = "yes" ]; then
        DEPENDENCY="afterany:$done_id" bash "$PIPELINE_DIR/04_plot_decomp.sh" "$1" "$2"
    fi
}

# Submit the plan, fit array and collect of <group> <rank>; set done_id to the collect job.
decompose_distributed() {
    local out plan_id fit_id
    out=$(model_dir "$1" "$2")
    local input=(--input "$(signal_table)" --ranks "$2" "${FIT_FLAGS[@]}")
    plan_array "$RESTARTS" 1
    local plan=(gmri decompose plan --output-dir "$out" "${input[@]}" --tasks-per-job "$FITS_PER_TASK")
    add_switch plan --overwrite "$OVERWRITE"
    plan_id=$(submit "$(job_name decompose_plan "$1" "$2")" "${SMALL_JOB[@]}" -- \
        --expect-jobs "$JOBS" "${plan[@]}")
    fit_id=$(submit "$(job_name decompose_fit "$1" "$2")" --array="$(rank_array 0)" \
        --dependency="afterany:$plan_id" "${FIT_JOB[@]}" -- \
        gmri decompose run --output-dir "$out" --job "{task}")
    # afterany: collect also runs when fit tasks failed; gmri collect names the missing restarts.
    done_id=$(submit "$(job_name decompose_collect "$1" "$2")" --dependency="afterany:$fit_id" \
        "${SMALL_JOB[@]}" -- gmri decompose collect --output-dir "$out")
    echo "decompose $1 rank $2: plan $plan_id, fit array $fit_id ($JOBS tasks), collect $done_id"
}

fit_cpus=$(job_cpus "${SINGLE_JOB[@]}")
if [ "$DISTRIBUTE" = "yes" ]; then fit_cpus=$(job_cpus "${FIT_JOB[@]}"); fi
RESTART_PROCS=$(local_procs "${RESTART_PROCS:-$fit_cpus}")
for group in $1; do
    use_group "$group"
    fit_flags
    for rank in $RANKS; do decompose "$group" "$rank"; done
done
