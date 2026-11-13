#!/bin/bash -l
# Replicability of each group's CP fits: fit random halves of its subjects (balanced on the
# variable that varies within the group, STRATIFY_BY) with random restarts, score how well the
# two halves' factors match (factor match score) per split and rank, and plot it against rank.
# The group's fit leaves out the subjects not in it, as in 02_decompose.sh.
# DISTRIBUTE=yes: one plan per group, a fit array per rank (TASKS_PER_JOB restarts per array
# task, settings.sh), collect, plot; DISTRIBUTE=no: one process, N_PROCS fits in parallel.
# Rerun to resume (DISTRIBUTE=yes): restarts stored with the same settings are reused, those
# fitted with other settings are refitted and overwritten; OVERWRITE=yes refits all.
# Writes replicability/<group>/replicability.csv and plots/replicability/<group>/
# (fms_by_rank.{pdf,png}, fms_summary.csv).
# Requires: 00_find_files.sh (subject_info.csv), 01_preprocessing.sh (roi_signal.parquet).
# Usage: bash code/pipeline/03_replicability.sh "<groups>" ["<ranks>"]
#   groups: all REF iNPH M F
#   e.g. bash code/pipeline/03_replicability.sh "all REF iNPH M F" "2 3 4 5 6"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# --- gmri replicability options (gMRI docs/cli.md, "gmri replicability") ----------
REPEATS=${REPEATS:-10}             # --repeats: random splits into halves
SEED=${SEED:-0}                    # --seed: seed of the splits, shared by every rank
STRATIFY_BY=${STRATIFY_BY:-}       # --stratify-by: subject-info column to balance the halves on;
                                   # empty: the group's (Stratum, i.e. Group and Sex, for all;
                                   # Sex for REF and iNPH; Group for M and F)
N_PROCS=${N_PROCS:-}               # --n-procs: parallel fits per process; empty: the CPUs of
                                   # the fit job (LOCAL_PROCS here)
# Fit options, as in 02_decompose.sh:
RANKS=${2:-$RANKS}                 # --ranks, or the optional 2nd argument
RESTARTS=${RESTARTS:-50}           # --restarts: per half and rank
STATISTIC=${STATISTIC:-median}     # --statistic
MIN_TIMEPOINTS=${MIN_TIMEPOINTS:-3}  # --min-timepoints
MAX_INVALID_FRACTION=${MAX_INVALID_FRACTION:-0.2}  # --max-invalid-fraction
MAX_ITER=${MAX_ITER:-5000}         # --max-iter
TOLERANCE=${TOLERANCE:-1e-7}       # --tolerance
DISTRIBUTE=${DISTRIBUTE:-auto}     # yes: plan/arrays/collect; no: one process; auto: yes if
                                   # jobs are submitted
OVERWRITE=${OVERWRITE:-no}         # yes: plan --overwrite, refit every restart

# --- SLURM options (sbatch) of the jobs this stage submits ----------------------
SMALL_JOB=(--partition=normal --cpus-per-task=1 --mem-per-cpu=4G --time=00:30:00)   # plan, collect, plot
FIT_JOB=(--partition=normal --cpus-per-task=4 --mem-per-cpu=2G --time=02:00:00)     # fit task (TASKS_PER_JOB restarts)
SINGLE_JOB=(--partition=normal --cpus-per-task=16 --mem-per-cpu=2G --time=24:00:00) # DISTRIBUTE=no

require_each group "${1:-}" "${SUBJECT_GROUPS[@]}"
require_ranks "$RANKS"
require_each DISTRIBUTE "$DISTRIBUTE" auto yes no
require_each OVERWRITE "$OVERWRITE" yes no
require_data
if [ "$DISTRIBUTE" = "auto" ]; then
    DISTRIBUTE=no
    if submits; then DISTRIBUTE=yes; fi
fi
read -r -a ranks <<< "$RANKS"
fits_per_rank=$(( 2 * REPEATS * RESTARTS ))
fit_cpus=$(job_cpus "${SINGLE_JOB[@]}")
if [ "$DISTRIBUTE" = "yes" ]; then fit_cpus=$(job_cpus "${FIT_JOB[@]}"); fi
N_PROCS=$(local_procs "${N_PROCS:-$fit_cpus}")

# Submit the jobs of the current group's (use_group) replicability, then its plot.
replicability() {
    local group=$1 stratify=$STRATIFY_BY out done_id plan_id i fit_id fit_ids=()
    if [ -z "$stratify" ]; then stratify=$(group_columns "$group" stratify); fi
    out="$RESULTS_DIR/replicability/$group"
    fit_flags replicability
    local input=(--input "$(signal_table)" --engine halfhalf --repeats "$REPEATS" --seed "$SEED"
        --n-procs "$N_PROCS" --subject-info "$SUBJECT_INFO" --stratify-by "$stratify"
        --ranks "${ranks[@]}" "${FIT_FLAGS[@]}")
    if [ "$DISTRIBUTE" = "no" ]; then
        done_id=$(submit "$(job_name replicability "$group")" "${SINGLE_JOB[@]}" -- \
            gmri replicability run --output-dir "$out" "${input[@]}")
        echo "replicability $group: single process $done_id"
    else
        plan_array "$fits_per_rank" "${#ranks[@]}"
        local plan=(gmri replicability plan --output-dir "$out" "${input[@]}"
            --tasks-per-job "$FITS_PER_TASK")
        add_switch plan --overwrite "$OVERWRITE"
        plan_id=$(submit "$(job_name replicability_plan "$group")" "${SMALL_JOB[@]}" -- \
            --expect-jobs "$JOBS" "${plan[@]}")
        for i in "${!ranks[@]}"; do
            fit_id=$(submit "$(job_name replicability_fit "$group" "${ranks[i]}")" \
                --array="$(rank_array "$i")" --dependency="afterany:$plan_id" "${FIT_JOB[@]}" -- \
                gmri replicability run --output-dir "$out" --job "{task}")
            fit_ids+=("$fit_id")
        done
        # afterany: collect also runs when fit tasks failed; gmri collect names the missing restarts.
        done_id=$(submit "$(job_name replicability_collect "$group")" \
            --dependency="$(IFS=:; echo "afterany:${fit_ids[*]}")" "${SMALL_JOB[@]}" -- \
            gmri replicability collect --output-dir "$out")
        echo "replicability $group: plan $plan_id, fit arrays ${fit_ids[*]} ($JOBS_PER_RANK tasks per rank), collect $done_id"
    fi
    local plot_id
    plot_id=$(submit "$(job_name replicability_plot "$group")" --dependency="afterany:$done_id" \
        "${SMALL_JOB[@]}" -- python code/analysis/plot_replicability.py "$out/replicability.csv" \
        "$RESULTS_DIR/plots/replicability/$group")
    echo "replicability $group: plot $plot_id"
}

for group in $1; do
    use_group "$group"
    replicability "$group"
done
