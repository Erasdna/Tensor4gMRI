#!/bin/bash
# Shared setup for the pipeline stages; sourced by every code/pipeline/*.sh stage.

# Sourced into an interactive shell (by hand), these options would make any failing command
# close the terminal; the stages themselves run non-interactively and always get them.
if [[ $- != *i* ]]; then set -euo pipefail; fi
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../scripts/common.sh"
: "${PROJECT_DIR:?set PROJECT_DIR in .env (the repository root)}"
: "${CONTAINER:?set CONTAINER in .env (e.g. \$PROJECT_DIR/apptainer/app.sif)}"
export RESULTS_DIR=${RESULTS_DIR:-${BASE_DATA_FOLDER:?set BASE_DATA_FOLDER in .env}/tensor}
export SUBJECT_INFO="$RESULTS_DIR/subject_info.csv"   # written by 00_find_files.sh
source "$PIPELINE_DIR/settings.sh"
JOB_SCRIPT="$PIPELINE_DIR/sbatch/job.sh"
LOG_DIR=$RESULTS_DIR

# Exit unless <list> is non-empty and each of its words is one of <allowed...>.
require_each() {
    local what=$1 list=$2 value
    shift 2
    if [ -z "$list" ]; then
        echo "No $what given (expected some of: $*)" >&2
        exit 1
    fi
    for value in $list; do
        case " $* " in
            *" $value "*) ;;
            *) echo "Unknown $what '$value' (expected some of: $*)" >&2; exit 1 ;;
        esac
    done
}

require_each RUN_LOCALLY "$RUN_LOCALLY" yes no
if ! [[ "$LOCAL_PROCS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid LOCAL_PROCS '$LOCAL_PROCS' (expected a positive integer)" >&2
    exit 1
fi
if [ "$RUN_LOCALLY" = "no" ] && command -v sbatch > /dev/null && [ -z "${SLURM_ACCOUNT:-}" ]; then
    echo "Set SLURM_ACCOUNT in .env to submit jobs (or RUN_LOCALLY=yes to run here)" >&2
    exit 1
fi

# Exit unless <list> is non-empty and each of its words is a positive integer.
require_ranks() {
    local rank
    if [ -z "$1" ]; then echo "No ranks given" >&2; exit 1; fi
    for rank in $1; do
        if ! [[ "$rank" =~ ^[1-9][0-9]*$ ]]; then
            echo "Unknown rank '$rank' (expected positive integers)" >&2
            exit 1
        fi
    done
}

# Call <function> once for every combination of the words in the list arguments,
# followed by the arguments after `--` unchanged:
#   for_each_combination f "all REF" "3 4"  ->  f all 3; f all 4; f REF 3; f REF 4
#   for_each_combination f "all REF" -- "a b"  ->  f all "a b"; f REF "a b"
for_each_combination() {
    local function=$1 lists=()
    shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do lists+=("$1"); shift; done
    if [ $# -gt 0 ]; then shift; fi
    _combine "$function" "" "${#lists[@]}" "${lists[@]}" "$@"
}

# _combine <function> <chosen> <n_lists> <lists...> <constants...>
_combine() {
    local function=$1 chosen=$2 n_lists=$3 value
    shift 3
    if [ "$n_lists" -eq 0 ]; then
        # shellcheck disable=SC2086  # $chosen holds one word per list
        "$function" $chosen "$@"
        return
    fi
    local list=$1
    shift
    for value in $list; do
        _combine "$function" "$chosen $value" $(( n_lists - 1 )) "$@"
    done
}

# Name of the <stage> job of <group> [<rank>]: <stage>[_rank<r>]_<group>.
job_name() {
    echo "$1${3:+_rank$3}_$2"
}

# The per-label table gmri preprocess writes, which decompose and replicability read.
signal_table() {
    echo "$RESULTS_DIR/preprocess/data/roi_signal.parquet"
}

# Output directory of the 03 fits of <group> <rank>; the model is <dir>/cp_rank_<rank>/.
model_dir() {
    echo "$RESULTS_DIR/decompose/cp/rank_$2_$1"
}

# Append <flag> to the array named <array> if <value> is "yes".
add_switch() {
    local -n array=$1
    case "$3" in
        yes) array+=("$2") ;;
        no) ;;
        *) echo "$2: expected yes or no, got '$3'" >&2; exit 1 ;;
    esac
}

# Append <flag> <value...> to the array named <array> unless <value> is empty.
add_optional() {
    local -n target=$1
    local flag=$2
    shift 2
    if [ -n "$*" ]; then
        # shellcheck disable=SC2206  # one word per value
        target+=("$flag" $*)
    fi
}

# Code version: code/VERSION if present, else git.
code_version() {
    if [ -f "$PROJECT_DIR/code/VERSION" ]; then
        paste -sd' ' "$PROJECT_DIR/code/VERSION"
    else
        echo "repo=$(git -C "$PROJECT_DIR" describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)"
    fi
}

# The arguments as one copy-pasteable command line, quoting only where needed.
command_line() {
    local arg words=()
    for arg in "$@"; do
        if [[ "$arg" =~ ^[A-Za-z0-9_./=:,+@%{}-]+$ ]]; then words+=("$arg"); else words+=("'${arg//\'/\'\\\'\'}'"); fi
    done
    echo "${words[*]}"
}

# Append the command to $LOG_DIR/commands.log (time, code version, command) and print it
# (to stderr, so a command's output can be captured).
log_cmd() {
    mkdir -p "$LOG_DIR"
    printf '%s %s %s\n' "$(date -Iseconds)" "$(code_version)" "$(command_line "$@")" >> "$LOG_DIR/commands.log"
    echo "+ $(command_line "$@")" >&2
}

# Submit one pipeline command as a SLURM job, log the full sbatch command and print the job id.
# Logs go to slurm/<name>_<job id>.out, or slurm/<name>_<array id>_<index>.out for an --array.
# Without sbatch, or with RUN_LOCALLY=yes, run the job here instead (see run_local).
# Usage: submit <name> [sbatch options...] -- [--expect-jobs N] <gmri|python> <args...>
submit() {
    local name=$1
    shift
    local options=()
    while [ "$1" != "--" ]; do options+=("$1"); shift; done
    shift
    if ! submits; then
        run_local "$name" "${options[@]}" -- "$@"
        return
    fi
    mkdir -p "$PROJECT_DIR/slurm"
    local log_id=%j
    if [[ " ${options[*]} " == *" --array="* ]]; then log_id=%A_%a; fi
    local cmd=(sbatch --parsable --export=ALL --account="$SLURM_ACCOUNT" --nodes=1 --ntasks=1
        --job-name="$name" --output="$PROJECT_DIR/slurm/${name}_${log_id}.out" "${options[@]}"
        "$JOB_SCRIPT" "$@")
    log_cmd "${cmd[@]}"
    "${cmd[@]}"
}

# Limit a command run outside SLURM to $LOCAL_PROCS threads in every library (limit_threads).
use_local_procs() {
    export SLURM_CPUS_PER_TASK=$LOCAL_PROCS
    limit_threads "$LOCAL_PROCS"
    echo "local run: LOCAL_PROCS=$LOCAL_PROCS threads" >&2
}

# <requested> parallel processes of a command (--n-procs, --restart-procs): as requested
# where it is submitted, at most LOCAL_PROCS where it runs here.
local_procs() {
    if submits || [ "$1" -le "$LOCAL_PROCS" ]; then
        echo "$1"
        return
    fi
    echo "running here: $1 processes lowered to LOCAL_PROCS=$LOCAL_PROCS" >&2
    echo "$LOCAL_PROCS"
}

# Run a `submit` job on this machine: each task of its --array in turn (with
# SLURM_ARRAY_TASK_ID set), output on stderr, and print "local" as its job id.
# Threads come from LOCAL_PROCS, not --cpus-per-task; the other sbatch options, dependencies
# included, are ignored since jobs run in submission order.
# Usage: run_local <name> [sbatch options...] -- [--expect-jobs N] <gmri|python> <args...>
run_local() {
    local name=$1 option tasks=("") task first last
    shift
    while [ "$1" != "--" ]; do
        option=$1
        shift
        case "$option" in
            --array=*)
                option=${option#--array=}
                option=${option%%\%*}
                first=${option%-*}
                last=${option#*-}
                mapfile -t tasks <<< "$(seq "$first" "$last")"
                ;;
        esac
    done
    shift
    use_local_procs
    log_cmd bash "$JOB_SCRIPT" "$@"
    for task in "${tasks[@]}"; do
        echo "$name: running locally${task:+ (task $task)}" >&2
        SLURM_ARRAY_TASK_ID=$task bash "$JOB_SCRIPT" "$@" >&2 || return 1
    done
    echo local
}

# Whether jobs are submitted: where sbatch exists, unless RUN_LOCALLY=yes asks to run here.
submits() {
    [ "$RUN_LOCALLY" = "no" ] && command -v sbatch > /dev/null
}

# Run <tool> <args...> as a SLURM job with the stage's SMALL_JOB options (or those in the
# array named by --sbatch), after $DEPENDENCY (e.g. afterany:123) if set, and set JOB_ID to
# its id. Without sbatch, or with RUN_LOCALLY=yes, run here instead (with LOCAL_PROCS
# threads) and set JOB_ID to "".
# Usage: run_or_submit <name> [--sbatch <array name>] <tool> <args...>
run_or_submit() {
    local name=$1 job_options=SMALL_JOB
    shift
    if [ "$1" = "--sbatch" ]; then job_options=$2; shift 2; fi
    local -n _job=$job_options
    JOB_ID=""
    if submits; then
        local dependency=(${DEPENDENCY:+"--dependency=$DEPENDENCY"})
        JOB_ID=$(submit "$name" "${dependency[@]}" "${_job[@]}" -- "$@")
        echo "$name: job $JOB_ID"
    else
        use_local_procs
        log_cmd "$@"
        run_tool "$@"
    fi
}

# --- Groups and fits (02-05) -------------------------------------------------------

# Exit unless 00 and 01 have written the subject table and the preprocessed data.
require_data() {
    local file
    for file in "$SUBJECT_INFO" "$(signal_table)"; do
        if [ ! -f "$file" ]; then
            echo "No $file: run code/pipeline/00_find_files.sh and 01_preprocessing.sh first" >&2
            exit 1
        fi
    done
}

# Set MODEL_SUFFIX to <group> and EXCLUDE_SUBJECTS to the subjects of the data not in it
# (empty for all), so its fits leave them out as if absent from the data.
use_group() {
    local cmd=(python code/preprocessing/group_subjects.py --subject-info "$SUBJECT_INFO"
        --input "$(signal_table)" --group "$1")
    MODEL_SUFFIX=$1
    log_cmd "${cmd[@]}"
    EXCLUDE_SUBJECTS=$(run_tool "${cmd[@]}")  # plain assignment, so a failure stops the stage
}

# The --cpus-per-task of the sbatch options <options...> (1 if none).
job_cpus() {
    local option
    for option in "$@"; do
        case "$option" in --cpus-per-task=*) echo "${option#*=}"; return ;; esac
    done
    echo 1
}

# FIT_FLAGS of the CP fits of the current group (use_group), from the fit options at the top of
# the calling stage (02_decompose.sh, 03_replicability.sh). With "replicability", no
# --restart-procs: gmri replicability spreads its fits with --n-procs.
fit_flags() {
    FIT_FLAGS=(--method cp --statistic "$STATISTIC"
        --min-timepoints "$MIN_TIMEPOINTS" --max-invalid-fraction "$MAX_INVALID_FRACTION"
        --restarts "$RESTARTS" --max-iter "$MAX_ITER" --tolerance "$TOLERANCE"
        --non-negative-modes 0,1,2 --store restarts)
    if [ "${1:-}" != "replicability" ]; then FIT_FLAGS+=(--restart-procs "$RESTART_PROCS"); fi
    add_optional FIT_FLAGS --exclude-subjects "$EXCLUDE_SUBJECTS"
}

# Set FITS_PER_TASK, JOBS_PER_RANK and JOBS (<n_ranks> * JOBS_PER_RANK), the array tasks for
# <fits_per_rank> restart fits of each of <n_ranks> ranks. FITS_PER_TASK starts at
# TASKS_PER_JOB and is raised to a divisor of <fits_per_rank>, so no task spans two ranks,
# and so that JOBS stays within $MAX_ARRAY_SIZE.
plan_array() {
    local fits=$1 n_ranks=$2 tasks=$TASKS_PER_JOB
    local needed=$(( (fits * n_ranks + MAX_ARRAY_SIZE - 1) / MAX_ARRAY_SIZE ))
    if [ "$tasks" -lt "$needed" ]; then tasks=$needed; fi
    if [ "$tasks" -gt "$fits" ]; then tasks=$fits; fi
    while [ $(( fits % tasks )) -ne 0 ]; do tasks=$(( tasks + 1 )); done
    if [ "$tasks" -ne "$TASKS_PER_JOB" ]; then
        echo "$fits fits per rank: TASKS_PER_JOB $TASKS_PER_JOB -> $tasks (divides the fits per rank, MAX_ARRAY_SIZE=$MAX_ARRAY_SIZE)" >&2
    fi
    FITS_PER_TASK=$tasks
    JOBS_PER_RANK=$(( fits / tasks ))
    JOBS=$(( n_ranks * JOBS_PER_RANK ))
}

# The --array range of the fit tasks of the <i>th rank (from 0); the plan runs rank by rank.
rank_array() {
    local first=$(( $1 * JOBS_PER_RANK ))
    echo "$first-$(( first + JOBS_PER_RANK - 1 ))%$ARRAY_LIMIT"
}

# Print the subject-info columns of <group>: "stratify" (03) or "covariates" (04, 05; the first
# is the group variable of the figures).
group_columns() {
    run_tool python code/preprocessing/group_subjects.py --group "$1" --print "$2"
}

# Set REFERENCE_SEGMENTATION and REFERENCE_BACKGROUND (default: the T1w of REFERENCE_SUBJECT),
# the merged segmentation and image the spatial maps are drawn on; exit unless both exist.
require_reference() {
    if [ -z "$REFERENCE_SUBJECT" ]; then
        echo "Set REFERENCE_SUBJECT (settings.sh, default \$SUB_REF_MEDIAN of .env): the subject" \
            "whose segmentation the spatial maps are drawn on" >&2
        exit 1
    fi
    REFERENCE_SEGMENTATION="$RESULTS_DIR/$REFERENCE_SUBJECT/$REFERENCE_SUBJECT-segmentation.nii.gz"
    if [ ! -f "$REFERENCE_SEGMENTATION" ]; then
        echo "No $REFERENCE_SEGMENTATION: is REFERENCE_SUBJECT=$REFERENCE_SUBJECT among the" \
            "subjects of code/pipeline/00_find_files.sh?" >&2
        exit 1
    fi
    REFERENCE_BACKGROUND=${REFERENCE_BACKGROUND:-"$RESULTS_DIR/$REFERENCE_SUBJECT/$REFERENCE_SUBJECT-T1w.nii.gz"}
    if [ ! -f "$REFERENCE_BACKGROUND" ]; then
        echo "No REFERENCE_BACKGROUND $REFERENCE_BACKGROUND" >&2
        exit 1
    fi
}
