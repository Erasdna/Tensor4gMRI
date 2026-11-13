#!/bin/bash -l
# Shared setup for the pipeline: the environment file and the container wrappers. Source it from
# any directory:
#   source code/scripts/common.sh

echo "$(hostname)"
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then source "$ENV_FILE";
else
    echo "Cannot find $ENV_FILE. Please make sure it exists and is readable."
    # In an interactive shell, exit would close the terminal.
    if [[ $- == *i* ]]; then return 1; else exit 1; fi
fi

# Network storage often does not support HDF5 file locks.
export HDF5_USE_FILE_LOCKING=FALSE
# Open files: as many as allowed (the hard limit). gmri's worker pools share each fit's data
# through a file descriptor, which a soft limit of 1024 (common on desktops) runs out of.
ulimit -Sn "$(ulimit -Hn)" 2>/dev/null || true

# Thread counts of the libraries the pipeline's commands use: torch and threadpoolctl in gmri
# (CPUS_PER_TASK), OpenMP, BLAS and numexpr. Each otherwise starts one thread per core of the
# machine.
THREAD_VARIABLES=(CPUS_PER_TASK OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
    VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS NUMEXPR_MAX_THREADS)

# Limit every library of THREAD_VARIABLES to <n> threads (container_exec passes them on).
limit_threads() {
    local variable
    for variable in "${THREAD_VARIABLES[@]}"; do export "$variable=$1"; done
}

# Run a command inside the container with the repo's code on PYTHONPATH, the raw and the
# processed data and the subject labels visible.
container_exec() {
    local thread_args=() bind_args=() variable
    if [ -n "${SUBJECT_LABELS:-}" ]; then bind_args+=(--bind "$(dirname "$SUBJECT_LABELS")"); fi
    # Thread limits from limit_threads (job.sh, or use_local_procs in code/pipeline/env.sh).
    for variable in "${THREAD_VARIABLES[@]}"; do
        if [ -n "${!variable:-}" ]; then thread_args+=(--env "$variable=${!variable}"); fi
    done
    apptainer exec \
        --cleanenv \
        "${thread_args[@]}" \
        --bind "${RAW_TRACER_DATA_FOLDER:?RAW_TRACER_DATA_FOLDER is not set; see .env}" \
        --bind "${BASE_DATA_FOLDER:?BASE_DATA_FOLDER is not set; see .env}" \
        "${bind_args[@]}" \
        --bind "$PROJECT_DIR":/mnt/project \
        --env PYTHONPATH=/mnt/project/code \
        --env HDF5_USE_FILE_LOCKING=FALSE \
        "$CONTAINER" "$@"
}

# Run a repo script (path relative to the repo root) with the container's Python, by its path
# in the container (the repo is bound at /mnt/project), so it is found from any directory.
container_python() {
    local script=$1
    shift
    container_exec python "/mnt/project/$script" "$@"
}

# Run the gMRItensor CLI (`gmri`) inside the container.
container_gmri() {
    container_exec gmri "$@"
}

# Dispatch `gmri <args>` or `python <script> <args>` to the container.
run_tool() {
    local tool=$1
    shift
    case "$tool" in
        gmri) container_gmri "$@" ;;
        python) container_python "$@" ;;
        *) echo "run_tool: unknown tool '$tool' (expected gmri or python)" >&2; return 1 ;;
    esac
}
