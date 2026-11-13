#!/bin/bash -l
# One pipeline command on a compute node; submitted by `submit` in code/pipeline/env.sh.
# Usage: job.sh [--expect-jobs N] <gmri|python> <args...>
# "{task}" in <args> is replaced by $SLURM_ARRAY_TASK_ID. With --expect-jobs the
# command is a `gmri ... plan`, and the job fails unless the plan has N jobs.
# Also run directly by `run_local` in env.sh where there is no SLURM.

set -euo pipefail
echo "job.sh: start $(date -Iseconds) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-} task=${SLURM_ARRAY_TASK_ID:-}"
trap 'status=$?; echo "job.sh: end $(date -Iseconds) exit=$status"' EXIT
source "$PROJECT_DIR/code/scripts/common.sh"
limit_threads "${SLURM_CPUS_PER_TASK:-1}"  # every library, not one thread per node core

expected=""
if [ "$1" = "--expect-jobs" ]; then
    expected=$2
    shift 2
fi
args=("${@//\{task\}/${SLURM_ARRAY_TASK_ID:-}}")
echo "+ ${args[*]}"

if [ -z "$expected" ]; then
    run_tool "${args[@]}"
else
    planned=$(run_tool "${args[@]}" | tail -n1)
    if [ "$planned" != "$expected" ]; then
        echo "The plan has $planned jobs but $expected array tasks were submitted" >&2
        exit 1
    fi
    echo "Plan has $planned jobs"
fi
