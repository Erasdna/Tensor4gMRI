#!/bin/bash
# Values shared by several stages, and cluster-wide SLURM settings. The gmri options and sbatch
# options of each stage are at the top of its script. Every value can be overridden from the
# environment, e.g. `ARRAY_LIMIT=50 bash code/pipeline/03_decompose.sh ...`.

# --- Study ---------------------------------------------------------------------
# Subject groups a model can be fitted on (code/tensor4gmri/config.py): every subject, one
# diagnosis or one sex. Each is its own fit, of the subjects in the group only.
SUBJECT_GROUPS=(all REF iNPH M F)
RANKS=${RANKS:-"2 3 4 5 6"}          # default ranks of 02-05 (each also takes them as an argument)
# Spatial maps of 04_plot_decomp.sh: drawn on the merged segmentation (00_find_files.sh) of
# REFERENCE_SUBJECT, default $SUB_REF_MEDIAN of .env, over REFERENCE_BACKGROUND, an image on the
# same grid (empty: that subject's pre-injection T1w, <subject>/<subject>-T1w.nii.gz from
# 00_find_files.sh), at SLICES, voxel indices I J K of the background reoriented to RAS (the mode
# grid shows I, sagittal; empty: centre of mass).
REFERENCE_SUBJECT=${REFERENCE_SUBJECT-${SUB_REF_MEDIAN:-}}
REFERENCE_BACKGROUND=${REFERENCE_BACKGROUND:-}
SLICES=${SLICES:-}

# --- SLURM (the sbatch options of each job are at the top of its stage) ------------
TASKS_PER_JOB=${TASKS_PER_JOB:-100}  # restarts fitted one after another by one array task
                                     # (raised to divide the fits per rank)
ARRAY_LIMIT=${ARRAY_LIMIT:-20}       # array tasks running at once, per rank's array
MAX_ARRAY_SIZE=${MAX_ARRAY_SIZE:-1000}

# --- Runs outside SLURM (this machine, or stages run on a login node) -------------
LOCAL_PROCS=${LOCAL_PROCS:-4}        # CPUs per command run here: caps its processes (--n-procs,
                                     # --restart-procs) and every library's threads; replaces --cpus-per-task
RUN_LOCALLY=${RUN_LOCALLY:-no}       # yes: run here even where sbatch exists
