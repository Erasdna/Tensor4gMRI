#!/bin/bash
# Values shared by several stages, and cluster-wide SLURM settings. The gmri options and sbatch
# options of each stage are at the top of its script. Every value can be overridden from the
# environment, e.g. `ARRAY_LIMIT=50 bash code/pipeline/03_decompose.sh ...`.

# --- Study ---------------------------------------------------------------------
# Subject groups a model can be fitted on (code/tensor4gmri/config.py): every subject, one
# diagnosis or one sex. Each is its own fit, of the subjects in the group only.
SUBJECT_GROUPS=(all REF iNPH M F)
RANKS=${RANKS:-"2 3 4 5 6"}          # default ranks of 02-05 (each also takes them as an argument)
# A subject's ROI with fewer voxels is removed (every time point), then any NaN value, then every
# label some subject has no row of left: 01 writes the cleaned table that 02-04 read
# (roi_signal_min<MIN_VOXELS>voxels.parquet). 0: no ROI is too small.
MIN_VOXELS=${MIN_VOXELS:-100}
# Spatial maps of 04_plot_decomp.sh: drawn on REFERENCE_SEGMENTATION (a merged segmentation, CSF
# labels + 10000) over REFERENCE_BACKGROUND (an image on its grid) at SLICES. .env may name them,
# e.g. a cropped plotting subject (code/adapt_example_subject.py); empty: the segmentation and
# pre-injection T1w of REFERENCE_SUBJECT (default $SUB_REF_MEDIAN of .env) from 00_find_files.sh.
# A REFERENCE_SEGMENTATION without a REFERENCE_BACKGROUND is drawn on its own mask.
REFERENCE_SUBJECT=${REFERENCE_SUBJECT-${SUB_REF_MEDIAN:-}}
REFERENCE_SEGMENTATION=${REFERENCE_SEGMENTATION:-}
REFERENCE_BACKGROUND=${REFERENCE_BACKGROUND:-}
# "I J K": gmri's --slices, voxel indices of the background reoriented to RAS (nibabel's
# as_closest_canonical; the mode grid shows I, sagittal). Empty: the centre of mass.
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
