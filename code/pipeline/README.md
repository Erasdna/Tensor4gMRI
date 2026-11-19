# Pipeline

Each stage is a script in this folder, run from the repository root. They wrap the `gmri` command
line of [gMRI-tensor](https://github.com/Erasdna/gMRI-tensor) (its `docs/cli.md` explains every
gmri flag) and run inside the Apptainer container `$CONTAINER`. All data is T1w: the signal is the
post/baseline intensity ratio per label of each subject's merged segmentation, and every model is a
non-negative CP decomposition.

## Setup

Build the container once (and again after `uv.lock` changes):

```bash
apptainer build apptainer/app.sif apptainer/app.def
```

`.env` in the repository root (not committed) sets, for the machine at hand:

| Variable | Meaning |
|---|---|
| `PROJECT_DIR` | The repository root. |
| `CONTAINER` | The container image, e.g. `$PROJECT_DIR/apptainer/app.sif`. |
| `BASE_DATA_FOLDER` | Results go to `$BASE_DATA_FOLDER/tensor` (`RESULTS_DIR`). |
| `RAW_TRACER_DATA_FOLDER` | One folder per subject with its normalized scans and `-sasparc`/`-refroi` segmentations. |
| `SUBJECT_LABELS` | The subject labels csv: `new_id`, `old_id`, `Group` (0 REF, 1 iNPH; other codes are left out), `Sex` (0 M, 1 F), `exclude`, `Age`, `Height`, `Weight`. |
| `SLURM_ACCOUNT` | Needed to submit jobs. |
| `SUB_REF_MEDIAN` | Default `REFERENCE_SUBJECT` of the spatial maps. |

## Groups

Every model is fitted on one group of subjects: `all`, a diagnosis (`REF`, `iNPH`) or a sex
(`M`, `F`). The data is preprocessed once; a group's fit leaves the other subjects out as if
they were absent from the data (`gmri --exclude-subjects`), so they shape neither the scaling nor
the fit. Its outputs carry the group in their path (`rank_<r>_<group>`). Within a group the
variable that varies is used where a group variable is needed (`code/tensor4gmri/config.py`):

| Group | Replicability halves balanced on | Primary categorical variables (stats), figure groups |
|---|---|---|
| `all` | `Stratum` (`Group` and `Sex`) | `Group`, `Sex` |
| `REF`, `iNPH` | `Sex` | `Sex` |
| `M`, `F` | `Group` | `Group` |

## Stages

| Script | What it does | Requires |
|---|---|---|
| `00_find_files.sh` | Writes `subject_info.csv`, the study's subjects: those of the labels not marked `exclude` and diagnosed REF or iNPH (Group 0 or 1), labels named, `Stratum` added; the others are listed as left out. For these subjects only, finds the scans closest to 2, 4, 8 and 24 h after the first, merges the SAS (labels + 10000) and parenchyma segmentations and copies the baseline (pre-injection) T1w into `<subject>/`, then writes `files_to_process.csv` and `manifest.csv`. Prints the subjects without scans and the number found per Group and Sex. Runs here, not as a job. | raw data, labels |
| `01_preprocessing.sh` | `gmri preprocess --input-type T1w`: `preprocess/data/roi_signal.parquet` (median and mean per label and scan). Then a job writes the table that 02–04 read, `roi_signal_min<N>voxels.parquet` (and `.csv`), cleaned in three steps: every subject's ROI with fewer than `MIN_VOXELS` voxels (default 100) is removed, then every row with NaN values, then, for every subject, each label some subject has no row of left. Every removed row is listed with its reason (`small`, `nan`, `incomplete`) in `removed_rois_min<N>voxels.csv`, the labels removed for everyone with the subjects without them in `removed_labels_min<N>voxels.csv`. A second job writes each subject's brain volumes to `volumes.csv` and adds them to `subject_info.csv` for 05: ventricles, SAS (CSF without the ventricles), parenchyma and cortical grey matter (gMRI's `roi_groups`), in mL (voxel count × voxel volume of each label in the uncleaned `roi_signal.parquet`) and as a fraction of all labelled volume; it prints their medians per Group. `PREPROCESS=no` only redoes these two jobs, e.g. after changing `MIN_VOXELS`. | 00 |
| `02_decompose.sh "<groups>" ["<ranks>"]` | CP fit of each group and rank with random restarts: the best restart and the restarts' reproducibility, in `decompose/cp/rank_<r>_<group>/`. Starts 04 for each fit once it is collected (`PLOT=no`: not). | 00, 01 |
| `03_replicability.sh "<groups>" ["<ranks>"]` | Half-half replicability: fits of random halves of each group's subjects, the factor match score per split and rank, and its plot against rank. | 00, 01 |
| `04_plot_decomp.sh "<groups>" ["<ranks>"]` | `gmri plot decomposition`: mode grid, subject mode, time mode, spatial maps on `REFERENCE_SUBJECT`, outliers, and the subject and time coefficients as csv. A small job also writes `cp_rank_<r>__components.tex`, the LaTeX `tabular` of each component's weight and share of the data's sum of squares, their sum and the model fit (1 − relative squared error), all read from `model.h5` (`code/analysis/component_table.py`); no caption, names blank, to `\input` in a `table`. | 02 |
| `05_subject_stats.sh "<groups>" ["<ranks>"]` | `gmri stats`: covariate-adjusted tests of the subject coefficients (primary variables, and Height, Weight and the brain volumes of 01 adjusted for them), then their figures and a LaTeX table of the variables per group (`code/plot_stats`); that job prints the tests and the pairs of components with p_perm < 0.05 (effect with 95% CI, p_perm, p_fdr) at the end of its output. See gMRI's `docs/subject-statistics.md`. With `COMPONENT_CORRELATIONS=yes` (default, rank ≥ 2) also a job of its own for the correlations between components: per pair, the partial r adjusted for the primary variables with 95% CI, permutation p and FDR over the pairs (gmri's own test, `gMRItensor.subject_statistics.clinical_effects`), and the unadjusted r, in `__subject_stats_components.csv`, drawn as a lower-triangle heatmap. | 02 |

`<groups>` and `<ranks>` are quoted, space-separated lists; every group and rank is its own fit.
Without `<ranks>`, `RANKS` of `settings.sh` (`2 3 4 5 6`) is used.

## Options

- **Per stage.** Each script starts with one variable per option of the command it runs, with a
  comment and its default, then the sbatch options of its jobs. Override any of them for one run
  from the environment, e.g. `RESTARTS=100 bash code/pipeline/02_decompose.sh REF 4`.
- **Shared.** `settings.sh`: `RANKS`, `MIN_VOXELS`, the spatial maps (`REFERENCE_SUBJECT`,
  `REFERENCE_BACKGROUND`, `SLICES`), the SLURM array settings (`TASKS_PER_JOB`, `ARRAY_LIMIT`,
  `MAX_ARRAY_SIZE`) and local runs (`LOCAL_PROCS`, `RUN_LOCALLY`).
- **Spatial maps.** Drawn on `REFERENCE_SEGMENTATION` over `REFERENCE_BACKGROUND` at `SLICES`.
  `.env` may name the two images, e.g. a cropped plotting subject (`code/adapt_example_subject.py`);
  otherwise they are `REFERENCE_SUBJECT`'s (default `$SUB_REF_MEDIAN`) segmentation and T1w from
  00. A segmentation without a background is drawn on its own mask. `SLICES` (`"I J K"`) is gmri's
  `--slices`: voxel indices of the background reoriented to RAS (nibabel's
  `as_closest_canonical`; the mode grid shows I). Empty: the centre of mass.
- **Cleaning (01).** `MIN_VOXELS` (default 100) is the smallest ROI, in voxels of a subject's own
  segmentation, that is used. 01 removes the rows of smaller ones (every time point), then the
  rows with NaN values (that scan only), then every label that some subject then has no row of,
  for every subject: the models only see labels present in every subject (and draw the others
  blank in the maps). The remaining gaps, a subject's single NaN scans of a label, are imputed by
  CP and do not count toward 02/03's `MAX_INVALID_FRACTION`, which only counts non-finite values.
  Each threshold has its own file, so fits of different thresholds are not mixed up.
- **Checks.** Every value is checked before anything is submitted: unknown groups or ranks,
  missing inputs, and (05) variables that are not columns of `subject_info.csv`.

## How they run

- **On SLURM** (where `sbatch` exists): 01–05 submit jobs and return. 02 and 03 run a plan job,
  one fit array per rank (`TASKS_PER_JOB` restarts per task) and a collect job
  (`DISTRIBUTE=no`: one process instead); 02's collect then starts 04's job, and 03's its plot.
- **Here** (no `sbatch`, or `RUN_LOCALLY=yes`): every job runs on this machine, one after another,
  in one process each, limited to `LOCAL_PROCS` CPUs (default 4): parallel fits
  (`RESTART_PROCS`, `N_PROCS`) are lowered to it, and every library gets that many threads. Each
  parallel fit process takes about 1 GB of memory. Example:
  `RUN_LOCALLY=yes LOCAL_PROCS=8 bash code/pipeline/02_decompose.sh "REF iNPH" "3 4"`.
- **Resuming.** Rerun the same command: 02 refits everything (`OVERWRITE=yes`); 03 reuses the
  stored restarts fitted with the same settings (`OVERWRITE=no`) and fits the rest. Plan and
  collect go through every stored restart one file at a time (03: 2 × `REPEATS` × `RESTARTS`
  per rank; collect reads each twice), so they need hours, not minutes. When only 03's collect
  failed (e.g. out of time), `COLLECT_ONLY=yes` reruns just collect and plot on the stored
  restarts: `COLLECT_ONLY=yes bash code/pipeline/03_replicability.sh all`.
- **Rerunning 00** rewrites `subject_info.csv` without the brain volumes; rerun 01
  (`PREPROCESS=no` is enough) before 05, which otherwise stops with "Not columns of …".
- **Record of what ran.** Every command is printed and appended to
  `$RESULTS_DIR/commands.log`, with the time and code version; each gmri command also writes
  its settings as json next to its outputs. SLURM logs go to `slurm/<job name>_<job id>.out`.

## Outputs

```
$RESULTS_DIR/                              ($BASE_DATA_FOLDER/tensor)
  files_to_process.csv  subject_info.csv  manifest.csv  volumes.csv (01)  commands.log
  <subject>/                               <subject>-segmentation.nii.gz, -brainmask.nii.gz, -T1w.nii.gz
  preprocess/data/                         roi_signal.parquet, roi_signal.csv, preprocess.json,
                                           roi_signal_min<N>voxels.{parquet,csv},
                                           removed_{rois,labels}_min<N>voxels.csv
  decompose/cp/rank_<r>_<group>/           cp_rank_<r>/ (model.h5, decompose.json, excluded_subjects.csv),
                                           fits.csv, plan.json, restarts/cp_rank_<r>/ (reproducibility.csv/.png)
  replicability/<group>/                   replicability.csv, plan.json, excluded_subjects.csv, restarts/
  plots/decomposition/cp/rank_<r>_<group>/ figures/decomposition/cp_rank_<r>__*.{pdf,png,csv,tex} (04),
                                           statistics/cp_rank_<r>__subject_stats_*.{csv,pdf,png,tex} (05,
                                           incl. __subject_stats_components.{csv,pdf,png})
  plots/replicability/<group>/             fms_by_rank.{pdf,png}, fms_summary.csv
```

## Run everything

```bash
bash code/pipeline/00_find_files.sh
bash code/pipeline/01_preprocessing.sh                          # wait for the job
bash code/pipeline/02_decompose.sh "all REF iNPH M F"            # also plots each fit (04)
bash code/pipeline/03_replicability.sh "all REF iNPH M F"
bash code/pipeline/05_subject_stats.sh "all REF iNPH M F" "3 4"  # once 02's fits are done
```

## Tests

`uv run pytest` runs the stages against a stub `sbatch` and a stub `apptainer` (which runs the
repository's Python and the installed `gmri` for real), including small end-to-end fits on
synthetic data (`tests/test_pipeline_scripts.py`).
