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
| `SUBJECT_LABELS` | The subject labels csv: `new_id`, `old_id`, `Group` (0 REF, 1 iNPH), `Sex` (0 M, 1 F), `exclude`, `Age`, `Height`, `Weight`. |
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
| `00_find_files.sh` | Finds each subject's scans closest to 2, 4, 8 and 24 h after the first, merges its SAS (labels + 10000) and parenchyma segmentations and copies its baseline (pre-injection) T1w into `<subject>/`. Writes `files_to_process.csv`, `subject_info.csv` (labels named, `Stratum` added; subjects marked `exclude` left out) and `manifest.csv`. Runs here, not as a job. | raw data, labels |
| `01_preprocessing.sh` | `gmri preprocess --input-type T1w`: `preprocess/data/roi_signal.parquet` (median and mean per label and scan). | 00 |
| `02_decompose.sh "<groups>" ["<ranks>"]` | CP fit of each group and rank with random restarts: the best restart and the restarts' reproducibility, in `decompose/cp/rank_<r>_<group>/`. Starts 04 for each fit once it is collected (`PLOT=no`: not). | 00, 01 |
| `03_replicability.sh "<groups>" ["<ranks>"]` | Half-half replicability: fits of random halves of each group's subjects, the factor match score per split and rank, and its plot against rank. | 00, 01 |
| `04_plot_decomp.sh "<groups>" ["<ranks>"]` | `gmri plot decomposition`: mode grid, subject mode, time mode, spatial maps on `REFERENCE_SUBJECT`, outliers, and the subject and time coefficients as csv. | 02 |
| `05_subject_stats.sh "<groups>" ["<ranks>"]` | `gmri stats`: covariate-adjusted tests of the subject coefficients (primary variables, and Height and Weight adjusted for them), then their figures and a LaTeX table of the variables per group (`code/plot_stats`). See gMRI's `docs/subject-statistics.md`. | 02 |

`<groups>` and `<ranks>` are quoted, space-separated lists; every group and rank is its own fit.
Without `<ranks>`, `RANKS` of `settings.sh` (`2 3 4 5 6`) is used.

## Options

- **Per stage.** Each script starts with one variable per option of the command it runs, with a
  comment and its default, then the sbatch options of its jobs. Override any of them for one run
  from the environment, e.g. `RESTARTS=100 bash code/pipeline/02_decompose.sh REF 4`.
- **Shared.** `settings.sh`: `RANKS`, the spatial maps (`REFERENCE_SUBJECT`,
  `REFERENCE_BACKGROUND`, `SLICES`), the SLURM array settings (`TASKS_PER_JOB`, `ARRAY_LIMIT`,
  `MAX_ARRAY_SIZE`) and local runs (`LOCAL_PROCS`, `RUN_LOCALLY`).
- **Spatial maps.** Drawn on `REFERENCE_SUBJECT`'s merged segmentation (default
  `$SUB_REF_MEDIAN`), over `REFERENCE_BACKGROUND` (default that subject's T1w from 00), at
  `SLICES` (`"I J K"`, voxel indices of the background reoriented to RAS; the mode grid shows `I`;
  empty: centre of mass).
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
  stored restarts with the same settings (`OVERWRITE=no`) and refits the rest.
- **Record of what ran.** Every command is printed and appended to
  `$RESULTS_DIR/commands.log`, with the time and code version; each gmri command also writes
  its settings as json next to its outputs. SLURM logs go to `slurm/<job name>_<job id>.out`.

## Outputs

```
$RESULTS_DIR/                              ($BASE_DATA_FOLDER/tensor)
  files_to_process.csv  subject_info.csv  manifest.csv  commands.log
  <subject>/                               <subject>-segmentation.nii.gz, -brainmask.nii.gz, -T1w.nii.gz
  preprocess/data/                         roi_signal.parquet, roi_signal.csv, preprocess.json
  decompose/cp/rank_<r>_<group>/           cp_rank_<r>/ (model.h5, decompose.json, excluded_subjects.csv),
                                           fits.csv, plan.json, restarts/cp_rank_<r>/ (reproducibility.csv/.png)
  replicability/<group>/                   replicability.csv, plan.json, excluded_subjects.csv, restarts/
  plots/decomposition/cp/rank_<r>_<group>/ figures/decomposition/cp_rank_<r>__*.{pdf,png,csv} (04),
                                           statistics/cp_rank_<r>__subject_stats_*.{csv,pdf,png,tex} (05)
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
