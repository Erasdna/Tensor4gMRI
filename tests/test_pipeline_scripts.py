"""The code/pipeline stages, run against a stub sbatch and a stub apptainer."""

import os
import re
import resource
import shlex
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import (
    labels_frame,
    subject_ids,
    subject_info_frame,
    write_raw_subject,
    write_roi_signal,
)
from gMRItensor.cli import _parser
from gMRItensor.model_io import load_decomposition

REPO = Path(__file__).resolve().parents[1]
SBATCH = """#!/bin/bash
if [ -n "${SBATCH_FAIL:-}" ]; then echo "sbatch: error: $SBATCH_FAIL" >&2; exit 1; fi
n=$(( $(cat "$SBATCH_COUNTER" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$SBATCH_COUNTER"
echo "$n"
"""
# Logs its options and the command after the container (CONTAINER=none), then runs python
# scripts and gmri for real.
APPTAINER = r"""#!/bin/bash
printf '%q ' "$@" >> "$APPTAINER_LOG.options"
echo >> "$APPTAINER_LOG.options"
ulimit -Sn >> "$APPTAINER_LOG.nofile"
while [ "$1" != none ]; do shift; done
shift
printf '%q ' "$@" >> "$APPTAINER_LOG"
echo >> "$APPTAINER_LOG"
set -- "${@//\/mnt\/project/$STUB_PROJECT}"  # the repo, bound at /mnt/project
if [ "$1" = python ] && [ -f "$2" ]; then PYTHONPATH="$STUB_PYTHONPATH" exec "$@"; fi
if [ "$1" = gmri ]; then exec "$@"; fi
"""
# A stage of one job, to test env.sh on its own.
TRIAL_STAGE = """#!/bin/bash -l
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
JOB=(--partition=normal --cpus-per-task=8 --mem-per-cpu=1G --time=00:10:00)
id=$(submit "$(job_name trial all 3)" "${JOB[@]}" -- gmri --help)
echo "job $id"
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    shutil.copytree(REPO / "code/analysis", project / "code/analysis")
    shutil.copytree(REPO / "code/plot_stats", project / "code/plot_stats")
    shutil.copytree(REPO / "code/pipeline", project / "code/pipeline")
    shutil.copytree(REPO / "code/scripts", project / "code/scripts")
    shutil.copytree(REPO / "code/preprocessing", project / "code/preprocessing")
    (project / "code/pipeline/trial.sh").write_text(TRIAL_STAGE)
    (project / ".env").write_text(
        f"export PROJECT_DIR={project}\nexport CONTAINER=none\n"
        f"export BASE_DATA_FOLDER={tmp_path / 'data'}\n"
        f"export RAW_TRACER_DATA_FOLDER={tmp_path / 'raw'}\n"
        f"export SUBJECT_LABELS={tmp_path / 'labels.csv'}\n",
    )
    (project / "code/VERSION").write_text("repo=test\n")
    for name, script in (("bin", APPTAINER), ("slurm_bin", SBATCH)):
        (tmp_path / name).mkdir()
    for path, script in (
        (tmp_path / "bin/apptainer", APPTAINER),
        (tmp_path / "slurm_bin/sbatch", SBATCH),
    ):
        path.write_text(script)
        path.chmod(0o755)
    return project


def run_stage(
    project: Path,
    stage: str,
    *args: str,
    env: dict[str, str] | None = None,
    sbatch: bool = True,
) -> subprocess.CompletedProcess[str]:
    bins = [project.parent / "bin"] + ([project.parent / "slurm_bin"] if sbatch else [])
    full_env = {
        "PATH": ":".join([*map(str, bins), os.environ["PATH"]]),
        "HOME": str(project.parent),
        "SBATCH_COUNTER": str(project.parent / "sbatch_counter"),
        "APPTAINER_LOG": str(project.parent / "apptainer.log"),
        "STUB_PYTHONPATH": str(REPO / "code"),
        "STUB_PROJECT": str(project),
        "SLURM_ACCOUNT": "acct",
        "PLOT": "no",  # 02 without 04's figures, unless a test asks for them
        "MIN_VOXELS": "0",  # no size filter: stages read roi_signal_min0voxels.parquet
        **(env or {}),
    }
    return subprocess.run(
        ["bash", str(project / "code/pipeline" / stage), *args],
        cwd=project,
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def logged(project: Path) -> list[str]:
    log = project.parent / "data/tensor/commands.log"
    return log.read_text().splitlines() if log.exists() else []


def ran(project: Path) -> list[list[str]]:
    """The commands the apptainer stub ran, in order."""
    log = project.parent / "apptainer.log"
    return (
        [shlex.split(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )


def container_options(project: Path) -> list[str]:
    log = project.parent / "apptainer.log.options"
    return log.read_text().splitlines() if log.exists() else []


def test_submits_with_the_account_and_logs_the_sbatch_command(project: Path) -> None:
    result = run_stage(project, "trial.sh")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("job 1")
    (line,) = logged(project)
    assert " repo=test sbatch --parsable " in line
    assert "--account=acct" in line
    assert "--job-name=trial_rank3_all" in line
    assert line.endswith("/sbatch/job.sh gmri --help")
    assert ran(project) == []


def test_runs_here_without_sbatch_limited_to_local_procs(project: Path) -> None:
    result = run_stage(project, "trial.sh", env={"LOCAL_PROCS": "2"}, sbatch=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("job local")
    assert ran(project) == [["gmri", "--help"]]
    assert "usage: gmri" in result.stderr
    (options,) = container_options(project)
    assert "--env OMP_NUM_THREADS=2" in options
    assert "--env PYTHONPATH=/mnt/project/code" in options
    assert "--cleanenv" in options
    assert f"--bind {project.parent}" in options  # the folder of $SUBJECT_LABELS


def test_run_locally_runs_here_although_sbatch_exists(project: Path) -> None:
    result = run_stage(project, "trial.sh", env={"RUN_LOCALLY": "yes"})
    assert result.returncode == 0, result.stderr
    assert ran(project) == [["gmri", "--help"]]
    assert not (project.parent / "sbatch_counter").exists()


def test_submitting_needs_an_account(project: Path) -> None:
    result = run_stage(project, "trial.sh", env={"SLURM_ACCOUNT": ""})
    assert result.returncode == 1
    assert "Set SLURM_ACCOUNT" in result.stderr
    assert logged(project) == []


def test_a_local_run_needs_no_account(project: Path) -> None:
    result = run_stage(project, "trial.sh", env={"SLURM_ACCOUNT": ""}, sbatch=False)
    assert result.returncode == 0, result.stderr


def test_failed_submission_stops_the_stage(project: Path) -> None:
    result = run_stage(project, "trial.sh", env={"SBATCH_FAIL": "no"})
    assert result.returncode != 0
    assert "sbatch: error: no" in result.stderr


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RUN_LOCALLY": "maybe"}, "Unknown RUN_LOCALLY"),
        ({"LOCAL_PROCS": "0"}, "LOCAL_PROCS"),
    ],
)
def test_invalid_settings_stop_before_anything_runs(
    project: Path,
    env: dict[str, str],
    message: str,
) -> None:
    result = run_stage(project, "trial.sh", env=env)
    assert result.returncode == 1
    assert message in result.stderr
    assert logged(project) == [] and ran(project) == []


def test_missing_env_file_stops_the_stage(project: Path) -> None:
    (project / ".env").unlink()
    result = run_stage(project, "trial.sh")
    assert result.returncode == 1
    assert "Cannot find" in result.stdout


def results(project: Path) -> Path:
    return project.parent / "data/tensor"


def parses_with_gmri(line: str) -> None:
    """The gmri command at the end of a logged line parses with the installed gmri."""
    words = shlex.split(line.split(" ", 2)[2].replace("{task}", "0"))  # as job.sh does
    _parser().parse_args(words[words.index("gmri") + 1 :])


def test_find_files_writes_the_scans_subjects_and_manifest(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    raw, labels = raw_data
    labels_frame().assign(
        exclude=[0, 0, 0, 0, 0, 0, 0, 1],  # PAT_007 excluded
        Group=[0, 0, 0, 0, 1, 2, 1, 1],  # PAT_005 neither REF nor iNPH
    ).to_csv(labels, index=False)
    shutil.rmtree(raw / "PAT_006")
    write_raw_subject(raw / "PAT_006", "PAT_006", second_day=False)

    result = run_stage(project, "00_find_files.sh")
    assert result.returncode == 0, result.stderr + result.stdout

    valid = sorted(set(subject_ids()) - {"PAT_005", "PAT_007"})
    found = sorted(set(valid) - {"PAT_006"})  # PAT_006 has no second-day scan
    for subject in ("PAT_005", "PAT_006", "PAT_007"):
        assert not (results(project) / subject).exists(), subject  # never processed
    assert "excluded: PAT_007" in result.stdout
    assert "not REF or iNPH: PAT_005" in result.stdout
    assert "No scans of 1 subject(s): PAT_006" in result.stdout
    files = pd.read_csv(
        results(project) / "files_to_process.csv",
        dtype={"old_id": str},
    )
    assert sorted(files["subject"].unique()) == found
    first = files[files["subject"] == "PAT_000"]
    assert first["time_point"].tolist() == [2, 4, 8, 24]
    assert first["actual_time"].tolist() == [2.0, 4.0, 8.0, 24.0]
    assert first["old_id"].tolist() == ["100"] * 4
    assert set(first["baseline_path"]) == {
        str(raw / "PAT_000/PAT_000-T1w-normalized-20170101_083000.nii.gz"),
    }
    assert first["post_injection_path"].iloc[-1].endswith("20170102_083000.nii.gz")

    segmentation = results(project) / "PAT_000/PAT_000-segmentation.nii.gz"
    assert set(first["segmentation_path"]) == {str(segmentation)}
    assert sorted(np.unique(nib.load(segmentation).get_fdata()).astype(int)) == [
        0,
        2,
        17,
        41,
        10024,
    ]
    mask = nib.load(results(project) / "PAT_000/PAT_000-brainmask.nii.gz").get_fdata()
    assert set(np.unique(mask)) == {0, 1}
    baseline = raw / "PAT_000/PAT_000-T1w-normalized-20170101_083000.nii.gz"
    t1w = results(project) / "PAT_000/PAT_000-T1w.nii.gz"
    assert t1w.read_bytes() == baseline.read_bytes()  # the pre-injection scan
    assert not any((raw / "PAT_000").glob("*segmentation*"))  # raw data untouched

    info = pd.read_csv(results(project) / "subject_info.csv")
    assert info["subjects"].tolist() == valid
    assert info.loc[0, ["Group", "Sex", "Stratum"]].tolist() == ["REF", "M", "REF_M"]

    manifest = pd.read_csv(results(project) / "manifest.csv")
    assert manifest.columns.tolist() == [
        "subject", "time_point", "baseline_path", "post_injection_path", "mask_path",
        "segmentation_path",
    ]  # fmt: skip
    assert len(manifest) == 4 * len(found)
    assert [line.split()[3].rsplit("/", 1)[-1] for line in logged(project)] == [
        "make_subject_info.py", "find_files.py", "make_manifest.py",
    ]  # fmt: skip


def test_preprocessing_submits_gmri_preprocess_of_t1w(project: Path) -> None:
    (results(project)).mkdir(parents=True)
    (results(project) / "manifest.csv").write_text("subject\n")
    (results(project) / "subject_info.csv").write_text("subjects\n")
    result = run_stage(project, "01_preprocessing.sh")
    assert result.returncode == 0, result.stderr
    line, clean, volumes = logged(project)
    assert "--job-name=clean_roi_signal " in clean and " --min-voxels 0 " in clean
    assert "--job-name=subject_volumes " in volumes
    assert "--job-name=preprocess " in line
    assert "--cpus-per-task=8" in line
    assert line.endswith(
        f"gmri preprocess --manifest {results(project)}/manifest.csv "
        f"--output-dir {results(project)}/preprocess --input-type T1w --n-procs 8",
    )
    parses_with_gmri(line)


def test_preprocessing_needs_the_manifest(project: Path) -> None:
    result = run_stage(project, "01_preprocessing.sh")
    assert result.returncode == 1
    assert "00_find_files.sh" in result.stderr
    assert logged(project) == []


def test_preprocessing_cleans_the_signal_after_gmri_preprocess(project: Path) -> None:
    results(project).mkdir(parents=True)
    (results(project) / "manifest.csv").write_text("subject\n")
    (results(project) / "subject_info.csv").write_text("subjects\n")
    result = run_stage(project, "01_preprocessing.sh", env={"MIN_VOXELS": "100"})
    assert result.returncode == 0, result.stderr
    preprocess, clean, volumes = logged(project)
    data = f"{results(project)}/preprocess/data"
    assert (
        "--job-name=subject_volumes " in volumes
        and " --dependency=afterok:1 " in volumes
    )
    assert volumes.endswith(
        f"python code/preprocessing/subject_volumes.py --input {data}/roi_signal.parquet "
        f"--subject-info {results(project)}/subject_info.csv --output "
        f"{results(project)}/volumes.csv",
    )
    assert "--job-name=preprocess " in preprocess
    assert (
        "--job-name=clean_roi_signal " in clean and " --dependency=afterok:1 " in clean
    )
    assert clean.endswith(
        f"python code/preprocessing/clean_roi_signal.py --input {data}/roi_signal.parquet "
        f"--min-voxels 100 --output {data}/roi_signal_min100voxels.parquet "
        f"--report {data}/removed_rois_min100voxels.csv "
        f"--label-report {data}/removed_labels_min100voxels.csv",
    )


@pytest.mark.parametrize("min_voxels", ["50", "0"])
def test_preprocessing_can_only_clean(project: Path, min_voxels: str) -> None:
    with_data(project)
    env = {"MIN_VOXELS": min_voxels, "PREPROCESS": "no"}
    result = run_stage(project, "01_preprocessing.sh", env=env)
    assert result.returncode == 0, result.stderr
    clean, volumes = logged(project)
    assert "--job-name=clean_roi_signal " in clean and " --dependency" not in clean
    assert "--job-name=subject_volumes " in volumes and " --dependency" not in volumes
    assert f" --min-voxels {min_voxels} " in clean


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PREPROCESS": "no", "MIN_VOXELS": "100"}, "roi_signal.parquet"),
        ({"MIN_VOXELS": "ten"}, "MIN_VOXELS"),
        ({"PREPROCESS": "maybe"}, "PREPROCESS"),
    ],
)
def test_preprocessing_checks_its_settings(
    project: Path,
    env: dict[str, str],
    message: str,
) -> None:
    results(project).mkdir(parents=True)
    (results(project) / "manifest.csv").write_text("subject\n")
    (results(project) / "subject_info.csv").write_text("subjects\n")
    result = run_stage(project, "01_preprocessing.sh", env=env)
    assert result.returncode == 1
    assert message in result.stderr
    assert logged(project) == []


def test_fits_read_the_masked_data(project: Path) -> None:
    data = with_data(project) / "preprocess/data"
    shutil.copy(data / "roi_signal.parquet", data / "roi_signal_min100voxels.parquet")
    result = run_stage(
        project,
        "02_decompose.sh",
        "REF",
        "3",
        env={"MIN_VOXELS": "100"},
    )
    assert result.returncode == 0, result.stderr
    (plan,) = gmri_lines(project, "plan")
    assert f" --input {data}/roi_signal_min100voxels.parquet " in plan
    (lookup,) = [cmd for cmd in ran(project) if "--group" in cmd]
    assert (
        lookup[lookup.index("--input") + 1] == f"{data}/roi_signal_min100voxels.parquet"
    )


def test_fits_need_the_masked_data(project: Path) -> None:
    with_data(project)
    result = run_stage(
        project,
        "02_decompose.sh",
        "REF",
        "3",
        env={"MIN_VOXELS": "100"},
    )
    assert result.returncode == 1
    assert "roi_signal_min100voxels.parquet" in result.stderr
    assert "01_preprocessing.sh" in result.stderr


def test_find_files_and_preprocessing_run_here(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    env = {"LOCAL_PROCS": "2", "MIN_VOXELS": "10"}
    for stage in ("00_find_files.sh", "01_preprocessing.sh"):
        result = run_stage(project, stage, env=env, sbatch=False)
        assert result.returncode == 0, result.stderr

    preprocess = next(cmd for cmd in ran(project) if cmd[:2] == ["gmri", "preprocess"])
    assert preprocess[-2:] == ["--n-procs", "2"]  # lowered to LOCAL_PROCS
    path = results(project) / "preprocess/data/roi_signal.parquet"
    metadata = pq.read_schema(path).metadata
    assert metadata[b"input_type"] == b"T1w" and metadata[b"signal"] == b"ratio"
    signal = pd.read_parquet(path)
    assert sorted(signal["subject"].unique()) == subject_ids()
    assert sorted(signal["time_point"].unique()) == [2, 4, 8, 24]
    assert sorted(signal["label"].unique()) == [2, 17, 41, 10024]
    ratio = signal.set_index(["subject", "time_point", "label"])["median"]
    assert ratio["PAT_000", 2, 17] == pytest.approx(1.1)
    assert ratio["PAT_000", 24, 10024] == pytest.approx(1.5)

    # Volumes from the 1 mm^3 synthetic segmentations: SAS label 10024 (36 voxels), parenchyma
    # labels 2, 41 (28 each) and 17 (8), no ventricles, no cortex.
    volumes = pd.read_csv(results(project) / "volumes.csv").set_index("subjects")
    assert volumes.loc["PAT_000", "sas_volume_ml"] == pytest.approx(0.036)
    assert volumes.loc["PAT_000", "parenchyma_volume_ml"] == pytest.approx(0.064)
    assert volumes.loc["PAT_000", "ventricles_volume_ml"] == 0
    assert volumes.loc["PAT_000", "cortical_grey_matter_volume_fraction"] == 0
    info = pd.read_csv(results(project) / "subject_info.csv")
    assert info["total_volume_ml"].tolist() == pytest.approx([0.1] * len(subject_ids()))

    # Label 17 has 8 voxels in every subject: removed; the others (28-36 voxels) kept.
    data = results(project) / "preprocess/data"
    cleaned = pd.read_parquet(data / "roi_signal_min10voxels.parquet")
    assert sorted(cleaned["label"].unique()) == [2, 41, 10024]
    assert cleaned["median"].notna().all()
    assert len(pd.read_csv(data / "roi_signal_min10voxels.csv")) == len(cleaned)
    report = pd.read_csv(data / "removed_rois_min10voxels.csv")
    assert set(report["label"]) == {17} and set(report["reason"]) == {"small"}
    labels = pd.read_csv(data / "removed_labels_min10voxels.csv")
    assert labels.to_dict("records") == [
        {
            "label": 17,
            "n_subjects_without": 8,
            "subjects_without": " ".join(subject_ids()),
        },
    ]

    fit = {"RESTARTS": "2", "MAX_ITER": "300", "TOLERANCE": "1e-4", **env}
    result = run_stage(project, "02_decompose.sh", "all", "2", env=fit, sbatch=False)
    assert result.returncode == 0, result.stderr
    model = load_decomposition(
        results(project) / "decompose/cp/rank_2_all/cp_rank_2/model.h5",
    )
    assert sorted(int(label) for label in model.labels) == [2, 41, 10024]


REF = subject_ids()[:4]
INPH = subject_ids()[4:]


def with_data(project: Path) -> Path:
    """The preprocessed data (gmri's and, without NaN rows, the copy the fits read with
    MIN_VOXELS=0) and subject table, as 00 and 01 write them."""
    data = results(project) / "preprocess/data"
    write_roi_signal(data / "roi_signal.parquet")
    shutil.copy(data / "roi_signal.parquet", data / "roi_signal_min0voxels.parquet")
    with_volumes(subject_info_frame()).to_csv(
        results(project) / "subject_info.csv",
        index=False,
    )
    return results(project)


VOLUME_NAMES = ("ventricles", "sas", "parenchyma", "cortical_grey_matter")
VOLUME_VARIABLES = [
    f"{name}_volume_{unit}" for unit in ("fraction", "ml") for name in VOLUME_NAMES
]
VOLUME_CATEGORIES = (
    "Anthropometrics=Height[cm],Weight[kg];"
    "Volume (as a fraction of total intracranial volume)="
    "ventricles_volume_fraction[,Ventricles],sas_volume_fraction[,SAS],"
    "parenchyma_volume_fraction[,Parenchyma],cortical_grey_matter_volume_fraction[,Cortical GM];"
    "Volume (mL)=ventricles_volume_ml[mL,Ventricles],sas_volume_ml[mL,SAS],"
    "parenchyma_volume_ml[mL,Parenchyma],cortical_grey_matter_volume_ml[mL,Cortical GM]"
)


def with_volumes(info: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """`info` with the volume columns 01 adds (code/preprocessing/subject_volumes.py)."""
    rng = np.random.default_rng(seed)
    ml = {name: rng.uniform(20, 400, len(info)) for name in VOLUME_NAMES}
    total = sum(ml.values())
    for name in VOLUME_NAMES:
        info[f"{name}_volume_ml"] = ml[name]
        info[f"{name}_volume_fraction"] = ml[name] / total
    return info.assign(total_volume_ml=total)


def gmri_lines(project: Path, command: str) -> list[str]:
    return [
        f"{line} "
        for line in logged(project)
        if f" gmri decompose {command} " in f"{line} "
    ]


def job_names(project: Path) -> list[str]:
    return [
        word.removeprefix("--job-name=")
        for line in logged(project)
        for word in line.split()
        if word.startswith("--job-name=")
    ]


def test_decompose_fits_each_group_without_the_other_subjects(project: Path) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", "all REF", "3")
    assert result.returncode == 0, result.stderr

    all_plan, ref_plan = gmri_lines(project, "plan")
    assert f"--output-dir {results(project)}/decompose/cp/rank_3_all " in all_plan
    assert "--exclude-subjects" not in all_plan
    assert f"--output-dir {results(project)}/decompose/cp/rank_3_REF " in ref_plan
    assert f" --exclude-subjects {' '.join(INPH)} " in ref_plan
    for line in (all_plan, ref_plan):
        assert " --method cp --statistic median --min-timepoints 3 " in line
        assert " --restarts 50 --max-iter 5000 --tolerance 1e-7 " in line
        assert " --non-negative-modes 0,1,2 --store restarts " in line
        assert " --overwrite " in line
        parses_with_gmri(line)
    for command in ("run", "collect"):
        for line in gmri_lines(project, command):
            parses_with_gmri(line)
    assert job_names(project) == [
        f"decompose_{stage}_rank3_{group}"
        for group in ("all", "REF")
        for stage in ("plan", "fit", "collect")
    ]


def test_decompose_reads_each_groups_subjects_once(project: Path) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", "REF iNPH M", "2 3 4")
    assert result.returncode == 0, result.stderr
    groups = [cmd[cmd.index("--group") + 1] for cmd in ran(project) if "--group" in cmd]
    assert groups == ["REF", "iNPH", "M"]
    assert len(gmri_lines(project, "plan")) == 9


@pytest.mark.parametrize(
    ("args", "env", "ranks"),
    [
        (("F", "3 4"), {"RANKS": "5"}, ["3", "4"]),
        (("F",), {"RANKS": "5"}, ["5"]),
        (("F",), {}, ["2", "3", "4", "5", "6"]),
    ],
)
def test_decompose_ranks(
    project: Path,
    args: tuple[str, ...],
    env: dict[str, str],
    ranks: list[str],
) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", *args, env=env)
    assert result.returncode == 0, result.stderr
    plans = gmri_lines(project, "plan")
    assert [line.split(" --ranks ")[1].split()[0] for line in plans] == ranks


def test_decompose_fit_array_waits_for_the_plan_and_collect_for_the_fits(
    project: Path,
) -> None:
    with_data(project)
    result = run_stage(
        project,
        "02_decompose.sh",
        "M",
        "3",
        env={"RESTARTS": "10", "TASKS_PER_JOB": "4"},
    )
    assert result.returncode == 0, result.stderr
    plan, fit, collect = (line for line in logged(project) if " sbatch " in line)
    assert (
        "job.sh --expect-jobs 2 gmri decompose plan " in plan
    )  # 10 restarts, 5 per task
    assert " --tasks-per-job 5 " in plan
    assert " --array=0-1%20 " in fit and " --dependency=afterany:1 " in fit
    assert fit.endswith(
        " gmri decompose run --output-dir "
        f"{results(project)}/decompose/cp/rank_3_M --job {{task}}",
    )
    assert " --dependency=afterany:2 " in collect


def test_decompose_options_override(project: Path) -> None:
    with_data(project)
    env = {
        "RESTARTS": "20", "MAX_ITER": "100", "TOLERANCE": "1e-5", "MIN_TIMEPOINTS": "4",
        "STATISTIC": "mean", "MAX_INVALID_FRACTION": "0.5", "OVERWRITE": "no",
    }  # fmt: skip
    result = run_stage(project, "02_decompose.sh", "all", "2", env=env)
    assert result.returncode == 0, result.stderr
    (plan,) = gmri_lines(project, "plan")
    assert " --statistic mean --min-timepoints 4 --max-invalid-fraction 0.5 " in plan
    assert " --restarts 20 --max-iter 100 --tolerance 1e-5 " in plan
    assert " --overwrite " not in plan


def test_decompose_in_one_process(project: Path) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", "REF", "3", env={"DISTRIBUTE": "no"})
    assert result.returncode == 0, result.stderr
    (line,) = [line for line in logged(project) if " sbatch " in line]
    assert "--job-name=decompose_rank3_REF " in line
    assert " gmri decompose run --output-dir " in line
    assert " --restart-procs 16 " in line  # the CPUs of the single job
    assert " --overwrite " not in line
    parses_with_gmri(line)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("PD",), "Unknown group 'PD'"),
        ((), "No group"),
        (("all", "0"), "Unknown rank '0'"),
    ],
)
def test_decompose_checks_its_arguments_first(
    project: Path,
    args: tuple[str, ...],
    message: str,
) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", *args)
    assert result.returncode == 1
    assert message in result.stderr
    assert logged(project) == [] and ran(project) == []


def test_decompose_needs_the_preprocessed_data(project: Path) -> None:
    result = run_stage(project, "02_decompose.sh", "all", "3")
    assert result.returncode == 1
    assert "01_preprocessing.sh" in result.stderr


def test_decompose_runs_here(project: Path) -> None:
    with_data(project)
    env = {"RESTARTS": "3", "MAX_ITER": "500", "TOLERANCE": "1e-4", "LOCAL_PROCS": "2"}
    result = run_stage(project, "02_decompose.sh", "REF", "2", env=env, sbatch=False)
    assert result.returncode == 0, result.stderr

    out = results(project) / "decompose/cp/rank_2_REF"
    model = load_decomposition(out / "cp_rank_2/model.h5")
    assert sorted(model.subjects) == REF
    excluded = pd.read_csv(out / "cp_rank_2/excluded_subjects.csv")
    assert sorted(excluded["subject"]) == INPH
    assert (out / "restarts/cp_rank_2/reproducibility.csv").is_file()


def test_distributed_decompose_runs_here(project: Path) -> None:
    with_data(project)
    env = {
        "RESTARTS": "4", "TASKS_PER_JOB": "2", "MAX_ITER": "500", "TOLERANCE": "1e-4",
        "DISTRIBUTE": "yes", "LOCAL_PROCS": "1",
    }  # fmt: skip
    result = run_stage(project, "02_decompose.sh", "iNPH", "2", env=env, sbatch=False)
    assert result.returncode == 0, result.stderr
    assert "Plan has 2 jobs" in result.stderr

    out = results(project) / "decompose/cp/rank_2_iNPH"
    assert sorted(load_decomposition(out / "cp_rank_2/model.h5").subjects) == INPH
    assert len(list((out / "restarts/cp_rank_2/full").glob("seed_*.pt"))) == 4
    assert [cmd[:3] for cmd in ran(project) if cmd[0] == "gmri"] == [
        ["gmri", "decompose", "plan"],
        ["gmri", "decompose", "run"],
        ["gmri", "decompose", "run"],
        ["gmri", "decompose", "collect"],
    ]


def with_reference(project: Path, raw_data: tuple[Path, Path]) -> Path:
    """The files of 00 (segmentations, T1w, manifest) and the data of 01; PAT_000 is the
    reference. Returns its T1w."""
    result = run_stage(project, "00_find_files.sh", sbatch=False)
    assert result.returncode == 0, result.stderr
    with_data(project)
    return results(project) / "PAT_000/PAT_000-T1w.nii.gz"


def plot_lines(project: Path) -> list[str]:
    return [
        f"{line} " for line in logged(project) if " gmri plot decomposition " in line
    ]


def table_lines(project: Path) -> list[str]:
    return [
        f"{line} "
        for line in logged(project)
        if " code/analysis/component_table.py " in line
    ]


def test_plot_decomposition_of_each_group(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    background = with_reference(project, raw_data)
    env = {"REFERENCE_SUBJECT": "PAT_000"}
    result = run_stage(project, "04_plot_decomp.sh", "all REF", "3", env=env)
    assert result.returncode == 0, result.stderr

    all_plot, ref_plot = plot_lines(project)
    out = results(project)
    for group, line in (("all", all_plot), ("REF", ref_plot)):
        assert f"--job-name=plot_decomposition_rank3_{group} " in line
        assert f" --model {out}/decompose/cp/rank_3_{group}/cp_rank_3 " in line
        assert f" --output-dir {out}/plots/decomposition/cp/rank_3_{group} " in line
        assert f" --subject-info {out}/subject_info.csv " in line
        assert f" --segmentation {out}/PAT_000/PAT_000-segmentation.nii.gz " in line
        assert f" --background {background} " in line
        assert " --numerical Age " in line and " --dpi 400 " in line
        assert " --slices " not in line
        assert " --page-width double " in line and " --min-panel-width " not in line
        parses_with_gmri(line)
    assert " --group-variable Group " in all_plot and " --categorical Sex " in all_plot
    assert " --group-variable Sex " in ref_plot and " --categorical " not in ref_plot
    all_table, ref_table = table_lines(project)
    for group, line in (("all", all_table), ("REF", ref_table)):
        figures = f"{out}/plots/decomposition/cp/rank_3_{group}/figures/decomposition"
        assert f"--job-name=component_table_rank3_{group} " in line
        assert f" --model {out}/decompose/cp/rank_3_{group}/cp_rank_3 " in line
        assert f" --output {figures}/cp_rank_3__components.tex " in line


def test_plot_decomposition_options(project: Path, raw_data: tuple[Path, Path]) -> None:
    with_reference(project, raw_data)
    other = results(project) / "PAT_001/PAT_001-T1w.nii.gz"
    env = {
        "REFERENCE_SUBJECT": "PAT_000", "SLICES": "3 3 3", "NUMERICAL": "",
        "DEPENDENCY": "afterany:7", "FORMATS": "png", "REFERENCE_BACKGROUND": str(other),
        "PAGE_WIDTH": "onehalf", "MIN_PANEL_WIDTH": "2",
    }  # fmt: skip
    result = run_stage(project, "04_plot_decomp.sh", "M", "2 4", env=env)
    assert result.returncode == 0, result.stderr
    lines = plot_lines(project)
    assert len(lines) == 2
    for line in lines:
        assert " --dependency=afterany:7 " in line
        assert " --slices 3 3 3 " in line and " --numerical " not in line  # stored RAS
        assert f" --background {other} " in line
        assert " --page-width onehalf " in line and " --min-panel-width 2 " in line
        assert " --formats png " in line
        assert " --group-variable Group " in line
        parses_with_gmri(line)
    tables = table_lines(project)
    assert len(tables) == 2
    assert all(" --dependency=afterany:7 " in line for line in tables)


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"REFERENCE_SUBJECT": ""}, "REFERENCE_SUBJECT"),
        ({"REFERENCE_SUBJECT": "PAT_999"}, "PAT_999-segmentation.nii.gz"),
        (
            {"REFERENCE_SUBJECT": "PAT_000", "REFERENCE_BACKGROUND": "/no/T1w.nii.gz"},
            "/no/T1w.nii.gz",
        ),
    ],
)
def test_plot_decomposition_needs_the_reference(
    project: Path,
    raw_data: tuple[Path, Path],
    env: dict[str, str],
    message: str,
) -> None:
    with_reference(project, raw_data)
    for stage, args in (
        ("04_plot_decomp.sh", ("REF", "3")),
        ("02_decompose.sh", ("REF", "3")),
    ):
        result = run_stage(
            project,
            stage,
            *args,
            env={**env, "PLOT": "yes"},
        )
        assert result.returncode == 1, stage
        assert message in result.stderr
    assert not any(" sbatch " in line for line in logged(project))


def test_decompose_plots_each_fit_once_collected(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    with_reference(project, raw_data)
    env = {"REFERENCE_SUBJECT": "PAT_000", "PLOT": "yes"}
    result = run_stage(project, "02_decompose.sh", "REF F", "3", env=env)
    assert result.returncode == 0, result.stderr
    names = job_names(project)
    assert names == [
        f"{stage}_rank3_{group}"
        for group in ("REF", "F")
        for stage in (
            "decompose_plan",
            "decompose_fit",
            "decompose_collect",
            "plot_decomposition",
            "component_table",
        )
    ]
    sbatch = [line for line in logged(project) if " sbatch " in line]
    assert " --dependency=afterany:3 " in sbatch[3]  # the collect job of REF


def test_decompose_without_plots(project: Path, raw_data: tuple[Path, Path]) -> None:
    with_data(project)
    result = run_stage(project, "02_decompose.sh", "REF", "3", env={"PLOT": "no"})
    assert result.returncode == 0, result.stderr
    assert plot_lines(project) == []


def test_decompose_and_plots_run_here(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    with_reference(project, raw_data)
    env = {
        "REFERENCE_SUBJECT": "PAT_000", "RESTARTS": "3", "MAX_ITER": "500",
        "TOLERANCE": "1e-4", "LOCAL_PROCS": "2", "FORMATS": "png", "PLOT": "yes",
    }  # fmt: skip
    result = run_stage(project, "02_decompose.sh", "all", "2", env=env, sbatch=False)
    assert result.returncode == 0, result.stderr

    figures = (
        results(project) / "plots/decomposition/cp/rank_2_all/figures/decomposition"
    )
    for part in (
        "mode_grid",
        "subject_mode",
        "time_mode",
        "spatial_parenchyma",
        "spatial_csf",
        "outliers",
    ):
        assert (figures / f"cp_rank_2__{part}.png").is_file(), part
    coefficients = pd.read_csv(figures / "cp_rank_2__subject_coefficients.csv")
    assert sorted(coefficients["subject"]) == subject_ids()
    table = (figures / "cp_rank_2__components.tex").read_text().splitlines()
    assert table[0] == r"\begin{tabular}{clcc}" and table[-1] == r"\end{tabular}"
    assert [line.split(" & ")[0] for line in table[4:6]] == ["1", "2"]


def replicability_lines(project: Path, command: str) -> list[str]:
    return [
        f"{line} "
        for line in logged(project)
        if f" gmri replicability {command} " in f"{line} "
    ]


def test_replicability_of_each_group(project: Path) -> None:
    with_data(project)
    result = run_stage(project, "03_replicability.sh", "all REF", "2 3")
    assert result.returncode == 0, result.stderr

    all_plan, ref_plan = replicability_lines(project, "plan")
    out = results(project)
    for group, line in (("all", all_plan), ("REF", ref_plan)):
        assert f" --output-dir {out}/replicability/{group} " in line
        assert f" --input {out}/preprocess/data/roi_signal_min0voxels.parquet " in line
        assert " --engine halfhalf --repeats 100 --seed 0 --n-procs 4 " in line
        assert f" --subject-info {out}/subject_info.csv " in line
        assert " --ranks 2 3 " in line
        assert (
            " --method cp --statistic median --min-timepoints 4 --max-invalid-fraction 0.2 "
            in line
        )
        assert " --restarts 100 --max-iter 5000 --tolerance 1e-7 " in line
        assert " --splits " not in line and " --restart-procs " not in line
        assert " --overwrite " not in line  # OVERWRITE=no by default
        parses_with_gmri(line)
    assert (
        " --stratify-by Stratum " in all_plan and " --exclude-subjects " not in all_plan
    )
    assert " --stratify-by Sex " in ref_plan
    assert f" --exclude-subjects {' '.join(INPH)} " in ref_plan
    for line in replicability_lines(project, "run") + replicability_lines(
        project,
        "collect",
    ):
        parses_with_gmri(line)


def test_replicability_arrays_per_rank_then_collect_and_plot(project: Path) -> None:
    with_data(project)
    env = {
        "RESTARTS": "10", "REPEATS": "2", "TASKS_PER_JOB": "100", "STRATIFY_BY": "Group",
        "OVERWRITE": "yes",
    }  # fmt: skip
    result = run_stage(project, "03_replicability.sh", "M", "2 3 4", env=env)
    assert result.returncode == 0, result.stderr

    sbatch = [line for line in logged(project) if " sbatch " in line]
    names = [line.split("--job-name=")[1].split()[0] for line in sbatch]
    assert names == [
        "replicability_plan_M",
        "replicability_fit_rank2_M",
        "replicability_fit_rank3_M",
        "replicability_fit_rank4_M",
        "replicability_collect_M",
        "replicability_plot_M",
    ]
    plan, *fits, collect, plot = (f"{line} " for line in sbatch)
    assert (
        "job.sh --expect-jobs 3 gmri replicability plan " in plan
    )  # 2 * 2 * 10 fits per rank
    assert " --tasks-per-job 40 " in plan and " --overwrite " in plan
    assert " --stratify-by Group " in plan
    assert [line.split("--array=")[1].split()[0] for line in fits] == [
        "0-0%20",
        "1-1%20",
        "2-2%20",
    ]
    assert all(" --dependency=afterany:1 " in line for line in fits)
    assert " --dependency=afterany:2:3:4 " in collect
    assert " --dependency=afterany:5 " in plot
    out = results(project)
    assert plot.rstrip().endswith(
        f"python code/analysis/plot_replicability.py {out}/replicability/M/replicability.csv "
        f"{out}/plots/replicability/M",
    )


def with_plan(project: Path, group: str) -> Path:
    """A `gmri replicability plan` output of `group` (03's collect reads only the store)."""
    out = results(project) / "replicability" / group
    out.mkdir(parents=True)
    (out / "plan.json").write_text("{}")
    return out


def test_replicability_collect_only(project: Path) -> None:
    with_data(project)
    out = with_plan(project, "M")
    result = run_stage(
        project,
        "03_replicability.sh",
        "M",
        "2 3",
        env={"COLLECT_ONLY": "yes"},
    )
    assert result.returncode == 0, result.stderr
    sbatch = [f"{line} " for line in logged(project) if " sbatch " in line]
    names = [line.split("--job-name=")[1].split()[0] for line in sbatch]
    assert names == ["replicability_collect_M", "replicability_plot_M"]
    collect, plot = sbatch
    assert " --dependency" not in collect
    assert f" gmri replicability collect --output-dir {out} " in collect
    assert " --dependency=afterany:1 " in plot
    parses_with_gmri(collect)


def test_replicability_collect_only_needs_a_plan(project: Path) -> None:
    with_data(project)
    with_plan(project, "M")
    result = run_stage(
        project,
        "03_replicability.sh",
        "M REF",
        env={"COLLECT_ONLY": "yes"},
    )
    assert result.returncode == 1
    assert f"{results(project)}/replicability/REF/plan.json" in result.stderr
    assert logged(project) == []


def test_replicability_in_one_process(project: Path) -> None:
    with_data(project)
    result = run_stage(
        project,
        "03_replicability.sh",
        "iNPH",
        "3",
        env={"DISTRIBUTE": "no"},
    )
    assert result.returncode == 0, result.stderr
    run, plot = (line for line in logged(project) if " sbatch " in line)
    assert "--job-name=replicability_iNPH " in run
    assert " gmri replicability run --output-dir " in run
    assert " --n-procs 16 " in run and " --tasks-per-job " not in run
    assert " --dependency=afterany:1 " in plot
    parses_with_gmri(run)


@pytest.mark.parametrize(
    ("args", "message"),
    [(("PD",), "Unknown group 'PD'"), (("all", "x"), "Unknown rank 'x'")],
)
def test_replicability_checks_its_arguments_first(
    project: Path,
    args: tuple[str, ...],
    message: str,
) -> None:
    with_data(project)
    result = run_stage(project, "03_replicability.sh", *args)
    assert result.returncode == 1
    assert message in result.stderr
    assert logged(project) == []


def test_replicability_needs_the_preprocessed_data(project: Path) -> None:
    result = run_stage(project, "03_replicability.sh", "all")
    assert result.returncode == 1
    assert "01_preprocessing.sh" in result.stderr


@pytest.mark.parametrize("distribute", ["no", "yes"])
def test_replicability_runs_here(project: Path, distribute: str) -> None:
    with_data(project)
    env = {
        "RESTARTS": "1", "REPEATS": "2", "TASKS_PER_JOB": "2", "MAX_ITER": "300",
        "TOLERANCE": "1e-4", "LOCAL_PROCS": "1", "DISTRIBUTE": distribute,
    }  # fmt: skip
    result = run_stage(
        project,
        "03_replicability.sh",
        "all",
        "1 2",
        env=env,
        sbatch=False,
    )
    assert result.returncode == 0, result.stderr

    fms = pd.read_csv(results(project) / "replicability/all/replicability.csv")
    assert sorted(fms["rank"].unique()) == [1, 2]
    assert len(fms) == 2 * 2  # repeats per rank
    assert fms["fms"].between(0, 1).all()
    plots = results(project) / "plots/replicability/all"
    assert (plots / "fms_by_rank.png").is_file()
    assert pd.read_csv(plots / "fms_summary.csv")["rank"].tolist() == [1, 2]
    if distribute == "yes":
        # Collect again from the stored restarts, without planning or fitting.
        table = results(project) / "replicability/all/replicability.csv"
        table.unlink()
        result = run_stage(
            project,
            "03_replicability.sh",
            "all",
            "1 2",
            env={**env, "COLLECT_ONLY": "yes"},
            sbatch=False,
        )
        assert result.returncode == 0, result.stderr
        assert pd.read_csv(table)["fms"].tolist() == fms["fms"].tolist()
        assert len(replicability_lines(project, "plan")) == 1
        assert len(replicability_lines(project, "collect")) == 2


def stats_lines(project: Path) -> list[str]:
    return [f"{line} " for line in logged(project) if " gmri stats " in line]


def stats_plot_lines(project: Path) -> list[str]:
    return [f"{line} " for line in logged(project) if "plot_subject_stats.py" in line]


def component_lines(project: Path) -> list[str]:
    return [
        f"{line} " for line in logged(project) if "component_correlations.py" in line
    ]


@pytest.mark.parametrize(
    ("env", "ranks"),
    [({"COMPONENT_CORRELATIONS": "no"}, "3"), ({}, "1"), ({"PLOTS_ONLY": "yes"}, "3")],
)
def test_subject_stats_without_component_correlations(
    project: Path,
    env: dict[str, str],
    ranks: str,
) -> None:
    with_data(project)
    result = run_stage(project, "05_subject_stats.sh", "all", ranks, env=env)
    assert result.returncode == 0, result.stderr
    assert component_lines(project) == []
    (plot,) = stats_plot_lines(project)
    if env.get("PLOTS_ONLY") != "yes":
        assert " --dependency=afterany:1 " in plot  # the stats job only


def test_component_correlations_setting_is_checked(project: Path) -> None:
    with_data(project)
    result = run_stage(
        project,
        "05_subject_stats.sh",
        "all",
        "3",
        env={"COMPONENT_CORRELATIONS": "maybe"},
    )
    assert result.returncode == 1
    assert "COMPONENT_CORRELATIONS" in result.stderr
    assert logged(project) == []


def test_subject_stats_of_each_group(project: Path) -> None:
    with_data(project)
    result = run_stage(project, "05_subject_stats.sh", "all REF", "3")
    assert result.returncode == 0, result.stderr

    out = results(project)
    all_stats, ref_stats = stats_lines(project)
    all_plot, ref_plot = stats_plot_lines(project)
    for group, stats, plot in (
        ("all", all_stats, all_plot),
        ("REF", ref_stats, ref_plot),
    ):
        model = f"{out}/decompose/cp/rank_3_{group}/cp_rank_3"
        stats_dir = f"{out}/plots/decomposition/cp/rank_3_{group}"
        assert f"--job-name=subject_stats_rank3_{group} " in stats
        assert (
            f" gmri stats --model {model} --subject-info {out}/subject_info.csv "
            in stats
        )
        assert f" --output-dir {stats_dir} " in stats
        assert (
            " --primary-numerical Age --clinical-numerical Height Weight "
            f"{' '.join(VOLUME_VARIABLES)} " in stats
        )
        assert " --n-permutations 10000 --seed 0 --alpha 0.05 " in stats
        assert (
            " --clinical-categorical " not in stats and " --interaction " not in stats
        )
        parses_with_gmri(stats)
        assert f"--job-name=plot_subject_stats_rank3_{group} " in plot
        assert f" --model {model} " in plot and f" --stats-dir {stats_dir} " in plot
        assert " --marker-variable Sex " in plot
        assert " --variable 'Age[years,Age]' " in plot  # quoted in the log
        logged_categories = plot.split(" --categories '")[1].split("'")[0]
        # a category may go on after a "," on the next line (05's default)
        assert re.sub(r",\s+", ",", logged_categories) == VOLUME_CATEGORIES
    assert " --primary-categorical Group Sex " in all_stats
    assert " --primary-categorical Sex " in ref_stats
    assert (
        " --group-variable Group " in all_plot and " --group-variable Sex " in ref_plot
    )
    assert " --dependency=afterany:1:2 " in all_plot  # its stats and component jobs
    all_components, ref_components = component_lines(project)
    for group, line, primary in (
        ("all", all_components, "Group Sex"),
        ("REF", ref_components, "Sex"),
    ):
        assert f"--job-name=component_stats_rank3_{group} " in line
        assert line.rstrip().endswith(
            f"python code/plot_stats/component_correlations.py --model {out}/decompose/cp/"
            f"rank_3_{group}/cp_rank_3 --subject-info {out}/subject_info.csv --output-dir "
            f"{out}/plots/decomposition/cp/rank_3_{group} --primary-categorical {primary} "
            "--primary-numerical Age --n-permutations 10000 --seed 0 --alpha 0.05",
        )


def test_subject_stats_variables_override(project: Path) -> None:
    with_data(project)
    env = {
        "PRIMARY_CATEGORICAL": "Sex", "PRIMARY_NUMERICAL": "Age Height",
        "CLINICAL_NUMERICAL": "Weight", "CLINICAL_CATEGORIES": "Body=Weight[kg]",
        "N_PERMUTATIONS": "500",
    }  # fmt: skip
    result = run_stage(project, "05_subject_stats.sh", "all", "2 3", env=env)
    assert result.returncode == 0, result.stderr
    stats = stats_lines(project)
    assert len(stats) == 2
    for line in stats:
        assert " --primary-categorical Sex --primary-numerical Age Height " in line
        assert (
            " --clinical-numerical Weight " in line and " --n-permutations 500 " in line
        )
    assert all(
        " --categories 'Body=Weight[kg]' " in line for line in stats_plot_lines(project)
    )
    assert all(" --group-variable Sex " in line for line in stats_plot_lines(project))


def test_subject_stats_plots_only(project: Path) -> None:
    with_data(project)
    result = run_stage(
        project,
        "05_subject_stats.sh",
        "M",
        "3",
        env={"PLOTS_ONLY": "yes"},
    )
    assert result.returncode == 0, result.stderr
    assert stats_lines(project) == []
    (plot,) = stats_plot_lines(project)
    assert " --dependency" not in plot


@pytest.mark.parametrize(
    ("env", "message"),
    [
        (
            {"CLINICAL_NUMERICAL": "Height BMI", "CLINICAL_CATEGORIES": "Body=BMI"},
            "Not columns of",
        ),
        ({"PRIMARY_NUMERICAL": "Age", "CLINICAL_NUMERICAL": "Age"}, "more than once"),
        ({"CLINICAL_CATEGORIES": "Body=BMI"}, "Not clinical variables"),
        ({"PLOTS_ONLY": "maybe"}, "PLOTS_ONLY"),
    ],
)
def test_subject_stats_checks_before_submitting(
    project: Path,
    env: dict[str, str],
    message: str,
) -> None:
    with_data(project)
    result = run_stage(project, "05_subject_stats.sh", "all", "3", env=env)
    assert result.returncode == 1
    assert message in result.stderr
    assert not any(" sbatch " in line for line in logged(project))


def test_subject_stats_runs_here(project: Path) -> None:
    with_data(project)
    env = {"RESTARTS": "3", "MAX_ITER": "500", "TOLERANCE": "1e-4", "LOCAL_PROCS": "2"}
    result = run_stage(project, "02_decompose.sh", "all", "2", env=env, sbatch=False)
    assert result.returncode == 0, result.stderr
    env = {"N_PERMUTATIONS": "200", "LOCAL_PROCS": "2"}
    result = run_stage(
        project,
        "05_subject_stats.sh",
        "all",
        "2",
        env=env,
        sbatch=False,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(
        r"^cp_rank_2: \d+ of \d+ tests with p_perm < 0.05$",
        result.stdout,
        re.M,
    )
    assert re.search(
        r"^cp_rank_2: [01] of 1 component pairs with p_perm < 0.05$",
        result.stdout,
        re.M,
    )

    statistics = results(project) / "plots/decomposition/cp/rank_2_all/statistics"
    primary = pd.read_csv(statistics / "cp_rank_2__subject_stats_primary.csv")
    assert set(primary["term"]) == {"Group", "Sex", "Age"}
    clinical = pd.read_csv(statistics / "cp_rank_2__subject_stats_clinical.csv")
    assert set(clinical["term"]) == {"Height", "Weight", *VOLUME_VARIABLES}
    for name in ("groups", "primary", "clinical"):
        assert (statistics / f"cp_rank_2__subject_stats_{name}.png").is_file(), name
    table = (statistics / "cp_rank_2__subject_stats_clinical_table.tex").read_text()
    assert "Anthropometrics" in table and "REF" in table and "iNPH" in table
    components = pd.read_csv(statistics / "cp_rank_2__subject_stats_components.csv")
    assert components[["component_a", "component_b"]].values.tolist() == [[1, 2]]
    assert (statistics / "cp_rank_2__subject_stats_components.png").is_file()


def test_containers_may_open_as_many_files_as_allowed(project: Path) -> None:
    """gmri's worker pools share each fit's data through a file descriptor; a soft limit of
    1024 open files (common on desktops) made 03's local runs fail."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    low = min(1024, hard)

    def lower_soft_limit() -> None:
        resource.setrlimit(resource.RLIMIT_NOFILE, (low, hard))

    env = {
        "PATH": f"{project.parent / 'bin'}:{os.environ['PATH']}",
        "HOME": str(project.parent),
        "APPTAINER_LOG": str(project.parent / "apptainer.log"),
        "STUB_PYTHONPATH": str(REPO / "code"),
    }
    result = subprocess.run(
        ["bash", str(project / "code/pipeline/trial.sh")],
        cwd=project,
        env=env,
        preexec_fn=lower_soft_limit,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    limits = (project.parent / "apptainer.log.nofile").read_text().split()
    expected = "unlimited" if hard == resource.RLIM_INFINITY else str(hard)
    assert limits == [expected]


def test_scripts_run_from_the_repo_bound_in_the_container(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    """The repo is bound at /mnt/project only, so its scripts must be called by that path,
    whatever directory the stage is started from."""
    result = subprocess.run(
        ["bash", str(project / "code/pipeline/00_find_files.sh")],
        cwd=project.parent,
        env={
            "PATH": f"{project.parent / 'bin'}:{os.environ['PATH']}",
            "HOME": str(project.parent),
            "APPTAINER_LOG": str(project.parent / "apptainer.log"),
            "STUB_PYTHONPATH": str(REPO / "code"),
            "STUB_PROJECT": str(project),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    scripts = [cmd[1] for cmd in ran(project) if cmd[0] == "python"]
    assert scripts == [
        "/mnt/project/code/preprocessing/make_subject_info.py",
        "/mnt/project/code/preprocessing/find_files.py",
        "/mnt/project/code/preprocessing/make_manifest.py",
    ]


def test_find_files_rounds_labels_stored_as_floats(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    raw = raw_data[0] / "PAT_000"
    refroi = nib.load(raw / "PAT_000-refroi.nii.gz")
    labels = refroi.get_fdata()
    labels[labels == 2] = 1.9999  # a label as a float or scaled image may read back
    image = nib.Nifti1Image(labels.astype(np.float32), refroi.affine)
    nib.save(image, raw / "PAT_000-refroi.nii.gz")

    result = run_stage(project, "00_find_files.sh", sbatch=False)
    assert result.returncode == 0, result.stderr
    segmentation = nib.load(results(project) / "PAT_000/PAT_000-segmentation.nii.gz")
    assert sorted(np.unique(segmentation.get_fdata()).astype(int)) == [
        0,
        2,
        17,
        41,
        10024,
    ]


def plotting_subject(project: Path) -> tuple[Path, Path]:
    """A cropped copy of PAT_000's segmentation and T1w, as code/adapt_example_subject.py makes,
    stored LIA like FreeSurfer's conformed volumes."""
    folder = results(project) / "plot_subject"
    folder.mkdir()
    files = []
    for name in ("PAT_000-segmentation.nii.gz", "PAT_000-T1w.nii.gz"):
        img = nib.load(results(project) / "PAT_000" / name).slicer[1:5, :, :]
        to_lia = nib.orientations.ornt_transform(
            nib.io_orientation(img.affine), nib.orientations.axcodes2ornt("LIA"),
        )  # fmt: skip
        files.append(folder / name)
        nib.save(img.as_reoriented(to_lia), files[-1])
    return files[0], files[1]


def test_plots_use_a_plotting_subject_named_in_the_settings(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    with_reference(project, raw_data)
    segmentation, background = plotting_subject(project)
    env = {
        "REFERENCE_SUBJECT": "", "REFERENCE_SEGMENTATION": str(segmentation),
        "REFERENCE_BACKGROUND": str(background), "SLICES": "1 2 3",
    }  # fmt: skip
    result = run_stage(project, "04_plot_decomp.sh", "REF", "3", env=env)
    assert result.returncode == 0, result.stderr
    (line,) = plot_lines(project)
    assert f" --segmentation {segmentation} " in line
    assert f" --background {background} " in line
    assert " --slices 1 2 3 " in line  # as given
    parses_with_gmri(line)


def test_a_plotting_segmentation_alone_draws_on_its_mask(
    project: Path,
    raw_data: tuple[Path, Path],
) -> None:
    with_reference(project, raw_data)
    segmentation, _ = plotting_subject(project)
    env = {"REFERENCE_SUBJECT": "", "REFERENCE_SEGMENTATION": str(segmentation)}
    result = run_stage(project, "04_plot_decomp.sh", "REF", "3", env=env)
    assert result.returncode == 0, result.stderr
    (line,) = plot_lines(project)
    assert f" --segmentation {segmentation} " in line and " --background " not in line


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"REFERENCE_SEGMENTATION": "/no/seg.nii.gz"}, "/no/seg.nii.gz"),
    ],
)
def test_plot_settings_are_checked_before_submitting(
    project: Path,
    raw_data: tuple[Path, Path],
    env: dict[str, str],
    message: str,
) -> None:
    with_reference(project, raw_data)
    result = run_stage(project, "04_plot_decomp.sh", "REF", "3", env=env)
    assert result.returncode != 0
    assert message in result.stderr
    assert plot_lines(project) == []


def test_subject_stats_need_the_volumes_of_01(project: Path) -> None:
    with_data(project)
    subject_info_frame().to_csv(results(project) / "subject_info.csv", index=False)
    result = run_stage(project, "05_subject_stats.sh", "all", "3")
    assert result.returncode == 1
    assert (
        "Not columns of" in result.stderr
        and "ventricles_volume_fraction" in result.stderr
    )
    assert not any(" sbatch " in line for line in logged(project))
