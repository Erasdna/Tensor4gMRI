"""code/preprocessing/group_subjects.py, whose stdout the stages capture."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def group_subjects(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / "code/preprocessing/group_subjects.py"), *args],
        env={**os.environ, "PYTHONPATH": str(REPO / "code")},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("group", "excluded"),
    [
        ("all", ""),
        ("REF", "PAT_004 PAT_005 PAT_006 PAT_007"),
        ("M", "PAT_001 PAT_003 PAT_005 PAT_007"),
    ],
)
def test_prints_only_the_subjects_to_exclude(
    subject_info_path: Path,
    roi_signal_path: Path,
    group: str,
    excluded: str,
) -> None:
    result = group_subjects(
        "--subject-info",
        str(subject_info_path),
        "--input",
        str(roi_signal_path),
        "--group",
        group,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{excluded}\n"


@pytest.mark.parametrize(
    ("what", "group", "printed"),
    [
        ("stratify", "all", "Stratum"),
        ("stratify", "iNPH", "Sex"),
        ("stratify", "F", "Group"),
        ("covariates", "all", "Group Sex"),
        ("covariates", "REF", "Sex"),
    ],
)
def test_prints_the_columns_of_a_group(what: str, group: str, printed: str) -> None:
    result = group_subjects("--group", group, "--print", what)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{printed}\n"


def test_unknown_group_fails(subject_info_path: Path, roi_signal_path: Path) -> None:
    result = group_subjects(
        "--subject-info",
        str(subject_info_path),
        "--input",
        str(roi_signal_path),
        "--group",
        "PD",
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "PD" in result.stderr


def test_subject_missing_from_the_subject_info_fails(
    tmp_path: Path,
    subject_info_path: Path,
    roi_signal_path: Path,
) -> None:
    info = tmp_path / "short.csv"
    lines = subject_info_path.read_text().splitlines()
    info.write_text("\n".join(lines[:-1]) + "\n")  # without PAT_007
    result = group_subjects(
        "--subject-info",
        str(info),
        "--input",
        str(roi_signal_path),
        "--group",
        "all",
    )
    assert result.returncode != 0
    assert "PAT_007" in result.stderr


def test_exclusion_needs_the_data_and_the_subject_info() -> None:
    result = group_subjects("--group", "REF")
    assert result.returncode != 0
    assert "--subject-info" in result.stderr
