import subprocess
from pathlib import Path

from tensor4gmri.config import (
    GROUP_FILTER,
    GROUPS,
    PRIMARY_CATEGORICAL,
    STRATIFY_BY,
    TIME_POINTS,
)

REPO = Path(__file__).resolve().parents[1]


def test_groups_match_the_pipeline_settings() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source {REPO}/code/pipeline/settings.sh && echo "${{SUBJECT_GROUPS[*]}}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert tuple(result.stdout.split()) == GROUPS


def test_every_group_has_a_filter_stratification_and_covariates() -> None:
    for groups in (set(GROUP_FILTER), set(STRATIFY_BY), set(PRIMARY_CATEGORICAL)):
        assert groups == set(GROUPS)


def test_groups_are_stratified_and_adjusted_by_what_varies_within_them() -> None:
    assert GROUP_FILTER["all"] is None
    for group, selection in GROUP_FILTER.items():
        if selection is None:
            continue
        column, _ = selection
        assert STRATIFY_BY[group] != column
        assert column not in PRIMARY_CATEGORICAL[group]
    assert STRATIFY_BY["all"] == "Stratum"
    assert PRIMARY_CATEGORICAL["all"] == ("Group", "Sex")


def test_time_points_are_the_hours_of_the_tensor() -> None:
    assert TIME_POINTS == (2, 4, 8, 24)
