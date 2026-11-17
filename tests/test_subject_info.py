from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import labels_frame, subject_ids, subject_info_frame
from tensor4gmri.subject_info import (
    excluded_for_group,
    group_members,
    left_out_labels,
    read_labels,
    read_subject_info,
    subject_info_from_labels,
    valid_labels,
)


def mixed_labels() -> pd.DataFrame:
    """Labels with excluded subjects (PAT_001; PAT_007 without an exclude value) and subjects of
    other diagnoses (PAT_002 coded 2, PAT_005 without a code)."""
    return labels_frame().assign(
        exclude=[0, 1, 0, 0, 0, 0, 0, np.nan],
        Group=[0, 0, 2, 0, 1, np.nan, 1, 1],
    )


VALID = ["PAT_000", "PAT_003", "PAT_004", "PAT_006"]


def test_labels_keep_ids_as_text_and_must_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    labels_frame().assign(old_id=["007"] * 8).to_csv(path, index=False)
    assert read_labels(path)["old_id"].tolist() == ["007"] * 8

    labels_frame().assign(new_id="PAT_000").to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than once.*PAT_000"):
        read_labels(path)


def test_valid_labels_are_included_ref_and_inph_subjects() -> None:
    assert valid_labels(mixed_labels())["new_id"].tolist() == VALID


def test_left_out_labels_say_why() -> None:
    assert left_out_labels(mixed_labels()) == {
        "excluded": ["PAT_001", "PAT_007"],
        "not REF or iNPH": ["PAT_002", "PAT_005"],
    }
    assert left_out_labels(labels_frame()) == {"excluded": [], "not REF or iNPH": []}


def test_subject_info_names_the_codes_of_the_valid_subjects() -> None:
    info = subject_info_from_labels(mixed_labels())

    assert info.columns[0] == "subjects"
    assert info["subjects"].tolist() == VALID
    assert info["Group"].tolist() == ["REF", "REF", "iNPH", "iNPH"]
    assert info["Sex"].tolist() == ["M", "F", "M", "M"]
    assert info["Stratum"].tolist() == ["REF_M", "REF_F", "iNPH_M", "iNPH_M"]
    assert info["Age"].tolist() == [60, 63, 64, 66]


def test_full_subject_info_matches_the_fixture() -> None:
    info = subject_info_from_labels(labels_frame())
    pd.testing.assert_frame_equal(info, subject_info_frame(), check_like=True)


def test_subject_info_rejects_unknown_sex_codes_of_valid_subjects() -> None:
    with pytest.raises(ValueError, match="Sex"):
        subject_info_from_labels(labels_frame().assign(Sex=2))


def test_read_subject_info_keeps_text(subject_info_path: Path) -> None:
    info = read_subject_info(subject_info_path)
    assert info["Age"].tolist()[0] == "60"


@pytest.mark.parametrize(
    ("group", "members"),
    [
        ("all", subject_ids()),
        ("REF", subject_ids()[:4]),
        ("iNPH", subject_ids()[4:]),
        ("M", subject_ids()[0::2]),
        ("F", subject_ids()[1::2]),
    ],
)
def test_group_members(group: str, members: list[str]) -> None:
    assert group_members(subject_info_frame(), group) == members


def test_excluded_for_group_lists_the_others_in_the_data_sorted() -> None:
    info = subject_info_frame()
    present = ["PAT_007", "PAT_001", "PAT_000", "PAT_004"]
    assert excluded_for_group(info, "all", present) == []
    assert excluded_for_group(info, "REF", present) == ["PAT_004", "PAT_007"]
    assert excluded_for_group(info, "F", present) == ["PAT_000", "PAT_004"]


def test_excluded_for_group_rejects_unknown_groups_and_subjects() -> None:
    info = subject_info_frame()
    with pytest.raises(ValueError, match="Unknown group 'PD'"):
        excluded_for_group(info, "PD", ["PAT_000"])
    with pytest.raises(ValueError, match="PAT_999"):
        excluded_for_group(info, "REF", ["PAT_000", "PAT_999"])
