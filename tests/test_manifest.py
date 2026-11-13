from pathlib import Path

import pandas as pd
import pytest
from tensor4gmri.manifest import MANIFEST_COLUMNS, image_manifest


def file_list() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["PAT_000", "PAT_000"],
            "time_point": [2, 24],
            "baseline_path": ["a/base.nii.gz", "/abs/base.nii.gz"],
            "post_injection_path": ["a/post2.nii.gz", "/abs/post24.nii.gz"],
            "mask_path": ["a/mask.nii.gz", "/abs/mask.nii.gz"],
            "segmentation_path": ["a/seg.nii.gz", "/abs/seg.nii.gz"],
            "actual_time": [2.1, 23.9],
            "old_id": ["100", "100"],
        },
    )


def test_manifest_has_gmris_columns_and_absolute_paths() -> None:
    manifest = image_manifest(file_list(), Path("/data"))

    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert manifest["time_point"].tolist() == [2, 24]
    assert manifest["baseline_path"].tolist() == [
        "/data/a/base.nii.gz",
        "/abs/base.nii.gz",
    ]
    assert manifest["segmentation_path"].tolist() == [
        "/data/a/seg.nii.gz",
        "/abs/seg.nii.gz",
    ]


def test_manifest_rejects_other_time_points() -> None:
    with pytest.raises(ValueError, match=r"\[3\]"):
        image_manifest(file_list().assign(time_point=[2, 3]), Path("/data"))


def test_manifest_rejects_an_empty_file_list() -> None:
    with pytest.raises(ValueError, match="No scans"):
        image_manifest(file_list().iloc[:0], Path("/data"))
