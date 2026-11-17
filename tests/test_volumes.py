"""Brain volumes per subject from gmri preprocess's roi_signal (code/tensor4gmri/volumes.py and
code/preprocessing/subject_volumes.py)."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tensor4gmri.volumes import VOLUME_COLUMNS, subject_volumes

REPO = Path(__file__).resolve().parents[1]
# Voxels per label: 4 and 10004 ventricles, 24 and 10024 SAS, 2 white matter, 17 hippocampus
# (subcortical grey matter: parenchyma, not cortical), 1001 cortex.
VOXELS = {
    "A": {4: 1000, 10004: 200, 24: 300, 10024: 500, 2: 4000, 17: 600, 1001: 2000},
    "B": {4: 3000, 10004: 600, 24: 100, 10024: 300, 2: 3600, 17: 400, 1001: 2000},
}


def roi_signal() -> pd.DataFrame:
    rows = [
        {
            "subject": s, "time_point": t, "label": label, "median": 1.0, "mean": 1.0,
            "n_voxels": n, "n_valid": n, "n_rejected": 0, "voxel_volume_mm3": 0.5,
        }
        for s, labels in VOXELS.items() for t in (2, 24) for label, n in labels.items()
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_volumes_in_ml_and_as_fractions_of_the_total() -> None:
    volumes = subject_volumes(roi_signal()).set_index("subjects")
    a = volumes.loc["A"]
    assert a["ventricles_volume_ml"] == pytest.approx(1200 * 0.5 / 1000)
    assert a["sas_volume_ml"] == pytest.approx(800 * 0.5 / 1000)
    assert a["parenchyma_volume_ml"] == pytest.approx(6600 * 0.5 / 1000)
    assert a["cortical_grey_matter_volume_ml"] == pytest.approx(2000 * 0.5 / 1000)
    assert a["total_volume_ml"] == pytest.approx(8600 * 0.5 / 1000)
    assert a["ventricles_volume_fraction"] == pytest.approx(1200 / 8600)
    assert a["cortical_grey_matter_volume_fraction"] == pytest.approx(2000 / 8600)
    b = volumes.loc["B"]
    assert b["ventricles_volume_ml"] == pytest.approx(3600 * 0.5 / 1000)
    for row in volumes.itertuples():
        parts = row.ventricles_volume_fraction + row.sas_volume_fraction
        assert parts + row.parenchyma_volume_fraction == pytest.approx(1)
    assert list(volumes.reset_index().columns) == ["subjects", *VOLUME_COLUMNS]


def test_a_segmentation_that_changes_between_time_points_is_an_error() -> None:
    frame = roi_signal()
    frame.loc[
        (frame["subject"] == "B") & (frame["time_point"] == 24) & (frame["label"] == 2),
        "n_voxels",
    ] = 3700  # fmt: skip
    with pytest.raises(ValueError, match="B"):
        subject_volumes(frame)


def run(source: Path, info: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(REPO / "code/preprocessing/subject_volumes.py"), "--input",
            str(source), "--subject-info", str(info), "--output", str(output),
        ],
        env={**os.environ, "PYTHONPATH": str(REPO / "code")},
        capture_output=True, text=True, check=False,
    )  # fmt: skip


def test_script_writes_the_volumes_and_adds_them_to_the_subject_info(
    tmp_path: Path,
) -> None:
    source = tmp_path / "roi_signal.parquet"
    pq.write_table(pa.Table.from_pandas(roi_signal(), preserve_index=False), source)
    info = tmp_path / "subject_info.csv"
    pd.DataFrame(
        {
            "subjects": ["A", "B", "C"],
            "Group": ["REF", "iNPH", "REF"],
            "Age": ["61", "74", "70"],
        },
    ).to_csv(info, index=False)
    output = tmp_path / "volumes.csv"

    for _ in range(2):  # a rerun replaces the columns
        result = run(source, info, output)
        assert result.returncode == 0, result.stderr
    assert pd.read_csv(output)["subjects"].tolist() == ["A", "B"]
    table = pd.read_csv(info)
    assert list(table.columns) == ["subjects", "Group", "Age", *VOLUME_COLUMNS]
    assert table.loc[0, "ventricles_volume_ml"] == pytest.approx(0.6)
    assert table.loc[2, VOLUME_COLUMNS].isna().all()  # C: no data
    assert table["Age"].tolist() == [61, 74, 70]
    assert "Median volumes per Group" in result.stdout
