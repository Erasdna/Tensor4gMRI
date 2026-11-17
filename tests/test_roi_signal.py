"""Cleaning gmri preprocess's per-label table (code/tensor4gmri/roi_signal.py and
code/preprocessing/clean_roi_signal.py): small ROIs, NaN values, then labels some subject lacks."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import subject_ids, write_roi_signal
from gMRItensor.cli import main
from gMRItensor.model_io import load_decomposition
from gMRItensor.preprocessing import read_signal_metadata
from tensor4gmri.roi_signal import clean_roi_signal

REPO = Path(__file__).resolve().parents[1]
METADATA = {"kind": "roi_signal", "input_type": "T1w", "signal": "ratio", "unit": "1"}
# Voxels of each subject's labels: label 1 is small in A only, label 2 large in both, label 3
# small in both, and B's segmentation has no label 4.
SIZES = {
    ("A", 1): 50,
    ("A", 2): 300,
    ("A", 3): 5,
    ("A", 4): 300,
    ("B", 1): 200,
    ("B", 2): 300,
    ("B", 3): 20,
}


def signal_frame() -> pd.DataFrame:
    rows = [
        {
            "subject": subject,
            "time_point": t,
            "label": label,
            "median": 1.0 + t / 100,
            "mean": 1.1 + t / 100,
            "n_voxels": n,
            "n_valid": n,
            "n_rejected": 0,
            "voxel_volume_mm3": 1.0,
        }
        for (subject, label), n in SIZES.items()
        for t in (2, 24)
    ]
    frame = pd.DataFrame(rows)
    # An invalid ROI in one scan, as gmri preprocess marks it: B's label 2 at 24 h.
    invalid = (
        (frame["subject"] == "B") & (frame["label"] == 2) & (frame["time_point"] == 24)
    )
    frame.loc[invalid, ["median", "mean", "n_valid"]] = [np.nan, np.nan, 0]
    return frame


def rows(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    return set(zip(frame["subject"], frame["label"], frame["time_point"]))


def test_keeps_only_labels_every_subject_has_after_small_and_nan_rows() -> None:
    kept, _, _ = clean_roi_signal(signal_frame(), 100)
    # label 1: small in A; label 3: small in both; label 4: not in B. Label 2 stays, without
    # B's NaN scan.
    assert rows(kept) == {("A", 2, 2), ("A", 2, 24), ("B", 2, 2)}
    assert kept["median"].notna().all()
    expected = signal_frame().loc[lambda f: (f["label"] == 2) & f["median"].notna()]
    pd.testing.assert_frame_equal(
        kept.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_every_removed_row_is_reported_with_its_reason() -> None:
    _, entries, _ = clean_roi_signal(signal_frame(), 100)
    assert entries.to_dict("records") == [
        {"subject": "A", "label": 1, "n_voxels": 50, "reason": "small", "time_points": "2 24"},
        {"subject": "A", "label": 3, "n_voxels": 5, "reason": "small", "time_points": "2 24"},
        {
            "subject": "A", "label": 4, "n_voxels": 300, "reason": "incomplete",
            "time_points": "2 24",
        },
        {
            "subject": "B", "label": 1, "n_voxels": 200, "reason": "incomplete",
            "time_points": "2 24",
        },
        {"subject": "B", "label": 2, "n_voxels": 300, "reason": "nan", "time_points": "24"},
        {"subject": "B", "label": 3, "n_voxels": 20, "reason": "small", "time_points": "2 24"},
    ]  # fmt: skip
    removed = sum(len(t.split()) for t in entries["time_points"])
    assert removed == len(signal_frame()) - 3


def test_removed_labels_name_the_subjects_without_them() -> None:
    _, _, labels = clean_roi_signal(signal_frame(), 100)
    assert labels.to_dict("records") == [
        {"label": 1, "n_subjects_without": 1, "subjects_without": "A"},
        {"label": 3, "n_subjects_without": 2, "subjects_without": "A B"},
        {"label": 4, "n_subjects_without": 1, "subjects_without": "B"},
    ]


def test_a_subject_keeps_a_label_with_one_valid_scan() -> None:
    frame = signal_frame()
    nan_in_a = (
        (frame["subject"] == "A") & (frame["label"] == 2) & (frame["time_point"] == 2)
    )
    frame.loc[nan_in_a, ["median", "mean"]] = np.nan
    kept, _, labels = clean_roi_signal(frame, 100)
    assert rows(kept) == {("A", 2, 24), ("B", 2, 2)}
    assert 2 not in set(labels["label"])


def test_threshold_is_inclusive_and_zero_keeps_every_size() -> None:
    kept, _, _ = clean_roi_signal(signal_frame(), 50)
    assert {label for _, label, _ in rows(kept)} == {1, 2}
    kept, entries, labels = clean_roi_signal(signal_frame(), 0)
    assert {label for _, label, _ in rows(kept)} == {1, 2, 3}
    assert set(entries["reason"]) == {"nan", "incomplete"}
    assert labels["label"].tolist() == [4]
    with pytest.raises(ValueError, match="min_voxels"):
        clean_roi_signal(signal_frame(), -1)


def run_script(
    source: Path,
    output: Path,
    data: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "code/preprocessing/clean_roi_signal.py"),
            "--input",
            str(source),
            "--min-voxels",
            "100",
            "--output",
            str(output),
            "--report",
            str(data / "removed_rois.csv"),
            "--label-report",
            str(data / "removed_labels.csv"),
        ],  # fmt: skip
        env={**os.environ, "PYTHONPATH": str(REPO / "code")},
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_writes_parquet_csv_and_reports(tmp_path: Path) -> None:
    source = tmp_path / "roi_signal.parquet"
    table = pa.Table.from_pandas(signal_frame(), preserve_index=False)
    pq.write_table(table.replace_schema_metadata(METADATA), source)
    output = tmp_path / "roi_signal_min100voxels.parquet"

    result = run_script(source, output, tmp_path)
    assert result.returncode == 0, result.stderr
    assert read_signal_metadata(output)["input_type"] == "T1w"
    assert pq.read_schema(output).metadata[b"min_voxels"] == b"100"
    assert pq.read_schema(output).types == pq.read_schema(source).types
    assert rows(pd.read_parquet(output)) == {("A", 2, 2), ("A", 2, 24), ("B", 2, 2)}
    csv = pd.read_csv(tmp_path / "roi_signal_min100voxels.csv")
    pd.testing.assert_frame_equal(csv, pd.read_parquet(output), check_dtype=False)
    assert len(pd.read_csv(tmp_path / "removed_rois.csv")) == 6
    assert pd.read_csv(tmp_path / "removed_labels.csv")["label"].tolist() == [1, 3, 4]
    out = result.stdout
    assert "3 subject x label entries (of 7) with fewer than 100 voxels removed" in out
    assert "1 more row (subject, time point, label) with NaN values removed" in out
    assert (
        "3 labels some subject lacks after that, removed for every subject: 1, 3, 4"
        in out
    )
    assert "1 of 4 labels kept" in out


def test_cleaned_table_fits_with_no_invalid_scans(tmp_path: Path) -> None:
    """Labels 17 and 2 are small in some subjects (removed for everyone) and one scan has an
    invalid ROI (removed, imputed); with --max-invalid-fraction 0 gmri keeps every subject.
    """
    source = write_roi_signal(tmp_path / "roi_signal.parquet")
    table = pq.read_table(source)
    frame = table.to_pandas()
    frame["n_voxels"] = 500
    first = frame["subject"].isin(subject_ids()[::2])
    frame.loc[first & (frame["label"] == 17), "n_voxels"] = 50
    frame.loc[~first & (frame["label"] == 2), "n_voxels"] = 50
    invalid = (frame["subject"] == "PAT_003") & (frame["time_point"] == 8)
    frame.loc[invalid & (frame["label"] == 41), ["median", "mean", "n_valid"]] = [
        np.nan,
    ] * 2 + [0]
    pq.write_table(
        pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False),
        source,
    )
    output = tmp_path / "roi_signal_min100voxels.parquet"
    assert run_script(source, output, tmp_path).returncode == 0

    out = tmp_path / "fit"
    with pytest.warns(UserWarning, match="imputing"):
        status = main(
            [
                "decompose", "run", "--output-dir", str(out), "--input", str(output),
                "--method", "cp", "--ranks", "2", "--restarts", "2", "--max-iter", "300",
                "--tolerance", "1e-4", "--min-timepoints", "4", "--max-invalid-fraction", "0",
            ],  # fmt: skip
        )
    assert status == 0
    model = load_decomposition(out / "cp_rank_2/model.h5")
    assert sorted(model.subjects) == subject_ids()
    assert sorted(int(label) for label in model.labels) == [41, 10024]
