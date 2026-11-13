from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

LABELS = (2, 17, 41, 10024)
TIME_POINTS = (2, 4, 8, 24)
N_SUBJECTS = 8
# Scan times (hhmmss) of the first day: baseline, then 2, 4 and 8 hours after it.
DAY_ONE = ("083000", "103000", "123000", "163000")


def subject_ids(n: int = N_SUBJECTS) -> list[str]:
    return [f"PAT_{s:03d}" for s in range(n)]


def labels_frame(n: int = N_SUBJECTS) -> pd.DataFrame:
    """$SUBJECT_LABELS: half REF (0) and half iNPH (1), alternating sex (0 M, 1 F)."""
    return pd.DataFrame(
        {
            "new_id": subject_ids(n),
            "old_id": [f"{100 + s}" for s in range(n)],
            "Group": [s * 2 // n for s in range(n)],
            "Sex": [s % 2 for s in range(n)],
            "exclude": [0] * n,
            "Age": [60 + s for s in range(n)],
            "Height": [170 + s for s in range(n)],
            "Weight": [70 + s for s in range(n)],
        },
    )


def subject_info_frame(n: int = N_SUBJECTS) -> pd.DataFrame:
    """subject_info.csv as 00_find_files.sh writes it from labels_frame."""
    labels = labels_frame(n)
    group = labels["Group"].map({0: "REF", 1: "iNPH"})
    sex = labels["Sex"].map({0: "M", 1: "F"})
    return pd.DataFrame(
        {
            "subjects": labels["new_id"],
            "old_id": labels["old_id"],
            "Group": group,
            "Sex": sex,
            "exclude": labels["exclude"],
            "Age": labels["Age"],
            "Height": labels["Height"],
            "Weight": labels["Weight"],
            "Stratum": group + "_" + sex,
        },
    )


def roi_signal_frame(n: int = N_SUBJECTS, seed: int = 0) -> pd.DataFrame:
    """A `gmri preprocess` roi_signal table of a rank-2 signal plus noise."""
    rng = np.random.default_rng(seed)
    subject_mode = rng.random((n, 2)) + 0.5
    label_mode = rng.random((len(LABELS), 2)) + 0.1
    rows = []
    for s, subject in enumerate(subject_ids(n)):
        for t in TIME_POINTS:
            time = np.array([np.exp(-t / 10), t / 24])
            values = (subject_mode[s] * time) @ label_mode.T + 0.01 * rng.random(
                len(LABELS),
            )
            for label, value in zip(LABELS, values):
                rows.append(
                    {
                        "subject": subject,
                        "time_point": t,
                        "label": label,
                        "median": value,
                        "mean": value,
                        "n_voxels": 10,
                        "n_valid": 10,
                        "n_rejected": 0,
                        "voxel_volume_mm3": 1.0,
                    },
                )
    return pd.DataFrame(rows)


def write_roi_signal(path: Path, n: int = N_SUBJECTS) -> Path:
    """roi_signal_frame with the metadata `gmri preprocess` writes for T1w input."""
    table = pa.Table.from_pandas(roi_signal_frame(n), preserve_index=False)
    metadata = {
        "kind": "roi_signal",
        "input_type": "T1w",
        "signal": "ratio",
        "unit": "1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


def _save(data: np.ndarray, path: Path) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


def write_raw_subject(folder: Path, subject: str, second_day: bool = True) -> None:
    """One subject's normalized T1w scans (day 1 and, unless not second_day, day 2) and its
    SAS and parenchyma segmentations, named as in $RAW_TRACER_DATA_FOLDER."""
    folder.mkdir(parents=True)
    shape = (6, 6, 6)
    parenchyma = np.zeros(shape, dtype=np.int16)
    parenchyma[1:3, 1:5, 1:5] = 2
    parenchyma[3:5, 1:5, 1:5] = 41
    parenchyma[2:4, 2:4, 2:4] = 17
    sas = np.zeros(shape, dtype=np.int16)
    sas[0, :, :] = 24
    _save(parenchyma, folder / f"{subject}-refroi.nii.gz")
    _save(sas, folder / f"{subject}-sasparc.nii.gz")
    scans = [(f"20170101_{t}", 1.0 + 0.1 * k) for k, t in enumerate(DAY_ONE)]
    if second_day:
        scans.append(("20170102_083000", 1.5))
    for stamp, factor in scans:
        image = np.full(shape, 100.0 * factor, dtype=np.float32)
        _save(image, folder / f"{subject}-T1w-normalized-{stamp}.nii.gz")


@pytest.fixture
def raw_data(tmp_path: Path) -> tuple[Path, Path]:
    """$RAW_TRACER_DATA_FOLDER with every subject of labels_frame, and $SUBJECT_LABELS."""
    raw = tmp_path / "raw"
    for subject in subject_ids():
        write_raw_subject(raw / subject, subject)
    labels = tmp_path / "labels.csv"
    labels_frame().to_csv(labels, index=False)
    return raw, labels


@pytest.fixture
def subject_info_path(tmp_path: Path) -> Path:
    path = tmp_path / "subject_info.csv"
    subject_info_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def roi_signal_path(tmp_path: Path) -> Path:
    return write_roi_signal(tmp_path / "roi_signal.parquet")
