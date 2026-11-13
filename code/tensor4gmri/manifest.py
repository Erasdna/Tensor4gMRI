"""Input for `gmri preprocess`: the image manifest."""

from pathlib import Path

import pandas as pd

from tensor4gmri.config import TIME_POINTS

MANIFEST_COLUMNS: tuple[str, ...] = (
    "subject",
    "time_point",
    "baseline_path",
    "post_injection_path",
    "mask_path",
    "segmentation_path",
)
PATH_COLUMNS: tuple[str, ...] = MANIFEST_COLUMNS[2:]


def image_manifest(file_list: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """`gmri preprocess --manifest` rows of a `files_to_process` table.

    Relative paths are resolved against `base_dir`. Raises `ValueError` if the table is empty
    or has a time point other than TIME_POINTS.
    """
    if file_list.empty:
        raise ValueError("No scans in the file list")
    other = sorted(set(file_list["time_point"]) - set(TIME_POINTS))
    if other:
        raise ValueError(f"Time points other than {list(TIME_POINTS)} h: {other}")
    manifest = file_list.copy()
    for column in PATH_COLUMNS:
        manifest[column] = [
            str((base_dir / Path(p)).resolve()) for p in manifest[column]
        ]
    return manifest.loc[:, list(MANIFEST_COLUMNS)].reset_index(drop=True)
