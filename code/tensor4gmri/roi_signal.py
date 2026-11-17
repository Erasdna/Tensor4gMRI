"""The per-label signal of gmri preprocess (roi_signal.parquet), cleaned for the fits: without
unreliably small ROIs, invalid values and labels some subject lacks."""

import numpy as np
import pandas as pd

ENTRY_COLUMNS = ["subject", "label", "n_voxels", "reason", "time_points"]
LABEL_COLUMNS = ["label", "n_subjects_without", "subjects_without"]


def _entries(rows: pd.DataFrame, reason: str) -> pd.DataFrame:
    """One row per subject and label of `rows`: its voxel count and time points."""
    grouped = rows.sort_values("time_point").groupby(["subject", "label"], sort=False)
    entries = grouped.agg(
        n_voxels=("n_voxels", "first"),
        time_points=("time_point", lambda t: " ".join(map(str, t))),
    )
    return entries.reset_index().assign(reason=reason)


def clean_roi_signal(
    frame: pd.DataFrame,
    min_voxels: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`frame` (a roi_signal table) cleaned in three steps:

    1. "small": the rows of every subject's label of fewer than `min_voxels` voxels (the
       subject's own count, so every time point) are removed;
    2. "nan": the rows whose `median` or `mean` is not finite (an invalid ROI in that scan);
    3. "incomplete": a label that some subject of `frame` has no row of left is removed for
       every subject (absent from its segmentation, small in it, or NaN in all its scans).

    Every label left has rows for every subject; gmri's CP fit treats a scan's missing row (step
    2) as missing and imputes it, without counting it toward `--max-invalid-fraction`.

    Returns the remaining rows, the removed entries (one row per subject, label and reason, with
    the voxel count and the removed time points, space-separated) and the removed labels of
    step 3 (`label, n_subjects_without, subjects_without`). Raises `ValueError` if
    `min_voxels` < 0 (0: no ROI is too small).
    """
    if min_voxels < 0:
        raise ValueError(f"min_voxels must be at least 0, not {min_voxels}")
    small = frame["n_voxels"] < min_voxels
    invalid = ~(np.isfinite(frame["median"]) & np.isfinite(frame["mean"])) & ~small
    remaining = frame[~small & ~invalid]

    subjects = set(frame["subject"])
    holders = remaining.groupby("label")["subject"].agg(set)
    complete = holders.index[holders.map(len) == len(subjects)]
    incomplete = ~remaining["label"].isin(complete)
    kept = remaining[~incomplete]

    entries = pd.concat(
        [
            _entries(frame[small], "small"),
            _entries(frame[invalid], "nan"),
            _entries(remaining[incomplete], "incomplete"),
        ],
        ignore_index=True,
    ).sort_values(["subject", "label", "reason"], ignore_index=True)

    removed = sorted(set(frame["label"]) - set(complete))
    without = [sorted(subjects - holders.get(label, set())) for label in removed]
    labels = pd.DataFrame(
        {
            "label": removed,
            "n_subjects_without": [len(s) for s in without],
            "subjects_without": [" ".join(s) for s in without],
        },
        columns=LABEL_COLUMNS,
    )
    return kept, entries.loc[:, ENTRY_COLUMNS], labels
