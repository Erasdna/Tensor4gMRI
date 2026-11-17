"""Brain volumes of each subject's merged segmentation, from gmri preprocess's per-label table:
ventricles, SAS (CSF without the ventricles), parenchyma and cortical grey matter, in mL and as a
share of the subject's total segmented volume. Regions as gMRI defines them (and as GRIP's
segmentation volumes)."""

import numpy as np
import pandas as pd
from gMRItensor.roi_groups import get_roi_presets, is_csf

VOLUMES = ("ventricles", "sas", "parenchyma", "cortical_grey_matter")
VOLUME_COLUMNS = [
    *(f"{name}_volume_{unit}" for name in VOLUMES for unit in ("ml", "fraction")),
    "total_volume_ml",
]


def subject_volumes(roi_signal: pd.DataFrame, csf_offset: int = 10000) -> pd.DataFrame:
    """One row per subject of `roi_signal` (columns `subject, label, n_voxels,
    voxel_volume_mm3`; one row per time point and label): `subjects` and VOLUME_COLUMNS.

    A label's volume is `n_voxels * voxel_volume_mm3`, every voxel of it in the subject's
    segmentation. CSF is gMRI's `is_csf` (a ventricle, FreeSurfer CSF 24, or an id from
    `csf_offset`), the ventricles its `ventricles` preset, SAS the CSF without them, parenchyma
    every other label and cortical grey matter its `cortical_grey_matter` preset. Fractions are
    shares of the total. Raises `ValueError` if a subject's label has different voxel counts at
    different time points (one segmentation per subject)."""
    per_label = roi_signal.groupby(["subject", "label"]).agg(
        n_voxels=("n_voxels", "first"),
        changing=("n_voxels", "nunique"),
        voxel_volume_mm3=("voxel_volume_mm3", "first"),
    )
    changing = sorted(
        set(per_label.index[per_label["changing"] > 1].get_level_values(0)),
    )
    if changing:
        raise ValueError(
            f"Voxel counts differ between time points for subjects {changing}",
        )
    per_label = per_label.reset_index()
    labels = per_label["label"].to_numpy(int)
    presets = get_roi_presets(csf_offset)
    csf = is_csf(labels, csf_offset)
    ventricles = np.isin(labels, presets["ventricles"])
    masks = {
        "ventricles": ventricles,
        "sas": csf & ~ventricles,
        "parenchyma": ~csf,
        "cortical_grey_matter": np.isin(labels, presets["cortical_grey_matter"]),
    }
    ml = per_label["n_voxels"] * per_label["voxel_volume_mm3"] / 1000
    by_subject = per_label["subject"]
    total = ml.groupby(by_subject).sum()
    volumes = pd.DataFrame(index=total.index)
    for name in VOLUMES:
        region = ml.where(masks[name], 0.0).groupby(by_subject).sum()
        volumes[f"{name}_volume_ml"] = region
        volumes[f"{name}_volume_fraction"] = region / total
    volumes["total_volume_ml"] = total
    return (
        volumes.rename_axis("subjects")
        .reset_index()
        .loc[:, ["subjects", *VOLUME_COLUMNS]]
    )
