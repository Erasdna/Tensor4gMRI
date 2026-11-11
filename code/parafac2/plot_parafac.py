import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import nibabel as nib
import h5py
from gMRItensor.plotting.evolving_mode import (
    reconstruct_evolving_factors,
    plot_evolving_mode,
)
from gMRItensor.plotting import plot_subject_mode
from gMRItensor import plotting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse tracer data and identify files")
    parser.add_argument(
        "decomposition",
        type=Path,
        help="Path to file containing decomposition",
    )
    parser.add_argument("subject_info", type=Path, help="Path to subject info file")
    parser.add_argument("csf", type=Path, help="Path to csf segmentation")
    parser.add_argument("parenchyma", type=Path, help="Path to parenchyma segmentation")
    parser.add_argument("background", type=Path, help="Path to T1w background")
    parser.add_argument("save", type=Path, help="Path to save file folder")

    args = parser.parse_args()

    with h5py.File(args.decomposition, "r") as f:
        fac = f["factors"]
        subject_mode = fac["subject_mode"][()]
        time_mode = fac["time_mode"][()]
        roi_mode = fac["roi_mode"][()]
        projections = fac["projections"][()]
        evolving_modes = reconstruct_evolving_factors(
            np.ones(time_mode.shape[-1]),
            (subject_mode, time_mode, roi_mode),
            projections,
        )
        subjects = f["subjects"][()].astype(str)
        time_points = f["time_points"][()]
        rois = f["rois"][()]

    cm = 1 / 2.54

    info = pd.read_csv(args.subject_info)
    info = info.rename(columns={"new_id": "subjects"})
    info["Group"] = info["Group"].map({0: "REF", 1: "iNPH"})
    print(info)
    fig, ax, sig = plot_evolving_mode(
        evolving_modes,
        [time_points] * len(subjects),
        subjects,
        info,
        group_variable="Group",
    )
    print(sig)

    save = args.save
    save.mkdir(exist_ok=True, parents=True)
    fig.savefig(save / "time.png", dpi=400)

    fig, ax = plot_subject_mode(
        subject_mode,
        subjects,
        info,
        "Group",
        ["Age", "Height", "Weight"],
        page_width=25 * cm,
    )
    fig.savefig(save / "subject.png", dpi=400)

    csf_seg = nib.as_closest_canonical(nib.load(args.csf)).get_fdata()
    parenchyma_seg = nib.as_closest_canonical(nib.load(args.parenchyma)).get_fdata()
    background = nib.as_closest_canonical(nib.load(args.background)).get_fdata()

    ventricles = [
        4,
        5,
        14,
        15,
        43,
        44,
        24,
    ]
    override = {}
    for el in ventricles:
        override[el] = "CSF"

    merged, segs, _ = plotting.utils.merge_segmentations(
        {"CSF": csf_seg, "Parenchyma": parenchyma_seg},
        override,
        {"CSF": 10000, "Parenchyma": 0},
    )

    print(segs.keys())

    values, index_list = plotting.utils.expand_roi_mode_to_voxels(
        roi_mode,
        rois,
        merged,
    )

    print(values.shape)

    print(segs["CSF"][*index_list.T] > 0)
    it = plotting.plot_spatial_mode(
        values,
        index_list,
        plotting.utils.region_masks_from_segmentations(index_list, segmentations=segs),
        background,
        [125, 127, 169],
        page_width=9 * cm,
        width_to_height_ratio=1,
    )
    for fig, ax, name in it:
        fig.savefig(save / f"{name}_rois.png", dpi=400)
