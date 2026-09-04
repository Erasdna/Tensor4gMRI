from gMRItensor.plotting import plot_spatial_mode
import numpy as np
import h5py
from pathlib import Path
import argparse
import nibabel as nib
import pandas as pd
from gMRItensor.preprocessing import prepare_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute average subject in a subset of the dataset",
    )
    parser.add_argument("model", type=Path, help="Path to h5py file with model weights")
    parser.add_argument(
        "data",
        type=Path,
        help="Path to csv file with the initial data",
    )
    parser.add_argument(
        "example_files",
        type=Path,
        help="Path to example files folder with segmentations",
    )
    parser.add_argument(
        "savepath",
        type=Path,
        help="Path to save output",
    )
    args = parser.parse_args()

    with h5py.File(args.model, "r") as f:
        subjects = f["subjects"][()].astype(str)
        scaling = f["factors/scaling"][()]

    group = args.model.name.split(".")[0].split("_")[-1]
    print(group)
    data = pd.read_csv(args.data).query("subject in @subjects")
    data, subjects, time_points, regions = prepare_tensor(
        data,
        ("Sex" if group in ["M", "F"] else "Group", group),
    )
    # Scale by standard deviation only
    print(data.shape)
    std_scaling = np.nanstd(data, axis=(0, 1), keepdims=False)
    print(std_scaling)
    data = data / std_scaling

    avg_data = (
        np.nanmean(data, axis=(0, 1), keepdims=False) / std_scaling
    )  # np.nanmean(data, axis=(0, 1), keepdims=False)

    print(avg_data)

    tissue_seg = nib.as_closest_canonical(
        nib.load(args.example_files / "parenchyma_seg.nii.gz"),
    ).get_fdata()
    csf_seg = nib.as_closest_canonical(
        nib.load(args.example_files / "csf_seg.nii.gz"),
    ).get_fdata()
    T1w_data = nib.as_closest_canonical(
        nib.load(args.example_files / "T1w.nii.gz"),
    ).get_fdata()

    ventricles = [
        4,
        5,
        14,
        15,
        43,
        44,
        24,
    ]

    # Transfer ventricles to csf segmentation
    csf_seg = np.where(np.isin(tissue_seg, ventricles), tissue_seg, csf_seg)
    tissue_seg = np.where(np.isin(tissue_seg, ventricles), 0, tissue_seg)
    combined_seg = np.where(csf_seg > 0, csf_seg + 10000, tissue_seg)

    brainmask = (csf_seg > 0) | (tissue_seg > 0)
    index_list = np.argwhere(brainmask)

    spatial_mode_voxel = np.zeros((len(index_list), 1))
    for i, index in enumerate(index_list):
        roi = combined_seg[*index]
        if roi - 10000 in ventricles:
            roi -= 10000
        component_index = np.where(regions == int(roi))
        if len(component_index[0]) == 0:
            spatial_mode_voxel[i, 0] = 0  # 1e-6
        else:
            spatial_mode_voxel[i, 0] = avg_data[component_index[0]][0]  # + 1e-6

    cm = 1 / 2.54
    for fig, ax, name in plot_spatial_mode(
        spatial_mode_voxel,
        index_list,
        csf_seg[*index_list.T] > 0,
        tissue_seg[*index_list.T] > 0,
        T1w_data,
        [125, 127, 169],
        page_width=9 * cm,
        width_to_height_ratio=1,
    ):
        seg_save_path = args.savepath.joinpath(
            "tissue" if name == "Parenchyma" else "csf",
        )
        seg_save_path.mkdir(parents=True, exist_ok=True)

        fig.suptitle(name)
        fig.savefig(
            seg_save_path.joinpath("spatial.png"),
            dpi=400,
            bbox_inches="tight",
        )
