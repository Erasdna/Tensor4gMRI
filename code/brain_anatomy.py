import nibabel as nib
from pathlib import Path
import numpy as np
from gMRItensor.plotting.spatial_mode import plot_enhancement_with_background
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os

example_file_path = Path(str(os.environ.get("BASE_DATA_FOLDER")), "example_files")


def plot_seg_on_background(background, seg, slice, name, view, cmap):
    fig, ax = plt.subplots()

    vmin = 1
    vmax = len(cmap.colors)
    print(vmin, vmax)
    if view == "sagittal":
        plot_enhancement_with_background(
            ax,
            np.flip(np.rot90(background[slice], 1), 1),
            signal=np.flip(np.rot90(seg[slice], 1), 1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    elif view == "axial":
        plot_enhancement_with_background(
            ax,
            np.rot90(background[:, slice]),
            signal=np.rot90(seg[:, slice]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    elif view == "coronal":
        plot_enhancement_with_background(
            ax,
            np.rot90(background[..., slice], 1),
            signal=np.rot90(seg[..., slice], 1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        raise ValueError("Invalid view parameter")

    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(
        example_file_path / f"{name}.png",
        dpi=500,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )


csf_seg_path = example_file_path / "csf_seg.nii.gz"
parenchyma_seg_path = example_file_path / "parenchyma_seg.nii.gz"
T1w_path = example_file_path / "T1w.nii.gz"

csf_seg_nifti = nib.as_closest_canonical(nib.load(str(csf_seg_path)))
csf_seg = csf_seg_nifti.get_fdata()
parenchyma_seg_nifti = nib.as_closest_canonical(nib.load(str(parenchyma_seg_path)))
parenchyma_seg = parenchyma_seg_nifti.get_fdata()
T1w_nifti = nib.as_closest_canonical(nib.load(T1w_path))
T1w = T1w_nifti.get_fdata()


# Move ventricle tags to csf:
ventricles = [
    4,
    5,
    14,
    15,
    31,
    63,
    43,
    44,
    24,
]
csf_seg = np.where(np.isin(parenchyma_seg, ventricles), parenchyma_seg, csf_seg)
parenchyma_seg = np.where(np.isin(parenchyma_seg, ventricles), 0, parenchyma_seg)


def mask_seg(seg, labels, tag):
    return np.where(np.isin(seg, labels), tag, seg)


# Simplify CSF segmentation:
basal_cistern = [16, 7, 8, 46, 47]
csf_seg = mask_seg(csf_seg, basal_cistern, 1)
csf_seg = mask_seg(csf_seg, ventricles, 2)
cerebral_SAS = np.unique(csf_seg[csf_seg >= 1000])
csf_seg = mask_seg(csf_seg, cerebral_SAS, 3)

# Plot the simplified CSF spaces:
category_colors = ["limegreen", "red", "dodgerblue"]
cmap = mcolors.ListedColormap(category_colors)
sagittal_slice = 125
plot_seg_on_background(T1w, csf_seg, sagittal_slice, "sagittal_slice", "sagittal", cmap)

# CSF
seg2 = np.where(csf_seg > 0, 0, 0)
# Cortical grey matter
seg2 = np.where((parenchyma_seg > 1000) & (parenchyma_seg < 3000), 2, seg2)
# Insula
seg2 = np.where(
    (parenchyma_seg == 1035) | (parenchyma_seg == 2035),
    seg2.max() + 1,
    seg2,
)
# Amygdala
seg2 = np.where((parenchyma_seg == 54) | (parenchyma_seg == 18), seg2.max() + 1, seg2)
# Hippocampus
seg2 = np.where((parenchyma_seg == 53) | (parenchyma_seg == 17), seg2.max() + 1, seg2)
# Enthorinal
seg2 = np.where(
    (parenchyma_seg == 2006)
    | (parenchyma_seg == 1006)
    | (parenchyma_seg == 3006)
    | (parenchyma_seg == 4006),
    seg2.max() + 1,
    seg2,
)
# Caudate
seg2 = np.where((parenchyma_seg == 50) | (parenchyma_seg == 11), 7, seg2)

coronal_slice = 169
axial_slice = 127
category_colors = ["white", "dodgerblue", "yellow", "pink", "cyan", "red", "purple"]

# White matter
seg2 = np.where((parenchyma_seg > 3000), seg2.max() + 1, seg2)
category_colors += ["lightgrey"]
cmap = mcolors.ListedColormap(category_colors)


medial_temporal_lobe_seg = np.where(np.isin(seg2, [4, 5, 6]), seg2, 0)
plot_seg_on_background(
    T1w,
    medial_temporal_lobe_seg,
    axial_slice,
    "medial_temporal_lobe",
    "axial",
    cmap,
)


lut_path = Path(str(os.environ.get("BASE_DATA_FOLDER")), "FreeSurferColorLUT.txt")

column_names = ["ID", "LabelName", "R", "G", "B", "A"]
fs_labels = pd.read_csv(
    lut_path,
    sep=r"\s+",
    comment="#",
    header=None,
    names=column_names,
    engine="python",
)
labels = np.unique(parenchyma_seg)
fs_labels = fs_labels.query("ID in @labels")
print(fs_labels)

lobe_mapping = {
    "Frontal": [
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parstriangularis",
        "parsorbitalis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "precentral",
        "paracentral",
        "frontalpole",
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
    ],
    "Parietal": [
        "superiorparietal",
        "inferiorparietal",
        "supramarginal",
        "postcentral",
        "precuneus",
        "posteriorcingulate",
        "isthmuscingulate",
    ],
    "Temporal": [
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "fusiform",
        "transversetemporal",
        "entorhinal",
        "temporalpole",
        "parahippocampal",
    ],
    "Occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    "Insula": ["insula"],
}

fs_labels["tag"] = fs_labels["LabelName"].apply(lambda x: x.split("-")[-1])
fs_labels["ctx"] = fs_labels["LabelName"].apply(lambda x: x.split("-")[0] == "ctx")

lobe_seg = np.zeros_like(parenchyma_seg)
for lobe, names in lobe_mapping.items():
    lobe_labels = fs_labels.query("tag in @names and ctx")
    print(lobe_labels)
    lobe_seg = np.where(
        np.isin(parenchyma_seg, lobe_labels["ID"]),
        lobe_seg.max() + 1,
        lobe_seg,
    )

# Add cerebellar gm
lobe_seg = np.where(np.isin(parenchyma_seg, [8, 47]), lobe_seg.max() + 1, lobe_seg)

category_colors = ["limegreen", "red", "dodgerblue", "pink", "yellow", "orange"]
cmap = mcolors.ListedColormap(category_colors)
for view, slice in zip(
    ["sagittal", "axial", "coronal"],
    [sagittal_slice, axial_slice, coronal_slice],
):
    plot_seg_on_background(T1w, lobe_seg, slice, f"{view}_lobes", view, cmap)


deep_brain_seg = np.zeros_like(parenchyma_seg)
for str_list in [
    ["White-Matter", "WhiteMatter", "wm-"],
    ["CC_"],
    ["Caudate", "Putamen", "Pallidum", "Accumbens"],
    ["Thalamus", "DC", "Hippocampus", "Amygdala"],
]:
    fs_labels["cat"] = fs_labels["LabelName"].apply(
        lambda x: any(word in x for word in str_list),
    )
    ids = fs_labels.query("cat")["ID"].values

    print(ids)
    print(str_list)
    deep_brain_seg = np.where(
        np.isin(parenchyma_seg, ids),
        deep_brain_seg.max() + 1,
        deep_brain_seg,
    )

category_colors = ["white", "red", "dodgerblue", "pink", "yellow", "orange"]
cmap = mcolors.ListedColormap(category_colors)
for view, slice in zip(
    ["sagittal", "axial", "coronal"],
    [sagittal_slice, axial_slice, coronal_slice],
):
    plot_seg_on_background(T1w, deep_brain_seg, slice, f"{view}_deep", view, cmap)
