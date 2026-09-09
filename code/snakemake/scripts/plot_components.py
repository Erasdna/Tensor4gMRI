import pandas as pd
from snakemake.script import snakemake
from gMRItensor.plotting import plot_mode_grid
from pathlib import Path
import h5py
import numpy as np
import nibabel as nib

data = Path(snakemake.input.CP_h5)
parenchyma_seg_path = Path(snakemake.input.parenchyma_segmentation)
csf_seg_path = Path(snakemake.input.csf_segmentation)
background_path = Path(snakemake.input.T1w)
subject_labels = Path(snakemake.input.INFO)
group = snakemake.params.group
R = int(snakemake.params.Rs)

cm = 1 / 2.54
base_width_per_col = 18 * cm / 4
base_height_per_row = base_width_per_col * 0.9
# figsize = (18 * cm, R * 4 * cm * 1.1)
print(f"Plotting {R}-component model")
with h5py.File(data, "r") as f:
    # Load spatial info
    scaling = f["factors/scaling"][()]
    rois = f["rois"][()]
    spatial_mode = f["factors/roi_mode"][()]  # * scaling[:, None]

    # Load subject info
    subject_mode = f["factors/subject_mode"][()]
    subjects = f["subjects"][()].astype(str)

    # Load time info
    time_mode = f["factors/time_mode"][()]

    weights = f["factors/weights"][()]

print(weights / weights.max())
subject_info = pd.read_csv(subject_labels)
subject_info = subject_info.rename(columns={"new_id": "subjects"})
subject_info["Group"] = subject_info["Group"].map({0: "REF", 1: "iNPH"})
subject_info["Sex"] = subject_info["Sex"].map({0: "M", 1: "F"})
if group in ["M", "F", "all"]:
    category = "Group"
elif group in ["REF", "iNPH"]:
    category = "Sex"

# segmetation and background
background_nifti = nib.as_closest_canonical(nib.load(str(background_path)))
background_data = background_nifti.get_fdata()
csf_seg_nifti = nib.as_closest_canonical(nib.load(str(csf_seg_path)))
csf_seg = csf_seg_nifti.get_fdata()
parenchyma_seg_nifti = nib.as_closest_canonical(nib.load(str(parenchyma_seg_path)))
tissue_seg = parenchyma_seg_nifti.get_fdata()

brainmask = (csf_seg > 0) | (tissue_seg > 0)
index_list = np.argwhere(brainmask)
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

spatial_mode_voxel = np.zeros((len(index_list), spatial_mode.shape[1]))
for i, index in enumerate(index_list):
    roi = combined_seg[*index]
    if roi - 10000 in ventricles:
        roi -= 10000
    component_index = np.where(rois == int(roi))
    if len(component_index[0]) == 0:
        spatial_mode_voxel[i, :] = 0  # 1e-6
    else:
        spatial_mode_voxel[i, :] = spatial_mode[component_index[0], :]  # + 1e-6

savepath = Path(snakemake.params.save).joinpath(f"R_{R}", "components")

fig, ax = plot_mode_grid(
    spatial_mode_voxel,
    time_mode,
    subject_mode,
    index_list,
    csf_seg[*index_list.T] > 0,
    tissue_seg[*index_list.T] > 0,
    background_data,
    125,
    [2, 4, 8, 24],
    subjects,
    subject_info,
    category,
    page_width=19 * cm,
    share_colorbar_scaling=False,
)
fig.savefig(savepath.joinpath(f"{group}_mode_grid.pdf"), dpi=400)
fig.savefig(savepath.joinpath(f"{group}_mode_grid.png"), dpi=400)
fig.savefig(savepath.joinpath("done.png"), dpi=50)
