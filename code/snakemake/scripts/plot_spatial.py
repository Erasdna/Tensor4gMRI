import h5py
from snakemake.script import snakemake
from pathlib import Path
from gMRItensor.plotting import plot_spatial_mode
import nibabel as nib
import numpy as np

data = Path(snakemake.input.CP_h5)
parenchyma_seg_path = Path(snakemake.input.parenchyma_segmentation)
csf_seg_path = Path(snakemake.input.csf_segmentation)
background_path = Path(snakemake.input.T1w)
R = int(snakemake.params.Rs)

print(f"Plotting spatial modes for {R}-component model")
cm = 1 / 2.54
# figsize = (9 * cm, 3 * cm * R)
savepath = Path(snakemake.params.save).joinpath(f"R_{R}", "spatial")
savepath.mkdir(exist_ok=True, parents=True)

with h5py.File(data, "r") as f:
    scaling = f["factors/scaling"][()]
    rois = f["rois"][()]
    spatial_mode = f["factors/roi_mode"][()]  # * scaling[:, None]

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

for fig, ax, name in plot_spatial_mode(
    spatial_mode_voxel,
    index_list,
    csf_seg[*index_list.T] > 0,
    tissue_seg[*index_list.T] > 0,
    background_data,
    [125, 127, 169],
    page_width=9 * cm,
    width_to_height_ratio=1,
    # figsize,
):
    seg_save_path = savepath.joinpath("tissue" if name == "Parenchyma" else "csf")
    seg_save_path.mkdir(parents=True, exist_ok=True)

    fig.suptitle("Spatial mode: " + name)
    fig.savefig(
        seg_save_path.joinpath("spatial.png"),
        dpi=400,
        bbox_inches="tight",
    )
    fig.savefig(seg_save_path.parent.joinpath("done.png"), dpi=10)
