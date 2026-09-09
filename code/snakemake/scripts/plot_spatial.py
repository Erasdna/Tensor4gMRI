import h5py
from snakemake.script import snakemake
from pathlib import Path
import nibabel as nib
from gMRItensor import plotting

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
    rois = f["rois"][()]
    spatial_mode = f["factors/roi_mode"][()]

background_nifti = nib.as_closest_canonical(nib.load(str(background_path)))
background_data = background_nifti.get_fdata()
csf_seg_nifti = nib.as_closest_canonical(nib.load(str(csf_seg_path)))
csf_seg = csf_seg_nifti.get_fdata()
parenchyma_seg_nifti = nib.as_closest_canonical(nib.load(str(parenchyma_seg_path)))
tissue_seg = parenchyma_seg_nifti.get_fdata()
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
    {"CSF": csf_seg, "Parenchyma": tissue_seg},
    override,
    {"CSF": 10000, "Parenchyma": 0},
)
values, index_list = plotting.utils.expand_roi_mode_to_voxels(
    spatial_mode,
    rois,
    merged,
)

for fig, ax, name in plotting.plot_spatial_mode(
    values,
    index_list,
    plotting.utils.region_masks_from_segmentations(index_list, segmentations=segs),
    background_data,
    [125, 127, 169],
    page_width=9 * cm,
    width_to_height_ratio=1,
    share_colorbar_scaling=False,
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
