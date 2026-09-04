import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib
from gMRItensor.plotting import plot_enhancement_with_background
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def make_colorbar(fig, ax, im, cax, label):
    cbar = fig.colorbar(
        im,
        cax=cax,
        ax=ax,
    )
    cbar.set_label(label=label)


subjects: dict[str, list[str]] = {
    "subject": [
        str(os.environ.get("SUB_iNPH_MEDIAN")),
        str(os.environ.get("SUB_REF_MEDIAN")),
        str(os.environ.get("SUB_REF_OUTLIER")),
    ],
    "names": ["iNPH median", "REF median", "REF extreme"],
}
subjects = pd.DataFrame(subjects)


enrichment_data = pd.read_csv(Path(str(os.environ.get("BASE_DATA_FOLDER")), "data.csv"))

enrichment_data = pd.merge(subjects, enrichment_data, on="subject").set_index(
    ["subject", "names"],
)
print(enrichment_data)
enrichment_data = enrichment_data.dropna()

# Load segmentations
segmentations: dict = {
    "new_id": [],
    "csf": [],
    "parenchyma": [],
    "background": [],
    "slice": [],
    "name": [],
}
subjects_path = Path(str(os.environ.get("RAW_TRACER_DATA_FOLDER")))
for subject_name, new_id in zip(
    subjects["names"],
    subjects["subject"],
):
    segmentations["new_id"].append(new_id)
    for name, suffix in zip(
        ["csf", "parenchyma"],
        ["-sasparc.nii.gz", "-refroi.nii.gz"],
    ):
        segmentations[name].append(
            nib.load(subjects_path.joinpath(new_id, f"{new_id}{suffix}")).get_fdata(),
        )
    segmentations["slice"].append(125)
    segmentations["name"].append(subject_name)
    segmentations["background"].append(
        nib.load(
            str(list(subjects_path.joinpath(new_id).glob("*normalized*"))[0]),
        ).get_fdata(),
    )

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

cm = 1 / 2.54
figsize = (18 * cm, 12.5 * cm)

superfig = plt.figure(figsize=figsize, layout="constrained")
subfigures = superfig.subfigures(1, 2, wspace=0.07)
csf_fig = subfigures[0]
parenchyma_fig = subfigures[1]
csf_ax = csf_fig.subplots(4, 4, gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
parenchyma_ax = parenchyma_fig.subplots(
    4,
    4,
    gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
)


time_points = enrichment_data["time_point"].unique()
for t in range(4):
    time_data = enrichment_data.query(f"time_point=={time_points[t]}")
    mi = 1.0
    csf_data = time_data.query("labels>10000 | labels in @ventricles")
    parenchyma_data = time_data.query("labels<10000")

    csf_ma = np.minimum(np.percentile(csf_data["values"], 90), 12)
    parenchyma_ma = np.minimum(
        np.percentile(parenchyma_data["values"], 90),
        3,
    )
    print(csf_ma, parenchyma_ma)
    cmap = plt.colormaps["plasma"]

    for i, (background, csf, parenchyma, new_id, slice, name) in enumerate(
        zip(
            segmentations["background"],
            segmentations["csf"],
            segmentations["parenchyma"],
            segmentations["new_id"],
            segmentations["slice"],
            segmentations["name"],
        ),
    ):
        print(t, new_id)
        sub_data = time_data.query(f"subject=='{new_id}'")
        csf_filled = np.zeros_like(background)
        parenchyma_filled = np.zeros_like(background)
        combined_seg = np.where(csf > 0, csf + 10000, parenchyma)

        for df_data, filled in zip(
            [parenchyma_data, csf_data],
            [parenchyma_filled, csf_filled],
        ):
            sub_data = df_data.query(f"subject=='{new_id}'")
            for row in sub_data.itertuples():
                filled = np.where(
                    np.isclose(combined_seg, row.labels),
                    row.values,
                    filled,
                )

        def plot_instance(ax_object, foreground, cbar_max, ylabel):
            cax = plot_enhancement_with_background(
                ax_object[t, i],
                np.flip(background[slice], 1),
                np.flip(foreground[slice], 1),
                cmap,
                mi,
                cbar_max,
            )
            ax_object[t, i].set_xticks([])
            ax_object[t, i].set_yticks([])

            if i == 0 and ylabel:
                ax_object[t, i].set_ylabel(f"{time_points[t]} hours")
            if t == 0:
                ax_object[t, i].set_title(name)

            return cax

        cax_csf = plot_instance(csf_ax, csf_filled, csf_ma, ylabel=True)
        cax_parenchyma = plot_instance(
            parenchyma_ax,
            parenchyma_filled,
            parenchyma_ma,
            ylabel=True,
        )

    for fig, ax, im, title in zip(
        [csf_fig, parenchyma_fig],
        [csf_ax[t, -1], parenchyma_ax[t, -1]],
        [cax_csf, cax_parenchyma],
        ["Signal increase: CSF", "Signal increase: Parenchyma"],
    ):
        ax.set_axis_off()
        cax = inset_axes(ax, width="100%", height="80%", loc="center right")
        make_colorbar(fig, ax, im, cax, "Signal increase")
        fig.suptitle(title, fontsize=10)

superfig.savefig(subjects_path / "outlier.png", dpi=300, bbox_inches="tight")
superfig.savefig(subjects_path / "outlier.pdf", dpi=300, bbox_inches="tight")
