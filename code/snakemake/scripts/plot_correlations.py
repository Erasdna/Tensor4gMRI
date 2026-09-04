from snakemake.script import snakemake
from pathlib import Path
from gMRItensor.plotting import plot_subject_mode_correlation
import h5py
import pandas as pd

data = Path(snakemake.input.CP_h5)
subject_labels = Path(snakemake.input.INFO)
group = snakemake.params.group
R = int(snakemake.params.Rs)

cm = 1 / 2.54
figsize = (23 * cm, R * 5.2 * cm)
print(f"Plotting {R}-component model")
savepath = Path(snakemake.params.save).joinpath(f"R_{R}")
savepath.mkdir(exist_ok=True, parents=True)

with h5py.File(data, "r") as f:
    subject_mode = f["factors/subject_mode"][()]
    subjects = f["subjects"][()].astype(str)
subject_info = pd.read_csv(subject_labels)
subject_info = subject_info.rename(columns={"new_id": "subjects"})
subject_info["Group"] = subject_info["Group"].map({0: "REF", 1: "iNPH"})
subject_info["Sex"] = subject_info["Sex"].map({0: "M", 1: "F"})

print(subject_info)
print(subjects)
if group in ["M", "F", "all"]:
    category = "Group"
elif group in ["REF", "iNPH"]:
    category = "Sex"
fig, ax = plot_subject_mode_correlation(
    subject_mode,
    subjects,
    subject_info,
    category,
    page_width=25 * cm,
    width_to_height_ratio=1,
)
fig.savefig(
    savepath.joinpath(f"{group}_component_correlations.png"),
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    savepath.joinpath(f"{group}_component_correlations.pdf"),
    dpi=300,
    bbox_inches="tight",
)
fig.savefig(
    savepath.joinpath("component_correlations_done.png"),
    dpi=50,
    bbox_inches="tight",
)
