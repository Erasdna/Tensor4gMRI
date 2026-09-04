from snakemake.script import snakemake
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

Rs = snakemake.params.Rs

# Read subjects
h5_path = Path(snakemake.input.CP_h5)
with h5py.File(h5_path, "r") as f:
    subjects = f["subjects"][()].astype(str)

savepath = Path(snakemake.params.save)

# Read labels
labels = pd.read_csv(snakemake.input.INFO)
labels = labels[labels["old_id"].isin(subjects)]
labels_df = labels[["old_id", "new_id", "Group", "Sex"]]
for R in Rs:
    with h5py.File(h5_path, "r") as f:
        subject_mode = f[str(Path(f"{R}/best_factors/subjects"))][()]

    df_dict = {"old_id": subjects}
    # Sort subjects by components in ascending order (smallest on top)
    for r in range(R):
        ids = np.argsort(subject_mode[:, r])
        print(subject_mode.shape)
        df_dict[f"component_{r+1}_id"] = ids
        df_dict[f"component_{r+1}_value"] = subject_mode[:, r]
        print(ids)
        print(subject_mode[ids, r])
        print(subject_mode[:, r])

    sorted_df = pd.merge(labels_df, pd.DataFrame(df_dict), on="old_id")
    sorted_df.to_csv(savepath.joinpath(f"R_{R}/subject/sorted.csv"))
