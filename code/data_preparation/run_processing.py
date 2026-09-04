from gMRItensor.preprocessing import compute_tracer_parallel
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse tracer data and compute signal increase",
    )
    parser.add_argument("files", type=Path, help="Path to csv with cases to run")
    parser.add_argument("labels", type=Path, help="Path to data labels")
    parser.add_argument("save", type=Path, help="Path to save file")
    args = parser.parse_args()

    files = pd.read_csv(args.files)
    files["func"] = np.nanmedian
    files["signal_type"] = "T1w"

    slurm_procs = os.environ.get("SLURM_CPUS_PER_TASK")
    n_procs = slurm_procs if slurm_procs is not None else 1
    out = compute_tracer_parallel(files.to_dict(orient="records"), n_procs=n_procs)

    merge_df = pd.read_csv(args.labels)
    merge_df = merge_df.rename(columns={"new_id": "subject"})
    merge_df["Group"] = merge_df["Group"].map({0: "REF", 1: "iNPH"})
    merge_df["Sex"] = merge_df["Sex"].map({0: "M", 1: "F"})
    merge_df = merge_df[merge_df["exclude"] == 0]
    out = pd.merge(out, merge_df, on="subject", how="inner")
    out.to_csv(args.save, index=False)
    print(np.isnan(out["values"]).sum())
