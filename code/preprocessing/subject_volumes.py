"""Write each subject's brain volumes (tensor4gmri.volumes: ventricles, SAS, parenchyma and
cortical grey matter, in mL and as a share of the total segmented volume) from gmri preprocess's
roi_signal.parquet to --output, and add them to the subject table (replacing earlier volume
columns; empty for subjects without data), where 05_subject_stats.sh tests them."""

import argparse
from pathlib import Path

import pandas as pd
from tensor4gmri.subject_info import read_subject_info
from tensor4gmri.volumes import VOLUME_COLUMNS, subject_volumes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="gmri's roi_signal.parquet",
    )
    parser.add_argument(
        "--subject-info",
        type=Path,
        required=True,
        help="subject_info.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="volumes.csv to write",
    )
    args = parser.parse_args()

    roi = pd.read_parquet(
        args.input,
        columns=["subject", "label", "n_voxels", "voxel_volume_mm3"],
    )
    volumes = subject_volumes(roi)
    volumes.to_csv(args.output, index=False)
    info = read_subject_info(args.subject_info)
    info = info.drop(columns=[c for c in VOLUME_COLUMNS if c in info.columns])
    info.merge(volumes, on="subjects", how="left").to_csv(
        args.subject_info,
        index=False,
    )

    print(
        f"Volumes of {len(volumes)} subjects written to {args.output} and {args.subject_info}",
    )
    if "Group" in info.columns:
        table = info.merge(volumes, on="subjects")
        columns = [c for c in VOLUME_COLUMNS if c.endswith("_ml")]
        print("Median volumes per Group (mL):")
        print(table.groupby("Group")[columns].median().round(1).T.to_string())
